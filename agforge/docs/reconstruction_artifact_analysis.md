# Surface Reconstruction Artifact Analysis & Research Request

## 1. Problem Statement

We have a working hybrid GPU-splatting / CPU-meshing surface reconstruction pipeline for an MPM (Material Point Method) metal forging simulation. The reconstruction generates a 3D mesh from ~5,900 particles in real-time.

**The mesh displays two persistent visual artifact categories:**

1. **Ripples / Waviness** — During deformation (pressing/releasing), the surface exhibits ripple-like waves that propagate across the mesh. These are especially visible on what should be smooth, flat surfaces.

2. **Static Bumps / Roughness** — Even when the material is stationary (idle state), the reconstructed surface has a "lumpy" or "cobblestone" texture rather than being smooth. The underlying cylinder should have smooth curved surfaces.

**We have already implemented the commonly recommended mitigations** (temporal density blending and Taubin mesh smoothing) but the artifacts persist. We need advice on what to try next.

## 2. Current Pipeline Architecture

```
MPM Particles (GPU, ~5901)
    │
    ▼
Density Splatting (GPU, Taichi kernel)
    │  Isotropic cubic spline: W(r) = (1 - r²)³
    │  Influence radius = 2.5 × particle_radius
    │  Grid: 64³ uniform
    │
    ▼
Temporal Density Blending (GPU, Taichi kernel)
    │  D_t = α·D_current + (1-α)·D_previous
    │  α = 0.35 (steady state), α = 1.0 (first frame)
    │
    ▼
GPU→CPU Transfer
    │  density.to_numpy()  (~64³ = 262,144 floats)
    │
    ▼
Marching Cubes (CPU, skimage.measure.marching_cubes)
    │  iso-value = 0.5
    │  Produces ~5,000-5,200 vertices, ~10,000 faces
    │
    ▼
Taubin Smoothing (CPU, trimesh.smoothing.filter_taubin)
    │  iterations = 3
    │  Fallback: Laplacian (lamb=0.5, iterations=2)
    │
    ▼
Output Mesh (trimesh.Trimesh)
```

### Performance Profile (from actual profiling run)

| Stage | Time | Notes |
|-------|------|-------|
| Full reconstruction (`teleop_recon_update`) | ~11.4ms avg | 1142 calls, 13025ms total |
| Physics step | ~11.3ms avg | 2072 calls, 23415ms total |
| Total frame (teleop_step) | ~22.8ms avg | Including IO, logic |

The reconstruction is 27.5% of active profiled time (the single largest leaf-node hotspot).

### Physical Setup

- **Material**: Elasto-plastic metal cylinder (steel-like, E=50GPa, ν=0.28, ρ=8000 kg/m³)
- **Cylinder diameter**: 1 inch (0.0254m), height: 6× radius
- **Particle count**: 5,901 (fixed, all active)
- **Particle radius**: ~0.000931m (inferred from `mpm_solver.particle_size / 2`)
- **Influence radius**: ~0.002327m (2.5× particle radius)
- **Deformation**: Up to 50% compressive strain via opposing grippers
- **Grid resolution**: 64³ (reconstruction grid, NOT the MPM simulation grid)
- **MPM grid density**: ~275 (base_grid_density = int(7 / cylinder_diameter))
- **dx (recon grid)**: ~0.00143m (varies with bounding box)

### Key Ratio Analysis

```
influence_radius / dx = 0.002327 / 0.00143 ≈ 1.63 grid cells
particle_radius / dx  = 0.000931 / 0.00143 ≈ 0.65 grid cells
```

This means each particle's influence covers roughly a 3×3×3 neighborhood of grid cells. This is relatively coarse — the kernel is only sampled at ~27 grid points per particle.

## 3. What We Have Already Tried

### A. Temporal Density Blending (Implemented)

Exponential moving average over the 3D density field between frames:

```python
@ti.kernel
def _blend_density_temporal(self, alpha: float):
    for I in ti.grouped(self.density):
        blended_val = alpha * self.density[I] + (1.0 - alpha) * self.prev_density[I]
        self.density[I] = blended_val
        self.prev_density[I] = blended_val
```

- `alpha = 0.35` in steady state, `alpha = 1.0` on first frame
- **Observation**: Reduces high-frequency temporal flickering somewhat, but ripples during active deformation are still clearly visible. The blending may actually be fighting against the rapidly changing ground truth during pressing.

### B. Taubin Mesh Smoothing (Implemented)

Post-process smoothing on the output mesh:

```python
if len(mesh.vertices) > 0:
    try:
        trimesh.smoothing.filter_taubin(mesh, iterations=3)
    except Exception:
        try:
            trimesh.smoothing.filter_laplacian(mesh, lamb=0.5, iterations=2)
        except Exception as e:
            gs.logger.debug(f"Smoothing failed: {e}")

    if np.isnan(mesh.vertices).any():
        gs.logger.warning("Smoothing produced NaN vertices, using unsmoothed mesh")
        mesh = trimesh.Trimesh(
            vertices=verts, faces=faces,
            vertex_normals=normals, process=False
        )
```

- **Observation**: Helps slightly with small-scale bumps, but 3 iterations of Taubin is not enough to remove the larger-scale waviness. Increasing iterations risks over-smoothing and losing actual geometric detail from the deformation.

### C. Active Particle Bounding Box (Implemented)

Grid bounds are computed from active particles only (not all particles), with `2× influence_radius` padding. This ensures the grid tightly covers the relevant volume.

## 4. Artifact Characterization

### Ripples During Deformation

- Appear as wave-like undulations on the mesh surface
- Most visible during the PRESSING phase (grippers compressing material)
- Frequency seems related to the grid cell size — roughly 3-5 cells wavelength
- Persist frame-to-frame despite temporal blending (likely because the particles are actively moving, so the density field is legitimately changing rapidly)

### Static Bumps at Rest

- Even when idle (no active deformation), the surface is not smooth
- Individual particle contributions are visible as subtle "bumps" in the isosurface
- The isotropic kernel creates roughly spherical density blobs per particle
- Where particle spacing is non-uniform (especially after deformation), the density field has peaks and valleys between particles
- This is the classic "bunch of grapes" artifact from isotropic kernels

### Suspected Root Causes (in priority order)

1. **Grid resolution too low** — At 64³, the ratio of influence_radius to dx (~1.63) means the kernel is very coarsely sampled. Each particle only affects ~27 grid cells.

2. **Isotropic kernel limitation** — After compression, particles are closer in one axis but the kernel is still spherical. This creates density oscillations along the compressed axis.

3. **Fixed bounding box per frame** — The bounding box (and thus dx) changes slightly each frame as particles move. This means the grid "slides" relative to particles, causing the density pattern to shift even when particles barely move.

4. **Temporal blending may be counterproductive during fast deformation** — At α=0.35, we're keeping 65% of the *previous* density field, but during pressing, the particle cloud is rapidly changing shape. The blended field may be a poor approximation of neither the current nor previous state.

## 5. Questions for Research

Given our constraints (real-time at ~11ms per reconstruction, ~5900 particles, GPU density computation + CPU meshing), we'd like advice on:

### A. Grid & Kernel Improvements

1. **Would increasing grid resolution to 96³ or 128³ significantly help?** Our current 64³ gives influence_radius/dx ≈ 1.6. At 128³ this would be ~3.2, quadrupling the sampling density per kernel. What is the quality/performance tradeoff?

2. **Should we increase the influence radius multiplier?** Currently 2.5× particle_radius. Would 3.0× or 3.5× help fill density gaps between particles? What's the expected effect on detail preservation?

3. **Is the kernel function `(1-r²)³` optimal, or would a different kernel (e.g., Wendland C2, poly6, cubic B-spline) produce smoother density fields?**

4. **Should we use a fixed grid origin/spacing across frames** instead of recomputing the bounding box each frame? This would eliminate the "grid sliding" artifact but might waste resolution on empty space.

### B. Temporal Strategy

5. **Should temporal blending be adaptive?** For example: α=0.35 when idle, α=0.8 during active deformation (to track the rapidly changing shape), and then fade α back down after deformation stops?

6. **Are there better temporal strategies than EMA?** For example, Kalman filtering on the density field, or blending at the mesh level (vertex interpolation) rather than the density level?

### C. Smoothing Strategy

7. **Is there a better smoothing approach than Taubin?** We're doing 3 iterations of Taubin post marching cubes. Would more iterations help, or would a different technique (bilateral mesh smoothing, mean curvature flow) be more appropriate for these specific artifacts?

8. **Should smoothing be applied to the density field (3D Gaussian blur before marching cubes) instead of / in addition to mesh smoothing?** This would smooth out individual particle bumps before isosurface extraction.

### D. Advanced Approaches (if basic tuning insufficient)

9. **Anisotropic kernels** — The literature (Yu & Turk 2013) suggests computing per-particle covariance matrices and using ellipsoidal kernels. Is there a simplified approximation that could work in real-time? For example, using the deformation gradient from MPM (which is already computed) to stretch the kernel?

10. **Would a different meshing algorithm help?** We're using standard Marching Cubes (skimage). Would Dual Contouring or a different MC variant reduce grid-aligned artifacts?

11. **Gaussian splatting for density** — Instead of the current kernel, would treating particles as 3D Gaussians with covariance derived from their local neighborhood produce smoother density fields?

### E. Practical Constraints

- Must run in real-time (~15ms budget for reconstruction)
- ~5,900 particles (fixed count)
- GPU: NVIDIA RTX 3060 Laptop (6GB)
- Using Taichi (gstaichi) for GPU kernels
- Using skimage.measure.marching_cubes for CPU meshing
- The mesh is sent over websocket to an external 3D visualization client

## 6. Complete Current Source Code

### agforge/reconstruction.py (Current Implementation)

```python
import torch
import numpy as np
import trimesh
import trimesh.smoothing
import gstaichi as ti
import genesis as gs
import genesis.utils.particle as pu
from skimage.measure import marching_cubes
from enum import Enum

# Compatibility Enum
class SamplingMethod(Enum):
    RANDOM = "random"
    VOXEL_STRATIFIED = "voxel_stratified"
    FPS = "fps"
    HALTON_LLOYD = "halton_lloyd"

@ti.data_oriented
class SurfaceReconstructor:
    def __init__(self, env, grid_res=128, backend='hybrid'):
        self.env = env
        self.reconstructed_mesh = trimesh.Trimesh()
        self.recon_enabled = True
        self.recon_frame_interval = 3  # Reconstruct every 3 frames
        self.frame_counter = 0
        
        # Compatibility / Configuration
        self.sampling_method = SamplingMethod.VOXEL_STRATIFIED
        self.recon_particle_fraction = 1.0
        self.skinning_enabled = False # Dummy flag
        self.bind_indices = None
        self.bind_weights = None
        
        # Backend Configuration
        self.backend = backend # 'hybrid' or 'splashsurf'
        
        # Caching for Visualization
        self._cached_particles = None
        
        # Grid Configuration
        self.grid_res = grid_res
        self.density = ti.field(dtype=float, shape=(self.grid_res, self.grid_res, self.grid_res))
        self.prev_density = ti.field(dtype=float, shape=(self.grid_res, self.grid_res, self.grid_res))
        self.temporal_alpha = 0.35 # Blend factor (0.0 = history only, 1.0 = no smoothing)
        self.density_initialized = False
        
        # Constants
        self.particle_radius = 0.005 
        try:
            self.particle_radius = self.env.scene.sim.mpm_solver.particle_size / 2.0
            gs.logger.info(f"Reconstruction: Inferred particle radius: {self.particle_radius}")
        except Exception:
            pass
        self.influence_radius = self.particle_radius * 2.5

    def get_state(self):
        return {
            'mesh': self.reconstructed_mesh.copy(),
            'recon_enabled': self.recon_enabled,
            'frame_counter': self.frame_counter,
        }

    def set_state(self, state):
        if state is None: return
        self.reconstructed_mesh = state['mesh'].copy()
        self.recon_enabled = state.get('recon_enabled', True)
        self.frame_counter = state.get('frame_counter', 0)
        self.density_initialized = False

    def reset(self):
        self.reconstructed_mesh = trimesh.Trimesh()
        self.frame_counter = 0
        self._cached_particles = None
        self.skinning_enabled = False
        self.density_initialized = False

    # --- Compatibility Interface ---
    def init_skinning(self):
        """No-op compatibility method."""
        pass
        
    def update_skinning(self):
        """No-op compatibility method."""
        pass
        
    def subdivide_long_edges(self) -> bool:
        """No-op compatibility method."""
        return False
        
    def get_active_particle_cache(self):
        """Returns cached particles for visualization."""
        if self._cached_particles is None:
             return self._get_active_particles(use_cache=False)
        return self._cached_particles
        
    def _get_active_particles(self, use_cache: bool = True, apply_subsampling: bool = True):
        """Fetches active particles from MPM entity."""
        if use_cache and self._cached_particles is not None:
            return self._cached_particles
            
        try:
            mpm_entity = self.env.mpm_entity
            parts = mpm_entity.get_particles_pos(envs_idx=0).squeeze(0)
            active = mpm_entity.get_particles_active(envs_idx=0).squeeze(0)
            parts = parts[active]
            
            p_numpy = parts.cpu().numpy()
            self._cached_particles = p_numpy
            return p_numpy
        except Exception:
            return None

    @ti.kernel
    def _compute_density_kernel(
        self, 
        particles_pos: ti.types.ndarray(),
        active_mask: ti.types.ndarray(),
        n_particles: int,
        lower_bound: ti.types.vector(3, float),
        dx: float,
        influence_radius: float
    ):
        for I in ti.grouped(self.density):
            self.density[I] = 0.0
            
        for i in range(n_particles):
            if active_mask[i]:
                pos = ti.Vector([particles_pos[i, 0], particles_pos[i, 1], particles_pos[i, 2]])
                grid_pos = (pos - lower_bound) / dx
                rad_cells = influence_radius / dx
                
                base_idx = ti.cast(ti.floor(grid_pos - rad_cells), ti.int32)
                end_idx = ti.cast(ti.ceil(grid_pos + rad_cells), ti.int32)
                
                for ix in range(base_idx[0], end_idx[0] + 1):
                    for iy in range(base_idx[1], end_idx[1] + 1):
                        for iz in range(base_idx[2], end_idx[2] + 1):
                            if (0 <= ix < self.grid_res and 
                                0 <= iy < self.grid_res and 
                                0 <= iz < self.grid_res):
                                cell_center = lower_bound + ti.Vector([ix, iy, iz]) * dx
                                dist_sq = (pos - cell_center).norm_sqr()
                                if dist_sq < influence_radius**2:
                                    r2 = dist_sq / (influence_radius**2)
                                    val = (1.0 - r2)**3
                                    self.density[ix, iy, iz] += val

    @ti.kernel
    def _blend_density_temporal(self, alpha: float):
        """Exponential moving average over the density field: D = alpha*D_new + (1-alpha)*D_prev."""
        for I in ti.grouped(self.density):
            blended_val = alpha * self.density[I] + (1.0 - alpha) * self.prev_density[I]
            self.density[I] = blended_val
            self.prev_density[I] = blended_val

    def update(self, should_reconstruct: bool):
        if not self.recon_enabled:
            return
        self.frame_counter += 1
        if not should_reconstruct and (self.frame_counter % self.recon_frame_interval != 0):
            return
        self.create_reconstructed_mesh()

    def create_reconstructed_mesh(self):
        # Legacy SplashSurf Path
        if self.backend == 'splashsurf':
            self._create_splashsurf_mesh()
            return

        # Hybrid Path
        try:
            mpm_entity = self.env.mpm_entity
            particles_pos = mpm_entity.get_particles_pos(envs_idx=0).squeeze(0)
            particles_active = mpm_entity.get_particles_active(envs_idx=0).squeeze(0)
            
            n_particles = particles_pos.shape[0]
            if n_particles == 0:
                self.reconstructed_mesh = trimesh.Trimesh()
                return

            active_particles = particles_pos[particles_active]
            n_active_particles = active_particles.shape[0]
            if n_active_particles == 0:
                gs.logger.warning(f"Reconstruction: 0 active of {n_particles} total particles")
                self._cached_particles = np.empty((0, 3), dtype=np.float32)
                self.reconstructed_mesh = trimesh.Trimesh()
                return

            # Update cache for visualization
            self._cached_particles = active_particles.cpu().numpy()

            min_bound = active_particles.min(dim=0).values.cpu().numpy()
            max_bound = active_particles.max(dim=0).values.cpu().numpy()
            
            padding = self.influence_radius * 2.0
            min_bound -= padding
            max_bound += padding
            
            extent = max_bound - min_bound
            max_extent = np.max(extent)
            if max_extent < 1e-4: max_extent = 1.0
                
            dx = max_extent / (self.grid_res - 1)
            lower_bound_ti = ti.Vector([min_bound[0], min_bound[1], min_bound[2]])
            
            self._compute_density_kernel(
                particles_pos, 
                particles_active, 
                n_particles, 
                lower_bound_ti, 
                float(dx), 
                float(self.influence_radius)
            )
            
            # Apply Temporal Blending
            # Use alpha=1.0 for first frame to avoid ghosting from zero-init
            alpha = 1.0 if not self.density_initialized else self.temporal_alpha
            self._blend_density_temporal(alpha)
            self.density_initialized = True
            
            density_cpu = self.density.to_numpy()
            max_dens = density_cpu.max()
            thresh = 0.5

            if max_dens < thresh:
                gs.logger.warning(
                    f"Reconstruction: max density {max_dens:.4f} below threshold {thresh}! "
                    f"Mesh will be empty."
                )
                return

            verts, faces, normals, values = marching_cubes(
                density_cpu,
                level=thresh,
                spacing=(dx, dx, dx),
                allow_degenerate=False
            )
            verts += min_bound

            mesh = trimesh.Trimesh(
                vertices=verts,
                faces=faces,
                vertex_normals=normals,
                process=False
            )

            # Taubin smoothing preserves volume better than Laplacian
            if len(mesh.vertices) > 0:
                try:
                    trimesh.smoothing.filter_taubin(mesh, iterations=3)
                except Exception:
                    try:
                        trimesh.smoothing.filter_laplacian(mesh, lamb=0.5, iterations=2)
                    except Exception as e:
                        gs.logger.debug(f"Smoothing failed: {e}")

                if np.isnan(mesh.vertices).any():
                    gs.logger.warning("Smoothing produced NaN vertices, using unsmoothed mesh")
                    mesh = trimesh.Trimesh(
                        vertices=verts, faces=faces,
                        vertex_normals=normals, process=False
                    )

            self.reconstructed_mesh = mesh
            
        except Exception as e:
            gs.logger.warning(f"Reconstruction failed: {e}", exc_info=True)

    def _create_splashsurf_mesh(self):
        """Legacy SplashSurf reconstruction."""
        try:
            particles = self._get_active_particles(use_cache=False)
            if particles is None or len(particles) == 0:
                return
                
            self.reconstructed_mesh = pu.particles_to_mesh(
                positions=particles,
                radius=self.particle_radius * 1.5,
                backend='splashsurf'
            )
        except Exception as e:
            gs.logger.warning(f"SplashSurf failed: {e}")
```

### Reconstruction Configuration (from agforge/options.py)

```python
class ReconstructionOptions(Options):
    grid_res: int = 64
    backend: str = "hybrid"  # 'hybrid' or 'splashsurf'
    enabled: bool = True

# MPM solver parameters (affect particle size/spacing):
self.mpm = MPMOptions(
    grid_density=self.robot.base_grid_density,  # ~275
    particle_size=0.8 * 0.01 * 64.0 / self.robot.base_grid_density,  # ~0.00186m
    lower_bound=self.robot.mpm_lower_bound,
    upper_bound=self.robot.mpm_upper_bound,
)
# particle_radius = particle_size / 2 ≈ 0.000931m
# influence_radius = particle_radius * 2.5 ≈ 0.002327m
```

### Runtime Log Excerpt (typical reconstruction during pressing)

```
Reconstruction: 5901/5901 particles, grid_res=64, dx=0.001421, influence_r=0.002327, alpha=0.35
Reconstruction: max_density=2.3539, threshold=0.5
Reconstruction: 5054 verts, 10076 faces

Reconstruction: 5901/5901 particles, grid_res=64, dx=0.001422, influence_r=0.002327, alpha=0.35
Reconstruction: max_density=2.4617, threshold=0.5
Reconstruction: 5062 verts, 10088 faces

Reconstruction: 5901/5901 particles, grid_res=64, dx=0.001423, influence_r=0.002327, alpha=0.35
Reconstruction: max_density=2.5695, threshold=0.5
Reconstruction: 5002 verts, 9968 faces
```

Note how `dx` changes slightly each frame (0.001421 → 0.001422 → 0.001423) as the bounding box shifts with particle movement. This means the grid is constantly "sliding" relative to particles.

## 7. Summary

We need recommendations for reducing surface ripples and bumps in our particle-based surface reconstruction, given that we've already implemented:
- Temporal density blending (EMA, α=0.35)
- Taubin mesh smoothing (3 iterations)
- Active-particle-only bounding box

The artifacts persist. We want to know:
1. What parameter tuning to try first (grid res, influence radius, kernel, threshold, smoothing iterations)
2. Whether architectural changes are needed (fixed grid, density-space smoothing, adaptive blending)
3. What advanced techniques are feasible within our ~15ms real-time budget
