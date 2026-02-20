import torch
import numpy as np
import trimesh
import trimesh.smoothing
import gstaichi as ti
import genesis as gs
import genesis.utils.particle as pu
from skimage.measure import marching_cubes
from scipy.ndimage import gaussian_filter
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
        
        # Grid snapping: fixed dx from physical scale, snapped origin
        self._grid_coverage_ratio = 1.6
        self._fixed_dx = None
        self._prev_grid_origin = None
        
        # Density-space smoothing before marching cubes (sigma in grid cells)
        self.density_blur_sigma = 0.75

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
        self._prev_grid_origin = None

    def reset(self):
        self.reconstructed_mesh = trimesh.Trimesh()
        self.frame_counter = 0
        self._cached_particles = None
        self.skinning_enabled = False
        self.density_initialized = False
        self._fixed_dx = None
        self._prev_grid_origin = None

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
             # Try to fetch if cache is empty
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

            # Compute fixed dx from physical scale (once, then stable)
            if self._fixed_dx is None:
                self._fixed_dx = self.influence_radius / self._grid_coverage_ratio
                gs.logger.info(
                    f"Reconstruction: Fixed dx={self._fixed_dx:.6f} "
                    f"(influence_r/dx={self.influence_radius / self._fixed_dx:.2f})"
                )
            dx = self._fixed_dx

            # Compute raw bounds with padding
            raw_min = active_particles.min(dim=0).values.cpu().numpy()
            raw_max = active_particles.max(dim=0).values.cpu().numpy()
            padding = self.influence_radius * 2.0
            raw_min -= padding
            raw_max += padding

            # Snap bounds to multiples of dx for temporal coherence
            min_bound = np.floor(raw_min / dx) * dx
            max_bound = np.ceil(raw_max / dx) * dx

            # Verify grid capacity — fall back to dynamic dx if particles exceed grid
            extent = max_bound - min_bound
            max_extent = float(np.max(extent))
            if max_extent < 1e-4:
                max_extent = 1.0
            required_cells = int(np.ceil(max_extent / dx)) + 1
            if required_cells > self.grid_res:
                dx = max_extent / (self.grid_res - 1)
                min_bound = np.floor(raw_min / dx) * dx
                gs.logger.warning(
                    f"Reconstruction: Particles exceed fixed grid ({required_cells} > {self.grid_res}), "
                    f"adjusting dx={dx:.6f} (influence_r/dx={self.influence_radius / dx:.2f})"
                )

            # Detect grid origin shift — reset temporal blending when grid jumps
            grid_origin = min_bound.copy()
            if self._prev_grid_origin is not None:
                if not np.allclose(grid_origin, self._prev_grid_origin, atol=dx * 0.01):
                    self.density_initialized = False
                    gs.logger.debug("Reconstruction: Grid origin shifted, resetting temporal blend")
            self._prev_grid_origin = grid_origin

            lower_bound_ti = ti.Vector([min_bound[0], min_bound[1], min_bound[2]])
            
            gs.logger.debug(
                f"Reconstruction: {n_active_particles}/{n_particles} particles, "
                f"grid_res={self.grid_res}, dx={dx:.6f}, influence_r={self.influence_radius:.6f}, "
                f"alpha={'1.0 (init)' if not self.density_initialized else str(self.temporal_alpha)}"
            )

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
            
            # Density-space smoothing: suppress per-particle bumps and grid aliasing
            if self.density_blur_sigma > 0:
                density_cpu = gaussian_filter(density_cpu, sigma=self.density_blur_sigma)

            max_dens = density_cpu.max()
            thresh = 0.5

            gs.logger.debug(f"Reconstruction: max_density={max_dens:.4f}, threshold={thresh}")

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

            gs.logger.debug(f"Reconstruction: {len(mesh.vertices)} verts, {len(mesh.faces)} faces")
            self.reconstructed_mesh = mesh
            
        except Exception as e:
            gs.logger.warning(f"Reconstruction failed: {e}", exc_info=True)

    def _create_splashsurf_mesh(self):
        """Legacy SplashSurf reconstruction."""
        try:
            particles = self._get_active_particles(use_cache=False)
            if particles is None or len(particles) == 0:
                return
                
            # Use ~1.0x particle radius for reconstruction
            # SplashSurf handles neighborhood search internally
            self.reconstructed_mesh = pu.particles_to_mesh(
                positions=particles,
                radius=self.particle_radius * 1.5,
                backend='splashsurf'
            )
        except Exception as e:
            gs.logger.warning(f"SplashSurf failed: {e}")