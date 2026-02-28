# Surface Reconstruction — Implementation Plan

Comprehensive guide for future development of the surface reconstruction pipeline
in `agforge/reconstruction.py`. Written for AI coding assistants and developers
continuing this work.

**Last updated**: 2026-02-19
**Current commit**: `5c414cf` (Fix #2: density-space Gaussian blur)

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Current Implementation State](#2-current-implementation-state)
3. [Known Issues & Artifacts](#3-known-issues--artifacts)
4. [Mathematical Parameter Framework](#4-mathematical-parameter-framework)
5. [Implementation Roadmap](#5-implementation-roadmap)
6. [Detailed Implementation Guides](#6-detailed-implementation-guides)
7. [What Was Tried and Reverted](#7-what-was-tried-and-reverted)
8. [Historical Context — Abandoned Approaches](#8-historical-context--abandoned-approaches)
9. [Alternative Reconstruction Approaches](#9-alternative-reconstruction-approaches)
10. [Reference Research Summary](#10-reference-research-summary)
11. [File Inventory & Cleanup](#11-file-inventory--cleanup)
12. [Appendix C: Performance Benchmarks](#appendix-c-performance-benchmarks)
13. [Appendix D: Platform Compatibility](#appendix-d-platform-compatibility)

---

## 1. Architecture Overview

### Pipeline

```
MPM Particles (GPU, Taichi)
    ↓ get_particles_pos / get_particles_active
Active Particles (torch tensor)
    ↓ _compute_density_kernel (Taichi GPU kernel)
3D Density Field (Taichi field, grid_res³)
    ↓ _blend_density_temporal (EMA smoothing)
Temporally Blended Density
    ↓ .to_numpy() transfer
Density on CPU (numpy)
    ↓ gaussian_filter (scipy)
Smoothed Density
    ↓ marching_cubes (skimage)
Raw Mesh (vertices on grid edges)
    ↓ filter_taubin (trimesh)
Final Mesh → sent to Unity via WebSocket
```

### Key Files

| File | Role |
|------|------|
| `agforge/reconstruction.py` | `SurfaceReconstructor` class — all reconstruction logic |
| `agforge/options.py` | `ReconstructionOptions` — config (grid_res, backend, enabled) |
| `agforge/strike_controller.py` | Creates reconstructor, calls `update()` per frame |
| `agforge/teleop_socket.py` | Simulation loop, sends mesh data to Unity client |

### Physical Setup

- **Material**: Elasto-plastic metal cylinder (steel-like, E=50GPa, ν=0.28, ρ=8000 kg/m³)
- **Cylinder**: 1 inch (0.0254m) diameter, height = 6× radius ≈ 0.0762m
- **Particle count**: ~5,901 (fixed, all active)
- **Deformation**: Up to 50% compressive strain via opposing grippers
- **MPM solver**: Genesis MPM with Taichi backend
- **MPM grid density**: ~275 (`base_grid_density = int(7 / cylinder_diameter)`)
- **Particle size formula**: `0.8 * 0.01 * 64.0 / base_grid_density` ≈ 0.00186m
- **GPU**: NVIDIA RTX 3060 Laptop (6GB) — development reference hardware

---

## 2. Current Implementation State

### Parameters (as of commit `5c414cf`)

| Parameter | Value | Location |
|-----------|-------|----------|
| `grid_res` | 64 | `options.py:81` |
| `influence_radius` | `particle_radius * 2.5` | `reconstruction.py:55` |
| `_grid_coverage_ratio` | 1.6 | `reconstruction.py:58` |
| `_fixed_dx` | `influence_radius / 1.6` ≈ 0.001455m | Computed at runtime |
| `density_blur_sigma` | 0.75 (grid cells) | `reconstruction.py:63` |
| `temporal_alpha` | 0.35 | `reconstruction.py:45` |
| `thresh` (iso-level) | 0.5 (hardcoded) | `reconstruction.py:277` |
| Taubin iterations | 3 | `reconstruction.py:306` |
| Kernel | `(1-r²)³` isotropic | `reconstruction.py:154` |

### Derived Values (for default cylinder)

| Derived | Value |
|---------|-------|
| `particle_radius` | ~0.000931m (from `mpm_solver.particle_size / 2`) |
| `influence_radius` | ~0.002327m |
| `dx` (fixed) | ~0.001455m |
| `influence_radius / dx` | 1.6 (each particle covers ~3×3×3 cells) |
| Grid physical extent | 64 × 0.001455 ≈ 0.093m per axis |
| Inter-particle spacing | `(V/N)^(1/3)` ≈ 0.00186m ≈ 1.28 cells |

### Features Implemented

- [x] GPU density splatting via Taichi kernel
- [x] Temporal EMA blending of density fields
- [x] Grid snapping — fixed dx from physical scale, origin snapped to dx multiples
- [x] Temporal blend reset when grid origin shifts
- [x] Density-space Gaussian blur before marching cubes
- [x] Taubin smoothing with Laplacian fallback
- [x] NaN vertex detection and recovery
- [x] Graceful fallback when particles exceed grid capacity
- [x] Legacy SplashSurf backend path

---

## 3. Known Issues & Artifacts

### 3.1 Contour Lines / MC Staircase (UNRESOLVED)

**Description**: Visible parallel ridges/terraces across the mesh surface, like topographic
contour lines. Most visible on smooth curved surfaces.

**Root cause**: Marching cubes constrains vertices to grid cell edges. On curved surfaces this
creates a stepped/banded appearance. The effect is fundamental to MC and cannot be fully
eliminated by pre-smoothing the density field alone.

**What was tried**:
- `density_blur_sigma` up to 1.8 — reduced but did not eliminate
- `grid_res` up to 128 with `coverage_ratio` up to 2.5 — reduced but added bloat/performance cost
- `thresh` lowered to 0.3 — caused mesh bloat
- `influence_radius * 3.0` — caused mesh bloat
- Taubin iterations up to 6 — insufficient

**Promising approaches not yet tried**:
- Mesh subdivision before smoothing (see §6.1)
- Dynamic threshold based on peak density (see §6.2)
- PyVista Flying Edges instead of skimage MC (see §6.4)

### 3.2 Grid Overflow During Heavy Compression (MINOR)

**Description**: Warning `Particles exceed fixed grid (65 > 64)` during RELEASE phase after
heavy compression. The fallback works (adjusts dx dynamically) but temporarily loses the
grid snap benefit.

**Fix**: Increase `grid_res` to 96 or compute it dynamically from the physical setup (see §4).

### 3.3 Residual Minor Bumps (MINOR)

**Description**: Small indents/bumps from individual particles. Significantly reduced from
the original "bunch of grapes" artifact but still faintly visible.

**Fix**: Addressed by mathematical parameter framework (see §4) — primarily by increasing
`influence_radius` to `3.0 * particle_radius` and `coverage_ratio` to 3.0.

---

## 4. Mathematical Parameter Framework

All reconstruction parameters can be derived from a single ground truth value:
**`particle_radius` ($r_p$)**. This framework is based on SPH theory and the Nyquist
sampling theorem.

### 4.1 Kernel Size: `influence_radius`

**Principle**: In SPH theory, to ensure no internal density gaps when particles are in a
resting grid, the support radius must overlap at least 2–3 neighboring particles.

$$R_{inf} = 3.0 \times r_p$$

Current value is `2.5 × r_p`, which is slightly too small for the `(1-r²)³` kernel and
contributes to bump artifacts.

### 4.2 Sampling Rate: `dx` (grid cell size)

**Principle**: Nyquist-Shannon sampling theorem. The kernel creates a density "wave" with
width $R_{inf}$. To digitize without aliasing, sample at least 3× per kernel width.

$$dx = \frac{R_{inf}}{3.0} = r_p$$

This means: **grid cells should be exactly the size of a particle**. Current ratio of 1.6
(`dx = R_inf / 1.6`) is significantly under-sampled.

The `_grid_coverage_ratio` parameter controls this:

$$\text{coverage\_ratio} = \frac{R_{inf}}{dx} = 3.0$$

### 4.3 Grid Resolution: `grid_res`

**Principle**: Grid must contain the maximum deformed object extent plus padding for the
kernel to drop smoothly to zero.

$$\text{Padding} = 2.0 \times R_{inf}$$
$$L_{max} = \text{maximum physical extent under deformation}$$
$$N_{grid} = \left\lceil \frac{L_{max} + 2 \times \text{Padding}}{dx} \right\rceil$$

For the default cylinder under 50% compression:
- Uncompressed: ~0.025m × 0.025m × 0.076m
- Compressed (50%): material spreads to ~0.05m × 0.05m × 0.038m
- With padding: max extent ≈ 0.067m
- At $dx = r_p \approx 0.000931$m: $N_{grid} \approx 72 + 12 = 84$

**Recommendation**: `grid_res = 96` gives comfortable headroom. `grid_res = 128` gives
generous headroom but at higher compute cost (8× cells vs 64³).

### 4.4 Anti-Aliasing Filter: `density_blur_sigma`

**Principle**: Low-pass filter to remove grid discretization noise (wavelength = $1 \times dx$)
without destroying particle-level geometry.

$$\sigma_{cells} = 1.25$$

Since $dx = r_p$, sigma of 1.25 cells blurs slightly more than one particle width — enough
to erase contour lines while preserving shape.

### 4.5 Isosurface Threshold: `thresh`

**Principle**: A fixed threshold (0.5) is fragile. Under compression, peak density spikes.
If the threshold is a small fraction of $D_{max}$, the isosurface cuts through the steep
tail of the density gradient where grid artifacts are most visible. Cut at the gradient's
midpoint instead.

$$\tau = 0.4 \times D_{max}$$

where $D_{max}$ is the peak density in the grid for the current frame. This automatically
adapts to compression state.

### 4.6 Summary: Optimal Parameters

For the default cylinder (`particle_size ≈ 0.00186m`):

| Parameter | Current | Optimal | Formula |
|-----------|---------|---------|---------|
| `influence_radius` | `r_p × 2.5` = 0.00233m | `r_p × 3.0` = 0.00279m | $3.0 \times r_p$ |
| `_grid_coverage_ratio` | 1.6 | 3.0 | $R_{inf} / dx = 3.0$ |
| `dx` (fixed) | 0.00145m | 0.000931m | $R_{inf} / 3.0 = r_p$ |
| `grid_res` | 64 | 96 | $\lceil (L_{max} + 2 \times \text{Pad}) / dx \rceil$ |
| `density_blur_sigma` | 0.75 | 1.25 | 1.25 cells |
| `thresh` | 0.5 (fixed) | `0.4 × D_max` | Dynamic |
| Taubin iterations | 3 | 5–6 | Empirical |

### 4.7 Implementation: Auto-Derive Parameters

The following code computes all parameters from `particle_radius`:

```python
# In __init__, after particle_radius is determined:

# SPH-optimal kernel size
self.influence_radius = self.particle_radius * 3.0

# Nyquist-optimal sampling rate (coverage_ratio = 3.0 → dx = particle_radius)
self._grid_coverage_ratio = 3.0

# Anti-aliasing filter
self.density_blur_sigma = 1.25

# Grid resolution: auto-compute from physical bounds if available, else use default
# This should be computed from the simulation's physical extent when known.
# For now, grid_res stays as a config parameter.
```

In `create_reconstructed_mesh`, replace the fixed threshold:

```python
# Dynamic thresholding: cut at 40% of peak density
max_dens = density_cpu.max()
thresh = max_dens * 0.4

if max_dens < 1e-4:
    return  # Empty grid
```

### 4.8 CAUTION: Performance Impact

Moving from `coverage_ratio=1.6` to `3.0` reduces `dx` by ~1.9× and increases the
effective grid cell count by ~6.7×. At `grid_res=64`, the grid only covers
64 × 0.000931 = 0.060m — too small for the cylinder. You **must** increase `grid_res`
to at least 96, which means ~3.4× more cells than the current 64³.

Combined with wider `influence_radius` (each particle touches more cells), the Taichi
kernel will do significantly more work. **Profile before committing to these values.**
If performance is a concern, intermediate values work:

| Preset | coverage_ratio | grid_res | dx | Grid extent | Perf impact |
|--------|---------------|----------|----|-------------|-------------|
| Current | 1.6 | 64 | 0.00145m | 0.093m | Baseline |
| Moderate | 2.0 | 96 | 0.00116m | 0.112m | ~3× splatting |
| Optimal | 3.0 | 96 | 0.00093m | 0.089m | ~6× splatting |
| Optimal+ | 3.0 | 128 | 0.00093m | 0.119m | ~8× splatting |

---

## 5. Implementation Roadmap

Ordered by expected impact-to-effort ratio. Each item is independent and can be
implemented/tested in isolation.

### Phase 1: Parameter Tuning (no structural changes)

**1A. Dynamic Threshold** — Highest ROI for contour lines
- Change `thresh = 0.5` to `thresh = max_dens * 0.4`
- One-line change in `create_reconstructed_mesh`
- Directly targets contour line artifact by cutting isosurface in flatter gradient region
- Zero performance cost

**1B. Apply Mathematical Parameter Framework**
- Set `influence_radius = particle_radius * 3.0`
- Set `_grid_coverage_ratio = 3.0`
- Set `density_blur_sigma = 1.25`
- Set `grid_res = 96` (in `options.py`)
- Increase Taubin iterations to 5
- **Profile after** — if too slow, use the "Moderate" preset from §4.8

### Phase 2: Mesh Post-Processing (low-risk structural changes)

**2A. Mesh Subdivision Before Smoothing** — Targets MC staircase directly
- Add `mesh.subdivide()` call before Taubin smoothing
- Breaks grid-edge vertex constraint, gives smoother more freedom
- Increases face count 4× (e.g. 10k → 40k) — acceptable for visuals
- If collider mesh is separate, apply only to visual mesh
- See §6.1 for detailed implementation

**2B. Decimation After Smoothing** (optional)
- If subdivision produces too many faces for the WebSocket/Unity pipeline
- `trimesh.simplify.simplify_quadric_decimation(mesh, face_count=target)`
- Apply after subdivision + smoothing to reduce poly count while preserving smooth shape

### Phase 3: Algorithm Improvements (moderate structural changes)

**3A. Adaptive Temporal Blending**
- High alpha during fast deformation (respond quickly to shape changes)
- Low alpha at rest (maximize smoothness)
- Compute deformation rate from particle velocity variance or force magnitude
- See §6.3 for implementation

**3B. Replace skimage MC with PyVista Flying Edges**
- Flying Edges is 2-5× faster than skimage MC, especially at higher resolutions
- Produces slightly different vertex placement (may help with contour lines)
- Requires `pyvista` dependency
- See §6.4 for implementation

### Phase 4: Advanced Techniques (significant structural changes)

**4A. Anisotropic Splatting Kernels**
- Use MPM deformation gradient to stretch splat kernels into ellipsoids
- Compressed particles produce "flat disk" density contributions instead of spheres
- Directly eliminates "bunch of grapes" under deformation
- Requires accessing deformation gradient from MPM solver
- Significant Taichi kernel changes

**4B. Density-Space Gaussian Blur in Taichi (GPU)**
- Move `gaussian_filter` from CPU (scipy) to GPU (Taichi separable blur kernel)
- Eliminates CPU work and the GPU→CPU→GPU round-trip for this step
- Only needed if blur becomes a performance bottleneck at higher grid resolutions

**4C. Unity-Side Visual Reconstruction (Dual-Mesh Architecture)**
- Keep Python reconstruction for collider mesh (low-res, lower frequency)
- Implement high-fidelity visual reconstruction in Unity compute shaders
- **Path A (Visuals)**: Send compressed density field to Unity. Unity Compute Shader
  (Metal/DX12) runs MC at 128³-256³ for high-fidelity rendering at 60fps.
- **Path B (Colliders)**: Downsample density to 32³/48³. `skimage.marching_cubes` on
  coarse grid (<3ms). Send low-poly mesh to Unity `MeshCollider`.
- Binary mesh streaming format: 12-byte header (`uint32 num_verts`, `uint32 num_tris`,
  `uint8 has_normals`, `uint8 has_colors`, `uint8 has_uv`, 1 byte padding), followed by
  vertex positions (N × 3 × float32), optional normals, indices (T × 3 × uint32).

---

## 6. Detailed Implementation Guides

### 6.1 Mesh Subdivision Before Smoothing

This is the most promising fix for the contour line artifact that hasn't been tried yet.

**Why it works**: MC places vertices exactly on grid edges. Even with perfect density
smoothing, the final vertices are constrained to grid lines. Subdivision injects new vertices
*not* locked to the grid, giving the smoother freedom to create truly smooth curves.

**Implementation** — replace the smoothing block in `create_reconstructed_mesh`:

```python
if len(mesh.vertices) > 0:
    try:
        # Subdivide to break grid-edge vertex constraint (1 iteration = 4× faces)
        mesh = mesh.subdivide()

        # Smooth the subdivided mesh — more iterations needed since
        # smoothing must propagate across more vertices
        trimesh.smoothing.filter_taubin(mesh, iterations=5)
    except Exception:
        try:
            trimesh.smoothing.filter_laplacian(mesh, lamb=0.5, iterations=2)
        except Exception as e:
            gs.logger.debug(f"Subdivision/Smoothing failed: {e}")
```

**Trade-offs**:
- Face count goes from ~10k to ~40k. Fine for Unity rendering, may be heavy for collider.
- Adds a few ms of CPU time. Profile at your grid resolution.
- If too many faces for the WebSocket pipeline, add decimation after smoothing:
  `mesh = mesh.simplify_quadric_decimation(face_count=12000)`

### 6.2 Dynamic Threshold

**Implementation** — in `create_reconstructed_mesh`, replace the fixed threshold:

```python
# BEFORE:
max_dens = density_cpu.max()
thresh = 0.5

# AFTER:
max_dens = density_cpu.max()
thresh = max_dens * 0.4  # Cut at 40% of peak — flatter gradient region

if max_dens < 1e-4:
    return
```

This automatically adapts to compression state. When particles compress and density
spikes, the threshold rises proportionally, keeping the isosurface cut in the
smooth part of the gradient.

### 6.3 Adaptive Temporal Blending

**Concept**: Use high alpha (respond fast) during active deformation, low alpha (smooth)
when at rest.

```python
# In create_reconstructed_mesh, before _blend_density_temporal:

# Estimate deformation rate from particle velocity
particle_velocities = mpm_entity.get_particles_vel(envs_idx=0).squeeze(0)
active_vels = particle_velocities[particles_active]
vel_magnitude = active_vels.norm(dim=1).mean().item()

# Adaptive alpha: high during motion, low at rest
# vel_threshold: typical pressing speed ~0.01 m/s
vel_threshold = 0.005
alpha_rest = 0.2    # Heavy smoothing when still
alpha_motion = 0.8  # Fast response during deformation
t = min(vel_magnitude / vel_threshold, 1.0)
alpha = alpha_rest + t * (alpha_motion - alpha_rest)

if not self.density_initialized:
    alpha = 1.0
self._blend_density_temporal(alpha)
self.density_initialized = True
```

### 6.4 PyVista Flying Edges

Flying Edges is a parallelized variant of marching cubes that is 2-5× faster, especially
at higher grid resolutions. It may also produce smoother vertex placement.

**Implementation** — replace the skimage marching_cubes call:

```python
import pyvista as pv

# Create PyVista grid wrapper (near zero-copy)
grid = pv.ImageData()
grid.dimensions = np.array(density_cpu.shape) + 1  # VTK needs N+1 for N cells
grid.spacing = (dx, dx, dx)
grid.origin = min_bound
grid.cell_data["density"] = density_cpu.flatten(order="F")

# Flying Edges extraction
contour = grid.contour(isosurfaces=[thresh], scalars="density", method='flying_edges')

if contour.n_points > 0:
    verts = np.array(contour.points)
    faces_flat = np.array(contour.faces)
    # PyVista uses VTK face format: [n, v0, v1, v2, n, v0, v1, v2, ...]
    # Reshape to (N, 3) triangle array
    faces = faces_flat.reshape(-1, 4)[:, 1:4]

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
```

**Notes**:
- Requires `pyvista` dependency: `uv pip install pyvista`
- VTK array ordering matters — use Fortran order for `flatten`
- Check `grid.dimensions` vs `density_cpu.shape` — VTK conventions differ from numpy
- Profile to confirm speedup at your resolution

**Alternative: Raw VTK API** (if you want to avoid the PyVista wrapper):

```python
import vtk
from vtk.util import numpy_support

def extract_surface_flying_edges(density_grid, spacing, origin, threshold):
    vtk_data = numpy_support.numpy_to_vtk(
        num_array=density_grid.ravel(), deep=True, array_type=vtk.VTK_FLOAT
    )
    img = vtk.vtkImageData()
    img.SetDimensions(density_grid.shape)
    img.SetSpacing(spacing)
    img.SetOrigin(origin)
    img.GetPointData().SetScalars(vtk_data)

    algo = vtk.vtkFlyingEdges3D()
    algo.SetInputData(img)
    algo.SetValue(0, threshold)
    algo.Update()

    poly = algo.GetOutput()
    verts = numpy_support.vtk_to_numpy(poly.GetPoints().GetData())
    faces = numpy_support.vtk_to_numpy(poly.GetPolys().GetData()).reshape(-1, 4)[:, 1:]
    return verts, faces
```

### 6.5 Alternative Smoothing Methods

**HC (Humphrey's Classes) Laplacian Smoothing**: Preserves volume and features better
than Taubin by pushing vertices back toward their original positions after each smoothing
iteration. Consider trying this if Taubin produces too much shrinkage:

```python
trimesh.smoothing.filter_humphrey(mesh, alpha=0.1, beta=0.5, iterations=5)
```

**Bilateral Mesh Filtering**: Non-iterative, feature-preserving smoothing analogous to
bilateral image filtering. Smooths flat areas aggressively while preserving sharp
edges/creases. Useful if the deformed cylinder has distinct flat faces that should stay
sharp. Not built into trimesh — would require a custom implementation.

### 6.6 Alternative Kernel Functions

The current `(1-r²)³` kernel is compact and fast but "peaky" — it has a sharp central
peak that produces visible per-particle density spikes.

**Wyvill Kernel** (broader plateau, smoother near center):
$$W(r) = (1 - \frac{4r^6}{9} + \frac{17r^4}{9} - \frac{22r^2}{9})$$
for $0 \leq r \leq 1$. The broader plateau near $r=0$ means nearby particles contribute
more uniformly, reducing the "bunch of grapes" look without increasing grid resolution.

**Wendland C2 Kernel** (standard SPH choice, smooth and compact):
$$W(r) = (1 - r)^4 (4r + 1)$$
for $0 \leq r \leq 1$. Well-studied in SPH literature, good balance of smoothness and
compact support.

To implement, replace the kernel computation in `_compute_density_kernel`:

```python
# Current: (1-r²)³
val = (1.0 - r2)**3

# Wendland C2: (1-r)⁴(4r+1) where r = sqrt(r2)
r = ti.sqrt(r2)
val = (1.0 - r)**4 * (4.0 * r + 1.0)
```

### 6.7 Vertex Correspondence Blending (Post-MC Temporal Smoothing)

Instead of (or in addition to) density-space temporal blending, blend the mesh vertex
positions directly between frames. This adds temporal coherence at the mesh level:

```python
from scipy.spatial import cKDTree

if self._prev_mesh is not None and len(self._prev_mesh.vertices) > 0:
    tree = cKDTree(self._prev_mesh.vertices)
    dists, indices = tree.query(verts)
    # Blend: 90% new position, 10% previous (low-pass filter)
    blend_factor = 0.1
    verts = (1.0 - blend_factor) * verts + blend_factor * self._prev_mesh.vertices[indices]
self._prev_mesh = mesh.copy()
```

**Trade-offs**: Adds ~1-2ms for the kNN query. May cause slight lag in surface response.
Best used with a low blend factor (0.05-0.15).

### 6.8 Anisotropic Splatting Kernels

The most impactful advanced technique. Instead of spherical kernels, stretch each
particle's density contribution into an ellipsoid aligned with the local deformation.

**Approach A: Velocity-Based** (simpler, uses particle velocities):

```python
@ti.kernel
def _compute_anisotropic_density(self, ...):
    for i in range(n_particles):
        if active_mask[i]:
            pos = ti.Vector([particles_pos[i, 0], particles_pos[i, 1], particles_pos[i, 2]])
            vel = ti.Vector([particles_vel[i, 0], particles_vel[i, 1], particles_vel[i, 2]])

            v_norm_sq = vel.norm_sqr()
            if v_norm_sq > 1e-6:
                k_stretch = 2.0
                # Stretch matrix: identity + k * v⊗v / |v|²
                G = ti.Matrix.identity(float, 3)
                for ii in ti.static(range(3)):
                    for jj in ti.static(range(3)):
                        G[ii, jj] += k_stretch * vel[ii] * vel[jj] / v_norm_sq
                G_inv = G.inverse()
                # Use G_inv to transform distance: dist_sq = (Δx)ᵀ G_inv (Δx)
            else:
                # Fall back to isotropic
                G_inv = ti.Matrix.identity(float, 3)
```

**Approach B: PCA-Based** (more accurate, uses particle neighborhoods):
Compute the covariance matrix of each particle's k-nearest neighbors. The eigenvectors
define the anisotropy axes, eigenvalues define the stretch. Under compression, particles
spread into flat disks, naturally producing smooth flat surfaces instead of lumpy ones.
This requires a neighbor search per particle, adding significant complexity.

**Approach C: Deformation Gradient** (most physically correct, uses MPM F matrix):
If the MPM solver exposes the deformation gradient $F$ per particle, use its SVD
($F = U \Sigma V^T$) to define ellipsoidal kernels. The singular values give the
stretch along each principal axis. This is the gold standard for MPM surface
reconstruction but requires access to per-particle $F$ from the Genesis solver.

---

## 7. What Was Tried and Reverted

These changes were implemented and tested but reverted due to insufficient improvement
or unwanted side effects. Documented here to prevent re-trying without modifications.

### 7.1 Aggressive Parameter Tuning (reverted from commit `5156751`)

| Parameter | Changed To | Result |
|-----------|-----------|--------|
| `grid_res` | 96, then 128 | Better quality but higher cost. 96 is the sweet spot. |
| `_grid_coverage_ratio` | 2.0, then 2.5 | 2.0 improved quality. 2.5 was tight at 96³, fine at 128³. |
| `density_blur_sigma` | 1.0, 1.3, 1.5, 1.8 | Reduced bumps but did NOT eliminate contour lines. |
| `thresh` | 0.3 | Caused mesh bloat (isosurface too far from particles). |
| `influence_radius` | `3.0 × r_p` | Caused mesh bloat when combined with low threshold. |
| Taubin iterations | 6 | Helped slightly, insufficient alone for contour lines. |

**Key lesson**: Contour lines are NOT a density smoothing problem — they are a
marching cubes vertex constraint problem. Density smoothing helps with bumps but
cannot eliminate MC staircase artifacts. The fix must happen at the mesh level
(subdivision) or algorithm level (Flying Edges / different iso-surface method).

### 7.2 Moderate Settings (tested, worked well but reverted for simplicity)

`grid_res=96` + `coverage_ratio=2.0` was tested and produced good results:
"a lot better, cleaned up big bumps." The only issue was contour lines becoming
more visible (because the smoother surface made them stand out). This is a viable
baseline to return to.

---

## 8. Historical Context — Abandoned Approaches

### 8.1 SplashSurf + LBS Skinning + Edge Splitting (OLD implementation)

The original reconstruction approach (commits `f7ad0861` / `cb071630`) used:
1. SplashSurf to create an initial mesh from particles
2. Linear Blend Skinning (LBS) to deform the mesh each frame (k=6 nearest particles, Gaussian weights)
3. Dynamic edge splitting via `trimesh.remesh.subdivide_to_size` to handle stretched triangles
4. Two-pass weight transfer (barycentric interpolation + kNN fallback) for new vertices
5. Taubin smoothing (lambda=0.5, mu=-0.53, 2 iterations) for surface quality
6. Pre-computed sparse Laplacian matrix for efficient smoothing

**Why it was abandoned** — detailed root cause analysis:

**Root Cause 1: Destructive Subdivision.** The `process=True` flag in trimesh (which
calls `merge_vertices`) was the most dangerous operation. After `subdivide` created new
vertices precisely on edge midpoints, floating-point error could place them ~1e-8m
off-edge. The `process` step would then either merge vertices that should remain separate
(e.g., thin material folding on itself) or fail to merge T-junctions, producing
non-manifold topology.

**Root Cause 2: Non-Linear Weight Transfer.** New vertices were placed at the spatial
midpoint of deformed edges, but the assumption that `Pos(0.5 × (p1+p2)) = 0.5 ×
(Pos(p1) + Pos(p2))` breaks under non-linear deformation (metal folding). This produced
skinning weight discontinuities at subdivided edges.

**Root Cause 3: The Fallback Loop.** When weight transfer failed (returning `old_weights`
of length N for `new_verts` of length N+M), the code fell back to `init_skinning()`,
which re-bound the *deformed* mesh to particles. This baked the deformation into the
rest pose, creating permanent distortion that accumulated over time.

**Root Cause 4: Disabled Post-Strike Reconstruction.** The code that should have triggered
a full reconstruction after each strike was accidentally disabled (a double-assignment of
`should_send` in `teleop_socket.py` that always evaluated to `False`). This meant skinning
drift was never corrected, compounding degradation across strikes.

**Failed Experiments with the Old Approach:**
- **Plastic Skinning (dynamic offsets):** Tried updating vertex binding offsets
  continuously to let the mesh "flow." Result: *ballooning* — the mesh learned its own
  smoothed volume and spiraled into infinite expansion.
- **Tighter Binding (k=4):** Result: *shrink-wrapping* — mesh looked like "dried fruit,"
  revealing every particle imperfection.

**Verdict**: This approach is a dead end for large deformations. Do not revisit unless
building a full deformable tracking system with regularized deformation constraints
(e.g., Position Based Dynamics with spring constraints between neighbors).

### 8.2 SplashSurf I/O Bottleneck

SplashSurf itself has an architectural bottleneck as used in Genesis. When
`pu.particles_to_mesh(backend='splashsurf')` is called, it:
1. Exports particle positions to disk (serialization)
2. Spawns an external SplashSurf (Rust) process
3. Waits for mesh file output
4. Reads mesh back from disk (deserialization)

Even though SplashSurf is fast internally (CPU-parallel Rust), the disk I/O round-trip
and process spawning dominate. Additionally, it requires moving particle data from VRAM
to RAM. The current Taichi GPU splatting approach bypasses all of this.

### 8.3 Grid-Velocity Advection (disabled in old code)

An alternative to LBS that used MPM grid velocities to advect mesh vertices.
Disabled due to a Taichi bug where `grid_vel` returned tuples instead of velocity data.
Not relevant to the current implicit-field approach.

---

## 9. Alternative Reconstruction Approaches

Reference of approaches evaluated during research. The current implicit-field pipeline
was chosen as the best fit, but these may be relevant for future work.

### 9.1 Comparison Table

| Approach | Output | Speed (6k pts) | Temporal Stability | Topology Changes | Best For |
|----------|--------|-----------------|-------------------|-----------------|----------|
| **Implicit Field + MC** (current) | Triangle mesh | ~11ms | Good (with snap) | Excellent | Real-time colliders |
| **SplashSurf** (old backend) | Triangle mesh | ~50ms+ | Low (popping) | Excellent | Offline quality |
| **3D Gaussian Splatting** | Visual render (no mesh) | Real-time | Excellent | N/A | Pure visualization |
| **CNN-Grid (Zhao et al.)** | Triangle mesh | ~2s (2M pts) | Good | Excellent | High particle counts |
| **FlexiCubes** (NVIDIA) | Triangle mesh | Slow (optimization) | Low | Excellent | Sharp features |
| **Dual Contouring** | Triangle mesh | Moderate | Good | Excellent | Sharp edges/corners |
| **Narrow-Band SDF** (scikit-fmm) | SDF → mesh | ~60ms (50k pts) | Good | Excellent | Physics-accurate SDF |
| **Screen-Space Rendering** | Visual render | Very fast | N/A | N/A | Visual-only fluid |
| **Advect-and-Repair** | Triangle mesh | Medium | Perfect (Lagrangian) | Good (via remesh) | Tracking with topology |

### 9.2 3D Gaussian Splatting (3DGS)

Treat each MPM particle as a 3D Gaussian ellipsoid. Instead of extracting a surface,
update the Gaussian's position, rotation (deformation gradient $F$), and scale directly
from the MPM simulation state. This **completely bypasses surface extraction** (no MC).
Techniques like "Gaussian Splashing" (CVPR 2025) achieve 60+ FPS rendering of millions of
particles. However, this produces a visual render only — no triangle mesh for physics
colliders.

### 9.3 CNN-Accelerated Grid Reconstruction

Rasterize particle attributes (mass/density) onto the MPM sparse grid, pass through a
lightweight 3D U-Net to predict an SDF, then extract mesh via MC. Recent implementations
(Zhao et al. 2025) reconstruct surfaces for 2M particles in ~2.2 seconds on a single GPU
— a ~30× speedup over traditional anisotropic methods. Requires training data and GPU
memory for the network.

### 9.4 Narrow-Band SDF via scikit-fmm

An alternative to direct density splatting: count particles per grid cell, threshold to
get a binary indicator, then compute a proper SDF using fast marching:

```python
import skfmm
# Create binary indicator (inside/outside)
indicator = (particle_count_grid > threshold).astype(float)
phi = np.where(indicator > 0.5, -1.0, 1.0)
# Compute SDF
sdf = skfmm.distance(phi, dx=dx)
# Extract surface
verts, faces, _, _ = marching_cubes(sdf, level=0.0, spacing=(dx, dx, dx))
```

Produces a mathematically correct SDF (useful for physics), but the fast marching step
is slower than direct density splatting (~40ms at 100³ vs ~2ms for splatting).

### 9.5 Semi-Lagrangian Advection Between Rebuilds

Instead of full reconstruction every frame, rebuild the density/SDF every N frames and
advect it between rebuilds using the MPM grid velocity:

$$\phi^{n+1}(x) = \phi^n(x - v(x) \Delta t)$$

This is cheaper than full reconstruction and provides excellent temporal coherence.
Requires access to the MPM grid velocity field.

### 9.6 Screen-Space Rendering (Visual Only)

"Narrow-Range Filter for Screen-Space Fluids" (Truong & Yuksel): Project particles as
spheres to the depth buffer, apply a 2D screen-space filter. Completely sidesteps 3D
meshing. ~2.4× speed improvement and ~44% memory savings vs volume-based methods.
Only produces a visual render, not a mesh.

---

## 10. Reference Research Summary

### Key Findings Across All Research

Multiple research models independently converged on the same conclusions:

1. **Keep the implicit-field pipeline** (particles → density grid → iso-surface). It
   avoids the tearing/hole failure mode of dynamic remeshing.

2. **The grid sliding artifact** was the biggest issue. Fixed by snapping dx and
   grid origin (implemented in commit `ac55f3a`).

3. **Contour lines** require mesh-level fixes (subdivision or better iso-surface algorithm),
   not just density smoothing.

4. **Parameters should be derived from physics** (SPH theory + Nyquist), not tuned
   empirically. The framework in §4 provides the formulas.

5. **Anisotropic kernels** are the "right" long-term fix for deformation artifacts but
   are complex to implement. Worth pursuing after the simpler fixes are exhausted.

6. **Unity-side reconstruction** is the best path for high-fidelity visuals, keeping
   Python reconstruction for the collider mesh.

### Research Sources

| Document | Key Contribution |
|----------|-----------------|
| `reconstruction_artifact_analysis.md` | Detailed artifact characterization, performance profiling |
| `reconstruction_research.md` | Phased roadmap from immediate fixes to advanced techniques |
| `reconstruction_research_responses.md` | Concrete code patches for temporal smoothing, adaptive radius |

---

## 11. File Inventory & Cleanup

### Current Organization (as of 2026-02-19)

**Active documentation in `agforge/docs/`:**
- `RECONSTRUCTION_IMPLEMENTATION_PLAN.md` — this document (comprehensive guide)
- `DEBUG_MESH_TEARING.md` — documents why the old skinning approach was abandoned

**Deleted / consolidated** (~9,800 lines total absorbed into this plan):
- `reconstruction_artifact_analysis.md` — profiling data, artifact characterization → §1-3, Appendix C
- `reconstruction_research.md` — phased roadmap → §4-6
- `reconstruction_research_responses.md` — code patches → §6
- 8 research model output files from project root → §6, §8, §9, Appendix C-D

---

## Appendix A: Current reconstruction.py Quick Reference

```
File: agforge/reconstruction.py (341 lines)
Class: SurfaceReconstructor (ti.data_oriented)

Constructor: __init__(env, grid_res=128, backend='hybrid')
  - Allocates density + prev_density Taichi fields
  - Infers particle_radius from MPM solver
  - Sets influence_radius, grid_coverage_ratio, fixed_dx

State: get_state() / set_state(state) / reset()
  - Checkpointing support for simulation save/load

Main entry: update(should_reconstruct: bool)
  - Frame counter + interval gating → create_reconstructed_mesh()

Reconstruction: create_reconstructed_mesh()
  - Lines 177-324: Full hybrid pipeline
  - Grid snapping (lines 205-245)
  - Density splatting (lines 255-262)
  - Temporal blending (lines 264-268)
  - Gaussian blur (lines 272-274)
  - Marching cubes (lines 288-294)
  - Taubin smoothing (lines 303-318)

Taichi kernels:
  - _compute_density_kernel: GPU particle-to-grid splatting with (1-r²)³ kernel
  - _blend_density_temporal: EMA between current and previous density fields

Legacy: _create_splashsurf_mesh() — SplashSurf backend path
```

## Appendix B: Testing Checklist

After any reconstruction change, verify:

1. **No overflow warnings**: Check logs for "Particles exceed fixed grid" — if present,
   increase `grid_res` or reduce `_grid_coverage_ratio`
2. **Mesh not empty**: Check "max density below threshold" warnings
3. **No NaN vertices**: Check "Smoothing produced NaN" warnings
4. **Visual inspection**: Run a full press-release cycle in Unity and check for:
   - Contour lines on curved surfaces
   - Bumps/grapes at rest
   - Shimmer/ripples during pressing
   - Mesh bloat (surface extending beyond particle cloud)
   - Tearing or holes (regression to old problems)
5. **Performance**: Reconstruction should complete within ~15ms budget at 64³

---

## Appendix C: Performance Benchmarks

### Current Pipeline Profiling (64³, ~6k particles)

| Stage | Time | Notes |
|-------|------|-------|
| Full reconstruction (`teleop_recon_update`) | ~11.4ms avg | 1142 calls, 13025ms total |
| Physics step | ~11.3ms avg | 2072 calls, 23415ms total |

Reconstruction is **27.5% of active profiled time** (the single largest leaf-node hotspot).

### Per-Approach Cost Estimates

| Approach | Quality Gain | Time Cost | Complexity |
|----------|-------------|-----------|------------|
| Temporal smoothing (EMA) | ++ (wavy) | +0.5ms | Easy |
| Density-space Gaussian blur | ++ (bumpy) | +1ms | Trivial |
| Adaptive influence radius | ++ (bumpy) | +2ms | Easy |
| Grid 64→96 | +++ (both) | +20ms | Trivial |
| Grid 64→128 | ++++ (both) | +40ms | Trivial |
| Mesh subdivision + smooth | +++ (contour) | +3ms | Easy |
| Anisotropic kernels | +++++ (both) | +10ms | Hard |

### Scaling Reference (Counting Particles + scikit-fmm + MC)

| Particle Count | Grid Resolution | Rasterize | SDF (fmm) | MC | **Total** |
|---------------|----------------|-----------|-----------|-----|----------|
| 10k | 50³ | <1ms | 5ms | 3ms | ~10ms |
| 50k | 100³ | 2ms | 40ms | 15ms | ~60ms |
| 100k | 128³ | 4ms | 80ms | 30ms | ~115ms |
| 500k | 200³ | 15ms | 250ms | 80ms | ~350ms |

### Tool Comparison (for future library decisions)

| Method | Processor | Data Transfer | Scalability | Best For |
|--------|-----------|--------------|-------------|----------|
| SplashSurf | CPU (Rust) | High (VRAM↔RAM + disk) | Poor (>500k) | Offline exports |
| Taichi Kernel (current) | GPU | None (zero-copy) | Excellent | Real-time |
| PyTorch3D/Kaolin | GPU | Low (Torch interop) | High | ML pipelines |
| PyVista/VTK | CPU (C++) | Low (numpy) | Good | Fast CPU meshing |

---

## Appendix D: Platform Compatibility (macOS ARM64)

Relevant if targeting Apple Silicon for development/deployment:

| Tool | macOS ARM64 | Notes |
|------|-------------|-------|
| Unity 6 | Full | Native Metal support |
| Open3D (CPU) | Full | pip install works |
| Open3D (CUDA/GPU) | None | Falls back to CPU |
| PyTorch (MPS) | Works | ~3× slower than CUDA |
| Taichi (Metal) | Works | Can run custom GPU kernels |
| CuPy / CUDA | None | CUDA doesn't exist on Apple Silicon |
| NVIDIA Kaolin | None | CUDA-only |
| isoext / torch-mcubes | None | CUDA-only |
| PyVista/VTK | Full | CPU-based, cross-platform |
| scipy/skimage | Full | CPU-based, cross-platform |

**Key takeaway**: The current Taichi (Metal) + scipy/skimage pipeline is cross-platform.
If adding GPU meshing, use Taichi kernels (Metal backend) rather than CUDA-only libraries
to maintain macOS compatibility.
