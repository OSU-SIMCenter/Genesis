# Surface Reconstruction Architecture

> ⚠️ **UNMAINTAINED — verify against source before trusting anything here.** Checked 2026-08-14:
> this file has not been updated since the 316L material change or the coupling/scaling work, and
> at least one of its statements was found to be factually wrong (see the Roadmap's Bug 3). It
> still describes the billet as **AISI 4340** in places; the billet is **316L**. Living references
> are `docs/THERMOMECHANICAL_COUPLING_AND_SCALING.md` (coupling, scaling, metrics),
> `docs/316L_MECHANICAL_PROPERTIES.md` (the material card and its validity limits), and
> `agforge/docs/Contact_Method_Research_And_Plan.md` (contact).


**Last updated**: 2026-03-06
**Context**: Phase 5 Optimized Pipeline for Genesis MPM

## 1. Abstract
This document outlines the architectural decisions, mathematical optimizations, and historical context for the real-time (<33ms per frame) surface reconstruction pipeline in `agforge/reconstruction.py`.

The primary challenge of this pipeline is taking a discrete Material Point Method (MPM) point cloud simulated by Genesis—which uses algorithms native to Smoothed-Particle Hydrodynamics (SPH) for fluids—and aggressively filtering it mathematically so that the resulting 3D geometry appears as a **solid, continuous, rigid metal object** under plastic deformation.

## 2. The Performance Constraint (Real-Time / 30 FPS)
Since this reconstruction pipeline runs live during a real-time teleoperation simulation loop, the *entire* pipeline must execute in **under 33 milliseconds per frame** to maintain a 30 FPS target.

Without this constraint, we could use CPU-based algorithms like SplashSurf (which takes upwards of 120ms to 500ms per frame). Therefore, every optimization is a careful trade-off: maximizing visual smoothness while minimizing computational cost using GPU parallelization, caching, and algorithmic shortcuts. Our current implementation averages ~28ms per frame.

## 3. Issues Fought
Because we use fluid algorithms for solid objects, we fought the following artifacts:
1.  **The "Bumpy Grapes" Texture:** Isotropic (spherical) kernels cause the surface to bulge outwards at particle centers.
2.  **Temporal Shimmering (Rippling):** Marching Cubes vertices fluctuate due to tiny floating-point particle movements, causing a rippling effect.
3.  **Contour Banding (Staircasing):** A consequence of uniform scalar grid extraction leading to topographical contour lines.
4.  **CPU Bottlenecks:** Original implementations took >80ms to process a 10,000 particle mesh on the CPU.
5.  **Mesh Tearing / Holes (Abandoned LBS Approach):** Dynamic edge splitting on the CPU caused unrecoverable mesh tearing.

## 4. The Optimized Pipeline (Phase 5)
To solve these issues, we built an optimized **Hybrid GPU-to-CPU Pipeline**.

### Foundational Optimizations
1.  **Taichi GPU Density Splatting:** We splat particles into a $128^3$ voxel density grid entirely on the GPU.
2.  **Mathematical SPH Grounding:** We dynamically derive the grid cell size (`dx`) and particle influence radius from the MPM engine's baseline `particle_radius` to obey Nyquist limits.
3.  **Wendland C2 Kernel:** We use `(1-r)⁴(4r+1)` for a flatter center plateau than classic isotropic kernels.
4.  **Temporal Density Blending:** A state-based exponential moving average (EMA) applied to the density grid (alpha=0.8 moving, 0.2 at rest) freezes the surface when stationary.
5.  **Fixed Grid Snapping:** We lock the origin of our 3D grid to multiples of `dx`.
6.  **Percentile-Based Dynamic Thresholding:** We dynamically calculate the marching cubes isosurface threshold based on the 95th percentile of the current density field. This drastically reduces the mesh "breathing" or pulsating when density spikes occur, compared to the old method of slicing by a flat percentage of `max_density`.
7.  **PyVista Flying Edges:** We use `pyvista.ImageData.contour(method='flying_edges')` as a highly optimized extraction method.
8.  **Vertex Correspondence Blending:** We use `scipy.spatial.cKDTree` to blend current mesh vertices 15% towards the previous frame.
9.  **Taubin Volume-Preserving Smoothing:** 3 iterations of Taubin smoothing (`trimesh.smoothing.filter_taubin`) round off sharp edges perfectly.

### Final Rigidity Polish (Phase 5-7)
10. **Post-Smooth Analytical Normals:** We discarded flat face-normals and disabled PyVista's built-in normal computation. Instead, we compute analytical normals *after* the Trimesh Taubin smoothing pass is complete, eliminating faceted artifacts.
11. **Bilateral (Edge-Aware) Density Blur (Phase 7B):** Our 3-pass GPU Gaussian blur includes a Range Weight ($\sigma_r$) scaled dynamically by cell threshold density. This stops the blur across sharp density drop-offs, preserving hard geometric corners struck by the gripper, directly combating contour banding.
12. **MPM-Driven Anisotropic Splatting:** We extract the Deformation Gradient Tensor ($F$) from the Genesis MPM solver and volume-normalize it (to preserve mass) before computing the Left Cauchy-Green tensor $B = F F^T$. Our Taichi kernel computes the anisotropic warped distance metric $r^2 = (x - x_p)^T B^{-1} (x - x_p)$. This forcefully flattens the kernels into plates under extreme compression, completely eradicating the "bumpy grapes" effect.

## 5. Mathematical Parameter Framework
All parameters derive from a single ground truth: **`particle_radius` ($r_p$)**:
*   **Kernel Size (`influence_radius`):** $3.0 \times r_p$
*   **Sampling Rate (`dx`):** $R_{inf} / 3.0 = r_p$ (Nyquist optimal)
*   **Grid Coverage Ratio:** $3.0$
*   **Anti-Aliasing Filter (`density_blur_sigma`):** ~1.25 cells

## 6. Historical Context — Abandoned Approaches
### 6.1 SplashSurf + LBS Skinning + Edge Splitting
We previously used SplashSurf + Linear Blend Skinning (LBS) + dynamic edge splitting on the CPU.
**Why it failed:**
*   Destructive subdivision caused non-manifold geometry and tearing holes.
*   Non-linear weight transfer broke under metal folding.
*   SplashSurf's I/O bottleneck (disk serialization + out-of-process execution) was disastrous for real-time performance.

## 7. Future Roadmap Items
Based on research and team discussions, further improvements could include:
1. **Shape/SDF-Based Priors:** Incorporating geometric shape priors (like an SDF of the initial billet) into the active density field to heavily stabilize the mesh. This ensures it maintains a rigid, solid appearance under extreme compression, preventing the metal from looking unexpectedly "fluid" or soft.
2. **Semi-Lagrangian Advection Between Rebuilds:** Instead of running the full Flying Edges reconstruction every frame, rebuild the density field periodically and advect it between rebuilds using the MPM grid velocity for extreme temporal coherence.
3. **Stabilize Anisotropic Splatting:** Revisit singular value clamping of the principal stretches of $B$ without introducing the shear wave artifacts seen in Phase 7A.
4. **CNN-Accelerated Grid Reconstruction:** An offline-trained CNN that maps particle-to-grid density to a clean SDF, explicitly learning away particle noise while preserving sharp features.

---

## Appendix A: Alternative Reconstruction Approaches
During the Phase 1-4 research periods, numerous alternative approaches were evaluated. The current implicit-field pipeline was chosen as the best fit for Genesis's GPU architecture, but this context is preserved for future architectural decisions.

### 1. 3D Gaussian Splatting (3DGS)
Treats each MPM particle as a 3D Gaussian ellipsoid, updating position, rotation ($F$), and scale directly from the MPM state.
**Why abandoned:** It completely bypasses surface extraction, producing a visual render only. The system requires a solid triangle mesh to act as a physical collider for the robot gripper in subsequent teleop stages.

### 2. CNN-Accelerated Grid Reconstruction (Zhao et al. 2025)
Rasterizes particles onto a sparse grid, passes through a 3D U-Net to predict an SDF, then extracts the mesh.
**Why abandoned:** While fast (2M particles in 2s), it requires training data generation and consumes significant VRAM. It remains a strong candidate for future work if rule-based methods plateau.

### 3. Narrow-Band SDF via scikit-fmm
Computes a mathematically correct Signed Distance Field using Fast Marching over the binary thresholded grid.
**Why abandoned:** While it produces a perfectly smooth SDF that removes contour lines, the CPU-bound Fast Marching step took ~40-60ms, violating the 33ms real-time constraint. Direct density splatting was chosen instead.

### 4. Dual Contouring (DC)
Places one vertex *inside* each cube by minimizing a Quadratic Error Function using gradient data, naturally preserving sharp edges and corners.
**Why abandoned:** Much more complex to implement than Flying Edges/Marching Cubes on the GPU, though it effectively permanently solves staircasing and contour banding without aggressive blur.

### 5. MACET (Marching Cubes via Edge Transformations)
Moves edge endpoints along $\nabla{f}$ or parallel to the isosurface before triangulation to eliminate staircase artifacts without changing topology.
**Why abandoned:** Slower than Taubin smoothing and analytical normals, but a viable future alternative if contour lines return.

### 6. SplashSurf (Rust Backend)
**Why abandoned:** Used during Phase 1, but its architectural bottleneck (serializing positions to disk -> spawning an external Rust process -> deserializing the mesh) caused >100ms lag. Moving all density calculations to Taichi GPU eliminated this I/O bottleneck entirely.

### 7. Phase 7A: SVD Anisotropy Clamping
Attempted to clamp the singular values of the Anisotropic splatting tensor ($F$) to prevent the ellipsoid kernels from becoming thinner than the voxel grid resolution $dx$.
**Why abandoned:** Introduced aggressive directional "shear wave" artifacts and visual ripples across the surface during deformation. Reverted in favor of Phase 7B (Dynamic Bilateral Blur) which successfully solved contour banding without breaking the surface visually.

### 8. Marching Cubes 33 (MC33) & Marching Tetrahedra
Standard Marching Cubes suffers from topological ambiguity (e.g., when diagonally opposite corners of a voxel have the same sign, creating potential for internal holes). MC33 resolves all 33 topologically distinct cases, while Marching Tetrahedra splits voxels into 6 tetrahedra to guarantee consistency.
**Why abandoned:** `pyvista`'s Flying Edges implementation handles standard cases natively with extreme parallel CPU optimization. The performance cost of writing a custom MC33 or Tetrahedra extractor outweighed the rare occurrences of ambiguous holes in our C2 continuous smoothed density field.
