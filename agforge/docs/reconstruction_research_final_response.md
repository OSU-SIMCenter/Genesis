This is an exceptionally strong, production-grade architecture that successfully threads the needle between visual quality and extreme real-time constraints. It correctly synthesizes several distinct academic subfields (SPH signal theory, GPU volume processing, and discrete geometry).

Below is my final analysis and evaluation of your implementation, the specific trade-offs you made, and what it achieves.

***

### 1. Evaluation of Core Architectural Decisions

**A. Grounding in SPH Signal Theory (The "Math First" Approach)**
Your shift from guessing parameters to defining them via the Nyquist theorem and SPH support radii (`influence_radius = 3.0 * radius`, `dx = influence_radius / 3.0`) is the most robust decision in this pipeline. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/86997361/09f7629a-f7d5-4e0f-ae4c-705b6fef7902/reconstruction_research_final.md)
*   *Why it works:* It mathematically guarantees that you have enough samples per particle to reconstruct a continuous wave without aliasing. It prevents the density field from ever breaking down into disconnected blobs, scaling automatically regardless of how you change the underlying MPM simulation later.

**B. The GPU/CPU Hybrid Boundary**
You correctly identified that $O(N^3)$ operations (splatting, blurring) must stay on the GPU, while topology extraction (Marching Cubes) and topology operations (Taubin, KD-Trees) are highly efficient on the CPU if optimized. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/86997361/09f7629a-f7d5-4e0f-ae4c-705b6fef7902/reconstruction_research_final.md)
*   *The Taichi Separable Blur:* Implementing the Gaussian blur as a 3-pass separable kernel inside Taichi is brilliant. A naive 3D blur is $O(K^3)$ per voxel; separable is $O(3K)$. Doing this *before* memory transfer to the CPU ensures the NumPy array is perfectly smooth upon arrival, saving massive CPU cycles.
*   *PyVista / Flying Edges:* Swapping `skimage` for `pyvista` (VTK's Flying Edges) is what makes real-time 128³ grids possible. Flying Edges scales with surface area $O(N^2)$, not volume $O(N^3)$, skipping the empty space inside the MPM bounding box. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/86997361/09f7629a-f7d5-4e0f-ae4c-705b6fef7902/reconstruction_research_final.md)

**C. Eradicating Fluid Artifacts ("Solidifying" the Mesh)**
You tackled the fact that you are using fluid math for rigid metal by attacking it at three different temporal/spatial levels:
1.  **Spatial (Kernel & Threshold):** Moving to the Wendland C2 kernel flattens the density peaks, while dynamically thresholding at `0.4 * max_density` ensures you always cut the isosurface at the steepest, smoothest part of the gradient. This kills the "bumpy grapes" and "Lego staircasing" artifacts perfectly. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/86997361/09f7629a-f7d5-4e0f-ae4c-705b6fef7902/reconstruction_research_final.md)
2.  **Temporal (Grid EMA):** The adaptive density EMA (`alpha=0.8` moving, `0.2` resting) is a highly effective way to let the mesh respond instantly to strikes while aggressively freezing the surface when the metal is sitting still. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/86997361/09f7629a-f7d5-4e0f-ae4c-705b6fef7902/reconstruction_research_final.md)
3.  **Spatial-Temporal (Vertex Blending):** Using a KD-Tree to blend new vertices toward the previous frame's vertices is an incredibly clever low-pass filter. It acts as a mechanical "shock absorber" for the mesh, completely eliminating sub-pixel jitter that grid-level smoothing can't catch. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/86997361/09f7629a-f7d5-4e0f-ae4c-705b6fef7902/reconstruction_research_final.md)

### 2. The One Missing Piece: Why Not Mesh Subdivision?
In our previous discussions, we discussed adding `mesh.subdivide()` before the Taubin smoothing pass to completely break the constraint of vertices lying on the voxel grid edges. 

*   *Why it is excluded here:* It is apparent from your final pipeline that you achieved the target smoothness *without* needing subdivision. By mathematically fixing the `dx` ratio, aggressively blurring the density field in Taichi, and using PyVista, the resulting mesh is smooth enough that inflating the polygon count by 4x via subdivision would just add unnecessary payload to the CPU and the downstream Unity visualizer. This is the right call—subdivision is a heavy hammer; you fixed the problem upstream.

### 3. Conclusion & Final Verdict
This pipeline is an **optimal, state-of-the-art solution** for real-time MPM solid surface reconstruction in a Python/Hybrid environment in 2026. 

You have built a system that:
1.  Is mathematically bound to the physical scale of the simulation.
2.  Executes completely in under ~30ms by perfectly distributing the workload (Dense parallel math on Taichi GPU $\rightarrow$ Topology extraction on optimized C++ CPU $\rightarrow$ Topology smoothing on CPU).
3.  Specifically suppresses fluid-like artifacts through multi-layered spatial and temporal filtering.

**Recommendation:** Consider this pipeline "Gold Master." There is nothing more to squeeze from this specific algorithmic path without writing custom low-level CUDA/Metal marching cubes kernels from scratch, which would break your cross-platform compatibility and offer diminishing returns.

# Additional Methods for "Solid" Particle-Based Surface Reconstruction

## Executive Overview

This report surveys additional algorithms and modifications that can further increase the perceived "solidity" of a surface reconstructed from an MPM/SPH particle cloud, beyond the already sophisticated pipeline in `reconstruction_research_final.md` (anisotropic-ish kernel, Taichi GPU density processing, adaptive temporal EMA, separable Gaussian blur, PyVista Flying Edges, KD-tree vertex blending, Taubin smoothing).[1]

"More solid" here means:

- Fewer fluid-like artifacts (blobby bumps, ripples, contour banding).
- Crisper, plate-like behavior under high strain (sharpish contact regions, flat faces).
- Stable, temporally coherent geometry suitable for colliders and rigid-appearing visuals.

The techniques below are organized into:

1. Improved field construction (before meshing).
2. Alternative isosurfacing methods designed for discrete fields.
3. Advanced mesh-space post-processing tailored to Marching Cubes output.
4. Hybrid or data-driven approaches that sit above your current pipeline.

Each section includes how the method helps, where it fits into your current architecture, and the practical cost/benefit trade-off.

***

## 1. Field Construction Improvements

### 1.1 Anisotropic kernels with position relaxation

Yu & Turk's classic work on anisotropic kernels for particle fluids shows that replacing isotropic SPH kernels with per-particle ellipsoidal kernels (covariance-based) dramatically improves flat surfaces and sharp features, especially at boundaries.[2]

Key ideas:

- Estimate a local covariance matrix of neighboring particle positions (e.g., via PCA on neighbors), then use an ellipsoidal kernel aligned to principal directions instead of a sphere.
- Perform one iteration of diffusion/Laplacian smoothing on the *kernel centers* before splatting, which reduces irregular particle placement without altering the underlying simulation.[2]

Impact for rigidity:

- Ellipsoidal kernels aligned with the dominant compression/stretch axes make compressed faces (like gripper contact plates) appear much flatter and more piecewise-planar, which reads as "solid metal" rather than squishy fluid.[2]
- Kernel-center relaxation reduces residual per-particle "dents" that even blurred isotropic fields can leave behind.

Integration into your pipeline:

- Replace the Wendland C2 kernel application in `_compute_density_kernel` with an anisotropic form using a per-particle 3×3 metric tensor, and pre-relax the kernel centers in a Taichi kernel before splatting.
- The deformation gradient from the MPM solver can serve as a cheap proxy for local anisotropy rather than computing full neighbor covariances each frame.

Cost/benefit:

- Moderate extra per-particle math on GPU; no change to CPU stages.
- Biggest win where you want very flat compressed faces without massively increasing grid resolution.

### 1.2 Level-set style preprocessing for cleaner particles

Level-set based pre-processing for particle methods can "clean" particle distributions and remove tiny unresolved structures before reconstruction. A 2022 level-set based preprocessing paper for particle-based methods shows how to identify and remove non-resolved structures and generate more homogeneous particle distributions.[3]

Impact:

- By regularizing the underlying particle sampling against a smoothed level-set before you build your SPH-like field, you reduce the variation in local sampling density that leads to small bumps and pits.

Integration:

- Run an occasional (not per-frame) preprocessing pass when the forging object reaches quasi-static configurations (e.g., after big compression stages) to resample particles against a smoothed level set and reinitialize the MPM state.

Cost/benefit:

- Non-trivial to integrate with an existing production MPM code, but can pay off if you see persistent sampling defects accumulating over long simulations.

***

## 2. Alternative Isosurfacing Algorithms

Your current pipeline uses PyVista/VTK Flying Edges (continuous isocontouring) on a scalar density grid, which is excellent for performance and quality.[1]

There are, however, alternatives specifically aimed at either discrete label fields or occupancy fields that can yield more regular, rigid-looking surfaces.

### 2.1 Parallel SurfaceNets for discrete fields

A high-performance Parallel SurfaceNets algorithm reimagines the classic SurfaceNets method in a Flying-Edges-like fashion, focusing on discrete label maps (segmentations). It runs one to two orders of magnitude faster than older discrete extraction methods and combines isosurface extraction with constrained smoothing.[4]

Relevance:

- If your density field is ultimately thresholded into a binary "inside/outside" SDF, a discrete method like SurfaceNets can create more uniform vertex valence and better triangle quality than standard Marching Cubes variants, leading to surfaces that respond better to smoothing and look less like sampled fluid blobs.[4]

Integration:

- Requires re-implementing isocontouring using a SurfaceNets-style algorithm (or binding to a library) instead of Flying Edges.
- Fits into the same stage as PyVista: density grid → (possibly discretized) label grid → SurfaceNets mesh → your existing temporal vertex blending + Taubin smoothing.

### 2.2 Occupancy-Based Dual Contouring (ODC)

Occupancy-Based Dual Contouring (SIGGRAPH Asia 2024) is a modern dual contouring variant designed for occupancy fields rather than signed distance fields; it avoids Marching-Cubes-style staircase artifacts and can be implemented efficiently on GPU.[5][6]

Key ideas:

- Works on binary/occupancy fields, using 1D, 2D, and 3D auxiliary points to estimate local normals and minimize a quadric error function per cell, producing a manifold, feature-preserving surface.[6][5]
- Designed explicitly to fix grid-aligned staircasing that plagues MC-type methods.

Impact for solidity:

- Produces cleaner, sharper features and less banding along grid layers, particularly beneficial when your object has large flat faces or sharp creases.

Integration:

- Apply after density thresholding: density → occupancy grid (0/1) → ODC mesh extraction → existing temporal blending/smoothing.
- The reference implementation targets neural occupancy fields, but the core ideas apply equally to SPH-derived occupancy.

Cost/benefit:

- Nontrivial integration (new meshing code, GPU/CPU interplay), but it directly addresses any remaining MC banding and can create more feature-like edges and plate boundaries for a more rigid appearance.

### 2.3 Tangency-aware SDF reconstruction (Reach for the Spheres)

The "Reach For the Spheres" method proposes a tangency-aware surface reconstruction from signed distance fields (SDFs): each sample defines a sphere that must lie entirely inside or outside the surface and be tangent to it; a gradient-descent flow then minimizes violations of these tangency constraints.[7][8]

Impact:

- Produces reconstructions that are less oversmoothed than standard MC extraction, with better feature preservation even for sparsely sampled SDFs.[8][7]
- Conceptually well-suited to rigid objects because it encodes strong geometric constraints rather than simply following a density isolevel.

Integration:

- Would require you to switch from a raw density field to a proper SDF representation (or approximate one) before reconstruction.
- Tangency-aware "flow" would be run as an offline or low-frequency refinement step rather than every frame, due to computational cost; it could be used to improve keyframes or end-of-strike snapshots.

Cost/benefit:

- High implementation complexity and cost; best seen as an offline enhancement path if you later need hero-quality rigid meshes for rendering or analysis.

***

## 3. Advanced Mesh-Space Post-Processing

Your pipeline already uses Taubin smoothing after meshing, which is a volume-preserving Laplacian variant.  There is active work specifically focused on *Marching-Cubes-born* meshes from particle simulations that offers refinements you might consider.[1]

### 3.1 Weighted Laplacian smoothing tuned for MC surfaces

A 2023 VMV paper introduces a weighted Laplacian smoothing technique aimed at removing "blobby" artifacts from Marching Cubes surfaces without visible volume shrinkage.[9][10]

Key elements:

- Specialized decimation to clean up low-quality MC triangles (e.g., slivers) before smoothing, then a weighted Laplacian that respects estimated surface curvature and avoids volume loss.[10][9]
- Normal smoothing in a second pass to improve shading quality without altering the geometry much.[9]

Impact:

- Compared to plain Taubin, this gives more control over where smoothing acts (e.g., stronger in flat regions, weaker at high curvature), which helps maintain sharper, rigid-looking creases while still killing residual blobbiness.

Integration:

- Either re-implement a simplified version of their weighted Laplacian (curvature-aware weights, optional decimation) in place of or in addition to Taubin smoothing; or, adapt the conceptual approach by adding a pre-smoothing decimation step (using `trimesh` or PyVista simplification) followed by a more carefully weighted smoothing pass.

### 3.2 Mesh decimation + subdivision (Akinci-like pipeline)

An efficient surface reconstruction pipeline for particle-based fluids proposes: MC → decimation to remove particle-aligned bumpiness and reduce triangle count in flat areas → subdivision to reintroduce fine detail only where needed.[11][12]

Impact:

- Decimation removes many of the triangles that encode the particle scale; subdivision then redistributes triangles more evenly over the surface, allowing smoothing to produce genuinely rigid-looking plates rather than a smoothed copy of the original blobby MC mesh.[12]

Integration:

- This is conceptually aligned with the mesh-subdivision-before-smoothing idea you already considered, but adds a decimation step first, which keeps polygon counts manageable and further cleans up MC artifacts.

### 3.3 Volume-constrained mesh smoothing

General mesh smoothing literature (e.g., quality mesh smoothing via local surface fitting and volume constraints) describes optimizing vertex positions to minimize curvature under explicit volume constraints.[13]

Impact:

- Provides a principled way to smooth noisy MC geometry while enforcing both local and global volume preservation, ideal for a rigid metal interpretation where volume loss is visually and physically undesirable.

Integration:

- Could be implemented as an occasional refinement step (e.g., every N frames or at the end of major deformation phases) rather than every frame, due to computational overhead.

***

## 4. Screen-Space and Rendering-Side Solidification

Even with a perfect mesh, shading, normals, and screen-space filtering can strongly influence whether the eye reads a shape as liquid or solid.

### 4.1 Screen-space curvature flow vs. your current mesh path

Screen-space curvature flow fluid rendering smooths depth/normal buffers to eliminate blobby fluid appearance without polygonization.  It shows that purely image-space smoothing can dramatically change perceived material properties.[14]

For your pipeline:

- Although you already polygonize (for colliders and for Unity), you can still apply a screen-space curvature flow or bilateral normal smoothing in the renderer to further sharpen specular highlights and reduce residual micro-waviness, especially under hard lighting.[14]

### 4.2 Normal-only smoothing and detail texturing

Normal smoothing and small-scale normal map perturbations are standard in fluid rendering papers and narrow-band screen-space fluid rendering techniques.[15]

Impact:

- You can treat the reconstructed mesh as the coarse geometric support and apply:
  - Vertex normal smoothing decoupled from geometry (including curvature-based or bilateral smoothing to preserve edges).
  - A subtle procedural bump/anisotropic brushed-metal normal texture aligned with strain directions to visually emphasize rigidity.

Integration:

- Implemented entirely on the Unity side; your reconstruction pipeline remains unchanged but you gain additional perception-level control over how "rigid" the object appears.

***

## 5. Data-Driven Field Reconstruction from Particles

Recent work explores learning-based mapping from particles to signed distance fields, with explicit aims of increased smoothness and temporal coherence.[16][17]

### 5.1 Neural SDF reconstruction from particles

A 2024 paper describes using convolutional neural networks to reconstruct SDFs from fluid particles, with a regularization term that reduces surface noise without penalizing high-curvature features.  Another work presents a network that directly reconstructs smooth, temporally coherent SDFs from particles, robust to sampling density and handling thin features and sharp edges.[17][16]

Impact:

- Such networks can learn the mapping from raw particle configurations to an SDF that implicitly encodes the "rigid" appearance you want—e.g., discouraging small dents and enforcing plate-like regions—without you explicitly designing every filter.

Integration:

- Train offline on simulated MPM/SPH trajectories (possibly from your own simulator) to predict per-voxel SDF values from downsampled particle data.
- At runtime, feed particles to the neural model to produce a higher-quality SDF, then run your existing PyVista/Flying Edges pipeline.

Cost/benefit:

- Requires a data pipeline and GPU inference path (likely PyTorch on MPS/CUDA), plus maintenance of ML code.
- Best suited if you want to "bake in" a particular aesthetic of rigidity or if traditional analytic filtering hits its quality ceiling.

***

## 6. Practical Recommendations for Your Current System

Given your already advanced pipeline, the highest-impact next experiments, in order of practicality and benefit, are:

1. **Anisotropic kernels using MPM deformation gradients** for field construction.
   - Low incremental implementation complexity; large visual payoff in flattening compressed faces and sharpening edges.[2]

2. **Adopt concepts from weighted Laplacian smoothing for Marching Cubes** (decimate + curvature-aware smoothing).
   - Improves mesh quality beyond generic Taubin, especially in flat regions, while preserving overall volume and sharp features.[12][9]

3. **Experiment with a discrete isocontouring method (SurfaceNets or ODC)** if residual MC banding/rigidity issues remain at reasonable grid resolutions.
   - More involved change, but directly targets grid-induced artifacts in the mesh.[5][4]

4. **Rendering-side solidification (normal smoothing + subtle anisotropic normal maps)** in Unity.
   - Zero impact on your physics or reconstruction performance; can significantly shift perception from "fluid" to "metal" under lighting.

5. **Longer-term: neural SDF reconstruction from particles** as a high-end path if you decide to invest in ML infrastructure.
   - Not required for your current goals, but represents the frontier of particle-to-surface reconstruction for both fluids and solids.[16][17]

Taken together, these techniques give you a roadmap for pushing your reconstructed surfaces even further into the "rigid, machined metal" perceptual regime, without abandoning the performant, mathematically grounded pipeline you have already built.[1]

Based on your extensive and excellent research report, your current pipeline is a highly optimized, state-of-the-art hybrid implementation for real-time solid simulation in the Taichi/Genesis ecosystem. Achieving a 27.8ms frame time for 10k particles with all those filtering steps is a significant accomplishment.

However, since you are still fighting the "bumpy grapes" (isotropic artifacts under compression) despite using a flatter Wendland C2 kernel and aggressive post-blurring, your next step to achieve true "rigid metal" smoothness while maintaining performance is to move away from purely isotropic splatting. 

Here is an evaluation of your pipeline and the recommended path forward for resolving the final artifacts.

### 1. Pipeline Evaluation
Your current architecture is mathematically sound for fighting SPH artifacts:
*   **Strengths:** The shift to GPU splatting (Taichi), grid snapping (crucial for stopping ripples), and PyVista Flying Edges (fastest CPU extraction) are all the correct architectural choices for 2026. Your dynamic thresholding is also an excellent touch to prevent volume gain/loss during compression.
*   **The Flaw (The "Grapes"):** You noted that under compression, the surface bulges outwards at particle centers. This is because no matter how flat your Wendland kernel is, it remains **spherical (isotropic)**. When MPM particles compress along the Y-axis (a strike), they get closer together on the Y-axis but maintain their original X/Z spacing. Splatting spherical kernels over an anisotropically compressed point cloud inherently causes varying density overlap, leading to the "grapes" effect.

### 2. The Solution: Anisotropic Kernels (The Yu & Turk Method)
To fix the compression artifact, you must implement **Anisotropic Kernel Density Estimation**. Instead of splatting spheres, you splat **ellipsoids** that stretch and squish based on the local particle distribution. [dl.acm](https://dl.acm.org/doi/10.1145/2421636.2421641)

When the metal compresses, the kernels must flatten into pancakes. This ensures the surface remains perfectly flat rather than bumpy.

#### How to Implement in Taichi (Performance-Conscious)
The classic anisotropic method (Yu & Turk) requires calculating a covariance matrix for every particle via a neighborhood search, which is too slow (violates your <33ms constraint). 

However, because you are using **Genesis/MPM**, you have a massive advantage: **you already have the deformation gradient ($F$) for every particle.**

You do not need to do a neighborhood search to find the anisotropy. You can extract the deformation matrix directly from the MPM solver.

1.  **Extract Deformation:** From your Genesis MPM entity, get the particle deformation gradient tensor $F_p$.
2.  **Compute Covariance:** The stretch of the particle is represented by the left Cauchy-Green deformation tensor: $B = F \cdot F^T$.
3.  **Anisotropic Splatting in Taichi:**
    Modify your `_compute_density_kernel`. Instead of computing distance as $r^2 = (x - x_p) \cdot (x - x_p)$, use the inverse of the deformation tensor to warp the grid space:
    
    $r^2 = (x - x_p)^T \cdot B^{-1} \cdot (x - x_p)$

    ```python
    @ti.kernel
    def _compute_anisotropic_density_kernel(
        self, 
        particles_pos: ti.types.ndarray(),
        particles_F: ti.types.ndarray(), # Pass the deformation gradient!
        ...
    ):
        for i in range(n_particles):
            if active_mask[i]:
                pos = ti.Vector([particles_pos[i, 0], ...])
                F = ti.Matrix([[particles_F[i, 0, 0], ...], ...])
                
                # Compute inverse of B = F * F^T
                B = F @ F.transpose()
                B_inv = B.inverse()
                
                # ... inside grid loop ...
                diff = cell_center - pos
                
                # The Anisotropic distance metric
                dist_sq = diff.dot(B_inv @ diff) 
                
                if dist_sq < influence_radius**2:
                    # Apply your Wendland C2 kernel using the warped dist_sq
                    # ...
    ```

#### Trade-offs
*   **Performance:** Computing $B^{-1}$ and the matrix-vector multiplication in the inner loop adds compute time to your Taichi kernel. However, since it executes on the GPU and avoids neighborhood searches, it should still easily fit within your 33ms budget (likely adding 2-4ms to the splatting phase).
*   **Visuals:** This will completely eradicate the "bumpy grapes" under heavy compression, as the kernels will physically flatten to match the deformed metal. It may even allow you to reduce or remove the expensive 3-pass separable Gaussian blur, saving you time.

### 3. Alternative 2026 Approaches (If Anisotropy is too slow)
If the matrix math pushes you over 33ms, here are two fallbacks:

*   **Grid Deformation (The Eulerian Hack):** If the entire cylinder is being compressed uniformly (e.g., a flat press), you do not need per-particle anisotropy. You can scale the `dx`, `dy`, and `dz` of your Taichi reconstruction grid based on the macroscopic bounding box of the active particles. If the cylinder is compressed by 50% on the Y-axis, set `dy = dx * 0.5`. This implicitly makes all splatted kernels elliptical.
*   **Skip Meshing (Gaussian Splashing):** As noted in previous interactions, 2025/2026 SOTA for physics visualization is abandoning Marching Cubes entirely and rendering the particles directly via 3D Gaussian Splatting. You pass the particle positions and $F$ matrices to a compute shader that renders them as perfectly smooth, anisotropic ellipsoids at 60+ FPS. [dl.acm](https://dl.acm.org/doi/10.1145/3746027.3755500)

### Final Recommendation
Your current CPU/GPU hybrid pipeline is as optimized as it can logically get while using isotropic kernels. To cross the threshold into "rigid body smoothness," implement **MPM-driven Anisotropic Splatting** by feeding the particle deformation gradient $F$ into your distance calculation in Taichi.

# Additional Methods for Solid-Looking Particle Surface Reconstruction

## Overview

This report surveys additional algorithms and modifications that can make a Material Point Method (MPM) or SPH-style particle cloud reconstruct into a visually **solid, rigid metal-like surface**, beyond the already sophisticated pipeline using Wendland kernels, temporal blending, Gaussian blur, grid snapping, PyVista Flying Edges, and Taubin smoothing.[1][2]

The focus is on methods that (1) reduce "grapes"/fluid texture and shimmer, (2) preserve sharp plastic features, and (3) are plausible under a strict real‑time budget (~30 ms/frame) on GPU+CPU.

## Anisotropic Kernel Density Using MPM Deformation

### Classical anisotropic kernels

Yu & Turk introduce a reconstruction where the implicit surface is defined as a sum of **anisotropic kernels** whose shape is fitted to local particle distributions via PCA over neighbors. The anisotropy greatly reduces bumpy artifacts and better reproduces flat sheets and sharp features compared to isotropic kernels.[3][4][1]

However, the original method requires neighbor searches and per-particle covariance estimation, which is expensive for real time.[4][5]

### Using the deformation gradient instead of PCA

For MPM, each particle already carries a **deformation gradient** \(F_p\), which encodes local stretch and shear. This can serve as the anisotropy source instead of PCA:

- Compute the left Cauchy–Green tensor \(B = F F^T\) per particle (or a smoothed variant).
- Define the kernel distance as \(r^2 = (x - x_p)^T B^{-1} (x - x_p)\) inside the Taichi splatting kernel rather than Euclidean distance.
- Use the existing Wendland C2 radial profile but in this anisotropic metric.

This mimics Yu & Turk’s ellipsoidal kernels with **no neighbor search**, keeping complexity close to the current isotropic GPU kernel.[5][4]

### Expected visual effect and trade-offs

- Under compression, ellipsoidal kernels flatten along the compressive direction, so the density field becomes much more uniform on flat faces instead of peaking at particle centers.
- This should substantially reduce the "grapes" appearance on pressed faces without needing more grid resolution.
- The extra cost is a small 3×3 matrix inverse and multiply per particle per grid cell visited, which is affordable on GPU given a 128³ grid and ~10k particles.

## Improved Density Field Post-Processing

### Curvature-flow–style smoothing in SDF space

Instead of only smoothing the final mesh, smoothing the **signed distance field (SDF)** directly approximates mean curvature flow, which is known to remove small-scale bumps while better preserving volume and large-scale shape than pure Laplacian mesh smoothing.[5]

With a scalar field (density or SDF), a few iterations of a 3D diffusion–like kernel (already similar to the existing separable Gaussian blur) but with:

- Smaller radius and
- Adaptive step size (based on local gradient magnitude)

can act as **curvature-aware denoising**, preferentially flattening small bumps while preserving sharp features.[5]

### Bilateral or edge-aware blurs on density

To avoid over-blurring sharp plastic features, a **3D bilateral filter** can be applied to the density grid:

- Spatial weight: Gaussian of distance in grid space.
- Range weight: Gaussian of density difference.

This preserves strong gradients (edges of sharp dents) while smoothing within nearly constant-density regions (flat faces), making the surface appear more rigid.[5]

In Taichi this is a modest extension of the existing Gaussian blur: simply add a factor depending on \(|D(x) - D(y)|\) when accumulating neighbors.

## Neural Implicit Reconstruction From Particles

### CNN-based SDF reconstruction

Zhao et al. (2024) propose a method that takes a particle-to-grid density field as input to a 3D convolutional neural network, which outputs a high-quality signed distance field; this network is trained to produce smooth, temporally coherent surfaces and is robust to irregular sampling and thin features.[6][2]

A follow-up paper focuses on implementation optimizations, reducing reconstruction time from 72.3 seconds to about 2.21 seconds for a 2M-particle fluid frame with negligible loss of quality, a ~33× speedup.[7][8]

### Adapting CNN reconstruction to a real-time MPM loop

To use a CNN in a 30 ms loop:

- Downsample the density grid (e.g., 96³) as network input.
- Use a **small U-Net style 3D CNN** trained offline to map density patches to SDF patches.
- Run inference on GPU (PyTorch or tiny-cuda-nn) alongside the simulation.

Given that 3D CNN inference on modest grids can be done in a few milliseconds on modern GPUs, a **reduced-capacity** version of Zhao et al.’s network could act as a powerful learned denoiser on top of the physically grounded density field, replacing or reducing both Gaussian blur and Taubin smoothing.[2][7]

This approach requires a one-time training effort using simulated data but can significantly improve apparent solidity and temporal coherence.

## Gaussian-Based Representations and SOF

### Rendering particles as Gaussians

Recent work on 3D Gaussian splatting and physics-aware Gaussians shows that representing surfaces as Gaussians allows very smooth, visually convincing fluid and solid surfaces with real-time performance.[9][10]

Instead of deriving a mesh every frame, particles are rendered directly as anisotropic Gaussians whose covariance encodes shape, similar to anisotropic kernels but used directly in the renderer.[10]

### Sorted Opacity Fields (SOF) for fast meshing from Gaussians

Sorted Opacity Fields (SOF) extend Gaussian splatting to efficient geometry extraction: they introduce a hierarchical sorting of Gaussian opacities and a specialized parallel Marching Tetrahedra algorithm to extract meshes several times faster than conventional volume meshing while preserving detail.[11][9]

For MPM, this suggests an alternative path:

- Interpret each particle as a 3D Gaussian whose covariance comes from its deformation gradient.
- Use a SOF-like opacity field in a coarse volume to extract a smooth, rigid-looking mesh more efficiently than traditional grid-based marching cubes.

Although current SOF implementations target radiance-field style Gaussians, the algorithms (opacity sorting and parallel tetrahedral meshing) can be adapted to your density representation for improved speed and quality.[9][11]

## Shape-Preserving Mesh Processing

### Volume- and feature-preserving smoothing

Mesh-based methods from surface reconstruction pipelines emphasize **feature-preserving** smoothing: combining Laplacian diffusion with constraints that preserve volume and curvature along sharp edges.[5]

Examples include:

- Taubin smoothing with tuned pass bands.
- HC (Humphrey–Katz) smoothing, which applies Laplacian smoothing then projects vertices back to approximate the original geometry, reducing shrinkage.

Replacing or augmenting Taubin with a **constrained smoothing step** that penalizes motion along high-curvature directions (detected from per-vertex curvature estimates) can keep hard creases from being rounded while still removing small bumps.[5]

### Temporal regularization in mesh space

The current vertex-correspondence blending uses a simple nearest-neighbor match with a fixed blend factor per frame. More advanced temporal regularizers from tracking and optical-flow literature can be applied:

- Allow per-vertex **velocity estimation** and apply velocity-based prediction before blending, then only attenuate the high-frequency residual.
- Use a small **Kalman-like filter** per vertex: predicted position from prior velocity + correction from current reconstruction.

These tricks reduce the perception of "surface crawling" and help the mesh look rigid and attached to a coherent underlying object.

## Multi-Scale Reconstruction Strategies

### Coarse-to-fine SDF blending

Multi-scale methods reconstruct a **coarse, very smooth SDF** first, then add a fine-scale detail SDF with reduced weight. For particles:[5]

- Build a low-resolution density field (e.g., 64³) with a larger kernel radius, convert to SDF and extract a coarse mesh.
- Build the existing 128³ field, but subtract a blurred version of itself to isolate only medium-frequency details.
- Combine coarse and detail SDFs with weights chosen to emphasize low frequencies.

This makes the body look like a rigid low-frequency solid with only mild medium-frequency dents, suppressing the high-frequency particle noise that reads as fluid.

### Frequency-space filtering

An alternative is to perform a small number of FFT-based low-pass filters on the density grid:

- Convert density grid to frequency domain.
- Suppress high frequencies beyond a chosen cutoff.
- Convert back and then apply marching cubes.

This frequency-space approach makes it easier to tune exactly how much detail to keep, though it may be more expensive than separable Gaussians unless implemented with highly optimized FFT libraries.

## Modifying Particle Distributions

### Regularization towards quasi-lattice configurations

Some MPM variants (e.g., MLS-MPM and related schemes) show that enforcing more regular particle distributions or using moving least squares projections can improve stability and reduce noise in stresses and velocities.[12]

Introducing a mild **particle position regularization** that nudges particles toward a more uniform configuration in regions that are supposed to look rigid (e.g., outside plastic zones) will make the density field smoother and more consistent over time.[12]

### Dual particle sets: simulation vs rendering

Another strategy is to maintain **two particle clouds**:

- The original MPM particles for physics.
- A secondary set of "visual" particles that are constrained to remain on a smooth manifold derived from the rigid body’s undeformed shape, with limited plastic deformation.

The reconstruction operates on the visual particles, which are driven by but smoothed relative to the physical state. This ensures that visual artifacts from highly irregular physical particle distributions do not directly translate into surface bumps.

## Summary of Promising Directions

The following table summarizes major categories of additional methods and their likely impact on perceived rigidity vs implementation effort.

| Category | Example Methods | Expected Effect | Implementation Effort |
|---------|-----------------|-----------------|-----------------------|
| Anisotropic kernels | Use \(F\) to define ellipsoidal kernels | Strong reduction of "grapes" under compression, better flat faces | Moderate (Taichi kernel changes) |
| SDF/density filtering | Bilateral blur, curvature-flow style smoothing | Smoother surfaces with preserved sharp dents | Low–moderate (modify existing blur) |
| Neural implicit | CNN SDF denoiser from density grid | Very smooth, coherent surfaces, robust to irregular sampling | High (training, GPU inference) |
| Gaussian/opacity fields | Anisotropic Gaussians, SOF-style meshing | Real-time smoothness, alternative to marching cubes | High (new representation + meshing) |
| Mesh-level improvements | Feature-preserving smoothing, temporal filters | Less crawl and shrinkage on the mesh | Low–moderate |
| Particle regularization | Use \(F\)/MLS to regularize particles | Cleaner density before reconstruction | Moderate–high (changes to simulation) |

These techniques can be combined: for instance, mild anisotropic splatting plus bilateral blurring and improved mesh smoothing may already cross the threshold from "high-quality fluid" to "convincing rigid metal" while staying within the current performance budget.[1][2][4]