Got it — v2 looks like a real “Phase 5” polish pass, and the new additions are directionally the right kind of upgrades for making the surface read as rigid/solid (especially under specular lighting). The key is: most of these don’t actually *change* the extracted geometry much; they either (a) change the implicit field so the geometry is intrinsically less “fluid”, or (b) change shading so remaining banding is hidden.

## What’s strongest in v2

### Anisotropic splatting using \(F\) (via \(B = FF^T\))
This is a very solid idea: Yu & Turk’s anisotropic-kernel reconstruction is a well-established way to get flatter surfaces and sharper features from particles by stretching kernels into ellipsoids based on local structure.  Your “MPM-driven” anisotropy is basically a physically-motivated shortcut for the same effect (using deformation instead of PCA-neighborhood covariance), which is perfect for MPM where \(F\) is already there. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/86997361/41a4f3b5-bf25-4b1f-8e79-f68f98290601/reconstruction_research_final_v2.md)

Practical caution: in your snippet you compute `dist_sq = diff.dot(B_inv @ diff)` and then compare it to `influence_radius**2`; that’s mixing a *dimensionless* anisotropic metric with a *length-squared* cutoff.  The fix is to define \(r^2 = (x-x_p)^T B^{-1}(x-x_p)\) and compare to a *dimensionless* threshold (e.g., \(r^2 < 1\)), or to scale the metric so units are consistent. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/86997361/41a4f3b5-bf25-4b1f-8e79-f68f98290601/reconstruction_research_final_v2.md)

### Bilateral blur in the density grid
Bilateral filtering is the right conceptual tool for “melt within regions, preserve edges,” because it combines a spatial Gaussian with a range (value-difference) Gaussian.  Iterating a bilateral filter can even push results toward piecewise-constant (more “machined plate” looking), but that can also look artificial if overdone. [sites.units](https://sites.units.it/ramponi/teaching/DIP/DIPmaterials/z08_Bilateral_filter.pdf)

### Analytical gradient normals
Using contour gradients for normals is a classic way to make MC/isosurface meshes shade smoothly and hide residual banding in lighting.  Also note VTK’s own docs: computing normals/gradients is non-trivial cost, and if you then run topology/geometry-modifying filters (like smoothing) you may want to compute normals *after* those operations. [vtk](https://vtk.org/doc/nightly/html/classvtkDiscreteFlyingEdges3D.html)

## Do these guarantee a more “solid” look?
Mostly yes, with one important nuance:

- **Normals** can hide contour lines under lighting but do not remove geometric banding if it’s visible in silhouette. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/86997361/41a4f3b5-bf25-4b1f-8e79-f68f98290601/reconstruction_research_final_v2.md)
- **Bilateral + anisotropic kernels** actually change the implicit field, so they can truly improve the geometry (including silhouette) without requiring a bigger grid. [dl.acm](https://dl.acm.org/doi/10.1145/2421636.2421641)

## One high-leverage “solid” upgrade you still may add (low bloat)
If you still see fluid-y micro-waviness: consider adopting ideas from *mesh-space* smoothing designed for particle/MC surfaces, e.g., weighted Laplacian approaches aimed at suppressing blobby artifacts while avoiding shrinkage.  This often gives better “solid” results than just increasing Taubin iterations. [diglib.eg](https://diglib.eg.org/bitstream/handle/10.2312/vmv20231245/219-227.pdf)

## Two small implementation suggestions (to avoid regressions)
- Compute normals after Taubin smoothing (or recompute them at the end), because smoothing moves vertices and can invalidate the gradient normals you cached. VTK explicitly flags normals/gradients as expensive and something you may want off if you’ll post-process geometry. [vtk](https://vtk.org/doc/nightly/html/classvtkContourFilter.html)
- Fix the anisotropic distance/cutoff units issue mentioned above, otherwise you can get inconsistent kernel support and odd “hard” artifacts depending on deformation magnitude. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/86997361/41a4f3b5-bf25-4b1f-8e79-f68f98290601/reconstruction_research_final_v2.md)

If you tell me whether the remaining “not solid” perception is (1) silhouette banding, (2) specular highlight waviness, or (3) tiny dents/bumps, I can point to the single next change with the highest ROI.

You’re already doing most of the “big wins” (GPU splat → temporal blend → edge-aware blur → Flying Edges → temporal vertex locking → Taubin), so the remaining path to a more *solid* look is mostly about (a) making the **anisotropy numerically stable and volume-consistent**, (b) fixing a couple of **edge-aware blur / thresholding gotchas** that can reintroduce shimmer, and (c) adding one or two **shape priors** (morphology / multi-scale) that suppress particle-scale texture without washing out real dents. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/86997361/b344ab5c-65d8-4e9f-b05d-2d395fbb1af1/reconstruction_research_final_v2.md)

## 1) Stabilize anisotropic splatting
Your Phase 5 move—anisotropic kernels via \(B = F F^T\) and \(r^2=(x-x_p)^T B^{-1}(x-x_p)\)—is the right class of solution for eliminating “grapes” under compression, and it’s closely related in spirit to classic anisotropic-kernel surface reconstruction (Yu & Turk). [dl.acm](https://dl.acm.org/doi/10.1145/2421636.2421641)

Two important upgrades usually make it look **more rigid** and less “alive”:
- **Clamp anisotropy**: do an SVD/polar step (even approximate) and clamp principal stretches (or clamp eigenvalues of \(B\)) so one particle can’t become an infinitely thin “pancake” and create streaky density ridges. [alexey.stomakhin](https://alexey.stomakhin.com/research/siggraph2016_mpm.pdf)
- **Volume-normalize the metric**: normalize \(B\) by \(\det(B)^{1/3}\) (or normalize \(F\) by \(\det(F)^{1/3}\) before building \(B\)) so anisotropy changes *shape* but not “effective kernel volume,” which reduces frame-to-frame density/threshold drift that reads as softness. [courses.washington](https://courses.washington.edu/mengr503/Chapter_3.pdf)

## 2) Fix bilateral blur + make it scale-aware
Your report/code shows a separable bilateral blur with a range term \(\exp(-0.5 * \text{diff} / \sigma_r^2)\); for a true bilateral you almost always want \(\exp(-0.5 * \text{diff}^2 / \sigma_r^2)\) (square the density difference), otherwise the filter becomes overly permissive and can smear or “pulse” across edges in a non-physical way. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/86997361/b344ab5c-65d8-4e9f-b05d-2d395fbb1af1/reconstruction_research_final_v2.md)

Also, because your density scale changes with particle count, kernel constants, and anisotropy, \(\sigma_r\) should usually be **relative to the current density scale** (e.g., \(\sigma_r = k \cdot \max(D)\) or \(k\cdot\) percentile), not a fixed 0.5, so the “edge stopping” strength stays consistent across strikes vs rest. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/86997361/b344ab5c-65d8-4e9f-b05d-2d395fbb1af1/reconstruction_research_final_v2.md)

## 3) Replace “0.4 * max” with a more stable isovalue rule
Dynamic thresholding like `thresh = 0.4 * max_density` can reduce banding, but it can also cause subtle volume breathing if max density spikes (often happens with anisotropy + blur + compression). [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/86997361/b344ab5c-65d8-4e9f-b05d-2d395fbb1af1/reconstruction_research_final_v2.md)

Two alternatives that often look more like rigid metal:
- **Percentile-based isovalue**: set `thresh = percentile(D, p)` (e.g., 97–99th of nonzero voxels). This tracks overall density distribution rather than a single extreme voxel.  
- **Mass/volume-calibrated field**: normalize your splat so “inside” density is roughly invariant (e.g., divide by expected kernel-sum under rest packing), then use a **fixed** isovalue. This reduces temporal “squishiness” without needing heavier temporal smoothing. [cg.informatik.uni-freiburg](https://cg.informatik.uni-freiburg.de/publications/2012_VRIPHYS_surfacePipeline.pdf)

## 4) Add a shape prior (cheap, very effective)
Once you’re extracting a binary-ish solid from a scalar grid, small pits/bumps are often better addressed with **morphology** than with more blur:
- **Closing in grid space** (dilate then erode) fills tiny surface dimples and makes the object read as machined/solid; you can approximate this with a couple of 3D max/min filter passes on the density or on a thresholded occupancy grid. [cg.informatik.uni-freiburg](https://cg.informatik.uni-freiburg.de/publications/2012_VRIPHYS_surfacePipeline.pdf)
- **Multi-scale blending**: mesh a coarse field (very smooth) and a fine field (detail), then blend them in SDF space (or blend densities before contouring) to keep real dents while suppressing particle-scale texture. [cg.informatik.uni-freiburg](https://cg.informatik.uni-freiburg.de/publications/2012_VRIPHYS_surfacePipeline.pdf)

## 5) If you want “ultimate” solidity: learned SDF denoiser
If you can afford an offline training step, the most powerful 2024–2026 direction is a **CNN that maps particle-to-grid density → clean SDF**, explicitly learning away particle noise while preserving sharp features and temporal coherence. [onlinelibrary.wiley](https://onlinelibrary.wiley.com/doi/10.1111/cgf.15181)

Recent work also focuses on making this fast (for big particle counts) by designing the network and pipeline for efficient inference, which is the key lever if you want to keep your current architecture but get a big perceptual jump in rigidity. [cs.ucr](https://www.cs.ucr.edu/~craigs/papers/2025-surface-reconstruction/paper.pdf)

One clarifying question that will change the best recommendation: do you need the reconstructed mesh for **physics/collisions**, or is it purely for **rendering/visual realism**?