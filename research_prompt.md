# Research Prompt: Incremental Mesh Reconstruction for MPM

## Context
I am working on a real-time metal forging simulation using the **Genesis** physics engine (MPM backend). We need to visualize the metal surface as a smooth, continuous mesh that deforms plastically.

## Current Tech Stack
- **Language:** Python
- **Physics:** Genesis (MPM / Particle-based)
- **Geometry:** `trimesh`, `splashsurf` (Marching Cubes)
- **Math/Compute:** PyTorch (GPU accelerated), NumPy

## The Problem
We currently use a "Hybrid" approach:
1.  **Initial:** Generate base mesh via Marching Cubes (`splashsurf`).
2.  **Runtime:** Use Linear Blend Skinning (k-NN binding) to deform vertices based on particle motion.

**Issues:**
*   **Topological Degradation:** During extreme compression (forging), the fixed mesh topology stretches, creating long sliver triangles.
*   **Mesh Jitter:** High-frequency particle noise transfers to the surface ("ripples").
*   **Popping:** To fix the degradation, we currently force a **Full Marching Cubes Rebuild** every ~15 frames. This causes visible "snapping" or "popping" artifacts.

## Failed Experiments (What We Tried)
1.  **Plastic Skinning (Dynamic Offsets):** We tried to update vertex binding offsets continuously to let the mesh "flow".
    *   *Result:* **Ballooning.** The mesh learned its own smoothed volume and spiraled into infinite expansion.
2.  **Idle Smoothing:** We continued smoothing the mesh when physics was paused.
    *   *Result:* Performance waste and volume loss/drift over time.
3.  **Tighter Binding (`k=4`):**
    *   *Result:* **Shrink-Wrapping.** The mesh looked like "dried fruit," revealing every particle imperfection.

## The Goal
I am looking for **efficient, incremental mesh reconstruction/adaptation algorithms** that can be implemented in **Python**.

Specifically, I need methods that can:
1.  **Adapt Topology:** incremental remeshing, dynamic edge splitting/collapsing, or "surface tracking" methods that can handle large plastic deformation without needing a full rebuild from scratch every frame.
2.  **Smooth Noise:** Filter out high-frequency particle jitter while maintaining volume.
3.  **Performance:** Find the most optimal and performant solution possible (ideally targeting 60Hz for ~10k-50k particles).

## Files to Review
Please review the attached **`reconstruction.py`** (Our current implementation).
*   Look at `init_skinning` vs `update_skinning`.
*   Look at how we currently use `Taubin Smoothing`.

**Question for Researcher:**
What are the state-of-the-art techniques or specific algorithms (e.g., Flip-based surface tracking, localized remeshing, moving least squares, etc.) that fit these criteria? Please provide papers, algorithm names, or pseudo-code concepts compatible with our Python/PyTorch stack.
