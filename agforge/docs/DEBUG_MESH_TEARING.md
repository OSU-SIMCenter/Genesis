# Research Report: Hybrid Reconstruction Failure Analysis

## Executive Summary
We attempted to implement **Dynamic Edge Splitting (CPU)** to repair mesh stretching during deformation. While we successfully implemented the geometric splitting and a robust weight transfer system (Barycentric interpolation + Nearest Neighbor fallback), the user reports persistent **mesh tearing and holes** during simulation.

This report analyzes the failure, the current approach, and proposes alternative strategies for a fresh start.

## Current Approach: "Split & Transfer"

### The Algorithm
1.  **Deformation:** Mesh moves via Linear Blend Skinning (LBS).
2.  **Detection:** Every 5 frames, we identify edges > 1.5x median length.
3.  **Splitting:** We use `trimesh.remesh.subdivide_to_size` to introduce new vertices.
4.  **Weight Transfer:**
    *   New vertices need skinning weights to move with particles.
    *   We map new vertices to the *old* mesh surface using `trimesh.proximity`.
    *   **Primary:** Barycentric interpolation from the parent face's 3 corners.
    *   **Fallback:** Nearest Neighbor (KDTree) if geometry is degenerate or weights map to NaNs.

### The Failure Mode
The "Holes" appear as tears in the mesh.
*   **Hypothesis 1 (Topological):** The `trimesh` subdivision might be creating non-manifold geometry or disconnected components that our "welding" (`process=True`) fails to fix perfectly in real-time.
*   **Hypothesis 2 (Skinning Discontinuity):** Even with interpolation, if a new vertex is created on a face that sits between two divergent particle clusters, its weight might not perfectly match the blend needed to bridge the gap. In the next frame, it gets pulled in a direction that rips it from its neighbors.
*   **Hypothesis 3 (Lag):** We split *after* deformation. If the mesh is already stretched, the new vertices are spawned in a stretched state. Computing weights on a stretched mesh is numerically unstable compared to the rest pose.

## "Weird Errors"
User reported "weird errors" and regressions.
*   **Likely Cause:** `trimesh` operations (especially `fix_normals` or `subdivide`) can be fragile on non-watertight meshes.
*   **Regression:** Our "Safe Abort" prevented crashes, but it also meant that when the mesh *needed* splitting the most (highly stretched), we likely aborted due to tolerance checks, leaving the mesh stretched and effectively invisible (degenerate triangles).

## Proposed Alternatives

### Option A: GPU-Based Remeshing (Taichi)
Instead of trying to patch the mesh on CPU (slow, buggy data transfer), we move the topology logic to Taichi.
*   **Pros:** Access to all particles instantly. Fast.
*   **Cons:** Extremely complex to implement dynamic topology in Taichi data structures.

### Option B: Full Frame-by-Frame Reconstruction (Marching Cubes)
Abandon "repairing" the mesh. Just rebuild it from scratch every frame.
*   **Old Problem:** Flicker/Popping because the topology changes every frame.
*   **Solution:** Use **Dual Contouring** or a temporal consistency filter?
*   **Performance:** Can we run Marching Cubes at 60Hz? (Likely yes at low res, maybe not at high res).

### Option C: Constraints-Based Cloth Simulation
Treat the surface not just as a visual mesh, but as a physical cloth coupled to the particles.
*   **Pros:** Guaranteed continuity (cloth doesn't tear).
*   **Cons:** Expensive. Hard to couple with MPM.

## Recommendation for Next Steps
1.  **Stop patching `reconstruction.py`.** The current path of "Split -> Transfer -> Fix -> Abort" has reached a dead end of complexity vs. reliability.
2.  **Analyze `trimesh` logs:** If we continue, we need to know exactly *why* the transfer fails.
3.  **Pilot Option B (Fast Rebuild):** Re-enable full reconstruction but optimize it to be fast. If we can make it fast enough, maybe the "popping" is acceptable compared to "tearing".

## Files for Review
*   `agforge/reconstruction.py`: The implementation of `subdivide_long_edges` and `_transfer_skinning_data`.
*   `agforge/teleop_socket.py`: The main loop calling the reconstruction.
