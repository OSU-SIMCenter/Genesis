"""Isosurface extraction from Quadrants density fields (GPU Warp or CPU PyVista)."""

from __future__ import annotations

import contextlib
from typing import Literal, Optional

import numpy as np
import pyvista as pv
import trimesh

import genesis as gs

McBackend = Literal["auto", "warp", "pyvista"]

_warp_initialized = False
_logged_backend: McBackend | None = None
_mc_contexts: dict[tuple[int, int, int, str], object] = {}


def _profile(profiler, name: str):
    if profiler is not None:
        return profiler.time(name)
    return contextlib.nullcontext()


def _init_warp() -> None:
    global _warp_initialized
    if _warp_initialized:
        return
    import warp as wp

    wp.init()
    _warp_initialized = True


def _density_to_torch(density_field):
    from genesis.utils.misc import qd_to_torch

    tensor = qd_to_torch(density_field, transpose=False, copy=False)
    return tensor.float().contiguous()


def resolve_mc_backend(backend: McBackend) -> McBackend:
    if backend != "auto":
        return backend
    try:
        import torch

        if torch.cuda.is_available():
            return "warp"
    except Exception:
        pass
    return "pyvista"


def compute_isovalue_threshold(
    density_field,
    *,
    density_cpu: np.ndarray | None = None,
    percentile: float = 95.0,
    scale: float = 0.3,
    min_val: float = 1e-4,
) -> float | None:
    """Match legacy PyVista threshold: percentile(valid, 95) * 0.3."""
    if density_cpu is not None:
        valid = density_cpu[density_cpu > min_val]
        if valid.size == 0:
            return None
        return float(np.percentile(valid, percentile)) * scale

    try:
        import torch
        import torch.nn.functional as F

        tensor = _density_to_torch(density_field)
        pooled = tensor.unsqueeze(0).unsqueeze(0)
        while pooled.shape[-1] > 32:
            kernel = min(3, pooled.shape[-1])
            pooled = F.max_pool3d(pooled, kernel_size=kernel, stride=kernel)
        small = pooled.squeeze().detach().cpu().numpy()
        valid = small[small > min_val]
        if valid.size == 0:
            return None
        return float(np.percentile(valid, percentile)) * scale
    except Exception as exc:
        gs.logger.warning(f"GPU isovalue estimate failed ({exc}); falling back to full-grid CPU threshold.")
        density_cpu = density_field.to_numpy()
        return compute_isovalue_threshold(
            density_field,
            density_cpu=density_cpu,
            percentile=percentile,
            scale=scale,
            min_val=min_val,
        )


def _extract_pyvista(
    density_cpu: np.ndarray,
    min_bound: np.ndarray,
    dx: float,
    threshold: float,
) -> trimesh.Trimesh | None:
    grid = pv.ImageData()
    grid.dimensions = np.array(density_cpu.shape)
    grid.spacing = (dx, dx, dx)
    grid.origin = min_bound
    grid.point_data["density"] = density_cpu.flatten(order="F")

    contour = grid.contour(
        isosurfaces=[threshold],
        scalars="density",
        method="flying_edges",
        compute_normals=False,
    )
    if contour.n_points == 0:
        return None

    verts = np.array(contour.points)
    faces = np.array(contour.faces).reshape(-1, 4)[:, 1:4]
    faces = faces[:, ::-1]
    return trimesh.Trimesh(vertices=verts, faces=faces, process=False)


def _extract_warp(
    density_field,
    min_bound: np.ndarray,
    dx: float,
    threshold: float,
    grid_res: int,
) -> trimesh.Trimesh | None:
    import torch
    import warp as wp
    from warp import MarchingCubes

    _init_warp()

    tensor = _density_to_torch(density_field)
    if tensor.device.type != "cuda":
        raise RuntimeError(f"Warp marching cubes requires a CUDA density field (got {tensor.device}).")

    if tensor.shape != (grid_res, grid_res, grid_res):
        raise ValueError(f"Density shape {tuple(tensor.shape)} != ({grid_res}, {grid_res}, {grid_res})")

    density_wp = wp.from_torch(tensor, dtype=wp.float32)

    lower = wp.vec3(float(min_bound[0]), float(min_bound[1]), float(min_bound[2]))
    upper = wp.vec3(
        float(min_bound[0] + (grid_res - 1) * dx),
        float(min_bound[1] + (grid_res - 1) * dx),
        float(min_bound[2] + (grid_res - 1) * dx),
    )

    cache_key = (grid_res, grid_res, grid_res, str(tensor.device))
    mc = _mc_contexts.get(cache_key)
    if mc is None:
        mc = MarchingCubes(grid_res, grid_res, grid_res)
        _mc_contexts[cache_key] = mc

    verts_wp, indices_wp = MarchingCubes.extract_surface_marching_cubes(
        density_wp,
        threshold=float(threshold),
        domain_bounds_lower_corner=lower,
        domain_bounds_upper_corner=upper,
    )

    if verts_wp is None or indices_wp is None or len(verts_wp) == 0:
        return None

    verts = verts_wp.numpy()
    # Warp outputs the same winding as PyVista after the VTK→Unity flip in _extract_pyvista.
    # Do not reverse again or normals invert in Unity / Genesis viewer.
    faces = indices_wp.numpy().reshape(-1, 3)
    return trimesh.Trimesh(vertices=verts, faces=faces, process=False)


def extract_isosurface_mesh(
    density_field,
    *,
    min_bound: np.ndarray,
    dx: float,
    grid_res: int,
    mc_backend: McBackend = "auto",
    profiler=None,
    profile_prefix: str = "hybrid",
) -> trimesh.Trimesh | None:
    """Extract a triangle mesh from a Quadrants density field."""
    backend = resolve_mc_backend(mc_backend)

    global _logged_backend
    if _logged_backend != backend:
        gs.logger.info(f"Marching cubes backend: {backend}")
        _logged_backend = backend

    with _profile(profiler, f"{profile_prefix}_threshold"):
        if backend == "pyvista":
            with _profile(profiler, f"{profile_prefix}_density_transfer"):
                density_cpu = density_field.to_numpy()
            threshold = compute_isovalue_threshold(density_field, density_cpu=density_cpu)
        else:
            density_cpu = None
            threshold = compute_isovalue_threshold(density_field)

        if threshold is None:
            return None

    with _profile(profiler, f"{profile_prefix}_marching_cubes"):
        if backend == "pyvista":
            assert density_cpu is not None
            mesh = _extract_pyvista(density_cpu, min_bound, dx, threshold)
        else:
            try:
                with _profile(profiler, f"{profile_prefix}_mesh_download"):
                    mesh = _extract_warp(density_field, min_bound, dx, threshold, grid_res)
            except Exception as exc:
                gs.logger.warning(f"Warp marching cubes failed ({exc}); falling back to PyVista.")
                with _profile(profiler, f"{profile_prefix}_density_transfer"):
                    density_cpu = density_field.to_numpy()
                with _profile(profiler, f"{profile_prefix}_marching_cubes_pyvista_fallback"):
                    mesh = _extract_pyvista(density_cpu, min_bound, dx, threshold)

    return mesh
