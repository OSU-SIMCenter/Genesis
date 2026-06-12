"""Micro-benchmark: SciPy cKDTree vs brute-force GPU cdist for vertex-temp k-NN.

Compares the two backends that were tried in teleop IO optimization:
  - scipy: cKDTree build + numpy gather (current production path)
  - gpu_cdist: chunked torch.cdist + topk build + GPU gather (reverted experiment)

Run:
  pixi run python agforge/benchmarks/benchmark_vertex_temp_knn.py
  pixi run python agforge/benchmarks/benchmark_vertex_temp_knn.py --n-verts 12000 --n-parts 8458
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch
from scipy.spatial import cKDTree


def _sync(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def bench_scipy_build(verts: np.ndarray, parts: np.ndarray, k: int, iters: int) -> float:
    for _ in range(3):
        tree = cKDTree(parts, leafsize=32)
        dists, indices = tree.query(verts, k=k)
        _ = dists.sum()

    start = time.perf_counter()
    for _ in range(iters):
        tree = cKDTree(parts, leafsize=32)
        dists, indices = tree.query(verts, k=k)
        if k == 1:
            dists = dists[:, np.newaxis]
            indices = indices[:, np.newaxis]
        dists = np.maximum(dists, 1e-6)
        weights = 1.0 / dists
        weights_norm = weights / weights.sum(axis=1, keepdims=True)
        indices = np.asarray(indices, dtype=np.int32)
        weights_norm = np.asarray(weights_norm, dtype=np.float32)
        _ = indices.sum() + weights_norm.sum()
    return (time.perf_counter() - start) / iters


def bench_scipy_map(
    temps: np.ndarray,
    indices: np.ndarray,
    weights: np.ndarray,
    fade_alpha: np.ndarray | None,
    iters: int,
) -> float:
    for _ in range(3):
        t = temps
        if fade_alpha is not None:
            target_vis = np.minimum(t, 900.0)
            t = t * (1.0 - fade_alpha) + target_vis * fade_alpha
        _ = (t[indices] * weights).sum(axis=1).sum()

    start = time.perf_counter()
    for _ in range(iters):
        t = temps
        if fade_alpha is not None:
            target_vis = np.minimum(t, 900.0)
            t = t * (1.0 - fade_alpha) + target_vis * fade_alpha
        out = (t[indices] * weights).sum(axis=1)
        _ = out.sum()
    return (time.perf_counter() - start) / iters


def bench_gpu_cdist_build(
    verts: np.ndarray,
    parts: np.ndarray,
    k: int,
    device: str,
    chunk_size: int,
    iters: int,
) -> float:
    parts_t = torch.as_tensor(parts, device=device, dtype=torch.float32)
    verts_t = torch.as_tensor(verts, device=device, dtype=torch.float32)

    def build_once() -> tuple[torch.Tensor, torch.Tensor]:
        idx_chunks: list[torch.Tensor] = []
        weight_chunks: list[torch.Tensor] = []
        for start in range(0, verts_t.shape[0], chunk_size):
            dists = torch.cdist(verts_t[start : start + chunk_size], parts_t)
            nearest_dists, nearest_idx = torch.topk(dists, k, dim=1, largest=False)
            inv = 1.0 / nearest_dists.clamp_min(1e-6)
            idx_chunks.append(nearest_idx)
            weight_chunks.append(inv / inv.sum(dim=1, keepdim=True))
        return torch.cat(idx_chunks, dim=0), torch.cat(weight_chunks, dim=0)

    for _ in range(3):
        indices_t, weights_t = build_once()
        _sync(device)
        _ = indices_t.sum().item()

    start = time.perf_counter()
    for _ in range(iters):
        indices_t, weights_t = build_once()
        _sync(device)
        _ = indices_t.sum().item()
    _sync(device)
    return (time.perf_counter() - start) / iters


def bench_gpu_map(
    temps: np.ndarray,
    indices_t: torch.Tensor,
    weights_t: torch.Tensor,
    fade_alpha: np.ndarray | None,
    device: str,
    iters: int,
) -> float:
    temps_t = torch.as_tensor(temps, device=device, dtype=torch.float32)
    fade_t = None
    if fade_alpha is not None:
        fade_t = torch.as_tensor(fade_alpha, device=device, dtype=torch.float32)

    def map_once() -> torch.Tensor:
        t = temps_t
        if fade_t is not None:
            target_vis = torch.minimum(t, torch.tensor(900.0, device=device))
            t = t * (1.0 - fade_t) + target_vis * fade_t
        neighbor_temps = t[indices_t]
        return (neighbor_temps * weights_t).sum(dim=1)

    for _ in range(3):
        out = map_once()
        _sync(device)
        _ = out.sum().item()

    start = time.perf_counter()
    for _ in range(iters):
        out = map_once()
        _sync(device)
        _ = out.sum().item()
    _sync(device)
    return (time.perf_counter() - start) / iters


def run_case(n_verts: int, n_parts: int, k: int, device: str, build_iters: int, map_iters: int) -> None:
    rng = np.random.default_rng(42)
    verts = rng.standard_normal((n_verts, 3), dtype=np.float32)
    parts = rng.standard_normal((n_parts, 3), dtype=np.float32)
    temps = rng.uniform(300.0, 900.0, size=n_parts).astype(np.float32)
    fade_alpha = np.clip(parts[:, 0], 0.0, 1.0).astype(np.float32)

    print(f"\n=== n_verts={n_verts:,}  n_parts={n_parts:,}  k={k}  device={device} ===")

    scipy_build_ms = bench_scipy_build(verts, parts, k, build_iters) * 1000
    tree = cKDTree(parts, leafsize=32)
    dists, indices = tree.query(verts, k=k)
    if k == 1:
        dists = dists[:, np.newaxis]
        indices = indices[:, np.newaxis]
    dists = np.maximum(dists, 1e-6)
    weights = 1.0 / dists
    weights_norm = weights / weights.sum(axis=1, keepdims=True)
    indices_np = np.asarray(indices, dtype=np.int32)
    weights_np = np.asarray(weights_norm, dtype=np.float32)
    scipy_map_ms = bench_scipy_map(temps, indices_np, weights_np, fade_alpha, map_iters) * 1000

    gpu_build_ms = gpu_map_ms = float("nan")
    if device.startswith("cuda") and torch.cuda.is_available():
        gpu_build_ms = bench_gpu_cdist_build(verts, parts, k, device, chunk_size=512, iters=build_iters) * 1000
        indices_t, weights_t = None, None
        parts_t = torch.as_tensor(parts, device=device, dtype=torch.float32)
        verts_t = torch.as_tensor(verts, device=device, dtype=torch.float32)
        idx_chunks, weight_chunks = [], []
        for start in range(0, verts_t.shape[0], 512):
            dists_t = torch.cdist(verts_t[start : start + 512], parts_t)
            nearest_dists, nearest_idx = torch.topk(dists_t, k, dim=1, largest=False)
            inv = 1.0 / nearest_dists.clamp_min(1e-6)
            idx_chunks.append(nearest_idx)
            weight_chunks.append(inv / inv.sum(dim=1, keepdim=True))
        indices_t = torch.cat(idx_chunks, dim=0)
        weights_t = torch.cat(weight_chunks, dim=0)
        _sync(device)
        gpu_map_ms = bench_gpu_map(temps, indices_t, weights_t, fade_alpha, device, map_iters) * 1000

    rebuild_rate = 0.22  # ~360/1665 from recent teleop profile
    scipy_frame_ms = scipy_map_ms + scipy_build_ms * rebuild_rate
    gpu_frame_ms = gpu_map_ms + gpu_build_ms * rebuild_rate if not np.isnan(gpu_build_ms) else float("nan")

    print(f"  scipy build (cache miss):     {scipy_build_ms:8.2f} ms")
    print(f"  scipy map   (cache hit):      {scipy_map_ms:8.2f} ms")
    print(f"  scipy amortized @22% rebuild: {scipy_frame_ms:8.2f} ms/frame")
    if not np.isnan(gpu_build_ms):
        print(f"  gpu_cdist build (cache miss): {gpu_build_ms:8.2f} ms  ({gpu_build_ms/scipy_build_ms:.2f}x vs scipy)")
        print(f"  gpu_cdist map (cache hit):    {gpu_map_ms:8.2f} ms  ({gpu_map_ms/scipy_map_ms:.2f}x vs scipy)")
        print(f"  gpu_cdist amortized:          {gpu_frame_ms:8.2f} ms/frame  ({gpu_frame_ms/scipy_frame_ms:.2f}x vs scipy)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark vertex-temp k-NN backends")
    parser.add_argument("--n-verts", type=int, default=0, help="Single vertex count (0 = sweep)")
    parser.add_argument("--n-parts", type=int, default=8458, help="Particle count (teleop default ~8458)")
    parser.add_argument("--build-iters", type=int, default=5)
    parser.add_argument("--map-iters", type=int, default=200)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    k = min(3, args.n_parts)
    vert_counts = [args.n_verts] if args.n_verts > 0 else [5000, 12000, 25000, 40000]

    print("Vertex-temp k-NN backend comparison")
    print(f"CUDA available: {torch.cuda.is_available()}  device={args.device}")
    for n_verts in vert_counts:
        run_case(n_verts, args.n_parts, k, args.device, args.build_iters, args.map_iters)


if __name__ == "__main__":
    main()
