#!/usr/bin/env python3
"""Profile g0_grid_alone 17-hit WITH Genesis viewer + MPM grid nodes visible.

Same timing/RAM/GPU instrumentation as profile_g0_17.py, but:
  - show_viewer=True (Genesis rendering)
  - visualize_mpm_grid=True
  - visualize_mpm_boundary=True
  - performance_mode=False so vis defaults keep grid overlays on

Requires WSLg DISPLAY. One GPU job at a time.
"""
from __future__ import annotations

import json
import os
import resource
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone

# WSLg GL defaults BEFORE any OpenGL/genesis import
sys.path.insert(0, os.path.expanduser("~/GitHub/Genesis/aims-genesis/nsf-demo"))
from agforge.wsl_graphics import apply_early_wsl_graphics_defaults  # noqa: E402

apply_early_wsl_graphics_defaults()

import numpy as np

T0 = time.perf_counter()
STAMPS: list[tuple[str, float]] = [("process_start", 0.0)]
MEM_SAMPLES: list[dict] = []
PEAK = {
    "proc_rss_mib": 0.0,
    "proc_vms_mib": 0.0,
    "host_used_mib": 0.0,
    "gpu_mem_used_mib": 0.0,
}


def mark(label: str) -> None:
    STAMPS.append((label, time.perf_counter() - T0))
    print(f"[mark +{STAMPS[-1][1]:7.2f}s] {label}", flush=True)


def _read_proc_status() -> dict:
    out = {}
    try:
        with open("/proc/self/status", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith(("VmRSS:", "VmHWM:", "VmSize:", "VmData:", "VmPeak:")):
                    key, val = line.split(":", 1)
                    out[key] = float(val.strip().split()[0]) / 1024.0
    except Exception as exc:
        out["error"] = repr(exc)
    return out


def _read_meminfo() -> dict:
    keys = ("MemTotal", "MemFree", "MemAvailable", "Buffers", "Cached", "SwapTotal", "SwapFree")
    raw = {}
    try:
        with open("/proc/meminfo", encoding="utf-8") as fh:
            for line in fh:
                for k in keys:
                    if line.startswith(k + ":"):
                        raw[k] = float(line.split()[1]) / 1024.0
    except Exception as exc:
        return {"error": repr(exc)}
    total = raw.get("MemTotal", 0.0)
    avail = raw.get("MemAvailable", raw.get("MemFree", 0.0))
    used = total - avail
    swap_total = raw.get("SwapTotal", 0.0)
    swap_free = raw.get("SwapFree", 0.0)
    return {
        "host_total_mib": round(total, 1),
        "host_available_mib": round(avail, 1),
        "host_used_mib": round(used, 1),
        "host_used_pct": round(100.0 * used / total, 2) if total else None,
        "buffers_mib": round(raw.get("Buffers", 0.0), 1),
        "cached_mib": round(raw.get("Cached", 0.0), 1),
        "swap_total_mib": round(swap_total, 1),
        "swap_used_mib": round(swap_total - swap_free, 1),
    }


def gpu_snapshot() -> dict:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=10,
        ).strip()
        parts = [p.strip() for p in out.split(",")]
        return {
            "name": parts[0],
            "util_gpu_pct": float(parts[1]),
            "util_mem_pct": float(parts[2]),
            "mem_used_mib": float(parts[3]),
            "mem_total_mib": float(parts[4]),
            "temp_c": float(parts[5]),
            "power_w": float(parts[6]) if parts[6] not in ("", "[N/A]") else None,
            "raw": out,
        }
    except Exception as exc:
        return {"error": repr(exc)}


def memory_snapshot(label: str, *, with_gpu: bool = True) -> dict:
    host = _read_meminfo()
    proc = _read_proc_status()
    try:
        ru_maxrss_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except Exception:
        ru_maxrss_mib = None
    snap = {
        "label": label,
        "t_s": round(time.perf_counter() - T0, 3),
        **host,
        "proc_rss_mib": round(proc.get("VmRSS", 0.0), 1),
        "proc_hwm_mib": round(proc.get("VmHWM", 0.0), 1),
        "proc_vms_mib": round(proc.get("VmSize", 0.0), 1),
        "proc_vmpeak_mib": round(proc.get("VmPeak", 0.0), 1),
        "proc_data_mib": round(proc.get("VmData", 0.0), 1),
        "ru_maxrss_mib": round(ru_maxrss_mib, 1) if ru_maxrss_mib is not None else None,
    }
    if with_gpu:
        g = gpu_snapshot()
        snap["gpu"] = g
        if isinstance(g, dict) and "mem_used_mib" in g:
            PEAK["gpu_mem_used_mib"] = max(PEAK["gpu_mem_used_mib"], float(g["mem_used_mib"]))
    PEAK["proc_rss_mib"] = max(PEAK["proc_rss_mib"], snap["proc_rss_mib"])
    PEAK["proc_vms_mib"] = max(PEAK["proc_vms_mib"], snap["proc_vms_mib"])
    if snap.get("host_used_mib") is not None:
        PEAK["host_used_mib"] = max(PEAK["host_used_mib"], float(snap["host_used_mib"]))
    MEM_SAMPLES.append(snap)
    print(
        f"[mem  +{snap['t_s']:7.2f}s] {label}: "
        f"RSS={snap['proc_rss_mib']:.0f} MiB (HWM={snap['proc_hwm_mib']:.0f})  "
        f"host_used={snap.get('host_used_mib')} / {snap.get('host_total_mib')} MiB "
        f"({snap.get('host_used_pct')}%)"
        + (
            f"  GPU={snap['gpu'].get('mem_used_mib')} / {snap['gpu'].get('mem_total_mib')} MiB"
            if with_gpu and isinstance(snap.get("gpu"), dict)
            else ""
        ),
        flush=True,
    )
    return snap


def deltas_from_stamps(stamps: list[tuple[str, float]]) -> list[dict]:
    rows = []
    prev = 0.0
    for label, t in stamps[1:]:
        rows.append({"phase": label, "delta_s": round(t - prev, 3), "cumul_s": round(t, 3)})
        prev = t
    return rows


def _build_viewer_adapter():
    """Genesis adapter with viewer + MPM grid nodes, without editing forge_common."""
    from forge_common.adapters.genesis_forge_adapter import GenesisForgeAdapter
    from forge_common.press_tool import die_contact_axial_width_mm
    from agforge import agforge_builder

    orig_build_env = agforge_builder.build_env

    def build_env_with_grid(cfg):
        cfg.general.show_viewer = True
        # performance_mode=True (default) leaves visualize_mpm_grid=False;
        # force the overlays on explicitly.
        cfg.performance_mode = False
        cfg.vis.visualize_mpm_grid = True
        cfg.vis.visualize_mpm_boundary = True
        cfg.vis.show_world_frame = True
        cfg.vis.particle_render_fraction = 1.0
        print(
            "[viewer] show_viewer=%s visualize_mpm_grid=%s visualize_mpm_boundary=%s "
            "performance_mode=%s DISPLAY=%r"
            % (
                cfg.general.show_viewer,
                cfg.vis.visualize_mpm_grid,
                cfg.vis.visualize_mpm_boundary,
                cfg.performance_mode,
                os.environ.get("DISPLAY"),
            ),
            flush=True,
        )
        return orig_build_env(cfg)

    agforge_builder.build_env = build_env_with_grid
    return GenesisForgeAdapter(
        press_width_mm=die_contact_axial_width_mm(),
        show_viewer=True,
        surface_mesh=False,
    )


def main() -> int:
    if os.environ.get("AGF_CONTACT_RUNTIME_SWITCH") != "1":
        print("ERROR: set AGF_CONTACT_RUNTIME_SWITCH=1", flush=True)
        return 2
    if not os.environ.get("DISPLAY"):
        print(
            "WARNING: DISPLAY is unset — Genesis viewer may fail under WSLg. "
            "Expected something like :0",
            flush=True,
        )

    n_hits = int(os.environ.get("AGF_PROFILE_N_HITS", "17"))
    out_name = os.environ.get("AGF_PROFILE_OUT", "profile_g0_17_render")
    out_dir = os.path.expanduser(f"~/GitHub/Genesis/forge_common/main/outputs/{out_name}")
    os.makedirs(out_dir, exist_ok=True)

    report: dict = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "arm": "g0_grid_alone",
        "mode": "genesis_viewer_mpm_grid",
        "n_hits_requested": n_hits,
        "out_dir": out_dir,
        "display": os.environ.get("DISPLAY"),
        "env": {
            k: os.environ.get(k)
            for k in [
                "AGF_MPM_X_PAD_LOWER",
                "AGF_CONTACT_RUNTIME_SWITCH",
                "AGF_DIAG_PENETRATION",
                "AGF_CELLS_PER_DIAMETER",
                "AGF_CFL_SAFETY",
                "AGF_ENABLE_CPIC",
                "AGF_APPROACH_CFL_RATIO",
                "DISPLAY",
                "GALLIUM_DRIVER",
                "MESA_D3D12_DEFAULT_ADAPTER_NAME",
                "LD_LIBRARY_PATH",
            ]
        },
    }
    report["mem_before"] = memory_snapshot("process_start")

    sys.path.insert(0, os.path.expanduser("~/GitHub/Genesis/forge_common/main"))

    from agforge.analysis.batch_arms import ARMS, configure  # noqa: E402

    arm = next(a for a in ARMS if a["tag"] == "g0_grid_alone")

    from forge_common.real_data import load_real_hits_for_sim  # noqa: E402
    from forge_common.real_scale import (  # noqa: E402
        REAL_STOCK_LENGTH_MM,
        REAL_STOCK_RADIUS_MM,
    )

    mark("imports_forge_common")
    memory_snapshot("after_forge_common_imports")

    import torch  # noqa: E402

    report["torch"] = {
        "version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    mark("import_torch")
    memory_snapshot("after_torch_import")

    import genesis  # noqa: E402,F401

    mark("import_genesis")
    memory_snapshot("after_genesis_import")

    if not report["torch"]["cuda_available"]:
        print("ERROR: CUDA not available — refusing CPU fallback", flush=True)
        report["status"] = "aborted_no_cuda"
        report["memory_samples"] = MEM_SAMPLES
        report["memory_peak"] = PEAK
        _write(out_dir, report)
        return 3

    hits = load_real_hits_for_sim("genesis", n_hits)
    mark("load_real_hits")
    report["n_hits_loaded"] = len(hits)

    adapter = _build_viewer_adapter()
    mark("build_adapter_viewer")
    memory_snapshot("after_build_adapter")

    state = adapter.init_stock(
        radius_mm=REAL_STOCK_RADIUS_MM, length_mm=REAL_STOCK_LENGTH_MM
    )
    mark("init_stock")
    memory_snapshot("after_init_stock")

    # Confirm vis flags actually stuck on the live scene config
    try:
        vis = state.env.cfg.vis
        report["vis_live"] = {
            "visualize_mpm_grid": bool(getattr(vis, "visualize_mpm_grid", None)),
            "visualize_mpm_boundary": bool(getattr(vis, "visualize_mpm_boundary", None)),
            "show_viewer": bool(getattr(state.env.cfg.general, "show_viewer", None)),
            "performance_mode": bool(getattr(state.env.cfg, "performance_mode", None)),
        }
        print("[viewer] live cfg:", report["vis_live"], flush=True)
    except Exception as exc:
        report["vis_live"] = {"error": repr(exc)}

    P0 = adapter.to_mesh(state).vertices
    report["fresh_bar"] = {
        "n_verts": int(len(P0)),
        "span_mm": [round(float(np.ptp(P0[:, k])), 3) for k in range(P0.shape[1])],
    }
    np.savez_compressed(
        os.path.join(out_dir, "_ref_fresh_verts.npz"), verts=P0.astype(np.float32)
    )
    mark("save_fresh_bar")

    coupler = state.env.scene.sim.coupler
    cfg = configure(coupler, arm)
    report["config"] = cfg
    mark("configure_g0_grid_alone")
    report["mem_after_setup"] = memory_snapshot("after_configure")

    ctrl = state.controller
    ctrl._diag_out = os.path.join(out_dir, "g0_grid_alone.diag.jsonl")
    ctrl._diag_strike_idx = 0
    ctrl._diag_acc = None
    if os.path.exists(ctrl._diag_out):
        os.remove(ctrl._diag_out)

    per_hit_times: list[dict] = []
    per_hit_clouds = {}
    status, n_done, err = "completed", 0, None
    t_hits_start = time.perf_counter()

    for i, h in enumerate(hits, 1):
        t_hit = time.perf_counter()
        try:
            state = adapter.apply_hit(state, h)
            n_done += 1
            dt = time.perf_counter() - t_hit
            Ph = adapter.to_mesh(state).vertices
            mem = memory_snapshot(f"after_hit_{i:02d}")
            row = {
                "hit": i,
                "wall_s": round(dt, 3),
                "cumul_from_hits_start_s": round(time.perf_counter() - t_hits_start, 3),
                "cumul_from_process_start_s": round(time.perf_counter() - T0, 3),
                "proc_rss_mib": mem["proc_rss_mib"],
                "proc_hwm_mib": mem["proc_hwm_mib"],
                "host_used_mib": mem.get("host_used_mib"),
                "host_available_mib": mem.get("host_available_mib"),
                "gpu_mem_used_mib": (
                    mem["gpu"].get("mem_used_mib")
                    if isinstance(mem.get("gpu"), dict)
                    else None
                ),
            }
            if Ph is not None and len(Ph):
                per_hit_clouds[f"hit_{i:02d}"] = np.asarray(Ph, dtype=np.float32)
                row["n_verts"] = int(len(Ph))
                row["span_mm"] = [
                    round(float(np.ptp(Ph[:, k])), 3) for k in range(Ph.shape[1])
                ]
                row["verts_finite"] = bool(np.all(np.isfinite(Ph)))
            per_hit_times.append(row)
            mark(f"hit_{i:02d}")
            print(
                f"  hit {i:02d}/{n_hits}: {dt:.3f}s  "
                f"RSS={row['proc_rss_mib']:.0f} MiB  "
                f"GPU={row.get('gpu_mem_used_mib')} MiB  "
                f"span={row.get('span_mm')}  verts={row.get('n_verts')}",
                flush=True,
            )
        except Exception as exc:
            dt = time.perf_counter() - t_hit
            status = f"failed_at_hit_{n_done + 1}"
            err = repr(exc)[:500]
            mem = memory_snapshot(f"after_hit_{i:02d}_FAILED")
            per_hit_times.append(
                {
                    "hit": i,
                    "wall_s": round(dt, 3),
                    "failed": True,
                    "error": err,
                    "proc_rss_mib": mem["proc_rss_mib"],
                    "host_used_mib": mem.get("host_used_mib"),
                    "gpu_mem_used_mib": (
                        mem["gpu"].get("mem_used_mib")
                        if isinstance(mem.get("gpu"), dict)
                        else None
                    ),
                }
            )
            mark(f"hit_{i:02d}_FAILED")
            print(f"  !! {status}: {err}", flush=True)
            traceback.print_exc()
            break

    t_hits_end = time.perf_counter()
    mark("hits_done")

    if per_hit_clouds:
        np.savez_compressed(os.path.join(out_dir, "g0_grid_alone_hits.npz"), **per_hit_clouds)

    P = adapter.to_mesh(state).vertices
    finite = bool(P is not None and len(P) and np.all(np.isfinite(P)))
    if P is not None and len(P) and finite:
        np.savez_compressed(
            os.path.join(out_dir, "g0_grid_alone_verts.npz"), verts=P.astype(np.float32)
        )

    report["mem_after"] = memory_snapshot("finished")
    report["status"] = status
    report["hits_done"] = n_done
    report["error"] = err
    report["final_bar"] = {
        "n_verts": int(len(P)) if P is not None else 0,
        "verts_finite": finite,
        "span_mm": (
            [round(float(np.ptp(P[:, k])), 3) for k in range(P.shape[1])]
            if P is not None and len(P) and finite
            else None
        ),
    }

    phases = deltas_from_stamps(STAMPS)
    report["phases"] = phases

    hit_walls = [r["wall_s"] for r in per_hit_times if not r.get("failed")]
    steady = hit_walls[1:] if len(hit_walls) > 1 else []
    setup_through = next(
        (p["cumul_s"] for p in phases if p["phase"] == "configure_g0_grid_alone"), 0.0
    )
    hits_wall = t_hits_end - t_hits_start
    total_wall = time.perf_counter() - T0
    rss_series = [r["proc_rss_mib"] for r in per_hit_times if "proc_rss_mib" in r]
    gpu_series = [
        r["gpu_mem_used_mib"]
        for r in per_hit_times
        if r.get("gpu_mem_used_mib") is not None
    ]

    summary = {
        "total_wall_s": round(total_wall, 3),
        "setup_through_configure_s": round(setup_through, 3),
        "hits_wall_s": round(hits_wall, 3),
        "hits_done": n_done,
        "per_hit_mean_s": round(float(np.mean(hit_walls)), 3) if hit_walls else None,
        "per_hit_std_s": round(float(np.std(hit_walls)), 3) if hit_walls else None,
        "per_hit_min_s": round(float(np.min(hit_walls)), 3) if hit_walls else None,
        "per_hit_max_s": round(float(np.max(hit_walls)), 3) if hit_walls else None,
        "hit1_s": hit_walls[0] if hit_walls else None,
        "steady_mean_s": round(float(np.mean(steady)), 3) if steady else None,
        "steady_std_s": round(float(np.std(steady)), 3) if steady else None,
        "jit_approx_s": (
            round(hit_walls[0] - float(np.mean(steady)), 3)
            if hit_walls and steady
            else None
        ),
        "setup_fraction": round(setup_through / total_wall, 3) if total_wall else None,
        "sim_fraction": round(hits_wall / total_wall, 3) if total_wall else None,
        "proc_rss_peak_mib": round(PEAK["proc_rss_mib"], 1),
        "proc_vms_peak_mib": round(PEAK["proc_vms_mib"], 1),
        "proc_rss_after_setup_mib": report["mem_after_setup"]["proc_rss_mib"],
        "proc_rss_final_mib": report["mem_after"]["proc_rss_mib"],
        "proc_rss_during_hits_mean_mib": (
            round(float(np.mean(rss_series)), 1) if rss_series else None
        ),
        "proc_rss_during_hits_max_mib": (
            round(float(np.max(rss_series)), 1) if rss_series else None
        ),
        "host_used_peak_mib": round(PEAK["host_used_mib"], 1),
        "host_total_mib": report["mem_after"].get("host_total_mib"),
        "gpu_mem_peak_mib": round(PEAK["gpu_mem_used_mib"], 1),
        "gpu_mem_during_hits_max_mib": (
            round(float(np.max(gpu_series)), 1) if gpu_series else None
        ),
        "gpu_mem_final_mib": (
            report["mem_after"]["gpu"].get("mem_used_mib")
            if isinstance(report["mem_after"].get("gpu"), dict)
            else None
        ),
    }
    report["summary"] = summary
    report["per_hit"] = per_hit_times
    report["memory_samples"] = MEM_SAMPLES
    report["memory_peak"] = {k: round(v, 1) for k, v in PEAK.items()}

    _write(out_dir, report)
    _print_tables(report)
    return 0 if status == "completed" else 1


def _write(out_dir: str, report: dict) -> None:
    path = os.path.join(out_dir, "profile.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nWrote {path}", flush=True)


def _print_tables(report: dict) -> None:
    print("\n" + "=" * 78)
    print("PHASE BREAKDOWN (WITH GENESIS RENDER + MPM GRID)")
    print("=" * 78)
    print(f"{'phase':42} {'delta_s':>10} {'cumul_s':>10}")
    print("-" * 78)
    for p in report["phases"]:
        print(f"{p['phase']:42} {p['delta_s']:10.3f} {p['cumul_s']:10.3f}")

    print("\n" + "=" * 78)
    print("PER-HIT TIMES + MEMORY")
    print("=" * 78)
    print(
        f"{'hit':>4} {'wall_s':>8} {'RSS_MiB':>8} {'HWM':>8} {'GPU_MiB':>8} "
        f"{'host_used':>10} {'n_verts':>8}"
    )
    print("-" * 78)
    for r in report["per_hit"]:
        if r.get("failed"):
            print(
                f"{r['hit']:4d} {r['wall_s']:8.3f} FAILED "
                f"RSS={r.get('proc_rss_mib')}  {r.get('error','')[:50]}"
            )
        else:
            print(
                f"{r['hit']:4d} {r['wall_s']:8.3f} {r.get('proc_rss_mib',0):8.0f} "
                f"{r.get('proc_hwm_mib',0):8.0f} {r.get('gpu_mem_used_mib') or 0:8.0f} "
                f"{r.get('host_used_mib') or 0:10.0f} {r.get('n_verts','-'):>8}"
            )

    s = report["summary"]
    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    for k, v in s.items():
        print(f"  {k:36} {v}")
    print(f"  {'status':36} {report['status']}")
    print(f"  {'vis_live':36} {report.get('vis_live')}")
    print(f"  {'torch.cuda':36} {report['torch']}")
    print(f"  {'memory_peak':36} {report.get('memory_peak')}")


if __name__ == "__main__":
    sys.exit(main())
