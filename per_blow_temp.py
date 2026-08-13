#!/usr/bin/env python3
"""Per-blow billet temperature across the whole 06-15 T4 session.

Only blow #1 had ever been read (session 12b6fa7e, re-derived 2026-08-13). The
other 46 carry the signal that no single blow can: the bar cools *within* a
forging bout and is reheated *between* bouts, so the sequence is a sawtooth. That
is the only real thermal ground truth this project has, and a coupled model has
to reproduce its within-bout slope.

Reads blow timings from analyze_press_mcap.py's segmentation and samples the
thermal camera around each. Both come from the same file, so the two clocks are
the same clock -- no alignment assumption is needed.

⚠️ The bar is NOT in frame between bouts. Presence is decided by blob area, which
separates cleanly: ~200-600 px with no bar, ~13,000-16,500 px with one. Blows
whose frames show no bar are reported as absent rather than given a number.
"""
import sys
import numpy as np

sys.path.insert(0, "/home/timothy/GitHub/Genesis/aims-genesis/thermal-st-invariance")
from agforge.mcap_thermal import ForgeMcap

MCAP = ("/mnt/c/Users/banko/Documents/forge-data-stage/2026-06-15_T4_bulk/"
        "20260615_180456_T4_bulk.mcap")
BLOWS = "/home/timothy/GitHub/Genesis/forge_common/main/outputs/t4_press_blows.npz"
OUT = "/home/timothy/GitHub/Genesis/forge_common/main/outputs/t4_per_blow_temp.npz"

PRESENT_PX = 5000        # blob area above which a bar is genuinely in frame
                         # (with the p85 rule this now rejects only true absences)
PRE_S, POST_S = 3.0, 3.0  # window sampled either side of contact


def largest_blob_mask(mask):
    try:
        from scipy import ndimage
        lab, n = ndimage.label(mask)
        if n == 0:
            return np.zeros_like(mask)
        sizes = ndimage.sum(mask, lab, range(1, n + 1))
        return lab == (int(np.argmax(sizes)) + 1)
    except ImportError:
        from collections import deque
        h, w = mask.shape
        if not mask.any():
            return np.zeros_like(mask)
        seen = np.zeros_like(mask)
        best, best_n = None, 0
        for y0 in range(h):
            for x0 in range(w):
                if not mask[y0, x0] or seen[y0, x0]:
                    continue
                q = deque([(y0, x0)]); seen[y0, x0] = True; cells = []
                while q:
                    y, x = q.popleft(); cells.append((y, x))
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                            seen[ny, nx] = True; q.append((ny, nx))
                if len(cells) > best_n:
                    best_n, best = len(cells), cells
        out = np.zeros_like(mask)
        if best:
            ys, xs = zip(*best)
            out[list(ys), list(xs)] = True
        return out


def workpiece(frame):
    """(area_px, p50, p90, p99, max) in Celsius for the largest hot region.

    Threshold is p85 with NO fixed floor. An earlier version used
    max(700 C, p85); the 700 C term was carried over from blow #1, which sits at
    the 95th percentile of this session, and it rejected the bar on the coolest
    blows -- biasing the session median high by dropping exactly the low tail.
    p85 is calibrated to the workpiece's own frame fraction (~16,400 of 110,016
    px = ~15%) and follows the bar down as it cools.
    """
    c = frame.celsius
    thr = float(np.percentile(c, 85))
    blob = largest_blob_mask(c >= thr)
    n = int(blob.sum())
    if n == 0:
        return 0, np.nan, np.nan, np.nan, np.nan
    v = c[blob]
    return (n, float(np.percentile(v, 50)), float(np.percentile(v, 90)),
            float(np.percentile(v, 99)), float(v.max()))


def main():
    b = np.load(BLOWS)
    t0 = b["t0"]
    peak = b["peak"]
    pos_start = b["pos_start"]
    pos_min = b["pos_min"]
    print(f"blows segmented from the press channel: {len(t0)}")

    m = ForgeMcap(MCAP)
    rows = []

    hdr = (f"{'#':>3} {'t0_s':>8} {'gap_in':>7} {'gap_out':>8} {'kN':>7} "
           f"{'dt_prev':>8} {'px':>6} {'T_pre':>7} {'T_post':>7} {'dT':>7} {'state':>9}")
    print()
    print(hdr)
    print("-" * len(hdr))

    prev_t = None
    for i, t in enumerate(t0, 1):
        frames = list(m.thermal_frames(float(t) - PRE_S, float(t) + POST_S))
        pre = [f for f in frames if f.t_s <= t]
        post = [f for f in frames if f.t_s > t]
        a_pre = workpiece(pre[-1]) if pre else (0,) + (np.nan,) * 4
        a_post = workpiece(post[-1]) if post else (0,) + (np.nan,) * 4

        area = max(a_pre[0], a_post[0])
        present = area >= PRESENT_PX
        dt_prev = np.nan if prev_t is None else float(t) - prev_t
        dT = (a_post[1] - a_pre[1]) if (present and np.isfinite(a_pre[1])
                                        and np.isfinite(a_post[1])) else np.nan
        state = "OK" if present else "no bar"

        print(f"{i:3d} {float(t):8.1f} {pos_start[i-1]:7.2f} {pos_min[i-1]:8.2f} "
              f"{peak[i-1]:7.1f} "
              f"{'' if not np.isfinite(dt_prev) else format(dt_prev, '8.1f'):>8} "
              f"{area:6d} "
              f"{'' if not (present and np.isfinite(a_pre[1])) else format(a_pre[1], '7.1f'):>7} "
              f"{'' if not (present and np.isfinite(a_post[1])) else format(a_post[1], '7.1f'):>7} "
              f"{'' if not np.isfinite(dT) else format(dT, '7.1f'):>7} {state:>9}")

        rows.append((float(t), float(pos_start[i-1]), float(pos_min[i-1]),
                     float(peak[i-1]), dt_prev, float(area),
                     a_pre[1], a_pre[2], a_pre[3], a_pre[4],
                     a_post[1], float(present)))
        prev_t = float(t)

    m.close()
    arr = np.array(rows, dtype=np.float64)
    np.savez_compressed(
        OUT, table=arr,
        columns=np.array(["t0_s", "gap_in_mm", "gap_out_mm", "peak_kN", "dt_prev_s",
                          "blob_px", "T_pre_p50", "T_pre_p90", "T_pre_p99",
                          "T_pre_max", "T_post_p50", "present"]))
    print(f"\nwrote {OUT}")

    ok = arr[arr[:, 11] > 0]
    print(f"\nblows with the bar in frame: {len(ok)} of {len(arr)}")
    if len(ok):
        print(f"T_pre p50 range: {np.nanmin(ok[:, 6]):.1f} - {np.nanmax(ok[:, 6]):.1f} C")
    return 0


sys.exit(main())
