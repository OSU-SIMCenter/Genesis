#!/usr/bin/env python3
"""Independently re-derive the ~960 C billet temperature at blow #1.

The figure is inherited from session 12b6fa7e and a pushed commit rests on it.
Two consecutive handoffs asked for it to be checked and it never was, because
the source file was believed to be pCloud-only. It is not -- it is staged on
Windows and readable over /mnt/c.

A naive re-run of the same extraction would be a MIRROR TEST: same method, same
systematic error, guaranteed agreement, no information. (This project has been
burned by exactly that -- 16 thermal mirror tests all passed because they
re-implemented the same units bug.) So this checks the things that are actually
independent of the original method:

  (a) STRUCTURAL -- does the 06-15 file carry hmr/sensors/thermalcam at all,
      and what else is in it? The claim that flipped once already is about WHICH
      SCENE the camera views, and that is what makes or breaks the number.
  (b) VALUE      -- reproduce the per-frame table around blow #1.
  (c) METHOD SENSITIVITY -- the original isolated the workpiece as the largest
      connected region above max(700 C, p85) and flagged "thresholding, not
      segmentation" as a caveat. Vary the isolation rule and see how far the
      answer actually moves. If it moves more than the +/-50 K emissivity band,
      the stated dominant uncertainty is wrong.
  (d) FRAME DUMP -- write PNGs so the press-vs-coil view can be eyeballed
      rather than argued about.
"""
import sys
from collections import deque

import numpy as np

sys.path.insert(0, "/home/timothy/GitHub/Genesis/aims-genesis/thermal-st-invariance")

from agforge.mcap_thermal import ForgeMcap, THERMAL_TOPIC

MCAP = ("/mnt/c/Users/banko/Documents/forge-data-stage/2026-06-15_T4_bulk/"
        "20260615_180456_T4_bulk.mcap")
OUT = "/home/timothy/billet_temp_check"

# Blow #1 contact is at ~299 s per the earlier decode. That timing is itself
# inherited, so sweep a window around it rather than trusting the point.
T_LO, T_HI = 292.0, 325.0


def largest_blob(mask):
    """Largest 4-connected component of a boolean mask, as a boolean mask."""
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    best = None
    best_n = 0
    for y0 in range(h):
        for x0 in range(w):
            if not mask[y0, x0] or seen[y0, x0]:
                continue
            q = deque([(y0, x0)])
            seen[y0, x0] = True
            cells = []
            while q:
                y, x = q.popleft()
                cells.append((y, x))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        q.append((ny, nx))
            if len(cells) > best_n:
                best_n = len(cells)
                best = cells
    out = np.zeros_like(mask, dtype=bool)
    if best:
        ys, xs = zip(*best)
        out[list(ys), list(xs)] = True
    return out


def stats(vals):
    return dict(n=int(vals.size),
                p50=float(np.percentile(vals, 50)),
                p90=float(np.percentile(vals, 90)),
                p99=float(np.percentile(vals, 99)),
                mx=float(vals.max()))


def main():
    import os
    os.makedirs(OUT, exist_ok=True)

    print("=" * 78)
    print("(a) STRUCTURAL")
    print("=" * 78)
    m = ForgeMcap(MCAP)
    print(f"duration        : {m.duration_s:.1f} s")
    print(f"chunks          : {len(m.chunks)}")
    print(f"channels        : {len(m.channels)}")
    for cid, topic in sorted(m.channels.items(), key=lambda kv: kv[1]):
        mark = "  <== THERMAL" if topic == THERMAL_TOPIC else ""
        print(f"   [{cid:3d}] {topic}{mark}")
    if THERMAL_TOPIC not in m.channels.values():
        print("\nFAIL: no thermal camera topic in this file")
        return 1

    print()
    print("=" * 78)
    print(f"(b)+(c) FRAMES AND METHOD SENSITIVITY, t in [{T_LO}, {T_HI}] s")
    print("=" * 78)

    frames = list(m.thermal_frames(T_LO, T_HI, frames_per_chunk=1))
    print(f"frames sampled  : {len(frames)}")
    if not frames:
        print("FAIL: no thermal frames in the window")
        return 1
    f0 = frames[0]
    print(f"frame shape     : {f0.kelvin.shape}  (expect 288x382)")
    print()

    hdr = (f"{'t_s':>7} {'meth':>10} {'npx':>7} "
           f"{'p50':>8} {'p90':>8} {'p99':>8} {'max':>8}")
    print(hdr)
    print("-" * len(hdr))

    table = {}
    for fr in frames:
        c = fr.celsius
        methods = {}

        # 1. the original: largest connected region above max(700, p85)
        thr = max(700.0, float(np.percentile(c, 85)))
        mask = c >= thr
        blob = largest_blob(mask)
        if blob.any():
            methods["blob>700/p85"] = c[blob]

        # 2. same threshold, NO connectivity requirement
        if mask.any():
            methods["thresh only"] = c[mask]

        # 3. a fixed hot threshold, no percentile coupling
        m900 = c >= 900.0
        if m900.any():
            methods["fixed >900"] = c[m900]

        # 4. hottest 10% of the frame, no threshold at all
        k = int(c.size * 0.10)
        methods["top 10%"] = np.sort(c.ravel())[-k:]

        for name, vals in methods.items():
            s = stats(vals)
            print(f"{fr.t_s:7.1f} {name:>10} {s['n']:7d} "
                  f"{s['p50']:8.1f} {s['p90']:8.1f} {s['p99']:8.1f} {s['mx']:8.1f}")
            table.setdefault(name, []).append((fr.t_s, s["p50"]))
        print()

    print("Cross-method spread is computed per frame by dump_frames.py -- keyed by\n"
          "frame rather than by column index, because a method can be absent from\n"
          "early frames (nothing exceeds 900 C before the bar arrives) and column\n"
          "alignment then silently compares different frames.")
    print()

    # (d) dump frames for visual confirmation of what the camera views
    print()
    print("=" * 78)
    print("(d) FRAME DUMP")
    print("=" * 78)
    try:
        from PIL import Image
        for fr in frames:
            if not (295.0 <= fr.t_s <= 303.0):
                continue
            c = fr.celsius
            lo, hi = np.percentile(c, 1), np.percentile(c, 99.9)
            img = np.clip((c - lo) / max(hi - lo, 1e-6), 0, 1)
            rgb = np.zeros(img.shape + (3,), dtype=np.uint8)
            rgb[..., 0] = (np.clip(img * 1.6, 0, 1) * 255).astype(np.uint8)
            rgb[..., 1] = (np.clip(img * 1.6 - 0.4, 0, 1) * 255).astype(np.uint8)
            rgb[..., 2] = (np.clip(img * 1.6 - 0.9, 0, 1) * 255).astype(np.uint8)
            p = f"{OUT}/t{fr.t_s:07.2f}.png"
            Image.fromarray(rgb).resize((img.shape[1] * 2, img.shape[0] * 2),
                                        Image.NEAREST).save(p)
            print(f"wrote {p}")
    except ImportError:
        print("PIL not available -- skipping render")

    m.close()
    return 0


sys.exit(main())
