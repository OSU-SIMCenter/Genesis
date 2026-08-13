#!/usr/bin/env python3
"""Render 06-15 thermal frames around blow #1, and key the method-spread by time.

The claim under test is not the number -- it is WHAT THE CAMERA IS POINTED AT.
That claim already flipped once (this project believed the 06-15 camera watched
the induction coil; it watches the press), and every temperature conclusion
rides on it. A render settles it by inspection instead of by argument.
"""
import sys
import numpy as np

sys.path.insert(0, "/home/timothy/GitHub/Genesis/aims-genesis/thermal-st-invariance")
from agforge.mcap_thermal import ForgeMcap

MCAP = ("/mnt/c/Users/banko/Documents/forge-data-stage/2026-06-15_T4_bulk/"
        "20260615_180456_T4_bulk.mcap")
OUT = "/home/timothy/billet_temp_check"


def main():
    import os
    os.makedirs(OUT, exist_ok=True)
    from PIL import Image

    m = ForgeMcap(MCAP)
    frames = list(m.thermal_frames(292.0, 312.0, frames_per_chunk=1))

    # method spread, keyed by frame so a method that is absent early cannot
    # shift the columns (the bug in the first version of this check)
    print(f"{'t_s':>7} {'blob':>8} {'thresh':>8} {'>900':>8} {'top10':>8} {'spread':>8}")
    for fr in frames:
        c = fr.celsius
        thr = max(700.0, float(np.percentile(c, 85)))
        mask = c >= thr
        vals = {}
        if mask.any():
            vals["thresh"] = float(np.median(c[mask]))
        m900 = c >= 900.0
        if m900.any():
            vals[">900"] = float(np.median(c[m900]))
        k = int(c.size * 0.10)
        vals["top10"] = float(np.median(np.sort(c.ravel())[-k:]))
        # blob = largest connected region, via a label-free flood from the hottest pixel
        if mask.any():
            from collections import deque
            h, w = c.shape
            seen = np.zeros_like(mask)
            iy, ix = np.unravel_index(np.argmax(np.where(mask, c, -1e9)), c.shape)
            q = deque([(iy, ix)]); seen[iy, ix] = True; cells = []
            while q:
                y, x = q.popleft(); cells.append((y, x))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True; q.append((ny, nx))
            ys, xs = zip(*cells)
            vals["blob"] = float(np.median(c[list(ys), list(xs)]))
        spread = max(vals.values()) - min(vals.values())
        row = f"{fr.t_s:7.1f}"
        for key in ("blob", "thresh", ">900", "top10"):
            row += f" {vals[key]:8.1f}" if key in vals else f" {'--':>8}"
        print(row + f" {spread:8.1f}")

    # renders
    print()
    for fr in frames:
        if not (295.0 <= fr.t_s <= 303.0):
            continue
        c = fr.celsius
        lo, hi = float(np.percentile(c, 2)), float(np.percentile(c, 99.8))
        img = np.clip((c - lo) / max(hi - lo, 1e-6), 0, 1)
        rgb = np.zeros(img.shape + (3,), dtype=np.uint8)
        rgb[..., 0] = (np.clip(img * 1.7, 0, 1) * 255).astype(np.uint8)
        rgb[..., 1] = (np.clip(img * 1.7 - 0.45, 0, 1) * 255).astype(np.uint8)
        rgb[..., 2] = (np.clip(img * 1.7 - 0.95, 0, 1) * 255).astype(np.uint8)
        p = f"{OUT}/t{fr.t_s:07.2f}.png"
        Image.fromarray(rgb).resize((img.shape[1] * 2, img.shape[0] * 2),
                                    Image.NEAREST).save(p)
        print(f"wrote {p}  range {lo:.0f}-{hi:.0f} C")
    m.close()
    return 0


sys.exit(main())
