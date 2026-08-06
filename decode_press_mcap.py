#!/usr/bin/env python3
"""Test the decode that the first blow suggests, against every episode.

Hypotheses:
  H1  live_position_mm + live_stroke_mm == 227.3 everywhere (one channel, two ends)
  H2  live_position_mm IS the die gap: contact starts at the billet diameter
  H3  the plan's `rho` is HALF-thickness, so the achieved gap should be 2*rho
  H4  episodes that miss 2*rho are the ones that hit the ~110 kN force limit
"""
import json
import sys

import numpy as np


def main(npz, blows_npz, plan_path):
    d = np.load(npz)
    f = d["press_force_kn"]
    s = d["press_stroke_mm"]
    p = d["press_position_mm"]
    b = np.load(blows_npz)

    print("=" * 74)
    print("H1  position + stroke is constant")
    print("=" * 74)
    tot = p + s
    print(f"  min {tot.min():.4f}  max {tot.max():.4f}  sd {tot.std():.6f}  "
          f"over {len(tot):,} samples")
    print("  => CONFIRMED: one displacement channel expressed from both ends"
          if tot.std() < 1e-6 else "  => NOT constant")
    print()

    plan = [json.loads(l) for l in open(plan_path) if l.strip()]
    rho = np.array([q["action"]["rho"] for q in plan])
    print("=" * 74)
    print("H3/H4  achieved gap vs commanded 2*rho, in plan order")
    print("=" * 74)
    pos_min = b["pos_min"]
    peak = b["peak"]
    n = min(len(rho), len(pos_min))
    print(f"{'#':>3} {'rho':>7} {'2*rho':>7} {'gap':>7} {'err_mm':>8} "
          f"{'peak_kN':>8}  verdict")
    print("-" * 66)
    hits = miss = 0
    miss_peaks, hit_peaks = [], []
    for i in range(n):
        target = 2 * rho[i]
        err = pos_min[i] - target
        if abs(err) < 0.15:
            v = "reached"
            hits += 1
            hit_peaks.append(peak[i])
        else:
            v = "SHORT" if err > 0 else "overshot"
            miss += 1
            miss_peaks.append(peak[i])
        print(f"{i:3d} {rho[i]:7.2f} {target:7.2f} {pos_min[i]:7.2f} {err:8.2f} "
              f"{peak[i]:8.1f}  {v}")
    print()
    print(f"reached commanded gap : {hits}/{n}")
    print(f"missed                : {miss}/{n}")
    if miss_peaks:
        mp = np.array(miss_peaks); hp = np.array(hit_peaks)
        print(f"\npeak force when gap REACHED : median {np.median(hp):6.1f} kN  "
              f"range {hp.min():.1f}-{hp.max():.1f}")
        print(f"peak force when gap MISSED  : median {np.median(mp):6.1f} kN  "
              f"range {mp.min():.1f}-{mp.max():.1f}")
        print(f"\n=> {(mp > 109.5).sum()}/{len(mp)} of the misses peak above 109.5 kN")
        print(f"=> {(hp > 109.5).sum()}/{len(hp)} of the hits do")
        if (mp > 109.5).mean() > 0.5 and (hp > 109.5).mean() < 0.2:
            print("\nH4 CONFIRMED: the press is FORCE-LIMITED near 110.2 kN.")
            print("Blows that miss their commanded reduction are the ones that")
            print("hit that limit -- so the limit is a CONTROL stop, and the")
            print("force reading itself is genuine (not a censored sensor).")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
