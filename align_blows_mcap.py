#!/usr/bin/env python3
"""Align blows to commands by TIME, not by list index.

The index alignment breaks the moment a retry inserts an extra episode. Each
u_taken carries its own rho plus a timestamp, so pair each command with the
force episode that follows it and before the next command.
"""
import json
import sys

import numpy as np


def main(npz, blows_npz, utaken_path):
    d = np.load(npz)
    t_ns = d["press_t"].astype(np.int64)
    b = np.load(blows_npz)
    t0s, pos_min, peak = b["t0"], b["pos_min"], b["peak"]

    ut = json.load(open(utaken_path))
    ut_t = np.array([(u["log_time_ns"] - int(t_ns[0])) / 1e9 for u in ut])
    order = np.argsort(ut_t)
    ut = [ut[i] for i in order]
    ut_t = ut_t[order]

    print(f"{'seq':>4} {'t_cmd':>9} {'rho':>6} {'2rho':>7} {'gap':>7} {'err':>7} "
          f"{'peak':>7}  verdict")
    print("-" * 68)
    used = set()
    reached, short_fl, short_other, unmatched = [], [], [], 0
    for k, (u, tu) in enumerate(zip(ut, ut_t)):
        t_next = ut_t[k + 1] if k + 1 < len(ut_t) else 1e18
        cand = np.flatnonzero((t0s >= tu - 0.5) & (t0s < t_next - 0.2))
        cand = [c for c in cand if c not in used]
        if not cand:
            unmatched += 1
            print(f"{u['seq']:4d} {tu:9.2f} {u['rho']:6.2f} {2*u['rho']:7.2f} "
                  f"{'--':>7} {'--':>7} {'--':>7}  no press")
            continue
        j = cand[0]
        used.add(j)
        tgt = 2 * u["rho"]
        err = pos_min[j] - tgt
        fl = peak[j] > 109.5
        if abs(err) < 0.15:
            v = "reached"; reached.append(peak[j])
        elif fl:
            v = "SHORT (force-limited)"; short_fl.append(err)
        else:
            v = "SHORT" if err > 0 else "overshot"; short_other.append(err)
        print(f"{u['seq']:4d} {tu:9.2f} {u['rho']:6.2f} {tgt:7.2f} {pos_min[j]:7.2f} "
              f"{err:7.2f} {peak[j]:7.1f}  {v}")

    n = len(reached) + len(short_fl) + len(short_other)
    print()
    print(f"commands           : {len(ut)}   (no press: {unmatched})")
    print(f"paired with a blow : {n}")
    print(f"  reached 2*rho          : {len(reached)}")
    print(f"  short, force-limited   : {len(short_fl)}")
    print(f"  other mismatch         : {len(short_other)}")
    if reached:
        r = np.array(reached)
        print(f"\npeak force on 'reached' blows: median {np.median(r):.1f} kN  "
              f"max {r.max():.1f}  (n={len(r)})")
    if short_fl:
        print(f"force-limited blows fall short by: "
              f"{np.mean(short_fl):.2f} mm mean, max {np.max(short_fl):.2f} mm")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
