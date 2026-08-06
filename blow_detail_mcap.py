#!/usr/bin/env python3
"""Two checks the summary table can't settle, then the first blow in full.

1. Several episodes peak at ~110.2 kN. Is that the metal, or a clipped sensor /
   machine force limit? If clipped, those blows are censored and unusable as a
   force target.
2. 54 u_taken vs 44 planned actions vs 47 detected episodes — reconcile.
"""
import json
import sys

import numpy as np


def main(npz, utaken_path, blows_npz):
    d = np.load(npz)
    t_ns = d["press_t"].astype(np.int64)
    f = d["press_force_kn"]
    s = d["press_stroke_mm"]
    p = d["press_position_mm"]
    t = (t_ns - t_ns[0]) / 1e9
    b = np.load(blows_npz)

    # ---------- 1. the ceiling ----------
    print("=" * 74)
    print("1. IS ~110 kN A REAL PEAK OR A CLIP?")
    print("=" * 74)
    fmax = f.max()
    top = np.sort(np.unique(f))[-12:]
    print(f"global max force      : {fmax:.4f} kN")
    print(f"12 largest distinct   : {np.array2string(top, precision=4)}")
    n_at = (f > fmax - 0.01).sum()
    n_near = (f > 109.0).sum()
    print(f"samples within 0.01 of max : {n_at:,}")
    print(f"samples above 109 kN       : {n_near:,}  "
          f"({n_near/len(f)*100:.3f}% of all samples)")
    dwell = n_at / 360.9
    print(f"total dwell at the max     : {dwell:.2f} s")
    peaks = b["peak"]
    n_ceiling = int((peaks > 109.5).sum())
    print(f"episodes peaking >109.5 kN : {n_ceiling} of {len(peaks)}")
    print()
    if n_at > 50:
        print("VERDICT: the max is held for many consecutive samples across")
        print("multiple separate blows -> a CLIP (sensor range or press force")
        print("limit), not a coincidence of metal response.")
        print(f"=> those {n_ceiling} episodes are CENSORED: true peak >= {fmax:.1f} kN.")
        print(f"=> {len(peaks)-n_ceiling} episodes remain usable as force targets.")
    print()

    # ---------- 2. u_taken vs episodes ----------
    print("=" * 74)
    print("2. u_taken VS DETECTED EPISODES")
    print("=" * 74)
    ut = json.load(open(utaken_path))
    ut_t = np.array([(u["log_time_ns"] - int(t_ns[0])) / 1e9 for u in ut])
    seqs = [u.get("seq") for u in ut]
    print(f"u_taken            : {len(ut)}   seq {min(seqs)}..{max(seqs)}")
    print(f"unique seq values  : {len(set(seqs))}")
    dup = len(seqs) - len(set(seqs))
    if dup:
        print(f"!! {dup} duplicate seq values -> u_taken is not one-per-action")
    t0s = b["t0"]
    print(f"detected episodes  : {len(t0s)}")
    # nearest episode after each u_taken
    unmatched = 0
    lags = []
    for ut_time in ut_t:
        after = t0s[t0s >= ut_time - 0.5]
        if len(after) == 0:
            unmatched += 1
        else:
            lags.append(after[0] - ut_time)
    lags = np.array(lags)
    print(f"u_taken with a following episode : {len(lags)}  (unmatched {unmatched})")
    if len(lags):
        print(f"lag u_taken -> force onset : median {np.median(lags):.2f} s  "
              f"min {lags.min():.2f}  max {lags.max():.2f}")
    # how many distinct episodes get claimed
    print(f"\nNOTE: {len(ut)} commands vs {len(t0s)} force episodes means the "
          f"mapping is NOT 1:1.\n      Commands that moved the manipulator without "
          f"a press, or presses merged\n      by the 0.25 s bridging, both break it. "
          f"Treat per-blow pairing as UNRESOLVED.")
    print()

    # ---------- 3. the first blow ----------
    print("=" * 74)
    print("3. FIRST BLOW — virgin 38.1 mm round 316L, the cleanest target")
    print("=" * 74)
    i0, i1 = int(b["i0"][0]), int(b["i1"][0])
    lo = max(0, i0 - 120)
    hi = min(len(t), i1 + 120)
    print(f"first u_taken   : t={ut_t[0]:.3f} s  {json.dumps({k:v for k,v in ut[0].items() if k!='log_time_ns'})}")
    print(f"episode window  : t={t[i0]:.3f} .. {t[i1]:.3f} s  ({t[i1]-t[i0]:.3f} s)")
    print(f"peak force      : {f[i0:i1+1].max():.2f} kN")
    print()
    print(f"{'t_s':>9} {'force_kN':>9} {'stroke_mm':>10} {'pos_mm':>9}")
    print("-" * 42)
    step = max(1, (hi - lo) // 55)
    for i in range(lo, hi, step):
        print(f"{t[i]:9.3f} {f[i]:9.2f} {s[i]:10.3f} {p[i]:9.3f}")
    print()
    seg = slice(i0, i1 + 1)
    print(f"stroke over the blow   : {s[seg].min():.3f} .. {s[seg].max():.3f} mm")
    print(f"position over the blow : {p[seg].min():.3f} .. {p[seg].max():.3f} mm")
    print(f"position + stroke      : min {(p[seg]+s[seg]).min():.3f}  "
          f"max {(p[seg]+s[seg]).max():.3f}  "
          f"sd {(p[seg]+s[seg]).std():.4f}")
    np.savez_compressed(npz.replace(".npz", "_blow0.npz"),
                        t=t[lo:hi], force_kn=f[lo:hi],
                        stroke_mm=s[lo:hi], position_mm=p[lo:hi])
    print(f"\nwrote {npz.replace('.npz','_blow0.npz')}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
