#!/usr/bin/env python3
"""Characterise the T4 press channel and segment it into individual blows.

Deliberately starts by describing the signal rather than assuming it: the sign
convention, the rest baseline and whether `live_stroke_mm` carries anything are
all unknown going in. The 07-17 file had stroke pinned at exactly 0.0.
"""
import json
import sys

import numpy as np


def main(npz_path, utaken_path):
    d = np.load(npz_path)
    t_ns = d["press_t"].astype(np.int64)
    f = d["press_force_kn"]
    s = d["press_stroke_mm"]
    p = d["press_position_mm"]
    flags = d["press_flags"]          # is_idle, ready, cycle_end, pass, end

    t = (t_ns - t_ns[0]) / 1e9
    print(f"press samples : {len(t):,}")
    print(f"duration      : {t[-1]:.2f} s   mean rate {len(t)/t[-1]:.1f} Hz")
    print()

    def describe(name, a):
        finite = np.isfinite(a)
        a = a[finite]
        uniq = np.unique(a)
        print(f"{name:20s} min={a.min():12.4f}  max={a.max():12.4f}  "
              f"mean={a.mean():10.4f}  n_unique={len(uniq):,}")
        return uniq

    describe("live_force_kn", f)
    su = describe("live_stroke_mm", s)
    describe("live_position_mm", p)
    print()

    if len(su) <= 5:
        print(f"!! live_stroke_mm is effectively constant: {su[:5]}")
        print("   -> stroke is NOT usable here; position is the displacement channel")
    print()

    # ---- flags ----
    names = ["is_idle", "ready", "cycle_end", "pass", "end"]
    print("flag duty (fraction of samples true):")
    for i, nm in enumerate(names):
        print(f"  {nm:12s} {flags[:, i].mean()*100:6.2f}%")
    print()

    # ---- baseline / noise, taken from idle samples ----
    idle = flags[:, 0]
    base = f[idle]
    print(f"idle samples  : {idle.sum():,}  ({idle.mean()*100:.1f}%)")
    print(f"idle force    : mean={base.mean():.4f} kN  sd={base.std():.4f}  "
          f"p01={np.percentile(base,1):.3f}  p99={np.percentile(base,99):.3f}")
    print()

    # Work with magnitude; sign convention is established below.
    absf = np.abs(f)
    thr = max(5.0, np.abs(base).mean() + 20 * base.std())
    print(f"blow threshold: |force| > {thr:.3f} kN")

    hot = absf > thr
    # segment contiguous runs, bridging gaps shorter than 0.25 s
    idx = np.flatnonzero(hot)
    if len(idx) == 0:
        print("!! no samples exceed the threshold — no blows detected")
        return
    breaks = np.flatnonzero(np.diff(t[idx]) > 0.25)
    starts = np.concatenate(([idx[0]], idx[breaks + 1]))
    ends = np.concatenate((idx[breaks], [idx[-1]]))

    print(f"\ndetected {len(starts)} force episodes\n")
    print(f"{'#':>3} {'t_start':>9} {'dur_s':>7} {'peak_kN':>9} {'signed':>9} "
          f"{'pos_start':>10} {'pos_min':>9} {'travel':>8}")
    print("-" * 78)
    rows = []
    for k, (a, b) in enumerate(zip(starts, ends)):
        sl = slice(max(0, a - 30), min(len(t), b + 30))
        seg_f = f[a:b + 1]
        seg_p = p[sl]
        pk = np.nanmax(np.abs(seg_f))
        signed = seg_f[np.nanargmax(np.abs(seg_f))]
        p0 = p[max(0, a - 30)]
        pmin = np.nanmin(seg_p)
        rows.append(dict(i=k, t0=float(t[a]), dur=float(t[b] - t[a]),
                         peak=float(pk), signed=float(signed),
                         pos_start=float(p0), pos_min=float(pmin),
                         travel=float(p0 - pmin), i0=int(a), i1=int(b)))
        if k < 60:
            print(f"{k:3d} {t[a]:9.2f} {t[b]-t[a]:7.3f} {pk:9.2f} {signed:9.2f} "
                  f"{p0:10.2f} {pmin:9.2f} {p0-pmin:8.2f}")
    if len(starts) > 60:
        print(f"... {len(starts)-60} more")

    peaks = np.array([r["peak"] for r in rows])
    print(f"\npeak force: min={peaks.min():.1f}  median={np.median(peaks):.1f}  "
          f"max={peaks.max():.1f} kN")

    # ---- u_taken alignment ----
    try:
        with open(utaken_path) as fh:
            ut = json.load(fh)
    except Exception as e:
        print(f"\n(u_taken unreadable: {e})")
        ut = []
    print(f"\nu_taken messages: {len(ut)}")
    if ut:
        ut_t = [(u["log_time_ns"] - int(t_ns[0])) / 1e9 for u in ut]
        print(f"  span {ut_t[0]:.2f} .. {ut_t[-1]:.2f} s")
        print("  first record:")
        print("   ", json.dumps({k: v for k, v in ut[0].items()
                                 if k != "log_time_ns"})[:400])

    np.savez_compressed(npz_path.replace(".npz", "_blows.npz"),
                        **{k: np.array([r[k] for r in rows]) for k in
                           ("t0", "dur", "peak", "signed", "pos_start",
                            "pos_min", "travel", "i0", "i1")})
    print(f"\nwrote {npz_path.replace('.npz', '_blows.npz')}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
