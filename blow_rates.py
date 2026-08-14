#!/usr/bin/env python3
"""Measured strain rate per blow, all 47 -- replaces the '1 /s recommended' placeholder.

docs/316L_MECHANICAL_PROPERTIES.md open item 1 recommends a nominal 1 /s and says
the measurement "needs one mcap recorded during a forging run". That mcap exists
and is extracted (t4_press*.npz, 2026-08-13). This derives the rate the flow-stress
fit should be evaluated at, for EVERY blow rather than for blow 1 alone -- which
matters because blow 1 is the 96th percentile of the session on temperature
(coupling doc 3.4), so anything taken from it is unrepresentative by construction.

`live_position_mm` is the die gap. For a compression blow closing h0 -> h1 in dt:

    engineering rate  ~  (h0 - h1) / dt / h_mean
    true rate         =  ln(h0 / h1) / dt          <- reported as the headline

Both are averages over the closing window, not instantaneous peaks.
"""
import sys

import numpy as np

BLOWS = "/home/timothy/GitHub/Genesis/forge_common/main/outputs/t4_press_blows.npz"

# Song2020 fit domain, from agforge/materials.py
FIT_LO, FIT_HI = 2e-4, 2e-2


def pct(a, p):
    return float(np.percentile(a, p))


def main():
    d = np.load(BLOWS, allow_pickle=True)
    h0 = d["pos_start"].astype(float)
    h1 = d["pos_min"].astype(float)
    dur = d["dur"].astype(float)

    ok = (h0 > 0) & (h1 > 0) & (h1 < h0) & (dur > 0)
    n_drop = int((~ok).sum())
    if n_drop:
        print("dropped %d of %d blows (non-closing or zero duration)" % (n_drop, len(h0)))

    h0, h1, dur = h0[ok], h1[ok], dur[ok]
    hm = 0.5 * (h0 + h1)
    eng = (h0 - h1) / dur / hm
    true = np.log(h0 / h1) / dur
    v = (h0 - h1) / dur

    print("\nn = %d blows" % len(true))
    print("%-22s %8s %8s %8s %8s %8s" % ("", "min", "p25", "median", "p75", "max"))
    for name, a in (("ram speed  [mm/s]", v),
                    ("closure    [mm]", h0 - h1),
                    ("duration   [s]", dur),
                    ("eng rate   [1/s]", eng),
                    ("TRUE rate  [1/s]", true)):
        print("%-22s %8.4f %8.4f %8.4f %8.4f %8.4f"
              % (name, a.min(), pct(a, 25), pct(a, 50), pct(a, 75), a.max()))

    b1 = float(true[0])
    med = pct(true, 50)
    print("\nblow #1 true rate      = %.4f /s   (percentile %.0f of the session)"
          % (b1, 100.0 * (true < b1).mean()))
    print("session median         = %.4f /s" % med)
    print("blow1 / median         = %.2fx" % (b1 / med))

    print("\nSong2020 fit domain    = %.0e .. %.0e /s" % (FIT_LO, FIT_HI))
    print("median is %.0fx ABOVE the top of the fit domain" % (med / FIT_HI))
    print("in-domain blows        = %d of %d" % (int(((true >= FIT_LO) & (true <= FIT_HI)).sum()),
                                                 len(true)))
    print("\ncurrent hardcoded arrhenius_process_strain_rate = 0.41 /s "
          "(blow 1 only); 316L doc open item 1 recommends 1.0 /s")
    print("ratio median/0.41      = %.2fx" % (med / 0.41))
    return 0


sys.exit(main())
