"""Is the particle piling a TIME-integration artifact or a SPATIAL/structural one?

CPIC is already enabled, so material is not leaking through the die -- yet packing still
collapses ~12%. That rules out the boundary-smearing explanation and points at sub-grid
particle clumping, which the grid's smooth velocity gradient cannot see (and which is
therefore invisible to det F by construction, not by accident).

If clumping accumulates per unit of integrated compression, halving dt should not help.
If it accumulates per timestep (a per-step projection/rounding effect), it should.

Free experiment: three cfl_safety levels already exist on disk at res 7, all completed
past hit 10. Compare packing efficiency at a COMMON hit number.
"""
import json
import os
import sqlite3

import numpy as np
from scipy.spatial import cKDTree

OUT = os.path.expanduser("~/GitHub/Genesis/forge_common/main/outputs")
RNG = np.random.default_rng(7)

# res 7 throughout, so psize is identical across every run compared here
PSIZE = 1000.0 / int(7 / 0.040) / 2.0

LADDER = [
    ("cfl 0.90 (stock dt)", ["velo_rep_c7_r1", "velo_rep_c7_r2", "velo_rep_c7_r3"]),
    ("cfl 0.45 (half dt)", ["velo_conf_045_r1", "velo_conf_045_r2", "velo_conf_045_r3"]),
    ("cfl 0.225 (quarter dt)", ["velo_conf_0225_r1", "velo_conf_0225_r2", "velo_conf_0225_r3"]),
]


def load(db, hit):
    if not os.path.exists(db):
        return None
    con = sqlite3.connect("file:%s?mode=ro" % db, uri=True)
    r = con.execute("SELECT vertices FROM hits WHERE step_number=?", (hit,)).fetchone()
    con.close()
    return None if r is None else np.asarray(json.loads(r[0]), dtype=np.float64).reshape(-1, 3)


def union_balls(P, r, nsamp=3_000_000):
    lo, hi = P.min(0) - 1.5 * r, P.max(0) + 1.5 * r
    box = float(np.prod(hi - lo))
    tree = cKDTree(P)
    hits = done = 0
    while done < nsamp:
        m = min(400_000, nsamp - done)
        d, _ = tree.query(RNG.uniform(lo, hi, size=(m, 3)), k=1, distance_upper_bound=r)
        hits += int(np.isfinite(d).sum())
        done += m
    return box * hits / done


def eta(P, r):
    v_ball = (4.0 / 3.0) * np.pi * r ** 3
    return union_balls(P, r) / (len(P) * v_ball)


def main():
    r = PSIZE / 2.0
    HIT = 10          # deepest hit reached by every stock-dt repeat
    print("=" * 78)
    print("PILING vs TIMESTEP   (res 7, psize=%.4f mm, ball r=%.4f mm, compared at hit %d)" % (PSIZE, r, HIT))
    print("=" * 78)
    print("\n%-24s %10s %10s %10s %10s" % ("setting", "run", "eta(h1)", "eta(h%d)" % HIT, "drop%"))
    summary = []
    for label, tags in LADDER:
        drops = []
        for t in tags:
            db = os.path.join(OUT, t + ".db")
            P1, PH = load(db, 1), load(db, HIT)
            if P1 is None or PH is None:
                print("%-24s %10s %10s" % (label, t.split("_")[-1], "  (no hit %d)" % HIT))
                continue
            e1, eH = eta(P1, r), eta(PH, r)
            d = (eH / e1 - 1) * 100
            drops.append(d)
            print("%-24s %10s %10.4f %10.4f %+10.2f" % (label, t.split("_")[-1], e1, eH, d))
            label = ""
        if drops:
            summary.append((tags[0], float(np.mean(drops)), float(np.std(drops)), len(drops)))
            print("%-24s %10s %10s %10s %+10.2f  <- mean of %d" % ("", "", "", "", np.mean(drops), len(drops)))
        print()

    print("-" * 78)
    print("VERDICT")
    if len(summary) >= 2:
        vals = [s[1] for s in summary]
        spread = max(vals) - min(vals)
        noise = float(np.mean([s[2] for s in summary]))
        print("  packing drop across a 4x dt range: %s" % ", ".join("%+.2f%%" % v for v in vals))
        print("  spread across dt = %.2f pp ; within-setting scatter = %.2f pp" % (spread, noise))
        if spread <= max(2.0 * noise, 1.0):
            print("  => dt does NOT drive the piling. It is a SPATIAL/sub-grid effect;")
            print("     refining the timestep cannot fix it, and neither will a contact")
            print("     mode that only changes how boundary VELOCITY is enforced.")
        else:
            print("  => dt DOES move the piling; a per-step integration effect is implicated.")
    json.dump([dict(run=s[0], mean_drop=s[1], std=s[2], n=s[3]) for s in summary],
              open(os.path.join(OUT, "review", "piling_vs_dt.json"), "w"), indent=1)
    print("\nwrote review/piling_vs_dt.json")


if __name__ == "__main__":
    main()
