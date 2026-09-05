"""WHERE does the interpenetration happen?  (corrected metric)

The first attempt binned local density inside R = 2*psize. That radius spans ~33 neighbours
and smooths over exactly the sub-psize crowding we are hunting, so it reported "no spatial
structure" for a metric that could not have seen it.

The right per-particle quantity is the one that actually sums to the union deficit:
    overlap_i = SUM_j V_lens(r, d_ij)   over neighbours with d_ij < 2r,  r = psize/2
    V_lens(r,d) = (pi/12) (4r + d) (2r - d)^2
Half the total (each pair counted twice) approximates N*v_ball - V_union, so we can check
it against the Monte-Carlo union volume before trusting the spatial breakdown.
"""
import json
import os
import sqlite3
import sys

import numpy as np
from scipy.spatial import cKDTree

OUT = os.path.expanduser("~/GitHub/Genesis/forge_common/main/outputs")


def geom(cpd):
    bgd = int(cpd / 0.040)
    dx = 1000.0 / bgd
    return dx, dx / 2.0


def load(db, hit):
    con = sqlite3.connect("file:%s?mode=ro" % db, uri=True)
    r = con.execute("SELECT vertices FROM hits WHERE step_number=?", (hit,)).fetchone()
    con.close()
    return None if r is None else np.asarray(json.loads(r[0]), dtype=np.float64).reshape(-1, 3)


def overlap_per_particle(P, r):
    """V_lens summed over each particle's overlapping neighbours."""
    tree = cKDTree(P)
    pairs = tree.query_pairs(2.0 * r, output_type="ndarray")
    out = np.zeros(len(P))
    if len(pairs) == 0:
        return out, 0.0
    d = np.linalg.norm(P[pairs[:, 0]] - P[pairs[:, 1]], axis=1)
    v = (np.pi / 12.0) * (4.0 * r + d) * (2.0 * r - d) ** 2
    np.add.at(out, pairs[:, 0], v)
    np.add.at(out, pairs[:, 1], v)
    return out, float(v.sum())


def bins(label, coord, val, nb=8):
    qs = np.quantile(coord, np.linspace(0, 1, nb + 1))
    print("   %-24s %8s %8s %10s %10s" % (label, "lo", "hi", "mean_ovl", "share%"))
    tot = val.sum()
    for i in range(nb):
        m = (coord >= qs[i]) & ((coord <= qs[i + 1]) if i == nb - 1 else (coord < qs[i + 1]))
        if m.sum() < 10:
            continue
        print("   %-24s %8.2f %8.2f %10.4f %10.1f"
              % ("", qs[i], qs[i + 1], val[m].mean(), 100.0 * val[m].sum() / tot))


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "velo_dlv_c10_r1"
    cpd = float(sys.argv[2]) if len(sys.argv) > 2 else 10.0
    db = os.path.join(OUT, tag + ".db")
    dx, ps = geom(cpd)
    r = ps / 2.0
    v_ball = (4.0 / 3.0) * np.pi * r ** 3

    con = sqlite3.connect("file:%s?mode=ro" % db, uri=True)
    H = con.execute("SELECT MAX(step_number) FROM hits").fetchone()[0]
    con.close()
    P1, PH = load(db, 1), load(db, H)
    o1, t1 = overlap_per_particle(P1, r)
    oH, tH = overlap_per_particle(PH, r)
    N = len(PH)

    print("=" * 78)
    print("%s  cpd=%g  psize=%.4f  r=%.4f mm  N=%d  final hit=%d" % (tag, cpd, ps, r, N, H))
    print("=" * 78)
    print("\nTOTAL PAIRWISE OVERLAP (cross-check against Monte-Carlo union deficit)")
    print("   hit 1 : sum V_lens = %9.1f mm3   (%.2f%% of N*v_ball=%.1f)" % (t1, 100 * t1 / (N * v_ball), N * v_ball))
    print("   hit %-2d: sum V_lens = %9.1f mm3   (%.2f%%)" % (H, tH, 100 * tH / (N * v_ball)))
    print("   growth in overlap  = %9.1f mm3" % (tH - t1))

    print("\nPER-PARTICLE OVERLAP DISTRIBUTION (mm3, ball volume = %.4f)" % v_ball)
    for nm, o in (("hit 1", o1), ("hit %d" % H, oH)):
        print("   %-7s zero:%5.1f%%  median %.4f  p90 %.4f  p99 %.4f  max %.4f"
              % (nm, 100.0 * (o == 0).mean(), np.median(o), np.quantile(o, .9), np.quantile(o, .99), o.max()))

    cx = PH[:, 0]
    yc, zc = PH[:, 1] - np.median(PH[:, 1]), PH[:, 2] - np.median(PH[:, 2])
    span = [np.ptp(PH[:, i]) / np.ptp(P1[:, i]) for i in (1, 2)]
    comp = 1 + int(np.argmin(span))
    cc = yc if comp == 1 else zc
    ff = zc if comp == 1 else yc
    rad = np.hypot(yc, zc)

    print("\n--- WHERE THE OVERLAP LIVES AT HIT %d ---" % H)
    print("\nBY AXIAL POSITION x")
    bins("x [mm]", cx, oH)
    print("\nBY COMPRESSION AXIS |%s| (large = die contact face)" % "xyz"[comp])
    bins("|%s| [mm]" % "xyz"[comp], np.abs(cc), oH)
    print("\nBY TRANSVERSE RADIUS (0 = core, large = skin)")
    bins("radius [mm]", rad, oH)

    q = np.quantile(rad, [0.33, 0.67])
    core, skin = oH[rad < q[0]], oH[rad > q[1]]
    print("\n   core (inner third) mean overlap = %.4f   share %.1f%%"
          % (core.mean(), 100 * core.sum() / oH.sum()))
    print("   skin (outer third) mean overlap = %.4f   share %.1f%%"
          % (skin.mean(), 100 * skin.sum() / oH.sum()))
    verdict = "SKIN/CONTACT-dominated" if skin.mean() > 1.25 * core.mean() else (
              "CORE/BULK-dominated" if core.mean() > 1.25 * skin.mean() else "BROADLY DISTRIBUTED")
    print("   => %s" % verdict)

    json.dump(dict(tag=tag, cpd=cpd, r=r, final_hit=int(H), N=N,
                   total_overlap_hit1=t1, total_overlap_final=tH,
                   core_mean=float(core.mean()), skin_mean=float(skin.mean()),
                   core_share=float(core.sum() / oH.sum()), skin_share=float(skin.sum() / oH.sum()),
                   verdict=verdict, comp_axis="xyz"[comp]),
              open(os.path.join(OUT, "review", "where2_%s.json" % tag), "w"), indent=1)
    print("\nwrote review/where2_%s.json" % tag)


if __name__ == "__main__":
    main()
