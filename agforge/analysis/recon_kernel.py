"""How much of the 'geometric volume loss' is a modelling choice?

recon_converge.py converged the marching-cubes GRID and got -6.1% at res10, where
Genesis's own mesher reported -11.95%. Two reconstructions of the same particle cloud
disagreeing by 2x means the number depends on the reconstruction, not just the physics.

The reconstruction has one remaining free parameter: the smoothing length h. So sweep it.
  * if the drop is flat in h  -> the number is robust and we can quote it
  * if the drop swings with h -> 'geometric volume' is not well defined for a particle
                                 cloud, and BOTH -6% and -12% are choices, not measurements

Also computes a parameter-light control: UNION-OF-BALLS volume at radius psize/2 (each
particle's own half-size, the one radius that is not a free choice -- at t=0 the lattice
spacing is exactly psize, so neighbouring balls precisely touch). Packing efficiency
    eta = V_union / (N * v_ball)
starts at ~1 and can ONLY fall by particles interpenetrating. That is the user's original
concern measured directly, with no kernel and no isosurface.
"""
import json
import os
import sqlite3
import sys

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree
from skimage.measure import marching_cubes

OUT = os.path.expanduser("~/GitHub/Genesis/forge_common/main/outputs")
RNG = np.random.default_rng(12345)


def geom(cpd):
    bgd = int(cpd / 0.040)
    dx = 1000.0 / bgd
    ps = dx / 2.0
    return dx, ps, ps ** 3


def load(db, hit):
    con = sqlite3.connect("file:%s?mode=ro" % db, uri=True)
    r = con.execute("SELECT vertices FROM hits WHERE step_number=?", (hit,)).fetchone()
    con.close()
    return None if r is None else np.asarray(json.loads(r[0]), dtype=np.float64).reshape(-1, 3)


def n_hits(db):
    con = sqlite3.connect("file:%s?mode=ro" % db, uri=True)
    n = con.execute("SELECT MAX(step_number) FROM hits").fetchone()[0]
    con.close()
    return n


def tri_vol(V, F):
    t = V[F]
    a, b, c = t[:, 0], t[:, 1], t[:, 2]
    return abs(np.einsum("ij,ij->i", a, np.cross(b, c)).sum()) / 6.0


def recon(P, g, h, alpha, Vp0):
    margin = 4.0 * h + 3.0 * g
    lo, hi = P.min(0) - margin, P.max(0) + margin
    dims = np.maximum(np.ceil((hi - lo) / g).astype(int), 4)
    H, _ = np.histogramdd(P, bins=dims, range=[(lo[i], lo[i] + dims[i] * g) for i in range(3)])
    rho = gaussian_filter(H.astype(np.float32) / g ** 3, sigma=h / g, mode="constant", cval=0.0)
    lvl = alpha * (1.0 / Vp0)
    if rho.max() <= lvl:
        return np.nan
    v, f, _, _ = marching_cubes(rho, level=lvl, spacing=(g, g, g))
    return tri_vol(v, f)


def calib(P, g, h, Vp0, Vt, lo=0.02, hi=0.98, it=24):
    for _ in range(it):
        m = 0.5 * (lo + hi)
        v = recon(P, g, h, m, Vp0)
        if not np.isfinite(v):
            hi = m
            continue
        if v > Vt:
            lo = m
        else:
            hi = m
    return 0.5 * (lo + hi)


def union_balls(P, r, nsamp=4_000_000):
    """Monte-Carlo volume of the union of balls radius r centred on the particles."""
    lo, hi = P.min(0) - 1.5 * r, P.max(0) + 1.5 * r
    box = np.prod(hi - lo)
    tree = cKDTree(P)
    hits = 0
    done = 0
    chunk = 500_000
    while done < nsamp:
        m = min(chunk, nsamp - done)
        pts = RNG.uniform(lo, hi, size=(m, 3))
        d, _ = tree.query(pts, k=1, distance_upper_bound=r)
        hits += int(np.isfinite(d).sum())
        done += m
    p = hits / done
    return box * p, box * np.sqrt(max(p * (1 - p), 1e-12) / done)


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "velo_dlv_c10_r1"
    cpd = float(sys.argv[2]) if len(sys.argv) > 2 else 10.0
    meanJ1 = float(sys.argv[3]) if len(sys.argv) > 3 else 0.9997071322274946
    meanJH = float(sys.argv[4]) if len(sys.argv) > 4 else 0.9702
    db = os.path.join(OUT, tag + ".db")
    dx, ps, Vp0 = geom(cpd)
    H = n_hits(db)
    P1, PH = load(db, 1), load(db, H)
    N = len(P1)
    Vt1 = N * Vp0 * meanJ1
    g = 0.5

    print("=" * 80)
    print("%s  cpd=%g  psize=%.4f mm  Vp0=%.4f  N=%d  hits=%d" % (tag, cpd, ps, Vp0, N, H))
    print("book: hit1=%.1f  hit%d=%.1f  (%+.2f%%)"
          % (Vt1, H, N * Vp0 * meanJH, (meanJH / meanJ1 - 1) * 100))
    print("=" * 80)

    print("\n--- KERNEL SWEEP (grid fixed at g=%.2f mm, alpha recalibrated at hit 1) ---" % g)
    print("%10s %8s %8s %12s %12s %10s" % ("h/psize", "h[mm]", "alpha", "V(hit1)", "V(hit%d)" % H, "drop%"))
    out = []
    for k in [0.7, 0.85, 1.0, 1.2, 1.5, 2.0]:
        h = k * ps
        a = calib(P1, g, h, Vp0, Vt1)
        v1, vH = recon(P1, g, h, a, Vp0), recon(PH, g, h, a, Vp0)
        d = (vH / v1 - 1) * 100
        out.append(dict(k=k, h=h, alpha=float(a), V1=float(v1), VH=float(vH), drop=float(d)))
        print("%10.2f %8.3f %8.4f %12.1f %12.1f %+10.2f" % (k, h, a, v1, vH, d))
        sys.stdout.flush()

    print("\n--- UNION-OF-BALLS packing efficiency (r = psize/2 = %.4f mm, no free params) ---" % (ps / 2))
    v_ball = (4.0 / 3.0) * np.pi * (ps / 2) ** 3
    print("%5s %14s %10s %10s %10s" % ("hit", "V_union", "+-", "eta", "d_eta%"))
    eta0 = None
    rows = []
    for hit in list(range(1, H + 1)):
        P = load(db, hit)
        if P is None:
            continue
        vu, se = union_balls(P, ps / 2)
        eta = vu / (len(P) * v_ball)
        if eta0 is None:
            eta0 = eta
        rows.append(dict(hit=hit, V_union=float(vu), se=float(se), eta=float(eta)))
        print("%5d %14.1f %10.1f %10.4f %+10.2f" % (hit, vu, se, eta, (eta / eta0 - 1) * 100))
        sys.stdout.flush()

    with open(os.path.join(OUT, "review", "kernel_%s.json" % tag), "w") as fh:
        json.dump(dict(tag=tag, cpd=cpd, psize=ps, Vp0=Vp0, N=N, g=g, Vt1=float(Vt1),
                       meanJ1=meanJ1, meanJH=meanJH, kernel_sweep=out, packing=rows), fh, indent=1)
    print("\nwrote review/kernel_%s.json" % tag)


if __name__ == "__main__":
    main()
