"""Is the -12/-15% geometric volume drop real, or an artifact of surface reconstruction?

The run DBs store RAW PARTICLE POSITIONS (surface_mesh=False), so the surface can be
rebuilt offline at any resolution -- no GPU, no re-simulation.

Construction (deliberately grid-independent in its physics):
  1. splat particles into a grid at spacing g  -> number density rho [particles/mm^3]
  2. smooth with a Gaussian of PHYSICAL width h [mm]  (h fixed while g varies)
  3. isosurface at rho = alpha * rho_bulk, where rho_bulk = 1/Vp0

Because h and alpha are physical, sweeping g alone isolates the GRID (marching-cubes)
bias; as g -> 0 the volume converges to the exact level-set volume.

The remaining KERNEL bias (alpha/h decide where the surface sits relative to the outermost
particles) is calibrated ONCE at hit 1, where the bar is essentially undeformed and the
true volume is known: V_true = N * Vp0 * meanJ(1).

The same alpha is then applied at every later hit. If the reconstructed volume still
collapses, the piling is real. We also record surface AREA at every hit, to test the
surface-offset model V_recon = V_true + delta*A -- the mechanism that would make the
bias drift as the bar flattens and its area grows.
"""
import json
import os
import sqlite3
import sys

import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.measure import marching_cubes

OUT = os.path.expanduser("~/GitHub/Genesis/forge_common/main/outputs")

# cells-per-diameter -> (dx_mm, psize_mm, Vp0_mm3).  base_grid_density = int(cpd/0.040) [cells/m]
def geom(cpd):
    bgd = int(cpd / 0.040)
    dx = 1000.0 / bgd          # mm
    ps = dx / 2.0              # particle_size = dx / AGF_PPC_DIVISOR(=2)
    return dx, ps, ps ** 3


def load_particles(db, hit):
    con = sqlite3.connect("file:%s?mode=ro" % db, uri=True)
    row = con.execute("SELECT vertices FROM hits WHERE step_number=?", (hit,)).fetchone()
    con.close()
    if row is None:
        return None
    return np.asarray(json.loads(row[0]), dtype=np.float64).reshape(-1, 3)


def n_hits(db):
    con = sqlite3.connect("file:%s?mode=ro" % db, uri=True)
    n = con.execute("SELECT MAX(step_number) FROM hits").fetchone()[0]
    con.close()
    return n


def tri_volume_area(V, F):
    """Enclosed volume (divergence theorem) and total surface area of a closed mesh."""
    t = V[F]
    a, b, c = t[:, 0], t[:, 1], t[:, 2]
    cr = np.cross(b - a, c - a)
    area = 0.5 * np.linalg.norm(cr, axis=1).sum()
    vol = abs(np.einsum("ij,ij->i", a, np.cross(b, c)).sum()) / 6.0
    return vol, area


def reconstruct(P, g, h, alpha, Vp0):
    """Volume+area of the isosurface rho = alpha*rho_bulk, on a grid of spacing g."""
    margin = 4.0 * h + 3.0 * g
    lo = P.min(axis=0) - margin
    hi = P.max(axis=0) + margin
    dims = np.maximum(np.ceil((hi - lo) / g).astype(int), 4)
    # number density [particles / mm^3]
    H, _ = np.histogramdd(P, bins=dims, range=[(lo[i], lo[i] + dims[i] * g) for i in range(3)])
    rho = H.astype(np.float32) / (g ** 3)
    rho = gaussian_filter(rho, sigma=h / g, mode="constant", cval=0.0)
    level = alpha * (1.0 / Vp0)
    if rho.max() <= level:
        return np.nan, np.nan, dims
    verts, faces, _, _ = marching_cubes(rho, level=level, spacing=(g, g, g))
    vol, area = tri_volume_area(verts, faces)
    return vol, area, dims


def calibrate_alpha(P, g, h, Vp0, V_true, lo=0.05, hi=0.95, iters=26):
    """Bisect alpha so the reconstructed volume at this hit equals the known true volume.
    Volume decreases as alpha increases (higher threshold -> tighter surface)."""
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        v, _, _ = reconstruct(P, g, h, mid, Vp0)
        if not np.isfinite(v):
            hi = mid
            continue
        if v > V_true:
            lo = mid          # need a tighter surface
        else:
            hi = mid
    return 0.5 * (lo + hi)


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "velo_dlv_c10_r1"
    cpd = float(sys.argv[2]) if len(sys.argv) > 2 else 10.0
    meanJ1 = float(sys.argv[3]) if len(sys.argv) > 3 else 0.9991116422662364
    db = os.path.join(OUT, tag + ".db")
    dx, ps, Vp0 = geom(cpd)
    N = len(load_particles(db, 1))
    V_true1 = N * Vp0 * meanJ1
    H = n_hits(db)
    h_kernel = 1.2 * ps

    print("=" * 78)
    print("run=%s  cpd=%g  dx=%.4f mm  psize=%.4f mm  Vp0=%.4f mm3" % (tag, cpd, dx, ps, Vp0))
    print("N=%d particles   V_book(hit1)=N*Vp0*meanJ1=%.1f mm3   kernel h=%.3f mm" % (N, V_true1, h_kernel))
    print("=" * 78)

    P1 = load_particles(db, 1)
    PH = load_particles(db, H)

    # ---- Phase 1: grid convergence, alpha recalibrated at each g so hit1 is exact -----
    print("\n--- PHASE 1: marching-cubes grid convergence (alpha recalibrated per g) ---")
    print("%8s %8s %12s %12s %10s %12s" % ("g[mm]", "alpha", "V(hit1)", "V(hit%d)" % H, "drop%", "A(hit%d)" % H))
    grids = [2.0, 1.5, 1.0, 0.75, 0.5]
    results = []
    for g in grids:
        alpha = calibrate_alpha(P1, g, h_kernel, Vp0, V_true1)
        v1, a1, _ = reconstruct(P1, g, h_kernel, alpha, Vp0)
        vH, aH, dims = reconstruct(PH, g, h_kernel, alpha, Vp0)
        drop = (vH / v1 - 1.0) * 100.0
        results.append((g, alpha, v1, vH, drop, aH, a1))
        print("%8.2f %8.4f %12.1f %12.1f %+10.2f %12.1f" % (g, alpha, v1, vH, drop, aH))
        sys.stdout.flush()

    # Richardson-style extrapolation of the drop to g -> 0 using the two finest grids
    (g2, _, _, _, d2, _, _) = results[-1]
    (g1, _, _, _, d1, _, _) = results[-2]
    drop0 = d2 + (d2 - d1) * (g2 ** 2) / (g1 ** 2 - g2 ** 2)
    print("\nextrapolated drop at g->0 (2nd-order Richardson): %+.2f%%" % drop0)

    # ---- Phase 2: all hits at the finest grid ---------------------------------------
    gf = grids[-1]
    alpha = calibrate_alpha(P1, gf, h_kernel, Vp0, V_true1)
    print("\n--- PHASE 2: every hit at g=%.2f mm, alpha=%.4f (calibrated on hit 1) ---" % (gf, alpha))
    print("%5s %12s %12s %10s %12s %10s" % ("hit", "V_geo", "A_geo", "dV%", "len_mm", "dA%"))
    rows = []
    v_ref = a_ref = None
    for hit in range(1, H + 1):
        P = load_particles(db, hit)
        if P is None:
            continue
        v, a, _ = reconstruct(P, gf, h_kernel, alpha, Vp0)
        if v_ref is None:
            v_ref, a_ref = v, a
        length = np.ptp(P[:, 0])
        rows.append(dict(hit=hit, V=v, A=a, length=float(length)))
        print("%5d %12.1f %12.1f %+10.2f %12.2f %+10.2f"
              % (hit, v, a, (v / v_ref - 1) * 100, length, (a / a_ref - 1) * 100))
        sys.stdout.flush()

    with open(os.path.join(OUT, "review", "recon_%s.json" % tag), "w") as fh:
        json.dump(dict(tag=tag, cpd=cpd, dx=dx, psize=ps, Vp0=Vp0, N=N, h_kernel=h_kernel,
                       V_true1=V_true1, alpha=alpha, g_fine=gf,
                       grid_sweep=[dict(g=r[0], alpha=r[1], V1=r[2], VH=r[3], drop=r[4],
                                        AH=r[5], A1=r[6]) for r in results],
                       drop_extrapolated=drop0, rows=rows), fh, indent=1)
    print("\nwrote review/recon_%s.json" % tag)


if __name__ == "__main__":
    main()
