"""General-purpose geometry accuracy of the sim billet against the real Agility Forge scans.

GROUND TRUTH. `2026-06-29.pt` carries, per hit, a full triangle mesh of the billet before
(`vertices_k`/`triangles_k`) and after (`_k1`) the hit -- ~100-137k vertices, ~200-271k
triangles, 99.99% edge-manifold, in millimetres. Only the raw point clouds were exposed
before (`load_real_point_cloud_frames`); the meshes are strictly better ground truth
because a closed surface supports exact inside/outside tests.

REPRESENTING THE SIM. The sim gives 9,266 particle CENTRES, not a surface. Measured against
the real mesh at hit 1, those centres sit ~1.0-1.3 mm inside the real surface -- which is
psize/2 (the particle radius). So the material is the UNION OF BALLS of radius psize/2
about the centres, and that union should land on the real surface. This is the same
representation the volume metric uses, and it is deliberately NOT a surface reconstruction:
meshing the particles reintroduces the smoothing width that produced -8% to -4% volume on a
single cloud and invalidated the earlier numbers.

METRICS, tiered because they fail differently:
  IoU / Dice on voxel occupancy   -- primary. Symmetric, bounded, handles the cloud-vs-mesh
                                     type mismatch natively. Its one parameter (voxel size)
                                     is SWEPT so convergence is visible rather than assumed.
  surface deviation               -- for each real mesh vertex, distance to the sim surface
                                     (nearest particle centre minus r). mean / p95 / max.
                                     p95 and max are where die-impression fidelity lives:
                                     a method can win on IoU and still miss the imprint.
  volume + extents                -- sanity scalars. Real volume is exact (closed mesh).

REGISTRATION is load-bearing, so it is reported at two levels rather than chosen:
  as-is       -- the adapter's canonical mm frame vs the dataset frame, no fitting.
                 ABSOLUTE error, including any pose mismatch.
  centroid    -- translation removed. SHAPE error only.
The DIFFERENCE between them is the pose error, which becomes its own reported number
instead of a hidden confound. (Rotation is deliberately not fitted: the billet is
near-cylindrical, so ICP is weakly constrained about the long axis and would happily
rotate away real error.)
"""
import argparse
import json
import os
import pathlib
import sqlite3

import gc
import resource

import numpy as np
import trimesh
from scipy.spatial import cKDTree

# --- HARD MEMORY GUARD ---------------------------------------------------------------
# The WSL VM is capped near 7 GB and the host runs close to its commit limit, so an
# unbounded allocation here kills the VM (taking any other agent's work with it) instead
# of raising. Cap the address space so python fails loudly and locally instead.
#   AGF_GEOM_MEM_GB=3 <cmd>
_MEM_GB = float(os.environ.get("AGF_GEOM_MEM_GB", "3.0"))
try:
    resource.setrlimit(resource.RLIMIT_AS, (int(_MEM_GB * 2 ** 30), int(_MEM_GB * 2 ** 30)))
except (ValueError, OSError):
    pass


def _peak_rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

OUT = pathlib.Path.home() / "GitHub/Genesis/forge_common/main/outputs"
# Per-hit real meshes, extracted once from the 319 MB .pt by extract_real_meshes.py so this
# script never has to import torch. Regenerable -- and outputs/ is gitignored, so treat it
# as a cache, not as data.
REAL_MESH_DIR = OUT / "real_meshes"
# The SIM's nominal stock (forge_common/real_scale.py), used only to derive the particle
# spacing: psize = (V_nominal / N)^(1/3). At res 10 this gives 2.0000 mm exactly.
# NB these are NOT the real billet's dimensions -- see the module docstring: the real scan
# implies a 38.30 mm diameter (1.5" stock) against the sim's 40 mm.
SIM_STOCK_R_MM, SIM_STOCK_L_MM = 20.0, 59.0

def real_mesh(hit, side="after"):
    """Closed surface of the billet BEFORE or AFTER `hit` (1-based).

    Reads the extracted per-hit .npz (~4 MB) rather than the 319 MB .pt, so this script
    never imports torch. Run extract_real_meshes.py once if the cache is absent.
    """
    p = REAL_MESH_DIR / ("hit_%02d.npz" % hit)
    if not p.exists():
        raise SystemExit(
            "missing %s -- run agforge/analysis/extract_real_meshes.py once first" % p)
    with np.load(p) as d:
        V = np.array(d["V_" + side], dtype=np.float64)
        T = np.array(d["T_" + side], dtype=np.int64)
    gc.collect()
    return trimesh.Trimesh(vertices=V, faces=T, process=False)


def sim_cloud(tag, hit):
    p = OUT / ((tag if tag.endswith(".db") else tag + ".db"))
    if not p.exists():
        return None
    con = sqlite3.connect("file:%s?mode=ro" % p, uri=True)
    try:
        r = con.execute("SELECT vertices FROM hits WHERE step_number=?", (hit,)).fetchone()
    finally:
        con.close()
    if r is None or r[0] is None:
        return None
    return np.asarray(json.loads(r[0]), dtype=np.float64).reshape(-1, 3)


def psize(n):
    """Particle lattice spacing in mm.

    psize is a property of the GRID -- the sim sets particle_size = dx / AGF_PPC_DIVISOR --
    NOT of how much material got seeded. Backing it out of the nominal cylinder volume only
    ever agreed with that because a Cylinder morph fills exactly the nominal volume.

    Seed the billet from a mesh (AGF_BILLET_MESH) and N drops ~10% while the nominal volume
    does not, so this derivation reads high: MEASURED 2.0696 mm against a true 2.0000 mm,
    +3.48%. Every metric built on psize -- packing eta, the IoU cube, surface deviation --
    then shifts silently. Set AGF_PSIZE_MM when scoring such a run; batch_arms writes the
    true value (read straight off the entity) into the batch's run_meta.json.
    """
    override = os.environ.get("AGF_PSIZE_MM", "").strip()
    if override:
        return float(override)
    return float((np.pi * SIM_STOCK_R_MM ** 2 * SIM_STOCK_L_MM / n) ** (1.0 / 3.0))


def occupancy(mesh, P, r, vox):
    """Voxel occupancy of real (mesh interior) and sim (union of balls) on a shared grid."""
    lo = np.minimum(mesh.vertices.min(0), P.min(0) - r) - vox
    hi = np.maximum(mesh.vertices.max(0), P.max(0) + r) + vox
    ax = [np.arange(lo[d] + vox / 2, hi[d], vox) for d in range(3)]
    G = np.stack(np.meshgrid(*ax, indexing="ij"), -1).reshape(-1, 3)
    # NOT mesh.contains(): trimesh falls back off embree here and builds an rtree R-tree
    # over all ~252k triangles per query batch, which raises std::bad_alloc (and previously
    # killed the whole WSL VM). Voxelize instead -- cost scales with surface area / pitch^2,
    # and occupancy is what is actually being measured, so this computes it directly rather
    # than casting rays to infer it.
    vg = mesh.voxelized(pitch=vox).fill()
    real = np.asarray(vg.is_filled(G), dtype=bool)
    # L-INFINITY, not Euclidean. A material point owns psize^3 of volume, so for an
    # occupancy comparison each particle is a CUBE of side psize, which tiles space. Balls
    # of radius psize/2 fill only pi/6 = 52.4% and understate the sim's volume by ~40%.
    # (The packing metric in score_arms.py deliberately keeps the ball representation --
    # different question: there, touching balls are the baseline interpenetration falls from.)
    sim = cKDTree(P).query(G, k=1, p=np.inf, distance_upper_bound=r)[0] < np.inf
    return real, sim, float(vox ** 3)


def surf_dev(mesh, P, r):
    """Distance from each real surface vertex to the sim surface (nearest centre - r).
    Positive => sim surface is inside the real one there (sim under-fills)."""
    d, _ = cKDTree(P).query(mesh.vertices, k=1)
    return d - r


def summarize(mesh, P, r, voxes):
    out = {}
    for v in voxes:
        real, sim, vv = occupancy(mesh, P, r, v)
        inter = np.count_nonzero(real & sim)
        union = np.count_nonzero(real | sim)
        out["iou_%.2f" % v] = inter / union if union else float("nan")
        out["dice_%.2f" % v] = 2 * inter / (real.sum() + sim.sum()) if (real.sum() + sim.sum()) else float("nan")
        out["vol_real_%.2f" % v] = real.sum() * vv
        out["vol_sim_%.2f" % v] = sim.sum() * vv
    # Surface deviation keeps the Euclidean radius: the question there is distance from a
    # real surface point to the nearest sim material, which is a round-particle question.
    dev = surf_dev(mesh, P, r)
    out["dev_mean"] = float(np.abs(dev).mean())
    out["dev_p95"] = float(np.percentile(np.abs(dev), 95))
    out["dev_max"] = float(np.abs(dev).max())
    out["dev_signed_mean"] = float(dev.mean())
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hit", type=int, required=True)
    ap.add_argument("--arms", nargs="+", required=True, help="name=tag1,tag2,...")
    ap.add_argument("--vox", nargs="+", type=float, default=[2.0, 1.5, 1.0])
    a = ap.parse_args()

    mesh = real_mesh(a.hit)
    print("=" * 100)
    print("GEOMETRY vs REAL SCAN   hit %d   real mesh: %d verts / %d tris, watertight=%s"
          % (a.hit, len(mesh.vertices), len(mesh.faces), mesh.is_watertight))
    _vr = abs(mesh.volume)
    _vs = np.pi * SIM_STOCK_R_MM ** 2 * SIM_STOCK_L_MM
    print("real volume (closed mesh) = %.0f mm^3   extents = %s"
          % (_vr, np.array2string(mesh.extents, precision=1)))
    print("sim NOMINAL stock volume  = %.0f mm^3  (r=%.1f, L=%.1f)  =>  sim stock is %+.1f%% "
          "vs this scan" % (_vs, SIM_STOCK_R_MM, SIM_STOCK_L_MM, 100.0 * (_vs / _vr - 1.0)))
    print("=" * 100)

    hdr = "%-20s %-9s %8s %8s %8s %9s %9s %9s %9s"
    print(hdr % ("arm", "regist.", "IoU@2.0", "IoU@1.5", "IoU@1.0", "dev mean", "dev p95",
                 "dev max", "sim vol"))
    print("-" * 100)

    for spec in a.arms:
        name, _, tags = spec.partition("=")
        for reg in ("as-is", "centroid"):
            rows = []
            for t in [x for x in tags.split(",") if x]:
                P = sim_cloud(t, a.hit)
                if P is None:
                    continue
                r = psize(len(P)) / 2.0
                Q = P.copy()
                if reg == "centroid":
                    Q = Q + (mesh.vertices.mean(0) - Q.mean(0))
                rows.append(summarize(mesh, Q, r, a.vox))
            if not rows:
                print("%-20s %-9s  no data at hit %d" % (name, reg, a.hit))
                continue
            m = {k: float(np.mean([x[k] for x in rows])) for k in rows[0]}
            print(hdr % (
                name if reg == "as-is" else "", reg,
                "%.4f" % m["iou_%.2f" % a.vox[0]],
                "%.4f" % m["iou_%.2f" % a.vox[1]] if len(a.vox) > 1 else "--",
                "%.4f" % m["iou_%.2f" % a.vox[2]] if len(a.vox) > 2 else "--",
                "%.3f" % m["dev_mean"], "%.3f" % m["dev_p95"], "%.3f" % m["dev_max"],
                "%.0f" % m["vol_sim_%.2f" % a.vox[-1]]))

    print()
    print("IoU       = |real AND sim| / |real OR sim| on voxel occupancy. 1.0 = identical.")
    print("            Swept over voxel size so convergence is visible, not assumed.")
    print("dev       = |distance| from each REAL surface vertex to the sim surface, mm.")
    print("            p95/max are where die-impression fidelity shows; a method can win on")
    print("            IoU and still miss the imprint.")
    print("regist.   = 'as-is' is the absolute error in the adapter's canonical frame.")
    print("            'centroid' removes translation, leaving shape error. The gap between")
    print("            the two rows IS the pose error.")
    print()
    print("peak RSS  = %.0f MB (cap %.1f GB via AGF_GEOM_MEM_GB)" % (_peak_rss_mb(), _MEM_GB))


if __name__ == "__main__":
    main()
