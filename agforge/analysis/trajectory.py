"""Per-hit TRAJECTORY of every metric, not just the final state.

WHY: a single end-state number cannot tell "steadily drifting" from "fine until hit N, then
falls apart". Those need completely different fixes, and the second is what a human watching an
animation actually notices. There is also standing suspicion that this sequence behaves well
through roughly hit 12 and degrades after -- and an earlier reviewer evidently thought so too,
since `outputs/review/` contains a `review_hits_01_12` render cut at exactly that point.

Geometry is scored against the REAL scan at the matching hit, so "error vs hit" means error
against the real billet at that stage, not against a fixed target.

PERFORMANCE NOTE: the real mesh's voxelization is arm-independent, so it is computed ONCE per hit
and reused across arms (17 voxelizations rather than 17*n_arms). The occupancy logic is therefore
inlined rather than calling geom_metrics.occupancy -- which recomputes it every call -- so
--selfcheck verifies this copy still agrees with that one before any of these numbers are used.
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from geom_metrics import real_mesh, psize, occupancy  # noqa: E402


def load_hits_npz(d, tag):
    p = os.path.join(d, "%s_hits.npz" % tag)
    if not os.path.exists(p):
        return {}
    out = {}
    with np.load(p) as z:
        for k in z.files:
            out[int(k.split("_")[1])] = np.asarray(z[k], dtype=np.float64)
    return out


def load_diag(d, tag):
    p = os.path.join(d, "%s.diag.jsonl" % tag)
    rows = {}
    if not os.path.exists(p):
        return rows
    with open(p, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("strike") is not None:
                rows[int(r["strike"])] = r
    return rows


def occ_cached(vg, mesh_v, P, r, vox):
    """Occupancy IoU/Dice reusing a prebuilt real voxel grid. Mirrors geom_metrics.occupancy."""
    lo = np.minimum(mesh_v.min(0), P.min(0) - r) - vox
    hi = np.maximum(mesh_v.max(0), P.max(0) + r) + vox
    ax = [np.arange(lo[d] + vox / 2, hi[d], vox) for d in range(3)]
    G = np.stack(np.meshgrid(*ax, indexing="ij"), -1).reshape(-1, 3)
    real = np.asarray(vg.is_filled(G), dtype=bool)
    # L-infinity: a material point owns psize^3, so for OCCUPANCY each particle is a cube that
    # tiles space. Balls of radius psize/2 fill only pi/6 and understate the volume ~40%.
    sim = cKDTree(P).query(G, k=1, p=np.inf, distance_upper_bound=r)[0] < np.inf
    inter = int(np.count_nonzero(real & sim))
    union = int(np.count_nonzero(real | sim))
    return (inter / union) if union else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", default="batch17")
    ap.add_argument("--arms", default="")
    ap.add_argument("--vox", type=float, default=2.0)
    ap.add_argument("--max-hit", type=int, default=17)
    ap.add_argument("--selfcheck", action="store_true",
                    help="verify the cached occupancy matches geom_metrics.occupancy")
    ap.add_argument("--out", default="trajectory.json")
    args = ap.parse_args()

    d = os.path.expanduser("~/GitHub/Genesis/forge_common/main/outputs/%s" % args.batch)
    tags = sorted({f.replace("_hits.npz", "") for f in os.listdir(d) if f.endswith("_hits.npz")})
    tags = [t for t in tags if not t.startswith("_ref")]
    if args.arms:
        want = [a.strip() for a in args.arms.split(",")]
        tags = [t for t in tags if t in want]

    clouds = {t: load_hits_npz(d, t) for t in tags}
    diags = {t: load_diag(d, t) for t in tags}
    print("arms: %s\n" % ", ".join(tags))

    traj = {t: {} for t in tags}
    for hit in range(1, args.max_hit + 1):
        present = [t for t in tags if hit in clouds[t]]
        if not present:
            continue
        mesh = real_mesh(hit, "after")
        vg = mesh.voxelized(pitch=args.vox).fill()      # once per hit, reused across arms
        mv = np.asarray(mesh.vertices, dtype=np.float64)

        for t in present:
            P = clouds[t][hit]
            r = psize(len(P)) / 2.0
            iou = occ_cached(vg, mv, P, r, args.vox)
            # Distance from each real surface vertex to the nearest sim material. Euclidean here
            # (unlike occupancy above): "how far to the nearest particle" is a round-particle
            # question, matching geom_metrics.surf_dev.
            dev = cKDTree(P).query(mv, k=1)[0] - r
            row = {
                "iou": float(iou),
                "dev_mean": float(np.abs(dev).mean()),
                "dev_p95": float(np.percentile(np.abs(dev), 95)),
                "dev_max": float(np.abs(dev).max()),
                "span_x": float(np.ptp(P[:, 0])),
            }
            dr = diags[t].get(hit, {})
            row.update({
                "pen_max_mm": (dr["pen_max"] * 1000.0) if dr.get("pen_max") is not None else None,
                "detF_mean": dr.get("detF_mean"),
                "f_press": dr.get("force_L_press_mean"),
                "n_press": dr.get("n_press_frames"),
            })
            traj[t][hit] = row
        print("hit %2d scored for %d arms" % (hit, len(present)))

    # ---------------------------------------------------------------- self-check
    if args.selfcheck:
        t0 = tags[0]
        h0 = sorted(clouds[t0])[0]
        mesh = real_mesh(h0, "after")
        P = clouds[t0][h0]
        r = psize(len(P)) / 2.0
        real, sim, vv = occupancy(mesh, P, r, args.vox)
        ref = np.count_nonzero(real & sim) / np.count_nonzero(real | sim)
        got = traj[t0][h0]["iou"]
        print("\nselfcheck %s hit %d: cached=%.6f geom_metrics=%.6f  diff=%.2e  %s"
              % (t0, h0, got, ref, abs(got - ref), "OK" if abs(got - ref) < 1e-9 else "MISMATCH"))

    outp = os.path.join(d, args.out)
    with open(outp, "w", encoding="utf-8") as fh:
        json.dump(traj, fh, indent=1)
    print("\nwrote %s" % outp)

    # ---------------------------------------------------------------- tables
    for metric, label, nd in (("iou", "IoU vs real scan (higher better)", 4),
                              ("dev_p95", "surface dev p95, mm (lower better)", 3),
                              ("pen_max_mm", "max penetration, mm", 3)):
        print("\n" + "=" * (14 + 6 * args.max_hit))
        print(label)
        print("=" * (14 + 6 * args.max_hit))
        print("%-20s" % "arm" + "".join("%6d" % h for h in range(1, args.max_hit + 1)))
        for t in tags:
            cells = ""
            for h in range(1, args.max_hit + 1):
                v = traj[t].get(h, {}).get(metric)
                cells += ("%6s" % "--") if v is None else (("%6." + str(nd) + "f") % v)[-6:]
            print("%-20s%s" % (t, cells))
    return 0


if __name__ == "__main__":
    sys.exit(main())
