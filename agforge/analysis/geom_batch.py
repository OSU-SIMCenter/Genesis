"""Score the batched sweep's geometry against the real Agility Forge scans.

Completes the third axis of the contact-method comparison. The batched driver writes particle
clouds as .npz rather than into a run DB, so this feeds geom_metrics.py's functions directly
instead of going through its DB reader -- deliberately by IMPORT, so the metric definition,
the union-of-cubes occupancy and the psize derivation stay identical to the numbers the banked
17-hit comparison produced. A lookalike reimplementation here would silently make the two
incomparable.

REGISTRATION is reported at both levels, because the difference between them IS the pose error:
  as-is     absolute error, pose mismatch included
  centroid  translation removed -- shape error only
That separation already mattered once: it showed the teleport arm has real pose drift while
grid-alone does not, which a single registration choice would have hidden.
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from geom_metrics import real_mesh, psize, summarize  # noqa: E402


def load_arm(batch_dir, tag, hit=None):
    """Prefer the per-hit cloud (so arms that died early still contribute at a fixed index)."""
    hp = os.path.join(batch_dir, "%s_hits.npz" % tag)
    if hit is not None and os.path.exists(hp):
        with np.load(hp) as d:
            key = "hit_%02d" % hit
            if key in d:
                return np.asarray(d[key], dtype=np.float64), "hit_%02d" % hit
        return None, None
    vp = os.path.join(batch_dir, "%s_verts.npz" % tag)
    if os.path.exists(vp):
        with np.load(vp) as d:
            return np.asarray(d["verts"], dtype=np.float64), "final"
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", default="batch", help="outputs/<batch> directory")
    ap.add_argument("--hit", type=int, required=True,
                    help="real-scan hit index to score against (1-based)")
    ap.add_argument("--sim-hit", type=int, default=None,
                    help="sim hit index to load; default = whatever final state was saved")
    ap.add_argument("--vox", nargs="+", type=float, default=[2.0, 1.0])
    args = ap.parse_args()

    batch_dir = os.path.expanduser(
        "~/GitHub/Genesis/forge_common/main/outputs/%s" % args.batch)
    if not os.path.isdir(batch_dir):
        print("no such batch dir: %s" % batch_dir)
        return 1

    tags = sorted({
        f.replace("_verts.npz", "").replace("_hits.npz", "")
        for f in os.listdir(batch_dir)
        if f.endswith("_verts.npz") or f.endswith("_hits.npz")
    })
    tags = [t for t in tags if not t.startswith("_ref")]
    if not tags:
        print("no arm clouds in %s" % batch_dir)
        return 1

    mesh = real_mesh(args.hit, "after")
    print("real scan: hit %d (after), %d verts, %d tris, volume %.0f mm^3"
          % (args.hit, len(mesh.vertices), len(mesh.faces), mesh.volume))
    print("scoring %d arms at vox %s\n" % (len(tags), args.vox))

    iou_key = "iou_%.2f" % args.vox[0]
    hdr = "%-20s %-8s %8s %8s %9s %9s %9s %9s"
    print(hdr % ("arm", "src", "IoU@%.1f" % args.vox[0], "IoU cen",
                 "dev_mean", "dev_p95", "dev_max", "pose"))
    print("-" * 96)

    rows = []
    for tag in tags:
        P, src = load_arm(batch_dir, tag, args.sim_hit)
        if P is None or not len(P):
            print("%-20s  (no cloud at requested hit)" % tag)
            continue
        r = psize(len(P)) / 2.0

        asis = summarize(mesh, P, r, args.vox)
        # Centroid registration: translation only. Rotation is deliberately not fitted -- the
        # billet is near-cylindrical, so ICP is weakly constrained about the long axis and would
        # happily rotate real error away.
        Pc = P - P.mean(0) + mesh.vertices.mean(0)
        cen = summarize(mesh, Pc, r, args.vox)

        pose = cen[iou_key] - asis[iou_key]
        print(hdr % (tag, src, "%.4f" % asis[iou_key], "%.4f" % cen[iou_key],
                     "%.3f" % asis["dev_mean"], "%.3f" % asis["dev_p95"],
                     "%.3f" % asis["dev_max"], "%+.4f" % pose))
        rows.append((tag, asis, cen, pose))

    print()
    print("IoU        = voxel-occupancy intersection-over-union vs the real scan, as-is")
    print("IoU cen    = same after centroid alignment (shape only)")
    print("pose       = IoU cen - IoU. POSITIVE means alignment helped => that arm has pose")
    print("             drift. NEGATIVE means alignment hurt => its pose was already right.")
    print("dev_*      = |distance| from each real surface vertex to the sim surface, mm.")
    print("             p95/max are where die-impression fidelity lives: an arm can win on IoU")
    print("             and still miss the imprint.")
    print()
    if rows:
        best = max(rows, key=lambda r: r[1][iou_key])
        floor = next((r for r in rows if r[0] == "ctl_none"), None)
        print("best as-is IoU: %s (%.4f)" % (best[0], best[1][iou_key]))
        if floor is not None:
            print("no-contact FLOOR (ctl_none, an essentially undeformed bar): %.4f"
                  % floor[1][iou_key])
            print("  => read every arm against that floor, not against 0. An arm barely above it")
            print("     has captured almost none of the available improvement.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
