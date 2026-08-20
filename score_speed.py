"""D1a re-scoring: IoU@2.0 vs press speed, on sweeps already on disk.

Mirrors agforge/analysis/geom_batch.py exactly -- same real_mesh(), same summarize(),
same r = psize(len(P))/2, same centroid-registration second pass -- so the numbers are
directly comparable to every other geometry number in this project.
"""
import os, sys, json, time, argparse

A = "/home/timothy/GitHub/Genesis/aims-genesis/nsf-demo/agforge/analysis"
sys.path.insert(0, A)
import numpy as np
from geom_metrics import real_mesh, psize, summarize
from geom_batch import load_arm

OUT = "/home/timothy/GitHub/Genesis/forge_common/main/outputs"
VOX = [2.0]
IOU = "iou_2.00"

# (batch, press_speed m/s, card)  -- card from the 6ee71236 (08-18) cutover
SETS = {
    "old": [("batch_speed_25p0", 25.0), ("batch_speed_12p5", 12.5),
            ("batch_speed_6p25", 6.25), ("batch_speed_3p125", 3.125)],
    "old_rep": [("batch_speed2_6p25", 6.25)],
    "new": [("batch_spdmx_25p0", 25.0), ("batch_spdmx_12p5", 12.5)],
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="+", default=["old", "old_rep", "new"])
    ap.add_argument("--hits", nargs="+", type=int, default=[1, 5, 10, 17])
    ap.add_argument("--out", default="/tmp/d1a_scores.jsonl")
    args = ap.parse_args()

    jobs = [(s, b, v) for s in args.sets for (b, v) in SETS[s]]
    rows, t0 = [], time.time()
    with open(args.out, "w") as fh:
        for hit in args.hits:
            mesh = real_mesh(hit, "after")
            for setname, batch, speed in jobs:
                bd = os.path.join(OUT, batch)
                if not os.path.isdir(bd):
                    print("  MISSING %s" % batch); continue
                tags = sorted({f.replace("_verts.npz", "").replace("_hits.npz", "")
                               for f in os.listdir(bd)
                               if f.endswith("_verts.npz") or f.endswith("_hits.npz")})
                tags = [t for t in tags if not t.startswith("_ref")]
                for tag in tags:
                    P, src = load_arm(bd, tag, hit)
                    if P is None or not len(P):
                        print("  %-14s %-22s %-18s no cloud @hit %d" % (setname, batch, tag, hit))
                        continue
                    r = psize(len(P)) / 2.0
                    asis = summarize(mesh, P, r, VOX)
                    Pc = P - P.mean(0) + mesh.vertices.mean(0)
                    cen = summarize(mesh, Pc, r, VOX)
                    row = dict(set=setname, batch=batch, speed=speed, hit=hit, arm=tag,
                               src=src, n=int(len(P)),
                               iou=float(asis[IOU]), iou_cen=float(cen[IOU]),
                               dev_mean=float(asis["dev_mean"]),
                               dev_p95=float(asis["dev_p95"]),
                               dev_max=float(asis["dev_max"]))
                    rows.append(row); fh.write(json.dumps(row) + "\n"); fh.flush()
                    print("  h%02d %-14s v=%-7.3f %-20s IoU %.4f  cen %.4f  dev %.3f"
                          % (hit, setname, speed, tag, row["iou"], row["iou_cen"], row["dev_mean"]))
    print("\n%d rows in %.1fs -> %s" % (len(rows), time.time() - t0, args.out))

main()
