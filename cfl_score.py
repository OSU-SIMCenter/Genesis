"""Break the speed/step-count degeneracy.

In the D1a speed sweep, press frames scale as 1/v to within 4%, so "slower press" and
"more accumulated steps" are collinear and indistinguishable. The CFL axis varies the
step count at FIXED press speed, so it separates them.

Prediction if the drift is a per-step accumulation: refining CFL (more steps) should move
geometry the SAME direction slowing the press does. If it moves the opposite way, the two
axes are different mechanisms and a time-proportional leak does not explain the speed drift.

Same scorer as everything else: geom_batch's summarize / real_mesh / psize.
"""
import os
import sys
import json

A = "/home/timothy/GitHub/Genesis/aims-genesis/nsf-demo/agforge/analysis"
sys.path.insert(0, A)
import numpy as np
from geom_metrics import real_mesh, psize, summarize
from geom_batch import load_arm

OUT = "/home/timothy/GitHub/Genesis/forge_common/main/outputs"
BATCHES = [("batch_cfl090", 0.90), ("batch_cfl045", 0.45), ("batch_cfl0225", 0.225)]
IOU = "iou_2.00"
C1 = 0.001

print("=== provenance: run_meta + stop reasons per batch ===")
for b, cfl in BATCHES:
    d = os.path.join(OUT, b)
    if not os.path.isdir(d):
        print("  %-16s MISSING" % b)
        continue
    meta = {}
    mp = os.path.join(d, "run_meta.json")
    if os.path.exists(mp):
        meta = json.load(open(mp))
    stops = {}
    for f in os.listdir(d):
        if f.endswith(".diag.jsonl"):
            for line in open(os.path.join(d, f), encoding="utf-8", errors="replace"):
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                stops[r.get("stop_reason")] = stops.get(r.get("stop_reason"), 0) + 1
    arms = sorted({f.replace("_verts.npz", "").replace("_hits.npz", "")
                   for f in os.listdir(d)
                   if f.endswith(("_verts.npz", "_hits.npz"))})
    arms = [a for a in arms if not a.startswith("_ref")]
    print("  %-16s cfl=%-6s arms=%-2d cpd=%s psize=%s  stops=%s"
          % (b, cfl, len(arms), meta.get("cells_per_diameter"), meta.get("psize_mm"), stops))

print()
rows = {}
for hit in (1, 5, 10, 17):
    mesh = real_mesh(hit, "after")
    for b, cfl in BATCHES:
        d = os.path.join(OUT, b)
        if not os.path.isdir(d):
            continue
        arms = sorted({f.replace("_verts.npz", "").replace("_hits.npz", "")
                       for f in os.listdir(d)
                       if f.endswith(("_verts.npz", "_hits.npz"))})
        for tag in [a for a in arms if not a.startswith("_ref")]:
            P, src = load_arm(d, tag, hit)
            if P is None or not len(P):
                continue
            r = psize(len(P)) / 2.0
            rows[(hit, cfl, tag)] = float(summarize(mesh, P, r, [2.0])[IOU])

tags = sorted({t for (_, _, t) in rows})
print("=== IoU@2.0 vs CFL at FIXED press speed ===")
print("%-20s %4s %9s %9s %9s | %9s %7s %s"
      % ("arm", "hit", "cfl0.90", "cfl0.45", "cfl0.225", "spread", "xC1", "direction"))
print("-" * 100)
for tag in tags:
    for hit in (1, 5, 10, 17):
        v = [rows.get((hit, c, tag)) for _, c in BATCHES]
        if any(x is None for x in v):
            continue
        sp = max(v) - min(v)
        # refining CFL = more steps. Does IoU go up or down as steps increase?
        direction = "UP with steps" if v[-1] > v[0] else "DOWN with steps"
        print("%-20s %4d %9.4f %9.4f %9.4f | %9.4f %6.0fx %s"
              % (tag, hit, v[0], v[1], v[2], sp, sp / C1, direction))
print()
print("Slowing the press moved IoU DOWN (0.5768 -> 0.5090 for g1_grid_prod at hit 17).")
print("If refining CFL moves it UP, the two axes are NOT the same mechanism and a")
print("per-step accumulation cannot explain the speed drift.")
