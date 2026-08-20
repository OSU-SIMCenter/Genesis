"""Score the controller-suppressed D1a re-run and compare it to the archived sweep.

Archived batch_speed_* ran with an active stall whose observed exposure grows 1.3 -> 13.7% of
frames as the press slows. batch_d1aclean_* is the same 8x speed step (25.0 -> 3.125 m/s per
jaw), same billet IC, same cpd/ppc/approach-CFL, with the controller suppressed:
gain 1.5e-5, imbalance threshold 1e12, max_force 1e9.

Same scorer as every other geometry number here: geom_batch's summarize / real_mesh / psize.

CAVEAT the output repeats: the clean run is on the sourced-316L card (nsf-demo 6ee71236);
the archived run predates it. A change in spread is controller OR card, not attributable to
one without a matched shipped-gain run on the same card.
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
C1 = 0.001
IOU = "iou_2.00"
ARMS = ["g1_grid_prod", "p5_penalty"]
HITS = [1, 5, 10, 17]

CLEAN = {25.0: "batch_d1aclean_25p0", 3.125: "batch_d1aclean_3p125"}
ARCH = {25.0: "batch_speed_25p0", 3.125: "batch_speed_3p125"}


def score(batch, tag, hit, mesh):
    d = os.path.join(OUT, batch)
    if not os.path.isdir(d):
        return None
    P, _ = load_arm(d, tag, hit)
    if P is None or not len(P):
        return None
    return float(summarize(mesh, P, psize(len(P)) / 2.0, [2.0])[IOU])


def stops(batch):
    d = os.path.join(OUT, batch)
    out = {}
    if not os.path.isdir(d):
        return out
    for f in os.listdir(d):
        if f.endswith(".diag.jsonl"):
            for line in open(os.path.join(d, f), encoding="utf-8", errors="replace"):
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                out[r.get("stop_reason")] = out.get(r.get("stop_reason"), 0) + 1
    return out


print("=== provenance ===")
for v, b in CLEAN.items():
    p = os.path.join(OUT, b, "controller_meta.json")
    m = json.load(open(p)) if os.path.exists(p) else {}
    print("  %-24s gain=%-8s thresh=%-8s max_force=%-8s stops=%s"
          % (b, m.get("force_balance_gain"), m.get("force_imbalance_threshold"),
             m.get("max_force"), stops(b)))
for v, b in ARCH.items():
    print("  %-24s (archived, shipped gain)                              stops=%s" % (b, stops(b)))

res = {}
for hit in HITS:
    mesh = real_mesh(hit, "after")
    for label, src in (("clean", CLEAN), ("arch", ARCH)):
        for v, b in src.items():
            for tag in ARMS:
                s = score(b, tag, hit, mesh)
                if s is not None:
                    res[(label, hit, v, tag)] = s

print()
print("=== CRITERION 1: IoU spread across the 8x speed step (25.0 -> 3.125 m/s per jaw) ===")
print("threshold < 0.001\n")
print("%-16s %4s | %9s %9s %9s %6s | %9s %9s %9s %6s | %s"
      % ("arm", "hit", "clean@25", "clean@3.1", "SPREAD", "xC1",
         "arch@25", "arch@3.1", "SPREAD", "xC1", "verdict"))
print("-" * 128)
for tag in ARMS:
    for hit in HITS:
        c1 = res.get(("clean", hit, 25.0, tag))
        c2 = res.get(("clean", hit, 3.125, tag))
        a1 = res.get(("arch", hit, 25.0, tag))
        a2 = res.get(("arch", hit, 3.125, tag))
        if c1 is None or c2 is None:
            print("%-16s %4d | (clean batch missing)" % (tag, hit))
            continue
        cs = abs(c1 - c2)
        astr = "%9.4f %9.4f %9.4f %5.0fx" % (a1, a2, abs(a1 - a2), abs(a1 - a2) / C1) \
            if a1 is not None and a2 is not None else "%39s" % "--"
        verdict = "PASSES criterion 1" if cs < C1 else "fails"
        if a1 is not None and a2 is not None and abs(a1 - a2) > 0:
            verdict += "  (%.2fx archived)" % (cs / abs(a1 - a2))
        print("%-16s %4d | %9.4f %9.4f %9.4f %5.0fx | %s | %s"
              % (tag, hit, c1, c2, cs, cs / C1, astr, verdict))

print()
print("NOTE: clean = sourced-316L card (6ee71236); archived = older card. A change in spread")
print("      is controller OR card. The self-contained question -- does spread fall under")
print("      0.001 with the controller out -- is unaffected by that.")
