"""Does the contact-arm RANKING depend on the die-balance gain?

gain_sweep.sh states the go/no-go it was launched for: "if arms reorder with
force_balance_gain, the sweep is measuring the controller, not the contact
method."  The batches ran 2026-08-18 and were never scored.

batch_gain_1p5em4 and batch_gain_1p5em5 are matched on everything the shared
MATERIAL_AND_CONFIG.txt records -- CFL 0.45, press speed 25.0, cpd 10,
psize 2.0, AGF_ENABLE_CPIC=0, AGF_MAX_FORCE disabled, 17 hits, reps 1, and the
SAME (old, 4340-derived) card -- and differ only in AGF_FORCE_BALANCE_GAIN.

Two caveats that belong on every number this prints.

  OLD CARD.  Both batches predate 6ee71236.  sigma_y at eps_p = 0.2 is 106 MPa
  here against 200 MPa on the shipped card, ~1.9x in flow stress.  The gain
  comparison is valid against ITSELF and against nothing on the new card; the
  provenance file says so explicitly.

  n = 1 vs n = 1.  Each cell is a single run.  Replicate scatter on this project
  is a per-arm/per-hit surface reaching 0.0867 IoU at hit 17, so a difference
  smaller than that arm's scatter at that hit is NOT resolvable here, however
  large it looks against criterion 1.  Differences are reported against both.

Same scorer as everything else: geom_batch's load_arm, geom_metrics' summarize.
"""
import os
import sys

A = "/home/timothy/GitHub/Genesis/aims-genesis/nsf-demo/agforge/analysis"
sys.path.insert(0, A)
import numpy as np                                              # noqa: E402
from geom_metrics import real_mesh, psize, summarize            # noqa: E402
from geom_batch import load_arm                                 # noqa: E402

OUT = "/home/timothy/GitHub/Genesis/forge_common/main/outputs"
BATCHES = [("batch_gain_1p5em4", "1.5e-4"), ("batch_gain_1p5em5", "1.5e-5")]
IOU = "iou_2.00"
C1 = 0.001
HITS = (1, 5, 10, 17)

# Observed stall exposure, read from gain_sweep.log with stall_audit.py after
# prepending a synthetic "pressing_speed=25.0 m/s" header (gain_sweep.sh pinned
# speed via env and never wrote the ### line the parser needs).
STALL = {
    "f3_pg2pvel_hyb": (23.2, 0.0), "f4_penalty_hyb": (15.1, 0.0),
    "g0_grid_alone": (21.9, 0.0), "g1_grid_prod": (14.4, 0.0),
    "p3_pg2p_pos": (32.7, 0.0), "p5_penalty": (1.6, 0.0),
}


def arms_in(d):
    tags = sorted({f.replace("_verts.npz", "").replace("_hits.npz", "")
                   for f in os.listdir(d)
                   if f.endswith(("_verts.npz", "_hits.npz"))})
    return [t for t in tags if not t.startswith("_ref")]


def main():
    if not os.environ.get("AGF_PSIZE_MM"):
        sys.stderr.write("REFUSING: set AGF_PSIZE_MM=2.0 (run_meta records the "
                         "true value; deriving it from a mesh-seeded billet "
                         "reads +3.5% high and shifts every metric).\n")
        raise SystemExit(1)

    iou, span = {}, {}
    for hit in HITS:
        mesh = real_mesh(hit, "after")
        for b, g in BATCHES:
            d = os.path.join(OUT, b)
            for tag in arms_in(d):
                P, src = load_arm(d, tag, hit)
                if P is None or not len(P):
                    continue
                r = psize(len(P)) / 2.0
                iou[(hit, g, tag)] = float(summarize(mesh, P, r, [2.0])[IOU])
                span[(hit, g, tag)] = float((P.max(axis=0) - P.min(axis=0)).max())

    tags = sorted({t for (_, _, t) in iou})

    print("=== IoU@2.0 vs die-balance gain, everything else matched ===")
    print("(OLD card. n=1 per cell. C1 = %.3f)" % C1)
    print()
    print("%-18s %4s %10s %10s | %9s %7s %s"
          % ("arm", "hit", "g=1.5e-4", "g=1.5e-5", "delta", "xC1", "stall 1.5e-4 -> 1.5e-5"))
    print("-" * 100)
    for tag in tags:
        for hit in HITS:
            a = iou.get((hit, "1.5e-4", tag))
            b = iou.get((hit, "1.5e-5", tag))
            if a is None and b is None:
                continue
            if a is None or b is None:
                have = "1.5e-5" if a is None else "1.5e-4"
                print("%-18s %4d %10s %10s | %9s %7s  MISSING at the other gain (has %s only)"
                      % (tag, hit,
                         "--" if a is None else "%.4f" % a,
                         "--" if b is None else "%.4f" % b,
                         "--", "--", have))
                continue
            d = b - a
            s = STALL.get(tag)
            stxt = "%.1f%% -> %.1f%%" % s if s else ""
            print("%-18s %4d %10.4f %10.4f | %+9.4f %6.0fx  %s"
                  % (tag, hit, a, b, d, abs(d) / C1, stxt))
        print()

    print("=== THE GO/NO-GO: does the ranking reorder? ===")
    print("Arms ordered best-to-worst by IoU@2.0 within each gain.")
    print()
    for hit in HITS:
        oa = sorted([t for t in tags if (hit, "1.5e-4", t) in iou],
                    key=lambda t: -iou[(hit, "1.5e-4", t)])
        ob = sorted([t for t in tags if (hit, "1.5e-5", t) in iou],
                    key=lambda t: -iou[(hit, "1.5e-5", t)])
        common_a = [t for t in oa if t in ob]
        common_b = [t for t in ob if t in oa]
        verdict = "SAME ORDER" if common_a == common_b else "*** REORDERS ***"
        print("  hit %2d  %s" % (hit, verdict))
        print("     1.5e-4: %s" % " > ".join(common_a))
        print("     1.5e-5: %s" % " > ".join(common_b))
        if len(oa) != len(ob):
            print("     (note: %d arms at 1.5e-4 vs %d at 1.5e-5 -- an arm did not survive)"
                  % (len(oa), len(ob)))
        print()

    print("=== max span (mm) -- the work proxy, so a geometry delta is not read as fidelity ===")
    print("%-18s %4s %10s %10s | %9s" % ("arm", "hit", "g=1.5e-4", "g=1.5e-5", "delta mm"))
    print("-" * 64)
    for tag in tags:
        for hit in (1, 17):
            a = span.get((hit, "1.5e-4", tag))
            b = span.get((hit, "1.5e-5", tag))
            if a is None or b is None:
                continue
            print("%-18s %4d %10.3f %10.3f | %+9.3f" % (tag, hit, a, b, b - a))
    print()
    print("Read a ranking change against the arm's own replicate scatter, not against C1.")
    print("No replicates exist inside these batches; both cells are n=1.")


if __name__ == "__main__":
    main()
