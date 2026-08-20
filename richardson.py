"""Richardson analysis of the D1a speed sweep, plus the controller-controlled comparison.

Speeds halve uniformly (25 -> 12.5 -> 6.25 -> 3.125), so the refinement ratio is r=2 and
successive-difference ratios give the observed order directly:

    ratio ~4 => 2nd order    ratio ~2 => 1st order    ratio ~1 => not converging
    ratio <1 => DIVERGING - differences growing; no limit to extrapolate to

Usage: python3 richardson.py [d1a_scores.jsonl]
"""
import json
import math
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/d1a_scores.jsonl"
R = [json.loads(l) for l in open(path)]
SPEEDS = [25.0, 12.5, 6.25, 3.125]
C1 = 0.001

# observed stall %, from stall_audit.py on speed_sweep.log
STALL = {("g1_grid_prod", 25.0): 1.3, ("g1_grid_prod", 12.5): 3.7,
         ("g1_grid_prod", 6.25): 8.1, ("g1_grid_prod", 3.125): 13.7,
         ("p5_penalty", 25.0): 0.6, ("p5_penalty", 12.5): 0.6,
         ("p5_penalty", 6.25): 2.6, ("p5_penalty", 3.125): 4.4}


def series(arm, hit):
    out = []
    for s in SPEEDS:
        v = next((r["iou"] for r in R
                  if r["set"] == "old" and r["arm"] == arm and r["hit"] == hit
                  and r["speed"] == s), None)
        out.append(v)
    return out


print("=" * 96)
print("RICHARDSON: successive-difference ratios, old card, r=2 (uniform halving)")
print("=" * 96)
print("%-16s %4s %9s %9s %9s %9s %9s %8s  %s"
      % ("arm", "hit", "d1", "d2", "d3", "d1/d2", "d2/d3", "order p", "verdict"))
print("-" * 96)
for arm in ("g1_grid_prod", "p5_penalty"):
    for hit in (1, 5, 10, 17):
        f = series(arm, hit)
        if any(x is None for x in f):
            continue
        d = [f[i] - f[i + 1] for i in range(3)]
        signs_ok = all(x > 0 for x in d) or all(x < 0 for x in d)
        r1 = d[0] / d[1] if d[1] else float("nan")
        r2 = d[1] / d[2] if d[2] else float("nan")
        if not signs_ok:
            verdict = "NON-MONOTONIC - Richardson inapplicable"
            p = float("nan")
        else:
            p = math.log(abs(r2), 2) if r2 > 0 else float("nan")
            if r2 > 1.5:
                verdict = "converging"
            elif r2 > 0.9:
                verdict = "NOT converging (ratio ~1)"
            else:
                verdict = "DIVERGING - differences GROWING"
        ps = "%8.2f" % p if p == p else "%8s" % "--"
        print("%-16s %4d %9.4f %9.4f %9.4f %9.3f %9.3f %s  %s"
              % (arm, hit, d[0], d[1], d[2], r1, r2, ps, verdict))

print()
print("=" * 96)
print("CONTROLLER-CONTROLLED COMPARISON: does the confound EXPLAIN the failure, or inflate it?")
print("=" * 96)
print("The die-balance stall is monotonic in speed, so it co-varies with N across the whole")
print("sweep. The test is to compare speed points whose stall exposure is closest.")
print()
print("%-16s %-14s %9s %9s %9s %10s %8s"
      % ("arm", "pair", "stall A", "stall B", "IoU A", "IoU B", "dIoU"))
print("-" * 96)
for arm in ("g1_grid_prod", "p5_penalty"):
    f = series(arm, 17)
    for i in range(3):
        a, b = SPEEDS[i], SPEEDS[i + 1]
        sa, sb = STALL[(arm, a)], STALL[(arm, b)]
        print("%-16s %-14s %8.1f%% %8.1f%% %9.4f %10.4f %8.4f  (%.0fx C1, stall delta %.1f pp)"
              % (arm, "%g->%g" % (a, b), sa, sb, f[i], f[i + 1],
                 abs(f[i] - f[i + 1]), abs(f[i] - f[i + 1]) / C1, abs(sb - sa)))
print()
p5 = series("p5_penalty", 17)
print("KEY: p5_penalty 25 -> 12.5 has IDENTICAL observed stall (0.6%% vs 0.6%%),")
print("     yet IoU moves %.4f = %.0fx criterion 1."
      % (abs(p5[0] - p5[1]), abs(p5[0] - p5[1]) / C1))
print("     => the controller inflates the headline 8x-range number, but a criterion-1")
print("        failure is present at CONSTANT controller exposure. It does not explain it away.")
