#!/usr/bin/env python3
"""Scan a run log for die-balance controller instability during PRESSING.

The press applies two corrections to the commanded die velocity when the two
dies disagree (`strike_controller.py`, PRESSING branch):

    adaptive_speed = pressing_speed * (SAFETY_THRESHOLD / |dF|)   when |dF| > 20 kN
    correction     = dF * force_balance_gain
    v_L, v_R       = clamp(adaptive_speed -/+ correction, min=0)

The base speed decays as 1/|dF| while the correction grows as |dF|, so they
cross. Past the crossing, `correction` exceeds `adaptive_speed`, one die clamps
to **zero** and the other is driven **faster than the nominal press speed** --
the press stops closing symmetrically and becomes one-sided. The threshold is
closed-form:

    |dF|_stall = sqrt(pressing_speed * SAFETY_THRESHOLD / force_balance_gain)

At the shipped 25 m/s, 20 kN and 1.5e-4 that is 57.7 kN -- the same order as the
total press force, so it takes a near-total asymmetry to reach, which is exactly
what the terminal transient produces.

    python3 press_balance_scan.py ~/profile_g0_17.log

Requires VERBOSE_LOGGING PRESSING telemetry in the log.
"""
import math
import re
import sys

PRESSING = re.compile(
    r"PRESSING\[(\d+)\]: F=\[([-0-9.]+),([-0-9.]+)\] dF=([-0-9.]+), v=\[([-0-9.]+),([-0-9.]+)\]")
NEW_HIT = re.compile(r"Strike -> PRESSING \(width=")
HOLDING = re.compile(r"Strike -> HOLDING \(([A-Za-z ]+), strain=([0-9.]+)")

SAFETY_THRESHOLD = 20000.0   # strike_controller.py, PRESSING branch
PRESSING_SPEED = 25.0        # options.py StrikeOptions.pressing_speed
GAIN = 1.5e-4                # options.py StrikeOptions.force_balance_gain


def stall_threshold(speed=PRESSING_SPEED, safety=SAFETY_THRESHOLD, gain=GAIN):
    """|dF| above which one die is commanded to zero velocity."""
    return math.sqrt(speed * safety / gain)


def main(path):
    thr = stall_threshold()
    print("stall threshold |dF| = %.0f N  (pressing_speed=%.1f, safety=%.0f, gain=%.1e)\n"
          % (thr, PRESSING_SPEED, SAFETY_THRESHOLD, GAIN))

    hits, cur = [], None
    for line in open(path, encoding="utf-8", errors="replace"):
        if NEW_HIT.search(line):
            cur = {"steps": [], "stop": None, "strain": None}
            hits.append(cur)
            continue
        if cur is None:
            continue
        m = PRESSING.search(line)
        if m:
            cur["steps"].append((int(m.group(1)), float(m.group(2)), float(m.group(3)),
                                 float(m.group(4)), float(m.group(5)), float(m.group(6))))
            continue
        m = HOLDING.search(line)
        if m:
            cur["stop"], cur["strain"] = m.group(1).strip(), float(m.group(2))

    if not hits:
        print("no PRESSING telemetry in %s (needs VERBOSE_LOGGING)" % path)
        return 1

    print("%-4s %-11s %-9s %-7s %-8s %s"
          % ("hit", "max|dF| N", "stalled", "over", "strain", "stop"))
    n_stall = n_over = 0
    by_stop = {}
    for i, h in enumerate(hits, start=1):
        if not h["steps"]:
            continue
        max_df = max(abs(s[3]) for s in h["steps"])
        stalled = any(s[4] == 0.0 or s[5] == 0.0 for s in h["steps"])
        over = max_df > thr
        n_stall += stalled
        n_over += over
        by_stop.setdefault(h["stop"], []).append(max_df)
        print("%-4d %-11.0f %-9s %-7s %-8s %s"
              % (i, max_df, "YES" if stalled else "-", "YES" if over else "-",
                 "%.4f" % h["strain"] if h["strain"] is not None else "?", h["stop"]))

    n = len([h for h in hits if h["steps"]])
    print("\n%d of %d hits drove a die to zero velocity; %d exceeded the closed-form "
          "threshold." % (n_stall, n, n_over))
    for stop, vals in sorted(by_stop.items(), key=lambda kv: str(kv[0])):
        print("  stop=%-14s n=%-3d mean max|dF| = %.0f N"
              % (stop, len(vals), sum(vals) / len(vals)))
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
