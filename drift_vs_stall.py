#!/usr/bin/env python3
"""Pair each arm's speed-driven geometry drift against its stalling, per card.

=============================================================================
SUPERSEDED. The stall side of this pairing is measured the wrong way, and the
conclusion I drew from it is retracted. Kept because the correction is more
instructive than the deletion.
=============================================================================

This infers stalling from |force_L_peak - force_R_peak| crossing the closed-form
bound. Those two per-strike peaks can occur at different frames, so the quantity
is a LOWER bound on max|dF| and the method under-reports.

How badly: it read p5_penalty as 0% stalling at every speed. The sweep logs,
which record v=[vL,vR] per pressing frame and therefore OBSERVE a die commanded
to exactly zero, show p5 stalling 0.6% / 0.6% / 2.6% / 4.4% across 25 -> 3.125
m/s. Not zero, and rising monotonically.

What I concluded from this script -- that p4 drifting 9.35 mm and p5 drifting
0.67 mm at "identical 0% stalling" refutes the controller explanation -- does
not survive that. The premise was an artifact of a blind proxy. And there are
no observed stall data for p3/p4 at all, so the between-method question this
script was built to answer is currently UNTESTED rather than answered.

Use ~/speed_sweep*.log and observed_stall.py instead. What still stands from
here is the geometry drift itself and its reproducibility across r1/r2, which
come from summary.json and do not depend on the stall proxy.
"""

_ORIGINAL_DOCSTRING = """Pair each arm's speed-driven geometry drift against its stalling, per card.

The controller hypothesis predicts these two travel together: arms that stall
more as the press slows should be the arms whose geometry moves with speed.
Arms that never approach the stall threshold should be speed-stable.

This tests that pairing directly instead of eyeballing two separate tables.
Duplicate runs (r1/r2) are kept apart rather than averaged, because agreement
between them is the only evidence available here that a drift is reproducible
rather than noise.
"""

import sys
print("SUPERSEDED -- see module docstring. The stall column here is a blind proxy.",
      file=sys.stderr)

import glob
import json
import math
import os
import re

OUT = "/home/timothy/GitHub/Genesis/forge_common/main/outputs"
GAIN = 1.5e-4


def stall_threshold(v):
    return math.sqrt(v * 20000.0 / GAIN)


def speed_of(name):
    m = re.search(r"_(\d+)p(\d+)$", name)
    return float("%s.%s" % (m.group(1), m.group(2))) if m else None


def collect(pattern):
    data = {}
    for d in sorted(glob.glob(os.path.join(OUT, pattern))):
        v = speed_of(os.path.basename(d))
        if v is None:
            continue
        thr = stall_threshold(v)
        spans = {}
        sp = os.path.join(d, "summary.json")
        if os.path.exists(sp):
            try:
                for row in json.load(open(sp)):
                    if row.get("span_mm"):
                        spans[row["tag"]] = row["span_mm"]
            except Exception:
                pass
        for p in sorted(glob.glob(os.path.join(d, "*.diag.jsonl"))):
            tag = os.path.basename(p)[: -len(".diag.jsonl")]
            dpk = []
            for line in open(p):
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                fl, fr = r.get("force_L_peak"), r.get("force_R_peak")
                if fl is not None and fr is not None:
                    dpk.append(abs(fl - fr))
            if not dpk:
                continue
            data.setdefault(tag, {})[v] = {
                "frac_over": sum(1 for x in dpk if x > thr) / len(dpk),
                "span": spans.get(tag),
                "thr": thr,
            }
    return data


def report(title, pattern):
    data = collect(pattern)
    print("=" * 86)
    print(title)
    print("=" * 86)
    print("%-22s %9s %9s %11s %11s   %s" % (
        "arm", "fast m/s", "slow m/s", "dX span mm", "stall f->s", "verdict"))
    rows = []
    for tag, bys in sorted(data.items()):
        speeds = sorted(bys, reverse=True)
        if len(speeds) < 2:
            continue
        f, s = speeds[0], speeds[-1]
        sf, ss = bys[f]["span"], bys[s]["span"]
        if not sf or not ss:
            continue
        dspan = ss[0] - sf[0]
        of, os_ = bys[f]["frac_over"], bys[s]["frac_over"]
        rows.append((tag, f, s, dspan, of, os_))
    for tag, f, s, dspan, of, os_ in sorted(rows, key=lambda r: r[3]):
        stalls = os_ > 0.05 or of > 0.05
        drifts = abs(dspan) > 2.0
        if drifts and stalls:
            v = "drifts AND stalls -> controller consistent"
        elif drifts and not stalls:
            v = "DRIFTS WITHOUT STALLING -> controller cannot explain"
        elif not drifts and not stalls:
            v = "stable, no stalling"
        else:
            v = "stalls but stable"
        print("%-22s %9.3f %9.3f %+11.3f %5.0f%%->%3.0f%%   %s" % (
            tag, f, s, dspan, 100 * of, 100 * os_, v))
    print()


report("OLD CARD  (batch_speed_*, 25 -> 3.125 m/s)", "batch_speed_*")
report("NEW CARD  (batch_spdmx_*, 25 -> 12.5 m/s)", "batch_spdmx_*")

print("Reproducibility of the drift, from the duplicate runs (new card):")
d = collect("batch_spdmx_*")
seen = {}
for tag, bys in d.items():
    base = re.sub(r"_r\d+$", "", tag)
    sp = sorted(bys, reverse=True)
    if len(sp) < 2 or not bys[sp[0]]["span"] or not bys[sp[-1]]["span"]:
        continue
    seen.setdefault(base, []).append(bys[sp[-1]]["span"][0] - bys[sp[0]]["span"][0])
for base, vals in sorted(seen.items()):
    if len(vals) >= 2:
        print("   %-20s %s   spread %.3f mm" % (
            base, " ".join("%+.3f" % v for v in vals), max(vals) - min(vals)))
