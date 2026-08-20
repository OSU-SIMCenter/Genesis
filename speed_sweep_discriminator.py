#!/usr/bin/env python3
"""Inertial artifact or controller artifact? Decided from data already on disk.

=============================================================================
SUPERSEDED -- DO NOT RUN THIS AND BELIEVE THE ANSWER.

It tests a two-way question that is actually three-way, and the verdict string
it prints does not say so.
=============================================================================

The script weighs INERTIAL (drift shrinks as the press slows) against
CONTROLLER (drift grows as the press slows) and prints one or the other.

There is a third explanation, DISCRETIZATION, and it makes the SAME
directional prediction as INERTIAL.  So an "inertial" verdict from here is
really "inertial OR discretization, indistinguishable", and nothing in the
output communicates that.  A reader would take a confounded answer as a clean
one.

Worse, the two are separated by an experiment this script cannot do, because
the archived data cannot support it: vary dt at FIXED press speed.  A-9 ran
exactly that on p4_pg2p_vel at hit 17 and the two routes to doubling the
substep count move Lx in OPPOSITE directions --

    speed 25 -> 12.5 m/s      Lx  -9.38 mm
    CFL 0.90 -> 0.45, v fixed Lx  +9.49 mm

Refining dt RECOVERS deformation that a coarse timestep was losing; slowing
the press REMOVES it.  Steps are not a shared error term, so "more steps" is
not a mechanism and this script's whole framing is too coarse.

The stall side is also wrong here for the same reason recorded in
drift_vs_stall.py: it infers stalling from a per-strike force proxy that is a
lower bound, and it read p5_penalty as 0% at every speed when the sweep logs
show p5 stalling 0.6% -> 4.4%.

Use ~/speed_sweep*.log with observed_stall.py for stall, and a fixed-speed dt
sweep for discretization.  Kept only so the reasoning and its correction stay
on the record.
"""

_SUPERSEDED_ORIGINAL = """Inertial artifact or controller artifact? Decided from data already on disk.

KE and IE cannot be recovered from the archived batches -- they store surface
mesh vertices per hit plus PER-STRIKE force aggregates, and nothing per-particle
or per-step.  So the energy criterion itself has to ride along on new runs.

But the two competing explanations for geometry drifting with press speed make
OPPOSITE predictions that the archived data can already separate:

  INERTIAL.  Slowing the press makes the problem more quasi-static.  The press
    ramp lengthens in proportion to 1/speed while the billet's natural period is
    fixed by its geometry and wave speed, so the ratio ramp/period IMPROVES.  If
    inertia were driving the drift, the drift should shrink as speed falls.

  CONTROLLER.  The die-balance loop stalls a die when the force imbalance
    exceeds |dF|_stall = sqrt(pressing_speed * 20000 / gain).  That threshold
    falls as sqrt(speed), so slowing the press LOWERS the bar for stalling and
    produces MORE of it.  If the controller were driving the drift, the drift
    should GROW as speed falls.

They point in opposite directions, which is what makes this decidable.

WHAT IS A PROXY HERE, AND WHY IT MATTERS

The archived diag records force_L_peak and force_R_peak per strike, not the
per-frame force traces.  Those two peaks can occur at different frames, so
|force_L_peak - force_R_peak| is NOT the peak of |dF|; it is a lower bound on
it.  The press-phase means are recorded too and give a second, smoother proxy.
Both are reported.  A conclusion that depends on which proxy is used is not a
conclusion, and this script says so rather than picking the flattering one.
"""

import sys as _sys
print("SUPERSEDED -- this script cannot distinguish INERTIAL from "
      "DISCRETIZATION; they predict the same direction. See module docstring.",
      file=_sys.stderr)

import glob
import json
import math
import os
import re

OUT = "/home/timothy/GitHub/Genesis/forge_common/main/outputs"

# Die-balance loop, as documented in the coupling doc section 4.7.4 and
# reproduced across 245 hits on two branches.
BALANCE_GAIN = 1.5e-4
IMBALANCE_THRESHOLD_N = 20000.0


def stall_threshold(pressing_speed_m_s, gain=BALANCE_GAIN):
    """|dF|_stall = sqrt(pressing_speed * 20000 / gain)."""
    return math.sqrt(pressing_speed_m_s * IMBALANCE_THRESHOLD_N / gain)


def speed_from_dirname(name):
    """batch_speed_3p125 -> 3.125 ; batch_spdmx_12p5 -> 12.5"""
    m = re.search(r"_(\d+)p(\d+)$", name)
    if not m:
        return None
    return float("%s.%s" % (m.group(1), m.group(2)))


def read_diag(path):
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    pass
    return rows


def analyse_batch(d):
    name = os.path.basename(d.rstrip("/"))
    speed = speed_from_dirname(name)
    if speed is None:
        return None
    thr = stall_threshold(speed)

    arms = {}
    for p in sorted(glob.glob(os.path.join(d, "*.diag.jsonl"))):
        tag = os.path.basename(p)[: -len(".diag.jsonl")]
        rows = read_diag(p)
        if not rows:
            continue
        dpeak, dmean, npf, stops = [], [], [], []
        for r in rows:
            fl, fr = r.get("force_L_peak"), r.get("force_R_peak")
            if fl is not None and fr is not None:
                dpeak.append(abs(fl - fr))
            ml, mr = r.get("force_L_press_mean"), r.get("force_R_press_mean")
            if ml is not None and mr is not None:
                dmean.append(abs(ml - mr))
            if r.get("n_press_frames"):
                npf.append(r["n_press_frames"])
            stops.append(r.get("stop_reason"))
        if not dpeak:
            continue
        arms[tag] = {
            "n_strikes": len(rows),
            "dF_peak_proxy_max": max(dpeak),
            "dF_peak_proxy_median": sorted(dpeak)[len(dpeak) // 2],
            "frac_strikes_over_stall_peakproxy": sum(1 for x in dpeak if x > thr) / len(dpeak),
            "dF_pressmean_max": max(dmean) if dmean else None,
            "frac_strikes_over_stall_meanproxy": (
                sum(1 for x in dmean if x > thr) / len(dmean)) if dmean else None,
            "n_press_frames_median": sorted(npf)[len(npf) // 2] if npf else None,
            "n_press_frames_total": sum(npf) if npf else None,
            "stop_reasons": {s: stops.count(s) for s in set(stops)},
        }

    spans = {}
    sp = os.path.join(d, "summary.json")
    if os.path.exists(sp):
        try:
            for row in json.load(open(sp)):
                if row.get("span_mm"):
                    spans[row["tag"]] = row["span_mm"]
        except Exception:
            pass

    return {"batch": name, "speed_m_s": speed, "stall_threshold_N": thr,
            "arms": arms, "span_mm": spans}


def main():
    families = {
        "old card (batch_speed_*)": sorted(glob.glob(os.path.join(OUT, "batch_speed_*"))),
        "new card (batch_spdmx_*)": sorted(glob.glob(os.path.join(OUT, "batch_spdmx_*"))),
    }

    for fam, dirs in families.items():
        results = [r for r in (analyse_batch(d) for d in dirs) if r]
        if not results:
            continue
        results.sort(key=lambda r: -r["speed_m_s"])
        print("=" * 78)
        print(fam)
        print("=" * 78)

        # The controller prediction, arm by arm, so a per-arm effect is visible
        # rather than averaged away.
        tags = sorted({t for r in results for t in r["arms"]})
        for tag in tags:
            print("\n  ARM: %s" % tag)
            print("    speed   stall_thr      dF_peak_prox   over(peak)  over(mean)  press_frames")
            for r in results:
                a = r["arms"].get(tag)
                if not a:
                    continue
                print("    %6.3f  %9.0f N   %10.0f N   %8.0f%%   %8s   %s" % (
                    r["speed_m_s"], r["stall_threshold_N"],
                    a["dF_peak_proxy_median"],
                    100 * a["frac_strikes_over_stall_peakproxy"],
                    ("%.0f%%" % (100 * a["frac_strikes_over_stall_meanproxy"]))
                    if a["frac_strikes_over_stall_meanproxy"] is not None else "--",
                    a["n_press_frames_median"]))
            # Direction of the effect across speed.
            pts = [(r["speed_m_s"], r["arms"][tag]["frac_strikes_over_stall_peakproxy"])
                   for r in results if tag in r["arms"]]
            if len(pts) >= 2:
                fast, slow = pts[0], pts[-1]
                d = slow[1] - fast[1]
                verdict = ("MORE stalling as speed falls -> CONTROLLER"
                           if d > 0.05 else
                           "LESS stalling as speed falls -> not controller"
                           if d < -0.05 else
                           "flat -> neither signature")
                print("    %.3f m/s: %.0f%%  ->  %.3f m/s: %.0f%%   [%s]" % (
                    fast[0], 100 * fast[1], slow[0], 100 * slow[1], verdict))

        print("\n  STOP REASONS")
        for r in results:
            for tag, a in sorted(r["arms"].items()):
                print("    %6.3f  %-22s %s" % (r["speed_m_s"], tag, a["stop_reasons"]))

        print("\n  FINAL SPAN (criterion iv proxy: does peak deformation settle?)")
        for r in results:
            for tag, s in sorted(r["span_mm"].items()):
                print("    %6.3f  %-22s  %s" % (r["speed_m_s"], tag,
                                                ["%.3f" % v for v in s]))
        print()


if __name__ == "__main__":
    main()
