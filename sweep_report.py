#!/usr/bin/env python3
"""Read the KE/IE speed sweep and answer the inertia question directly.

The prediction that matters: if inertia were driving the geometry drift, KE/IE
should be non-negligible and should FALL as the press slows.  If KE/IE is tiny
everywhere and falls the way simple scaling says it must, inertia is not
available as an explanation and the drift has to come from somewhere else.

Simple scaling: kinetic energy goes as v^2 while the plastic work done over a
fixed displacement is roughly speed-independent, so KE/IE should drop by about
4x per halving of press speed.  A measured ratio that tracks that is evidence
the monitor is reading real inertia; one that does not is worth explaining
before anything is concluded from it.
"""

import glob
import json
import os

O = "/home/timothy/GitHub/Genesis/forge_common/main/outputs/energy_a1"


def speed_of(p):
    t = os.path.basename(p)[len("sweep_"):-len(".json")]
    return float(t.replace("p", "."))


rows = []
for p in sorted(glob.glob(os.path.join(O, "sweep_*.json"))):
    if p.endswith(".jsonl"):
        continue
    try:
        d = json.load(open(p))
    except Exception as e:
        print("skip", p, e)
        continue
    s = d["summary"]
    rows.append((speed_of(p), s, d.get("samples", [])))

rows.sort(key=lambda r: -r[0])

print("=" * 100)
print("A1 QUASI-STATIC CRITERIA vs PRESS SPEED   (KE = MPM particles only, dies excluded)")
print("=" * 100)
print("%7s %9s %11s %12s %12s %11s %9s" % (
    "m/s", "KE_max_J", "IE_final_J", "KE/IE@peak", "KE/IE_ss", "dIE/IE_ss", "status"))
for v, s, _ in rows:
    ie = s.get("E_plastic_if_per_step_J")
    ie_tot = None
    if ie is not None:
        ie_tot = ie
    ss = s.get("criterion_ii_KE_over_IE_lt_0p001_at_steady_state", {})
    iii = s.get("criterion_iii_dIE_dt_negligible_at_steady_state", {})
    print("%7.3f %9.3f %11.2f %12.3e %12.3e %11.3e %9s" % (
        v, s.get("KE_J_max", float("nan")), ie_tot if ie_tot else float("nan"),
        s.get("KE_over_IE_at_peak_KE") or float("nan"),
        ss.get("max_at_steady_state") or float("nan"),
        iii.get("IE_fractional_drift_over_steady_window") or float("nan"),
        s.get("status", "?")))

print("\nCRITERIA (i)-(iv):")
for v, s, _ in rows:
    print("  %6.3f m/s  i=%-5s ii=%-5s iii=%-5s iv=%-5s   invalid=%s" % (
        v,
        s.get("criterion_i_KE_le_5pct_of_IE", {}).get("passes"),
        s.get("criterion_ii_KE_over_IE_lt_0p001_at_steady_state", {}).get("passes"),
        s.get("criterion_iii_dIE_dt_negligible_at_steady_state", {}).get("passes"),
        s.get("criterion_iv_peak_deformation_constant_at_steady_state", {}).get("passes"),
        s.get("n_invalid_samples_excluded")))

print("\nDOES KE SCALE AS v^2 ?  (the check that the monitor is reading real inertia)")
if len(rows) >= 2:
    v0, s0, _ = rows[0]
    for v, s, _ in rows:
        pred = s0["KE_J_max"] * (v / v0) ** 2
        got = s["KE_J_max"]
        print("   %6.3f m/s: KE_max measured %8.3f J   v^2-predicted %8.3f J   ratio %.2f" % (
            v, got, pred, got / pred if pred else float("nan")))

print("\nFINAL SPAN vs SPEED  (does the geometry drift reproduce in-monitor?)")
for v, s, samples in rows:
    valid = [x for x in samples if x.get("physics_valid", True) and x.get("span_mm")]
    if valid:
        sp = valid[-1]["span_mm"]
        print("   %6.3f m/s: %s" % (v, ["%.3f" % q for q in sp]))

print("\nPEAK-KE PHASE ATTRIBUTION  (is peak KE the press, or the balance loop?)")
for v, s, samples in rows:
    valid = [x for x in samples if x.get("physics_valid", True)]
    if not valid:
        continue
    pk = max(valid, key=lambda x: x["KE_J"])
    print("   %6.3f m/s: peak KE %7.3f J at step %4s  state=%-10s v_max=%6.2f m/s" % (
        v, pk["KE_J"], pk.get("step"), pk.get("strike_state"), pk.get("v_max_m_s")))
