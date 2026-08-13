"""Score a batched contact-arm sweep: packing, det F, force, penetration.

Reuses score_arms.py's eta/union-of-balls definition by IMPORT rather than reimplementation, so
these numbers are the same metric as the banked 17-hit comparison and not a lookalike.

TWO THINGS THIS GETS RIGHT THAT THE FIRST VERSION DID NOT:

  FIXED HIT INDEX. Arms reach different depths before going unstable, so comparing each at its
  own final state compares different amounts of deformation. Everything here is read at one
  requested hit; an arm that never reached it is reported as absent rather than silently scored
  at a shallower state.

  A REAL t=0 REFERENCE. eta drop is measured against the fresh bar saved by the driver. The
  earlier version referenced the no-contact control's final state, which only works if that
  control truly did nothing -- an assumption, not a measurement.

eta is MONTE CARLO: scoring the reference against itself yields ~0.16 pp, so differences below
about 0.2 pp are noise, not signal.
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

try:
    from score_arms import eta, psize_from_N, spans as spans_of
except Exception as exc:
    print("could not import score_arms (%r); eta will be skipped" % exc)
    eta = None


def load_diag(d, tag, upto=None):
    path = os.path.join(d, "%s.diag.jsonl" % tag)
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            if upto is None or (r.get("strike") or 0) <= upto:
                rows.append(r)
    return rows


def load_cloud(d, tag, hit):
    hp = os.path.join(d, "%s_hits.npz" % tag)
    if os.path.exists(hp):
        with np.load(hp) as z:
            key = "hit_%02d" % hit
            if key in z:
                return np.asarray(z[key], dtype=np.float64)
        return None                      # reached the file but not that depth
    vp = os.path.join(d, "%s_verts.npz" % tag)
    if os.path.exists(vp):
        with np.load(vp) as z:
            return np.asarray(z["verts"], dtype=np.float64)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", default="batch")
    ap.add_argument("--hit", type=int, default=None,
                    help="score every arm at this hit; default = each arm's final state")
    args = ap.parse_args()

    d = os.path.expanduser("~/GitHub/Genesis/forge_common/main/outputs/%s" % args.batch)
    spath = os.path.join(d, "summary.json")
    if not os.path.exists(spath):
        print("no summary.json in %s" % d)
        return 1
    with open(spath, "r", encoding="utf-8") as fh:
        summary = json.load(fh)

    # ------------------------------------------------------------------ reference
    ref_eta = None
    rp = os.path.join(d, "_ref_fresh_verts.npz")
    if os.path.exists(rp) and eta is not None:
        with np.load(rp) as z:
            R = np.asarray(z["verts"], dtype=np.float64)
        ref_eta = eta(R, psize_from_N(len(R)) / 2.0)
        selfcheck = eta(R, psize_from_N(len(R)) / 2.0)
        print("reference: fresh bar (t=0), %d verts, spans %s, eta=%.4f"
              % (len(R), tuple(round(s, 2) for s in spans_of(R)), ref_eta))
        print("  eta self-check on the SAME cloud: %.4f  =>  Monte-Carlo noise %.3f pp."
              % (selfcheck, abs(selfcheck - ref_eta) / ref_eta * 100.0))
        print("  Treat eta differences below ~0.2 pp as noise.")
    else:
        print("no _ref_fresh_verts.npz -- eta drops omitted (rerun the sweep to capture it)")
    print()

    hdr = "%-24s %-18s %5s %8s %9s %9s %9s %9s %9s"
    print(hdr % ("arm", "status", "hit", "eta", "d_eta%", "detF%", "F_press", "pen_max", "pen/dx"))
    print("-" * 116)

    for rec in summary:
        tag, status = rec.get("tag"), rec.get("status", "?")
        hit = args.hit if args.hit is not None else rec.get("hits_done")
        P = load_cloud(d, tag, hit) if hit else None
        if P is None:
            print("%-24s %-18s %5s   (did not reach this hit)" % (tag, status[:18], hit))
            continue

        e = d_eta = None
        if eta is not None and len(P):
            e = eta(P, psize_from_N(len(P)) / 2.0)
            if ref_eta:
                d_eta = (e - ref_eta) / ref_eta * 100.0

        diag = load_diag(d, tag, upto=hit)
        last, first = (diag[-1] if diag else {}), (diag[0] if diag else {})
        detF_drop = None
        if last.get("detF_mean") is not None and first.get("detF_first_mean"):
            detF_drop = (last["detF_mean"] - first["detF_first_mean"]) / first["detF_first_mean"] * 100.0

        # Pressing mean only: peak force is contaminated by gripper actuation during approach
        # (it reads ~128 kN with nothing touching the bar), so it is not a contact metric.
        fp = [r.get("force_L_press_mean") for r in diag if r.get("force_L_press_mean") is not None]
        fp = float(np.mean(fp)) if fp else None
        pens = [r.get("pen_max") for r in diag if r.get("pen_max") is not None]
        pmax = max(pens) if pens else None
        dx = next((r.get("pen_dx") for r in diag if r.get("pen_dx")), None)

        def f(v, nd=3):
            return "--" if v is None else ("%%.%df" % nd) % v

        print(hdr % (tag, status[:18], hit, f(e, 4), f(d_eta, 2), f(detF_drop, 2), f(fp, 1),
                     f(pmax * 1000.0 if pmax is not None else None, 3),
                     f(pmax / dx if (pmax is not None and dx) else None, 3)))

    print()
    print("eta      = union-of-balls packing efficiency; falls only via interpenetration")
    print("d_eta%   = packing drop vs the t=0 fresh bar")
    print("detF%    = det(F) change up to this hit; the volume change F can SEE")
    print("F_press  = mean left-die force over PRESSING frames only")
    print("pen_max  = deepest penetration into the die up to this hit, mm (pen/dx = grid cells)")
    print("           NOTE teleport-on arms read exactly 0 BY CONSTRUCTION -- the projection")
    print("           targets signed_dist >= margin -- so this discriminates only teleport-off arms.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
