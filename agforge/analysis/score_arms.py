"""Score contact-method arms on volume, force and shape -- at a FIXED hit index.

This is the harness the contact-method comparison is scored with. It exists to make the
audited confounds structurally impossible to reintroduce rather than merely remembered:

  #5  arms reach different final hits  -> every arm is read at the SAME --hit, and an arm
                                          that never reached it is reported as INCOMPLETE
                                          rather than silently compared at its own endpoint.
  #7  det(F) was never captured        -> read from the diag JSONL and paired with eta, so
                                          the GAP (the actual quantity of interest) is
                                          computed rather than inferred.
  #8  force was never captured         -> peak and pressing-mean, per gripper.
  #9  force-reporting coverage varies  -> each arm carries a `reports_force` label; an arm
                                          whose mechanism applies no reaction force to the
                                          die (fluidlab, the bare teleport) will read low
                                          for a REPORTING reason, not a physical one, and
                                          the table says so instead of letting the number
                                          stand bare.
  #12 nondeterminism                   -> mean +- half-range over repeats, and the spread
                                          is printed, never collapsed to a point estimate.
  #13 geometric divergence             -> spans are printed beside volume, because two arms
                                          that deformed differently are not the same
                                          experiment minus one term.

METRIC OF RECORD: union-of-balls packing efficiency eta = V_union / (N * v_ball), ball
radius psize/2. At t=0 the lattice spacing IS psize, so the balls exactly touch and eta can
only fall through interpenetration. Parameter-free and resolution-independent -- unlike the
marching-cubes "enclosed volume" recipe, which moves from -8% to -4% on ONE cloud purely by
varying smoothing width and is therefore retracted.

Usage:
    score_arms.py --hit 10 --arms grid=velo_cm_grid_r1,velo_cm_grid_r2 particle=velo_cm_particle_r1
"""
import argparse
import glob
import json
import os
import sqlite3

import numpy as np
from scipy.spatial import cKDTree

OUT = os.path.expanduser("~/GitHub/Genesis/forge_common/main/outputs")
RNG = np.random.default_rng(0)

# Real stock the replay is initialised from (forge_common.real_scale).
REAL_R_MM, REAL_L_MM = 20.0, 58.57

# Arms whose contact mechanism applies NO reaction force to the die. Verified by resolving
# the call graph to _func_apply_coupling_force, not by grepping bodies for a string.
NO_FORCE_ARMS = {"fluidlab", "teleport_only", "pconly"}


# ---------------------------------------------------------------------------- loading
def db_path(tag):
    return tag if tag.endswith(".db") else os.path.join(OUT, tag + ".db")


def load_cloud(tag, hit):
    p = db_path(tag)
    if not os.path.exists(p):
        return None
    con = sqlite3.connect("file:%s?mode=ro" % p, uri=True)
    try:
        r = con.execute("SELECT vertices FROM hits WHERE step_number=?", (hit,)).fetchone()
    finally:
        con.close()
    if r is None or r[0] is None:
        return None
    return np.asarray(json.loads(r[0]), dtype=np.float64).reshape(-1, 3)


def hits_reached(tag):
    p = db_path(tag)
    if not os.path.exists(p):
        return -1
    con = sqlite3.connect("file:%s?mode=ro" % p, uri=True)
    try:
        r = con.execute("SELECT MAX(step_number) FROM hits").fetchone()
    finally:
        con.close()
    return int(r[0]) if r and r[0] is not None else 0


def load_diag(tag, hit):
    """Per-strike force/det(F) row for `hit`, from the diagnostics tap."""
    stem = tag[:-3] if tag.endswith(".db") else tag
    for cand in (os.path.join(OUT, stem + ".diag.jsonl"),
                 "/tmp/diag_%s.jsonl" % os.path.basename(stem)):
        if os.path.exists(cand):
            with open(cand, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except ValueError:
                        continue
                    if row.get("strike") == hit:
                        return row
    return None


# ---------------------------------------------------------------------------- metrics
def union_balls(P, r, nsamp=3_000_000):
    lo, hi = P.min(0) - 1.5 * r, P.max(0) + 1.5 * r
    box = float(np.prod(hi - lo))
    tree = cKDTree(P)
    hits = done = 0
    while done < nsamp:
        m = min(400_000, nsamp - done)
        d, _ = tree.query(RNG.uniform(lo, hi, size=(m, 3)), k=1, distance_upper_bound=r)
        hits += int(np.isfinite(d).sum())
        done += m
    return box * hits / done


def eta(P, r):
    return union_balls(P, r) / (len(P) * ((4.0 / 3.0) * np.pi * r ** 3))


def psize_from_N(n):
    """psize = (V_bar / N)^(1/3): particles sit on a cubic lattice of spacing psize, so each
    occupies psize^3. Derived rather than hardcoded so a resolution change cannot silently
    invalidate the ball radius."""
    return float((np.pi * REAL_R_MM ** 2 * REAL_L_MM / n) ** (1.0 / 3.0))


def spans(P):
    return tuple(float(P[:, i].max() - P[:, i].min()) for i in range(3))


def agg(vals):
    """mean +- half-range; half-range (not stdev) because n is small and the honest
    statement at n=3 is the observed spread, not an estimated one."""
    v = [x for x in vals if x is not None and np.isfinite(x)]
    if not v:
        return None, None
    return float(np.mean(v)), float((max(v) - min(v)) / 2.0)


def fmt(m, s, scale=1.0, unit="", nd=2):
    if m is None:
        return "--"
    if s is None or s == 0:
        return ("%%.%df%%s" % nd) % (m * scale, unit)
    return ("%%.%df+-%%.%df%%s" % (nd, nd)) % (m * scale, s * scale, unit)


# ---------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hit", type=int, required=True,
                    help="fixed hit index every arm is compared at")
    ap.add_argument("--ref-hit", type=int, default=1,
                    help="baseline hit for the eta drop (default 1)")
    ap.add_argument("--arms", nargs="+", required=True,
                    help="name=tag1,tag2,... (repeats of one arm)")
    a = ap.parse_args()

    parsed = []
    for spec in a.arms:
        name, _, tags = spec.partition("=")
        parsed.append((name, [t for t in tags.split(",") if t]))

    print("=" * 108)
    print("CONTACT-METHOD COMPARISON   compared at hit %d (ref hit %d)" % (a.hit, a.ref_hit))
    print("=" * 108)

    rows = []
    for name, tags in parsed:
        reached = [hits_reached(t) for t in tags]
        usable = [t for t, h in zip(tags, reached) if h >= a.hit]
        if not usable:
            rows.append((name, reached, None))
            continue

        e_ref, e_hit, dets, fL, fR, fLp, sp = [], [], [], [], [], [], []
        for t in usable:
            P0, PH = load_cloud(t, a.ref_hit), load_cloud(t, a.hit)
            if PH is None:
                continue
            r = psize_from_N(len(PH)) / 2.0
            if P0 is not None:
                e_ref.append(eta(P0, r))
            e_hit.append(eta(PH, r))
            sp.append(spans(PH))
            d = load_diag(t, a.hit)
            if d:
                dets.append(d.get("detF_mean"))
                # CONFOUND #14, found by the t1_nocontact control: force_*_peak reads
                # ~128 kN even with NOTHING touching the bar -- it picks up gripper
                # actuation during APPROACH. force_*_press_mean is gated to the
                # PRESSING state and reads exactly 0.0 on that control, so it is the
                # only trustworthy force metric. Peak is retained but not compared.
                fL.append(d.get("force_L_press_mean"))
                fR.append(d.get("force_R_press_mean"))
                fLp.append(d.get("force_L_peak"))

        rows.append((name, reached, {
            "n": len(usable), "eta_ref": agg(e_ref), "eta_hit": agg(e_hit),
            "det": agg(dets), "fL": agg(fL), "fR": agg(fR), "fLp": agg(fLp),
            "spans": [agg([s[i] for s in sp]) for i in range(3)],
        }))

    hdr = "%-18s %5s %6s %14s %14s %12s %16s %14s"
    print(hdr % ("arm", "runs", "hits", "eta drop %", "det F drop %", "GAP pp",
                 "press force kN", "span x mm"))
    print("-" * 108)
    for name, reached, m in rows:
        hits_s = ",".join(str(h) for h in reached)
        if m is None:
            print("%-18s %5s %6s   INCOMPLETE -- never reached hit %d" % (name, "-", hits_s, a.hit))
            continue
        d_eta = (None, None)
        if m["eta_ref"][0] is not None and m["eta_hit"][0] is not None:
            d_eta = ((m["eta_hit"][0] - m["eta_ref"][0]) / m["eta_ref"][0] * 100.0,
                     (m["eta_hit"][1] or 0) / m["eta_ref"][0] * 100.0)
        d_det = ((m["det"][0] - 1.0) * 100.0, (m["det"][1] or 0) * 100.0) \
            if m["det"][0] is not None else (None, None)
        gap = (d_eta[0] - d_det[0]) if (d_eta[0] is not None and d_det[0] is not None) else None
        force = m["fL"][0] + m["fR"][0] if (m["fL"][0] and m["fR"][0]) else None
        flag = " (no reaction force)" if name.lower() in NO_FORCE_ARMS else ""
        print(hdr % (
            name, m["n"], hits_s,
            fmt(*d_eta, nd=2), fmt(*d_det, nd=2),
            "%.2f" % gap if gap is not None else "--",
            (fmt(force, None, 1e-3, nd=1) + flag) if force else "--",
            fmt(*m["spans"][0], nd=2),
        ))

    print()
    print("eta drop  = union-of-balls packing efficiency, hit %d vs hit %d. Falls only via"
          % (a.hit, a.ref_hit))
    print("            interpenetration. GAP = eta drop - det F drop: the part of the volume")
    print("            loss the deformation gradient cannot see.")
    print("force     = MEAN (L+R) resistance while PRESSING. Peak is deliberately NOT used:")
    print("            the no-contact control reports ~128 kN peak with nothing touching")
    print("            (gripper actuation during approach) but 0.0 pressing-mean.")
    print("            Arms flagged 'no reaction force' read low for a REPORTING reason.")
    print("spans     = bar extent; two arms with different spans deformed differently and are")
    print("            not the same experiment minus one term (confound #13).")


if __name__ == "__main__":
    main()
