"""Run every contact arm in ONE process, sharing a single scene build.

WHY: measured with startup_probe.py, `init_stock` (scene + robot MJCF + die SDF + particle
sampling + eager kernel compilation) costs ~117 s warm against ~2.8 s per hit. A 3-hit arm is
therefore ~93% setup. Running N arms as N processes pays that N times; the contact mode was a
build-time `qd.static` flag, so it had to. With the runtime-switchable port it does not: the
adapter's `init_stock` already has a fast path that calls `controller.reset_simulation()` when the
stock geometry is unchanged, so every arm after the first costs only its hits.

Expected: ~9 arms x 3 hits goes from ~18 min to ~2 min.

THE CONTROL THAT EARNS ITS PLACE: sharing one process means arms are no longer isolated from each
other. An arm that goes unstable can leave non-finite particle state behind, and if that survives
`reset_simulation()` every subsequent arm is silently contaminated -- it would look like a real
result. So the fresh state is checked for finiteness and plausible extent after every reset, and
the sweep aborts rather than producing quietly-wrong numbers for the remaining arms. Isolation is
the thing being traded away here, so it is the thing that has to be verified.

Arms are ordered by information yield, so an early stop still leaves the valuable part done.

  AGF_CONTACT_RUNTIME_SWITCH=1 AGF_DIAG_PENETRATION=1 python batch_arms.py [--n-hits 3]
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.expanduser("~/GitHub/Genesis/forge_common/main"))

OUT = os.path.expanduser("~/GitHub/Genesis/forge_common/main/outputs/batch")


# Each arm is (tag, contact mode, teleport mech, refinements). `mech` is the custom
# apply_particle_contact teleport -- the axis the headline grid-vs-grid+teleport result turns on,
# and the reason the port had to be extended past what interactive-bench makes switchable.
ARMS = [
    # Grid contact is the baseline and every other method composes with it: the grid projection
    # supplies the non-penetration floor, the named correction supplies what the grid step cannot
    # see. Selecting a non-grid mode does not disable grid contact (see legacy_coupler's implicit
    # floor), so no configuration here resolves contact particle-side alone.
    dict(tag="grid",                     mode="grid",     mech=0),
    dict(tag="grid_position_correction", mode="grid",     mech=1),
    dict(tag="grid_fluidlab",            mode="fluidlab", mech=0, grid_floor=1),
    dict(tag="grid_particle_sdf",        mode="particle", mech=0, grid_floor=1),
    dict(tag="grid_penalty",             mode="penalty",  mech=0, grid_floor=1),
    # Control: no coupler-level contact at all.
    dict(tag="no_contact",               mode="none",     mech=0),
]


def configure(coupler, arm):
    """Apply one arm's contact configuration to the live coupler."""
    coupler.set_contact_mode(arm["mode"])
    coupler.set_refinement(
        per_node=bool(arm.get("per_node", 0)),
        c_injection=bool(arm.get("c_injection", 0)),
        ftmp_proj=bool(arm.get("ftmp", 0)),
        grid_floor=bool(arm.get("grid_floor", 0)),
    )
    coupler.set_particle_contact(
        mech=bool(arm.get("mech", 0)),
        c_project=bool(arm.get("c_project", 0)),
        f_feedback=bool(arm.get("f_feedback", 0)),
    )
    return coupler.get_contact_config()


def check_fresh(adapter, state, expect_n, expect_span_mm, tag):
    """
    Verify the reset actually produced a clean bar. Returns None if fine, else a reason string.

    This is the price of sharing one process. A previous arm that went unstable can leave
    non-finite or wildly displaced particles behind, and a reset that failed to clear them would
    hand the next arm a corrupted billet while every downstream metric still computed happily.
    """
    P = adapter.to_mesh(state).vertices
    if P is None or len(P) == 0:
        return "no vertices after reset"
    if not np.all(np.isfinite(P)):
        return "non-finite vertices after reset (%d bad)" % int((~np.isfinite(P)).sum())
    if expect_n is not None and len(P) != expect_n:
        return "particle count changed: %d -> %d" % (expect_n, len(P))
    span = float(np.ptp(P[:, 2])) if P.shape[1] > 2 else 0.0
    if expect_span_mm is not None and abs(span - expect_span_mm) > 0.05 * max(expect_span_mm, 1e-9):
        return "fresh-bar extent drifted: %.3f -> %.3f" % (expect_span_mm, span)
    return None


def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-hits", type=int, default=3)
    ap.add_argument("--arms", default="", help="comma-separated tags; default all")
    ap.add_argument("--out", default="batch",
                    help="subdirectory of outputs/ to write to; use a distinct name per sweep so "
                         "a longer run does not overwrite a shorter one's results")
    ap.add_argument("--reps", type=int, default=1,
                    help="repeats per arm, tagged _r1.._rN. n=1 cannot distinguish a real arm "
                         "difference from this sim's known run-to-run spread")
    args = ap.parse_args()
    OUT = os.path.expanduser("~/GitHub/Genesis/forge_common/main/outputs/%s" % args.out)

    if os.environ.get("AGF_CONTACT_RUNTIME_SWITCH") != "1":
        print("ERROR: this driver requires AGF_CONTACT_RUNTIME_SWITCH=1 -- without it the contact "
              "mode is baked into the kernels at scene build and every arm after the first would "
              "silently run the FIRST arm's configuration.")
        return 2

    os.makedirs(OUT, exist_ok=True)

    from forge_common.adapter_build import build_adapter
    from forge_common.real_data import load_real_hits_for_sim
    from forge_common.real_scale import REAL_STOCK_RADIUS_MM, REAL_STOCK_LENGTH_MM

    # AGF_STOCK_RADIUS_MM overrides the seeded billet radius WITHOUT touching shared
    # forge_common. REAL_STOCK_RADIUS_MM = 20.0 is a bare constant there with no derivation;
    # measured, it is the BOUNDING BOX of the hit-1 scan, so it seeds +10.9% too much
    # material. Useful values at L = 59 mm:
    #     20.000  as-shipped, bounding box
    #     19.635  mid-body equivalent diameter, where the dies actually strike
    #     18.987  volume-matched to the hit-1-before scan (66,825 mm^3)
    # Grid sizing follows cylinder_diameter, so changing this ALSO moves dx and the timestep
    # unless AGF_CELLS_PER_DIAMETER is rescaled to compensate -- see the note where it is used.
    _stock_r = float(os.environ.get("AGF_STOCK_RADIUS_MM", REAL_STOCK_RADIUS_MM))

    wanted = [a.strip() for a in args.arms.split(",") if a.strip()]
    arms = [a for a in ARMS if not wanted or a["tag"] in wanted]

    # Repeats are REP-MAJOR (all arms at r1, then all at r2, ...) rather than grouped per arm.
    # This sim is not bitwise deterministic and arms sit near a stability cliff, so a single run
    # cannot tell "this arm is worse" from "this seed died early". Interleaving also means a
    # drift over the life of the process shows up as spread WITHIN an arm instead of masquerading
    # as a difference BETWEEN arms, and an early abort still leaves every arm covered at r1.
    if args.reps > 1:
        arms = [dict(a, tag="%s_r%d" % (a["tag"], r))
                for r in range(1, args.reps + 1) for a in arms]

    hits = load_real_hits_for_sim("genesis", args.n_hits)
    adapter = build_adapter("genesis")

    t_start = time.time()
    results = []
    expect_n = expect_span = None

    for i, arm in enumerate(arms, 1):
        tag = arm["tag"]
        t0 = time.time()
        print("\n" + "=" * 78)
        print("ARM %d/%d  %s   (t+%.0fs)" % (i, len(arms), tag, time.time() - t_start))
        print("=" * 78)

        # init_stock rebuilds on the first call and resets on every later one.
        state = adapter.init_stock(radius_mm=_stock_r, length_mm=REAL_STOCK_LENGTH_MM)
        t_setup = time.time() - t0

        if expect_n is None:
            P0 = adapter.to_mesh(state).vertices
            expect_n = len(P0)
            expect_span = float(np.ptp(P0[:, 2]))
            # Save the genuine t=0 state. Without it, eta drop has to be referenced to the
            # no-contact control's FINAL state, which is only a stand-in and quietly assumes the
            # control really did nothing. Cheap to store, removes the assumption.
            np.savez_compressed(os.path.join(OUT, "_ref_fresh_verts.npz"),
                                verts=P0.astype(np.float32))
            print("  baseline fresh bar: %d verts, z-span %.3f (saved as _ref_fresh)"
                  % (expect_n, expect_span))
        else:
            bad = check_fresh(adapter, state, expect_n, expect_span, tag)
            if bad is not None:
                print("  ABORT: reset did not produce a clean bar -- %s" % bad)
                print("  Remaining arms would run on contaminated state; stopping instead.")
                results.append({"tag": tag, "status": "aborted_dirty_reset", "reason": bad})
                break

        coupler = state.env.scene.sim.coupler
        cfg = configure(coupler, arm)
        print("  config: %s" % json.dumps(cfg, sort_keys=True))

        # Route this arm's per-strike diagnostics to its own file.
        ctrl = state.controller
        ctrl._diag_out = os.path.join(OUT, "%s.diag.jsonl" % tag)
        ctrl._diag_strike_idx = 0
        ctrl._diag_acc = None
        if os.path.exists(ctrl._diag_out):
            os.remove(ctrl._diag_out)

        status, n_done, err = "completed", 0, None
        per_hit = {}
        for h in hits:
            try:
                state = adapter.apply_hit(state, h)
                n_done += 1
                # Keep the cloud at every hit, not just the last. Geometry has to be scored at a
                # FIXED hit index to be comparable across arms that reach different depths, and
                # an arm that dies at hit 3 otherwise contributes nothing at all.
                Ph = adapter.to_mesh(state).vertices
                if Ph is not None and len(Ph):
                    per_hit["hit_%02d" % n_done] = np.asarray(Ph, dtype=np.float32)
            except Exception as exc:                     # instability raises; record and move on
                status, err = "failed_at_hit_%d" % (n_done + 1), repr(exc)[:300]
                print("  !! %s: %s" % (status, err))
                break
        if per_hit:
            np.savez_compressed(os.path.join(OUT, "%s_hits.npz" % tag), **per_hit)

        P = adapter.to_mesh(state).vertices
        finite = np.all(np.isfinite(P)) if P is not None and len(P) else False
        if P is not None and len(P) and finite:
            np.savez_compressed(os.path.join(OUT, "%s_verts.npz" % tag), verts=P.astype(np.float32))

        row = {
            "tag": tag, "status": status, "hits_done": n_done, "error": err,
            "config": cfg, "setup_s": round(t_setup, 1), "arm_s": round(time.time() - t0, 1),
            "n_verts": int(len(P)) if P is not None else 0, "verts_finite": bool(finite),
        }
        if P is not None and len(P) and finite:
            row["span_mm"] = [round(float(np.ptp(P[:, k])), 3) for k in range(P.shape[1])]
        results.append(row)
        print("  %s  hits=%d  setup=%.1fs  arm=%.1fs" % (status, n_done, t_setup, row["arm_s"]))

        with open(os.path.join(OUT, "summary.json"), "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2)

        # Record what scoring needs but cannot infer from a particle cloud. psize is read
        # STRAIGHT OFF THE ENTITY rather than derived, because every derivation of it in
        # this tree assumed a cylinder IC and silently breaks under AGF_BILLET_MESH.
        try:
            _ent = state.env.mpm_entity
            _meta = {
                "psize_mm": float(_ent.particle_size) * 1000.0,
                "n_particles": int(_ent.n_particles),
                "billet_mesh": os.environ.get("AGF_BILLET_MESH") or None,
                # These fall back to LITERALS, so they re-guess the configuration rather than
                # observing it: with no AGF_ var set they record whatever is written here, not
                # what the sim used. Keep them equal to the defaults in options.py or run_meta
                # will confidently describe a run that did not happen. (This is why the batches
                # of 2026-08-20 needed a hand-written RUN_PROVENANCE.txt alongside run_meta.)
                "cells_per_diameter": os.environ.get("AGF_CELLS_PER_DIAMETER", "10"),
                "ppc_divisor": os.environ.get("AGF_PPC_DIVISOR", "2.0"),
                "approach_cfl_ratio": os.environ.get("AGF_APPROACH_CFL_RATIO", "0.05"),
                "max_particle_velocity": os.environ.get("AGF_MAX_PARTICLE_VELOCITY", "100.0"),
            }
            with open(os.path.join(OUT, "run_meta.json"), "w", encoding="utf-8") as fh:
                json.dump(_meta, fh, indent=2)
        except Exception as _e:  # never let bookkeeping kill a sweep
            print("  (run_meta not written: %s)" % _e)

    print("\n" + "=" * 78)
    print("SWEEP DONE  %d arms  %.1f min total" % (len(results), (time.time() - t_start) / 60.0))
    print("=" * 78)
    print("%-22s %-20s %6s %8s %8s" % ("arm", "status", "hits", "setup_s", "arm_s"))
    for r in results:
        print("%-22s %-20s %6s %8s %8s" % (
            r["tag"], r["status"], r.get("hits_done", "-"),
            r.get("setup_s", "-"), r.get("arm_s", "-")))
    print("\nresults in %s" % OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
