#!/usr/bin/env python3
"""Live check of forge_energy.EnergyMonitor against a real scene.

The module's own smoke test only exercises the numeric core. This runs the
monitor against an actual built entity, which is where the assumptions that
matter can fail: that get_particles_vel returns what is expected, that the
mass self-check passes against real solver state, that dT_adiabatic is
populated at all in the batch path, and that criterion 4 produces a number
rather than a None.

Sampling is done by wrapping scene.step, so samples land exactly at
macro-step boundaries -- which is where the monitor is designed to read, and
which is the only place a 'frozen' press is actually isothermal. Sampling only
at hit boundaries would read KE at rest, i.e. near zero, and would let
criterion 4 pass for the wrong reason.

Setup mirrors material_arms.py's verified path, including its abort-on-mismatch
contract: an experiment that cannot be verified is not run.
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-hits", type=int, default=1)
    ap.add_argument("--cells", type=int, default=None)
    ap.add_argument("--billet-k", type=float, default=1233.15)
    ap.add_argument("--sample-every", type=int, default=1)
    ap.add_argument("--out", default=None)
    ap.add_argument("--diag-out", default=None,
                    help="per-macro-step energy JSONL (streamed, survives a crash)")
    ap.add_argument("--pressing-speed", type=float, default=None)
    # Thermal FLIP/PIC blend for the temperature G2P (runtime field, no
    # recompile). 0.0 = pure PIC, unconditionally stable for a diffusion field.
    #
    # Safe for THIS measurement, and it is worth saying why rather than just
    # doing it. The blend writes only particles.temp, from the diffusion /
    # convection / radiation / bulk channels. dT_adiabatic is accumulated
    # separately in p2g_post_constitutive and is NOT part of the blend, so the
    # plastic channel that internal energy is derived from is untouched. And on
    # the Johnson-Cook path the flow stress is temperature-blind below
    # jc_T_ref = 1273.15 K, so as long as the billet stays under that, the
    # temperature field cannot reach the mechanics at all. The run reports
    # T_max so that assumption is checkable rather than assumed.
    ap.add_argument("--flip-frac", type=float, default=None)
    # Raise (or effectively disable) the Max Force stop.
    #
    # This branch has no AGF_MAX_FORCE override and material_arms.py has no way
    # to set it, which is what workstream B identified as blocking D1a. Without
    # it every hit terminates wherever the force profile crosses 200 kN, and
    # that crossing point moves with press speed -- so a speed sweep compares
    # hits that stopped at DIFFERENT strains and measures the stop, not the
    # thing being swept. Confirmed live: at 25/12.5/6.25 m/s the same hit
    # stopped at strain 0.2160 / 0.2243 / 0.2140 against a 0.2484 target, all
    # on Max Force. Set this high to compare like with like.
    ap.add_argument("--max-force", type=float, default=None)
    # Die-balance loop gain. The stall bound is
    # sqrt(pressing_speed * 20000 / gain), so LOWERING the gain RAISES the
    # bound and suppresses stalling.
    #
    # This exists because of a confound found in this monitor's own output:
    # 72-93% of a run's total kinetic energy lands in a ~15-step burst that
    # co-occurs with a die-stall episode. That makes KE/IE a measurement of the
    # stall transient as much as of inertia, and the two cannot be separated at
    # the shipped gain. Re-running with stalling suppressed is what separates
    # them. 1.5e-4 is the shipped value; 1.5e-5 raises the bound ~3.2x.
    ap.add_argument("--force-balance-gain", type=float, default=None)
    args = ap.parse_args()

    t0 = time.time()

    from agforge.options import TeleopOptions

    shared_mat = TeleopOptions.model_fields["mat"].default
    if shared_mat is None or not hasattr(shared_mat, "use_arrhenius"):
        raise SystemExit("TeleopOptions.mat is no longer a shared instance -- "
                         "re-read the module before assuming how to override it")
    # Johnson-Cook, the DEFAULT flow rule (options.py use_johnson_cook = True).
    # Deliberately the default: the point is to instrument what production runs.
    shared_mat.use_arrhenius = False

    _orig_post_init = TeleopOptions.model_post_init

    def _post_init(self, __context):
        _orig_post_init(self, __context)
        self.mpm.default_initial_temperature = float(args.billet_k)
        if args.cells is not None:
            self.robot.grid_cells_across_billet = int(args.cells)
        if args.pressing_speed is not None:
            self.strike.pressing_speed = float(args.pressing_speed)
        if args.max_force is not None:
            self.strike.max_force = float(args.max_force)
        if args.force_balance_gain is not None:
            self.strike.force_balance_gain = float(args.force_balance_gain)

    TeleopOptions.model_post_init = _post_init

    # Same import path material_arms.py uses. The stock dimensions are imported
    # rather than written down, so this cannot silently drift from the arm runs
    # it is meant to be comparable with.
    from forge_common.adapter_build import build_adapter
    from forge_common.real_data import load_real_hits_for_sim
    from forge_common.real_scale import (REAL_STOCK_RADIUS_MM,
                                         REAL_STOCK_LENGTH_MM)

    hits = load_real_hits_for_sim("genesis", args.n_hits)
    adapter = build_adapter("genesis")
    state = adapter.init_stock(radius_mm=REAL_STOCK_RADIUS_MM,
                               length_mm=REAL_STOCK_LENGTH_MM)
    t_setup = time.time() - t0
    print("[setup] %.1f s" % t_setup, flush=True)

    env = state.env
    ent = getattr(env, "mpm_entity", None)
    if ent is None:
        raise SystemExit("no mpm_entity on the built env -- refusing to measure "
                         "an entity I cannot reach")

    solver = next((s for s in env.scene.sim.solvers
                   if hasattr(s, "_particle_volume_scale")), None)
    if solver is None:
        raise SystemExit("cannot locate the MPM solver -- refusing to run unverified")

    print("[verify] material  = %s" % type(ent.material).__name__, flush=True)
    print("[verify] enable_thermal = %s" % getattr(solver, "_enable_thermal", None),
          flush=True)
    print("[verify] n_particles = %d" % int(ent.n_particles), flush=True)

    if args.flip_frac is not None:
        if not hasattr(solver, "_rt_thermal_flip_frac"):
            raise SystemExit("MPM solver has no _rt_thermal_flip_frac -- refusing "
                             "to run a mislabelled arm")
        before = float(solver._rt_thermal_flip_frac[None])
        solver._rt_thermal_flip_frac[None] = float(args.flip_frac)
        after = float(solver._rt_thermal_flip_frac[None])
        # float32 storage: only exactly-representable values survive a tight
        # read-back. 0.0 is fine; 0.97 is not, which is why arms on disk that
        # asked for 0.97 recorded null and silently ran the solver default.
        if abs(after - args.flip_frac) > 1e-6:
            raise SystemExit("FLIP MISMATCH: asked %.6f, field reads %.6f"
                             % (args.flip_frac, after))
        print("[verify] thermal FLIP fraction %.4f -> %.4f" % (before, after), flush=True)

    import forge_energy

    # Pass the built config so the mass check compares against the CONFIGURED
    # density rather than a remembered constant. The card ships rho = 7334
    # (NIST2021 SRM 1155a at 1273.15 K), which is neither the room-temperature
    # 8000 it replaced nor any handbook 316L figure -- checking against a
    # plausible-looking literature value produces a false alarm.
    mon = forge_energy.EnergyMonitor(ent, solver, cfg=env.cfg,
                                     jsonl_path=args.diag_out)
    print("[verify] rho_expected = %.1f (%s)" % (mon.rho_expected, mon.rho_source),
          flush=True)
    print("[verify] mass self-check: %s" % json.dumps(mon.check), flush=True)
    print("[verify] cp source: %s" % mon.cp_source, flush=True)
    print("[verify] elastic source: %s" % mon.elastic_source, flush=True)
    print("[verify] pressing_speed = %.4f m/s" % float(env.cfg.strike.pressing_speed),
          flush=True)
    # Abort-on-mismatch, same contract material_arms.py uses: a run whose config
    # cannot be confirmed is not a measurement.
    if args.pressing_speed is not None:
        got = float(env.cfg.strike.pressing_speed)
        if abs(got - args.pressing_speed) > 1e-6:
            raise SystemExit("PRESSING SPEED MISMATCH: asked %.4f, cfg reads %.4f"
                             % (args.pressing_speed, got))
    if args.max_force is not None:
        got = float(env.cfg.strike.max_force)
        if abs(got - args.max_force) > 1e-3:
            raise SystemExit("MAX FORCE MISMATCH: asked %.1f, cfg reads %.1f"
                             % (args.max_force, got))
        print("[verify] max_force = %.1f N" % got, flush=True)
    gain = float(env.cfg.strike.force_balance_gain)
    if args.force_balance_gain is not None:
        if abs(gain - args.force_balance_gain) / args.force_balance_gain > 1e-6:
            raise SystemExit("BALANCE GAIN MISMATCH: asked %g, cfg reads %g"
                             % (args.force_balance_gain, gain))
    import math as _m
    print("[verify] force_balance_gain = %g  -> stall bound %.0f N at %.3f m/s"
          % (gain, _m.sqrt(float(env.cfg.strike.pressing_speed) * 20000.0 / gain),
             float(env.cfg.strike.pressing_speed)), flush=True)

    # Sample at macro-step boundaries by wrapping scene.step. material_arms.py
    # already establishes monkey-patching as the way this project overrides
    # behaviour without touching shared source.
    # Find the strike controller so each energy record can carry the stop reason
    # it belongs to. Without that, comparing energies across arms silently mixes
    # hits that terminated for different reasons -- which this project has
    # already been burned by once on the geometry side.
    # It lives on the adapter STATE, not on env -- GenesisForgeState holds
    # (env, controller). Scanning env for it silently found nothing and left
    # every stop_reason null, which is the sort of quiet absence that makes an
    # export look complete when it is not.
    ctrl = getattr(state, "controller", None)
    print("[verify] controller reachable: %s" % (type(ctrl).__name__ if ctrl else "NO"),
          flush=True)

    scene = env.scene
    orig_step = scene.step
    counter = {"n": 0, "err": None, "hit": 0}

    def stepping(*a, **kw):
        out = orig_step(*a, **kw)
        counter["n"] += 1
        if counter["n"] % args.sample_every == 0:
            try:
                extra = {}
                if ctrl is not None:
                    extra["stop_reason"] = getattr(ctrl, "_diag_stop_reason", None)
                    st = getattr(ctrl, "strike_state", None)
                    extra["strike_state"] = getattr(st, "name", str(st)) if st else None
                # Passed IN, not attached after: the JSONL line is written
                # inside sample(), so a post-hoc mutation never reaches it.
                mon.sample(step=counter["n"], tag="hit_%02d" % (counter["hit"] + 1),
                           extra=extra)
            except Exception as exc:
                if counter["err"] is None:
                    counter["err"] = repr(exc)[:300]
        return out

    scene.step = stepping

    status = "completed"
    n_done = 0
    try:
        for h in hits:
            state = adapter.apply_hit(state, h)
            n_done += 1
            counter["hit"] = n_done
            print("[hit %d] steps=%d  samples=%d" % (n_done, counter["n"],
                                                     len(mon.samples)), flush=True)
    except Exception as exc:
        status = "failed_at_hit_%d" % (n_done + 1)
        print("  !! %s: %r" % (status, exc), flush=True)
    finally:
        scene.step = orig_step
        mon.close()

    summ = mon.summary()
    summ["status"] = status
    summ["hits_done"] = n_done
    summ["steps"] = counter["n"]
    summ["sample_error"] = counter["err"]
    summ["setup_s"] = round(t_setup, 1)
    summ["total_s"] = round(time.time() - t0, 1)

    print("\n=== ENERGY SUMMARY ===", flush=True)
    print(json.dumps(summ, indent=2, default=str), flush=True)

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"summary": summ, "samples": mon.samples}, f,
                      indent=2, default=str)
        print("\n[wrote] %s" % args.out, flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
