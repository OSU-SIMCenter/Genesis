"""Score MATERIAL choices on GEOMETRY instead of force.

Force cannot discriminate a flow rule in this sim (a 10% flow-stress change moves
peak force 1.1%; a purely elastic change moved it 10%). Shape can. One material
configuration per process -- material selection is compile-time, baked into the
kernel at scene build, so arms CANNOT be batched the way contact arms can.

The arm that matters is m2 vs m3: identical except the Arrhenius temperature
ceiling (1273.15 = the old clamp, 1473.15 = what b8854afc shipped). At a 1473 K
billet those are a 2.33x flow-stress difference (181.1 vs 77.8 MPa, CPU
reference), so if shape cannot separate them that is a real negative result
about a change shipped on literature grounds alone.

=========================== HOW THE OVERRIDES WORK ===========================
The first version of this script patched pydantic FIELD DEFAULTS and silently
did nothing -- all four arms built JohnsonCookPlasticity. Three distinct traps,
all worth knowing before touching agforge config from outside:

  1. `TeleopOptions.mat` is declared `mat: MaterialOptions = MaterialOptions()`
     -- a SHARED INSTANCE built at import time. Patching the MaterialOptions
     field default never reaches it. Must mutate that instance.

  2. `default_initial_temperature=293.0` is passed as an EXPLICIT KWARG at the
     MPMOptions construction site inside TeleopOptions.model_post_init, so a
     field-default patch loses to it. Must fix up cfg.mpm after post-init.

  3. `enable_particle_contact` is NOT passed explicitly, so for that one a
     field-default patch is the correct mechanism.

Because of 1-3, this script now VERIFIES against the built scene and aborts if
what got built is not what was asked for. Reporting the request rather than the
result is what made the first run look like a physics finding.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

OUTDIR = os.path.expanduser("~/GitHub/Genesis/forge_common/main/outputs/material_arms")

# tag -> (use_arrhenius, billet_K, T_fit_max or None)
ARMS = {
    "m0_jc_293":      (False, 293.0,   None),
    "m1_jc_1273":     (False, 1273.15, None),
    "m2_arr_clamped": (True,  1473.15, 1273.15),
    "m3_arr_raised":  (True,  1473.15, 1473.15),

    # --- billet-temperature identification, 2026-08-11 ---
    # The real sequence temperature is known only as "900 or 1000, C or K".
    # Johnson-Cook CANNOT discriminate (jc_T_ref = 1273.15, so T* clamps to zero
    # at or below every candidate), so these are all Arrhenius. And Arrhenius
    # clamps to [1073.15, 1473.15], so 900 K and 1000 K BOTH land on 1073.15 and
    # are the same experiment -- four candidates, three distinct arms.
    # Ceiling is held at the shipped 1473.15 throughout so the only variable is
    # the billet temperature.
    "t_K_either":  (True, 1000.0,  1473.15),   # 900 K or 1000 K -> clamps to 1073.15
    "t_900C":      (True, 1173.15, 1473.15),
    "t_1000C":     (True, 1273.15, 1473.15),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=sorted(ARMS))
    ap.add_argument("--n-hits", type=int, default=1)
    # All three Arrhenius temperature arms died at hit 2 with "Thermal Detonation"
    # (which also fires on temp < 0, i.e. numerical blowup, not overheating).
    # Hypothesis: enable_thermal is on and thermal_time_scale ~ 171,653 runs
    # radiation ~97x fast; on a HOT billet radiation goes as T^4, so the surface
    # field can crash. 'mechanical' ties the thermal clock to the press instead.
    ap.add_argument("--thermal-mode", default=None, choices=["cfl", "mechanical"])
    ap.add_argument("--tag-suffix", default="")
    # Grid resolution. This branch defaults to 7 cells across the billet; the
    # contact workstream runs 10 (via their AGF_CELLS_PER_DIAMETER, which is not
    # committed anywhere). That is a 2.48 vs 3.37 cell resolution of the 13.48 mm
    # die contact band, and it is the leading suspect for why sequences die here
    # but complete for them.
    ap.add_argument("--cells", type=int, default=None)
    # Johnson-Cook is temperature-BLIND mechanically (T* clamps at jc_T_ref), yet a
    # hot billet still dies at hit 2 under JC. That points at the THERMAL solver,
    # not the mechanics. A press is frozen isothermal anyway (StrikeController
    # snapshots/restores temps around every step), so for pure forging work the
    # thermal solve may be dead weight that only contributes instability.
    ap.add_argument("--no-thermal", action="store_true")
    # Override the arm's billet temperature. Added to test whether the hot-billet
    # instability tracks jc_T_ref (1273.15 K) rather than "hotness": below T_ref the
    # Johnson-Cook thermal_softening term is clamped off entirely, so the
    # adiabatic-heating -> softening -> more-plastic-work feedback cannot start.
    ap.add_argument("--billet-k", type=float, default=None)
    # Thermal FLIP/PIC blend for the temperature G2P. Default 0.97 (solver's own).
    # 0.0 = pure PIC, which is unconditionally stable for a diffusion field.
    # Poked into the RUNTIME field after build -- no source change, no recompile.
    ap.add_argument("--flip-frac", type=float, default=None)
    # Override the parametric approach speed (options.py:940 gives 273.2 m/s, vs a
    # 100 m/s stability intercept). The contact workstream found 35 m/s takes their
    # arms to 17/17.
    ap.add_argument("--approach-speed", type=float, default=None)
    args = ap.parse_args()

    use_arr, billet_k, t_fit_max = ARMS[args.arm]
    if args.billet_k is not None:
        billet_k = float(args.billet_k)
    tag = args.arm + args.tag_suffix
    os.makedirs(OUTDIR, exist_ok=True)

    from agforge.options import TeleopOptions, MPMOptions
    from agforge import materials as mat_mod

    # (1) mutate the shared MaterialOptions instance
    shared_mat = TeleopOptions.model_fields["mat"].default
    if shared_mat is None or not hasattr(shared_mat, "use_arrhenius"):
        raise SystemExit("TeleopOptions.mat is no longer a shared instance -- re-read the "
                         "module before assuming how to override it")
    shared_mat.use_arrhenius = use_arr
    if use_arr:
        # The sim presses 1773x too fast (656 /s vs the real blow's 0.41), so without
        # this every Arrhenius arm is evaluated far outside its fit domain and the
        # comparison measures the rate error rather than the ceiling.
        shared_mat.arrhenius_process_strain_rate = 0.41

    # (2) beat the explicit default_initial_temperature kwarg
    _orig_post_init = TeleopOptions.model_post_init

    def _post_init(self, __context):
        _orig_post_init(self, __context)
        self.mpm.default_initial_temperature = billet_k
        if args.approach_speed is not None:
            self.strike.approach_speed = float(args.approach_speed)
        if args.no_thermal:
            self.mpm.enable_thermal = False

    TeleopOptions.model_post_init = _post_init

    # (3) grid-only contact; this one IS a field default
    MPMOptions.model_fields["enable_particle_contact"].default = False
    MPMOptions.model_rebuild(force=True)

    # (4) optional thermal-clock mode. 'mechanical' ties S_T to the press speed
    # instead of the diffusion CFL ceiling -- d982104d.
    if args.thermal_mode is not None:
        if "thermal_time_scale_mode" not in TeleopOptions.model_fields:
            raise SystemExit("no thermal_time_scale_mode field -- API drifted")
        TeleopOptions.model_fields["thermal_time_scale_mode"].default = args.thermal_mode
        TeleopOptions.model_rebuild(force=True)

    # (5) optional grid resolution
    if args.cells is not None:
        from agforge.options import RobotOptions
        if "grid_cells_across_billet" not in RobotOptions.model_fields:
            raise SystemExit("no grid_cells_across_billet field -- API drifted")
        RobotOptions.model_fields["grid_cells_across_billet"].default = args.cells
        RobotOptions.model_rebuild(force=True)

    if t_fit_max is not None:
        mat_mod.ArrheniusPlasticity.model_fields["T_fit_max"].default = t_fit_max
        mat_mod.ArrheniusPlasticity.model_rebuild(force=True)

    from forge_common.adapter_build import build_adapter
    from forge_common.real_data import load_real_hits_for_sim
    from forge_common.real_scale import REAL_STOCK_RADIUS_MM, REAL_STOCK_LENGTH_MM

    t0 = time.time()
    hits = load_real_hits_for_sim("genesis", args.n_hits)
    adapter = build_adapter("genesis")
    state = adapter.init_stock(radius_mm=REAL_STOCK_RADIUS_MM,
                               length_mm=REAL_STOCK_LENGTH_MM)
    t_setup = time.time() - t0

    # ---- VERIFY THE BUILT SCENE, and abort rather than produce a pretty artifact ----
    ent = getattr(state.env, "mpm_entity", None)
    built = getattr(ent, "material", None)
    if built is None:
        raise SystemExit("cannot reach the built material via env.mpm_entity.material -- "
                         "refusing to run an experiment I cannot verify")
    got_cls = type(built).__name__
    want_cls = "ArrheniusPlasticity" if use_arr else "JohnsonCookPlasticity"
    if got_cls != want_cls:
        raise SystemExit("MATERIAL MISMATCH: asked for %s, scene built %s" % (want_cls, got_cls))
    if t_fit_max is not None:
        got_tmax = float(getattr(built, "T_fit_max", float("nan")))
        if abs(got_tmax - t_fit_max) > 1e-6:
            raise SystemExit("T_fit_max MISMATCH: asked %.2f, built %.2f" % (t_fit_max, got_tmax))

    # Billet temperature, checked at BOTH places that matter and with no getattr
    # fallback -- a default would let an unverifiable run look verified, which is
    # exactly how the first version of this experiment produced four identical
    # arms that were all secretly Johnson-Cook.
    cfg_temp = float(state.env.cfg.mpm.default_initial_temperature)  # StrikeController reads this
    mpm_solver = next((s for s in state.env.scene.sim.solvers
                       if hasattr(s, "_default_initial_temperature")), None)
    if mpm_solver is None:
        raise SystemExit("cannot locate the MPM solver -- refusing to run unverified")
    solver_temp = float(mpm_solver._default_initial_temperature)
    if abs(cfg_temp - billet_k) > 1e-6 or abs(solver_temp - billet_k) > 1e-6:
        raise SystemExit("BILLET TEMP MISMATCH: asked %.2f K, cfg %.2f, solver %.2f"
                         % (billet_k, cfg_temp, solver_temp))

    if not hasattr(mpm_solver, "_enable_particle_contact"):
        raise SystemExit("MPM solver has no _enable_particle_contact -- cannot confirm "
                         "contact mode, refusing to run")
    pc = bool(mpm_solver._enable_particle_contact)
    if pc:
        raise SystemExit("CONTACT MISMATCH: particle/teleport contact is still ON; this "
                         "experiment must be grid-only or it confounds material with contact")

    # --- approach-speed read-back ---
    approach_used = float(state.env.cfg.strike.approach_speed)
    if args.approach_speed is not None:
        if abs(approach_used - args.approach_speed) > 1e-6:
            raise SystemExit("APPROACH SPEED MISMATCH: asked %.3f m/s, cfg reads %.3f"
                             % (args.approach_speed, approach_used))
        print("[verify] approach_speed = %.3f m/s" % approach_used)

    # --- thermal FLIP/PIC override, with read-back (runtime field, no recompile) ---
    flip_used = None
    if args.flip_frac is not None:
        if not hasattr(mpm_solver, "_rt_thermal_flip_frac"):
            raise SystemExit("MPM solver has no _rt_thermal_flip_frac -- cannot set the "
                             "thermal FLIP fraction, refusing to run a mislabelled arm")
        before = float(mpm_solver._rt_thermal_flip_frac[None])
        mpm_solver._rt_thermal_flip_frac[None] = float(args.flip_frac)
        after = float(mpm_solver._rt_thermal_flip_frac[None])
        # The field is gs.qd_float and Genesis runs precision 32, so what comes
        # back is the float32 nearest the request -- 0.97 stores as
        # 0.9700000286..., 2.9e-8 out. A 1e-9 tolerance is therefore satisfiable
        # ONLY for values exactly representable in float32 (0.0, 0.5, 0.25...),
        # and 0.97 aborts every time. That is why the 0.97 arms on disk record
        # thermal_flip_frac: null -- the flag was dropped rather than the guard
        # relaxed, so those arms ran on the solver default unverified.
        # 1e-6 is ~8x float32 eps near 1.0 and orders below anything physical.
        if abs(after - args.flip_frac) > 1e-6:
            raise SystemExit("FLIP MISMATCH: asked %.6f, field reads %.6f"
                             % (args.flip_frac, after))
        flip_used = after
        print("[verify] thermal FLIP fraction %.4f -> %.4f" % (before, after))

    n_particles = int(len(adapter.to_mesh(state).vertices))
    if args.cells is not None:
        got_cells = int(state.env.cfg.robot.grid_cells_across_billet)
        if got_cells != args.cells:
            raise SystemExit("RESOLUTION MISMATCH: asked %d cells, built %d"
                             % (args.cells, got_cells))

    got_thermal = bool(state.env.cfg.mpm.enable_thermal)
    if args.no_thermal and got_thermal:
        raise SystemExit("THERMAL MISMATCH: asked to disable thermal, it is still on")

    observed = {"material_class": got_cls, "n_particles": n_particles,
                "enable_thermal": got_thermal,
                "cells_across_billet": int(state.env.cfg.robot.grid_cells_across_billet),
                "T_fit_max": float(getattr(built, "T_fit_max", float("nan")))
                if use_arr else None,
                "billet_K_cfg": cfg_temp, "billet_K_solver": solver_temp,
                "enable_particle_contact": pc,
                "thermal_flip_frac": flip_used,
                "approach_speed": approach_used}
    print("[verify] OK -- %s" % json.dumps(observed))

    status, n_done, err = "completed", 0, None
    per_hit = {}
    for h in hits:
        try:
            state = adapter.apply_hit(state, h)
            n_done += 1
            P = adapter.to_mesh(state).vertices
            if P is not None and len(P):
                per_hit["hit_%02d" % n_done] = np.asarray(P, dtype=np.float32)
        except Exception as exc:
            status, err = "failed_at_hit_%d" % (n_done + 1), repr(exc)[:300]
            print("  !! %s: %s" % (status, err))
            break

    if per_hit:
        np.savez_compressed(os.path.join(OUTDIR, "%s_hits.npz" % tag), **per_hit)
    P = adapter.to_mesh(state).vertices
    finite = bool(P is not None and len(P) and np.all(np.isfinite(P)))
    if finite:
        np.savez_compressed(os.path.join(OUTDIR, "%s_verts.npz" % tag),
                            verts=P.astype(np.float32))

    row = {"tag": tag, "status": status, "hits_done": n_done, "error": err,
           "requested": {"use_arrhenius": use_arr, "billet_K": billet_k,
                         "T_fit_max": t_fit_max},
           "thermal_mode": args.thermal_mode, "observed": observed, "setup_s": round(t_setup, 1),
           "total_s": round(time.time() - t0, 1),
           "n_verts": int(len(P)) if P is not None else 0, "verts_finite": finite}
    with open(os.path.join(OUTDIR, "%s.summary.json" % tag), "w") as f:
        json.dump(row, f, indent=2)
    print(json.dumps(row, indent=2))
    return 0 if finite else 1


if __name__ == "__main__":
    sys.exit(main())
