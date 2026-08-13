#!/usr/bin/env python3
"""Pin down WHICH failure "Thermal Detonation" actually is.

The label covers two opposite failures. options.py:799-800 fires the intercept on
temp > 4000 K OR temp < 0 K. Negative temperature on a diffusion field is classic
FLIP null-space undershoot -- the gather re-injecting a residual the grid solve
just smoothed away. Above 4000 K is runaway. They have different fixes, and the
coupling document explicitly refuses to guess between them.

The forensic trace in strike_controller.py cannot answer it. It prints the
offending particle's temperature, but it locates that particle through
_per_particle_field_cpu, which at line 35 does

    t.reshape(-1, n_particles)[0]

i.e. keeps only the FIRST frame buffer, while the trigger at line 1657 tests the
whole tensor. Genesis double-buffers particle state -- base_mpm_solver.py:745,
the FLIP blend itself, writes to particles[f + 1]. So a detonation in the newly
gathered buffer fires the intercept and then vanishes from the trace, leaving
bad_indices empty and no GROUND ZERO block. That is exactly what a live run does.

This reads solver.particles.temp directly, across every frame buffer, and reports
which side of the limit was crossed. It changes no shared source -- the thermal
solver belongs to session B-3 and this worktree is shared with it.
"""
import os
import sys
import numpy as np

BILLET_K = 1273.15
N_HITS = 3


def report(tag, T):
    print(f"\n--- {tag} ---")
    print(f"shape {T.shape}  dtype {T.dtype}")
    print(f"global  min {np.nanmin(T):14.4f}   max {np.nanmax(T):14.4f}   "
          f"nan {int(np.isnan(T).sum())}")
    lead = T.shape[0] if T.ndim >= 2 else 1
    for f in range(lead):
        Tf = T[f] if T.ndim >= 2 else T
        below = int((Tf < 0.0).sum())
        above = int((Tf > 4000.0).sum())
        nan = int(np.isnan(Tf).sum())
        flag = ""
        if below:
            flag += "  <== NEGATIVE (FLIP undershoot signature)"
        if above:
            flag += "  <== >4000 K (runaway signature)"
        print(f"  frame {f}: min {np.nanmin(Tf):13.4f}  max {np.nanmax(Tf):13.4f}  "
              f"n<0 {below:5d}  n>4000 {above:5d}  nan {nan:5d}{flag}")
        if below or above:
            bad = np.where((Tf < 0.0) | (Tf > 4000.0))
            idx = [int(a[0]) for a in bad]
            print(f"           first offender at index {idx}, value "
                  f"{float(Tf[tuple(a[0] for a in bad)]):.4f} K")
            vals = Tf[(Tf < 0.0) | (Tf > 4000.0)]
            print(f"           offending values: min {float(vals.min()):.4f}  "
                  f"max {float(vals.max()):.4f}  n {vals.size}")


def main():
    from agforge.options import TeleopOptions, MPMOptions

    shared_mat = TeleopOptions.model_fields["mat"].default
    if shared_mat is None or not hasattr(shared_mat, "use_arrhenius"):
        raise SystemExit("TeleopOptions.mat is no longer a shared instance")
    shared_mat.use_arrhenius = False

    _orig = TeleopOptions.model_post_init

    def _post_init(self, __context):
        _orig(self, __context)
        self.mpm.default_initial_temperature = BILLET_K

    TeleopOptions.model_post_init = _post_init
    MPMOptions.model_fields["enable_particle_contact"].default = False
    MPMOptions.model_rebuild(force=True)

    from forge_common.adapter_build import build_adapter
    from forge_common.real_data import load_real_hits_for_sim
    from forge_common.real_scale import REAL_STOCK_RADIUS_MM, REAL_STOCK_LENGTH_MM

    hits = load_real_hits_for_sim("genesis", N_HITS)
    adapter = build_adapter("genesis")
    state = adapter.init_stock(radius_mm=REAL_STOCK_RADIUS_MM,
                               length_mm=REAL_STOCK_LENGTH_MM)

    solver = next((s for s in state.env.scene.sim.solvers
                   if hasattr(s, "_default_initial_temperature")), None)
    if solver is None:
        raise SystemExit("cannot locate MPM solver")
    if abs(float(solver._default_initial_temperature) - BILLET_K) > 1e-6:
        raise SystemExit("billet temp mismatch")
    flip = float(solver._rt_thermal_flip_frac[None])
    print(f"[verify] billet {BILLET_K} K, flip {flip!r}, "
          f"thermal {bool(state.env.cfg.mpm.enable_thermal)}, "
          f"particle_contact {bool(solver._enable_particle_contact)}")

    T0 = solver.particles.temp.to_numpy()
    report("BEFORE ANY HIT", T0)

    for n, h in enumerate(hits, 1):
        try:
            state = adapter.apply_hit(state, h)
            T = solver.particles.temp.to_numpy()
            report(f"AFTER HIT {n} (survived)", T)
        except Exception as exc:
            print(f"\n!!! hit {n} FAILED: {exc!r}")
            T = solver.particles.temp.to_numpy()
            report(f"AT FAILURE (hit {n})", T)
            print("\n" + "=" * 70)
            print("VERDICT")
            print("=" * 70)
            lead = T.shape[0] if T.ndim >= 2 else 1
            neg = any(int((T[f] < 0.0).sum()) for f in range(lead))
            hot = any(int((T[f] > 4000.0).sum()) for f in range(lead))
            nan = bool(np.isnan(T).any())
            if neg and not hot:
                print("NEGATIVE TEMPERATURE ONLY -> undershoot, not runaway.")
            elif hot and not neg:
                print("ABOVE 4000 K ONLY -> runaway, not undershoot.")
            elif neg and hot:
                print("BOTH SIDES present.")
            elif nan:
                print("NaN present but neither limit crossed in this snapshot.")
            else:
                print("NEITHER limit crossed in this snapshot -- the offending value "
                      "did not survive to the point this probe reads it.")
            break

    return 0


if __name__ == "__main__":
    sys.exit(main())
