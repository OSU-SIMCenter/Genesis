"""Replay a recorded forge toolpath through the Genesis adapter.

One press per non-no_op record in a ``*_toolpath.jsonl``: the die closes to the
commanded gap, at the commanded angle, at the commanded distance from the chuck
face. Works for any of the recorded series; nothing here is specific to one.

Two conventions in the file format, both applied here:

* ``rho`` is the HALF gap. ``Hit.rho`` is the full die gap, so it is doubled.
* ``phi`` is in degrees. ``Hit.phi`` is radians.

Getting either wrong produces a run that completes and is quietly wrong, so both
are asserted in ``tests/test_toolpath_hits.py`` against a recorded file.

The material is chosen with ``AGF_MATERIAL`` and verified after the scene is
built: the card that was selected must be the card in effect. Running an
aluminium program on the steel card is the failure this guards -- it completes,
and the bar is 27% too strong at yield.

Run one press, which is what to time first:

    PYTHONPATH=forge_common/main \
    AGF_MATERIAL=6063T52 \
    python -u -m agforge.scripts.record_toolpath_hits \
      --jsonl path/to/roll2_toolpath.jsonl --n-hits 1 \
      --length-from-toolpath --hit-dir out/roll2_smoke

Add ``--n-hits 0`` for every press in the file. See ``agforge/README.md`` for the
full sequence and the settings the published runs used.

Was ``record_roll1_hits.py`` before the series were generalised; old logs name it
that way.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve()
_CANDIDATES = [
    _HERE.parents[4] / "forge_common" / "main",
    _HERE.parents[3] / "forge_common" / "main",
]
for _p in _CANDIDATES:
    if (_p / "forge_common").is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
        break

try:
    from forge_common.adapters.genesis_forge_adapter import GenesisForgeAdapter
    from forge_common.hit import Hit
    from forge_common.policies import run_hit_sequence
    from forge_common.press_tool import die_contact_axial_width_mm
except ImportError as e:
    sys.exit(
        """ERROR: forge_common is required and was not importable.

It is a separate repository (OSU-SIMCenter/forge_common). Clone it beside this
one and point PYTHONPATH at its `main` directory:

    PYTHONPATH=/path/to/forge_common/main

Import failed: """
        + str(e)
    )

# forge_common's adapter prepends its own Genesis checkout to sys.path. Put this
# tree back at the front so the options module that builds the scene is this one.
_WORKTREE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_WORKTREE))

from agforge.env_knobs import env_float  # noqa: E402  (after the path fix-ups)

RHO_SCALE = 2.0  # toolpath rho is the half gap; Hit.rho is the full die gap
DEFAULT_RADIUS_MM = 19.05  # 38.1 mm stock, the diameter every recorded series uses

# The adapter moves the slider and hinge to the next pose, and a teleport inside one
# scene.step implies a rigid speed of dx/dt -- at dt ~ 1e-5 s a 100 degree swing is
# thousands of m/s and trips the particle-velocity guard before the press even starts.
# The pose is therefore walked in steps bounded by this speed. 35 m/s is the value
# that completed full sequences; it is a numerical limit, not the real ram speed.
#   AGF_MAX_RIGID_SPEED=20 <cmd>
MAX_RIGID_SPEED_M_S = env_float("AGF_MAX_RIGID_SPEED", 35.0)
# Lever arm turning a hinge angle into an arc length, for bounding that speed only.
JAW_RADIUS_M = 0.06


class ToolpathGenesisAdapter(GenesisForgeAdapter):
    """Open the jaws, then walk the pose so the implied rigid speed stays bounded."""

    def apply_hit(self, state, hit: Hit):
        controller = state.controller
        qpos = self._run(controller.get_qpos())
        opened = qpos.clone()
        opened[0, 2] = controller.gripper_open_pos
        opened[0, 3] = controller.gripper_open_pos
        self._run(controller.set_qpos(opened))
        for _ in range(max(self.settle_steps, 5)):
            self._run(controller.step_simulation())
        qpos = opened

        target = qpos.clone()
        target[0, 0] = state.x_pinned_face_m - float(hit.z) / 1000.0
        target[0, 1] = -float(hit.phi)
        target[0, 2] = controller.gripper_open_pos
        target[0, 3] = controller.gripper_open_pos

        dt = float(state.env.cfg.sim.dt)
        d_slide = abs(float(target[0, 0] - qpos[0, 0]))
        d_arc = abs(float(target[0, 1] - qpos[0, 1])) * JAW_RADIUS_M
        dist = max(d_slide, d_arc, 1e-9)
        n = max(self.settle_steps, int(math.ceil(dist / (MAX_RIGID_SPEED_M_S * dt))))
        print(
            f"[toolpath] pose walk n={n} d_slide={d_slide * 1e3:.2f}mm "
            f"d_arc={d_arc * 1e3:.2f}mm dt={dt:.3e} cap={MAX_RIGID_SPEED_M_S}m/s"
        )
        for i in range(1, n + 1):
            a = i / n
            self._run(controller.set_qpos(qpos + (target - qpos) * a))
            self._run(controller.step_simulation())
        return super().apply_hit(state, hit)


def load_toolpath(jsonl_path: Path, n_hits: int = 0, z_shift_mm: float = 0.0):
    """Presses from a recorded toolpath, plus what the file says about the stock.

    ``n_hits`` of 0 means every press in the file. ``no_op`` records carry the
    opening state rather than a press and are skipped, but their declared
    workpiece length is still the length before press 1, so the length is read
    from the first record of either kind.

    Returns ``(hits, meta)``. ``meta["exposed_length_mm"]`` is the toolpath's own
    ``workpiece_length_mm`` before the first press, or None when the file omits it.

    Only the FIRST record's length is read, and deliberately so. The rest of the
    series is a forecast, not a measurement: it rises monotonically to the length
    the bar is expected to reach, and on the hot series it tracks the slab model's
    own prediction to about 0.03 mm. Scoring simulated elongation against it would
    be comparing one model to another and calling it validation. Use the scans for
    that. The first value is the stock that went in, which is an input.
    """
    records = []
    with jsonl_path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError("%s line %d is not JSON: %s" % (jsonl_path, lineno, e))
    if not records:
        raise ValueError("%s is empty" % jsonl_path)

    exposed = records[0].get("workpiece_length_mm")
    actions = [r.get("action") or r for r in records]
    presses = [a for a in actions if not a.get("no_op") and "rho" in a]
    if not presses:
        raise ValueError("%s has no press records" % jsonl_path)
    if n_hits and n_hits > len(presses):
        raise ValueError(
            "%s has %d press actions, requested %d" % (jsonl_path, len(presses), n_hits)
        )
    if n_hits:
        presses = presses[:n_hits]

    hits = [
        Hit(
            rho=float(a["rho"]) * RHO_SCALE,
            phi=math.radians(float(a["phi"])),
            z=float(a["z"]) + z_shift_mm,
            duration=1.0,
        )
        for a in presses
    ]
    meta = {
        "exposed_length_mm": None if exposed is None else float(exposed),
        "presses_in_file": sum(1 for a in actions if not a.get("no_op") and "rho" in a),
        "records": len(records),
    }
    return hits, meta


def verify_material(mat, mpm):
    """The card that was selected must be the card in effect.

    Every value the active preset declares is compared against the built scene.
    A mismatch means the preset did not reach the solver -- which is how a run
    completes on the wrong material and reads as a valid result. Setting only
    part of a flow curve once left an aluminium bar on hot-steel constants and
    27% too strong at yield, and the run looked fine.
    """
    from agforge.material_presets import active_material_preset

    name = os.environ.get("AGF_MATERIAL") or "316L (default)"
    preset = active_material_preset()
    wrong = []
    for key, want in preset.items():
        if key in ("enable_thermal", "billet_temp_k"):
            continue
        got = getattr(mat, key, None)
        if got is None or abs(float(got) - float(want)) > abs(float(want)) * 1e-9 + 1e-12:
            wrong.append("  %-26s preset %-14s scene %s" % (key, want, got))
    want_thermal = preset.get("enable_thermal", True)
    if bool(mpm.enable_thermal) != bool(want_thermal):
        wrong.append("  %-26s preset %-14s scene %s"
                     % ("enable_thermal", want_thermal, mpm.enable_thermal))
    if wrong:
        raise SystemExit(
            ("ABORT: AGF_MATERIAL=%s did not reach the solver." % name)
            + chr(10) + chr(10).join(wrong)
        )
    print("[toolpath] material %s verified in the built scene" % name)


def main() -> None:
    # Imported after the sys.path fix-ups above so this tree's modules win.
    from agforge.scripts.record_real_hits import _export_named_episode, _flush_episode
    import agforge.options as _opts

    p = argparse.ArgumentParser(
        description="Replay a recorded forge toolpath through the Genesis adapter."
    )
    p.add_argument("--jsonl", type=Path, required=True,
                   help="Recorded toolpath, one JSON record per line.")
    p.add_argument("--n-hits", type=int, default=0,
                   help="Presses to run. 0 (the default) runs every press in the file.")
    p.add_argument("--radius-mm", type=float, default=DEFAULT_RADIUS_MM,
                   help="Stock radius. Default %(default)s mm, i.e. 38.1 mm stock.")
    p.add_argument("--length-mm", type=float, default=None,
                   help="Exposed stock length. Required unless --length-from-toolpath.")
    p.add_argument("--length-from-toolpath", action="store_true",
                   help="Take the exposed length from the file's own "
                        "workpiece_length_mm before press 1. Only that first value "
                        "is used; the rest of the series is a forecast, not a "
                        "measurement (see load_toolpath).")
    p.add_argument("--z-shift-mm", type=float, default=0.0,
                   help="Added to every z. Use with AGF_PIN_EXTRA_MM when stock is held "
                        "on the chuck side, so z stays measured from the chuck face.")
    p.add_argument("--hit-dir", default=None,
                   help="Directory for per-press npz and frames.txt. Without it, nothing "
                        "per-press is written and only the episode is kept.")
    p.add_argument("--record-out", default=None,
                   help="Also export the episode to this .h5 path.")
    p.add_argument("--show-viewer", action="store_true")
    args = p.parse_args()

    hits, meta = load_toolpath(args.jsonl, args.n_hits, z_shift_mm=args.z_shift_mm)

    if args.length_from_toolpath:
        if meta["exposed_length_mm"] is None:
            raise SystemExit(
                "--length-from-toolpath was given but %s has no workpiece_length_mm on "
                "its first record. Pass --length-mm instead." % args.jsonl
            )
        length_mm = meta["exposed_length_mm"]
    elif args.length_mm is not None:
        length_mm = args.length_mm
        if (meta["exposed_length_mm"] is not None
                and abs(args.length_mm - meta["exposed_length_mm"]) > 0.05):
            print("[toolpath] NOTE --length-mm %.2f differs from the file's "
                  "workpiece_length_mm %.2f"
                  % (args.length_mm, meta["exposed_length_mm"]))
    else:
        raise SystemExit(
            "Give --length-mm, or --length-from-toolpath to read it from the file. "
            "There is no default: the exposed length is a property of the run, and "
            "guessing it moves every press relative to the free end."
        )

    print("[toolpath] %s: %d of %d press(es), %d record(s)"
          % (args.jsonl, len(hits), meta["presses_in_file"], meta["records"]))
    print("[toolpath] stock r=%.2f mm, exposed L=%.2f mm%s, z_shift=%.2f mm, "
          "rho x%.1f, phi deg->rad"
          % (args.radius_mm, length_mm,
             " (from toolpath)" if args.length_from_toolpath else "",
             args.z_shift_mm, RHO_SCALE))
    for i, h in enumerate(hits, start=1):
        print("  %3d: gap=%7.2fmm  phi=%7.4frad  z=%7.2fmm  reduction=%6.2fmm"
              % (i, h.rho, h.phi, h.z, args.radius_mm * 2.0 - h.rho))

    adapter = ToolpathGenesisAdapter(
        press_width_mm=die_contact_axial_width_mm(),
        show_viewer=args.show_viewer,
    )
    state = adapter.init_stock(radius_mm=args.radius_mm, length_mm=length_mm)
    mpm = state.env.cfg.mpm
    mat = state.env.cfg.mat
    robot = state.env.cfg.robot
    print("[toolpath] options from %s" % _opts.__file__)
    print("[toolpath] grid: cells_across_diameter=%.2f dx=%.3fmm"
          % (robot.base_grid_density * robot.cylinder_diameter,
             1000.0 / robot.base_grid_density))
    print("[toolpath] stock: length=%.2fmm held=%.2fmm pinned_face=%.2fmm clamp_x=%s"
          % (robot.cylinder_height * 1e3, (robot.pin_extra_m or 0.0) * 1e3,
             state.x_pinned_face_m * 1e3,
             [round(v * 1e3, 2) for v in robot.fixed_region_bounds[0].tolist()]))
    print("[toolpath] material: jc_A=%.4e jc_B=%.4e jc_n=%s E=%.4e nu=%s rho=%s "
          "thermal=%s T0=%s"
          % (mat.jc_A, mat.jc_B, mat.jc_n, mat.E, mat.nu, mat.rho,
             mpm.enable_thermal, mpm.default_initial_temperature))
    print("[toolpath] press: approach=%.2fm/s pressing=%.2fm/s max_force=%.3e "
          "contact=%s die=%s cfl=%s"
          % (state.env.cfg.strike.approach_speed, state.env.cfg.strike.pressing_speed,
             state.env.cfg.strike.max_force,
             os.environ.get("AGF_CONTACT_MODE", "grid"),
             os.environ.get("AGF_DIE_MESH", "") or "box",
             os.environ.get("AGF_CFL_SAFETY", "0.45")))

    verify_material(mat, mpm)

    state.controller.recorder.is_recording = True
    if args.hit_dir:
        import numpy as _np0
        os.makedirs(args.hit_dir, exist_ok=True)
        _p0 = state.env.mpm_entity.get_state().pos[0].detach().cpu().numpy()
        _np0.savez(os.path.join(args.hit_dir, "hit_00.npz"), pos=_p0.astype(_np0.float32))
        print("[toolpath] saved the bar before press 1 (%d particles)" % _p0.shape[0])

    hit_i = [0]
    prev_n = [0]

    def on_hit(state, hit, mesh):
        hit_i[0] += 1
        print("  press %d/%d: gap=%.3fmm phi=%.4frad z=%.3fmm"
              % (hit_i[0], len(hits), hit.rho, hit.phi, hit.z))
        if not args.hit_dir:
            return

        # The recorder buffer holds every frame's particle state, which reaches GB
        # scale over a full sequence. Write this press's last frame, then release the
        # buffer and the stale checkpoints so the next press has the memory.
        import gc as _gc
        import numpy as _np

        _rec = state.controller.recorder
        _buf = _rec.buffer
        _this = len(_buf["qpos"])
        _n = prev_n[0] + _this
        os.makedirs(args.hit_dir, exist_ok=True)
        with open(os.path.join(args.hit_dir, "frames.txt"), "a", encoding="utf-8") as _fh:
            print("%d %d" % (hit_i[0], _n), file=_fh)

        _dst = os.path.join(args.hit_dir, "hit_%02d.npz" % hit_i[0])
        _tmp = os.path.join(args.hit_dir, "_writing_%02d.npz" % hit_i[0])
        _np.savez(
            _tmp,
            pos=_np.asarray(_buf["particles_pos"][-1], dtype=_np.float32),
            detF=_np.asarray(_buf["particles_detF"][-1], dtype=_np.float32),
            force=_np.asarray(_buf["force_torque"], dtype=_np.float32),
            n_frames=_np.int32(_this),
        )
        os.replace(_tmp, _dst)  # so a reader never sees a half-written file
        _npart = int(_buf["particles_pos"][-1].shape[0])
        _vol = _buf["particle_vol"]
        _rec._init_buffer()
        _rec.buffer["particle_vol"] = _vol
        _rec.is_recording = True
        prev_n[0] = _n
        _ck = getattr(state.controller, "checkpoints", None)
        if _ck is not None and len(_ck) > 2:
            del _ck[:-2]
        _gc.collect()
        try:
            import torch as _torch
            if _torch.cuda.is_available():
                _torch.cuda.empty_cache()
        except Exception:
            pass
        print("    saved %s (%d particles)" % (os.path.basename(_dst), _npart))

    t0 = time.time()
    result = run_hit_sequence(adapter, state, hits, on_hit=on_hit)
    elapsed = time.time() - t0

    if result.succeeded:
        print("[toolpath] all %d/%d presses in %.1fs (%.1fs per press)"
              % (result.hits_completed, result.total_hits_planned, elapsed,
                 elapsed / max(result.hits_completed, 1)))
    else:
        print("[toolpath] STOPPED after %d/%d presses (%.1fs) -- %s: %s"
              % (result.hits_completed, result.total_hits_planned, elapsed,
                 type(result.error).__name__, result.error))

    if len(state.controller.recorder.buffer["qpos"]) == 0 and prev_n[0] > 0:
        print("[toolpath] %d per-press file(s) in %s" % (hit_i[0], args.hit_dir))
        if not result.succeeded:
            sys.exit(1)
        return

    episode = _flush_episode(
        state,
        success=result.succeeded,
        language_instruction="%s replay (%d/%d presses)"
        % (args.jsonl.stem, result.hits_completed, result.total_hits_planned),
    )
    if episode is None:
        print("[toolpath] no episode written (recorder buffer was empty).")
        sys.exit(1)

    shard_path, ep_name = episode
    replay_data, replay_ep = shard_path, ep_name
    if args.record_out:
        replay_data, replay_ep = _export_named_episode(shard_path, ep_name, args.record_out)
        print("[toolpath] episode exported to %s (%s)" % (replay_data, replay_ep))
    print("[toolpath] replay it with:")
    print("  python -m agforge.replay_episode --data %s -e %s --loop"
          % (replay_data, replay_ep))
    if not result.succeeded:
        sys.exit(1)


if __name__ == "__main__":
    main()
