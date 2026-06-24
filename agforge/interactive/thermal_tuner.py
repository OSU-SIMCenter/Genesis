"""Live thermal / induction parameter tuning for Genesis-viewer teleop."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from agforge.environment import AgilityForgeEnv
    from agforge.strike_controller import StrikeController


@dataclass
class ThermalTunerState:
    q_peak: float
    skin_depth: float
    coil_length: float
    coil_radius: float
    coil_offset_x: float
    thermal_time_scale: float
    fixed_end_L_eff: float
    fixed_end_ambient: float
    fixed_end_blend: float


def _clamp(value: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, value)))


def measure_held_end_particle_temp(
    controller: "StrikeController",
    *,
    margin_cells: float = 2.0,
) -> float:
    """Mean particle temperature near the held-end cut plane (for fixed-end blend)."""
    mpm = controller.env.mpm_entity
    solver = controller.env.scene.sim.mpm_solver
    x_cut = float(solver._fixed_end_x_cut)
    dx = float(solver.dx)
    margin = dx * float(margin_cells)

    pos = mpm.get_particles_pos().detach().cpu().numpy().reshape(-1, 3)
    temps = mpm.get_particles_temp().detach().cpu().numpy().reshape(-1)
    mask = pos[:, 0] >= (x_cut - margin)
    if not np.any(mask):
        return float(np.mean(temps))
    return float(np.mean(temps[mask]))


def measure_billet_temps(controller: "StrikeController") -> dict[str, float]:
    """Quick volume stats for the HUD."""
    temps = controller.env.mpm_entity.get_particles_temp().detach().cpu().numpy().reshape(-1)
    temps = temps[np.isfinite(temps)]
    if temps.size == 0:
        return {"mean_k": 0.0, "max_k": 0.0, "min_k": 0.0}
    return {
        "mean_k": float(np.mean(temps)),
        "max_k": float(np.max(temps)),
        "min_k": float(np.min(temps)),
    }


class ThermalTuner:
    """Holds live thermal tuning parameters and pushes them into the GPU solver."""

    def __init__(self, controller: "StrikeController"):
        self.controller = controller
        self.state = self._state_from_config()
        self._saved: ThermalTunerState | None = None
        self._dirty = True

    def _state_from_config(self) -> ThermalTunerState:
        cfg = self.controller.env.cfg
        mpm = cfg.mpm
        robot = cfg.robot
        return ThermalTunerState(
            q_peak=float(self.controller.heating_power),
            skin_depth=float(self.controller.skin_depth),
            coil_length=float(robot.coil_length),
            coil_radius=float(robot.coil_radius),
            coil_offset_x=float(robot.coil_offset_x),
            thermal_time_scale=float(mpm.thermal_time_scale),
            fixed_end_L_eff=float(mpm.fixed_end_conduction_length),
            fixed_end_ambient=float(mpm.fixed_end_ambient),
            fixed_end_blend=float(getattr(mpm, "fixed_end_blend", 0.0)),
        )

    def save_preset(self) -> None:
        self._saved = copy.deepcopy(self.state)

    def restore_preset(self) -> None:
        if self._saved is None:
            return
        self.state = copy.deepcopy(self._saved)
        self._dirty = True
        self.apply(force=True)

    def mark_dirty(self) -> None:
        self._dirty = True

    def adjust(self, field: str, delta: float, *, fine: bool = False) -> None:
        scale = 0.1 if fine else 1.0
        d = float(delta) * scale
        s = self.state

        if field == "q_peak":
            s.q_peak = _clamp(s.q_peak + d * 2.5e7, 1.0e7, 1.0e10)
        elif field == "skin_depth":
            s.skin_depth = _clamp(s.skin_depth + d * 0.0005, 0.0005, 0.05)
        elif field == "coil_length":
            s.coil_length = _clamp(s.coil_length + d * 0.002, 0.005, 0.20)
        elif field == "coil_radius":
            s.coil_radius = _clamp(s.coil_radius + d * 0.001, 0.005, 0.08)
        elif field == "coil_offset_x":
            s.coil_offset_x += d * 0.002
        elif field == "thermal_time_scale":
            s.thermal_time_scale = _clamp(s.thermal_time_scale + d * max(50.0, s.thermal_time_scale * 0.05), 10.0, 50000.0)
        elif field == "fixed_end_L_eff":
            s.fixed_end_L_eff = _clamp(s.fixed_end_L_eff + d * 0.005, 0.005, 0.50)
        elif field == "fixed_end_blend":
            s.fixed_end_blend = _clamp(s.fixed_end_blend + d * 0.05, 0.0, 1.0)
        else:
            raise ValueError(f"Unknown thermal tuner field: {field}")

        self._dirty = True
        self.apply()

    def apply(self, *, force: bool = False) -> None:
        if not force and not self._dirty:
            return
        self._dirty = False

        ctrl = self.controller
        cfg = ctrl.env.cfg
        solver = ctrl.env.scene.sim.mpm_solver
        s = self.state

        ctrl.heating_power = s.q_peak
        ctrl.skin_depth = s.skin_depth
        cfg.heating_power = s.q_peak
        cfg.skin_depth = s.skin_depth
        cfg.robot.coil_length = s.coil_length
        cfg.robot.coil_radius = s.coil_radius
        cfg.robot.coil_offset_x = s.coil_offset_x
        cfg.mpm.thermal_time_scale = s.thermal_time_scale
        cfg.mpm.fixed_end_conduction_length = s.fixed_end_L_eff
        cfg.mpm.fixed_end_ambient = s.fixed_end_ambient
        cfg.mpm.fixed_end_blend = s.fixed_end_blend

        sink_temp = (
            measure_held_end_particle_temp(ctrl)
            if s.fixed_end_blend > 0.0
            else s.fixed_end_ambient
        )
        if hasattr(solver, "set_thermal_runtime_tuning"):
            solver.set_thermal_runtime_tuning(
                thermal_time_scale=s.thermal_time_scale,
                fixed_end_conduction_length=s.fixed_end_L_eff,
                fixed_end_ambient=s.fixed_end_ambient,
                fixed_end_blend=s.fixed_end_blend,
                fixed_end_sink_temp=sink_temp,
            )

        ctrl._last_induction_key = None

    def on_pre_physics_step(self) -> None:
        """Refresh held-end sink temp and push any pending parameter changes."""
        if self.state.fixed_end_blend > 0.0:
            self._dirty = True
        self.apply()

    def format_status_lines(self) -> list[str]:
        s = self.state
        temps = measure_billet_temps(self.controller)
        held = measure_held_end_particle_temp(self.controller)
        return [
            f"Thermal tuner  T mean/max {temps['mean_k']:.0f}/{temps['max_k']:.0f} K  held {held:.0f} K",
            (
                f"q_peak {s.q_peak:.2e}  d {s.skin_depth*1e3:.2f} mm  "
                f"S_T {s.thermal_time_scale:.0f}"
            ),
            (
                f"coil L {s.coil_length*1e3:.1f} R {s.coil_radius*1e3:.1f} off {s.coil_offset_x*1e3:.1f} mm"
            ),
            f"L_eff {s.fixed_end_L_eff*1e3:.1f} mm  blend {s.fixed_end_blend:.2f}",
            "Keys: 1/2 q  3/4 d  5/6 coilL  7/8 coilR  9/0 S_T  -/= L_eff  [/] blend  ;/' off  p print  z/x preset",
        ]

    def print_status(self) -> None:
        import genesis as gs

        lines = self.format_status_lines()
        for line in lines:
            gs.logger.info(line)
        gs.logger.info(f"state={asdict(self.state)}")


def register_thermal_tuner_keybinds(
    env: "AgilityForgeEnv",
    tuner: ThermalTuner | None,
) -> None:
    if tuner is None:
        return
    vis = env.scene.visualizer
    if vis is None or vis.viewer is None:
        return

    import genesis as gs
    from genesis.vis.keybindings import Key, KeyAction, KeyMod, Keybind

    viewer = vis.viewer

    def _refresh():
        from agforge.vis.status_overlay import refresh_viewer_status_with_tuner

        refresh_viewer_status_with_tuner(env, tuner=tuner)
        if vis is not None:
            vis.update(force=False, auto=True)

    def _bind(field: str, delta: float):
        def _cb():
            tuner.adjust(field, delta, fine=False)
            val = getattr(tuner.state, field if field != "fixed_end_L_eff" else "fixed_end_L_eff", None)
            gs.logger.info(f"thermal tuner: {field} = {val}")
            _refresh()

        return _cb

    def _bind_fine(field: str, delta: float):
        def _cb():
            tuner.adjust(field, delta, fine=True)
            gs.logger.info(f"thermal tuner (fine): {field}")
            _refresh()

        return _cb

    pairs = [
        ("dec_q_peak", Key.NUM_1, "q_peak", -1.0),
        ("inc_q_peak", Key.NUM_2, "q_peak", 1.0),
        ("dec_skin", Key.NUM_3, "skin_depth", -1.0),
        ("inc_skin", Key.NUM_4, "skin_depth", 1.0),
        ("dec_coil_len", Key.NUM_5, "coil_length", -1.0),
        ("inc_coil_len", Key.NUM_6, "coil_length", 1.0),
        ("dec_coil_r", Key.NUM_7, "coil_radius", -1.0),
        ("inc_coil_r", Key.NUM_8, "coil_radius", 1.0),
        ("dec_st", Key.NUM_9, "thermal_time_scale", -1.0),
        ("inc_st", Key.NUM_0, "thermal_time_scale", 1.0),
        ("dec_leff", Key.MINUS, "fixed_end_L_eff", -1.0),
        ("inc_leff", Key.EQUAL, "fixed_end_L_eff", 1.0),
        ("dec_blend", Key.BRACKETLEFT, "fixed_end_blend", -1.0),
        ("inc_blend", Key.BRACKETRIGHT, "fixed_end_blend", 1.0),
        ("dec_off", Key.SEMICOLON, "coil_offset_x", -1.0),
        ("inc_off", Key.APOSTROPHE, "coil_offset_x", 1.0),
    ]

    keybinds = [
        Keybind(name, key, key_action=KeyAction.PRESS, callback=_bind(field, delta), allow_overload=True)
        for name, key, field, delta in pairs
    ]
    keybinds.extend(
        Keybind(
            f"fine_{name}",
            key,
            key_mods=(KeyMod.SHIFT,),
            key_action=KeyAction.PRESS,
            callback=_bind_fine(field, delta),
            allow_overload=True,
        )
        for name, key, field, delta in pairs
    )
    keybinds.extend(
        [
            Keybind(
                "thermal_print",
                Key.P,
                key_action=KeyAction.PRESS,
                callback=lambda: (tuner.print_status(), _refresh()),
                allow_overload=True,
            ),
            Keybind(
                "thermal_save_preset",
                Key.Z,
                key_action=KeyAction.PRESS,
                callback=lambda: (tuner.save_preset(), gs.logger.info("thermal preset saved"), _refresh()),
                allow_overload=True,
            ),
            Keybind(
                "thermal_restore_preset",
                Key.X,
                key_action=KeyAction.PRESS,
                callback=lambda: (tuner.restore_preset(), gs.logger.info("thermal preset restored"), _refresh()),
                allow_overload=True,
            ),
        ]
    )

    viewer.register_keybinds(*keybinds, overwrite=False)
    from agforge.vis.status_overlay import _refresh_keybind_help

    _refresh_keybind_help(env)


def install_thermal_tuner(
    env: "AgilityForgeEnv",
    controller: "StrikeController",
    *,
    register_keybinds: bool = True,
) -> ThermalTuner | None:
    """Attach a live thermal tuner to a teleop StrikeController."""
    if not getattr(env.cfg.general, "show_viewer", False):
        return None
    if not getattr(env.cfg.mpm, "enable_thermal", False):
        return None

    tuner = ThermalTuner(controller)
    tuner.apply(force=True)
    controller.thermal_tuner = tuner

    if register_keybinds:
        register_thermal_tuner_keybinds(env, tuner)

    from agforge.vis.status_overlay import refresh_viewer_status_with_tuner

    refresh_viewer_status_with_tuner(env, tuner=tuner)
    return tuner
