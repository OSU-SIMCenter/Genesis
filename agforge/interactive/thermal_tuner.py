"""Live thermal / induction parameter tuning for Genesis-viewer teleop."""

from __future__ import annotations

import copy
import time
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from agforge.environment import AgilityForgeEnv
    from agforge.strike_controller import StrikeController

# HUD temperature readout interval (seconds). Held-end sink updates run only when blend > 0.
_TELEMETRY_INTERVAL_S = 0.5

_THERMAL_TUNER_KEY_PAIRS = [
    ("dec_q_peak", "_1", "q_peak", -1.0),
    ("inc_q_peak", "_2", "q_peak", 1.0),
    ("dec_skin", "_3", "skin_depth", -1.0),
    ("inc_skin", "_4", "skin_depth", 1.0),
    ("dec_coil_len", "_5", "coil_length", -1.0),
    ("inc_coil_len", "_6", "coil_length", 1.0),
    ("dec_coil_r", "_7", "coil_radius", -1.0),
    ("inc_coil_r", "_8", "coil_radius", 1.0),
    ("dec_st", "_9", "thermal_time_scale", -1.0),
    ("inc_st", "_0", "thermal_time_scale", 1.0),
    ("dec_leff", "MINUS", "fixed_end_L_eff", -1.0),
    ("inc_leff", "EQUAL", "fixed_end_L_eff", 1.0),
    ("dec_blend", "BRACKETLEFT", "fixed_end_blend", -1.0),
    ("inc_blend", "BRACKETRIGHT", "fixed_end_blend", 1.0),
    ("dec_off", "SEMICOLON", "coil_offset_x", -1.0),
    ("inc_off", "APOSTROPHE", "coil_offset_x", 1.0),
]
# Shift+number row sends punctuation on US keyboards (!@#...), not the digit + shift mod.
_THERMAL_TUNER_FINE_KEY_NAMES = {
    "_1": "EXCLAMATION",
    "_2": "AT",
    "_3": "HASH",
    "_4": "DOLLAR",
    "_5": "PERCENT",
    "_6": "ASCIICIRCUM",
    "_7": "AMPERSAND",
    "_8": "ASTERISK",
    "_9": "PARENLEFT",
    "_0": "PARENRIGHT",
    "MINUS": "UNDERSCORE",
    "EQUAL": "PLUS",
    "BRACKETLEFT": "BRACELEFT",
    "BRACKETRIGHT": "BRACERIGHT",
    "SEMICOLON": "COLON",
    "APOSTROPHE": "DOUBLEQUOTE",
}


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
        self._cached_temps: dict[str, float] = {"mean_k": 0.0, "max_k": 0.0, "min_k": 0.0}
        self._cached_held_k: float = 0.0
        self._last_telemetry_mono: float = 0.0
        # Baseline snapshot so X can restore without pressing Z first.
        self.save_preset()

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

    def restore_preset(self) -> bool:
        if self._saved is None:
            return False
        self.state = copy.deepcopy(self._saved)
        self._dirty = True
        return True

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

    def _apply_config(self) -> None:
        """CPU-side mirrors only (safe from viewer keybind thread)."""
        ctrl = self.controller
        cfg = ctrl.env.cfg
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

    def _apply_gpu(self) -> None:
        """Push live scalars into Taichi fields (sim thread only)."""
        ctrl = self.controller
        solver = ctrl.env.scene.sim.mpm_solver
        s = self.state

        sink_temp = (
            self._cached_held_k
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

        ctrl._invalidate_induction_params_cache()

    def apply(self, *, force: bool = False) -> None:
        if not force and not self._dirty:
            return
        self._dirty = False
        self._apply_config()
        self._apply_gpu()

    def refresh_held_end_temp(self) -> None:
        """Sample held-end temperature for blend sink (sim thread only)."""
        self._cached_held_k = measure_held_end_particle_temp(self.controller)

    def refresh_billet_temps(self, *, force: bool = False) -> None:
        """Throttled billet stats for the HUD (sim thread only)."""
        now = time.monotonic()
        if not force and (now - self._last_telemetry_mono) < _TELEMETRY_INTERVAL_S:
            return
        self._last_telemetry_mono = now
        self._cached_temps = measure_billet_temps(self.controller)

    def on_pre_physics_step(self) -> None:
        """Push pending tuning and refresh telemetry at a safe rate."""
        if self.state.fixed_end_blend > 0.0:
            self.refresh_held_end_temp()
            self._dirty = True
        if self._dirty:
            self.apply()
        self.refresh_billet_temps()
        guides = getattr(self.controller.env, "_visual_guides", None)
        if guides is not None:
            guides.sync_coil_visual_if_needed()

    def flush_pending(self) -> None:
        """Apply queued tuning on the asyncio/sim thread when physics is idle."""
        if not self._dirty:
            return
        if self.state.fixed_end_blend > 0.0:
            self.refresh_held_end_temp()
        self.apply()
        self.refresh_billet_temps()
        guides = getattr(self.controller.env, "_visual_guides", None)
        if guides is not None:
            guides.sync_coil_visual_if_needed()

    def format_status_lines(self) -> list[str]:
        s = self.state
        temps = self._cached_temps
        held = self._cached_held_k
        pending = " *" if self._dirty else ""
        return [
            f"Thermal tuner{pending}  T mean/max {temps['mean_k']:.0f}/{temps['max_k']:.0f} K  held {held:.0f} K",
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

        self.refresh_billet_temps(force=True)
        if self.state.fixed_end_blend > 0.0:
            self.refresh_held_end_temp()
        lines = self.format_status_lines()
        for line in lines:
            gs.logger.info(line)
        gs.logger.info(f"state={asdict(self.state)}")


def _thermal_key(name: str):
    from genesis.vis.keybindings import Key

    return getattr(Key, name)


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
    from genesis.vis.keybindings import Key, KeyAction, Keybind

    viewer = vis.viewer

    def _refresh_hud():
        from agforge.vis.status_overlay import refresh_viewer_status_with_tuner

        refresh_viewer_status_with_tuner(env, tuner=tuner)

    def _restore_preset():
        if not tuner.restore_preset():
            gs.logger.warning("thermal preset restore: nothing saved (press Z to snapshot first)")
            return
        tuner._apply_config()
        gs.logger.info("thermal preset restored (queued)")
        _refresh_hud()

    def _bind(field: str, delta: float):
        def _cb():
            tuner.adjust(field, delta, fine=False)
            val = getattr(tuner.state, field, None)
            gs.logger.info(f"thermal tuner: {field} = {val} (queued)")
            tuner._apply_config()
            _refresh_hud()

        return _cb

    def _bind_fine(field: str, delta: float):
        def _cb():
            tuner.adjust(field, delta, fine=True)
            val = getattr(tuner.state, field, None)
            gs.logger.info(f"thermal tuner (fine): {field} = {val} (queued)")
            tuner._apply_config()
            _refresh_hud()

        return _cb

    keybinds = [
        Keybind(
            name,
            _thermal_key(key_name),
            key_action=KeyAction.PRESS,
            callback=_bind(field, delta),
            allow_overload=True,
            show_in_help=False,
        )
        for name, key_name, field, delta in _THERMAL_TUNER_KEY_PAIRS
    ]
    keybinds.extend(
        Keybind(
            f"fine_{name}",
            _thermal_key(_THERMAL_TUNER_FINE_KEY_NAMES[key_name]),
            key_action=KeyAction.PRESS,
            callback=_bind_fine(field, delta),
            allow_overload=True,
            show_in_help=False,
        )
        for name, key_name, field, delta in _THERMAL_TUNER_KEY_PAIRS
    )
    keybinds.extend(
        [
            Keybind(
                "thermal_print",
                Key.P,
                key_action=KeyAction.PRESS,
                callback=lambda: (tuner.print_status(), _refresh_hud()),
                allow_overload=True,
                show_in_help=False,
            ),
            Keybind(
                "thermal_save_preset",
                Key.Z,
                key_action=KeyAction.PRESS,
                callback=lambda: (tuner.save_preset(), gs.logger.info("thermal preset saved"), _refresh_hud()),
                allow_overload=True,
                show_in_help=False,
            ),
            Keybind(
                "thermal_restore_preset",
                Key.X,
                key_action=KeyAction.PRESS,
                callback=_restore_preset,
                allow_overload=True,
                show_in_help=False,
            ),
        ]
    )

    viewer.register_keybinds(*keybinds, overwrite=False)


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
    if not getattr(env.cfg.general, "interactive_thermal_tuner", False):
        return None

    tuner = ThermalTuner(controller)
    tuner.refresh_billet_temps(force=True)
    tuner.apply(force=True)
    controller.thermal_tuner = tuner

    if register_keybinds:
        register_thermal_tuner_keybinds(env, tuner)

    from agforge.vis.status_overlay import refresh_viewer_status_with_tuner

    refresh_viewer_status_with_tuner(env, tuner=tuner)
    return tuner
