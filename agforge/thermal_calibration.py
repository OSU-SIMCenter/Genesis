"""Analytical thermal calibration helpers for AgForge forging simulation.

Provides energy-balance estimates that mirror the GPU kernels in
``base_mpm_solver`` and ``legacy_coupler``.  Used by ``thermal_calibrate`` to
set literature-backed defaults and recommend ``q_peak``, ``L_eff``, and
``h_contact`` before or after headless benchmark fitting.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from agforge.thermal import get_steel_cp_numpy
from agforge.thermal_field import biot_savart_f_axial, skin_weight

STEFAN_BOLTZMANN = 5.67e-8  # W/(m^2 K^4)
STEEL_RHO_THERMAL = 7850.0  # kg/m^3 — matches thermal kernels (mech uses 8000)


def get_steel_k_numpy(temp: np.ndarray | float) -> np.ndarray:
    """Thermal conductivity [W/mK] — mirrors ``get_steel_thermal_conductivity``."""
    t = np.asarray(temp, dtype=np.float64)
    k = np.full_like(t, 44.0)
    mask_high = t >= 1000.0
    k[mask_high] = 27.0
    mask_mid = (t >= 700.0) & (t < 1000.0)
    if np.any(mask_mid):
        u = (t[mask_mid] - 700.0) / 300.0
        k[mask_mid] = 35.0 - u * 8.0
    mask_low = (t > 293.15) & (t < 700.0)
    if np.any(mask_low):
        u = (t[mask_low] - 293.15) / 406.85
        k[mask_low] = 44.0 - u * 9.0
    return k


@dataclass
class CalibrationTargets:
    """Application targets used by predict/fit/verify."""

    initial_temp_k: float = 293.15
    target_surface_temp_k: float = 1200.0
    # Thermal (scaled) seconds to reach target under coil center, ignoring losses.
    target_heating_time_thermal_s: float = 45.0
    # Max acceptable held-end temperature during idle heating.
    target_held_end_temp_k: float = 400.0
    # Ambient / chuck reference.
    ambient_temp_k: float = 293.15


@dataclass
class ThermalCalibrationConfig:
    """Snapshot of solver thermal parameters for analysis."""

    # Geometry
    cylinder_radius_m: float
    cylinder_height_m: float
    skin_depth_m: float
    coil_half_length_m: float
    coil_radius_m: float
    coil_center_x_m: float
    fixed_end_x_cut_m: float
    cylinder_pos: tuple[float, float, float]
    cylinder_euler: tuple[float, float, float]
    mpm_lower_bound: tuple[float, float, float]
    mpm_upper_bound: tuple[float, float, float]
    dx_m: float
    particle_radius_m: float

    # Material
    rho_kg_m3: float = STEEL_RHO_THERMAL

    # Timing
    macro_dt_s: float = 1.0e-5
    substeps: int = 8
    thermal_time_scale: float = 1.0

    # Coefficients
    q_peak_w_m3: float = 2.0e8
    h_air_w_m2k: float = 15.0
    h_contact_w_m2k: float = 5000.0
    emissivity: float = 0.8
    fixed_end_conduction_length_m: float = 0.05
    fixed_end_ambient_k: float = 293.15

    @property
    def substep_dt_s(self) -> float:
        return self.macro_dt_s / self.substeps

    @property
    def thermal_dt_per_macro_s(self) -> float:
        return self.macro_dt_s * self.thermal_time_scale

    @property
    def billet_volume_m3(self) -> float:
        return math.pi * self.cylinder_radius_m**2 * self.cylinder_height_m

    @property
    def billet_mass_kg(self) -> float:
        return self.rho_kg_m3 * self.billet_volume_m3

    @property
    def lateral_surface_area_m2(self) -> float:
        r = self.cylinder_radius_m
        h = self.cylinder_height_m
        return 2.0 * math.pi * r * h + 2.0 * math.pi * r * r


def load_config_from_teleop(cfg=None) -> ThermalCalibrationConfig:
    """Build a calibration config from ``TeleopOptions`` (or a fresh instance)."""
    if cfg is None:
        from agforge.options import TeleopOptions

        cfg = TeleopOptions()

    robot = cfg.robot
    mpm = cfg.mpm
    dx = 1.0 / robot.base_grid_density
    return ThermalCalibrationConfig(
        cylinder_radius_m=float(robot.cylinder_radius),
        cylinder_height_m=float(robot.cylinder_height),
        skin_depth_m=float(cfg.skin_depth),
        coil_half_length_m=float(robot.coil_length) / 2.0,
        coil_radius_m=float(robot.coil_radius),
        coil_center_x_m=float(robot.cylinder_pos[0]),
        fixed_end_x_cut_m=float(mpm.fixed_end_x_cut),
        cylinder_pos=tuple(float(x) for x in robot.cylinder_pos),
        cylinder_euler=tuple(float(x) for x in robot.cylinder_euler),
        mpm_lower_bound=tuple(float(x) for x in robot.mpm_lower_bound),
        mpm_upper_bound=tuple(float(x) for x in robot.mpm_upper_bound),
        dx_m=dx,
        particle_radius_m=float(mpm.particle_size) / 2.0,
        rho_kg_m3=STEEL_RHO_THERMAL,
        macro_dt_s=float(cfg.sim.dt),
        substeps=int(cfg.sim.substeps),
        thermal_time_scale=float(mpm.thermal_time_scale),
        q_peak_w_m3=float(cfg.heating_power),
        h_air_w_m2k=float(mpm.thermal_air_conductivity),
        h_contact_w_m2k=float(mpm.thermal_contact_conductivity),
        emissivity=float(mpm.emissivity),
        fixed_end_conduction_length_m=float(mpm.fixed_end_conduction_length),
        fixed_end_ambient_k=float(mpm.fixed_end_ambient),
    )


def cp_at(temp_k: float) -> float:
    return float(get_steel_cp_numpy(np.array([temp_k]))[0])


def k_at(temp_k: float) -> float:
    return float(get_steel_k_numpy(temp_k))


def alpha_at(temp_k: float, rho: float = STEEL_RHO_THERMAL) -> float:
    return k_at(temp_k) / (rho * cp_at(temp_k))


def surface_induction_heating_rate_k_per_s(
    q_peak: float,
    thermal_time_scale: float,
    temp_k: float,
    *,
    depth_m: float = 0.0,
    f_axial: float = 1.0,
    rho: float = STEEL_RHO_THERMAL,
) -> float:
    """dT/dt [K/s thermal-time] at a particle from induction only.

    ``S_T`` cancels when converting per-macro deposition to a thermal-time rate:
    dT/dt = q_peak * w * f_axial / (rho * Cp).
    """
    del thermal_time_scale  # kept for API compatibility
    cp = cp_at(temp_k)
    w = math.exp(-2.0 * depth_m / max(1e-12, 1.0)) if depth_m else 1.0
    return q_peak * f_axial * w / (rho * cp)


def linearized_radiation_h(
    temp_k: float,
    ambient_k: float,
    emissivity: float,
) -> float:
    """h_rad from Stefan–Boltzmann linearization [W/m²K] (unscaled)."""
    t = temp_k
    t0 = ambient_k
    return emissivity * STEFAN_BOLTZMANN * (t * t + t0 * t0) * (t + t0)


def lumped_surface_cooling_rate_k_per_s(
    temp_k: float,
    ambient_k: float,
    area_m2: float,
    mass_kg: float,
    *,
    h_air: float,
    emissivity: float,
    thermal_time_scale: float,
    rho: float = STEEL_RHO_THERMAL,
) -> float:
    """Approximate net surface cooling rate [K/s thermal-time] (air + radiation)."""
    del thermal_time_scale  # S_T cancels in thermal-time rate
    cp = cp_at(temp_k)
    h_rad = linearized_radiation_h(temp_k, ambient_k, emissivity)
    h_total = h_air + h_rad
    return h_total * area_m2 * (temp_k - ambient_k) / (mass_kg * cp)


def bulk_cutface_cooling_rate_k_per_s(
    temp_k: float,
    bulk_temp_k: float,
    cut_area_m2: float,
    cell_thermal_mass_kg: float,
    *,
    l_eff_m: float,
    thermal_time_scale: float,
    rho: float = STEEL_RHO_THERMAL,
) -> float:
    """Robin cut-face cooling rate for one grid cell's thermal mass."""
    del thermal_time_scale
    cp = cp_at(temp_k)
    h_bulk = k_at(temp_k) / l_eff_m
    return h_bulk * cut_area_m2 * (temp_k - bulk_temp_k) / (cell_thermal_mass_kg * cp)


def recommend_l_eff_m(t_char_thermal_s: float, temp_k: float = 293.15) -> float:
    """Effective chuck conduction length from representative heating time."""
    alpha = alpha_at(temp_k)
    return math.sqrt(math.pi * alpha * t_char_thermal_s)


def recommend_q_peak_w_m3(
    delta_t_k: float,
    t_thermal_s: float,
    temp_k: float,
    *,
    depth_m: float = 0.0,
    f_axial: float = 1.0,
    rho: float = STEEL_RHO_THERMAL,
) -> float:
    """q_peak to achieve ``delta_t_k`` in ``t_thermal_s`` (lossless, at coil center)."""
    cp = cp_at(temp_k)
    w = math.exp(-2.0 * depth_m / max(1e-12, 1.0)) if depth_m else 1.0
    denom = t_thermal_s * f_axial * w
    if denom <= 0.0:
        return float("inf")
    return delta_t_k * rho * cp / denom


def macro_steps_for_delta_t(
    delta_t_k: float,
    q_peak: float,
    cfg: ThermalCalibrationConfig,
    *,
    temp_k: float = 600.0,
    depth_m: float = 0.0,
    f_axial: float = 1.0,
) -> float:
    """Macro steps needed for ``delta_t_k`` induction-only heating at coil center."""
    dT_per_step = (
        q_peak
        * cfg.thermal_time_scale
        * cfg.macro_dt_s
        * f_axial
        * (math.exp(-2.0 * depth_m / cfg.skin_depth_m) if depth_m else 1.0)
        / (cfg.rho_kg_m3 * cp_at(temp_k))
    )
    if dT_per_step <= 0.0:
        return float("inf")
    return delta_t_k / dT_per_step


def estimate_steady_surface_temp_k(
    cfg: ThermalCalibrationConfig,
    targets: CalibrationTargets | None = None,
    *,
    depth_m: float = 0.0,
    f_axial: float = 1.0,
) -> float | None:
    """Surface T where induction ≈ air+radiation (0D, held-end ignored)."""
    targets = targets or CalibrationTargets()
    t_amb = targets.ambient_temp_k

    def residual(temp_k: float) -> float:
        q_in = surface_induction_heating_rate_k_per_s(
            cfg.q_peak_w_m3,
            cfg.thermal_time_scale,
            temp_k,
            depth_m=depth_m,
            f_axial=f_axial,
            rho=cfg.rho_kg_m3,
        )
        # Lumped billet: use half the lateral area as an order-of-magnitude surface.
        area = 0.5 * cfg.lateral_surface_area_m2
        q_out = lumped_surface_cooling_rate_k_per_s(
            temp_k,
            t_amb,
            area,
            cfg.billet_mass_kg,
            h_air=cfg.h_air_w_m2k,
            emissivity=cfg.emissivity,
            thermal_time_scale=cfg.thermal_time_scale,
            rho=cfg.rho_kg_m3,
        )
        return q_in - q_out

    lo, hi = t_amb, 2000.0
    r_lo, r_hi = residual(lo), residual(hi)
    if r_lo <= 0.0:
        return lo
    if r_hi >= 0.0:
        return None
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if residual(mid) > 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


@dataclass
class AnalysisReport:
    config: ThermalCalibrationConfig
    targets: CalibrationTargets
    heating_rate_surface_k_per_s: float
    dT_per_macro_step_surface_k: float
    macro_steps_to_target_lossless: float
    thermal_seconds_to_target_lossless: float
    recommended_q_peak_lossless_w_m3: float
    recommended_l_eff_m: float
    steady_surface_temp_k: float | None
    cooling_rate_at_1000k_k_per_s: float
    net_heating_at_1000k_k_per_s: float
    coil_center_f_axial: float
    skin_weight_at_surface: float
    skin_weight_at_radius: float
    literature_defaults: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["config"] = asdict(self.config)
        d["targets"] = asdict(self.targets)
        return d


def analyze(
    cfg: ThermalCalibrationConfig | None = None,
    targets: CalibrationTargets | None = None,
) -> AnalysisReport:
    """Full analytical report for current or supplied configuration."""
    cfg = cfg or load_config_from_teleop()
    targets = targets or CalibrationTargets()

    f_ax = float(
        biot_savart_f_axial(
            np.array([0.0]),
            cfg.coil_half_length_m,
            cfg.coil_radius_m,
        )[0]
    )
    w_surface = 1.0
    w_at_r = float(skin_weight(np.array([cfg.cylinder_radius_m]), cfg.skin_depth_m)[0])

    heat_surf = surface_induction_heating_rate_k_per_s(
        cfg.q_peak_w_m3,
        cfg.thermal_time_scale,
        targets.target_surface_temp_k,
        f_axial=f_ax,
        rho=cfg.rho_kg_m3,
    )
    dT_step = heat_surf * cfg.thermal_dt_per_macro_s
    delta_t = targets.target_surface_temp_k - targets.initial_temp_k
    steps = macro_steps_for_delta_t(delta_t, cfg.q_peak_w_m3, cfg, f_axial=f_ax)
    t_thermal = steps * cfg.thermal_dt_per_macro_s

    q_rec = recommend_q_peak_w_m3(
        delta_t,
        targets.target_heating_time_thermal_s,
        0.5 * (targets.initial_temp_k + targets.target_surface_temp_k),
        f_axial=f_ax,
    )
    l_rec = recommend_l_eff_m(targets.target_heating_time_thermal_s)

    cool_1k = lumped_surface_cooling_rate_k_per_s(
        1000.0,
        targets.ambient_temp_k,
        0.5 * cfg.lateral_surface_area_m2,
        cfg.billet_mass_kg,
        h_air=cfg.h_air_w_m2k,
        emissivity=cfg.emissivity,
        thermal_time_scale=cfg.thermal_time_scale,
        rho=cfg.rho_kg_m3,
    )
    net_1k = heat_surf - cool_1k
    t_ss = estimate_steady_surface_temp_k(cfg, targets, f_axial=f_ax)

    literature = {
        "skin_depth_rule": "cylinder_radius / 3 (through-heating rule of thumb)",
        "h_air_natural_convection": "5–25 W/m²K; default 15",
        "emissivity_oxidized_steel": "0.75–0.85; default 0.80",
        "h_contact_range": "1000–10000+ W/m²K depending on die pressure",
        "l_eff_formula": "sqrt(pi * alpha * t_char)",
    }

    notes = [
        "thermal_time_scale is derived from thermal CFL — treat as fixed unless rescaling globally.",
        "Tune q_peak for heating rate; L_eff for held-end gradient; h_contact for strike die chilling.",
        "Idle heating: contact cooling inactive; air + radiation + bulk + diffusion compete with induction.",
        "Mapping: t_thermal [s] ≈ n_macro_steps * macro_dt * thermal_time_scale.",
    ]
    if t_ss is not None and t_ss < targets.target_surface_temp_k:
        notes.append(
            f"At q_peak={cfg.q_peak_w_m3:.2e}, steady 0D surface estimate ({t_ss:.0f} K) "
            f"is below target {targets.target_surface_temp_k:.0f} K — increase q_peak or reduce cooling."
        )
    if net_1k <= 0.0:
        notes.append("Net heating at 1000 K is <= 0 — cooling dominates; increase q_peak.")

    return AnalysisReport(
        config=cfg,
        targets=targets,
        heating_rate_surface_k_per_s=heat_surf,
        dT_per_macro_step_surface_k=dT_step,
        macro_steps_to_target_lossless=steps,
        thermal_seconds_to_target_lossless=t_thermal,
        recommended_q_peak_lossless_w_m3=q_rec,
        recommended_l_eff_m=l_rec,
        steady_surface_temp_k=t_ss,
        cooling_rate_at_1000k_k_per_s=cool_1k,
        net_heating_at_1000k_k_per_s=net_1k,
        coil_center_f_axial=f_ax,
        skin_weight_at_surface=w_surface,
        skin_weight_at_radius=w_at_r,
        literature_defaults=literature,
        notes=notes,
    )


def format_analysis_report(report: AnalysisReport) -> str:
    """Human-readable analysis summary."""
    c = report.config
    t = report.targets
    lines = [
        "=== Thermal calibration analysis ===",
        "",
        "--- Geometry ---",
        f"  Billet: R={c.cylinder_radius_m*1e3:.2f} mm, H={c.cylinder_height_m*1e3:.1f} mm",
        f"  Coil: L={2*c.coil_half_length_m*1e3:.1f} mm, R_solenoid={c.coil_radius_m*1e3:.1f} mm",
        f"  skin_depth δ={c.skin_depth_m*1e3:.2f} mm, dx={c.dx_m*1e3:.2f} mm",
        "",
        "--- Timing ---",
        f"  macro_dt={c.macro_dt_s:.3e} s, substeps={c.substeps}, S_T={c.thermal_time_scale:.0f}",
        f"  thermal_dt/macro={c.thermal_dt_per_macro_s:.4f} s",
        "",
        "--- Current coefficients ---",
        f"  q_peak={c.q_peak_w_m3:.3e} W/m³",
        f"  h_air={c.h_air_w_m2k} W/m²K, ε={c.emissivity}, h_contact={c.h_contact_w_m2k} W/m²K",
        f"  L_eff={c.fixed_end_conduction_length_m*1e3:.1f} mm, T_bulk={c.fixed_end_ambient_k:.1f} K",
        "",
        "--- Induction (coil center, surface) ---",
        f"  f_axial={report.coil_center_f_axial:.3f}, w_skin(d=0)={report.skin_weight_at_surface:.3f}",
        f"  w_skin(d=R)={report.skin_weight_at_radius:.4f}",
        f"  Heating rate: {report.heating_rate_surface_k_per_s:.2f} K/s (thermal time)",
        f"  ΔT/macro-step: {report.dT_per_macro_step_surface_k:.3f} K",
        f"  Lossless reach {t.target_surface_temp_k:.0f} K: "
        f"{report.macro_steps_to_target_lossless:.0f} macro-steps "
        f"({report.thermal_seconds_to_target_lossless:.1f} s thermal)",
        "",
        "--- Cooling @ 1000 K (0D lumped estimate) ---",
        f"  Cooling rate: {report.cooling_rate_at_1000k_k_per_s:.2f} K/s",
        f"  Net heating:  {report.net_heating_at_1000k_k_per_s:.2f} K/s",
        f"  Steady surface (induction ≈ cooling): "
        f"{report.steady_surface_temp_k:.0f} K"
        if report.steady_surface_temp_k is not None
        else "  Steady surface: no balance below 2000 K (q_peak too high)",
        "",
        "--- Recommendations (targets) ---",
        f"  Target: {t.initial_temp_k:.0f} → {t.target_surface_temp_k:.0f} K "
        f"in {t.target_heating_time_thermal_s:.0f} s thermal (lossless)",
        f"  Recommended q_peak: {report.recommended_q_peak_lossless_w_m3:.3e} W/m³",
        f"  Recommended L_eff: {report.recommended_l_eff_m*1e3:.1f} mm "
        f"(for t_char={t.target_heating_time_thermal_s:.0f} s)",
        "",
        "--- Notes ---",
    ]
    lines.extend(f"  • {n}" for n in report.notes)
    return "\n".join(lines)


@dataclass
class FitResult:
    q_peak_w_m3: float
    fixed_end_conduction_length_m: float
    h_contact_w_m2k: float
    score: float
    metrics: dict[str, Any]
    method: str


def fit_coefficients_analytical(
    cfg: ThermalCalibrationConfig | None = None,
    targets: CalibrationTargets | None = None,
) -> FitResult:
    """Fast analytical fit for q_peak and L_eff (h_contact unchanged)."""
    cfg = cfg or load_config_from_teleop()
    targets = targets or CalibrationTargets()
    base = analyze(cfg, targets)

    f_ax = base.coil_center_f_axial
    delta_t = targets.target_surface_temp_k - targets.initial_temp_k
    t_mid = 0.5 * (targets.initial_temp_k + targets.target_surface_temp_k)

    # Start from lossless recommendation, then bump if cooling eats margin.
    q_peak = recommend_q_peak_w_m3(
        delta_t,
        targets.target_heating_time_thermal_s,
        t_mid,
        f_axial=f_ax,
    )

    # If net at 1000K would be negative with that q, scale up.
    trial_cfg = ThermalCalibrationConfig(**{**asdict(cfg), "q_peak_w_m3": q_peak})
    net = analyze(trial_cfg, targets).net_heating_at_1000k_k_per_s
    if net <= 0.0:
        q_peak *= 1.5

    l_eff = recommend_l_eff_m(targets.target_heating_time_thermal_s)

    final_cfg = ThermalCalibrationConfig(
        **{
            **asdict(cfg),
            "q_peak_w_m3": q_peak,
            "fixed_end_conduction_length_m": l_eff,
        }
    )
    rep = analyze(final_cfg, targets)
    score = 0.0
    if rep.steady_surface_temp_k is not None:
        score += abs(rep.steady_surface_temp_k - targets.target_surface_temp_k)
    score += abs(rep.thermal_seconds_to_target_lossless - targets.target_heating_time_thermal_s)

    return FitResult(
        q_peak_w_m3=q_peak,
        fixed_end_conduction_length_m=l_eff,
        h_contact_w_m2k=cfg.h_contact_w_m2k,
        score=score,
        metrics={
            "thermal_seconds_to_target": rep.thermal_seconds_to_target_lossless,
            "steady_surface_temp_k": rep.steady_surface_temp_k,
            "net_heating_at_1000k": rep.net_heating_at_1000k_k_per_s,
        },
        method="analytical",
    )


def apply_fit_to_options_dict(fit: FitResult) -> dict[str, Any]:
    """Patch dict suitable for merging into TeleopOptions / MPMOptions."""
    return {
        "heating_power": fit.q_peak_w_m3,
        "mpm": {
            "fixed_end_conduction_length": fit.fixed_end_conduction_length_m,
            "thermal_contact_conductivity": fit.h_contact_w_m2k,
        },
    }


def save_calibration_json(path: str, fit: FitResult, report: AnalysisReport) -> None:
    payload = {
        "fit": asdict(fit),
        "analysis": report.to_dict(),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
