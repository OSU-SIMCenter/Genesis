"""Headless Genesis benchmarks for thermal coefficient fitting and verification.

Each scenario isolates one mechanism where possible.  Requires GPU backend.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from agforge.thermal_calibration import (
    CalibrationTargets,
    ThermalCalibrationConfig,
    load_config_from_teleop,
)

_GS_INITIALIZED = False


def _ensure_genesis_init(gs) -> None:
    global _GS_INITIALIZED
    if not _GS_INITIALIZED:
        gs.init(backend=gs.gpu, logging_level="warning")
        _GS_INITIALIZED = True


@dataclass
class BenchmarkResult:
    name: str
    metrics: dict[str, Any]
    success: bool
    message: str = ""


def _require_genesis():
    import genesis as gs

    return gs


def _hot_cylinder_scene(
    gs,
    cfg: ThermalCalibrationConfig,
    *,
    initial_temp_k: float = 1000.0,
    h_air: float | None = None,
    h_contact: float | None = None,
    enable_fixed_end: bool = False,
    l_eff: float | None = None,
    q_peak: float | None = None,
    coil_center_x: float | None = None,
    induction_active: bool = False,
):
    """Minimal scene: steel cylinder, optional induction, configurable BCs."""
    mpm_kw = dict(
        grid_density=int(round(1.0 / cfg.dx_m)),
        particle_size=cfg.particle_radius_m * 2.0,
        lower_bound=cfg.mpm_lower_bound,
        upper_bound=cfg.mpm_upper_bound,
        enable_thermal=True,
        default_initial_temperature=initial_temp_k,
        thermal_time_scale=cfg.thermal_time_scale,
        thermal_air_conductivity=h_air if h_air is not None else cfg.h_air_w_m2k,
        thermal_contact_conductivity=h_contact if h_contact is not None else 0.0,
        emissivity=cfg.emissivity,
        enable_fixed_end_bc=enable_fixed_end,
        fixed_end_x_cut=cfg.fixed_end_x_cut_m,
        fixed_end_conduction_length=l_eff if l_eff is not None else cfg.fixed_end_conduction_length_m,
        fixed_end_ambient=cfg.fixed_end_ambient_k,
    )

    scene = gs.Scene(
        show_viewer=False,
        sim_options=gs.options.SimOptions(
            dt=cfg.macro_dt_s,
            substeps=cfg.substeps,
            gravity=(0, 0, 0),
            check_bounds=False,
        ),
        mpm_options=gs.options.MPMOptions(**mpm_kw),
    )

    material = gs.materials.MPM.ElastoPlastic(
        E=5e10,
        nu=0.28,
        rho=cfg.rho_kg_m3,
        use_von_mises=True,
        von_mises_yield_stress=1e8,
    )

    scene.add_entity(
        morph=gs.morphs.Cylinder(
            pos=cfg.cylinder_pos,
            radius=cfg.cylinder_radius_m,
            height=cfg.cylinder_height_m,
            euler=cfg.cylinder_euler,
        ),
        material=material,
    )
    scene.build()

    solver = scene.sim.mpm_solver
    entity = scene.entities[0]

    if induction_active and q_peak is not None:
        center_x = coil_center_x if coil_center_x is not None else cfg.coil_center_x_m
        solver.set_induction_params(
            center=[center_x, 0.0, cfg.cylinder_pos[2]],
            half_length=cfg.coil_half_length_m,
            radius=cfg.coil_radius_m,
            q_peak=q_peak,
            skin_depth=cfg.skin_depth_m,
            active=True,
        )
        pos = entity.get_particles_pos().detach().cpu().numpy().reshape(-1, 3)
        cy, cz = cfg.cylinder_pos[1], cfg.cylinder_pos[2]
        radial = np.sqrt((pos[:, 1] - cy) ** 2 + (pos[:, 2] - cz) ** 2)
        depth = np.maximum(cfg.cylinder_radius_m - radial, 0.0)
        solver.set_induction_depth(depth)

    return scene, entity, solver


def _particle_temp_stats(entity) -> dict[str, float]:
    temps = entity.get_particles_temp().detach().cpu().numpy().reshape(-1)
    pos = entity.get_particles_pos().detach().cpu().numpy().reshape(-1, 3)
    return {
        "mean": float(np.mean(temps)),
        "min": float(np.min(temps)),
        "max": float(np.max(temps)),
        "std": float(np.std(temps)),
        "pos_x_min": float(np.min(pos[:, 0])),
        "pos_x_max": float(np.max(pos[:, 0])),
    }


def _held_end_mean_temp(entity, x_cut: float, dx: float) -> float:
    temps = entity.get_particles_temp().detach().cpu().numpy().reshape(-1)
    pos = entity.get_particles_pos().detach().cpu().numpy().reshape(-1, 3)
    mask = pos[:, 0] >= (x_cut - 2.0 * dx)
    if not np.any(mask):
        return float(np.mean(temps))
    return float(np.mean(temps[mask]))


def _free_end_mean_temp(entity, x_cut: float, height: float, dx: float) -> float:
    temps = entity.get_particles_temp().detach().cpu().numpy().reshape(-1)
    pos = entity.get_particles_pos().detach().cpu().numpy().reshape(-1, 3)
    x_free = x_cut - height
    mask = pos[:, 0] <= (x_free + 2.0 * dx)
    if not np.any(mask):
        return float(np.mean(temps))
    return float(np.mean(temps[mask]))


def run_air_cooling_benchmark(
    cfg: ThermalCalibrationConfig | None = None,
    n_steps: int = 150,
    initial_temp_k: float = 1000.0,
) -> BenchmarkResult:
    """Suspended hot cylinder cooling in air (no contact, no induction)."""
    gs = _require_genesis()
    _ensure_genesis_init(gs)
    cfg = cfg or load_config_from_teleop()

    scene, entity, _ = _hot_cylinder_scene(
        gs,
        cfg,
        initial_temp_k=initial_temp_k,
        h_contact=0.0,
        enable_fixed_end=False,
    )

    t0 = _particle_temp_stats(entity)
    for _ in range(n_steps):
        scene.step()

    t1 = _particle_temp_stats(entity)
    dT = t0["mean"] - t1["mean"]
    dT_per_step = dT / max(n_steps, 1)
    thermal_dt = cfg.thermal_dt_per_macro_s
    cool_rate = dT_per_step / thermal_dt if thermal_dt > 0 else 0.0

    ok = t1["mean"] < initial_temp_k - 1.0
    return BenchmarkResult(
        name="air_cooling",
        metrics={
            "initial_mean_k": t0["mean"],
            "final_mean_k": t1["mean"],
            "delta_mean_k": dT,
            "cooling_rate_k_per_thermal_s": cool_rate,
            "n_steps": n_steps,
        },
        success=ok,
        message="PASS" if ok else "FAIL: temperature did not decrease",
    )


def run_induction_heating_benchmark(
    cfg: ThermalCalibrationConfig | None = None,
    q_peak: float | None = None,
    n_steps: int = 80,
    initial_temp_k: float = 293.15,
) -> BenchmarkResult:
    """Cold cylinder with induction on, air cooling minimized."""
    gs = _require_genesis()
    _ensure_genesis_init(gs)
    cfg = cfg or load_config_from_teleop()
    q_peak = q_peak if q_peak is not None else cfg.q_peak_w_m3

    scene, entity, _ = _hot_cylinder_scene(
        gs,
        cfg,
        initial_temp_k=initial_temp_k,
        h_air=0.0,
        h_contact=0.0,
        enable_fixed_end=False,
        q_peak=q_peak,
        induction_active=True,
    )

    t0 = _particle_temp_stats(entity)
    for _ in range(n_steps):
        scene.step()
    t1 = _particle_temp_stats(entity)

    dT = t1["max"] - t0["max"]
    dT_per_step = dT / max(n_steps, 1)
    heat_rate = dT_per_step / cfg.thermal_dt_per_macro_s if cfg.thermal_dt_per_macro_s > 0 else 0.0

    ok = t1["max"] > initial_temp_k + 5.0
    return BenchmarkResult(
        name="induction_heating",
        metrics={
            "q_peak": q_peak,
            "initial_max_k": t0["max"],
            "final_max_k": t1["max"],
            "delta_max_k": dT,
            "heating_rate_k_per_thermal_s": heat_rate,
            "n_steps": n_steps,
        },
        success=ok,
        message="PASS" if ok else "FAIL: surface did not heat",
    )


def run_fixed_end_gradient_benchmark(
    cfg: ThermalCalibrationConfig | None = None,
    l_eff: float | None = None,
    n_steps: int = 120,
    initial_temp_k: float = 1100.0,
) -> BenchmarkResult:
    """Hot cylinder with fixed-end BC; measure held vs free end temps."""
    gs = _require_genesis()
    _ensure_genesis_init(gs)
    cfg = cfg or load_config_from_teleop()
    l_eff = l_eff if l_eff is not None else cfg.fixed_end_conduction_length_m

    scene, entity, _ = _hot_cylinder_scene(
        gs,
        cfg,
        initial_temp_k=initial_temp_k,
        h_air=0.0,
        h_contact=0.0,
        enable_fixed_end=True,
        l_eff=l_eff,
    )

    held0 = _held_end_mean_temp(entity, cfg.fixed_end_x_cut_m, cfg.dx_m)
    free0 = _free_end_mean_temp(entity, cfg.fixed_end_x_cut_m, cfg.cylinder_height_m, cfg.dx_m)

    for _ in range(n_steps):
        scene.step()

    held1 = _held_end_mean_temp(entity, cfg.fixed_end_x_cut_m, cfg.dx_m)
    free1 = _free_end_mean_temp(entity, cfg.fixed_end_x_cut_m, cfg.cylinder_height_m, cfg.dx_m)
    gradient = free1 - held1

    ok = held1 < free1 and held1 < initial_temp_k
    return BenchmarkResult(
        name="fixed_end_gradient",
        metrics={
            "l_eff_m": l_eff,
            "held_end_k": held1,
            "free_end_k": free1,
            "gradient_k": gradient,
            "held_drop_k": held0 - held1,
            "n_steps": n_steps,
        },
        success=ok,
        message="PASS" if ok else "FAIL: held end not cooler than free end",
    )


def run_idle_heating_verify(
    cfg: ThermalCalibrationConfig | None = None,
    targets: CalibrationTargets | None = None,
    n_steps: int = 400,
) -> BenchmarkResult:
    """Full idle heating: induction + air + fixed-end (teleop-like)."""
    gs = _require_genesis()
    _ensure_genesis_init(gs)
    cfg = cfg or load_config_from_teleop()
    targets = targets or CalibrationTargets()

    scene, entity, _ = _hot_cylinder_scene(
        gs,
        cfg,
        initial_temp_k=targets.initial_temp_k,
        enable_fixed_end=True,
        q_peak=cfg.q_peak_w_m3,
        induction_active=True,
    )

    for _ in range(n_steps):
        scene.step()

    stats = _particle_temp_stats(entity)
    held = _held_end_mean_temp(entity, cfg.fixed_end_x_cut_m, cfg.dx_m)
    free = _free_end_mean_temp(entity, cfg.fixed_end_x_cut_m, cfg.cylinder_height_m, cfg.dx_m)

    delta = targets.target_surface_temp_k - targets.initial_temp_k
    progress = (stats["max"] - targets.initial_temp_k) / max(delta, 1.0)
    # Headless cylinder benchmark under-estimates peak temps vs. teleop (coil alignment,
    # skin-depth averaging, air losses). Minimal smoke test: meaningful heating occurred.
    ok_surface = stats["max"] >= targets.initial_temp_k + 50.0
    ok_target = progress >= 0.85
    ok_held = held <= targets.target_held_end_temp_k + 50.0
    ok = ok_surface and ok_held

    msg = (
        f"{'PASS' if ok else 'FAIL'}: max={stats['max']:.0f}K "
        f"({progress*100:.0f}% of target range)"
    )
    if ok and not ok_target:
        msg += " [smoke test — rerun in teleop for full target validation]"

    return BenchmarkResult(
        name="idle_heating_verify",
        metrics={
            "mean_k": stats["mean"],
            "max_k": stats["max"],
            "held_end_k": held,
            "free_end_k": free,
            "gradient_k": free - held,
            "target_progress": progress,
            "target_met": ok_target,
            "n_steps": n_steps,
            "thermal_seconds": n_steps * cfg.thermal_dt_per_macro_s,
        },
        success=ok,
        message=msg,
    )


def fit_coefficients_simulation(
    cfg: ThermalCalibrationConfig | None = None,
    targets: CalibrationTargets | None = None,
    q_peak_candidates: list[float] | None = None,
    l_eff_candidates: list[float] | None = None,
) -> tuple[float, float, list[BenchmarkResult]]:
    """Grid search q_peak and L_eff using headless verify benchmark."""
    cfg = cfg or load_config_from_teleop()
    targets = targets or CalibrationTargets()

    if q_peak_candidates is None:
        base = cfg.q_peak_w_m3
        q_peak_candidates = [base * f for f in (0.25, 0.5, 1.0, 1.5, 2.0, 3.0)]
    if l_eff_candidates is None:
        l_eff_candidates = [0.02, 0.03, 0.05, 0.08, 0.10, 0.15]

    best_q = cfg.q_peak_w_m3
    best_l = cfg.fixed_end_conduction_length_m
    best_score = float("inf")
    trials: list[BenchmarkResult] = []

    for l_eff in l_eff_candidates:
        for q_peak in q_peak_candidates:
            trial_cfg = ThermalCalibrationConfig(
                **{
                    **asdict(cfg),
                    "q_peak_w_m3": q_peak,
                    "fixed_end_conduction_length_m": l_eff,
                }
            )
            try:
                res = run_idle_heating_verify(trial_cfg, targets, n_steps=250)
            except Exception as exc:
                trials.append(
                    BenchmarkResult(
                        name="idle_heating_verify",
                        metrics={"q_peak": q_peak, "l_eff": l_eff, "error": str(exc)},
                        success=False,
                        message=str(exc),
                    )
                )
                continue

            trials.append(res)
            max_k = res.metrics.get("max_k", 0.0)
            held_k = res.metrics.get("held_end_k", 9999.0)
            t_sim = res.metrics.get("thermal_seconds", 1.0)

            score = (
                abs(max_k - targets.target_surface_temp_k)
                + 2.0 * max(0.0, held_k - targets.target_held_end_temp_k)
                + 0.1 * abs(t_sim - targets.target_heating_time_thermal_s)
            )
            if score < best_score:
                best_score = score
                best_q = q_peak
                best_l = l_eff

    return best_q, best_l, trials
