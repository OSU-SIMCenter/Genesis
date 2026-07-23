"""Thermal time-scale (S_T) should not change steady-state temperature.

Induction heating is linear in S_T·dt. Surface cooling must use the same linear
scaling — exponential (1-exp(-h·S_T·dt)) saturates when h·S_T·dt is O(1) and
breaks equilibrium when S_T is changed at runtime.
"""

from __future__ import annotations

import math

from agforge.thermal_calibration import (
    linearized_radiation_h,
    lumped_surface_cooling_rate_k_per_s,
    surface_induction_heating_rate_k_per_s,
)


def _linear_surface_cooling_step(
    temp_k: float,
    *,
    s_t: float,
    dt: float,
    h_air_base: float,
    emissivity: float,
    area_m2: float,
    mass_kg: float,
    t_amb: float,
    rho: float = 7850.0,
    cp: float = 450.0,
) -> float:
    """One substep of linear surface cooling (matches GPU kernel after fix)."""
    h_rad = linearized_radiation_h(temp_k, t_amb, emissivity)
    h_total = h_air_base + h_rad
    dt_th = dt * s_t
    return h_total * area_m2 * (temp_k - t_amb) * dt_th / (mass_kg * cp)


def _lumped_step_linear(
    temp_k: float,
    *,
    q_peak: float,
    s_t: float,
    dt: float,
    h_air: float,
    emissivity: float,
    area_m2: float,
    mass_kg: float,
    t_amb: float,
    rho: float = 7850.0,
    cp: float = 450.0,
) -> float:
    dt_th = dt * s_t
    dT_heat = q_peak * dt_th / (rho * cp)
    dT_cool = _linear_surface_cooling_step(
        temp_k,
        s_t=s_t,
        dt=dt,
        h_air_base=h_air,
        emissivity=emissivity,
        area_m2=area_m2,
        mass_kg=mass_kg,
        t_amb=t_amb,
        rho=rho,
        cp=cp,
    )
    return temp_k + dT_heat - dT_cool


def _lumped_step_exponential(
    temp_k: float,
    *,
    q_peak: float,
    s_t: float,
    dt: float,
    h_air: float,
    emissivity: float,
    area_m2: float,
    mass_kg: float,
    t_amb: float,
    rho: float = 7850.0,
    cp: float = 450.0,
) -> float:
    """Legacy exponential surface cooling (saturates at large S_T)."""
    dT_heat = q_peak * s_t * dt / (rho * cp)
    h_rad = linearized_radiation_h(temp_k, t_amb, emissivity)
    h_total = (h_air + h_rad) * s_t
    k = h_total * area_m2 * dt / (mass_kg * cp)
    dT_cool = (temp_k - t_amb) * (1.0 - math.exp(-k))
    return temp_k + dT_heat - dT_cool


def _run_to_steady(step_fn, s_t: float, *, steps: int = 12000, dt: float = 2e-4) -> float:
    t_amb = 293.15
    q_peak = 2.5e8
    h_air = 15.0
    emissivity = 0.8
    area = 0.02
    mass = 0.5
    temp = t_amb
    for _ in range(steps):
        temp = step_fn(
            temp,
            q_peak=q_peak,
            s_t=s_t,
            dt=dt,
            h_air=h_air,
            emissivity=emissivity,
            area_m2=area,
            mass_kg=mass,
            t_amb=t_amb,
        )
    return temp


def test_linear_lumped_equilibrium_independent_of_s_t():
    t_lo = _run_to_steady(_lumped_step_linear, 400.0)
    t_hi = _run_to_steady(_lumped_step_linear, 12000.0)
    assert abs(t_hi - t_lo) < 0.1


def test_exponential_lumped_equilibrium_depends_on_s_t():
    """Documents the bug: exp cooling breaks S_T invariance at large S_T."""
    t_lo = _run_to_steady(_lumped_step_exponential, 400.0)
    t_hi = _run_to_steady(_lumped_step_exponential, 12000.0)
    assert abs(t_hi - t_lo) > 5.0


def test_heating_and_cooling_rates_cancel_s_t():
    temp = 900.0
    s_t = 3000.0
    q_in = surface_induction_heating_rate_k_per_s(2.5e8, s_t, temp)
    q_out = lumped_surface_cooling_rate_k_per_s(
        temp,
        293.15,
        0.02,
        0.5,
        h_air=15.0,
        emissivity=0.8,
        thermal_time_scale=s_t,
    )
    q_in2 = surface_induction_heating_rate_k_per_s(2.5e8, s_t * 0.25, temp)
    q_out2 = lumped_surface_cooling_rate_k_per_s(
        temp,
        293.15,
        0.02,
        0.5,
        h_air=15.0,
        emissivity=0.8,
        thermal_time_scale=s_t * 0.25,
    )
    assert abs((q_in / q_out) - (q_in2 / q_out2)) < 1e-9


if __name__ == "__main__":
    test_linear_lumped_equilibrium_independent_of_s_t()
    test_exponential_lumped_equilibrium_depends_on_s_t()
    test_heating_and_cooling_rates_cancel_s_t()
    print("ok")
