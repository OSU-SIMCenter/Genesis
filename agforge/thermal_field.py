"""Analytical induction heating field helpers (mirrors the MPM `p2g_induction` kernel).

Used for offline validation plots — same formulas as
`genesis/engine/solvers/base_mpm_solver.py::p2g_induction`.
"""

from __future__ import annotations

import numpy as np

_EPS = 1e-12


def biot_savart_b_axial(
    x: np.ndarray,
    half_length: float,
    radius: float,
) -> np.ndarray:
    """On-axis magnetic field B(x) for a finite solenoid (arbitrary units, linear in current)."""
    x = np.asarray(x, dtype=np.float64)
    h = float(half_length)
    r = float(radius)
    t1 = (x + h) / np.sqrt((x + h) ** 2 + r ** 2)
    t2 = (x - h) / np.sqrt((x - h) ** 2 + r ** 2)
    return 0.5 * (t1 - t2)


def biot_savart_f_axial(
    x: np.ndarray,
    half_length: float,
    radius: float,
) -> np.ndarray:
    """Normalized axial profile f_axial(x) = B(x)^2 / B(0)^2."""
    x = np.asarray(x, dtype=np.float64)
    h = float(half_length)
    r = float(radius)
    b = biot_savart_b_axial(x, h, r)
    b_peak = h / np.sqrt(h * h + r * r)
    return (b * b) / np.maximum(b_peak * b_peak, _EPS)


def skin_weight(depth: np.ndarray, skin_depth: float) -> np.ndarray:
    """Skin-effect power weight w_skin = exp(-2 * depth / delta)."""
    depth = np.asarray(depth, dtype=np.float64)
    delta = float(skin_depth)
    if delta <= 0.0:
        return np.zeros_like(depth)
    return np.exp(-2.0 * depth / delta)


def q_volumetric(
    x_axial: np.ndarray,
    depth: np.ndarray,
    *,
    q_peak: float,
    half_length: float,
    radius: float,
    skin_depth: float,
    thermal_time_scale: float = 1.0,
) -> np.ndarray:
    """Peak volumetric heating rate q_dot [W/m^3] at (x, depth).

    q_dot = q_peak * S_T * f_axial(x) * exp(-2 * depth / delta)
    """
    f_ax = biot_savart_f_axial(x_axial, half_length, radius)
    w = skin_weight(depth, skin_depth)
    return float(q_peak) * float(thermal_time_scale) * f_ax * w


def coil_axial_bounds(coil_center_x: float, half_length: float) -> tuple[float, float]:
    """Return (x_min, x_max) of the coil interior along the solenoid axis."""
    c = float(coil_center_x)
    h = float(half_length)
    return c - h, c + h


def particle_q_ind(
    positions: np.ndarray,
    depths: np.ndarray,
    *,
    coil_center_x: float,
    half_length: float,
    radius: float,
    q_peak: float,
    skin_depth: float,
    thermal_time_scale: float = 1.0,
) -> np.ndarray:
    """Per-particle volumetric heating rate [W/m^3] using the engine's induction model."""
    pos = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    depth = np.asarray(depths, dtype=np.float64).reshape(-1)
    x_axial = pos[:, 0] - float(coil_center_x)
    return q_volumetric(
        x_axial,
        depth,
        q_peak=q_peak,
        half_length=half_length,
        radius=radius,
        skin_depth=skin_depth,
        thermal_time_scale=thermal_time_scale,
    )
