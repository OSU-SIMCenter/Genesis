#!/usr/bin/env python3
"""Plot analytical induction heating profiles (Biot-Savart + skin depth).

Validates the same formulas used in the GPU `p2g_induction` kernel without
running a simulation. Defaults match `TeleopOptions` coil geometry.

Usage:
    pixi run python -m agforge.scripts.plot_induction_profiles
    pixi run python -m agforge.scripts.plot_induction_profiles --output induction_profiles.png
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

# Allow running as script or module from repo root
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from agforge.options import TeleopOptions
from agforge.thermal_field import (
    biot_savart_f_axial,
    coil_axial_bounds,
    q_volumetric,
    skin_weight,
)


def _default_coil_center_x(cfg: TeleopOptions) -> float:
    """Coil center when the slider is at the billet midpoint (approximate teleop idle pose)."""
    return float(cfg.robot.cylinder_pos[0])


def plot_profiles(
    cfg: TeleopOptions | None = None,
    coil_center_x: float | None = None,
    output_path: str | None = None,
    show: bool = False,
) -> str:
    cfg = cfg or TeleopOptions()
    half_length = cfg.robot.coil_length / 2.0
    radius = cfg.robot.coil_radius
    skin_depth = cfg.skin_depth
    q_peak = cfg.heating_power
    thermal_time_scale = cfg.mpm.thermal_time_scale

    if coil_center_x is None:
        coil_center_x = _default_coil_center_x(cfg)

    x_min, x_max = coil_axial_bounds(coil_center_x, half_length)
    # Extend past coil ends to show fringing
    pad = half_length * 2.5
    x_abs = np.linspace(x_min - pad, x_max + pad, 500)
    x_rel = x_abs - coil_center_x

    f_axial = biot_savart_f_axial(x_rel, half_length, radius)
    q_surface = q_volumetric(
        x_rel,
        np.zeros_like(x_rel),
        q_peak=q_peak,
        half_length=half_length,
        radius=radius,
        skin_depth=skin_depth,
        thermal_time_scale=thermal_time_scale,
    )

    depths = np.linspace(0.0, skin_depth * 3.0, 200)
    w_skin = skin_weight(depths, skin_depth)

    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=False)

    ax0 = axes[0]
    ax0.plot(x_rel * 1e3, f_axial, color="C0", lw=2)
    ax0.axvline(-half_length * 1e3, color="C3", ls="--", alpha=0.7, label="coil edge")
    ax0.axvline(half_length * 1e3, color="C3", ls="--", alpha=0.7)
    ax0.axvspan(-half_length * 1e3, half_length * 1e3, color="C3", alpha=0.08)
    ax0.set_ylabel(r"$f_{\mathrm{axial}}(x) = B^2/B_0^2$")
    ax0.set_xlabel("Axial offset from coil center [mm]")
    ax0.set_title("Biot–Savart axial profile (finite solenoid)")
    ax0.grid(True, alpha=0.35)
    ax0.legend(loc="upper right")
    ax0.set_ylim(0.0, 1.05)

    ax1 = axes[1]
    ax1.plot(depths * 1e3, w_skin, color="C1", lw=2)
    ax1.axvline(skin_depth * 1e3, color="C3", ls="--", alpha=0.7, label=r"$\delta$")
    ax1.set_ylabel(r"$w_{\mathrm{skin}} = e^{-2d/\delta}$")
    ax1.set_xlabel("Depth below surface [mm]")
    ax1.set_title(f"Skin-effect weight (δ = {skin_depth * 1e3:.2f} mm)")
    ax1.grid(True, alpha=0.35)
    ax1.legend(loc="upper right")

    ax2 = axes[2]
    ax2.plot(x_rel * 1e3, q_surface / 1e6, color="C2", lw=2)
    ax2.axvline(-half_length * 1e3, color="C3", ls="--", alpha=0.7, label="coil edge")
    ax2.axvline(half_length * 1e3, color="C3", ls="--", alpha=0.7)
    ax2.axvspan(-half_length * 1e3, half_length * 1e3, color="C3", alpha=0.08)
    ax2.set_ylabel(r"$\dot{q}$ at surface [MW/m³]")
    ax2.set_xlabel("Axial offset from coil center [mm]")
    ax2.set_title(
        f"Volumetric intensity at d=0  (q_peak={q_peak:.2e} W/m³, S_T={thermal_time_scale:.3g})"
    )
    ax2.grid(True, alpha=0.35)
    ax2.legend(loc="upper right")

    fig.suptitle(
        f"Induction field — coil R={radius * 1e3:.1f} mm, L={2 * half_length * 1e3:.1f} mm, "
        f"center x={coil_center_x:.4f} m",
        fontsize=11,
    )
    fig.tight_layout()

    if output_path is None:
        output_path = os.path.join(os.path.dirname(__file__), "..", "induction_profiles.png")
    output_path = os.path.abspath(output_path)
    fig.savefig(output_path, dpi=150)
    print(f"Saved: {output_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Plot analytical induction heating profiles.")
    parser.add_argument(
        "--output", "-o", type=str, default=None, help="Output PNG path (default: agforge/induction_profiles.png)"
    )
    parser.add_argument(
        "--coil-center-x",
        type=float,
        default=None,
        help="World-frame coil center x [m] (default: billet cylinder center x)",
    )
    parser.add_argument("--show", action="store_true", help="Show interactive window after saving")
    args = parser.parse_args()

    plot_profiles(coil_center_x=args.coil_center_x, output_path=args.output, show=args.show)


if __name__ == "__main__":
    main()
