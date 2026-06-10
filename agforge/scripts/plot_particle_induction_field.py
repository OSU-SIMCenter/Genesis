#!/usr/bin/env python3
"""2D cross-section plots of per-particle induction depth and skin-weight fields.

Builds the teleop scene, uploads the SDF depth field from the surface mesh, and
plots particle scalars on axial (X–Z) and transverse (Y–Z) slices.

Usage:
    pixi run python -m agforge.scripts.plot_particle_induction_field
    pixi run python -m agforge.scripts.plot_particle_induction_field --field q_ind -o field.png
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import genesis as gs
from agforge.agforge_builder import build_env
from agforge.options import TeleopOptions
from agforge.thermal import InductionHeater
from agforge.thermal_field import particle_q_ind, skin_weight


def _slice_mask(pos: np.ndarray, plane: str, coord: float, thickness: float) -> np.ndarray:
    axis = {"xy": 2, "xz": 1, "yz": 0}[plane]
    return np.abs(pos[:, axis] - coord) <= thickness


def _scatter_slice(ax, pos, values, plane: str, coord: float, title: str, cmap: str, vmin, vmax, cbar_label: str):
    if plane == "xz":
        x, y = pos[:, 0], pos[:, 2]
        xlabel, ylabel = "X [m]", "Z [m]"
    elif plane == "yz":
        x, y = pos[:, 1], pos[:, 2]
        xlabel, ylabel = "Y [m]", "Z [m]"
    else:
        x, y = pos[:, 0], pos[:, 1]
        xlabel, ylabel = "X [m]", "Y [m]"

    sc = ax.scatter(x, y, c=values, s=4, cmap=cmap, vmin=vmin, vmax=vmax, linewidths=0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title}\n({plane} slice @ {coord:.4f} m ± thickness)")
    ax.set_aspect("equal")
    plt.colorbar(sc, ax=ax, label=cbar_label)


def plot_fields(
    cfg: TeleopOptions | None = None,
    field: str = "skin_weight",
    output_path: str | None = None,
    plane_primary: str = "xz",
    plane_secondary: str = "yz",
    slice_thickness: float = 0.0005,
    show: bool = False,
) -> str:
    cfg = cfg or TeleopOptions()
    cfg.general.show_viewer = False
    cfg.vis.visualize_mpm_grid = False
    cfg.vis.visualize_mpm_boundary = False

    gs.init(backend=gs.gpu)
    env = build_env(cfg)

    from agforge.reconstruction import SurfaceReconstructor

    reconstructor = SurfaceReconstructor(env, grid_res=cfg.reconstruction.grid_res, backend=cfg.reconstruction.backend)
    heater = InductionHeater(
        solver=env.scene.sim.mpm_solver,
        entity=env.mpm_entity,
        reconstructor=reconstructor,
    )
    reconstructor.create_reconstructed_mesh(is_deforming=False)
    if not heater.recompute_and_upload():
        raise RuntimeError("Failed to upload induction depth field — is the surface mesh valid?")

    # Publish coil uniforms so q_ind uses realistic values
    center_x = float(cfg.robot.cylinder_pos[0])
    env.scene.sim.mpm_solver.set_induction_params(
        center=[center_x, 0.0, float(cfg.robot.cylinder_pos[2])],
        half_length=cfg.robot.coil_length / 2.0,
        radius=cfg.robot.coil_radius,
        q_peak=cfg.heating_power,
        skin_depth=cfg.skin_depth,
        active=True,
    )

    pos = env.mpm_entity.get_particles_pos().detach().cpu().numpy().reshape(-1, 3)
    depths = env.mpm_entity.get_particles_induction_depth().detach().cpu().numpy().reshape(-1)
    skin_depth = float(cfg.skin_depth)
    weights = skin_weight(depths, skin_depth)

    center, half_length, radius, q_peak, sd, _ = env.scene.sim.mpm_solver.get_induction_uniforms_numpy(0)
    q = particle_q_ind(
        pos,
        depths,
        coil_center_x=float(center[0]),
        half_length=half_length,
        radius=radius,
        q_peak=q_peak,
        skin_depth=sd if sd > 0 else skin_depth,
        thermal_time_scale=float(cfg.mpm.thermal_time_scale),
    )

    fields = {
        "induction_depth": (depths, "plasma", 0.0, 3.0 * skin_depth, "Depth [m]"),
        "skin_weight": (weights, "inferno", 0.0, 1.0, r"$e^{-2d/\delta}$"),
        "q_ind": (q, "inferno", 0.0, float(np.percentile(q, 99.5)), r"$\dot{q}$ [W/m³]"),
    }
    if field not in fields:
        raise ValueError(f"Unknown field '{field}'. Choose from {list(fields)}")
    values, cmap, vmin, vmax, cbar_label = fields[field]

    cyl = cfg.robot.cylinder_pos
    slice_coords = {
        "xz": float(cyl[1]),
        "yz": float(cyl[0]),
        "xy": float(cyl[2]),
    }

    fig, axes = plt.subplots(2, 2, figsize=(13, 11))

    for ax, plane in zip(axes[0], (plane_primary, plane_secondary)):
        coord = slice_coords[plane]
        mask = _slice_mask(pos, plane, coord, slice_thickness)
        if mask.sum() == 0:
            ax.set_title(f"No particles in {plane} slice")
            continue
        _scatter_slice(ax, pos[mask], values[mask], plane, coord, field, cmap, vmin, vmax, cbar_label)

    ax_hist = axes[1, 0]
    finite = np.isfinite(depths)
    ax_hist.hist(depths[finite], bins=60, color="C0", alpha=0.85)
    ax_hist.axvline(skin_depth, color="C3", ls="--", label=rf"$\delta$ = {skin_depth*1e3:.2f} mm")
    ax_hist.set_xlabel("SDF depth [m]")
    ax_hist.set_ylabel("Particle count")
    ax_hist.set_title("Depth distribution (all particles)")
    ax_hist.legend()
    ax_hist.grid(True, alpha=0.3)

    ax_prof = axes[1, 1]
    x = pos[:, 0]
    order = np.argsort(x)
    x_sorted = x[order]
    # Bin-mean profiles along billet axis
    n_bins = 40
    edges = np.linspace(x_sorted.min(), x_sorted.max(), n_bins + 1)
    bin_centers = 0.5 * (edges[:-1] + edges[1:])
    mean_depth = np.zeros(n_bins)
    mean_w = np.zeros(n_bins)
    for i in range(n_bins):
        m = (x >= edges[i]) & (x < edges[i + 1])
        if m.any():
            mean_depth[i] = depths[m].mean()
            mean_w[i] = weights[m].mean()
    ax_prof.plot(bin_centers * 1e3, mean_depth * 1e3, label="mean depth [mm]", color="C0")
    ax_prof2 = ax_prof.twinx()
    ax_prof2.plot(bin_centers * 1e3, mean_w, label=r"mean $e^{-2d/\delta}$", color="C1")
    ax_prof.set_xlabel("X [mm]")
    ax_prof.set_ylabel("Mean SDF depth [mm]", color="C0")
    ax_prof2.set_ylabel(r"Mean skin weight", color="C1")
    ax_prof.set_title("Axial depth / skin-weight profile")
    ax_prof.grid(True, alpha=0.3)

    fig.suptitle(
        f"Induction SDF field — δ={skin_depth*1e3:.2f} mm, "
        f"displaying '{field}' (surface mesh SDF → exp(-2d/δ) in kernel)",
        fontsize=11,
    )
    fig.tight_layout()

    if output_path is None:
        output_path = os.path.join(os.path.dirname(__file__), "..", f"particle_{field}_field.png")
    output_path = os.path.abspath(output_path)
    fig.savefig(output_path, dpi=150)
    print(f"Saved: {output_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Plot particle induction depth / skin-weight cross-sections.")
    parser.add_argument(
        "--field",
        choices=["induction_depth", "skin_weight", "q_ind"],
        default="skin_weight",
        help="Scalar to color slices (default: skin_weight)",
    )
    parser.add_argument("--output", "-o", default=None)
    parser.add_argument("--plane", default="xz", choices=["xy", "xz", "yz"])
    parser.add_argument("--plane2", default="yz", choices=["xy", "xz", "yz"])
    parser.add_argument("--thickness", type=float, default=0.0005, help="Half-thickness of slice [m]")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    plot_fields(
        field=args.field,
        output_path=args.output,
        plane_primary=args.plane,
        plane_secondary=args.plane2,
        slice_thickness=args.thickness,
        show=args.show,
    )


if __name__ == "__main__":
    main()
