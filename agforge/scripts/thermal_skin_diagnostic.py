#!/usr/bin/env python3
"""Induction skin-depth / SDF diagnostic — mesh fidelity vs. analytic cylinder.

This answers one question before any parameter tuning: **is the per-particle
induction depth field trustworthy, or is the reconstructed-mesh SDF introducing
error?** It builds the *production* induction path (InductionPhysicsMesher →
igl SDF, exactly what teleop heats with), then compares the mesh-derived depth
against the analytic cylindrical depth computed straight from the particle cloud.

It separates three independent effects that all make heating "look shallow":

  1. Mesh fidelity   — mesh_depth - analytic_nearest_depth.  A constant offset is
                       an iso-level bias; spread is faceting/quantization noise.
  2. Coil-facing gap — analytic_nearest_depth vs analytic_radial_depth.  Shows
                       where the |SDF| "nearest surface anywhere" picks the end
                       cap instead of the lateral (coil-facing) surface.
  3. Convention/regime — w = exp(-2 d / delta) e-folds power at delta/2, and a
                       fixed thin delta is only valid below the Curie point.

Outputs a full text report (works headless) and an optional PNG.

Usage:
    pixi run python -m agforge.scripts.thermal_skin_diagnostic
    pixi run python -m agforge.scripts.thermal_skin_diagnostic --backend hybrid_high -o skin_diag.png
    pixi run python -m agforge.scripts.thermal_skin_diagnostic --no-plot
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import genesis as gs
from agforge.agforge_builder import build_env
from agforge.options import TeleopOptions
from agforge.thermal import InductionHeater
from agforge.thermal_field import skin_weight


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _detect_axis(pos: np.ndarray) -> int:
    """Return the index (0/1/2) of the longest particle-cloud extent (the cylinder axis)."""
    extent = pos.max(axis=0) - pos.min(axis=0)
    return int(np.argmax(extent))


def analytic_cylinder_depths(pos: np.ndarray, axis: int):
    """Compute analytic depths from the particle cloud itself (no mesh).

    Returns a dict with, per particle (metres):
      r                  radial distance from the cloud axis
      cloud_radius       robust outer radius of the cloud (scalar)
      depth_radial       cloud_radius - r  (lateral / coil-facing depth, clamped >=0)
      depth_nearest      min(lateral, free-cap, held-cap)  — honest analog of |SDF|
      which              0=lateral, 1=free cap, 2=held cap  (which surface is nearest)
    """
    lat_axes = [a for a in range(3) if a != axis]
    a1, a2 = lat_axes
    # Axis location from the cloud centroid (robust to small placement offsets).
    c1 = float(np.median(pos[:, a1]))
    c2 = float(np.median(pos[:, a2]))
    r = np.sqrt((pos[:, a1] - c1) ** 2 + (pos[:, a2] - c2) ** 2)

    # Outer radius / end-cap planes from the actual particle envelope.
    cloud_radius = float(np.percentile(r, 99.5))
    x = pos[:, axis]
    x_free = float(np.percentile(x, 0.1))
    x_held = float(np.percentile(x, 99.9))

    depth_radial = np.clip(cloud_radius - r, 0.0, None)
    depth_free = np.clip(x - x_free, 0.0, None)
    depth_held = np.clip(x_held - x, 0.0, None)

    stacked = np.stack([depth_radial, depth_free, depth_held], axis=1)
    which = np.argmin(stacked, axis=1)
    depth_nearest = stacked.min(axis=1)

    return {
        "r": r,
        "cloud_radius": cloud_radius,
        "x_free": x_free,
        "x_held": x_held,
        "depth_radial": depth_radial,
        "depth_nearest": depth_nearest,
        "which": which,
        "axis_center": (c1, c2),
        "lat_axes": (a1, a2),
    }


# ---------------------------------------------------------------------------
# Scene build (mirrors strike_controller production induction path)
# ---------------------------------------------------------------------------

def build_production_depths(cfg: TeleopOptions, backend: str | None, use_reconstructor: bool):
    """Build env + production physics mesh, upload SDF, return (pos, mesh_depth, mesh, info)."""
    cfg.general.show_viewer = False
    cfg.vis.visualize_mpm_grid = False
    cfg.vis.visualize_mpm_boundary = False

    gs.init(backend=gs.gpu)
    env = build_env(cfg)

    from agforge.reconstruction import SurfaceReconstructor

    recon_cfg = cfg.reconstruction
    reconstructor = SurfaceReconstructor(
        env,
        grid_res=recon_cfg.grid_res,
        backend=recon_cfg.backend,
        physics_grid_res=recon_cfg.physics_mesh_grid_res,
        mc_backend=getattr(recon_cfg, "mc_backend", "auto"),
    )

    mesh = None
    physics_mesher = None
    if not use_reconstructor:
        from agforge.physics_mesh import InductionPhysicsMesher

        physics_mesher = InductionPhysicsMesher(env, reconstructor, cfg)
        if backend is not None:
            physics_mesher.set_backend(backend)
        if not physics_mesher.rebuild():
            gs.logger.warning("Physics mesh rebuild failed; falling back to plain reconstructor.")
            physics_mesher = None
        else:
            v, fcs = physics_mesher.get_verts_faces()
            if v is not None:
                import trimesh

                mesh = trimesh.Trimesh(vertices=v, faces=fcs, process=False)

    if physics_mesher is None:
        reconstructor.create_reconstructed_mesh(is_deforming=False)
        mesh = reconstructor.reconstructed_mesh

    heater = InductionHeater(
        solver=env.scene.sim.mpm_solver,
        entity=env.mpm_entity,
        physics_mesher=physics_mesher,
        reconstructor=reconstructor,
    )
    if not heater.recompute_and_upload():
        raise RuntimeError("Failed to upload induction depth field — surface mesh invalid?")

    pos = env.mpm_entity.get_particles_pos().detach().cpu().numpy().reshape(-1, 3).astype(np.float64)
    mesh_depth = env.mpm_entity.get_particles_induction_depth().detach().cpu().numpy().reshape(-1).astype(np.float64)

    info = {
        "n_particles": pos.shape[0],
        "mesh_verts": 0 if mesh is None else len(mesh.vertices),
        "mesh_faces": 0 if mesh is None else len(mesh.faces),
        "mesh_source": "reconstructor" if physics_mesher is None else f"physics_mesher[{physics_mesher.backend.value}]",
    }
    return pos, mesh_depth, mesh, info


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _fmt_mm(x) -> str:
    return f"{x * 1e3:8.3f} mm"


def run(cfg: TeleopOptions, backend: str | None, use_reconstructor: bool, plot_path: str | None):
    skin_depth = float(cfg.skin_depth)
    dx = 1.0 / float(cfg.robot.base_grid_density)
    dp = float(getattr(cfg.mpm, "particle_size", dx / 2.0))  # nominal inter-particle spacing
    R = float(cfg.robot.cylinder_radius)

    pos, mesh_depth, mesh, info = build_production_depths(cfg, backend, use_reconstructor)
    axis = _detect_axis(pos)
    geo = analytic_cylinder_depths(pos, axis)

    # ---- VERIFICATION: rule out artifacts in this script before trusting the verdict ----
    print("\n" + "-" * 78)
    print("  [0] VERIFICATION  (is the production depth field self-consistent?)")
    print("-" * 78)
    a1, a2 = geo["lat_axes"]
    if mesh is not None:
        v = np.asarray(mesh.vertices, dtype=np.float64)
        c1, c2 = geo["axis_center"]
        v_r = np.sqrt((v[:, a1] - c1) ** 2 + (v[:, a2] - c2) ** 2)
        print(f"  mesh vertex radius   : min {_fmt_mm(v_r.min())}  median {_fmt_mm(np.median(v_r))}  max {_fmt_mm(v_r.max())}")
        print(f"  mesh axial extent    : {_fmt_mm(v[:, axis].min())} .. {_fmt_mm(v[:, axis].max())}")
        print(f"  cloud radius (ref)   : {_fmt_mm(geo['cloud_radius'])}   (mesh should hug this if faithful)")
        fcs = np.asarray(mesh.faces, dtype=np.int64)
        # Mesh quality
        nan_v = int(np.sum(~np.isfinite(v)))
        bad_idx = int(np.sum((fcs < 0) | (fcs >= len(v))))
        tri = v[fcs]
        e0 = tri[:, 1] - tri[:, 0]
        e1 = tri[:, 2] - tri[:, 0]
        area2 = np.linalg.norm(np.cross(e0, e1), axis=1)
        degen = int(np.sum(area2 < 1e-14))
        print(f"  mesh quality         : watertight={getattr(mesh, 'is_watertight', '?')}  NaN_verts={nan_v}  bad_face_idx={bad_idx}  degenerate_faces={degen}/{len(fcs)}")
        # Recompute distance two ways: the buggy signed_distance (winding-number default, 3x on
        # MC meshes) and the correct point_mesh_squared_distance. The production field should now
        # match the latter (after the thermal.py fix).
        try:
            import igl

            fcs_i32 = np.asarray(mesh.faces, dtype=np.int32)
            d_signed = np.abs(igl.signed_distance(pos, v, fcs_i32)[0])
            d_correct = np.sqrt(np.maximum(igl.point_mesh_squared_distance(pos, v, fcs_i32)[0], 0.0))
            print(f"  igl.signed_distance  : median {_fmt_mm(np.nanmedian(d_signed))}  max {_fmt_mm(np.nanmax(d_signed))}  (buggy: winding-number 3x on MC meshes)")
            print(f"  point_mesh_sqr_dist  : median {_fmt_mm(np.nanmedian(d_correct))}  max {_fmt_mm(np.nanmax(d_correct))}  (correct reference)")
            diff = np.abs(d_correct - mesh_depth)
            print(f"  field vs correct ref : max|diff| {_fmt_mm(np.nanmax(diff))}  mean|diff| {_fmt_mm(np.nanmean(diff))}  (~0 => thermal.py fix active)")
        except Exception as e:
            print(f"  (manual igl check skipped: {e})")
        # Independent ground truth: KD-tree nearest mesh VERTEX distance (upper bound on
        # true nearest-surface distance). If this is sane but igl is 3x bigger, igl/BVH is wrong.
        try:
            from scipy.spatial import cKDTree

            d_kd = cKDTree(v).query(pos, k=1)[0]
            print(f"  KDtree nearest-vert  : min {_fmt_mm(np.nanmin(d_kd))}  median {_fmt_mm(np.nanmedian(d_kd))}  max {_fmt_mm(np.nanmax(d_kd))}")
            print(f"  igl/KDtree ratio     : median {np.nanmedian(mesh_depth)/max(np.nanmedian(d_kd),1e-9):.2f}x  (should be <=1; igl=surface, KD=vertex)")
        except Exception as e:
            print(f"  (KDtree check skipped: {e})")
        # Dump for offline forensics (no GPU rebuild needed to iterate).
        dump = os.path.join(os.path.dirname(__file__), "..", "skin_diag_dump.npz")
        np.savez(os.path.abspath(dump), pos=pos, verts=v, faces=fcs, field=mesh_depth)
        print(f"  dumped mesh+field    : {os.path.abspath(dump)}")
    print(f"  field depth (in use) : min {_fmt_mm(np.nanmin(mesh_depth))}  median {_fmt_mm(np.nanmedian(mesh_depth))}  max {_fmt_mm(np.nanmax(mesh_depth))}")
    print(f"     -> a faithful field for this billet should be ~0 at surface, ~{_fmt_mm(geo['cloud_radius'])} at the axis.")

    finite = np.isfinite(mesh_depth)
    # Apples-to-apples fidelity error: both are "nearest surface" depths.
    err = mesh_depth - geo["depth_nearest"]

    # Mid-length lateral particles isolate the radial behavior from end-cap effects.
    x = pos[:, axis]
    mid = (x > np.percentile(x, 20)) & (x < np.percentile(x, 80))
    lateral = geo["which"] == 0
    core_mask = finite & mid & lateral

    bias = float(np.mean(err[core_mask])) if core_mask.any() else float("nan")
    noise = float(np.std(err[core_mask])) if core_mask.any() else float("nan")

    # Implied mesh radius (lateral interior): mesh_depth ≈ R_mesh - r  =>  R_mesh ≈ r + mesh_depth.
    implied_radius = geo["r"][core_mask] + mesh_depth[core_mask]
    iso_offset = float(np.median(implied_radius) - geo["cloud_radius"]) if core_mask.any() else float("nan")

    # Surface mis-classification: particles that should be ~surface but mesh calls deep.
    should_surface = finite & (geo["depth_radial"] < dp)
    if should_surface.any():
        mis_frac = float(np.mean(mesh_depth[should_surface] > 2.0 * dp))
        surf_depth_med = float(np.median(mesh_depth[should_surface]))
    else:
        mis_frac = float("nan")
        surf_depth_med = float("nan")

    # ---- text report ----
    print("\n" + "=" * 78)
    print("  INDUCTION SKIN-DEPTH / SDF DIAGNOSTIC")
    print("=" * 78)
    print(f"  particles            : {info['n_particles']}")
    print(f"  mesh source          : {info['mesh_source']}  ({info['mesh_verts']} verts, {info['mesh_faces']} faces)")
    print(f"  cylinder axis        : {'XYZ'[axis]}")
    print(f"  nominal billet radius: {_fmt_mm(R)}    cloud outer radius: {_fmt_mm(geo['cloud_radius'])}")
    print(f"  grid dx              : {_fmt_mm(dx)}    particle spacing dp: {_fmt_mm(dp)}")
    print(f"  skin_depth delta     : {_fmt_mm(skin_depth)}  (= R/{R/skin_depth:.2f})")
    print(f"  power e-folding depth : {_fmt_mm(skin_depth/2)}  (delta/2, because w = exp(-2 d/delta))")

    print("\n" + "-" * 78)
    print("  [1] MESH FIDELITY   (mesh_depth - analytic_nearest_depth, lateral mid-length)")
    print("-" * 78)
    print(f"  bias (mean error)    : {_fmt_mm(bias)}   ({bias/dp:+.2f} particle spacings)")
    print(f"  noise (std error)    : {_fmt_mm(noise)}   ({noise/dp:.2f} particle spacings)")
    print(f"  iso-level offset     : {_fmt_mm(iso_offset)}   (implied mesh radius - cloud radius)")
    print(f"     -> mesh sits {'OUTSIDE' if iso_offset > 0 else 'INSIDE'} the particle cloud by ~{abs(iso_offset)*1e3:.2f} mm")
    print(f"  surface particles (analytic depth < dp): median mesh_depth = {_fmt_mm(surf_depth_med)}")
    print(f"     -> fraction the mesh mis-labels as deep (>2 dp): {mis_frac*100:5.1f} %")

    print("\n" + "-" * 78)
    print("  [2] RADIAL DEPTH PROFILE (mid-length lateral particles, by layer)")
    print("-" * 78)
    print(f"  {'r [mm]':>8} {'analytic d':>12} {'mesh d':>12} {'w(analytic)':>12} {'w(mesh)':>10} {'n':>6}")
    n_layers = max(4, int(round(geo["cloud_radius"] / dp)))
    edges = np.linspace(0.0, geo["cloud_radius"], n_layers + 1)
    prof = []
    for i in range(n_layers):
        lo, hi = edges[i], edges[i + 1]
        m = core_mask & (geo["r"] >= lo) & (geo["r"] < hi)
        if not m.any():
            continue
        r_mid = 0.5 * (lo + hi)
        a_d = float(np.mean(geo["depth_radial"][m]))
        m_d = float(np.mean(mesh_depth[m]))
        w_a = float(np.mean(skin_weight(geo["depth_radial"][m], skin_depth)))
        w_m = float(np.mean(skin_weight(mesh_depth[m], skin_depth)))
        prof.append((r_mid, a_d, m_d, w_a, w_m, int(m.sum())))
        print(f"  {r_mid*1e3:8.2f} {_fmt_mm(a_d)} {_fmt_mm(m_d)} {w_a:12.3f} {w_m:10.3f} {int(m.sum()):6d}")

    print("\n" + "-" * 78)
    print("  [3] CONVENTION / REGIME  (independent of mesh)")
    print("-" * 78)
    for k in range(5):
        d = k * dp
        print(f"  depth {k} layer(s) ({_fmt_mm(d)}): w_skin = {np.exp(-2.0*d/skin_depth):.3f}")
    # delta needed for ~50% power at one billet-radius depth (through-heating intent)
    delta_for_through = 2.0 * R / np.log(2.0)
    print(f"  delta for 50% power at depth R ({_fmt_mm(R)}): {_fmt_mm(delta_for_through)}  (current is {_fmt_mm(skin_depth)})")
    print(f"  NOTE: above Curie (~1043 K) real steel skin depth >~10 mm; fixed thin delta under-penetrates the hot regime.")

    # ---- verdict ----
    print("\n" + "=" * 78)
    print("  VERDICT")
    print("=" * 78)
    mesh_bad = (abs(bias) >= dp) or (noise >= dp) or (np.isfinite(mis_frac) and mis_frac > 0.10)
    if mesh_bad:
        print("  >> MESH FIDELITY is a MATERIAL contributor.")
        print(f"     bias={bias*1e3:+.2f} mm, noise={noise*1e3:.2f} mm vs particle spacing {dp*1e3:.2f} mm.")
        print("     Recommend Option A: analytic cylindrical depth (depth = cloud_radius - r),")
        print("     which removes the reconstruction/iso-level error entirely and is cheaper.")
    else:
        print("  >> MESH FIDELITY looks OK (error << particle spacing).")
        print("     The shallow-heating look is then dominated by CONVENTION/REGIME, not the mesh:")
        print("     the exp(-2d/delta) factor-of-2 makes power e-fold at delta/2 (~one layer), and a")
        print("     fixed thin delta is wrong above the Curie point. Fix delta (larger / T-dependent).")
    print("  Either way, enlarging/temperature-ramping delta desensitizes the model to mesh error.")
    print("=" * 78 + "\n")

    if plot_path is not None:
        _make_plot(pos, axis, geo, mesh_depth, err, core_mask, skin_depth, dp, prof, plot_path)


def _make_plot(pos, axis, geo, mesh_depth, err, core_mask, skin_depth, dp, prof, plot_path):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        gs.logger.warning(f"matplotlib unavailable, skipping plot: {e}")
        return

    fig, axes = plt.subplots(2, 2, figsize=(13, 11))

    # (a) the money plot: mesh depth vs analytic nearest depth. Perfect mesh -> y=x.
    ax = axes[0, 0]
    d_an = geo["depth_nearest"][core_mask] * 1e3
    d_me = mesh_depth[core_mask] * 1e3
    ax.scatter(d_an, d_me, s=4, alpha=0.4, linewidths=0)
    lim = max(d_an.max(), d_me.max()) if core_mask.any() else 1.0
    ax.plot([0, lim], [0, lim], "r--", label="y = x (perfect mesh)")
    ax.set_xlabel("analytic nearest depth [mm]")
    ax.set_ylabel("mesh (igl) depth [mm]")
    ax.set_title("Mesh vs analytic depth (lateral mid-length)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (b) error histogram
    ax = axes[0, 1]
    ax.hist(err[core_mask] * 1e3, bins=50, color="C1", alpha=0.85)
    ax.axvline(0, color="k", lw=1)
    ax.axvline(dp * 1e3, color="C3", ls="--", label=f"+1 dp ({dp*1e3:.2f} mm)")
    ax.axvline(-dp * 1e3, color="C3", ls="--")
    ax.set_xlabel("mesh_depth - analytic_nearest [mm]")
    ax.set_ylabel("particles")
    ax.set_title("Mesh fidelity error")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (c) radial w_skin profile: analytic vs mesh
    ax = axes[1, 0]
    if prof:
        r_mid = np.array([p[0] for p in prof]) * 1e3
        w_a = np.array([p[3] for p in prof])
        w_m = np.array([p[4] for p in prof])
        ax.plot(r_mid, w_a, "o-", label="w(analytic depth)", color="C0")
        ax.plot(r_mid, w_m, "s-", label="w(mesh depth)", color="C1")
    ax.set_xlabel("radius r [mm]   (surface at right)")
    ax.set_ylabel(r"mean skin weight $e^{-2d/\delta}$")
    ax.set_title("Radial heating profile by layer")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (d) transverse slice colored by mesh depth
    ax = axes[1, 1]
    a1, a2 = geo["lat_axes"]
    xq = pos[:, a1] * 1e3
    yq = pos[:, a2] * 1e3
    sc = ax.scatter(xq, yq, c=mesh_depth * 1e3, s=6, cmap="plasma", linewidths=0)
    ax.set_aspect("equal")
    ax.set_xlabel(f"{'XYZ'[a1]} [mm]")
    ax.set_ylabel(f"{'XYZ'[a2]} [mm]")
    ax.set_title("Mesh depth, transverse view")
    plt.colorbar(sc, ax=ax, label="mesh depth [mm]")

    fig.suptitle("Induction skin-depth / SDF diagnostic", fontsize=12)
    fig.tight_layout()
    plot_path = os.path.abspath(plot_path)
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot: {plot_path}")


def main():
    parser = argparse.ArgumentParser(description="Induction skin-depth / SDF mesh-fidelity diagnostic.")
    parser.add_argument("--backend", default=None, choices=["hybrid_low", "hybrid_high", "splashsurf"],
                        help="Override physics_mesh_backend (default: config value).")
    parser.add_argument("--reconstructor", action="store_true",
                        help="Use the plain visual reconstructor instead of the physics mesher.")
    parser.add_argument("--output", "-o", default=None, help="PNG output path (default: agforge/skin_diagnostic.png).")
    parser.add_argument("--no-plot", action="store_true", help="Text report only.")
    args = parser.parse_args()

    cfg = TeleopOptions()
    plot_path = None
    if not args.no_plot:
        plot_path = args.output or os.path.join(os.path.dirname(__file__), "..", "skin_diagnostic.png")

    run(cfg, backend=args.backend, use_reconstructor=args.reconstructor, plot_path=plot_path)


if __name__ == "__main__":
    main()
