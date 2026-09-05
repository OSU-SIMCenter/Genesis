"""Animated side-by-side comparison of contact methods against the real scan.

LAYOUT: columns = real scan + one per contact method; rows = three views of the same instant.
  row 0  isometric A                    overall shape / elongation
  row 1  isometric B (opposite azimuth) the other side, so a dent hidden by the silhouette in A
                                        is still visible
  row 2  LONGITUDINAL CROSS-SECTION     a mid-plane slab. This row earns its place: MPM
                                        particles are volumetric, so a section shows sub-surface
                                        piling and how material actually flows under the die,
                                        which no exterior view can. The real scan is a surface
                                        line-scan with NO interior, so its section is an outline
                                        -- that asymmetry is real, not a rendering artifact.

FRAMING IS FIXED, NOT AUTO-FIT: one shared parallel-projection scale and focal point computed
over every arm and every hit. Auto-framing rescales each panel to fill itself, which would make
a bar that elongates look identical to one that stalled -- precisely the difference under
examination (the domain-limit bug was exactly that failure).

WHY TILE-AND-COMPOSITE rather than a pyvista subplot grid: a (3 x 4) Plotter allocates its whole
multi-viewport framebuffer up front and dies with `LLVM ERROR: out of memory` / `std::bad_alloc`
under this VM's software GL, which other agents share. One small reused render context per tile,
composited with PIL, keeps peak memory flat and also allows real text rendering.

Arms that go unstable HOLD their last good frame and are labelled FAILED (matching
forge_common.viz.render_common_gif's convention) so a short arm never silently vanishes.
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.expanduser("~/GitHub/Genesis/forge_common/main"))

# Deliberately NOT `from geom_metrics import real_mesh, psize`. That module sets
# RLIMIT_AS to 3 GB at import time (a guard added after trimesh.contains() killed this VM), and
# VTK's software GL reserves far more than that in VIRTUAL address space -- so importing it here
# makes rendering die as `LLVM ERROR: out of memory` / `std::bad_alloc` with no traceback. Both
# helpers are a few lines and neither needs trimesh, since only vertices are used.
REAL_MESH_DIR = os.path.expanduser("~/GitHub/Genesis/forge_common/main/outputs/real_meshes")
SIM_STOCK_R_MM, SIM_STOCK_L_MM = 20.0, 59.0


def real_surface(hit, side="after"):
    """Vertices AND triangles of the real billet surface scan after `hit` (mm).

    The faces matter: drawn as a solid surface the real part reads as a shape, whereas a
    randomly-subsampled point cloud next to the sims' regular particle lattice reads as noise and
    makes the eye compare sampling density instead of geometry.
    """
    p = os.path.join(REAL_MESH_DIR, "hit_%02d.npz" % hit)
    if not os.path.exists(p):
        raise SystemExit("missing %s -- run agforge/analysis/extract_real_meshes.py once" % p)
    with np.load(p) as z:
        return np.array(z["V_" + side], dtype=np.float64), np.array(z["T_" + side], dtype=np.int64)


def psize(n):
    """Particle lattice spacing in mm.

    psize is a property of the GRID -- the sim sets particle_size = dx / AGF_PPC_DIVISOR --
    NOT of how much material got seeded. Backing it out of the nominal cylinder volume only
    ever agreed with that because a Cylinder morph fills exactly the nominal volume.

    Seed the billet from a mesh (AGF_BILLET_MESH) and N drops ~10% while the nominal volume
    does not, so this derivation reads high: MEASURED 2.0696 mm against a true 2.0000 mm,
    +3.48%. Every metric built on psize -- packing eta, the IoU cube, surface deviation --
    then shifts silently. Set AGF_PSIZE_MM when scoring such a run; batch_arms writes the
    true value (read straight off the entity) into the batch's run_meta.json.
    """
    override = os.environ.get("AGF_PSIZE_MM", "").strip()
    if override:
        return float(override)
    return float((np.pi * SIM_STOCK_R_MM ** 2 * SIM_STOCK_L_MM / n) ** (1.0 / 3.0))

PRETTY = {
    "grid": "grid (baseline)",
    "grid_position_correction": "grid + position correction",
    "grid_fluidlab": "grid + FluidLab correction",
    "grid_particle_sdf": "grid + particle SDF correction",
    "grid_penalty": "grid + penalty correction",
    "no_contact": "NO CONTACT (control)",
}
COLORS = ["#e8833a", "#3aa7e8", "#7ac74f", "#c77adc", "#dcc84f"]
REAL_COLOR = "#3f6fb0"

# (label, camera offset direction, up vector, slab axis or None)
# Slab axis is the axis a cross-section is cut ALONG: 1 = cut the y mid-plane and look down y
# (side section, shows the pressing direction), 2 = cut the z mid-plane and look down z (top
# section). Profiles are the same cameras without the cut.
VIEWS = [
    ("iso", np.array([1.0, -1.0, 0.55]), np.array([0.0, 0.0, 1.0]), None),
    ("side profile", np.array([0.0, -1.0, 0.0]), np.array([0.0, 0.0, 1.0]), None),
    ("top profile", np.array([0.0, 0.0, 1.0]), np.array([0.0, 1.0, 0.0]), None),
    ("side cross-section", np.array([0.0, -1.0, 0.0]), np.array([0.0, 0.0, 1.0]), 1),
    ("top cross-section", np.array([0.0, 0.0, 1.0]), np.array([0.0, 1.0, 0.0]), 2),
]

# MPM grid geometry in the CANONICAL mm frame. Nodes sit on integer multiples of dx in SIM
# coordinates (see base_mpm_solver: array index I maps to world cell I + _lower_bound_cell), and
# the adapter's canonical transform is
#     canon_x = (x_pinned_face_m - sim_x)*1000,  canon_y = sim_y*1000,
#     canon_z = (z_axis_m - sim_z)*1000
# with x_pinned_face_m = 29.5 mm and z_axis_m = 120 mm. A pure reflection plus translation, so the
# lattice stays axis-aligned with the same 4 mm spacing -- only its PHASE shifts.
GRID_DX_MM = 4.0
GRID_PHASE = (29.5 % GRID_DX_MM, 0.0, 120.0 % GRID_DX_MM)   # -> x = 1.5, y = 0.0, z = 0.0 (mod 4)


def load_hits(d, tag):
    p = os.path.join(d, "%s_hits.npz" % tag)
    if not os.path.exists(p):
        return {}
    with np.load(p) as z:
        return {int(k.split("_")[1]): np.asarray(z[k], dtype=np.float64) for k in z.files}


def _font(size):
    from PIL import ImageFont
    for c in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if os.path.exists(c):
            try:
                return ImageFont.truetype(c, size)
            except Exception:
                pass
    return ImageFont.load_default()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", default="batch17fix")
    ap.add_argument("--arms", default="grid,grid_position_correction,grid_fluidlab")
    ap.add_argument("--max-hit", type=int, default=17)
    ap.add_argument("--out", default="contact_methods")
    ap.add_argument("--fps", type=int, default=2)
    ap.add_argument("--tile", type=int, default=440, help="tile width in px")
    ap.add_argument("--tile-h", type=int, default=330)
    ap.add_argument("--real-points", type=int, default=9000)
    ap.add_argument("--color-error", action="store_true",
                    help="colour sim particles by SIGNED distance to the real surface: red = sim "
                         "material where the real part has none (over-fill), blue = inside the "
                         "real body. Turns 'the shapes differ' into 'here is WHERE it is wrong'.")
    ap.add_argument("--clim", type=float, default=3.0, help="colour range, +/- mm")
    ap.add_argument("--grid", action="store_true",
                    help="overlay the MPM grid nodes at the viewing plane, so the grid resolution "
                         "is visible against the particle spacing (dx is 2x the particle spacing)")
    args = ap.parse_args()

    import pyvista as pv
    from PIL import Image, ImageDraw

    d = os.path.expanduser("~/GitHub/Genesis/forge_common/main/outputs/%s" % args.batch)
    tags = [t.strip() for t in args.arms.split(",") if t.strip()]
    clouds = {t: load_hits(d, t) for t in tags}
    if [t for t in tags if not clouds[t]]:
        print("no per-hit clouds for: %s" % ", ".join(t for t in tags if not clouds[t]))
        return 1

    hits = sorted({h for t in tags for h in clouds[t] if h <= args.max_hit})
    last_ok = {t: max(clouds[t]) for t in tags}
    print("arms: %s" % ", ".join("%s(->%d)" % (t, last_ok[t]) for t in tags))

    rng = np.random.default_rng(0)
    real, real_pts = {}, {}
    for h in hits:
        V, T = real_surface(h)
        real[h] = (V, T)
        # Section row still uses points: a slab through a surface mesh is an outline either way,
        # and points keep it visually consistent with the sim sections beside it.
        real_pts[h] = V if len(V) <= args.real_points else V[
            rng.choice(len(V), args.real_points, replace=False)]

    allpts = np.concatenate([real[h][0] for h in hits]
                            + [clouds[t][h] for t in tags for h in clouds[t] if h <= args.max_hit])
    lo, hi = allpts.min(0), allpts.max(0)
    ctr = (lo + hi) / 2.0
    span = float(np.max(hi - lo))
    # parallel_scale is a HALF-HEIGHT in world units, so the visible WIDTH is
    # 2*pscale*(tile_w/tile_h). Solve it so the longest state fills ~92% of the tile width
    # instead of floating in whitespace, and keep it FIXED across arms and hits.
    pscale = (span / 0.92) / (2.0 * (args.tile / float(args.tile_h)))
    dist = span * 3.0
    r0 = psize(len(next(iter(clouds[tags[0]].values())))) / 2.0
    slab = 1.6 * r0

    # ---- error field: signed distance from each sim particle to the real surface -------------
    # UNSIGNED distance alone is useless here: an interior particle is legitimately far from the
    # surface and would light up as "wrong". The SIGN is what carries the meaning, and it comes
    # from a voxel inside/outside test -- NOT trimesh.contains(), which falls off embree onto an
    # rtree over ~270k triangles and has twice killed this VM.
    err_ctx = {}
    if args.color_error:
        import trimesh
        from scipy.spatial import cKDTree
        for h in hits:
            V, T = real[h]
            m = trimesh.Trimesh(vertices=V, faces=T, process=False)
            err_ctx[h] = (cKDTree(V), m.voxelized(pitch=1.5).fill())
        print("built error reference for %d hits" % len(err_ctx))

    def signed_err(P, h):
        tree, vg = err_ctx[h]
        d, _ = tree.query(P, k=1)
        inside = np.asarray(vg.is_filled(P), dtype=bool)
        return np.where(inside, -d, d)

    def grid_plane(axis):
        """MPM grid nodes on the single lattice plane nearest the centre along `axis`.

        Only ONE plane, not the whole volume: the full 3-D node lattice would bury the particles
        it is meant to be compared against. One plane at the viewing/cutting depth shows the
        spacing honestly.
        """
        ph = GRID_PHASE[axis]
        val = ph + round((ctr[axis] - ph) / GRID_DX_MM) * GRID_DX_MM
        others = [i for i in range(3) if i != axis]

        def rng(i):
            start = GRID_PHASE[i] + GRID_DX_MM * np.floor((lo[i] - GRID_PHASE[i]) / GRID_DX_MM)
            return np.arange(start, hi[i] + GRID_DX_MM, GRID_DX_MM)

        A, B = np.meshgrid(rng(others[0]), rng(others[1]), indexing="ij")
        G = np.zeros((A.size, 3))
        G[:, others[0]] = A.ravel()
        G[:, others[1]] = B.ravel()
        G[:, axis] = val
        return G

    grids = {1: grid_plane(1), 2: grid_plane(2)} if args.grid else {}
    if args.grid:
        print("grid overlay: dx=%.1f mm, %d nodes per plane (particle spacing %.2f mm)"
              % (GRID_DX_MM, len(grids[1]), 2.0 * r0))

    cols = ["__real__"] + tags
    TW, TH = args.tile, args.tile_h
    # Title and column headers get their OWN bands; drawing them at overlapping y collides.
    TITLE_H, COLHDR_H, LBL = 38, 26, 18
    HDR = TITLE_H + COLHDR_H
    W, H = TW * len(cols), HDR + (TH + LBL) * len(VIEWS)

    f_title, f_col, f_view = _font(19), _font(13), _font(11)
    pl = pv.Plotter(off_screen=True, window_size=(TW, TH))
    pl.set_background("white")

    frames = []
    for h in hits:
        canvas = Image.new("RGB", (W, H), "white")
        drw = ImageDraw.Draw(canvas)
        drw.text((12, 12), "Rigid-MPM contact methods vs real forge scan   —   hit %d / %d"
                 % (h, args.max_hit), fill="black", font=f_title)

        if args.color_error:
            def _cmap(t):
                try:
                    import matplotlib
                    r, g, b, _ = matplotlib.colormaps["coolwarm"](t)
                    return int(r * 255), int(g * 255), int(b * 255)
                except Exception:                      # blue-white-red fallback
                    if t < 0.5:
                        k = t / 0.5
                        return (int(60 + 195 * k), int(80 + 175 * k), 255)
                    k = (t - 0.5) / 0.5
                    return (255, int(255 - 175 * k), int(255 - 195 * k))
            bw, bh = 230, 13
            bx, by = W - bw - 250, 15
            for i in range(bw):
                drw.line([(bx + i, by), (bx + i, by + bh)], fill=_cmap(i / (bw - 1.0)))
            drw.text((bx - 132, by), "inside real  −%.0f mm" % args.clim,
                     fill="#333333", font=f_col)
            drw.text((bx + bw + 8, by), "+%.0f mm  outside real (over-fill)" % args.clim,
                     fill="#333333", font=f_col)

        for c, name in enumerate(cols):
            is_real = name == "__real__"
            if is_real:
                P, failed, label = real_pts[h], False, "REAL SCAN (ground truth)"
            else:
                use = h if h in clouds[name] else last_ok[name]
                P, failed = clouds[name][use], h > last_ok[name]
                label = PRETTY.get(name, name)
            color = REAL_COLOR if is_real else COLORS[(c - 1) % len(COLORS)]

            for r, (vname, direction, up, cut) in enumerate(VIEWS):
                pl.clear()
                if args.grid:
                    # Drawn first so particles sit ON TOP of the grid rather than behind it.
                    gnodes = grids[cut if cut is not None else 1]
                    pl.add_mesh(pv.PolyData(gnodes), color="#b9b2a6", point_size=3,
                                render_points_as_spheres=True, opacity=0.85)
                if is_real and cut is None:
                    # Solid surface for the real part: it IS a closed scan, and drawing it as a
                    # surface stops the eye comparing sampling density instead of geometry.
                    V, T = real[h]
                    faces = np.hstack([np.full((len(T), 1), 3, dtype=np.int64), T])
                    pl.add_mesh(pv.PolyData(V, faces), color=color, smooth_shading=True)
                else:
                    pts = P
                    sc = signed_err(pts, h) if (args.color_error and not is_real) else None
                    if cut is not None:
                        keep = np.abs(pts[:, cut] - ctr[cut]) <= slab
                        if int(keep.sum()) >= 10:
                            pts = pts[keep]
                            if sc is not None:
                                sc = sc[keep]
                    kw = dict(point_size=5 if cut is None else 8,
                              render_points_as_spheres=True)
                    if sc is not None:
                        # Diverging map clamped tight: most particles sit deep inside and would
                        # otherwise saturate the scale, hiding the surface detail that matters.
                        pl.add_mesh(pv.PolyData(pts), scalars=sc, cmap="coolwarm",
                                    clim=(-args.clim, args.clim), show_scalar_bar=False, **kw)
                    else:
                        pl.add_mesh(pv.PolyData(pts), color=color, **kw)
                # CAMERA LAG FIX (2026-08-11). Assigning cam.position / cam.up / cam.focal_point
                # attribute-by-attribute does NOT commit before screenshot(), so every tile used
                # to render ONE VIEW BEHIND -- every row shifted by one and the LAST view (top
                # cross-section) never appeared at all. All renders shipped before this date, and
                # the published review artifact, have mislabelled rows.
                # pl.camera_position = [pos, focal, up] is the assignment that actually commits.
                pos = tuple(ctr + direction / np.linalg.norm(direction) * dist)
                pl.camera_position = [pos, tuple(ctr), tuple(up)]
                cam = pl.camera
                cam.parallel_projection = True
                cam.parallel_scale = pscale
                # Set the clipping range explicitly. The camera is positioned manually without a
                # reset_camera(), so VTK keeps whatever range the previous actor left behind --
                # which slices the real surface mesh and renders it as a bare silhouette outline
                # from some directions but solid from others.
                cam.clipping_range = (max(1e-3, dist - 2.0 * span), dist + 2.0 * span)
                pl.render()          # force the pipeline to commit the new camera before capture
                tile = np.asarray(pl.screenshot(return_img=True))
                y = HDR + r * (TH + LBL)
                canvas.paste(Image.fromarray(tile).resize((TW, TH)), (c * TW, y))
                if c == 0:
                    note = vname
                    if cut is not None:
                        # Say it, or the real column's thin arcs read as missing data.
                        note = vname + "  (real = surface scan, no interior)"
                    drw.text((6, y + 3), note, fill="#777777", font=f_view)

            txt = label + ("   [FAILED @ hit %d]" % last_ok[name] if failed else "")
            drw.text((c * TW + 10, TITLE_H + 5), txt,
                     fill="#b03030" if failed else "black", font=f_col)

        frames.append(np.asarray(canvas))
        print("  hit %d" % h)

    pl.close()

    outdir = os.path.join(d, "render")
    os.makedirs(outdir, exist_ok=True)
    from forge_common.viz import write_gif
    gif = os.path.join(outdir, "%s.gif" % args.out)
    write_gif(frames, gif, fps=args.fps)
    print("wrote %s (%.1f MB)" % (gif, os.path.getsize(gif) / 1e6))
    try:
        import imageio
        mp4 = os.path.join(outdir, "%s.mp4" % args.out)
        # macro_block_size=1 preserves exact pixel dims; the default 16 silently crops and
        # would shift the panel grid.
        with imageio.get_writer(mp4, fps=args.fps, macro_block_size=1) as w:
            for f in frames:
                w.append_data(f)
        print("wrote %s (%.1f MB)" % (mp4, os.path.getsize(mp4) / 1e6))
    except Exception as exc:
        print("mp4 skipped: %r" % exc)
    return 0


if __name__ == "__main__":
    sys.exit(main())
