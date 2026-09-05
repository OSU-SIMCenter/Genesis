"""Build the billet initial condition from the real scan, for AGF_BILLET_MESH.

WHY THIS EXISTS. The sim seeds its billet from gs.morphs.Cylinder(r=0.02, h=0.059) -- which
is the BOUNDING BOX of the hit-1-before scan. forge_common's REAL_STOCK_RADIUS_MM = 20.0 is
a bare constant with no derivation in that file, and it is the bar's circumscribed extent,
not its volume. Measured against the scan:

    cylinder morph   9266 particles   74,128 mm^3   +10.9% vs the real 66,825
    this mesh        8364 particles   66,912 mm^3    +0.1%

and the cylinder additionally cannot represent the tapered ends, which the scan clearly has
(equivalent diameter falls to ~19.5 mm in the last 4 mm against ~40 mm mid-body).

THREE THINGS THIS SCRIPT HAS TO GET RIGHT, each of which broke a first attempt:

  1. DECIMATE. Genesis binds every visual vertex to every particle in one dense
     (V, P, 3) float32 array. The raw scan is 119,659 verts, which asks for 11.2 GiB and
     dies. 8,000 faces costs -0.14% of volume; at 2 mm particle spacing the discarded
     detail was never resolvable.
  2. CENTRE IT. The scan's own frame puts the bar at x in [-0.66, 58.6] mm. The morph's
     `pos` translates, it does not re-centre, so an uncentred mesh lands outside the solver
     domain and Genesis raises "particles outside solver boundary".
  3. DO NOT CHASE WATERTIGHTNESS. The scans are not watertight and fill_holes does not fix
     them (it closed one face out of 237k). It does not matter: the volume is stable to
     0.1% under repair, and the MPM sampler accepts the mesh as-is. Separately verified:
     the meshes have ZERO open boundary edges, so the tapered ends are real geometry rather
     than tong-occlusion holes.
  4. ROTATE INTO SIM SPACE. The scans live in the adapter's CANONICAL frame, which sits a
     180-degree rotation about Y away from sim space:
         canonical_x = x_pinned_face - sim_x ,   canonical_z = z_axis - sim_z
     Feed a scan straight in and the exported cloud comes back MIRRORED end-for-end against
     the scan it is scored against. This bar is strongly asymmetric -- one end tapers to
     ~31 mm equivalent diameter against ~40 mm mid-body -- so the mirror puts the taper at
     the wrong end. Measured cost before the fix: a flat -0.08 IoU at hits 1, 2 and 3.

Usage:
    python make_billet_mesh.py [--hit 1] [--side before] [--faces 8000]
    AGF_BILLET_MESH=<printed path> <your run command>
"""
import argparse
import os
import pathlib
import sys

import numpy as np
import trimesh

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from geom_metrics import real_mesh, SIM_STOCK_R_MM, SIM_STOCK_L_MM  # noqa: E402

OUT_DIR = pathlib.Path.home() / "GitHub/Genesis/forge_common/main/outputs/real_meshes"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hit", type=int, default=1)
    ap.add_argument("--side", default="before", choices=("before", "after"))
    ap.add_argument("--faces", type=int, default=8000,
                    help="decimation target; 8000 costs -0.14%% volume (default)")
    ap.add_argument("--no-rotate", action="store_true",
                    help="skip the canonical->sim 180-about-Y rotation (see note 4); the "
                         "exported cloud will then be mirrored against the scans")
    a = ap.parse_args()
    no_rotate = a.no_rotate

    m = real_mesh(a.hit, a.side)
    v_raw = abs(m.volume)
    print("scan hit %d %s: %d verts / %d faces, volume %.0f mm^3, watertight=%s"
          % (a.hit, a.side, len(m.vertices), len(m.faces), v_raw, m.is_watertight))

    m.merge_vertices()
    m.update_faces(m.unique_faces())
    m.update_faces(m.nondegenerate_faces())
    m.remove_unreferenced_vertices()

    d = m.simplify_quadric_decimation(face_count=a.faces) if a.faces else m
    d.merge_vertices()
    trimesh.repair.fix_normals(d)
    if d.volume < 0:
        d.invert()

    # (4) canonical -> sim: 180 degrees about Y, which negates x and z exactly as
    # _to_canonical_mm does, so the export round-trips back to the scan's orientation. A
    # ROTATION, not a reflection, so face winding and normals stay valid.
    if not no_rotate:
        d.apply_transform(trimesh.transformations.rotation_matrix(np.pi, [0, 1, 0]))

    # (2) centre on the bounding-box centroid so the morph's `pos` reproduces the cylinder's
    # placement exactly.
    d.apply_translation(-d.bounding_box.centroid)

    v = abs(d.volume)
    v_cyl = np.pi * SIM_STOCK_R_MM ** 2 * SIM_STOCK_L_MM
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / ("billet_hit%02d_%s_d%d%s.obj"
                     % (a.hit, a.side, a.faces, "_unrot" if no_rotate else ""))
    d.export(out)

    print("decimated  : %d verts / %d faces   volume %.0f mm^3  (%+.3f%% vs raw scan)"
          % (len(d.vertices), len(d.faces), v, 100 * (v / v_raw - 1)))
    print("centred    : bounds %s .. %s mm"
          % (np.round(d.bounds[0], 2), np.round(d.bounds[1], 2)))
    print("vs cylinder: nominal %.0f mm^3, i.e. the cylinder IC runs %+.1f%% heavy"
          % (v_cyl, 100 * (v_cyl / v_raw - 1)))
    print()
    print("wrote %s" % out)
    print()
    print("use it with:")
    print("    AGF_BILLET_MESH=%s <run command>" % out)


if __name__ == "__main__":
    raise SystemExit(main())
