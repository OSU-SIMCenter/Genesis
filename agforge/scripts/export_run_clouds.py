"""Turn a run's saved presses into point clouds in the measurement frame.

``record_toolpath_hits.py --hit-dir DIR`` writes one ``hit_NN.npz`` per press, in
the simulator's own frame and metres. Scans and slab-model clouds are published in
a different frame and in millimetres, so a comparison means nothing until both
sides are in one frame. This does that conversion once, here, rather than in every
script that wants to look at a run.

The measurement frame, which is the frame the scans and the slab-model
``sm_pcd_NNN.ply`` files use:

* **x** millimetres from the chuck face toward the free end. The held stock is at
  x < 0, so the exposed bar starts at 0 regardless of how much was held.
* **y, z** millimetres, bar axis at the origin.
* a press at toolpath angle ``phi`` closes along ``(cos phi, sin phi)`` in (y, z).

The simulator runs with x increasing toward the pinned end and the bar sitting at
z = 6 * radius, so both of those axes are reversed on the way out. Getting that
wrong produces a bar that deforms at the wrong end, which is why the geometry is
taken as explicit arguments and echoed into ``params.json`` instead of being
inferred.

Two clouds per press are written. ``cloudN.ply`` is the reconstructed outer
surface, which is what a shape comparison should use; ``particles/particlesN.ply``
is the raw particle centres, which sit about half a particle inside that surface
and are the right thing for anything counting material.

    python -m agforge.scripts.export_run_clouds \
      --hit-dir out/roll2_c13 --out out/roll2-al --cells 13 \
      --exposed-mm 100 --held-mm 25
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from agforge.analysis.surface_recon import reconstruct, write_ply

DEFAULT_RADIUS_MM = 19.05


def to_measurement_frame(pos_m, radius_mm, exposed_mm, held_mm):
    """Simulator metres to measurement-frame millimetres.

    The pinned face is at ``held/2 + (exposed + held)/2`` in the simulator's x,
    because the bar is centred on the held stock's half-length (see
    AGF_PIN_EXTRA_MM in options.py) and runs ``exposed + held`` long. Subtracting
    the held length then puts x = 0 on the chuck face.
    """
    held_m = held_mm / 1000.0
    exposed_m = exposed_mm / 1000.0
    pin_x_m = held_m / 2.0 + (exposed_m + held_m) / 2.0
    z_axis_m = 6.0 * (radius_mm / 1000.0)
    out = np.empty_like(pos_m)
    out[:, 0] = (pin_x_m - pos_m[:, 0]) * 1000.0 - held_mm
    out[:, 1] = pos_m[:, 1] * 1000.0
    out[:, 2] = (z_axis_m - pos_m[:, 2]) * 1000.0
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hit-dir", type=Path, required=True,
                   help="Directory of hit_NN.npz written by record_toolpath_hits.py.")
    p.add_argument("--out", type=Path, required=True,
                   help="Series folder to write. Clouds go in <out>/mpm/.")
    p.add_argument("--cells", type=float, required=True,
                   help="Cells across the bar diameter the run used "
                        "(AGF_CELLS_PER_DIAMETER).")
    p.add_argument("--exposed-mm", type=float, required=True,
                   help="Exposed stock length the run used.")
    p.add_argument("--held-mm", type=float, default=0.0,
                   help="Stock held in the chuck (AGF_PIN_EXTRA_MM). Default 0.")
    p.add_argument("--radius-mm", type=float, default=DEFAULT_RADIUS_MM)
    p.add_argument("--ppc-divisor", type=float, default=2.0,
                   help="Particle size as dx / this. Must match AGF_PPC_DIVISOR "
                        "for the run. Default %(default)s.")
    p.add_argument("--n-points", type=int, default=150_000,
                   help="Surface points per cloud. Default %(default)s.")
    p.add_argument("--subdir", default="mpm")
    p.add_argument("--keep-held", action="store_true",
                   help="Keep the held stock (x < 0) in the surface clouds. Off by "
                        "default: it is rigid and is not part of the forged shape.")
    args = p.parse_args(argv)

    files = sorted(args.hit_dir.glob("hit_*.npz"))
    if not files:
        raise SystemExit("no hit_*.npz in %s" % args.hit_dir)

    dx_mm = (2.0 * args.radius_mm) / args.cells
    psize_mm = dx_mm / args.ppc_divisor
    out_dir = args.out / args.subdir
    (out_dir / "particles").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)  # fixed, so a re-export is byte-identical

    rows = []
    for f in files:
        k = int(f.stem.split("_")[1])
        with np.load(f) as z:
            pos = z["pos"].astype(np.float64)
        P = to_measurement_frame(pos, args.radius_mm, args.exposed_mm, args.held_mm)
        surf = reconstruct(P, psize_mm)
        V = surf["V"]
        if not args.keep_held:
            V = V[V[:, 0] >= 0.0]
        if len(V) > args.n_points:
            V = V[rng.choice(len(V), args.n_points, replace=False)]
        V = V[np.argsort(V[:, 0], kind="stable")]
        write_ply(out_dir / ("cloud%d.ply" % k), V)
        write_ply(out_dir / "particles" / ("particles%d.ply" % k), P)
        rows.append(dict(press=k, n_surface_points=int(len(V)), n_particles=int(len(P)),
                         free_end_mm=float(V[:, 0].max()),
                         volume_mm3=surf["vol_mm3"], surface_pieces=surf["n_pieces"]))
        print("  press %2d: %d surface points, free end %.2f mm, %d piece(s)"
              % (k, len(V), V[:, 0].max(), surf["n_pieces"]))

    params = dict(
        n_presses=len(rows),
        cells_across_diameter=args.cells,
        grid_dx_mm=dx_mm,
        particle_size_mm=psize_mm,
        exposed_length_mm=args.exposed_mm,
        held_length_mm=args.held_mm,
        stock_radius_mm=args.radius_mm,
        frame=("x mm from the chuck face toward the free end; y, z mm with the bar axis "
               "at the origin; a press at toolpath angle phi closes along "
               "(cos phi, sin phi) in (y, z). The frame of the slab-model sm_pcd files "
               "and the real_*.ply scans."),
        cloud=("cloudN.ply: the reconstructed outer material surface after press N -- "
               "union of the particles' own cubes, gaps closed, interior voids filled, "
               "largest piece kept, smoothed and contoured -- subsampled to at most %d "
               "points%s. particles/particlesN.ply: every particle centre, which sits "
               "about half a particle (%.2f mm) inside that surface."
               % (args.n_points,
                  "" if args.keep_held else " and cropped to the exposed bar (x >= 0)",
                  psize_mm / 2.0)),
        presses=rows,
    )
    (out_dir / "params.json").write_text(json.dumps(params, indent=1), encoding="utf-8")
    print("wrote %s (%d press(es))" % (out_dir, len(rows)))


if __name__ == "__main__":
    main()
