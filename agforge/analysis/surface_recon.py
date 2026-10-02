"""The material's outer surface, rebuilt from the particles that carry it.

MPM particles are sample points, not a surface, and the quantity a forging run is
judged on is a shape. This rebuilds the boundary of the material the particles
occupy, with no fitted parameters: each particle contributes a cube of its own
size, the union is closed, interior voids are filled, the largest piece is kept,
and the result is lightly smoothed and contoured.

Why the closing step exists, since it looks like cosmetic smoothing and is not.
Where forging stretches or shears the particle lattice, consecutive particles sit
further apart along an axis than their own cube width, so the raw union has slits
running through solid material. Marching cubes then meshes both faces of every
slit and the result is riddled with interior surface: measured on one press of a
hot series at 13 cells, 930 separate surface pieces with 44% of vertices more
than 1 mm inside the outer envelope. Closing by two voxels bridges those gaps and
leaves one piece, while moving the undeformed bar's radius by less than 0.001 mm
and its volume by under 1%.

The closing has a limit, and it is arithmetic rather than taste. The occupancy
grid is ``SUBDIV`` voxels per particle half-width, so a closing of ``close_vox``
voxels bridges ``close_vox / SUBDIV`` of a particle width -- two thirds of one at
the defaults. A lattice stretched past about 1.67x therefore keeps its slits, and
measurement agrees: stretches of 1.2 to 1.7 come back as a single piece, 1.9 does
not. Past that the answer is a finer grid, not a larger closing radius, because a
closing wide enough to bridge the gap also rounds off features of the same size.

This is deliberately not the reconstruction in ``recon_kernel.py``. That one
calibrates a density level against a target volume to study grid convergence;
this one has nothing to calibrate, which is what makes it usable as a deliverable
surface rather than a measurement.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import ndimage
from skimage import measure

PLY_TYPES = {
    "char": "i1", "uchar": "u1", "short": "i2", "ushort": "u2", "int": "i4",
    "uint": "u4", "float": "f4", "double": "f8", "float32": "f4",
    "float64": "f8", "int32": "i4", "uint8": "u1", "int8": "i1", "uint32": "u4",
}

#: Voxels per particle half-width. The occupancy grid is this much finer than the
#: particle cubes, so a cube is resolved by 2*SUBDIV+1 voxels per axis.
SUBDIV = 3


def reconstruct(P, particle_size_mm, close_vox=2, smooth_sigma=0.7):
    """Outer surface of the material occupied by particle centres ``P``.

    ``particle_size_mm`` is the particle's own width, i.e. the cube side each
    particle contributes. ``close_vox`` is the morphological closing radius in
    occupancy voxels; 0 disables it and reproduces the unclosed union, which is
    the shape that produced the interior-surface defect described in the module
    docstring.

    Returns a dict with the mesh (``V``, ``F``), the occupancy field and its
    origin and spacing (``field``, ``lo``, ``h``) for point-in-material queries,
    and the occupied volume (``vol_mm3``).
    """
    P = np.asarray(P, dtype=np.float64)
    if P.ndim != 2 or P.shape[1] != 3:
        raise ValueError("P must be (N, 3), got %r" % (P.shape,))
    if not np.isfinite(P).all():
        raise ValueError("P contains non-finite coordinates")
    if particle_size_mm <= 0:
        raise ValueError("particle_size_mm must be positive, got %r" % (particle_size_mm,))

    s = float(particle_size_mm)
    h = s / (2.0 * SUBDIV)
    # Room for the cube, the closing, and the marching-cubes stencil, so the
    # surface is never clipped by the edge of the grid.
    pad = s + (3 + close_vox + 1) * h
    lo = P.min(0) - pad
    hi = P.max(0) + pad
    shape = np.ceil((hi - lo) / h).astype(int) + 1

    grid = np.zeros(shape, dtype=np.uint8)
    idx = np.round((P - lo) / h).astype(int)
    grid[idx[:, 0], idx[:, 1], idx[:, 2]] = 1
    # Each marked voxel becomes the particle's cube.
    occ = ndimage.maximum_filter(grid, size=2 * SUBDIV + 1)
    if close_vox:
        c = 2 * close_vox + 1
        occ = ndimage.minimum_filter(ndimage.maximum_filter(occ, size=c), size=c)
    occ = ndimage.binary_fill_holes(occ)
    lab, n = ndimage.label(occ)
    if n > 1:
        # Keep the bar, drop anything that detached.
        occ = lab == (np.argmax(np.bincount(lab.ravel())[1:]) + 1)
    occ = occ.astype(np.uint8)

    field = ndimage.gaussian_filter(occ.astype(np.float32), sigma=smooth_sigma)
    verts, faces, _, _ = measure.marching_cubes(field, 0.5, spacing=(h, h, h))
    return {
        "V": verts + lo,
        "F": faces,
        "field": field,
        "lo": lo,
        "h": h,
        "vol_mm3": float(occ.sum()) * h ** 3,
        "n_pieces": int(n),
    }


def inside(surf, pts):
    """Whether each point lies in the reconstructed material."""
    c = (np.asarray(pts, dtype=np.float64) - surf["lo"]) / surf["h"]
    i = np.round(c).astype(int)
    ok = np.all((i >= 0) & (i < np.array(surf["field"].shape)), axis=1)
    out = np.zeros(len(i), dtype=bool)
    good = i[ok]
    out[ok] = surf["field"][good[:, 0], good[:, 1], good[:, 2]] >= 0.5
    return out


def write_ply(path, V, F=None):
    """Binary little-endian PLY, x/y/z as doubles, millimetres."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    V = np.ascontiguousarray(V, dtype="<f8")
    head = ["ply", "format binary_little_endian 1.0", "element vertex %d" % len(V),
            "property double x", "property double y", "property double z"]
    if F is not None:
        head += ["element face %d" % len(F), "property list uchar int vertex_indices"]
    head.append("end_header")
    with open(path, "wb") as fh:
        fh.write(("\n".join(head) + "\n").encode("ascii"))
        fh.write(V.tobytes())
        if F is not None:
            F = np.asarray(F, dtype="<i4")
            rec = np.empty(len(F), dtype=[("n", "u1"), ("i", "<i4", 3)])
            rec["n"] = 3
            rec["i"] = F
            fh.write(rec.tobytes())


def read_ply_xyz(path):
    """Vertex coordinates from an ascii or binary PLY, as (N, 3) float64."""
    with open(path, "rb") as fh:
        header = []
        while True:
            line = fh.readline()
            if not line:
                raise ValueError("%s ended before end_header" % path)
            s = line.decode("ascii", "replace").strip()
            header.append(s)
            if s == "end_header":
                break
        fmt, n, props, in_vertex = None, 0, [], False
        for s in header:
            p = s.split()
            if not p:
                continue
            if p[0] == "format":
                fmt = p[1]
            elif p[0] == "element":
                in_vertex = p[1] == "vertex"
                if in_vertex:
                    n = int(p[2])
            elif p[0] == "property" and in_vertex:
                props.append((p[-1], p[1]))
        if fmt == "ascii":
            return np.loadtxt(fh, max_rows=n)[:, :3].astype(np.float64)
        end = "<" if fmt == "binary_little_endian" else ">"
        dt = np.dtype([(name, end + PLY_TYPES[t]) for name, t in props])
        arr = np.frombuffer(fh.read(dt.itemsize * n), dtype=dt)
        return np.stack([arr["x"], arr["y"], arr["z"]], 1).astype(np.float64)
