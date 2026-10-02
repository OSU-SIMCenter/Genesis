"""The reconstruction has to recover a shape we already know, and the closing step
has to be shown to do something -- and to have a limit.

The undeformed bar is the easy half: a lattice filling a 38.1 mm cylinder must come
back as a 38.1 mm cylinder. That check caught nothing when the exporter was building
interior surfaces, because an undeformed lattice has no slits to build them from. So
the rest of this file stretches the lattice until it does, pins where closing fixes
that, and pins where it stops working. Without the last of those, nobody can tell
whether close_vox=2 is sufficient for a given run or merely the value in the file.

Figures here are measured from this synthetic lattice, which is a weaker instrument
than the real pre-press check (that one reads the solver's own seeding and lands
within 0.2 mm of nominal). The assertions are therefore about relationships that
hold regardless of seeding, not absolute tolerances.
"""
import numpy as np
import pytest

from agforge.analysis.surface_recon import (
    SUBDIV,
    inside,
    read_ply_xyz,
    reconstruct,
    write_ply,
)

RADIUS_MM = 19.05
CELLS = 13
PSIZE = (2.0 * RADIUS_MM / CELLS) / 2.0

#: Closing bridges a gap of close_vox/SUBDIV particle widths, so a lattice stretched
#: by more than 1 + that ratio keeps its slits. Measured below, and it matches.
HEALABLE_STRETCH = 1.0 + 2.0 / SUBDIV  # 1.667 at close_vox=2


def _lattice(length=40.0, psize=PSIZE, stretch=1.0):
    """Particles on a psize lattice inside the cylinder, optionally stretched along
    the bar axis the way forging stretches them."""
    n_ax = int(round(length / psize))
    n_r = int(np.ceil(RADIUS_MM / psize)) + 1
    xs = (np.arange(n_ax) + 0.5) * psize * stretch
    g = np.arange(-n_r, n_r + 1) * psize
    Y, Z = np.meshgrid(g, g, indexing="ij")
    keep = (Y ** 2 + Z ** 2) <= RADIUS_MM ** 2
    y, z = Y[keep], Z[keep]
    P = np.empty((len(xs) * len(y), 3))
    P[:, 0] = np.repeat(xs, len(y))
    P[:, 1] = np.tile(y, len(xs))
    P[:, 2] = np.tile(z, len(xs))
    return P


def test_surface_sits_between_the_outer_particles_and_one_particle_out():
    """The relationship the delivered clouds rely on: particle centres sit inside
    the material surface by up to their own width, so a comparison against a scanned
    surface must use the reconstruction and not the centres."""
    P = _lattice()
    surf = reconstruct(P, PSIZE)
    V = surf["V"]
    barrel = V[(V[:, 0] > 8.0) & (V[:, 0] < 32.0)]  # skip the end caps
    r_surface = float(np.median(np.hypot(barrel[:, 1], barrel[:, 2])))
    r_particles = float(np.hypot(P[:, 1], P[:, 2]).max())
    assert r_particles <= r_surface <= r_particles + PSIZE
    assert surf["n_pieces"] == 1


def test_undeformed_volume_is_close_to_the_nominal_cylinder():
    P = _lattice()
    surf = reconstruct(P, PSIZE)
    span = P[:, 0].max() - P[:, 0].min() + PSIZE
    nominal = np.pi * RADIUS_MM ** 2 * span
    assert surf["vol_mm3"] == pytest.approx(nominal, rel=0.05)


def test_closing_is_inert_when_there_are_no_slits():
    """It must not quietly inflate a bar that has nothing to bridge."""
    P = _lattice()
    a = reconstruct(P, PSIZE, close_vox=0)
    b = reconstruct(P, PSIZE, close_vox=2)
    assert b["vol_mm3"] == pytest.approx(a["vol_mm3"], rel=1e-9)
    assert a["n_pieces"] == b["n_pieces"] == 1


def test_closing_removes_the_interior_surface_a_stretch_creates():
    """This is the defect closing exists for. Unclosed, a 1.4x stretched lattice
    fragments into tens of pieces and the largest-piece rule then keeps a sliver;
    closed, it is one solid bar again."""
    P = _lattice(stretch=1.4)
    unclosed = reconstruct(P, PSIZE, close_vox=0)
    closed = reconstruct(P, PSIZE, close_vox=2)
    assert unclosed["n_pieces"] > 10, "expected the stretched lattice to fragment"
    assert closed["n_pieces"] == 1
    # The fragments were interior surface: with the slits closed, far more material
    # is enclosed than the surviving sliver held.
    assert closed["vol_mm3"] > 10.0 * unclosed["vol_mm3"]


def test_closing_has_a_limit_and_it_matches_the_voxel_arithmetic():
    """Closing bridges close_vox/SUBDIV of a particle width, so it cannot heal an
    arbitrary stretch. Asserted so that close_vox=2 is never assumed sufficient:
    a run whose lattice stretches past this needs a finer grid, not a bigger
    closing radius."""
    ok = reconstruct(_lattice(stretch=1.5), PSIZE, close_vox=2)
    too_far = reconstruct(_lattice(stretch=1.9), PSIZE, close_vox=2)
    assert 1.5 < HEALABLE_STRETCH < 1.9, "the bound moved; re-measure this test"
    assert ok["n_pieces"] == 1
    assert too_far["n_pieces"] > 1, "expected closing to fail past the bound"


def test_inside_separates_the_axis_from_the_outside():
    P = _lattice()
    surf = reconstruct(P, PSIZE)
    got = inside(surf, np.array([[20.0, 0.0, 0.0], [20.0, 100.0, 0.0]]))
    assert bool(got[0]) is True
    assert bool(got[1]) is False


def test_ply_round_trip(tmp_path):
    V = np.array([[1.5, -2.25, 3.0], [0.0, 0.0, 0.0], [1e3, 1e-3, -7.5]])
    p = tmp_path / "c.ply"
    write_ply(p, V)
    back = read_ply_xyz(p)
    assert back.shape == V.shape
    assert np.allclose(back, V)


def test_rejects_malformed_input():
    with pytest.raises(ValueError):
        reconstruct(np.zeros((4, 2)), PSIZE)
    with pytest.raises(ValueError):
        reconstruct(np.array([[0.0, 0.0, np.nan]]), PSIZE)
    with pytest.raises(ValueError):
        reconstruct(_lattice(length=10.0), 0.0)
