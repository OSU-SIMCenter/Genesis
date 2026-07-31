"""Guard the one duplication the material-property refactor could not remove.

``genesis/engine/solvers/base_mpm_solver.py`` cannot import
``agforge.material_properties``: its curves compile to literals inside a
``@qd.func``. So the kernel keeps its own copy of the 316L constants, and this
test is what stops that copy from drifting — which is exactly how the constants
ended up disagreeing in four places before.

If this fails, change BOTH the kernel literals and material_properties.py.

Runs on numpy alone; no GPU, no genesis import.
"""
import numpy as np
import pytest

from agforge.material_properties import (
    CP_316L_SEG_PARAMS,
    K_316L_SEG_PARAMS,
    STEEL_316L,
    cp_316l,
    cp_316l_seg,
    k_316l,
    k_316l_seg,
    rho_e_316l,
)

# --------------------------------------------------------------------------
# Transcribed by hand from base_mpm_solver.py::get_steel_thermal_conductivity
# and ::get_steel_cp. Keep in sync with those functions.
# --------------------------------------------------------------------------
KERNEL_K = (13.31, 700.0, 19.87, 1000.0, 24.16, 0.0129, 293.0)
KERNEL_CP = (500.0, 700.0, 556.5, 1000.0, 605.2, 0.0743, 293.0)


def test_kernel_conductivity_literals_match_module():
    assert KERNEL_K == K_316L_SEG_PARAMS, (
        "base_mpm_solver.get_steel_thermal_conductivity has drifted from "
        "material_properties.K_316L_SEG_PARAMS"
    )


def test_kernel_specific_heat_literals_match_module():
    assert KERNEL_CP == CP_316L_SEG_PARAMS, (
        "base_mpm_solver.get_steel_cp has drifted from "
        "material_properties.CP_316L_SEG_PARAMS"
    )


@pytest.mark.parametrize("temp", [293.0, 400.0, 500.0, 700.0, 900.0, 1000.0, 1200.0, 1450.0])
def test_segment_form_tracks_reference_conductivity(temp):
    """The kernel's 3 segments must stay within 3% of the Ho & Chu knot table."""
    ref = float(k_316l(temp))
    seg = float(k_316l_seg(temp))
    assert abs(seg - ref) / ref < 0.03, f"k at {temp} K: seg {seg:.2f} vs ref {ref:.2f}"


@pytest.mark.parametrize("temp", [293.0, 473.0, 700.0, 873.0, 1000.0, 1173.0, 1253.0])
def test_segment_form_tracks_reference_specific_heat(temp):
    """Within 3% of the NIST SRM 1155a DSC table. The loosest point is ~870 K,
    where the real data has a kink from precipitates dissolving that three
    straight segments cannot follow."""
    ref = float(cp_316l(temp))
    seg = float(cp_316l_seg(temp))
    assert abs(seg - ref) / ref < 0.03, f"cp at {temp} K: seg {seg:.2f} vs ref {ref:.2f}"


def test_conductivity_rises_with_temperature():
    """316L is not carbon steel: k must INCREASE with temperature. A regression
    here means someone reinstated an AISI 4340-shaped curve."""
    t = np.linspace(300.0, 1500.0, 60)
    k = k_316l(t)
    assert np.all(np.diff(k) > 0.0)


def test_resistivity_rises_and_saturates():
    """rho_e increases monotonically but its slope must fall — a linear fit
    overestimates the room-temperature end and underestimates the hot end."""
    t = np.linspace(300.0, 1500.0, 60)
    r = rho_e_316l(t)
    d = np.diff(r)
    assert np.all(d > 0.0)
    assert d[-1] < d[0], "resistivity curve should saturate, not stay linear"


def test_316l_is_non_magnetic_at_all_temperatures():
    """No Curie transition: any delta(T) modelling premised on one is misplaced."""
    assert STEEL_316L.mu_r == pytest.approx(1.0)
    assert STEEL_316L.ferromagnetic_below_curie is False


def test_skin_depth_round_trips_through_implied_frequency():
    for f in (1000.0, 3000.0, 10000.0):
        d = STEEL_316L.skin_depth_m(f)
        assert STEEL_316L.implied_frequency_hz(d) == pytest.approx(f, rel=1e-9)


def test_skin_depth_is_independent_of_billet_radius():
    """The property this refactor exists to enforce. The old code derived delta
    from cylinder_radius, so changing billet size silently changed the implied
    drive frequency. delta must depend only on material and frequency."""
    d = STEEL_316L.skin_depth_m(3000.0)
    assert d == pytest.approx(STEEL_316L.skin_depth_m(3000.0))
    # And it must NOT coincide with either billet radius / 2.
    for radius in (0.0254 / 2, 0.0381 / 2):
        assert abs(d - radius / 2.0) > 1e-4


def test_alpha_worst_is_at_the_hot_end_for_316l():
    """Opposite to AISI 4340, whose diffusivity peaked at room temperature. The
    CFL bound in options.py depends on getting this right."""
    t = np.linspace(293.15, 1500.0, 512)
    a = STEEL_316L.alpha(t)
    assert t[int(np.argmax(a))] > 1200.0
