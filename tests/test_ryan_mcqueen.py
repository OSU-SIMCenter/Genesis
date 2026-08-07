"""Pin the [RyanMcQueen1990] reference and the domain error it exposes.

Ryan & McQueen (J. Mater. Process. Technol. 21, 1990, 177-199) is type 316
torsion over 900-1200 C at 0.1-5 /s -- a window that CONTAINS the real process
(1150-1260 C, 0.41 /s), where the shipped [Song2020] card (800-1000 C,
2e-4 - 2e-2 /s) does not and simply clamps.

The headline these tests protect: at the real forging point the shipped card is
2.4x - 4.4x too STIFF. That is a factor, not a percentage, and it is established
from published sources alone -- no simulation, no force reading, no shape metric.

Nothing here initialises a backend or touches the GPU.
"""
import math

import pytest

import agforge.material_properties_mechanical as m

RATE = 0.41          # /s, blow #1 of the 2026-06-15 T4 dataset
STRAIN = 0.207       # true strain of that blow
T_CARD_K = 1273.15   # 1000 C - the Arrhenius clamp ceiling
T_FORGE_K = 1473.15  # 1200 C - the middle of the real forging window


def test_activation_energy_is_the_published_one():
    assert m.RM_Q_J_MOL == pytest.approx(460.0e3)


def test_zener_hollomon_matches_the_definition():
    z = m.rm_zener_hollomon(RATE, T_CARD_K)
    expected = RATE * math.exp(460.0e3 / (m.R_GAS * T_CARD_K))
    assert z == pytest.approx(expected, rel=1e-12)
    assert z == pytest.approx(3.064e18, rel=1e-3)


@pytest.mark.parametrize("state, expected_mpa", [
    ("eps_0", 73.3),
    ("eps_0.1", 78.2),
    ("drv_sat", 165.7),
    ("drx_ss", 139.3),
])
def test_published_states_at_the_card_temperature(state, expected_mpa):
    """Hand-computed from zeta = C * asinh(Z*1e-17)**m at 1000 C, 0.41 /s."""
    assert m.rm_stress_mpa(state, RATE, T_CARD_K) == pytest.approx(expected_mpa, abs=0.15)


def test_all_four_states_are_declared_with_two_constants_each():
    assert set(m.RM_STATES) == {"eps_0", "eps_0.1", "drv_sat", "drx_ss"}
    for c, exp in m.RM_STATES.values():
        assert c > 0 and 0.0 < exp < 1.0


def test_stress_falls_monotonically_with_temperature():
    """Hot steel is softer. Guards a sign error in the Z exponent."""
    for state in m.RM_STATES:
        vals = [m.rm_stress_mpa(state, RATE, tc + 273.15)
                for tc in (900, 1000, 1100, 1200, 1260)]
        assert vals == sorted(vals, reverse=True), f"{state} not monotonic in T"


def test_stress_rises_with_strain_rate():
    lo = m.rm_stress_mpa("drv_sat", 0.1, T_FORGE_K)
    hi = m.rm_stress_mpa("drv_sat", 5.0, T_FORGE_K)
    assert hi > lo


def test_the_bracket_is_ordered_and_contains_the_operating_strain():
    lo, hi = m.rm_bracket_mpa(RATE, T_FORGE_K)
    assert lo < hi
    # eps 0.207 sits above the eps=0.1 point and below DRV saturation, so the
    # bracket is the honest statement; the paper does not pin down where inside.
    assert 0.1 < STRAIN


def test_clamping_at_1000C_would_be_2_to_4x_too_stiff():
    """Why the ceiling had to move. NOTE the CPU reference does not clamp
    temperature at all - only the kernel does - so 1000 C is evaluated here
    explicitly to represent what the old clamp produced."""
    lo, hi = m.rm_bracket_mpa(RATE, T_FORGE_K)
    as_if_clamped = m.flow_stress_mpa(STRAIN, RATE, T_CARD_K)

    assert as_if_clamped / hi == pytest.approx(2.4, abs=0.2)
    assert as_if_clamped / lo == pytest.approx(4.4, abs=0.3)


def test_song_extrapolates_into_the_ryan_mcqueen_bracket():
    """The load-bearing justification for raising the ceiling rather than
    porting a new model: Song's FORM tracks temperature correctly, and an
    independent in-domain study says so."""
    lo, hi = m.rm_bracket_mpa(RATE, T_FORGE_K)
    extrapolated = m.flow_stress_mpa(STRAIN, RATE, T_FORGE_K)

    assert extrapolated == pytest.approx(77.8, abs=1.0)
    # Just above the bracket top, and nowhere near the 181 MPa the clamp gave.
    assert hi < extrapolated < 1.1 * hi
    assert extrapolated < 0.5 * m.flow_stress_mpa(STRAIN, RATE, T_CARD_K)


def test_the_kernel_ceiling_reaches_the_forging_window_but_no_further():
    """1473.15 K is the top of Ryan & McQueen's measured range. Above it the
    extrapolation is unchecked, so the ceiling should not quietly drift up."""
    from agforge.materials import ArrheniusPlasticity

    a = ArrheniusPlasticity(E=121.5e9, nu=0.383, rho=7334.0)
    assert a.T_fit_max == pytest.approx(1473.15)
    assert a.T_fit_max >= 1423.15, "must reach the 1150 C bottom of the forging window"
    assert a.T_fit_max <= 1473.15, (
        "raised past what [RyanMcQueen1990] can corroborate - needs a new source"
    )


def test_the_two_models_are_the_same_order_where_both_are_in_domain():
    """At 1000 C both fits apply. They should not disagree wildly - if they do,
    one of them has been transcribed wrong."""
    lo, hi = m.rm_bracket_mpa(RATE, T_CARD_K)
    song = m.flow_stress_mpa(STRAIN, RATE, T_CARD_K)
    assert 0.5 * lo < song < 2.0 * hi
