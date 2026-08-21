"""Pin the [RyanMcQueen1990] reference and the domain error it exposes.

Ryan & McQueen (J. Mater. Process. Technol. 21, 1990, 177-199) is type 316
torsion over 900-1200 C at 0.1-5 /s -- a window that CONTAINS the real process
(1150-1260 C, 0.41 /s), where the shipped [Song2020] card (800-1000 C,
2e-4 - 2e-2 /s) does not and simply clamps.

🚩 THE HEADLINE THESE TESTS ONCE PROTECTED WAS WRONG ON THREE COUNTS, corrected
2026-08-20. It read "at the real forging point the shipped card is 2.7x - 4.7x
too STIFF". Every part of that has since failed:
  (a) the "real forging point" of 1150-1260 C is the GENERIC LITERATURE window
      for 316L, not a measurement of this billet. The camera reads 615-967 C,
      so this forge never reaches it. See doc section 0.1.
  (b) the 2.7x-4.7x compares the CLAMPED Song value (181.1 MPa) against the
      Tier-2 bracket. That clamp was removed when T_fit_max went to 1473.15 K,
      so the configuration the factor describes no longer runs.
  (c) the bracket itself is wrong. The published Tier-1 equation returns
      77.4 MPa at 1200 C / 0.41 /s -- outside the 38.5-67.2 bracket entirely.
Matched-rate against the published form, the card AS IT NOW RUNS is within 0.5%
of Ryan & McQueen at 1200 C. It runs 7-24% stiffer across 615-967 C, which is
the band that actually matters and which nothing here yet pins.

⚠️ The (C, m) constants the Tier-2 numbers come from are an UNVERIFIED secondary
extraction whose functional form does not match the paper's published equation
-- see the provenance note in material_properties_mechanical.py. These tests pin
what the module currently computes; they do not certify the source.

Nothing here initialises a backend or touches the GPU.
"""
import math

import pytest

import agforge.material_properties_mechanical as m

RATE = 0.41          # /s, blow #1 of the 2026-06-15 T4 dataset
STRAIN = 0.207       # true strain of that blow
T_CARD_K = 1273.15   # 1000 C - the Arrhenius clamp ceiling
T_FORGE_K = 1473.15  # 1200 C - the middle of the real forging window


def test_activation_energy_is_the_worked_condition_from_the_paper():
    """454 kJ/mol is Ryan & McQueen's own measurement for WORKED 316, which is
    what our bar is. 402 is their as-cast value. 460 -- which this module shipped
    until 2026-08-11 -- is the mean of the 21-study survey in their Table 1, not
    a measurement of theirs."""
    assert m.RM_Q_J_MOL == pytest.approx(454.0e3)
    assert m.RM_Q_CAST_J_MOL == pytest.approx(402.0e3)
    assert m.RM_Q_J_MOL != pytest.approx(460.0e3), "regressed to the survey mean"


def test_the_paper_s_own_sinh_constants_are_recorded():
    """These are the Tier-1 values read straight off the paper. Unlike the
    (C, m) pairs they are verified, so a port should start from them."""
    assert m.RM_ALPHA_INV_MPA == pytest.approx(83.33, abs=0.01)  # alpha = 1.2e-2
    assert m.RM_N == pytest.approx(4.5)
    assert m.RM_Q_DRX_STEADY_J_MOL == pytest.approx(296.0e3)


def test_zener_hollomon_matches_the_definition():
    z = m.rm_zener_hollomon(RATE, T_CARD_K)
    expected = RATE * math.exp(454.0e3 / (m.R_GAS * T_CARD_K))
    assert z == pytest.approx(expected, rel=1e-12)
    assert z == pytest.approx(1.7383e18, rel=1e-3)


@pytest.mark.parametrize("state, expected_mpa", [
    ("eps_0", 72.43),
    ("eps_0.1", 76.37),
    ("drv_sat", 160.71),
    ("drx_ss", 135.04),
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


def test_pre_exponential_recovered_via_characteristic_temperature():
    """Thesis eqn. (19): T' = Q_HW / (R ln A), so A = exp(Q / (R T'))."""
    assert m.RM_TPRIME_W_K == pytest.approx(1521.9, abs=0.1)   # figure prints 1522
    assert m.RM_A_W_PER_S == pytest.approx(3.826e15, rel=1e-3)
    # round trip back through eqn. (19)
    assert m.RM_Q_J_MOL / (8.314 * math.log(m.RM_A_W_PER_S)) == pytest.approx(
        m.RM_TPRIME_W_K, rel=1e-3)


@pytest.mark.parametrize("rate, temp_c, expected_mpa", [
    (0.373, 1200, 76.11),   # doc comparison point
    (0.373, 615, 513.6),    # coldest measured blow
    (0.41, 1200, 77.38),    # rate-matched against Song's 77.8
])
def test_tier1_peak_stress_matches_independent_evaluation(rate, temp_c, expected_mpa):
    """Anchored on two INDEPENDENT evaluations of the published equation that
    agreed to 4 s.f., not on a re-derivation of this module's own arithmetic --
    a mirror test would only pin consistency. See the thermal-solver episode."""
    assert m.rm_peak_stress_mpa(rate, temp_c + 273.15) == pytest.approx(
        expected_mpa, rel=2e-3)


def test_tier2_bracket_does_not_contain_the_published_answer():
    """Why RM_STATES is superseded: at 1200 C / 0.41 /s the Tier-2 bracket is
    38.5-67.2 MPa while the published form gives 77.4. Pinned so that anyone
    reinstating the bracket as authoritative trips this."""
    lo, hi = m.rm_bracket_mpa(0.41, 1473.15)
    published = m.rm_peak_stress_mpa(0.41, 1473.15)
    assert published > hi, "bracket unexpectedly contains the published value"
    assert lo < hi < published


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

    assert as_if_clamped / hi == pytest.approx(2.70, abs=0.15)
    assert as_if_clamped / lo == pytest.approx(4.71, abs=0.25)


def test_song_extrapolates_to_just_above_the_ryan_mcqueen_bracket():
    """The load-bearing justification for raising the ceiling rather than
    porting a new model: Song's FORM tracks temperature correctly, and an
    independent in-domain study says so.

    NOTE the name: Song lands just ABOVE the bracket, not inside it. At the
    corrected Q = 454 the margin is ~16% (it read ~5% at the old, wrong Q = 460).
    The argument still holds comfortably -- the clamped value is 2.7x the bracket
    top -- but it is a weaker corroboration than it first appeared."""
    lo, hi = m.rm_bracket_mpa(RATE, T_FORGE_K)
    extrapolated = m.flow_stress_mpa(STRAIN, RATE, T_FORGE_K)

    assert extrapolated == pytest.approx(77.8, abs=1.0)
    # Just above the bracket top, and nowhere near the 181 MPa the clamp gave.
    assert hi < extrapolated < 1.2 * hi
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
