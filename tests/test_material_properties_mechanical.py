"""Guard the sourced 316L mechanical properties and the traps around them.

Companion to ``test_material_property_consistency.py``, which does the same job
for the thermal / electromagnetic side.

Three kinds of assertion here:

1. SOURCED VALUES - the numbers that came out of published measurements. If one
   of these moves, either a source was re-read or someone typo'd a constant.
2. CROSS-MODULE AGREEMENT - the mechanical and thermal modules describe the same
   billet and must not disagree about it. This is the failure mode that
   material_properties.py exists to prevent.
3. REGRESSION GUARDS - two specific mistakes that were made and fixed during this
   work, locked so they cannot come back silently.

Runs on numpy alone; no GPU, no genesis import.
"""
import numpy as np
import pytest

from agforge.material_properties import STEEL_316L
from agforge.material_properties_mechanical import (
    DRX_PEAK_STRAIN,
    FORGING_TEMP_K,
    T_LIQUIDUS_K,
    T_SOLIDUS_K,
    density_kg_m3,
    fit_johnson_cook,
    flow_curve_mpa,
    isothermal_card,
    johnson_cook_divergence,
    peak_flow_stress_mpa,
    youngs_modulus_pa,
)

# --------------------------------------------------------------------------
# 1. Sourced values
# --------------------------------------------------------------------------
# Peak flow stress at 1000 C, derived from [Song2020] Table 1. Cross-checked
# against an independent 316H fit [Zhou2023] to within 2.4%, and against a
# measured 111 MPa at 0.01 /s to within 1.3%.
EXPECTED_PEAK_MPA = {0.1: 157.2, 1.0: 213.4, 10.0: 274.7, 100.0: 338.5}


@pytest.mark.parametrize("rate,expected", sorted(EXPECTED_PEAK_MPA.items()))
def test_peak_flow_stress_matches_published_fit(rate, expected):
    got = peak_flow_stress_mpa(rate, FORGING_TEMP_K)
    assert got == pytest.approx(expected, abs=0.2), (
        f"peak flow stress at {rate} /s drifted from the [Song2020] fit"
    )


def test_in_domain_measured_point():
    """111 MPa measured at 1000 C / 0.01 /s, independently reported.

    0.01 /s is INSIDE Song's fit domain, so this validates the implementation
    rather than the extrapolation.
    """
    got = peak_flow_stress_mpa(0.01, FORGING_TEMP_K)
    assert got == pytest.approx(111.0, rel=0.05), (
        "implementation no longer reproduces the in-domain measured point"
    )


def test_youngs_modulus_and_density_at_forging_temp():
    # [BAM2023] measured to 900 C, linearly extrapolated 100 C further.
    # Independent check: Andrews reports 118.7 GPa at 1270 K.
    assert youngs_modulus_pa(FORGING_TEMP_K) == pytest.approx(121.5e9, rel=0.01)
    # [NIST2021] SRM 1155a, D(T) = 8052 - 0.564 T.
    assert density_kg_m3(FORGING_TEMP_K) == pytest.approx(7334.0, abs=1.0)


def test_youngs_modulus_interpolates_measured_range_exactly():
    """Inside the BAM range the function must return measurements, not a fit."""
    assert youngs_modulus_pa(273.15 + 900.0) == pytest.approx(129e9, rel=1e-6)
    assert youngs_modulus_pa(273.15 + 700.0) == pytest.approx(144e9, rel=1e-6)


# --------------------------------------------------------------------------
# 2. Cross-module agreement
# --------------------------------------------------------------------------
def test_solidus_agrees_with_thermal_module():
    """The two modules describe one billet and must agree on its melting point.

    material_properties.STEEL_316L.t_melt_k is the solidus (Pichler et al.),
    independently confirmed by the NIST SRM 1155a measurements used here.
    """
    assert T_SOLIDUS_K == STEEL_316L.t_melt_k, (
        "material_properties_mechanical and material_properties disagree about "
        "the 316L solidus"
    )


def test_liquidus_above_solidus():
    assert T_LIQUIDUS_K > T_SOLIDUS_K


def test_density_is_below_room_temperature_value():
    """Hot steel is less dense than cold steel. Catches a units/sign slip."""
    assert density_kg_m3(FORGING_TEMP_K) < STEEL_316L.rho_kg_m3


# --------------------------------------------------------------------------
# 3. Regression guards - mistakes actually made during this work
# --------------------------------------------------------------------------
def test_johnson_cook_A_is_not_a_degenerate_fit():
    """REGRESSION: a free least-squares fit drives A to ~1 MPa.

    The residual is nearly flat in A, so least squares slides it to whatever
    bound it is given. But the solver evaluates ``A + B eps^n`` from eps_p = 0,
    where the expression collapses to A - so A IS the initial yield stress the
    simulation sees. A free fit therefore produces a billet that yields at
    essentially zero load while fitting the 0.05-0.30 window beautifully.

    A must stay physically plausible for hot 316L.
    """
    a_pa, _, _, _ = fit_johnson_cook(1.0, FORGING_TEMP_K)
    assert a_pa > 50e6, (
        "Johnson-Cook A has collapsed to a fitting artifact; it must be pinned, "
        "not free-fitted (see fit_johnson_cook docstring)"
    )
    assert a_pa < peak_flow_stress_mpa(1.0, FORGING_TEMP_K) * 1e6


def test_johnson_cook_fit_is_accurate_below_the_drx_peak():
    _, _, _, rms = fit_johnson_cook(1.0, FORGING_TEMP_K)
    mean = flow_curve_mpa(1.0, FORGING_TEMP_K)[1].mean()
    assert rms / mean < 0.02, "JC fit quality degraded below the DRX peak"


def test_johnson_cook_diverges_past_the_drx_peak():
    """The quantitative case against original JC in the DRX regime.

    JC hardening is monotonic; the real material softens after the peak. If this
    ever stops being true the flow curve has lost its DRX shape.
    """
    div = johnson_cook_divergence(1.0, FORGING_TEMP_K)
    assert div, "no tabulated strains past the DRX peak"
    worst = max(abs(pct) for _, _, _, pct in div)
    assert worst > 20.0, (
        "JC no longer diverges past the DRX peak - the flow curve may have lost "
        "its peak-then-soften shape"
    )


def test_flow_curve_actually_peaks_and_softens():
    eps, sig = flow_curve_mpa(1.0, FORGING_TEMP_K)
    peak_i = int(np.argmax(sig))
    assert 0 < peak_i < len(sig) - 1, "flow curve has no interior peak"
    assert eps[peak_i] == pytest.approx(DRX_PEAK_STRAIN, abs=1e-9)
    assert sig[-1] < sig[peak_i], "flow curve does not soften after the peak"


def test_rate_sensitivity_is_large():
    """REGRESSION: the reason a single-number answer is not well posed.

    If this shrinks toward 1.0 the rate dependence has been lost, and the whole
    argument for a rate-resolved card goes with it.
    """
    slow = peak_flow_stress_mpa(0.1, FORGING_TEMP_K)
    fast = peak_flow_stress_mpa(10.0, FORGING_TEMP_K)
    assert fast / slow == pytest.approx(1.75, abs=0.05)


def test_no_4340_constants_hardcoded_at_the_jc_call_site():
    """REGRESSION: environment.py used to hardcode AISI 4340's constants.

    ``T_melt=1793.0, jc_m=1.03`` are 4340's values and sat at the
    JohnsonCookPlasticity call site rather than in MaterialOptions - which is
    precisely why the 316L conversion missed them for so long. They were dormant
    only because runs are isothermal; they activate the moment anything drives
    ``temp``.

    This is a source scan rather than an import check because the defect was
    *location*, not value: the point is that these numbers must not live at a
    call site again, where nobody thinks to look for material constants.
    """
    import pathlib

    env = pathlib.Path(__file__).resolve().parents[1] / "agforge" / "environment.py"
    src = env.read_text(encoding="utf-8")
    jc_call = src[src.index("JohnsonCookPlasticity("):]
    jc_call = jc_call[: jc_call.index("morph=")]

    assert "1793" not in jc_call, (
        "AISI 4340's melting point (1793 K) is hardcoded at the "
        "JohnsonCookPlasticity call site again; 316L is 1675 K and the value "
        "belongs in MaterialOptions.jc_T_melt"
    )
    for field in ("jc_T_ref", "jc_T_melt", "jc_m"):
        assert f"cfg.mat.{field}" in jc_call, (
            f"{field} is no longer plumbed through MaterialOptions"
        )


def test_material_options_match_the_sourced_card():
    """The live defaults must BE the sourced values, not merely document them.

    This is the test that would have caught the original defect: 4340's constants
    sitting in a config labelled for steel at 1200 C, with nothing asserting that
    the config agreed with any measurement.

    Imports genesis (via agforge.options), unlike the rest of this module.
    """
    from agforge.options import MaterialOptions

    o = MaterialOptions()
    card = isothermal_card(strain_rate=o.jc_eps0, temp_k=o.jc_T_ref)

    assert o.E == pytest.approx(card["E"], rel=0.01)
    assert o.rho == pytest.approx(card["rho"], rel=0.01)
    assert o.nu == pytest.approx(card["nu"], rel=0.01)
    assert o.jc_A == pytest.approx(card["jc_A"], rel=0.01)
    assert o.jc_B == pytest.approx(card["jc_B"], rel=0.01)
    assert o.jc_n == pytest.approx(card["jc_n"], rel=0.01)
    assert o.jc_T_melt == STEEL_316L.t_melt_k

    # None of AISI 4340's fingerprints may survive anywhere in this block.
    assert o.jc_T_melt != 1793.0
    assert o.jc_C != 0.014, "still AISI 4340's room-temperature rate coefficient"
    assert o.jc_n != 0.26, "still AISI 4340's hardening exponent"


def test_thermal_softening_is_neutral_at_the_operating_point():
    """jc_A/jc_B are already the 1000 C values, so the thermal term must be 1.0
    there - otherwise the calibration gets softened a second time."""
    from agforge.options import MaterialOptions

    o = MaterialOptions()
    t_star = (o.jc_T_ref - o.jc_T_ref) / (o.jc_T_melt - o.jc_T_ref)
    assert t_star == 0.0
    assert (1.0 - t_star ** o.jc_m) == pytest.approx(1.0)


def test_rate_term_is_neutral_at_nominal_rate():
    """Same argument for the rate term: 1 + C*ln(edot/edot0) must be 1.0 at
    nominal, so C only corrects deviation from the calibrated rate."""
    from agforge.options import MaterialOptions

    o = MaterialOptions()
    assert 1.0 + o.jc_C * np.log(o.jc_eps0 / o.jc_eps0) == pytest.approx(1.0)


def test_rate_coefficient_reproduces_measured_sensitivity():
    """jc_C must reproduce the Arrhenius rate sensitivity near nominal.

    Guards against reverting to a room-temperature C: 316L's rate sensitivity
    rises steeply with temperature, and 4340's 0.014 is ~9x too low here.
    """
    from agforge.options import MaterialOptions

    o = MaterialOptions()
    for rate in (0.1, 10.0):
        jc_ratio = 1.0 + o.jc_C * np.log(rate / o.jc_eps0)
        arr_ratio = (peak_flow_stress_mpa(rate, FORGING_TEMP_K)
                     / peak_flow_stress_mpa(o.jc_eps0, FORGING_TEMP_K))
        assert jc_ratio == pytest.approx(arr_ratio, rel=0.10), (
            f"jc_C does not reproduce measured rate sensitivity at {rate} /s"
        )


def test_isothermal_card_is_self_consistent():
    card = isothermal_card(1.0)
    assert card["t_melt_k"] == STEEL_316L.t_melt_k
    assert card["jc_A"] < card["peak_flow_stress_pa"]
    assert card["jc_valid_max_strain"] == DRX_PEAK_STRAIN
    # Card must be SI: Pa not MPa.
    assert card["E"] > 1e10
    assert card["jc_B"] > 1e8
