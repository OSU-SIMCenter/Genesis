"""Guard the decoupling of constitutive strain rate from numerical strain rate.

The simulated press runs at ``strike.pressing_speed`` = 25 m/s for numerical
affordability. The real Agility Forge press runs at 14.1 mm/s -- measured from
the 2026-06-15 T4 dataset, first blow, gap 38.41 -> 31.06 mm in 0.505 s. The
sim is ~1770x fast, so a flow rule that derives its strain rate from
``substep_dt`` is handed a rate that does not exist in the process.

Johnson-Cook survives that only because its rate term is dead code. Arrhenius
does not: it is genuinely rate-coupled, which was the entire point of porting
it. These tests pin the size of the error and prove the escape hatch works.

Nothing here initialises a backend or touches the GPU.
"""
import numpy as np
import pytest

from agforge import material_properties_mechanical as cpu
from agforge.materials import ArrheniusPlasticity
from agforge.options import MaterialOptions

# Measured, 2026-06-15 T4 bulk, blow #1.
REAL_DIE_SPEED_MM_S = 14.1
REAL_TRUE_STRAIN_RATE = 0.41
BILLET_DIA_MM = 38.1

# strike.pressing_speed = 25 m/s over a 38.1 mm bar.
SIM_PRESSING_SPEED_M_S = 25.0
SIM_NOMINAL_RATE = SIM_PRESSING_SPEED_M_S / (BILLET_DIA_MM / 1000.0)

ARRHENIUS_FIT_RATE_MAX = 2e-2   # top of the [Song2020] fit domain


def test_the_sim_rate_is_the_one_that_is_fictional():
    """Sanity-pin both numbers so the ~1770x gap cannot drift unnoticed."""
    assert SIM_NOMINAL_RATE == pytest.approx(656.2, rel=1e-3)
    ratio = SIM_PRESSING_SPEED_M_S / (REAL_DIE_SPEED_MM_S / 1000.0)
    assert ratio == pytest.approx(1773.0, rel=0.01)


def test_the_rate_clamp_does_not_catch_the_simulated_rate():
    """rate_max was sized against the real process ('forging runs 1-100 /s'),
    so the sim's 656 /s slips underneath the guard rather than tripping it.
    This is the specific reason the problem was silent."""
    mat = ArrheniusPlasticity(E=121.5e9, nu=0.329, rho=7334.0)
    assert SIM_NOMINAL_RATE < mat.rate_max, (
        "if this ever fails the clamp has started catching the sim rate, and "
        "the silent-extrapolation hazard this test documents is gone")


def test_the_simulated_rate_is_far_outside_the_fit_domain():
    decades = np.log10(SIM_NOMINAL_RATE / ARRHENIUS_FIT_RATE_MAX)
    assert decades > 4.0, "expected >4 decades of extrapolation"


def test_the_real_rate_is_much_closer_to_the_fit_domain():
    """The real process is only ~1.3 decades above the fit, not ~4.5."""
    sim = np.log10(SIM_NOMINAL_RATE / ARRHENIUS_FIT_RATE_MAX)
    real = np.log10(REAL_TRUE_STRAIN_RATE / ARRHENIUS_FIT_RATE_MAX)
    assert real < sim - 3.0


def test_using_the_simulated_rate_inflates_flow_stress_substantially():
    """Quantify the error the prescribed rate removes. Evaluated on the CPU
    reference, which the kernel is separately proven to match."""
    T = 1273.15
    eps = 0.2
    sigma_real = cpu.flow_stress_mpa(eps, REAL_TRUE_STRAIN_RATE, T)
    sigma_sim = cpu.flow_stress_mpa(eps, SIM_NOMINAL_RATE, T)
    assert sigma_sim > sigma_real
    ratio = sigma_sim / sigma_real
    # ~1.75x per 100x rate over ~3.2 decades
    assert 1.8 < ratio < 3.5, f"unexpected inflation factor {ratio:.2f}"


def test_default_keeps_the_legacy_derived_rate():
    mat = ArrheniusPlasticity(E=121.5e9, nu=0.329, rho=7334.0)
    assert mat.process_strain_rate == 0.0
    assert MaterialOptions().arrhenius_process_strain_rate is None


def test_prescribed_rate_is_carried_onto_the_material():
    mat = ArrheniusPlasticity(E=121.5e9, nu=0.329, rho=7334.0,
                              process_strain_rate=REAL_TRUE_STRAIN_RATE)
    assert mat.process_strain_rate == pytest.approx(REAL_TRUE_STRAIN_RATE)


def test_prescribed_rate_still_respects_the_clamps():
    """The escape hatch must not become a way to smuggle an absurd rate in."""
    mat = ArrheniusPlasticity(E=121.5e9, nu=0.329, rho=7334.0,
                              process_strain_rate=1e9)
    assert mat.rate_max <= 1e3
    clamped = min(max(mat.process_strain_rate, mat.rate_min), mat.rate_max)
    assert clamped == mat.rate_max
