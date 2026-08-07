"""Guard the Arrhenius flow rule ported into the MPM kernel.

``agforge/materials.py`` carries its own copy of the [Song2020] constants,
because kernel constants have to resolve at compile time and cannot be read out
of a numpy module from inside a ``qd.func``. Duplicated constants drift, so the
first job here is to prove the two copies still agree.

The second job is the interpolation. The kernel evaluates the strain dependence
as a branch-free hat-function sum rather than calling ``numpy.interp``; that is
an implementation detail which must not change the answer. The reference
implementation of that sum lives in this file and is checked against the CPU
module across the whole operating domain.

The third job is the guards. The Arrhenius fit is a HOT-WORKING model. Outside
its window it does not merely lose accuracy, it goes physically absurd
(~3.8 GPa at room temperature) and numerically unrepresentable (the exp()
argument reaches 176, and float32 overflows above ~88). The clamps are load
bearing, not cosmetic.

Imports agforge.materials, so quadrants/genesis must import - but nothing here
initialises a backend or touches the GPU.
"""
import math

import numpy as np
import pytest

from agforge import material_properties_mechanical as cpu
from agforge.materials import (
    _ARR_A,
    _ARR_ALPHA,
    _ARR_DSTRAIN,
    _ARR_N,
    _ARR_Q,
    _ARR_STRAIN0,
    _R_GAS,
    ArrheniusPlasticity,
    JohnsonCookPlasticity,
)
from agforge.options import MaterialOptions

# The domain the sim actually runs in, plus the edges of the fit.
_TEMPS_K = (1073.15, 1173.15, 1273.15)
_RATES = (0.1, 1.0, 10.0, 100.0)
_STRAINS = (0.0, 0.05, 0.07, 0.12, 0.20, 0.30, 0.45, 0.60)


def _kernel_hat_sum(eps_p, rate, temp_k):
    """Reference for the kernel's branch-free interpolation.

    Mirrors ``ArrheniusPlasticity._flow_stress_pa`` exactly, in plain Python.
    Returns MPa.
    """
    t = min(max((eps_p - _ARR_STRAIN0) / _ARR_DSTRAIN, 0.0), 8.0)
    total = 0.0
    for k in range(9):
        w = max(0.0, 1.0 - abs(t - k))
        if w > 0.0:
            z = rate * math.exp(_ARR_Q[k] / (_R_GAS * temp_k))
            x = (z / _ARR_A[k]) ** (1.0 / _ARR_N[k])
            total += w * math.asinh(x) / _ARR_ALPHA[k]
    return total


# --------------------------------------------------------------------------
# 1. The duplicated constant tables must not drift apart
# --------------------------------------------------------------------------
def test_kernel_constants_match_the_cpu_reference_module():
    assert _ARR_ALPHA == tuple(cpu._ALPHA)
    assert _ARR_N == tuple(cpu._N_EXP)
    assert _ARR_A == tuple(cpu._A_FAC)
    # The CPU module stores Q in kJ/mol; the kernel needs J/mol.
    assert _ARR_Q == tuple(q * 1e3 for q in cpu._Q_KJ)


def test_kernel_strain_grid_matches_the_cpu_reference_module():
    assert _ARR_STRAIN0 == cpu._STRAIN[0]
    # The hat-sum is only equivalent to numpy.interp on a UNIFORM grid.
    spacing = np.diff(np.array(cpu._STRAIN))
    assert np.allclose(spacing, _ARR_DSTRAIN), (
        "the [Song2020] strain grid stopped being uniform; the kernel's "
        "hat-function interpolation is no longer equivalent to numpy.interp"
    )
    assert len(cpu._STRAIN) == 9


def test_gas_constant_matches():
    assert _R_GAS == cpu.R_GAS


# --------------------------------------------------------------------------
# 2. The kernel's interpolation must equal the CPU implementation
# --------------------------------------------------------------------------
@pytest.mark.parametrize("temp_k", _TEMPS_K)
@pytest.mark.parametrize("rate", _RATES)
def test_hat_sum_reproduces_the_cpu_flow_stress(rate, temp_k):
    for eps in _STRAINS:
        got = _kernel_hat_sum(eps, rate, temp_k)
        want = cpu.flow_stress_mpa(eps, rate, temp_k)
        assert got == pytest.approx(want, rel=1e-9), (
            "kernel interpolation diverged from the CPU reference at "
            "eps=%s rate=%s T=%s" % (eps, rate, temp_k)
        )


def test_strain_is_clamped_at_both_ends_of_the_table():
    """Below 0.05 and above 0.45 the fit has no data; both must flatten."""
    lo = _kernel_hat_sum(0.0, 1.0, 1273.15)
    lo_edge = _kernel_hat_sum(_ARR_STRAIN0, 1.0, 1273.15)
    hi = _kernel_hat_sum(2.0, 1.0, 1273.15)
    hi_edge = _kernel_hat_sum(0.45, 1.0, 1273.15)
    assert lo == pytest.approx(lo_edge, rel=1e-12)
    assert hi == pytest.approx(hi_edge, rel=1e-12)


# --------------------------------------------------------------------------
# 3. The validity-window guards
# --------------------------------------------------------------------------
def test_fit_window_matches_the_published_domain():
    mat = ArrheniusPlasticity(E=121.5e9, nu=0.329, rho=7334.0)
    assert mat.T_fit_min == pytest.approx(1073.15)  # 800 C
    assert mat.T_fit_max == pytest.approx(1273.15)  # 1000 C


def test_room_temperature_is_absurd_which_is_why_the_clamp_exists():
    """Documents the number the clamp is protecting against.

    If this ever stops being large, the clamp may look unnecessary - it is not.
    """
    absurd = cpu.flow_stress_mpa(0.20, 1.0, 293.15)
    forging = cpu.flow_stress_mpa(0.20, 1.0, 1273.15)
    assert absurd > 3000.0
    assert absurd / forging > 15.0


def test_clamping_temperature_keeps_the_exponent_inside_float32():
    """exp() overflows float32 above ~88. The clamp must keep us well under."""
    worst_q = max(_ARR_Q)
    at_clamp = worst_q / (_R_GAS * 1073.15)
    unclamped = worst_q / (_R_GAS * 293.15)
    assert at_clamp < 60.0, "clamped exponent %.1f is close to float32 limits" % at_clamp
    assert unclamped > 88.0, (
        "room temperature no longer overflows float32; re-check whether the "
        "clamp is still doing the job it was written for"
    )


def test_rate_clamp_brackets_the_forging_band():
    mat = ArrheniusPlasticity(E=121.5e9, nu=0.329, rho=7334.0)
    assert mat.rate_min <= 1.0 <= mat.rate_max
    assert mat.rate_max >= 100.0, "forging runs to ~100 /s; the clamp must not bite there"


def test_substep_dt_default_is_a_plausible_cfl_timestep():
    """It is overwritten by environment.py, but a silly default would hide bugs."""
    mat = ArrheniusPlasticity(E=121.5e9, nu=0.329, rho=7334.0)
    assert 1e-8 < mat.substep_dt < 1e-4


# --------------------------------------------------------------------------
# 4. Wiring and regression guards
# --------------------------------------------------------------------------
def test_arrhenius_is_off_by_default():
    """Deliberate: the billet still starts at 293 K, where the clamp - not the
    physics - would decide the flow stress. See the option's docstring."""
    assert MaterialOptions().use_arrhenius is False
    assert MaterialOptions().use_johnson_cook is True


def test_johnson_cook_class_defaults_are_316l_not_4340():
    """Regression: these were AISI 4340's room-temperature constants.

    They were only harmless because environment.py overrides every one of them,
    so a partial construction anywhere would have silently forged the wrong
    alloy at the wrong temperature.
    """
    mat = JohnsonCookPlasticity(E=121.5e9, nu=0.329, rho=7334.0)
    assert mat.A != pytest.approx(792e6), "4340's A is back"
    assert mat.B != pytest.approx(510e6), "4340's B is back"
    assert mat.n != pytest.approx(0.26), "4340's n is back"
    assert mat.C != pytest.approx(0.014), "4340's C is back"
    assert mat.T_melt != pytest.approx(1793.0), "4340's melting point is back"
    assert mat.jc_m != pytest.approx(1.03), "4340's m is back"
    # And they should agree with the sourced card that MaterialOptions applies.
    opts = MaterialOptions()
    assert mat.A == pytest.approx(opts.jc_A)
    assert mat.B == pytest.approx(opts.jc_B)
    assert mat.n == pytest.approx(opts.jc_n)
    assert mat.T_ref == pytest.approx(opts.jc_T_ref)
    assert mat.T_melt == pytest.approx(opts.jc_T_melt)


# --------------------------------------------------------------------------
# 5. The CFL margin that pins nu
# --------------------------------------------------------------------------
# These live here rather than with the pure-numpy property tests because they
# are about the material card's effect on SOLVER STABILITY, which is what
# actually stops the sourced Poisson ratio from being adopted.
def _wave_speeds(E, nu, rho):
    mu = E / (2.0 * (1.0 + nu))
    K = E / (3.0 * (1.0 - 2.0 * nu))
    return math.sqrt(E / rho), math.sqrt((K + 4.0 * mu / 3.0) / rho)


def test_timestep_is_derived_from_the_p_wave():
    """The 0.90 safety margin is now measured against the wave that governs.

    This test used to assert the OPPOSITE - that substep_dt came from
    sqrt(E/rho) and so sat at ~1.09x the strict P-wave CFL limit - and its
    docstring said "if this ever starts passing at < 1.0, someone fixed the
    derivation and the nu constraint below can be revisited". That happened on
    2026-08-07. Both halves had to ship together: raising nu is only safe once
    the timestep comes from c_p.
    """
    from agforge.options import TeleopOptions

    m = MaterialOptions()
    dx = 1.0 / 183
    c_bar, c_p = _wave_speeds(m.E, m.nu, m.rho)
    assert c_p > c_bar, "P-wave must be the faster of the two"

    cfg = TeleopOptions(mat=m)
    substep_dt = cfg.sim.dt / cfg.sim.substeps
    assert substep_dt / (dx / c_p) == pytest.approx(0.90, abs=1e-3), (
        "substep_dt no longer honours the 0.90 margin against the P-wave"
    )

    # What the OLD bar-wave derivation would give at the now-shipped nu: 1.24,
    # far past the measured cliff at ~1.13. That is precisely why nu was pinned
    # to 0.329 before, and why it no longer has to be.
    assert (0.90 * dx / c_bar) / (dx / c_p) == pytest.approx(1.243, abs=0.01)


def test_the_sourced_poisson_ratio_would_cross_the_measured_stability_cliff():
    """nu = 0.382 (from BAM's own E and G) was measured to break the press.

    Stable at ratio 1.117 (nu 0.34), broken at 1.140 (nu 0.35). This pins WHY
    the shipped nu is not the sourced one - so nobody 'corrects' it and silently
    destabilises every consumer of this branch.
    """
    m = MaterialOptions()
    dx = 1.0 / 183
    c_bar, _ = _wave_speeds(m.E, m.nu, m.rho)
    substep_dt = 0.90 * dx / c_bar

    def ratio_for(nu):
        _, c_p = _wave_speeds(m.E, nu, m.rho)
        return substep_dt / (dx / c_p)

    assert ratio_for(0.34) < 1.13, "nu=0.34 was measured stable"
    assert ratio_for(0.35) > 1.13, "nu=0.35 was measured to break"
    assert ratio_for(0.3823) > 1.2, "the BAM-derived nu is far past the cliff"

    # The cliff only applies while the timestep comes from the bar wave. Raising
    # nu is CORRECT once cfl_use_pwave is on - that is the whole point of the
    # flag - so this guard must not fire on that (currently non-default) config.
    if not m.cfl_use_pwave:
        assert m.nu < 0.35, (
            "nu was raised past the measured stability cliff while the timestep "
            "is still derived from the bar wave; set cfl_use_pwave first"
        )


def test_pwave_cfl_option_is_on_by_default():
    """On since 2026-08-07. The P-wave is the criterion that actually governs
    stability, and turning it on is what lets nu be the sourced 0.383 rather
    than a value held down to keep the bar-wave timestep from blowing up."""
    assert MaterialOptions().cfl_use_pwave is True


def test_the_shipped_nu_is_the_sourced_one():
    """options.nu and poisson_ratio() must not drift apart -- the card's
    provenance story depends on them being the same number."""
    from agforge.material_properties_mechanical import poisson_ratio

    nu = MaterialOptions().nu
    assert nu == pytest.approx(poisson_ratio(1273.15), abs=5e-4), (
        "options.nu no longer matches poisson_ratio at forging temperature"
    )
    assert nu > 0.37, "nu fell back toward the old stability-pinned 0.329"


def test_pwave_cfl_option_puts_the_timestep_back_inside_the_limit():
    """With the option on, the derived dt should honour the 0.90 margin against
    the P-wave - which is what the nominal safety factor always claimed."""
    from agforge.options import TeleopOptions

    for nu in (0.329, 0.3823):
        mat = MaterialOptions()
        mat.nu = nu
        mat.cfl_use_pwave = True
        cfg = TeleopOptions(mat=mat)
        substep_dt = cfg.sim.dt / cfg.sim.substeps
        _, c_p = _wave_speeds(mat.E, nu, mat.rho)
        ratio = substep_dt / ((1.0 / 183) / c_p)
        assert ratio == pytest.approx(0.90, abs=1e-3), (
            "cfl_use_pwave should give exactly the 0.90 margin against the "
            "P-wave at nu=%s, got %.4f" % (nu, ratio)
        )


def test_pwave_cfl_only_ever_shrinks_the_timestep():
    """It cannot destabilise a configuration that already runs; the cost is
    wall-clock. Guards against someone 'optimising' it into a larger dt."""
    from agforge.options import TeleopOptions

    off = MaterialOptions()
    off.cfl_use_pwave = False
    on = MaterialOptions()
    on.cfl_use_pwave = True

    cfg_off = TeleopOptions(mat=off)
    cfg_on = TeleopOptions(mat=on)
    dt_off = cfg_off.sim.dt / cfg_off.sim.substeps
    dt_on = cfg_on.sim.dt / cfg_on.sim.substeps
    assert dt_on < dt_off, "the P-wave criterion must never enlarge the timestep"


def test_johnson_cook_reference_temperature_is_the_forging_temperature():
    """Not room temperature: A and B are already the 1000 C values, so T* must
    be 0 there or the card gets thermally softened twice."""
    mat = JohnsonCookPlasticity(E=121.5e9, nu=0.329, rho=7334.0)
    assert mat.T_ref == pytest.approx(1273.15)
