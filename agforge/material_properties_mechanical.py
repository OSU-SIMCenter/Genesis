"""Single documented source of truth for billet MECHANICAL properties at forging
temperature.

Companion to ``material_properties.py``, which owns the thermal / electromagnetic
side (k, cp, rho_e, emissivity, mu_r). That module deliberately carries no
mechanical data; this one carries no thermal data. Same material, two axes.

Deliberately PORTABLE
---------------------
Nothing here imports genesis, quadrants, torch or anything else from this repo -
only the standard library and numpy. That is intentional: the shared material
library proposed for ``forge_common/materials/`` should be able to take this file
verbatim. Keep it that way. Anything that needs a solver type belongs in
``agforge/materials.py`` or ``agforge/options.py``, not here.

What "316L at 1000 C" actually means
------------------------------------
It is not a single number. 1000 C is squarely in the DYNAMIC RECRYSTALLIZATION
(DRX) regime for 316L, where flow stress depends strongly on strain rate:

    peak flow stress at 1000 C     0.1 /s -> 157 MPa
                                     1 /s -> 213 MPa
                                    10 /s -> 275 MPa
                                   100 /s -> 339 MPa

A 100x change in rate moves the flow stress 1.75x. Forging rates (~1-100 /s) are
therefore nowhere near quasi-static values, and any single-number "yield stress at
1000 C" is underdetermined until a rate is named.

The flow curves also PEAK near strain 0.30 and then soften - that is the DRX
signature. See the Johnson-Cook caveat below.

Constitutive model choice is NOT made here
------------------------------------------
This module supplies data and a reference Arrhenius implementation. Which
constitutive model the solver should use is a separate decision that belongs with
the sim-accuracy workstream. The literature is consistent that for 316-family
austenitics in the hot-working domain, hyperbolic-sine (Arrhenius / Zener-Hollomon)
models track the data well while the ORIGINAL Johnson-Cook does not - JC multiplies
three uncoupled terms and its (A + B eps^n) hardening branch is monotonic, so it
structurally cannot represent the DRX peak-then-soften shape. Quantified in
``fit_johnson_cook`` below: a JC fit is excellent (<1.2% RMS) up to the peak and
then diverges to +25% by strain 0.45.

Sources
-------
[Song2020]  S-H. Song, "A Comparison Study of Constitutive Equation, Neural
            Networks, and Support Vector Regression for Modeling Hot Deformation
            of 316L Stainless Steel", Materials 13(17):3766 (2020), Table 1.
            doi:10.3390/ma13173766
            316L, 800-1000 C, strain rate 2e-4 .. 2e-2 /s, hot tensile.
            PRIMARY flow-stress source: strain-dependent alpha, n, Q, A.
[Zhou2023]  "Microstructure evolution and constitutive analysis of nuclear grade
            AISI-316H austenitic stainless steel during thermal deformation",
            Mater. Res. Express 10 (2023) 115604. doi:10.1088/2053-1591/ad07cb
            316H, 900-1200 C, 0.01-10 /s, Gleeble-3800.
            Independent CROSS-CHECK: agrees with [Song2020] to 0.9-2.4%.
[Benc2023]  M. Benc et al., "Influence of Deformation Temperature and Strain Rate
            on the Maximum Flow Stress Level of the 3D printed AISI 316L Steel",
            METAL 2023. Measured peak 381 MPa at 1173 K / 100 /s and 65 MPa at
            1523 K / 0.1 /s. Used to BOUND the rate extrapolation (see below).
[BAM2023]   B. Rehmer, F. Bayram, L.A. Avila Calderon, G. Mohr, B. Skrotzki,
            "BAM reference data: Temperature-dependent Young's and shear modulus
            data for ... AISI 316L", Zenodo 10.5281/zenodo.7813836 (2023);
            described in Scientific Data. Dynamic resonance, ASTM E1875,
            room temperature to 900 C. PRIMARY elastic source.
[NIST2021]  "Measurements of thermophysical properties of solid and liquid NIST
            SRM 316L stainless steel", PMC8193647. Ohmic pulse heating + DSC on
            SRM 1155a. PRIMARY density source. Independently confirms the
            solidus 1675 K / liquidus 1708 K already used in
            ``material_properties.py``.
[ISIJ1993]  "Elastic Moduli and Internal Friction of Low Carbon and Stainless
            Steels as a Function of Temperature", ISIJ International 33(4):508
            (1993). Cites Andrews: 118.7 GPa at 1270 K for stainless - an
            independent check on the extrapolated E(1000 C).

Confidence
----------
HIGH    E(T) and rho(T) - direct measurements from a national metrology institute
        (BAM) and a certified reference material (NIST SRM 1155a).
HIGH    Flow stress at 1000 C for 0.01-10 /s - two independent fits agree to
        within 2.4%, and the target temperature is inside both fit domains.
MEDIUM  Flow stress at 100 /s - this extrapolates [Song2020] roughly 4 orders of
        magnitude beyond its 2e-2 /s ceiling. Checked against [Benc2023]:
        over-predicts +14% at 1173 K / 100 /s. Treat +/-15% as the honest bar.
MEDIUM  E(1000 C) - extrapolated 100 C beyond the BAM range. Two independent
        estimates (122 GPa extrapolated, 118.7 GPa from Andrews) bracket ~120 GPa.
LOW     Poisson's ratio at 1000 C - published values scatter badly and
        non-monotonically. See ``poisson_ratio``.
"""

from __future__ import annotations

import math

import numpy as np

R_GAS = 8.314  # J/(mol K)

# Solidus. Matches STEEL_316L.t_melt_k in material_properties.py; independently
# confirmed by [NIST2021]. Liquidus is 1708 K.
T_SOLIDUS_K = 1675.0
T_LIQUIDUS_K = 1708.0


# --------------------------------------------------------------------------
# Flow stress: strain-compensated Arrhenius (Zener-Hollomon)
# --------------------------------------------------------------------------
# [Song2020] Table 1. Material constants as a function of plastic strain.
# alpha in 1/MPa, Q in kJ/mol, A in 1/s. Fit domain: 800-1000 C, 2e-4..2e-2 /s.
_STRAIN = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45)
_ALPHA = (0.0091, 0.0081, 0.0076, 0.0073, 0.0071, 0.0070, 0.0071, 0.0073, 0.0081)
_N_EXP = (6.0452, 5.4551, 5.2624, 5.1774, 5.0818, 5.0140, 4.9624, 4.8867, 5.2208)
_Q_KJ = (476.67, 446.37, 433.83, 430.11, 426.47, 426.65, 429.75, 432.76, 458.30)
_A_FAC = (7.3454e17, 4.737e16, 1.493e16, 1.069e16, 7.41e15, 7.49e15,
          9.96e15, 1.21e16, 9.634e16)

#: Strain at which the DRX peak occurs in the [Song2020] fit at forging
#: temperature. Johnson-Cook hardening is only defensible below this.
DRX_PEAK_STRAIN = 0.30


def _sigma_arrhenius(strain_rate, temp_k, alpha, n, q_j_mol, a_factor):
    """Hyperbolic-sine flow stress in MPa for one set of Arrhenius constants."""
    z = strain_rate * math.exp(q_j_mol / (R_GAS * temp_k))
    return math.asinh((z / a_factor) ** (1.0 / n)) / alpha


def flow_stress_mpa(plastic_strain, strain_rate, temp_k):
    """316L flow stress in MPa, from [Song2020] strain-compensated Arrhenius.

    ``plastic_strain`` is linearly interpolated between the tabulated strains and
    clamped at the ends of the table (0.05 .. 0.45).

    Valid: 1073-1273 K. Rates above ~1e-2 /s extrapolate the fit - see the
    module docstring on confidence.
    """
    eps = min(max(float(plastic_strain), _STRAIN[0]), _STRAIN[-1])
    sig = [
        _sigma_arrhenius(strain_rate, temp_k, a, n, q * 1e3, af)
        for a, n, q, af in zip(_ALPHA, _N_EXP, _Q_KJ, _A_FAC)
    ]
    return float(np.interp(eps, _STRAIN, sig))


def flow_curve_mpa(strain_rate, temp_k):
    """The tabulated flow curve: ``(strains, stresses_mpa)`` at fixed rate/temp."""
    return (
        np.array(_STRAIN),
        np.array([
            _sigma_arrhenius(strain_rate, temp_k, a, n, q * 1e3, af)
            for a, n, q, af in zip(_ALPHA, _N_EXP, _Q_KJ, _A_FAC)
        ]),
    )


def peak_flow_stress_mpa(strain_rate, temp_k):
    """Peak (DRX) flow stress in MPa - the number to quote for "strength at T"."""
    return float(flow_curve_mpa(strain_rate, temp_k)[1].max())


# ==========================================================================
# [RyanMcQueen1990] - the fit that actually covers the FORGING window
# --------------------------------------------------------------------------
# N.D. Ryan and H.J. McQueen, J. Mater. Process. Technol. 21 (1990) 177-199.
# Type 316 torsion, as-cast AND worked, 900-1200 C at 0.1-5 /s.
#
# WHY THIS EXISTS ALONGSIDE [Song2020]. Song is fitted over 800-1000 C and
# 2e-4 - 2e-2 /s. The real process sits at 1150-1260 C and 0.41 /s, so the
# shipped card is out of domain on BOTH axes and the kernel simply clamps to
# its 1273.15 K ceiling. Ryan & McQueen contains the operating point outright.
#
# --------------------------------------------------------------------------
# PROVENANCE, 2026-08-11. Two tiers. Read this before quoting any number here.
# --------------------------------------------------------------------------
# TIER 1 - READ OFF THE PAPER ITSELF, safe to rely on:
#     alpha = 1.2e-2 MPa^-1        (so 1/alpha = 83.33 MPa)
#     n     = 4.5                  (peak stress; same for 316W and 316C)
#     Q     = 454 kJ/mol  WORKED   <- our billet is worked bar
#     Q     = 402 kJ/mol  AS-CAST
# The paper states Q is CONSTANT over the whole range: "the analysis associated
# with eqn. (3) finds Q = 454 kJ/mol across the entire range". Published form is
# its eqn. (3)/(4):
#     Z = strain_rate * exp(Q/RT) = A [sinh(alpha * sigma)] ** n
# The pre-exponential A is NOT recoverable from the text available to us, so
# that equation cannot be evaluated here. DO NOT INVENT AN A.
#
# 🚨 TIER 2 - SECONDARY EXTRACTION, NOT VERIFIED AGAINST THE PAPER: the
# RM_STATES (C, m) pairs below. They reached this repo through an LLM research
# pass, not the PDF. Two concrete reasons to distrust them:
#   (a) their form, zeta = C * [asinh(Z * 1e-17)] ** m, puts the exponent
#       OUTSIDE the asinh. The paper's eqn. (4) puts it INSIDE, on (Z/A).
#       Those are not the same function.
#   (b) the values 65.7 / 62.2 / 123.8 / 103.5 could not be located anywhere in
#       the paper's text.
# The same research pass also produced a paper that does not exist (a
# "Dehghan-Manshadi 2008" on 316L - the real one is on 304) and a wrong year for
# Venugopal, so its base error rate is known to be non-trivial. Treat RM_STATES
# as a PLAUSIBILITY RANGE of unconfirmed provenance, not as published law.
#
# 🚩 Q WAS WRONG UNTIL 2026-08-11. This module shipped Q = 460 kJ/mol. In the
# paper, 460 is the MEAN of a 21-study literature survey in Table 1 (reported
# alongside n = 4.3 +/- 0.8), not Ryan & McQueen's own measurement. Corrected to
# 454. Effect at 1200 C / 0.41 /s: the bracket moves from 41.6-74.3 MPa down to
# 38.5-67.2, about -9%, and Song's extrapolation goes from 4.7% above the
# bracket top to 16% above it. The decision that rests on this - raising
# T_fit_max instead of clamping - survives easily either way, because the
# clamped value is 2.7x the bracket top. But it is less clean than it read.
#
# ⚠️ ONE Q FOR ALL FOUR STATES IS AN ASSUMPTION, NOT THE PAPER'S POSITION. The
# paper derives a separate and much lower activation energy for the DRX
# steady-state stress, Q_DRX ~ 296 kJ/mol, "considerably less than the one for
# sigma_p, i.e. Q_HW = 454". Applying 454 to drx_ss is therefore not what the
# paper does. Left alone for now only because these states feed NOTHING in the
# kernel - they are a reference bracket. Fix it before any of this ships as a
# flow rule.
#
# Corroboration on Q: DeAlmeida & Barbosa 2005 (ISIJ Int. 45(2) 296) get
# 450 +/- 20 over the same 900-1200 C window, and Song's own per-strain table
# runs 426-477. Ferreira 2020 gets 347 and is the outlier; it is also the only
# delta-ferrite-free study.
#
# 🚨 THE STRAIN AXIS IS UNDERDETERMINED *HERE* - AND THE BLOCKER IS OUR SOURCE,
# NOT THE PAPER. Of the four states only two carry a strain: zeta_0 at eps = 0
# and zeta_0.1 at eps = 0.1. The other two are LIMITS, not points -- zeta_e is
# the dynamic-recovery saturation stress and zeta_ss the dynamic-recrystallisa-
# tion steady state -- so placing them on the strain axis needs the critical and
# peak strains (eps_c, eps_p).
#
# THE PAPER PUBLISHES THOSE. It derives eps_c and eps_p and reports
# eps_c = 0.64 eps_p for the worked condition. They were simply absent from the
# extraction this module was built from. So "we cannot port this" is FALSE as
# stated -- getting the PDF would unblock a real kernel flow rule. Inventing the
# strains would not, and that remains the failure this workstream is undoing.
# Until then this module exposes the states and an honest BRACKET, and stops.
#
# Note also: type 316, not 316L, and torsion rather than compression.
# ==========================================================================

#: Activation energy, WORKED 316 -- the applicable condition for our bar.
RM_Q_J_MOL = 454.0e3
#: As-cast 316. Reference only; do not use for wrought bar.
RM_Q_CAST_J_MOL = 402.0e3
#: The paper's SEPARATE activation energy for the DRX steady-state stress.
#: Not currently applied -- see the note above.
RM_Q_DRX_STEADY_J_MOL = 296.0e3
#: 1/alpha in MPa, from the paper's alpha = 1.2e-2 MPa^-1.
RM_ALPHA_INV_MPA = 83.33
#: Stress exponent at the peak. Same for 316W and 316C.
RM_N = 4.5

#: state -> (C [MPa], m).
#: 🚨 TIER 2 -- UNVERIFIED secondary extraction, and its functional form does
#: not match the paper's published equation. See the provenance note above
#: before relying on, quoting, or porting these.
RM_STATES = {
    "eps_0": (65.7, 0.077),    # zero strain
    "eps_0.1": (62.2, 0.162),  # eps = 0.1
    "drv_sat": (123.8, 0.206),  # dynamic-recovery saturation
    "drx_ss": (103.5, 0.210),   # dynamic-recrystallisation steady state
}


def rm_zener_hollomon(strain_rate, temp_k):
    """Z = strain_rate * exp(Q/RT) for the Ryan & McQueen activation energy."""
    return float(strain_rate) * math.exp(RM_Q_J_MOL / (R_GAS * float(temp_k)))


def rm_stress_mpa(state, strain_rate, temp_k):
    """One published Ryan & McQueen stress state, in MPa.

    ``state`` is a key of :data:`RM_STATES`. No strain interpolation is done or
    implied -- see the module note above on why.
    """
    c, m = RM_STATES[state]
    z = rm_zener_hollomon(strain_rate, temp_k)
    return float(c * math.asinh(z * 1e-17) ** m)


def rm_bracket_mpa(strain_rate, temp_k):
    """(low, high) MPa bracketing flow stress between eps 0.1 and DRV saturation.

    The honest statement for an intermediate strain such as the 0.207 of the
    first blow of the T4 dataset: it lies inside this bracket, and the paper
    does not pin down where.

    ⚠️ NOT A BOUND, in two separate ways. First, ``drv_sat`` is a DRV saturation
    limit, not a maximum: [Song2020]'s own curve peaks at 80.6 MPa (eps ~ 0.3),
    ABOVE the 67.2 MPa this returns at the forging point, then softens. Second,
    the (C, m) constants behind it are an unverified extraction -- see the
    provenance note at the top of this section. Read this as a plausibility
    range, never as a published bound.
    """
    return (rm_stress_mpa("eps_0.1", strain_rate, temp_k),
            rm_stress_mpa("drv_sat", strain_rate, temp_k))



# --------------------------------------------------------------------------
# Elastic constants and density
# --------------------------------------------------------------------------
# [BAM2023] hot-rolled 316L, dynamic resonance (ASTM E1875). E in GPa.
_BAM_T_C = (24.0, 100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0, 850.0, 900.0)
_BAM_E_GPA = (197.0, 193.0, 184.0, 174.0, 167.0, 159.0, 152.0, 144.0, 137.0, 133.0, 129.0)
_BAM_G_GPA = (74.0, 73.0, 69.0, 69.0, 62.0, 59.0, 56.0, 53.0, 50.0, 48.0, 47.0)

# Linear fit over the top of the measured range (700-900 C), used to reach 1000 C.
_E_SLOPE_GPA_PER_C = (129.0 - 144.0) / (900.0 - 700.0)  # -0.075 GPa/C


def youngs_modulus_pa(temp_k):
    """Young's modulus in Pa. [BAM2023] below 900 C, linear extrapolation above.

    E(1000 C) ~= 122 GPa by this route; [ISIJ1993] cites Andrews at 118.7 GPa for
    1270 K. Take ~120 GPa with roughly +/-3% spread.
    """
    t_c = temp_k - 273.15
    if t_c <= _BAM_T_C[-1]:
        return float(np.interp(t_c, _BAM_T_C, _BAM_E_GPA)) * 1e9
    return (_BAM_E_GPA[-1] + _E_SLOPE_GPA_PER_C * (t_c - _BAM_T_C[-1])) * 1e9


def shear_modulus_pa(temp_k):
    """Shear modulus in Pa. [BAM2023] below 900 C, linear extrapolation above."""
    t_c = temp_k - 273.15
    if t_c <= _BAM_T_C[-1]:
        return float(np.interp(t_c, _BAM_T_C, _BAM_G_GPA)) * 1e9
    slope = (47.0 - 53.0) / (900.0 - 700.0)
    return (_BAM_G_GPA[-1] + slope * (t_c - _BAM_T_C[-1])) * 1e9


# nu = E/(2G) - 1 fitted across the whole [BAM2023] table. Computed from the
# tables above rather than hardcoded, so it follows if they are ever corrected.
_NU_VS_T_FIT = np.polyfit(
    np.array(_BAM_T_C, dtype=float),
    np.array(_BAM_E_GPA, dtype=float) / (2.0 * np.array(_BAM_G_GPA, dtype=float)) - 1.0,
    1,
)


def poisson_ratio(temp_k):
    """Poisson's ratio, derived from [BAM2023]'s own E and G.

    nu = E/(2G) - 1 is the standard identity, and it is exactly how [ORNL1985]
    derived nu for types 304/316. Applied POINT BY POINT to the BAM table it is
    noisy, because E and G are tabulated to whole GPa and a 1 GPa change in G
    moves nu by ~0.03: the 300 C row gives 0.261 (E fell 10 GPa while G did not
    move at all) against 0.385 at 850 C. A linear fit across the whole table
    averages that rounding out and recovers the monotonic rise that is physically
    expected -- [ISIJ1993] establishes that nu RISES with temperature for
    austenitic steel:

        nu(T_C) ~= 0.3068 + 7.638e-05 * T_C

    That gives 0.383 at 1000 C and 0.308 at room temperature. Evaluating the live
    E and G functions directly at 1000 C instead gives 0.381, so the two routes
    agree to 0.002 -- inside the rounding scatter.

    ⚠️ WHAT THIS REPLACED, and why it was wrong. Until 2026-08-07 this returned
    ``clip(0.28 + 0.00005*(T_C - 20), 0.28, 0.33)`` -- an interpolation invented
    for this repo, giving 0.329 at forging temperature. It was low by ~0.05 and
    disagreed with the very dataset that supplies E. It was kept only because
    nu = 0.382 had been MEASURED TO BREAK THE SIMULATION: substep_dt came from
    sqrt(E/rho), which does not depend on nu at all, so a higher bulk modulus
    pushed the run past the P-wave CFL limit. That was a TIMESTEP defect, not a
    material fact, and the old docstring said as much -- "the honest fix is the
    timestep, not this constant". That fix now ships: see cfl_use_pwave in
    agforge/options.py.

    ⚠️ Still not high-confidence. [ORNL1985] warns that a derived nu degrades at
    high temperature because small errors in E and G reinforce, so the ~0.005
    agreement above is rounding scatter and understates the true uncertainty.
    [ISIJ1993] measured nu directly over 300-1500 K and would settle it outright
    if someone can pull its tabulated values. The 0.49 ceiling below is numerical
    insurance only (nu -> 0.5 sends the bulk modulus to infinity); the fit does
    not reach it below ~2500 C, far above melting.
    """
    t_c = temp_k - 273.15
    return float(np.clip(np.polyval(_NU_VS_T_FIT, t_c), 0.25, 0.49))


def density_kg_m3(temp_k):
    """Density in kg/m3. [NIST2021] SRM 1155a, D(T) = 8052 - 0.564 T, T in K.

    Fit domain 500 K <= T <= solidus. At 1273 K this gives 7334 kg/m3 - 9.1%
    below the 8000 currently used in agforge/options.py, and 8.8% below the
    room-temperature 7980 in material_properties.py.

    KNOWN INTER-LAB SPREAD, roughly 2-3% at forging temperature:

        NIST SRM 1155a, pulse heating, at 1273 K      7334 kg/m3  (this function)
        [BAM2023] measured 7535 @ 900 C, extrapolated 7481 kg/m3
        room-T density + mean CTE to 1000 C           ~7570 kg/m3

    At 1173 K the two direct measurements differ by 1.9% (NIST 7391 vs BAM 7535),
    so this is genuine inter-laboratory disagreement on slightly different heats,
    not an arithmetic error. NIST is used as primary because SRM 1155a is a
    certified reference material measured directly at temperature; the CTE route
    is weakest and is listed only because it is what most datasheets imply.

    Treat 7330-7570 as the defensible band. The choice matters little for MPM
    dynamics (it enters the wave speed as a square root: the full 3.2% spread
    moves dt by 1.6%) but it should not be quoted to four figures.
    """
    return 8052.0 - 0.564 * float(temp_k)


# --------------------------------------------------------------------------
# Johnson-Cook, for the current solver
# --------------------------------------------------------------------------
#: A is pinned to this fraction of the peak flow stress rather than free-fitted.
#: See the long note in ``fit_johnson_cook`` for why that is deliberate.
JC_A_FRACTION_OF_PEAK = 0.47


def fit_johnson_cook(strain_rate, temp_k, max_strain=DRX_PEAK_STRAIN,
                     a_fixed_mpa=None):
    """Least-squares fit of ``sigma_y = A + B eps^n`` to the Arrhenius curve.

    Returns ``(A_pa, B_pa, n, rms_mpa)``. With ``a_fixed_mpa=None`` (default) A is
    pinned to ``JC_A_FRACTION_OF_PEAK`` x peak stress and only B and n are fitted.

    WHY A IS PINNED RATHER THAN FITTED
    ----------------------------------
    The fit cannot constrain A. The residual is nearly flat in it - at 1000 C and
    1 /s, RMS runs 0.85% at A=1 MPa to 1.55% at A=100 MPa, so least squares just
    slides A to whatever bound it is given and pays almost nothing for it.

    That does not make A unimportant. The solver evaluates this expression from
    ``eps_p = 0``, which is OUTSIDE the fit domain (the source data starts at
    strain 0.05). At eps_p = 0 the expression collapses to A, so A *is* the
    initial yield stress the simulation sees. A free fit happily returns A = 1 MPa,
    which fits the 0.05-0.30 window beautifully and gives the sim a billet that
    yields at essentially zero load.

    So: pin A somewhere physically defensible for hot 316L, and let B and n absorb
    the rest. The cost is ~0.7 percentage points of RMS; the benefit is an initial
    yield that is not a fitting artifact. ``JC_A_FRACTION_OF_PEAK`` = 0.47 puts A
    near 100 MPa at 1 /s while keeping every point within ~2.3%.

    Fitted only up to ``max_strain`` (default: the DRX peak). Beyond the peak the
    real material SOFTENS and this functional form cannot follow it - see
    ``johnson_cook_divergence``.
    """
    eps, sig = flow_curve_mpa(strain_rate, temp_k)
    mask = eps <= max_strain + 1e-9
    eps, sig = eps[mask], sig[mask]

    if a_fixed_mpa is None:
        a_fixed_mpa = JC_A_FRACTION_OF_PEAK * float(sig.max())
    if a_fixed_mpa >= sig.min():
        raise ValueError(
            f"a_fixed_mpa={a_fixed_mpa:.1f} MPa must sit below the smallest "
            f"fitted flow stress ({sig.min():.1f} MPa)"
        )

    n, ln_b = np.polyfit(np.log(eps), np.log(sig - a_fixed_mpa), 1)
    b_mpa = math.exp(ln_b)
    rms = float(np.sqrt(np.mean((a_fixed_mpa + b_mpa * eps ** n - sig) ** 2)))
    return a_fixed_mpa * 1e6, b_mpa * 1e6, float(n), rms


def johnson_cook_divergence(strain_rate, temp_k):
    """How wrong a pre-peak JC fit goes past the DRX peak, as a percentage.

    Returns ``[(strain, arrhenius_mpa, jc_mpa, pct_error), ...]`` for the
    tabulated strains beyond the peak. This is the quantitative case against
    using original Johnson-Cook in the DRX regime.
    """
    a_pa, b_pa, n, _ = fit_johnson_cook(strain_rate, temp_k)
    eps, sig = flow_curve_mpa(strain_rate, temp_k)
    out = []
    for e, s in zip(eps, sig):
        if e <= DRX_PEAK_STRAIN + 1e-9:
            continue
        jc = a_pa / 1e6 + (b_pa / 1e6) * e ** n
        out.append((float(e), float(s), float(jc), 100.0 * (jc - s) / s))
    return out


# --------------------------------------------------------------------------
# Ready-to-use isothermal card
# --------------------------------------------------------------------------
#: Nominal forging condition for the Agility Forge digital twin.
#: 1 /s is the low end of the forging band; see FORGING_STRAIN_RATE_BAND.
FORGING_TEMP_K = 1273.15
FORGING_STRAIN_RATE_BAND = (1.0, 100.0)


def isothermal_card(strain_rate=1.0, temp_k=FORGING_TEMP_K):
    """Everything needed to configure an isothermal run, in SI.

    ``use_johnson_cook`` consumers can read jc_A / jc_B / jc_n straight out of
    this. NOTE the thermal-softening term is NOT folded in: this card already
    represents the material AT ``temp_k``, so a solver applying an additional
    (1 - T*^m) factor would double-count. Keep the run isothermal.
    """
    a_pa, b_pa, n, rms = fit_johnson_cook(strain_rate, temp_k)
    return {
        "temp_k": temp_k,
        "strain_rate_per_s": strain_rate,
        "E": youngs_modulus_pa(temp_k),
        "G": shear_modulus_pa(temp_k),
        "nu": poisson_ratio(temp_k),
        "rho": density_kg_m3(temp_k),
        "t_melt_k": T_SOLIDUS_K,
        "peak_flow_stress_pa": peak_flow_stress_mpa(strain_rate, temp_k) * 1e6,
        "jc_A": a_pa,
        "jc_B": b_pa,
        "jc_n": n,
        "jc_fit_rms_mpa": rms,
        "jc_valid_max_strain": DRX_PEAK_STRAIN,
    }


if __name__ == "__main__":  # pragma: no cover - reporting aid
    print("316L at 1000 C - peak flow stress vs strain rate")
    for rate in (0.1, 1.0, 10.0, 100.0):
        print(f"  {rate:>6g} /s   {peak_flow_stress_mpa(rate, FORGING_TEMP_K):7.1f} MPa")

    print("\nisothermal card @ 1 /s")
    for k, v in isothermal_card(1.0).items():
        print(f"  {k:<24}{v}")

    print("\nJohnson-Cook divergence past the DRX peak (1 /s)")
    for e, s, jc, err in johnson_cook_divergence(1.0, FORGING_TEMP_K):
        print(f"  strain {e:.2f}   real {s:6.1f}   JC {jc:6.1f}   {err:+5.1f}%")
