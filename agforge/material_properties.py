"""Single documented source of truth for billet thermal / electromagnetic properties.

Why this module exists
----------------------
Before this, the same material constants were hardcoded in four places that could
drift independently:

  * ``genesis/engine/solvers/base_mpm_solver.py`` — ``get_steel_cp`` /
    ``get_steel_thermal_conductivity`` (the GPU kernel path, the one that actually
    integrates the heat equation)
  * ``agforge/thermal.py`` — numpy/torch CPU mirrors
  * ``agforge/thermal_calibration.py`` — ``get_steel_k_numpy`` + ``STEEL_RHO_THERMAL``
  * ``agforge/options.py`` — the CFL ``alpha_worst`` bound and the ``skin_depth``
    heuristic

The GPU kernel cannot import this module (its curves compile to literals inside a
``@qd.func``), so it necessarily keeps its own copy. ``tests/test_material_property_
consistency.py`` asserts the kernel literals still agree with the values here.

Material selection
------------------
The simulation was originally parameterised for **AISI 4340** low-alloy steel. The
2026-07-17 Colton Wright calibration dataset is **316L austenitic stainless**, so
316L is now the default. Both are kept: 4340 is still the right material if the
production forge runs low-alloy steel, and that is a real open question.

The two are not interchangeable, and not merely offset:

  * ``k`` runs the OPPOSITE direction with temperature — 4340 falls 44 -> 27 W/m-K,
    316L rises 13.3 -> ~30. They differ ~3x at room temperature and nearly coincide
    at forging temperature, so the error is concentrated in the heat-up transient,
    which is exactly what a calibration heating run exercises.
  * 4340's ``cp`` jumps at ~1000 K from the ferrite/austenite transformation. 316L is
    austenitic at all temperatures: no transformation, no Curie point. Any
    "delta(T) Curie transition" modelling is meaningless for 316L.
  * 316L is paramagnetic (mu_r ~ 1) at ALL temperatures, whereas carbon/low-alloy
    steel is ferromagnetic below its Curie point. For induction this *simplifies*
    316L: no mu_r(T) collapse to model.

Sources
-------
316L curves are now anchored on published measurements rather than datasheet
round numbers:

  [1] Ho & Chu, "Electrical Resistivity and Thermal Conductivity of Nine Selected
      AISI Stainless Steels", CINDAS/AFML report (DTIC ADA129160). Critically
      evaluated recommended values for AISI 316 synthesised from 23 conductivity
      and 8 resistivity datasets, 1 K to melt. Used for k(T) and rho_e(T) — taking
      both from ONE evaluation keeps them mutually consistent, which matters
      because the Wiedemann-Franz link ties them together.
  [2] Pichler et al., "Measurements of thermophysical properties of solid and
      liquid NIST SRM 1155a (AISI 316L)", J. Mater. Sci. (NIST pub 928362).
      Ohmic pulse-heating + DSC on the NIST 316L standard reference material.
      Used for cp(T). Note their DSC shows a kink near 820 K attributed to
      precipitates dissolving.
  [3] Balat-Pichelin, Sans & Beche, "Spectral directional and total hemispherical
      emissivity of virgin and oxidized 316L stainless steel from 1000 to 1650 K",
      Infrared Physics & Technology. Used for emissivity.
  [4] Hunnewell et al., "Total Hemispherical Emissivity of SS 316L...",
      Nuclear Technology 198(3), and Al Zubaidi et al. (2018) for the
      as-received / lightly-oxidised low-temperature end.

Spread between sources is real: [2]'s conductivity fit runs ~8% below [1] at
1000 K on a different heat of material. Treat k as +/- ~10%.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np

MU0 = 4.0e-7 * math.pi  # vacuum permeability [H/m]


def _piecewise(temp, knots):
    """Piecewise-linear interpolation through (T, value) knots, flat-extrapolated
    below the first knot and linearly extrapolated above the last using the final
    segment's slope. Mirrors what the GPU kernel does with compiled literals."""
    t = np.asarray(temp, dtype=np.float64)
    ts = np.array([k[0] for k in knots], dtype=np.float64)
    vs = np.array([k[1] for k in knots], dtype=np.float64)
    out = np.interp(t, ts, vs)
    slope = (vs[-1] - vs[-2]) / (ts[-1] - ts[-2])
    hi = t > ts[-1]
    out = np.where(hi, vs[-1] + (t - ts[-1]) * slope, out)
    return out


# --------------------------------------------------------------------------- 316L
# Ho & Chu [1], AISI 316 recommended values (W/m-K). NOTE: rises with temperature.
_K_316L = [(293.0, 13.31), (400.0, 15.16), (500.0, 16.80), (600.0, 18.36),
           (700.0, 19.87), (800.0, 21.39), (900.0, 22.79), (1000.0, 24.16),
           (1100.0, 25.46), (1200.0, 26.74), (1300.0, 28.02), (1400.0, 29.32),
           (1500.0, 30.61)]

# Pichler et al. [2], NIST SRM 1155a DSC (J/kg-K). 293 K anchored on the standard
# datasheet value; their DSC series starts at 473 K.
_CP_316L = [(293.0, 500.0), (473.0, 528.0), (573.0, 537.0), (673.0, 553.0),
            (773.0, 566.0), (873.0, 595.0), (973.0, 603.0), (1073.0, 611.0),
            (1173.0, 619.0), (1253.0, 624.0)]

# Ho & Chu [1], AISI 316 recommended values (Ohm-m). Saturating, NOT linear —
# the slope roughly halves between room temperature and forging temperature.
_RHO_E_316L = [(293.0, 77.1e-8), (400.0, 85.2e-8), (500.0, 91.7e-8),
               (600.0, 97.7e-8), (700.0, 103.1e-8), (800.0, 108.0e-8),
               (900.0, 112.1e-8), (1000.0, 115.7e-8), (1100.0, 118.9e-8),
               (1200.0, 121.7e-8), (1300.0, 124.3e-8), (1400.0, 126.8e-8),
               (1500.0, 129.2e-8)]


def k_316l(temp):
    """Thermal conductivity of 316L [W/m-K], Ho & Chu recommended values."""
    return _piecewise(temp, _K_316L)


def cp_316l(temp):
    """Specific heat of 316L [J/kg-K], NIST SRM 1155a DSC."""
    return _piecewise(temp, _CP_316L)


def rho_e_316l(temp):
    """Electrical resistivity of 316L [Ohm-m], Ho & Chu recommended values."""
    return _piecewise(temp, _RHO_E_316L)


def _three_segment(temp, v0, t1, v1, t2, v2, slope_hi, t0=293.0):
    """The 3-segment piecewise-linear form the GPU kernel compiles to.

    The kernel cannot interpolate a knot table inside a @qd.func, so it uses three
    linear segments plus a slope above t2. These helpers exist so the CPU mirrors
    use the SAME approximation as the kernel rather than the fuller knot table —
    otherwise CPU and GPU would quietly disagree by a percent or two. The test
    bounds this approximation against the full tables.
    """
    t = np.asarray(temp, dtype=np.float64)
    out = np.full_like(t, v0)
    hi = t >= t2
    out[hi] = v2 + (t[hi] - t2) * slope_hi
    mid = (t >= t1) & (t < t2)
    out[mid] = v1 + (t[mid] - t1) / (t2 - t1) * (v2 - v1)
    lo = (t > t0) & (t < t1)
    out[lo] = v0 + (t[lo] - t0) / (t1 - t0) * (v1 - v0)
    return out


#: (v0, t1, v1, t2, v2, slope_above_t2, t0) — the literals the GPU kernel compiles.
#: Exported so torch/numpy mirrors can share them instead of re-typing the numbers.
K_316L_SEG_PARAMS = (13.31, 700.0, 19.87, 1000.0, 24.16, 0.0129, 293.0)
CP_316L_SEG_PARAMS = (500.0, 700.0, 556.5, 1000.0, 605.2, 0.0743, 293.0)


def k_316l_seg(temp):
    """316L conductivity in the kernel's 3-segment form [W/m-K]."""
    return _three_segment(temp, *K_316L_SEG_PARAMS[:6], t0=K_316L_SEG_PARAMS[6])


def cp_316l_seg(temp):
    """316L specific heat in the kernel's 3-segment form [J/kg-K]."""
    return _three_segment(temp, *CP_316L_SEG_PARAMS[:6], t0=CP_316L_SEG_PARAMS[6])


def emissivity_316l(temp, oxidation="in_situ_air"):
    """Total hemispherical emissivity of 316L [-].

    ⚠️ This is the single largest uncertainty in the thermal calibration, and it is
    NOT a constant. Balat-Pichelin et al. [3] measured as-received 316L oxidising in
    situ in air: emissivity climbs from ~0.25 to ~0.70 across 1100-1500 K as the
    oxide layer forms, then stabilises near 0.75. Virgin (unoxidised) 316L sits at
    only ~0.25-0.30 [3][4].

    Two consequences for the 2026-07-17 calibration run, which heated a bare rod in
    air for ~9.5 minutes:

      1. The radiative loss term is a moving target during the run. The value 0.8
         previously assumed in calibration.json is at the very top of the range and
         is only reached once the oxide is fully developed.
      2. The thermal camera's reported temperature depends on the emissivity it was
         configured with. If it assumed a fixed value while the true surface
         emissivity was climbing, part of the apparent temperature rise is an
         artefact. The camera's emissivity setting is not recorded anywhere in the
         mcap and needs to come from Colton.

    ``oxidation`` selects a regime:
      "virgin"        - bright/unoxidised, ~0.25-0.30, weakly T-dependent
      "in_situ_air"   - as-received oxidising in air (DEFAULT, matches the run)
      "fully_oxidised"- developed oxide layer, ~0.75
    """
    t = np.asarray(temp, dtype=np.float64)
    if oxidation == "virgin":
        return _piecewise(t, [(436.0, 0.25), (1166.0, 0.36), (1650.0, 0.40)])
    if oxidation == "fully_oxidised":
        return _piecewise(t, [(1000.0, 0.70), (1500.0, 0.75), (1650.0, 0.75)])
    return _piecewise(t, [(436.0, 0.25), (1000.0, 0.30), (1100.0, 0.35),
                          (1300.0, 0.52), (1500.0, 0.70), (1650.0, 0.75)])


# -------------------------------------------------------------------------- 4340
def k_4340(temp):
    """Thermal conductivity of AISI 4340 [W/m-K]. FALLS with temperature.
    Transcribed unchanged from the values previously in the solver."""
    t = np.asarray(temp, dtype=np.float64)
    k = np.full_like(t, 44.0)
    hi = t >= 1000.0
    k[hi] = 27.0
    mid = (t >= 700.0) & (t < 1000.0)
    k[mid] = 35.0 - (t[mid] - 700.0) / 300.0 * 8.0
    lo = (t > 293.15) & (t < 700.0)
    k[lo] = 44.0 - (t[lo] - 293.15) / 406.85 * 9.0
    return k


def cp_4340(temp):
    """Specific heat of AISI 4340 [J/kg-K]. Jump at 1000 K = ferrite/austenite."""
    t = np.asarray(temp, dtype=np.float64)
    cp = np.full_like(t, 450.0)
    hi = t >= 1000.0
    cp[hi] = 750.0
    mid = (t >= 700.0) & (t < 1000.0)
    cp[mid] = 580.0 + (t[mid] - 700.0) / 300.0 * 70.0
    lo = (t > 293.15) & (t < 700.0)
    cp[lo] = 450.0 + (t[lo] - 293.15) / 406.85 * 130.0
    return cp


def rho_e_4340(temp):
    """Electrical resistivity of low-alloy steel [Ohm-m], crude linear form.

    Only meaningful ABOVE the Curie point (~1043 K) where the steel is
    paramagnetic; below it mu_r is large and strongly field-dependent, so a
    single-valued skin depth is not physical. Present for comparison only.
    """
    t = np.asarray(temp, dtype=np.float64)
    return 1.6e-7 + (np.clip(t, 293.15, 1500.0) - 293.15) * 8.6e-10


@dataclass(frozen=True)
class BilletMaterial:
    """Thermal + electromagnetic properties of the billet material."""

    name: str
    rho_kg_m3: float
    t_melt_k: float
    mu_r: float
    k: Callable
    cp: Callable
    rho_e: Callable
    ferromagnetic_below_curie: bool
    emissivity: Callable | None = None
    notes: str = ""

    def alpha(self, temp):
        """Thermal diffusivity k/(rho*cp) [m^2/s] — governs the thermal CFL bound."""
        return self.k(temp) / (self.rho_kg_m3 * self.cp(temp))

    def alpha_worst(self, t_lo: float = 293.15, t_hi: float = 1500.0) -> float:
        """Max diffusivity over the operating range — the CFL-limiting value."""
        return float(self.alpha(np.linspace(t_lo, t_hi, 512)).max())

    def skin_depth_m(self, freq_hz: float, temp_k: float = 1273.15) -> float:
        """EM skin depth delta = sqrt(rho_e / (pi * f * mu0 * mu_r)) [m].

        A MATERIAL + FREQUENCY property. It does not depend on billet radius; see
        the note in options.py about the legacy ``radius / 2`` heuristic.
        """
        if freq_hz <= 0.0:
            raise ValueError("freq_hz must be positive")
        return math.sqrt(float(self.rho_e(temp_k)) / (math.pi * freq_hz * MU0 * self.mu_r))

    def implied_frequency_hz(self, skin_depth_m: float, temp_k: float = 1273.15) -> float:
        """Inverse of :meth:`skin_depth_m` — what frequency a given delta implies.

        Useful for auditing a hardcoded delta: if the implied frequency is not a
        plausible number for the real power supply, the delta is wrong.
        """
        if skin_depth_m <= 0.0:
            raise ValueError("skin_depth_m must be positive")
        return float(self.rho_e(temp_k)) / (math.pi * skin_depth_m ** 2 * MU0 * self.mu_r)


STEEL_316L = BilletMaterial(
    name="316L austenitic stainless",
    rho_kg_m3=7980.0,
    t_melt_k=1675.0,          # solidus, Pichler et al. [2] (liquidus 1708 K)
    mu_r=1.0,                 # paramagnetic at ALL temperatures
    k=k_316l,
    cp=cp_316l,
    rho_e=rho_e_316l,
    emissivity=emissivity_316l,
    ferromagnetic_below_curie=False,
    notes=("Material of the 2026-07-17 Colton Wright calibration run "
           "(38.1 mm rod, 420.2 A coil current)."),
)

AISI_4340 = BilletMaterial(
    name="AISI 4340 low-alloy steel",
    rho_kg_m3=7850.0,
    t_melt_k=1793.0,
    mu_r=1.0,                 # above Curie only; meaningless below it
    k=k_4340,
    cp=cp_4340,
    rho_e=rho_e_4340,
    emissivity=None,
    ferromagnetic_below_curie=True,
    notes=("The simulation's original material. Retained because whether the "
           "production forge actually runs low-alloy steel is still open."),
)

MATERIALS = {"316L": STEEL_316L, "4340": AISI_4340}

#: Active billet material. Switch here to re-target the whole thermal stack.
#: NOTE: the GPU kernel curves in base_mpm_solver.py do NOT read this — they are
#: compiled literals. Changing this alone silently desynchronises them; the
#: consistency test will fail and say so.
ACTIVE_MATERIAL = STEEL_316L
