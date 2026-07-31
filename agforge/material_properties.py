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
consistency.py`` asserts the kernel literals still agree with the values here — that
test is what keeps the two honest, not convention.

Material selection
------------------
The simulation was originally parameterised for **AISI 4340** low-alloy steel. The
2026-07-17 Colton Wright calibration dataset is **316L austenitic stainless**, so
316L is now the default. Both are kept: 4340 is still the right material if the
production forge runs low-alloy steel, and the choice is a real open question
rather than a settled one.

The two are not interchangeable, and not merely offset:

  * ``k`` runs the OPPOSITE direction with temperature — 4340 falls 44 -> 27 W/m-K,
    316L rises 14.6 -> ~29 W/m-K. They differ by 3x at room temperature and nearly
    coincide at forging temperature, so the error is concentrated in the heat-up
    transient, which is exactly what a steady-state calibration run exercises.
  * 4340's ``cp`` jumps at ~1000 K because of the ferrite/austenite transformation.
    316L is austenitic at all temperatures: no transformation, no Curie point. Any
    "delta(T) Curie transition" modelling work is meaningless for 316L.
  * 316L is paramagnetic (mu_r ~ 1) at ALL temperatures, whereas carbon/low-alloy
    steel is ferromagnetic below its Curie point. For induction this actually
    *simplifies* 316L: no mu_r(T) collapse to model.

Sources / uncertainty
---------------------
316L values are anchored on the Outokumpu / AK Steel 316L (1.4404) datasheets and
ASM Handbook Vol. 1. Published room-temperature ``k`` for 316L spans roughly
14.6-16.3 W/m-K depending on source and product form, so treat these curves as
+/- ~10%, not exact. The 4340 curves are transcribed unchanged from the values
that were already in the solver ("Data from ASM Handbook Vol. 1 & MatWeb 4340
datasheet").
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np

MU0 = 4.0e-7 * math.pi  # vacuum permeability [H/m]


# --------------------------------------------------------------------------- 316L
def k_316l(temp: np.ndarray | float) -> np.ndarray:
    """Thermal conductivity of 316L [W/m-K]. RISES with temperature.

    Anchors: 15.0 @ 293 K, 16.3 @ 373 K, ~21 @ 773 K (datasheet), extrapolated
    linearly at ~0.0127 W/m-K per K to forging temperature.
    """
    t = np.asarray(temp, dtype=np.float64)
    k = np.full_like(t, 14.6)
    hi = t >= 1000.0
    k[hi] = 23.6 + (t[hi] - 1000.0) * 0.0127
    mid = (t >= 700.0) & (t < 1000.0)
    k[mid] = 19.8 + (t[mid] - 700.0) / 300.0 * 3.8
    lo = (t > 293.15) & (t < 700.0)
    k[lo] = 14.6 + (t[lo] - 293.15) / 406.85 * 5.2
    return k


def cp_316l(temp: np.ndarray | float) -> np.ndarray:
    """Specific heat of 316L [J/kg-K]. Flat-ish — no phase transformation spike."""
    t = np.asarray(temp, dtype=np.float64)
    cp = np.full_like(t, 500.0)
    hi = t >= 1000.0
    cp[hi] = 585.0 + (t[hi] - 1000.0) * 0.05
    mid = (t >= 700.0) & (t < 1000.0)
    cp[mid] = 550.0 + (t[mid] - 700.0) / 300.0 * 35.0
    lo = (t > 293.15) & (t < 700.0)
    cp[lo] = 500.0 + (t[lo] - 293.15) / 406.85 * 50.0
    return cp


def rho_e_316l(temp: np.ndarray | float) -> np.ndarray:
    """Electrical resistivity of 316L [ohm-m].

    7.4e-7 at room temperature rising to ~1.05e-6 near 1200 K. Modest and roughly
    linear — austenitic stainless has a much flatter resistivity curve than carbon
    steel, which is one reason induction coupling into 316L is comparatively stable.
    """
    t = np.asarray(temp, dtype=np.float64)
    return 7.4e-7 + (np.clip(t, 293.15, 1500.0) - 293.15) * 3.42e-10


# -------------------------------------------------------------------------- 4340
def k_4340(temp: np.ndarray | float) -> np.ndarray:
    """Thermal conductivity of AISI 4340 [W/m-K]. FALLS with temperature."""
    t = np.asarray(temp, dtype=np.float64)
    k = np.full_like(t, 44.0)
    hi = t >= 1000.0
    k[hi] = 27.0
    mid = (t >= 700.0) & (t < 1000.0)
    k[mid] = 35.0 - (t[mid] - 700.0) / 300.0 * 8.0
    lo = (t > 293.15) & (t < 700.0)
    k[lo] = 44.0 - (t[lo] - 293.15) / 406.85 * 9.0
    return k


def cp_4340(temp: np.ndarray | float) -> np.ndarray:
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


def rho_e_4340(temp: np.ndarray | float) -> np.ndarray:
    """Electrical resistivity of low-alloy steel [ohm-m], crude linear form.

    Only meaningful ABOVE the Curie point (~1043 K) where the steel is
    paramagnetic; below it, mu_r is large and strongly field-dependent, so a
    single-valued skin depth is not physical. Present for comparison only.
    """
    t = np.asarray(temp, dtype=np.float64)
    return 1.6e-7 + (np.clip(t, 293.15, 1500.0) - 293.15) * 8.6e-10


@dataclass(frozen=True)
class BilletMaterial:
    """Thermal + electromagnetic properties of the billet material."""

    name: str
    rho_kg_m3: float           # density used by the THERMAL kernels
    t_melt_k: float            # Johnson-Cook thermal-softening reference
    mu_r: float                # relative permeability at forging temperature
    k: Callable[[np.ndarray | float], np.ndarray]
    cp: Callable[[np.ndarray | float], np.ndarray]
    rho_e: Callable[[np.ndarray | float], np.ndarray]
    ferromagnetic_below_curie: bool
    notes: str = ""

    def alpha(self, temp: np.ndarray | float) -> np.ndarray:
        """Thermal diffusivity k/(rho*cp) [m^2/s] — governs the thermal CFL bound."""
        return self.k(temp) / (self.rho_kg_m3 * self.cp(temp))

    def alpha_worst(self, t_lo: float = 293.15, t_hi: float = 1500.0) -> float:
        """Max diffusivity over the operating range — the CFL-limiting value."""
        grid = np.linspace(t_lo, t_hi, 512)
        return float(self.alpha(grid).max())

    def skin_depth_m(self, freq_hz: float, temp_k: float = 1273.15) -> float:
        """EM skin depth delta = sqrt(rho_e / (pi * f * mu0 * mu_r)) [m].

        This is a MATERIAL + FREQUENCY property. It does not depend on billet
        radius; see the note in ``agforge/options.py`` about the legacy
        ``radius / 2`` heuristic that made it appear to.
        """
        if freq_hz <= 0.0:
            raise ValueError("freq_hz must be positive")
        rho_e = float(self.rho_e(temp_k))
        return math.sqrt(rho_e / (math.pi * freq_hz * MU0 * self.mu_r))

    def implied_frequency_hz(self, skin_depth_m: float, temp_k: float = 1273.15) -> float:
        """Inverse of :meth:`skin_depth_m` — what frequency a given delta implies.

        Useful for auditing a hardcoded delta: if the implied frequency is not a
        plausible number for the real power supply, the delta is wrong.
        """
        if skin_depth_m <= 0.0:
            raise ValueError("skin_depth_m must be positive")
        rho_e = float(self.rho_e(temp_k))
        return rho_e / (math.pi * skin_depth_m ** 2 * MU0 * self.mu_r)


STEEL_316L = BilletMaterial(
    name="316L austenitic stainless",
    rho_kg_m3=7980.0,
    t_melt_k=1673.0,          # liquidus ~1400 C; solidus ~1375 C
    mu_r=1.0,                 # paramagnetic at ALL temperatures
    k=k_316l,
    cp=cp_316l,
    rho_e=rho_e_316l,
    ferromagnetic_below_curie=False,
    notes=(
        "Material of the 2026-07-17 Colton Wright calibration run "
        "(38.1 mm rod, 420.2 A coil current)."
    ),
)

AISI_4340 = BilletMaterial(
    name="AISI 4340 low-alloy steel",
    rho_kg_m3=7850.0,
    t_melt_k=1793.0,
    mu_r=1.0,                 # above Curie; meaningless below it
    k=k_4340,
    cp=cp_4340,
    rho_e=rho_e_4340,
    ferromagnetic_below_curie=True,
    notes=(
        "The simulation's original material. Retained because whether the "
        "production forge actually runs low-alloy steel is still an open question."
    ),
)

MATERIALS = {"316L": STEEL_316L, "4340": AISI_4340}

#: Active billet material. Switch here to re-target the whole thermal stack.
#: NOTE: the GPU kernel curves in base_mpm_solver.py do NOT read this — they are
#: compiled literals. Changing this alone will silently desynchronise them; the
#: consistency test will fail and tell you so.
ACTIVE_MATERIAL = STEEL_316L
