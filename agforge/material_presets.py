"""Selectable billet material presets (Option M: mechanical-only, no registry).

Selected with AGF_MATERIAL. Unset is equivalent to "316L".

DESIGN INVARIANT, and the reason "316L" is an EMPTY dict:
    With AGF_MATERIAL unset or "316L", every resulting option value must be
    byte-identical to the values this tree shipped before presets existed.

The 316L card IS the declared field defaults in options.MaterialOptions. Writing
those numbers out again here would create a second source of truth that has to be
kept in sync by hand -- the same shape as `rho` already being carried twice
(BilletMaterial 7980.0 vs MaterialOptions 7334., 8.8% apart, nothing saying which
consumer reads which). So "316L" overrides nothing and the invariant holds by
construction rather than by transcription.

The JC flow-curve triple (jc_A/jc_B/jc_n) must always land together; see the
note beside them.

Deliberately NOT here: `billet_length_m` and `target_cfl_ratio`. Those are stock
geometry and solver time-step policy, not material properties. Putting them in a
preset would mean selecting a material silently changes the simulation domain and
the stability policy. In particular the roll1 overlay's `target_cfl_ratio` 0.041
is a SYMPTOM -- an approach-CFL cut from 0.35 after a 299 m/s abort -- and baking
it in would permanently hide the instability it signals.

stdlib only, no agforge imports, so this file moves verbatim like the T52 card did.
"""
from __future__ import annotations

# Keys here are MaterialOptions field names, except the two marked (builder).
MATERIAL_PRESETS: dict[str, dict] = {
    # The shipped hot-steel card. Empty on purpose -- see the module docstring.
    "316L": {},

    # Colton's roll1 stock: cold 6063-T52 aluminium, room temperature, isothermal.
    # Sourced in docs/6063_T52_MECHANICAL_PROPERTIES.md, which states its own
    # limits: no measured T52 flow curve exists, jc_n is a typical Al Ludwik
    # exponent and not measured on this bar, and the card is bounded to a cold
    # shallow bite. Do not use it past ~0.30 strain without a measured curve.
    "6063T52": {
        "E": 68.9e9,
        "nu": 0.33,
        "rho": 2700.0,
        "von_mises_yield_stress": 137.9e6,
        # use_johnson_cook defaults True, so environment.py takes the JohnsonCookPlasticity
        # branch and von_mises_yield_stress above is NOT read -- it is here for the
        # ElastoPlastic branch and to document the card. The live flow curve is
        #     sigma_y = jc_A + jc_B * eps_p**jc_n
        # so the JC triple is what actually sets strength, and all three must land together.
        # Setting only jc_n left jc_A/jc_B at 316L's 1000 C values, which made this card
        # 27.3% too soft at eps=0 and 10.5% too soft at eps=0.30 -- aluminium geometry with
        # hot-steel flow constants. Values below are the roll1 T52 overlay's.
        "jc_A": 137.9e6,        # T52 typical YS; shipped default is 100.3e6 (316L @ 1000 C)
        "jc_B": 173.6e6,        # fitted to T52 typical UTS at eps=0.08; shipped is 195.0e6
        "jc_n": 0.45,
        "jc_C": 0.0,            # dead code in materials.py; carried so it is right if wired up
        "jc_T_ref": 293.15,     # RT card: reference IS room temperature
        "jc_T_melt": 933.15,
        "jc_m": 1.0,
        # (builder) consumed in AgilityForgeOptions, not MaterialOptions fields:
        "enable_thermal": False,   # cold, no induction; adiabatic rise ~8 K total
        "billet_temp_k": 293.15,
    },
}

DEFAULT_MATERIAL = "316L"


class UnknownMaterialError(ValueError):
    """Raised for an AGF_MATERIAL value with no preset -- fail loud, never silently 316L."""


def active_material_preset(name: str | None = None) -> dict:
    """Return the preset dict for `name`, or for AGF_MATERIAL when name is None."""
    if name is None:
        import os
        name = os.environ.get("AGF_MATERIAL") or DEFAULT_MATERIAL
    name = name.strip()
    if name not in MATERIAL_PRESETS:
        raise UnknownMaterialError(
            "AGF_MATERIAL=%r is not a known material. Known: %s"
            % (name, ", ".join(sorted(MATERIAL_PRESETS)))
        )
    return dict(MATERIAL_PRESETS[name])
