"""Option M: AGF_MATERIAL must be a superset, never a swap.

The governing invariant, enforced here rather than asserted in prose:

    With AGF_MATERIAL unset or "316L", every MaterialOptions value must be
    identical to what this tree shipped before presets existed.

That is what keeps the 2026-09-17 proof run of 1423bd87 valid on this line.
"""
import pytest

from agforge.material_presets import (
    MATERIAL_PRESETS,
    UnknownMaterialError,
    active_material_preset,
)
from agforge.options import MaterialOptions

_FIELDS = sorted(MaterialOptions.model_fields)


def _card(monkeypatch, material=None):
    monkeypatch.delenv("AGF_MATERIAL", raising=False)
    if material is not None:
        monkeypatch.setenv("AGF_MATERIAL", material)
    m = MaterialOptions()
    return {k: getattr(m, k) for k in _FIELDS}


def test_316L_preset_is_empty():
    """The declared field defaults ARE the 316L card. A second copy of those
    numbers would be a second source of truth -- the shape that already bit us
    with rho carried as both 7980.0 and 7334."""
    assert MATERIAL_PRESETS["316L"] == {}


def test_unset_equals_316L(monkeypatch):
    assert _card(monkeypatch, None) == _card(monkeypatch, "316L")


def test_316L_does_not_move_any_field(monkeypatch):
    baseline = _card(monkeypatch, None)
    assert baseline["E"] == 121.5e9
    assert baseline["nu"] == 0.383
    assert baseline["rho"] == 7334.0
    assert baseline["jc_T_ref"] == 1273.15
    assert baseline["jc_T_melt"] == 1675.0


def test_6063t52_applies_the_cold_card(monkeypatch):
    al = _card(monkeypatch, "6063T52")
    assert al["E"] == 68.9e9
    assert al["nu"] == 0.33
    assert al["rho"] == 2700.0
    assert al["jc_T_ref"] == 293.15
    assert al["jc_T_melt"] == 933.15


def test_explicit_caller_is_never_clobbered(monkeypatch):
    """An external driver setting a field explicitly must win over the preset."""
    monkeypatch.setenv("AGF_MATERIAL", "6063T52")
    m = MaterialOptions(E=999.0e9)
    assert m.E == 999.0e9
    assert m.rho == 2700.0


def test_unknown_material_fails_loud(monkeypatch):
    monkeypatch.setenv("AGF_MATERIAL", "6061T6")
    with pytest.raises(UnknownMaterialError):
        active_material_preset()


def test_geometry_and_cfl_are_not_material_properties():
    """billet_length_m and target_cfl_ratio stay OUT of presets: selecting a
    material must not silently change the domain or the stability policy."""
    for preset in MATERIAL_PRESETS.values():
        assert "billet_length_m" not in preset
        assert "target_cfl_ratio" not in preset


def test_6063_400C_is_the_hot_card(monkeypatch):
    """400 C is carried in jc_A with thermal off, so the elastic card does not move."""
    hot = _card(monkeypatch, "6063_400C")
    cold = _card(monkeypatch, "6063T52")
    assert hot["jc_A"] == 75.0e6
    assert hot["E"] == cold["E"] and hot["nu"] == cold["nu"] and hot["rho"] == cold["rho"]


def test_6063_400C_is_perfectly_plastic(monkeypatch):
    """jc_B = 0. With hardening left on the cold card's value the hot bar would be
    STRONGER than the cold one by 0.2 strain, which is the opposite of the intent."""
    hot = _card(monkeypatch, "6063_400C")
    cold = _card(monkeypatch, "6063T52")
    assert hot["jc_B"] == 0.0
    for eps in (0.0, 0.1, 0.2, 0.3):
        hot_flow = hot["jc_A"] + hot["jc_B"] * eps ** hot["jc_n"]
        cold_flow = cold["jc_A"] + cold["jc_B"] * eps ** cold["jc_n"]
        assert hot_flow < cold_flow, f"hot card is not softer at eps={eps}"


def test_johnson_cook_triple_always_lands_together():
    """jc_A, jc_B and jc_n set the flow curve jointly. A preset that moves one and
    inherits the others mixes two materials' constants -- the bug that made the
    first 6063 card 27.3% too soft at yield."""
    triple = ("jc_A", "jc_B", "jc_n")
    for name, preset in MATERIAL_PRESETS.items():
        present = [k for k in triple if k in preset]
        assert len(present) in (0, 3), f"{name} sets only {present} of the flow-curve triple"
