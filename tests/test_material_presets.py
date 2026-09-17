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
