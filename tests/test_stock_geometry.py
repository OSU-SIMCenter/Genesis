"""Where the bar sits, and how much of it the clamp holds.

Every number here was measured from the resolved options, not predicted. They are
asserted because the two knobs under test exist to stop a specific failure: the
clamp growing with the bar until it holds the material the dies are striking.

The frame: the bar runs along x, is pinned at +x, and elongates into -x. The clamp
box is centred on the pinned face, so half of it hangs off the end of the stock and
the length it actually grips is (pinned face - box inner edge).
"""
import pytest

from agforge.options import RobotOptions

# The aluminium programs' two innermost blows, as millimetres from the chuck face.
NEAREST_BLOWS_MM = (15.68, 17.87)


def _bar(monkeypatch, length_mm, **knobs):
    for k in ("AGF_PIN_EXTRA_MM", "AGF_CLAMP_DEPTH_MM", "AGF_CLAMP_FRACTION"):
        monkeypatch.delenv(k, raising=False)
    for k, v in knobs.items():
        monkeypatch.setenv(k, str(v))
    r = RobotOptions(cylinder_diameter=0.0381, cylinder_height=length_mm / 1000.0)
    bounds = r.fixed_region_bounds.numpy()
    half = r.cylinder_height / 2.0
    centre = float(r.cylinder_pos[0])
    return {
        "length_mm": r.cylinder_height * 1e3,
        "free_end_mm": (centre - half) * 1e3,
        "pinned_face_mm": (centre + half) * 1e3,
        "clamp_inner_mm": float(bounds[0][0]) * 1e3,
        "held_mm": ((centre + half) - float(bounds[0][0])) * 1e3,
    }


def test_default_geometry_is_unchanged(monkeypatch):
    """No knob set: bar centred on the origin, clamp 0.35 * 75 mm biting half of that."""
    b = _bar(monkeypatch, 75.0)
    assert b["length_mm"] == pytest.approx(75.0)
    assert b["free_end_mm"] == pytest.approx(-37.5)
    assert b["pinned_face_mm"] == pytest.approx(37.5)
    assert b["clamp_inner_mm"] == pytest.approx(24.375)
    assert b["held_mm"] == pytest.approx(13.125)


def test_held_stock_does_not_move_the_free_end(monkeypatch):
    """The whole point of adding on +x: z is measured from the chuck face, so the
    free end has to stay put or every blow moves relative to it."""
    plain = _bar(monkeypatch, 75.0)
    held = _bar(monkeypatch, 75.0, AGF_PIN_EXTRA_MM=25)
    assert held["free_end_mm"] == pytest.approx(plain["free_end_mm"])
    assert held["length_mm"] == pytest.approx(100.0)
    assert held["pinned_face_mm"] == pytest.approx(62.5)


def test_fraction_clamp_swallows_the_nearest_blow_on_a_long_bar(monkeypatch):
    """This is the failure the absolute clamp exists for. At the shipped 0.35 a
    100 mm bar is gripped 17.50 mm back from the pinned face, and the innermost
    blow is at 15.68 mm -- inside the rigid region, where material cannot flow."""
    b = _bar(monkeypatch, 100.0)
    assert b["held_mm"] == pytest.approx(17.5)
    assert min(NEAREST_BLOWS_MM) < b["held_mm"], "expected the clamp to reach past the blow"


def test_depth_clamp_stops_at_the_chuck_face(monkeypatch):
    """Depth equal to the held stock puts the clamp's inner edge exactly on the
    chuck face, so every blow lands outside the rigid region."""
    b = _bar(monkeypatch, 75.0, AGF_PIN_EXTRA_MM=25, AGF_CLAMP_DEPTH_MM=25)
    assert b["clamp_inner_mm"] == pytest.approx(37.5)
    assert b["held_mm"] == pytest.approx(25.0)
    # The chuck face is where the exposed bar begins: 25 mm of held stock behind a
    # 100 mm bar pinned at +62.5 leaves it at +37.5. A blow z mm out from that face
    # sits at (37.5 - z), and the clamp occupies everything at or above 37.5.
    chuck_face_mm = b["pinned_face_mm"] - 25.0
    assert chuck_face_mm == pytest.approx(b["clamp_inner_mm"])
    for z in NEAREST_BLOWS_MM:
        blow_x = chuck_face_mm - z
        assert blow_x < b["clamp_inner_mm"], (
            f"blow at z={z} mm sits at x={blow_x:.2f}, inside the clamp "
            f"(inner edge {b['clamp_inner_mm']:.2f})"
        )


def test_the_clamp_test_can_actually_fail(monkeypatch):
    """Guard on the assertion above: run the SAME check against the fraction clamp
    on a bar with no held stock, where it must fail. Without this, a test that
    only ever passes proves nothing about where the clamp is."""
    b = _bar(monkeypatch, 100.0)
    chuck_face_mm = b["pinned_face_mm"]
    inside = [z for z in NEAREST_BLOWS_MM if (chuck_face_mm - z) >= b["clamp_inner_mm"]]
    assert inside, "expected the fraction clamp to swallow at least one blow"


def test_both_clamp_spellings_is_an_error(monkeypatch):
    with pytest.raises(Exception) as excinfo:
        _bar(monkeypatch, 75.0, AGF_CLAMP_DEPTH_MM=25, AGF_CLAMP_FRACTION=0.2625)
    assert "AGF_CLAMP_DEPTH_MM" in str(excinfo.value)


def test_negative_held_stock_is_an_error(monkeypatch):
    with pytest.raises(Exception) as excinfo:
        _bar(monkeypatch, 75.0, AGF_PIN_EXTRA_MM=-5)
    assert "AGF_PIN_EXTRA_MM" in str(excinfo.value)
