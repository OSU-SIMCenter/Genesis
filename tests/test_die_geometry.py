"""Guard the die contact footprint against drifting back to a scaled billet.

The dies used to be a box derived from the BILLET radius (0.5r x 1.2r), which
makes the tool a function of the workpiece -- so changing billet diameter
silently changed the press. The real dies are forge_common's ``Tool2_clip.obj``,
and the 2026-06-15 T4 dataset names them explicitly: every action in its
embedded plan carries ``anvil_tool: 2, hammer_tool: 2``.

Contact area sets force, so the along-bar width is the load-bearing number.
These tests pin it to the tool measurement and, just as importantly, prove the
footprint change did NOT move the press kinematics -- the closed gap is set by
the die thickness along the approach axis, which deliberately stays on the old
rule.

Nothing here initialises a backend or touches the GPU.
"""
import numpy as np
import pytest

from agforge.agforge_builder import RobotXMLGenerator
from agforge.options import RobotOptions

# forge_common/main/forge_common/press_tool.py + real_scale.py
TOOL2_TIP_AXIAL_MM = 9.49        # narrow contact tip
TOOL2_EFFECTIVE_AXIAL_MM = 13.48  # die_contact_axial_width_mm(), 25% plateau
TOOL2_SHANK_AXIAL_MM = 15.8187    # REAL_PRESS_AXIAL_BAND_MM, whole-tool bbox
TOOL2_LATERAL_MM = 54.7116        # REAL_PRESS_LATERAL_MM

LEGACY_AXIAL_MM = 19.05           # the old 0.5 * cylinder_radius * 2 box
BILLET_DIA_MM = 38.1


@pytest.fixture
def cfg():
    return RobotOptions()


@pytest.fixture
def die_full_mm(cfg):
    """Die full extents in mm: (along-bar, thickness, across-bar)."""
    return np.asarray(RobotXMLGenerator(cfg).gripper_size) * 2000.0


def test_along_bar_width_is_the_tool_effective_contact_band(die_full_mm):
    assert die_full_mm[0] == pytest.approx(TOOL2_EFFECTIVE_AXIAL_MM, abs=1e-6)


def test_along_bar_width_lies_between_the_tool_tip_and_shank(die_full_mm):
    """The taper means any flat-plane width must sit inside these bounds."""
    assert TOOL2_TIP_AXIAL_MM <= die_full_mm[0] <= TOOL2_SHANK_AXIAL_MM


def test_die_is_no_longer_the_legacy_billet_derived_box(die_full_mm):
    assert die_full_mm[0] != pytest.approx(LEGACY_AXIAL_MM, abs=0.01), (
        "the along-bar die width is back on the billet-radius rule; it must "
        "come from the Tool2 measurement")


def test_the_legacy_box_overstated_contact_area(die_full_mm):
    """Regression direction matters: the old box was too WIDE, so it
    over-predicted force. Peak force scales linearly with this."""
    assert die_full_mm[0] < LEGACY_AXIAL_MM
    ratio = die_full_mm[0] / LEGACY_AXIAL_MM
    assert ratio == pytest.approx(0.7076, abs=1e-3)


def test_lateral_span_covers_the_bar_so_it_never_limits_contact(die_full_mm):
    assert die_full_mm[2] == pytest.approx(TOOL2_LATERAL_MM, abs=1e-6)
    assert die_full_mm[2] > BILLET_DIA_MM


def test_die_footprint_does_not_depend_on_billet_diameter():
    """A tool is not a function of the workpiece. This is the structural bug
    the old 0.5r / 1.2r rule encoded."""
    small = np.asarray(RobotXMLGenerator(
        RobotOptions(billet_diameter_in=1.0)).gripper_size) * 2000.0
    large = np.asarray(RobotXMLGenerator(
        RobotOptions(billet_diameter_in=2.0)).gripper_size) * 2000.0
    assert small[0] == pytest.approx(large[0])   # along-bar
    assert small[2] == pytest.approx(large[2])   # across-bar


def test_die_thickness_still_tracks_the_billet_on_purpose(cfg, die_full_mm):
    """The approach-axis thickness stays on the legacy rule because
    gripper_closed_y derives the closed gap from it. Changing it would move
    the kinematics rather than the contact area."""
    assert die_full_mm[1] == pytest.approx(cfg.cylinder_radius * 0.4 * 2000.0)


def test_press_kinematics_are_unchanged_by_the_footprint(cfg):
    """Slide range and start position must depend only on the Y half-extent,
    so re-sizing the contact face cannot alter how far the press can close."""
    gen = RobotXMLGenerator(cfg)
    r = cfg.cylinder_radius
    y_half = r * 0.4
    expected_open = r * 2.5
    expected_closed = r - y_half * 0.5
    assert gen.gripper_start_pos_y == pytest.approx(expected_open)
    assert gen.gripper_slide_range[1] == pytest.approx(
        expected_open - expected_closed)


def test_options_carry_the_tool_numbers(cfg):
    assert cfg.die_axial_width_m * 1000.0 == pytest.approx(
        TOOL2_EFFECTIVE_AXIAL_MM, abs=1e-6)
    assert cfg.die_lateral_span_m * 1000.0 == pytest.approx(
        TOOL2_LATERAL_MM, abs=1e-6)
