"""AGF_DIE_MESH swaps the contact surface and nothing else.

The box die is the shipped behaviour, so the first test is byte-identity of the
generated XML against the generator before the knob existed: a mesh option that
quietly perturbs the box runs would invalidate every comparison made with them.
"""
import subprocess

import pytest

from agforge.agforge_builder import RobotXMLGenerator
from agforge.options import RobotOptions

BAR = dict(cylinder_diameter=0.0381, cylinder_height=0.075)


def _xml(monkeypatch, mesh=None):
    monkeypatch.delenv("AGF_DIE_MESH", raising=False)
    if mesh is not None:
        monkeypatch.setenv("AGF_DIE_MESH", str(mesh))
    return RobotXMLGenerator(RobotOptions(**BAR)).generate_content()


def test_box_die_is_the_default(monkeypatch):
    xml = _xml(monkeypatch)
    assert xml.count('type="box" size=') == 2
    assert 'type="mesh"' not in xml
    assert "die_mesh" not in xml


def test_box_xml_is_unchanged_from_before_the_knob(monkeypatch):
    """Generated with the generator as it stood on origin/main. If this drifts, the
    box runs are no longer comparable with the ones already published."""
    ref_src = subprocess.run(
        ["git", "show", "origin/main:agforge/agforge_builder.py"],
        capture_output=True, text=True,
    )
    if ref_src.returncode != 0:
        pytest.skip("origin/main not available in this checkout")
    import re
    src = ref_src.stdout
    # Run only the reference template, not its imports: pull out the class body's
    # generate_content by exec'ing the file with the package already imported.
    ns = {}
    src = src.replace("from agforge.wsl_graphics import apply_early_wsl_graphics_defaults",
                      "apply_early_wsl_graphics_defaults = lambda: None")
    src = re.sub(r"^apply_early_wsl_graphics_defaults\(\)$", "", src, flags=re.M)
    exec(compile(src, "ref_builder", "exec"), ns)
    ref_xml = ns["RobotXMLGenerator"](RobotOptions(**BAR)).generate_content()
    assert _xml(monkeypatch) == ref_xml


def test_mesh_replaces_both_jaw_geoms(monkeypatch, tmp_path):
    stl = tmp_path / "tool.stl"
    stl.write_text("solid x endsolid x")
    xml = _xml(monkeypatch, stl)
    assert 'type="box" size=' not in xml
    assert xml.count('type="mesh" mesh="die_mesh"') == 2
    assert f'<mesh name="die_mesh" file="{stl}" scale="0.001 0.001 0.001"/>' in xml


def test_mesh_face_sits_where_the_box_face_was(monkeypatch, tmp_path):
    """Each jaw's mesh is placed at its own contact face, 0.4 * radius out, and the
    jaw closing along +y is turned 180 degrees so its body is behind that face."""
    stl = tmp_path / "tool.stl"
    stl.write_text("solid x endsolid x")
    xml = _xml(monkeypatch, stl)
    half_y = 0.0381 / 2.0 * 0.4
    assert f'pos="0 {half_y:.6f} 0" euler="180 0 0"' in xml
    assert f'pos="0 {-half_y:.6f} 0" euler="0 0 0"' in xml


def test_missing_mesh_is_an_error(monkeypatch, tmp_path):
    """Falling back to the box would log as a mesh run."""
    with pytest.raises(FileNotFoundError):
        _xml(monkeypatch, tmp_path / "absent.stl")
