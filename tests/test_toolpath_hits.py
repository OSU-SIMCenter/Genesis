"""The two unit conventions in a recorded toolpath, and what the loader does with them.

A toolpath records ``rho`` as the HALF gap and ``phi`` in DEGREES. ``Hit`` wants the
full gap in millimetres and radians. Getting either wrong produces a sequence that
runs to completion and is quietly wrong -- a factor of two on every gap, or a
rotation in the wrong units -- so both are asserted here rather than described in a
comment.

The expected values below were read from the recorded files and cross-check against
figures derived independently of this code: roll1 press 1 is a 17.07 mm half-gap at
z = 60.1 mm, and roll3 press 1 is a 14.76 mm commanded half-thickness at z = 17.36 mm.
"""
import json
import math
import sys
from pathlib import Path

import pytest

# The loader is exercised without forge_common or Genesis present: the module imports
# both at module scope, so the function is extracted and given a stand-in Hit.
_DRIVER = Path(__file__).resolve().parents[1] / "agforge" / "scripts" / "record_toolpath_hits.py"


class _Hit:
    def __init__(self, rho, phi, z, duration):
        self.rho, self.phi, self.z, self.duration = rho, phi, z, duration


def _load_fn():
    src = _DRIVER.read_text(encoding="utf-8")
    body = src[src.index("def load_toolpath("):src.index("def verify_material(")]
    ns = {"json": json, "math": math, "Path": Path, "Hit": _Hit, "RHO_SCALE": 2.0}
    exec(compile(body, str(_DRIVER), "exec"), ns)
    return ns["load_toolpath"]


load_toolpath = _load_fn()


def _write(tmp_path, records):
    p = tmp_path / "toolpath.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    return p


def _press(rho, phi, z, **extra):
    act = {"rho": rho, "phi": phi, "z": z, "anvil_tool": 2, "hammer_tool": 2,
           "scan_after": False, "heat_after": False, "no_op": False}
    act.update(extra)
    return act


def test_rho_is_doubled_into_a_full_gap(tmp_path):
    """roll1 press 1 is recorded as 17.07 and is a 34.14 mm die gap."""
    p = _write(tmp_path, [{"action": _press(17.07, 184.1, 60.1), "workpiece_length_mm": 75.0}])
    hits, _ = load_toolpath(p)
    assert hits[0].rho == pytest.approx(34.14)


def test_phi_is_converted_from_degrees(tmp_path):
    p = _write(tmp_path, [{"action": _press(14.76, 136.5, 17.36), "workpiece_length_mm": 100.0}])
    hits, _ = load_toolpath(p)
    assert hits[0].phi == pytest.approx(math.radians(136.5))
    assert hits[0].phi < math.tau, "phi is radians, so it cannot exceed one turn"


def test_no_op_records_are_not_presses_but_still_give_the_length(tmp_path):
    """Every recorded series opens with a no_op that carries the starting length."""
    p = _write(tmp_path, [
        {"action": {"no_op": True, "scan_after": True}, "workpiece_length_mm": 100.0},
        {"action": _press(16.43, 172.0, 77.06), "workpiece_length_mm": 100.0},
    ])
    hits, meta = load_toolpath(p)
    assert len(hits) == 1
    assert meta["records"] == 2
    assert meta["exposed_length_mm"] == pytest.approx(100.0)


def test_n_hits_zero_means_every_press(tmp_path):
    p = _write(tmp_path, [{"action": _press(15.0 + i, 90.0, 20.0 + i)} for i in range(5)])
    assert len(load_toolpath(p, 0)[0]) == 5
    assert len(load_toolpath(p, 2)[0]) == 2


def test_asking_for_more_presses_than_exist_is_an_error(tmp_path):
    p = _write(tmp_path, [{"action": _press(15.0, 90.0, 20.0)}])
    with pytest.raises(ValueError):
        load_toolpath(p, 9)


def test_z_shift_moves_every_press_together(tmp_path):
    """Used with chuck-side stock: z stays measured from the chuck face."""
    recs = [{"action": _press(15.0, 90.0, 20.0)}, {"action": _press(14.0, 0.0, 40.0)}]
    p = _write(tmp_path, recs)
    base = load_toolpath(p)[0]
    moved = load_toolpath(p, z_shift_mm=25.0)[0]
    for b, m in zip(base, moved):
        assert m.z == pytest.approx(b.z + 25.0)
        assert m.rho == pytest.approx(b.rho), "a z shift must not touch the gap"


def test_a_file_with_no_presses_is_an_error(tmp_path):
    p = _write(tmp_path, [{"action": {"no_op": True}, "workpiece_length_mm": 75.0}])
    with pytest.raises(ValueError):
        load_toolpath(p)


def test_missing_length_is_reported_as_missing_not_guessed(tmp_path):
    """The caller refuses to invent an exposed length, so the loader must say None
    rather than substituting a plausible default."""
    p = _write(tmp_path, [{"action": _press(15.0, 90.0, 20.0)}])
    _, meta = load_toolpath(p)
    assert meta["exposed_length_mm"] is None


def test_malformed_json_names_the_line(tmp_path):
    p = tmp_path / "bad.jsonl"
    p.write_text('{"action": {"rho": 15, "phi": 0, "z": 20}}\nnot json\n', encoding="utf-8")
    with pytest.raises(ValueError) as e:
        load_toolpath(p)
    assert "line 2" in str(e.value)
