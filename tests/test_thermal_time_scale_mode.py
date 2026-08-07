"""Pin the relationship between the thermal and mechanical time scales.

``thermal_time_scale`` (S_T) is derived from the explicit-diffusion CFL ceiling,
which is a NUMERICAL bound -- it says nothing about how fast the press moves.
At the shipped card that makes one macro step advance 1.66 s of thermal time
while the mechanics advance 17.1 ms of real-equivalent time: a ~97x disagreement.

That is harmless while a press runs with temperatures frozen (StrikeController
defaults to ``thermal_enabled = False``, snapshotting and restoring particle
temperatures around every physics step). It stops being harmless the moment
thermal physics runs during a strike, which is what die chill would require.
Measured idle over 100 steps from a uniform 1273.15 K billet:

    S_T = 171653 -> mean 965.8 K, min 473.4 K   (166 s of thermal time)
    S_T =   1773 -> mean 1267.4 K, min 1232.7 K (1.7 s of thermal time)

``thermal_time_scale_mode = 'mechanical'`` ties the two clocks together.

The ratio assertions below are deliberately scoped to ``mode == 'cfl'`` so that
switching the default to 'mechanical' does not fail a test on a CORRECT config.

Nothing here initialises a backend or touches the GPU.
"""
import pytest

from agforge.options import StrikeOptions, TeleopOptions

# Measured from blow #1 of the 2026-06-15 T4 dataset: 7.12 mm of bite in 0.505 s.
REAL_DIE_SPEED_MS = 0.0141
SIM_PRESS_SPEED_MS = 25.0
MECH_ACCEL = SIM_PRESS_SPEED_MS / REAL_DIE_SPEED_MS  # ~1773x


@pytest.fixture
def cfl_cfg():
    return TeleopOptions()


@pytest.fixture
def mech_cfg():
    return TeleopOptions(thermal_time_scale_mode="mechanical")


def test_the_real_die_speed_is_the_measured_one():
    assert StrikeOptions().real_die_speed == pytest.approx(REAL_DIE_SPEED_MS)
    assert StrikeOptions().pressing_speed == pytest.approx(SIM_PRESS_SPEED_MS)


def test_default_mode_is_cfl(cfl_cfg):
    """The induction workstream is tuned against the CFL value; keep it default."""
    assert cfl_cfg.thermal_time_scale_mode == "cfl"


def test_mechanical_mode_matches_the_press_acceleration(mech_cfg):
    assert mech_cfg.mpm.thermal_time_scale == pytest.approx(MECH_ACCEL, rel=1e-9)


def test_mechanical_mode_is_strictly_more_stable(cfl_cfg, mech_cfg):
    """S_T only ever shrinks, so 'mechanical' cannot destabilise a working config."""
    assert mech_cfg.mpm.thermal_time_scale < cfl_cfg.mpm.thermal_time_scale


def test_the_cfl_default_disagrees_with_the_mechanical_clock(cfl_cfg):
    """Document the ~97x gap that motivates the mode. Scoped to the cfl default."""
    if cfl_cfg.thermal_time_scale_mode != "cfl":
        pytest.skip("default is no longer 'cfl'; this guard describes that regime")
    ratio = cfl_cfg.mpm.thermal_time_scale / MECH_ACCEL
    assert ratio > 50.0, (
        f"S_T/mech_accel = {ratio:.1f}. If this dropped, the CFL derivation "
        f"changed and the comment in options.py needs re-measuring."
    )


def test_thermal_time_per_macro_step(cfl_cfg, mech_cfg):
    """One macro step should advance real-process time under 'mechanical' mode."""
    macro_dt = cfl_cfg.sim.dt
    cfl_thermal_s = macro_dt * cfl_cfg.mpm.thermal_time_scale
    mech_thermal_s = macro_dt * mech_cfg.mpm.thermal_time_scale

    # Under 'mechanical', thermal time per step == real time per step.
    real_time_per_step = macro_dt * MECH_ACCEL
    assert mech_thermal_s == pytest.approx(real_time_per_step, rel=1e-9)

    # A 0.505 s blow is ~74 steps in the sim; under 'cfl' that is minutes of cooling.
    assert cfl_thermal_s > 1.0
    assert mech_thermal_s < 0.05


def test_an_unknown_mode_is_rejected():
    """Validated by the Literal field, not by a raise in model_post_init.

    Raising ValueError there produced "not enough values to unpack" -- pydantic
    mangles exceptions from model_post_init, so the field type does the work.
    """
    with pytest.raises(Exception) as excinfo:
        TeleopOptions(thermal_time_scale_mode="physical")
    msg = str(excinfo.value)
    # Genesis wraps pydantic's ValidationError in GenesisException, which is not
    # a ValueError, so assert on the message rather than the wrapper type.
    assert "thermal_time_scale_mode" in msg
    assert "cfl" in msg and "mechanical" in msg  # allowed values are named


def test_mechanical_mode_stays_within_the_diffusion_cfl_ceiling(cfl_cfg, mech_cfg):
    """The ceiling is 4x the shipped value (shipped = 0.25 of it)."""
    ceiling = cfl_cfg.mpm.thermal_time_scale / 0.25
    assert mech_cfg.mpm.thermal_time_scale < ceiling
