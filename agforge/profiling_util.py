"""Shared helpers for hierarchical teleop / simulator profiling."""

from __future__ import annotations

import contextlib
from typing import Any


def _profiling_options(target: Any):
    if target is None:
        return None
    po = getattr(target, "profiling_options", None)
    if po is not None:
        return po
    scene = getattr(target, "scene", None)
    if scene is not None:
        return getattr(scene, "profiling_options", None)
    return None


def teleop_profile(target: Any, name: str):
    """Profile a teleop section when enabled in ProfilingOptions.configs.teleop."""
    profiling = _profiling_options(target)
    if profiling is None or not profiling.enabled:
        return contextlib.suppress()
    opt_name = name.replace("teleop_", "")
    if getattr(profiling.configs.teleop, opt_name, False):
        return profiling.profiler.time(name)
    return contextlib.suppress()


def simulator_profile(target: Any, name: str, *, flag: str | None = None):
    """Profile a simulator section when enabled in ProfilingOptions.configs.simulator."""
    profiling = _profiling_options(target)
    if profiling is None or not profiling.enabled:
        return contextlib.suppress()
    flag_name = flag or name
    if getattr(profiling.configs.simulator, flag_name, True):
        return profiling.profiler.time(name)
    return contextlib.suppress()
