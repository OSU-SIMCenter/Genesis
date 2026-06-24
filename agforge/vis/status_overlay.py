"""Minimal top-right status text for the Genesis viewer (color mode + SDF mesh)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from agforge.environment import AgilityForgeEnv
    from agforge.physics_mesh import InductionPhysicsMesher


def _viewer_safe_text(text: str) -> str:
    return "".join(c if 32 <= ord(c) < 127 else "?" for c in str(text))


from genesis.vis.viewer_plugins import ViewerPlugin


class ViewerStatusPlugin(ViewerPlugin):
    """Renders one or two short status lines at the top-right of the viewer."""

    def __init__(self):
        super().__init__()
        self._lines: list[str] = []

    def set_lines(self, lines: list[str]) -> None:
        self._lines = [_viewer_safe_text(line) for line in lines if line]

    def on_draw(self) -> None:
        if not self._lines or self.viewer is None:
            return

        from genesis.ext.pyrender.constants import FONT_SIZE, TEXT_PADDING, TextAlign

        viewer = self.viewer
        x = viewer._viewport_size[0] - TEXT_PADDING
        y = viewer._viewport_size[1] - TEXT_PADDING
        color = np.array([0.85, 0.92, 1.0, 0.95], dtype=np.float32)

        for i, line in enumerate(self._lines):
            viewer._renderer.render_text(
                line,
                x,
                int(y - i * FONT_SIZE * 1.15),
                font_pt=FONT_SIZE,
                color=color,
                align=TextAlign.TOP_RIGHT,
            )


def _refresh_keybind_help(env: "AgilityForgeEnv") -> None:
    vis = env.scene.visualizer
    if vis is None or vis.viewer is None:
        return
    pr = getattr(vis.viewer, "_pyrender_viewer", None)
    if pr is not None and hasattr(pr, "_update_instr_texts"):
        pr._update_instr_texts()


def update_viewer_status(
    env: "AgilityForgeEnv",
    *,
    physics_mesher: "InductionPhysicsMesher | None" = None,
    status_plugin: ViewerStatusPlugin | None = None,
    extra_lines: list[str] | None = None,
) -> None:
    """Show current particle color mode and induction SDF mesh backend (top-right)."""
    from agforge.vis.temperature_particles import PARTICLE_COLOR_MODE_LABELS

    if status_plugin is None:
        status_plugin = getattr(env, "_viewer_status_plugin", None)
    if status_plugin is None:
        return

    mode = getattr(env.cfg.general, "particle_color_mode", "temperature")
    color_label = PARTICLE_COLOR_MODE_LABELS.get(mode, mode)
    if getattr(env.cfg.general, "particle_simple_color", False):
        color_label = f"{color_label} (simple)"

    perf = getattr(env.cfg, "performance", None)
    viewer_fps = int(getattr(perf, "target_viewer_fps", 0) or 0) if perf else 0

    unified = bool(getattr(env.cfg.reconstruction, "unified_mesh", True))
    mesh_label = ""
    if physics_mesher is not None:
        backend = physics_mesher.backend_label
        mesh_label = f"{backend} (unified)" if unified else backend

    lines = [f"Color: {color_label}"]
    if viewer_fps > 0:
        lines.append(f"Viewer: {viewer_fps} FPS")
    if mesh_label:
        prefix = "Mesh" if unified else "SDF mesh"
        lines.append(f"{prefix}: {mesh_label}")
    if extra_lines:
        lines.extend(extra_lines)

    status_plugin.set_lines(lines)


def refresh_viewer_status_with_tuner(
    env: "AgilityForgeEnv",
    *,
    tuner=None,
    physics_mesher: "InductionPhysicsMesher | None" = None,
) -> None:
    """Refresh the HUD including optional thermal tuner readout."""
    extra = tuner.format_status_lines() if tuner is not None else None
    update_viewer_status(env, physics_mesher=physics_mesher, extra_lines=extra)


def install_viewer_status_plugin(env: "AgilityForgeEnv") -> ViewerStatusPlugin | None:
    if not getattr(env.cfg.general, "show_viewer", False):
        return None
    vis = env.scene.visualizer
    if vis is None or vis.viewer is None:
        return None

    plugin = ViewerStatusPlugin()
    vis.viewer.add_plugin(plugin)
    env._viewer_status_plugin = plugin

    # Clear any legacy caption HUD so it does not overlap this overlay.
    pr = getattr(vis.viewer, "_pyrender_viewer", None)
    if pr is not None:
        pr.viewer_flags["caption"] = None

    return plugin
