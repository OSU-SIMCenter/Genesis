"""Reconstructed mesh overlays in the Genesis viewer (visual vs physics meshes)."""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Callable

import numpy as np
import trimesh

if TYPE_CHECKING:
    from agforge.environment import AgilityForgeEnv
    from agforge.physics_mesh import InductionPhysicsMesher
    from agforge.reconstruction import SurfaceReconstructor


class MeshDisplayMode(str, Enum):
    OFF = "off"
    OPAQUE = "opaque"
    TRANSPARENT = "transparent"


MESH_DISPLAY_MODES: tuple[MeshDisplayMode, ...] = (
    MeshDisplayMode.OFF,
    MeshDisplayMode.OPAQUE,
    MeshDisplayMode.TRANSPARENT,
)

MESH_DISPLAY_MODE_LABELS: dict[str, str] = {
    MeshDisplayMode.OFF.value: "off",
    MeshDisplayMode.OPAQUE.value: "opaque",
    MeshDisplayMode.TRANSPARENT.value: "translucent",
}


def mesh_for_genesis_viewer(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Flip winding so outward normals face the camera in pyrender.

    Reconstruction stores Unity-oriented winding (faces reversed from MC output).
    Genesis/pyrender needs the opposite convention.
    """
    out = mesh.copy()
    if len(out.faces) > 0:
        out.faces = out.faces[:, ::-1]
    return out


def _cycle_display_mode(current: MeshDisplayMode) -> MeshDisplayMode:
    modes = MESH_DISPLAY_MODES
    idx = modes.index(current)
    return modes[(idx + 1) % len(modes)]


class ReconMeshOverlay:
    """Draws optional visual and physics reconstruction meshes in the Genesis viewer."""

    def __init__(
        self,
        ctx,
        *,
        visual_color=(0.25, 0.75, 0.95, 1.0),
        physics_color=(0.95, 0.55, 0.15, 1.0),
        transparent_alpha: float = 0.35,
    ):
        from genesis.ext import pyrender

        self._ctx = ctx
        self._pyrender = pyrender
        self._visual_color = np.array(visual_color, dtype=np.float32)
        self._physics_color = np.array(physics_color, dtype=np.float32)
        self._transparent_alpha = float(transparent_alpha)
        self.visual_mode = MeshDisplayMode.OFF
        self.physics_mode = MeshDisplayMode.OFF
        self._visual_node = None
        self._physics_node = None
        self._visual_mesh: trimesh.Trimesh | None = None
        self._physics_mesh: trimesh.Trimesh | None = None
        self._visual_stamp: int = -1
        self._physics_stamp: int = -1

    @classmethod
    def from_env(cls, env: "AgilityForgeEnv") -> "ReconMeshOverlay":
        ctx = env.scene.visualizer._context
        return cls(ctx)

    def cycle_visual_mode(self) -> MeshDisplayMode:
        self.visual_mode = _cycle_display_mode(self.visual_mode)
        self._refresh_node("visual")
        return self.visual_mode

    def cycle_physics_mode(self) -> MeshDisplayMode:
        self.physics_mode = _cycle_display_mode(self.physics_mode)
        self._refresh_node("physics")
        return self.physics_mode

    def cycle_unified_display(self) -> MeshDisplayMode:
        """Cycle mesh overlay display for unified visual+physics mesh."""
        self.visual_mode = _cycle_display_mode(self.visual_mode)
        self.physics_mode = MeshDisplayMode.OFF
        self._remove_node("physics")
        self._refresh_node("visual")
        return self.visual_mode

    def sync_meshes(
        self,
        visual_mesh: trimesh.Trimesh | None,
        physics_mesh: trimesh.Trimesh | None,
        *,
        visual_stamp: int | None = None,
        physics_stamp: int | None = None,
    ) -> None:
        if visual_stamp is not None and visual_stamp == self._visual_stamp:
            visual_changed = False
        else:
            visual_changed = True
            self._visual_stamp = -1 if visual_stamp is None else int(visual_stamp)
            self._visual_mesh = (
                visual_mesh.copy()
                if visual_mesh is not None and len(visual_mesh.vertices) >= 4
                else None
            )

        if physics_stamp is not None and physics_stamp == self._physics_stamp:
            physics_changed = False
        else:
            physics_changed = True
            self._physics_stamp = -1 if physics_stamp is None else int(physics_stamp)
            self._physics_mesh = (
                physics_mesh.copy()
                if physics_mesh is not None and len(physics_mesh.vertices) >= 4
                else None
            )

        if visual_changed:
            self._refresh_node("visual")
        if physics_changed:
            self._refresh_node("physics")

    def _sync_unified_mesh(self, mesh: trimesh.Trimesh | None, stamp: int) -> None:
        """Single overlay in unified mode (one color, no visual/physics double-draw)."""
        self.physics_mode = MeshDisplayMode.OFF
        if stamp == self._visual_stamp:
            return
        self._visual_stamp = int(stamp)
        self._physics_stamp = int(stamp)
        self._visual_mesh = (
            mesh.copy() if mesh is not None and len(mesh.vertices) >= 4 else None
        )
        self._physics_mesh = None
        self._remove_node("physics")
        self._refresh_node("visual")

    def sync_from_controller(self, controller) -> None:
        unified = bool(getattr(controller.env.cfg.reconstruction, "unified_mesh", True))
        if unified:
            mesh = getattr(controller.physics_mesher, "physics_mesh", None)
            if mesh is None or len(mesh.vertices) < 4:
                mesh = getattr(controller.reconstructor, "reconstructed_mesh", None)
            stamp = getattr(controller.physics_mesher, "version", 0)
            self._sync_unified_mesh(mesh, stamp)
            return
        visual = getattr(controller.reconstructor, "reconstructed_mesh", None)
        physics = getattr(controller.physics_mesher, "physics_mesh", None)
        self.sync_meshes(
            visual,
            physics,
            visual_stamp=getattr(controller.reconstructor, "mesh_version", 0),
            physics_stamp=getattr(controller.physics_mesher, "version", 0),
        )

    def _material_for(self, rgba: np.ndarray, mode: MeshDisplayMode):
        color = rgba.copy()
        if mode == MeshDisplayMode.TRANSPARENT:
            color[3] = self._transparent_alpha
            alpha_mode = "BLEND"
        else:
            color[3] = 1.0
            alpha_mode = "OPAQUE"
        return self._pyrender.MetallicRoughnessMaterial(
            baseColorFactor=color,
            metallicFactor=0.0,
            roughnessFactor=0.85,
            alphaMode=alpha_mode,
        )

    def _remove_node(self, which: str) -> None:
        node_attr = f"_{which}_node"
        node = getattr(self, node_attr)
        if node is not None:
            self._ctx.remove_node(node)
            setattr(self, node_attr, None)

    def _refresh_node(self, which: str) -> None:
        mode: MeshDisplayMode = getattr(self, f"{which}_mode")
        mesh: trimesh.Trimesh | None = getattr(self, f"_{which}_mesh")
        color = self._visual_color if which == "visual" else self._physics_color

        self._remove_node(which)
        if mode == MeshDisplayMode.OFF or mesh is None or len(mesh.vertices) < 4:
            return

        material = self._material_for(color, mode)
        display_mesh = mesh_for_genesis_viewer(mesh)
        pr_mesh = self._pyrender.Mesh.from_trimesh(display_mesh, material=material, smooth=True)
        node = self._ctx.add_node(pr_mesh)
        setattr(self, f"_{which}_node", node)


def update_mesh_overlay_display(
    env: "AgilityForgeEnv",
    overlay: ReconMeshOverlay | None,
    physics_mesher: "InductionPhysicsMesher | None" = None,
    *,
    flash: bool = False,
    temp_renderer=None,
) -> None:
    from agforge.vis.temperature_particles import update_particle_color_display

    update_particle_color_display(env, physics_mesher=physics_mesher)


def register_mesh_overlay_keybinds(
    env: "AgilityForgeEnv",
    overlay: ReconMeshOverlay | None,
    *,
    on_physics_backend_change: Callable[[], None] | None = None,
    on_ensure_physics_mesh: Callable[[], None] | None = None,
    physics_mesher: "InductionPhysicsMesher | None" = None,
    temp_renderer=None,
) -> None:
    if overlay is None:
        return

    import genesis as gs
    from genesis.vis.keybindings import Key, KeyAction, Keybind

    vis = env.scene.visualizer
    if vis is None or vis.viewer is None:
        return

    viewer = vis.viewer

    def _refresh_viewer():
        vis.update(force=False, auto=True)

    def _cycle_visual():
        mode = overlay.cycle_visual_mode()
        gs.logger.info(f"Visual mesh display: {MESH_DISPLAY_MODE_LABELS.get(mode.value, mode.value)}")
        update_mesh_overlay_display(env, overlay, physics_mesher, temp_renderer=temp_renderer)
        _refresh_viewer()

    def _cycle_physics():
        if on_ensure_physics_mesh is not None:
            on_ensure_physics_mesh()
        mode = overlay.cycle_physics_mode()
        gs.logger.info(f"Physics mesh display: {MESH_DISPLAY_MODE_LABELS.get(mode.value, mode.value)}")
        update_mesh_overlay_display(env, overlay, physics_mesher, temp_renderer=temp_renderer)
        _refresh_viewer()

    def _cycle_physics_backend():
        if on_physics_backend_change is not None:
            on_physics_backend_change()
        else:
            update_mesh_overlay_display(env, overlay, physics_mesher, flash=True, temp_renderer=temp_renderer)
            if temp_renderer is not None:
                temp_renderer.sync_from_env(env)
        _refresh_viewer()

    unified = bool(getattr(env.cfg.reconstruction, "unified_mesh", True))

    def _cycle_unified():
        mode = overlay.cycle_unified_display()
        gs.logger.info(f"Surface mesh display: {MESH_DISPLAY_MODE_LABELS.get(mode.value, mode.value)}")
        update_mesh_overlay_display(env, overlay, physics_mesher, temp_renderer=temp_renderer)
        _refresh_viewer()

    # M/N/Y avoid Genesis defaults: V=vertex normals, P=reload shader, H=shadow.
    if unified:
        viewer.register_keybinds(
            Keybind(
                "cycle_surface_mesh",
                Key.M,
                key_action=KeyAction.PRESS,
                callback=_cycle_unified,
                allow_overload=True,
            ),
            overwrite=False,
        )
    else:
        viewer.register_keybinds(
            Keybind(
                "cycle_visual_mesh",
                Key.M,
                key_action=KeyAction.PRESS,
                callback=_cycle_visual,
                allow_overload=True,
            ),
            Keybind(
                "cycle_physics_mesh",
                Key.N,
                key_action=KeyAction.PRESS,
                callback=_cycle_physics,
                allow_overload=True,
            ),
            Keybind(
                "cycle_sdf_mesh",
                Key.Y,
                key_action=KeyAction.PRESS,
                callback=_cycle_physics_backend,
                allow_overload=True,
            ),
            overwrite=False,
        )
    from agforge.vis.status_overlay import _refresh_keybind_help

    update_mesh_overlay_display(env, overlay, physics_mesher, temp_renderer=temp_renderer)
    _refresh_keybind_help(env)


def install_mesh_overlay(
    env: "AgilityForgeEnv",
    controller,
    temp_renderer=None,
    register_keybinds: bool = True,
) -> ReconMeshOverlay | None:
    if not getattr(env.cfg.general, "show_viewer", False):
        return None
    if env.scene.visualizer is None:
        return None

    overlay = ReconMeshOverlay.from_env(env)
    controller._mesh_overlay = overlay

    if register_keybinds:
        register_mesh_overlay_keybinds(
            env,
            overlay,
            on_physics_backend_change=controller.cycle_physics_mesh_backend,
            on_ensure_physics_mesh=controller.ensure_physics_mesh,
            physics_mesher=controller.physics_mesher,
            temp_renderer=temp_renderer,
        )

    overlay.sync_from_controller(controller)
    return overlay
