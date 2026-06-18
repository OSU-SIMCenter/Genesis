"""Reconstructed mesh overlays in the Genesis viewer (visual vs physics meshes)."""

from __future__ import annotations

import contextlib
from enum import Enum
from typing import TYPE_CHECKING, Callable

import numpy as np
import trimesh

from agforge.profiling_util import teleop_profile

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
    return _display_mesh_from_source(mesh)


def _display_mesh_from_source(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Single viewer-oriented copy with flipped winding and warmed normals."""
    out = mesh.copy()
    if len(out.faces) > 0:
        out.faces = out.faces[:, ::-1]
        # Compute once per mesh version; pyrender smooth path copies these arrays.
        _ = out.vertex_normals
    return out


def _topology_key(mesh: trimesh.Trimesh) -> tuple[int, int]:
    return len(mesh.vertices), len(mesh.faces)


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
        env: "AgilityForgeEnv | None" = None,
        visual_color=(0.25, 0.75, 0.95, 1.0),
        physics_color=(0.95, 0.55, 0.15, 1.0),
        transparent_alpha: float = 0.35,
    ):
        from genesis.ext import pyrender

        self._ctx = ctx
        self._env = env
        self._pyrender = pyrender
        self._visual_color = np.array(visual_color, dtype=np.float32)
        self._physics_color = np.array(physics_color, dtype=np.float32)
        self._transparent_alpha = float(transparent_alpha)
        self.visual_mode = MeshDisplayMode.TRANSPARENT
        self.physics_mode = MeshDisplayMode.OFF
        self._visual_node = None
        self._physics_node = None
        self._visual_mesh: trimesh.Trimesh | None = None
        self._physics_mesh: trimesh.Trimesh | None = None
        self._visual_stamp: int = -1
        self._physics_stamp: int = -1
        self._visual_topology: tuple[int, int] | None = None
        self._physics_topology: tuple[int, int] | None = None

    @classmethod
    def from_env(cls, env: "AgilityForgeEnv") -> "ReconMeshOverlay":
        ctx = env.scene.visualizer._context
        return cls(ctx, env=env)

    def cycle_visual_mode(self) -> MeshDisplayMode:
        self.visual_mode = _cycle_display_mode(self.visual_mode)
        self._refresh_node("visual", allow_inplace=False)
        return self.visual_mode

    def cycle_physics_mode(self) -> MeshDisplayMode:
        self.physics_mode = _cycle_display_mode(self.physics_mode)
        self._refresh_node("physics", allow_inplace=False)
        return self.physics_mode

    def cycle_unified_display(self) -> MeshDisplayMode:
        """Cycle mesh overlay display for unified visual+physics mesh."""
        self.visual_mode = _cycle_display_mode(self.visual_mode)
        self.physics_mode = MeshDisplayMode.OFF
        self._remove_node("physics")
        self._refresh_node("visual", allow_inplace=False)
        return self.visual_mode

    def _bind_display_mesh_from_source(self, which: str, source: trimesh.Trimesh | None) -> None:
        """Cache viewer-oriented mesh; copy topology only when counts change."""
        mesh_attr = f"_{which}_mesh"
        topo_attr = f"_{which}_topology"
        if source is None or len(source.vertices) < 4:
            setattr(self, mesh_attr, None)
            setattr(self, topo_attr, None)
            return

        display: trimesh.Trimesh | None = getattr(self, mesh_attr)
        source_topo = _topology_key(source)
        if (
            display is not None
            and getattr(self, topo_attr) == source_topo
            and len(display.vertices) == len(source.vertices)
        ):
            with teleop_profile(self._env, "teleop_render_mesh_overlay_vertices"):
                display.vertices[:] = source.vertices
            return

        display_mesh = _display_mesh_from_source(source)
        setattr(self, mesh_attr, display_mesh)
        setattr(self, topo_attr, _topology_key(display_mesh))

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
            self._bind_display_mesh_from_source("visual", visual_mesh)

        if physics_stamp is not None and physics_stamp == self._physics_stamp:
            physics_changed = False
        else:
            physics_changed = True
            self._physics_stamp = -1 if physics_stamp is None else int(physics_stamp)
            self._bind_display_mesh_from_source("physics", physics_mesh)

        if visual_changed:
            self._refresh_node("visual", allow_inplace=True)
        if physics_changed:
            self._refresh_node("physics", allow_inplace=True)

    def _sync_unified_mesh(self, mesh: trimesh.Trimesh | None, stamp: int) -> None:
        """Single overlay in unified mode (one color, no visual/physics double-draw)."""
        self.physics_mode = MeshDisplayMode.OFF
        if stamp == self._visual_stamp:
            return
        self._visual_stamp = int(stamp)
        self._physics_stamp = int(stamp)
        self._bind_display_mesh_from_source("visual", mesh)
        self._physics_mesh = None
        self._physics_topology = None
        self._remove_node("physics")
        self._refresh_node("visual", allow_inplace=True)

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
        setattr(self, f"_{which}_topology", None)

    def _viewer_buffer_lock(self):
        """Serialize GPU buffer queue updates with the threaded viewer render pass."""
        if self._env is None:
            return contextlib.nullcontext()
        vis = self._env.scene.visualizer
        if vis is None or vis.viewer is None or not vis.viewer._is_built:
            return contextlib.nullcontext()
        pyv = vis.viewer._pyrender_viewer
        if pyv is None or not getattr(pyv, "run_in_thread", False):
            return contextlib.nullcontext()
        return vis.viewer.lock

    def _try_update_node_geometry(self, which: str, mesh: trimesh.Trimesh) -> bool:
        """Update GPU vertex/normal buffers when topology is unchanged."""
        node = getattr(self, f"_{which}_node")
        topology = getattr(self, f"_{which}_topology")
        if node is None or topology is None or topology != _topology_key(mesh):
            return False

        primitive = node.mesh.primitives[0]
        if len(mesh.vertices) != len(primitive.positions):
            return False

        scene = self._ctx._scene
        update_data = scene.reorder_vertices(node, mesh.vertices.astype(np.float32))
        with self._viewer_buffer_lock():
            self._ctx.jit.update_buffer(scene.get_buffer_id(node, "pos"), update_data)
            normal_data = self._ctx.jit.update_normal(node, update_data)
            if normal_data is not None:
                self._ctx.jit.update_buffer(scene.get_buffer_id(node, "normal"), normal_data)
        return True

    def _refresh_node(self, which: str, *, allow_inplace: bool = False) -> None:
        profile_target = self._env
        with teleop_profile(profile_target, "teleop_render_mesh_overlay_refresh"):
            mode: MeshDisplayMode = getattr(self, f"{which}_mode")
            mesh: trimesh.Trimesh | None = getattr(self, f"_{which}_mesh")
            color = self._visual_color if which == "visual" else self._physics_color

            if mode == MeshDisplayMode.OFF or mesh is None or len(mesh.vertices) < 4:
                self._remove_node(which)
                return

            if allow_inplace and self._try_update_node_geometry(which, mesh):
                return

            self._remove_node(which)
            material = self._material_for(color, mode)
            pr_mesh = self._pyrender.Mesh.from_trimesh(mesh, material=material, smooth=True)
            node = self._ctx.add_node(pr_mesh)
            setattr(self, f"_{which}_node", node)
            setattr(self, f"_{which}_topology", _topology_key(mesh))


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
                temp_renderer.prepare_render_frame(env)
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
