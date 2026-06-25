"""Toggle and sync non-physics visual guides (coil cylinder, fixed-end box)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import genesis as gs

if TYPE_CHECKING:
    from agforge.environment import AgilityForgeEnv

COIL_VGEOM_NAME = "induction_coil_visual"
COIL_RGBA = (1.0, 0.5, 0.0, 0.4)


def _ctx(env: "AgilityForgeEnv"):
    vis = env.scene.visualizer
    if vis is None:
        return None
    return vis._context


def _viewer_lock(env: "AgilityForgeEnv"):
    vis = env.scene.visualizer
    if vis is None:
        from contextlib import nullcontext

        return nullcontext()
    return vis.viewer_lock


def _find_coil_vgeom(env: "AgilityForgeEnv"):
    robot = getattr(env, "robot", None)
    if robot is None:
        return None
    for link in robot.entity.links:
        for vgeom in link.vgeoms:
            if vgeom.metadata.get("name") == COIL_VGEOM_NAME:
                return vgeom
    base = robot.entity.get_link("base_plate")
    for vgeom in base.vgeoms:
        if vgeom.type == gs.GEOM_TYPE.CYLINDER:
            return vgeom
    return None


def _iter_entity_vgeoms(entity):
    if entity is None:
        return
    for link in entity.links:
        yield from link.vgeoms


def _vgeom_node(ctx, vgeom):
    if ctx is None or vgeom is None:
        return None
    return ctx.rigid_nodes.get(vgeom.uid)


def _set_vgeom_visible(env: "AgilityForgeEnv", vgeom, visible: bool) -> None:
    """Hide/show without removing scene nodes (avoids pyrender graph corruption)."""
    ctx = _ctx(env)
    if ctx is None or vgeom is None:
        return
    with _viewer_lock(env):
        node = _vgeom_node(ctx, vgeom)
        if node is not None and getattr(node, "mesh", None) is not None:
            node.mesh.is_visible = bool(visible)


def _set_entity_visible(env: "AgilityForgeEnv", entity, visible: bool) -> None:
    for vgeom in _iter_entity_vgeoms(entity):
        _set_vgeom_visible(env, vgeom, visible)


def _add_vgeom_node(env: "AgilityForgeEnv", vgeom, mesh=None) -> None:
    from genesis.ext import pyrender

    ctx = _ctx(env)
    if ctx is None or vgeom is None:
        return
    if vgeom.uid in ctx.rigid_nodes:
        return

    solver = env.scene.sim.rigid_solver
    geom_envs_idx = ctx._get_geom_active_envs_idx(vgeom, ctx.rendered_envs_idx)
    if len(geom_envs_idx) == 0:
        return

    if mesh is None:
        mesh = vgeom.get_trimesh()
    geom_T = solver._vgeoms_render_T[vgeom.idx][geom_envs_idx]
    with _viewer_lock(env):
        ctx.add_rigid_node(
            vgeom,
            pyrender.Mesh.from_trimesh(
                mesh=mesh,
                poses=geom_T,
                smooth=vgeom.surface.smooth,
                double_sided=vgeom.surface.double_sided,
                is_floor=False,
                env_shared=not ctx.env_separate_rigid,
            ),
        )


class VisualGuidesController:
    """Viewer-only controls for coil and fixed-end guide geometry."""

    def __init__(self, env: "AgilityForgeEnv"):
        self.env = env
        self._coil_vgeom = _find_coil_vgeom(env)
        self._fixed_region_entity = getattr(env, "_fixed_region_guide_entity", None)
        self._target_bounds_entity = getattr(env, "_target_bounds_entity", None)

        self.coil_visible = True
        self.fixed_region_visible = True
        self.target_bounds_visible = bool(self._target_bounds_entity is not None)

        self._coil_radius = float(env.cfg.robot.coil_radius)
        self._coil_length = float(env.cfg.robot.coil_length)

    def toggle_coil(self) -> bool:
        self.coil_visible = not self.coil_visible
        _set_vgeom_visible(self.env, self._coil_vgeom, self.coil_visible)
        return self.coil_visible

    def toggle_fixed_region(self) -> bool:
        self.fixed_region_visible = not self.fixed_region_visible
        _set_entity_visible(self.env, self._fixed_region_entity, self.fixed_region_visible)
        return self.fixed_region_visible

    def toggle_target_bounds(self) -> bool:
        if self._target_bounds_entity is None:
            return False
        self.target_bounds_visible = not self.target_bounds_visible
        _set_entity_visible(self.env, self._target_bounds_entity, self.target_bounds_visible)
        return self.target_bounds_visible

    def sync_coil_visual_if_needed(self) -> bool:
        """Refresh coil mesh radius/length from cfg (sim thread). Offset needs XML regen."""
        if not self.coil_visible or self._coil_vgeom is None:
            self._coil_radius = float(self.env.cfg.robot.coil_radius)
            self._coil_length = float(self.env.cfg.robot.coil_length)
            return False

        radius = float(self.env.cfg.robot.coil_radius)
        length = float(self.env.cfg.robot.coil_length)
        if abs(radius - self._coil_radius) < 1e-9 and abs(length - self._coil_length) < 1e-9:
            return False

        self._coil_radius = radius
        self._coil_length = length
        self._rebuild_coil_mesh(radius, length)
        return True

    def _rebuild_coil_mesh(self, radius: float, length: float) -> None:
        import genesis.utils.mesh as mu
        from genesis.ext import pyrender

        vgeom = self._coil_vgeom
        ctx = _ctx(self.env)
        if vgeom is None or ctx is None:
            return

        # Z-aligned mesh; MJCF euler (0 90 0) is applied via geom_T poses (same as initial vgeom).
        mesh = mu.create_cylinder(radius=radius, height=length, color=COIL_RGBA)

        solver = self.env.scene.sim.rigid_solver
        geom_envs_idx = ctx._get_geom_active_envs_idx(vgeom, ctx.rendered_envs_idx)
        if len(geom_envs_idx) == 0:
            return
        geom_T = solver._vgeoms_render_T[vgeom.idx][geom_envs_idx]

        with _viewer_lock(self.env):
            node = _vgeom_node(ctx, vgeom)
            pr_mesh = pyrender.Mesh.from_trimesh(
                mesh=mesh,
                poses=geom_T,
                smooth=vgeom.surface.smooth,
                double_sided=vgeom.surface.double_sided,
                is_floor=False,
                env_shared=not ctx.env_separate_rigid,
            )
            if node is not None and ctx._scene.has_node(node):
                node.mesh = pr_mesh
                node.mesh.is_visible = self.coil_visible
            else:
                ctx.add_rigid_node(vgeom, pr_mesh)

    def status_line(self) -> str:
        coil = "on" if self.coil_visible else "off"
        fixed = "on" if self.fixed_region_visible else "off"
        return f"Guides coil {coil}  fixed {fixed}  [K] coil  [U] fixed"


def register_visual_guide_keybinds(
    env: "AgilityForgeEnv",
    guides: VisualGuidesController | None,
) -> None:
    if guides is None:
        return
    vis = env.scene.visualizer
    if vis is None or vis.viewer is None:
        return

    from genesis.vis.keybindings import Key, KeyAction, Keybind

    viewer = vis.viewer

    def _refresh_hud():
        from agforge.vis.status_overlay import refresh_viewer_status_with_tuner

        refresh_viewer_status_with_tuner(env, tuner=None)

    def _log_coil():
        state = "on" if guides.toggle_coil() else "off"
        gs.logger.info(f"Induction coil visual: {state}")
        _refresh_hud()

    def _log_fixed():
        state = "on" if guides.toggle_fixed_region() else "off"
        gs.logger.info(f"Fixed-end guide box: {state}")
        _refresh_hud()

    viewer.register_keybinds(
        Keybind(
            "toggle_coil_visual",
            Key.K,
            key_action=KeyAction.PRESS,
            callback=_log_coil,
            allow_overload=True,
            show_in_help=False,
        ),
        Keybind(
            "toggle_fixed_region_guide",
            Key.U,
            key_action=KeyAction.PRESS,
            callback=_log_fixed,
            allow_overload=True,
            show_in_help=False,
        ),
        overwrite=False,
    )


def install_visual_guides(env: "AgilityForgeEnv", *, register_keybinds: bool = True) -> VisualGuidesController | None:
    if not getattr(env.cfg.general, "show_viewer", False):
        return None

    guides = VisualGuidesController(env)
    env._visual_guides = guides
    if register_keybinds:
        register_visual_guide_keybinds(env, guides)
    from agforge.vis.status_overlay import refresh_viewer_status_with_tuner

    refresh_viewer_status_with_tuner(env, tuner=getattr(env, "thermal_tuner", None))
    return guides
