"""Temperature-colored MPM particle rendering for the Genesis viewer.

Buckets particles into a small number of inferno-colored sphere instances so
`jit.update_buffer` can be called from inside `RasterizerContext.update()`.
Used by live teleop and episode replay.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from agforge.profiling_util import teleop_profile

if TYPE_CHECKING:
    from agforge.environment import AgilityForgeEnv

N_BUCKETS = 8
OFF_SCREEN = np.array([0.0, 0.0, -0.05])

PARTICLE_COLOR_MODES = ("temperature", "induction_depth", "skin_weight", "q_ind")
PARTICLE_COLOR_MODE_LABELS = {
    "temperature": "Temperature",
    "induction_depth": "SDF depth",
    # ASCII only — the viewer font lacks Greek/special glyphs (e.g. delta crashes on_draw).
    "skin_weight": "Skin weight",
    "q_ind": "Heating rate q",
}


def _bucket_colors(cmap_name: str = "inferno", n_buckets: int = N_BUCKETS):
    import matplotlib as mpl

    cmap = mpl.colormaps.get_cmap(cmap_name)
    colors = []
    for i in range(n_buckets):
        t = 0.15 + (i / max(n_buckets - 1, 1)) * 0.85
        colors.append(tuple(cmap(t)))
    return colors


def _scalars_to_bucket_indices(
    values: np.ndarray, vmin: float, vmax: float, n_buckets: int = N_BUCKETS
):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    values = np.nan_to_num(values, nan=float(vmin), posinf=float(vmax), neginf=float(vmin))
    span = max(float(vmax) - float(vmin), 1e-12)
    tn = np.clip((values - vmin) / span, 0.0, 1.0)
    return np.minimum((tn * n_buckets).astype(np.int32), n_buckets - 1)


def _temps_to_bucket_indices(temps: np.ndarray, temp_min: float, temp_max: float, n_buckets: int = N_BUCKETS):
    return _scalars_to_bucket_indices(temps, temp_min, temp_max, n_buckets)


class TemperatureParticleRenderer:
    """Replaces default single-color MPM spheres with temperature-bucketed spheres."""

    def __init__(
        self,
        ctx,
        mpm_entity,
        particle_radius: float,
        n_particles: int,
        *,
        temp_min: float = 293.0,
        temp_max: float = 1450.0,
        bucket_max_sizes: np.ndarray | None = None,
        cmap_name: str = "inferno",
    ):
        from genesis.ext import pyrender
        import genesis.utils.mesh as mu

        self._ctx = ctx
        self._mpm_entity = mpm_entity
        self._physics_radius = float(particle_radius)
        self._render_scale = 1.0
        self._n_particles = n_particles
        self._temp_min = temp_min
        self._temp_max = temp_max
        self._n_buckets = N_BUCKETS
        self._colors = _bucket_colors(cmap_name, N_BUCKETS)

        if bucket_max_sizes is None:
            # Live teleop: any single bucket may hold ALL particles when temperature is
            # uniform (e.g. room-temp billet → bucket 0 only). Start with a modest
            # allocation; _write_bucket_poses grows buckets on demand.
            base = max(256, int(np.ceil(n_particles / N_BUCKETS)))
            bucket_max_sizes = np.full(N_BUCKETS, base, dtype=np.int32)
        else:
            bucket_max_sizes = np.asarray(bucket_max_sizes, dtype=np.int32)

        self._bucket_max = bucket_max_sizes.tolist()
        self._bucket_nodes: list = []
        self._remove_default_nodes()

        for b in range(N_BUCKETS):
            n_slots = int(self._bucket_max[b])
            if n_slots <= 0:
                self._bucket_nodes.append(None)
                continue
            mesh = mu.create_sphere(self._draw_radius(), subdivisions=1, color=self._colors[b])
            tfs = np.tile(np.eye(4), (n_slots, 1, 1))
            tfs[:, :3, 3] = OFF_SCREEN
            pr_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=True, poses=tfs)
            self._bucket_nodes.append(self._ctx.add_node(pr_mesh))

        self._frame_pos: np.ndarray | None = None
        self._bucket_idx: np.ndarray | None = None
        self._env = None
        self._original_update_mpm = ctx.update_mpm
        ctx.update_mpm = self._update_mpm_hook

    @classmethod
    def from_env(
        cls,
        env: "AgilityForgeEnv",
        *,
        temp_min: float = 293.0,
        temp_max: float = 1450.0,
        bucket_max_sizes: np.ndarray | None = None,
    ) -> "TemperatureParticleRenderer":
        ctx = env.scene.visualizer._context
        mpm_entity = env.mpm_entity
        radius = env.scene.sim.mpm_solver.particle_radius
        return cls(
            ctx,
            mpm_entity,
            radius,
            mpm_entity.n_particles,
            temp_min=temp_min,
            temp_max=temp_max,
            bucket_max_sizes=bucket_max_sizes,
        )

    def _draw_radius(self) -> float:
        return self._physics_radius * self._render_scale

    def render_scale_label(self) -> str:
        return "normal" if self._render_scale >= 0.99 else "small"

    def cycle_render_scale(self, env: "AgilityForgeEnv") -> float:
        """Toggle viewer sphere radius between full size and a smaller debug scale."""
        small = float(getattr(env.cfg.general, "particle_render_scale_small", 0.3))
        if self._render_scale >= 0.99:
            self._render_scale = small
        else:
            self._render_scale = 1.0
        self._rebuild_all_bucket_meshes()
        if self._frame_pos is not None and self._bucket_idx is not None:
            self._write_bucket_poses(self._frame_pos, self._bucket_idx, env=self._env)
        return self._render_scale

    def _rebuild_all_bucket_meshes(self):
        for b in range(self._n_buckets):
            n_slots = int(self._bucket_max[b])
            if n_slots > 0:
                self._recreate_bucket_node(b, n_slots)

    def _remove_default_nodes(self):
        for idx in self._ctx.rendered_envs_idx:
            key = (idx, self._mpm_entity.uid)
            if key in self._ctx.static_nodes:
                old_node = self._ctx.static_nodes.pop(key)
                self._ctx.remove_node(old_node)

    def _update_mpm_hook(self):
        if self._frame_pos is None or self._bucket_idx is None:
            return
        self._write_bucket_poses(self._frame_pos, self._bucket_idx, env=self._env)

    def _recreate_bucket_node(self, b: int, n_slots: int):
        from genesis.ext import pyrender
        import genesis.utils.mesh as mu

        old = self._bucket_nodes[b]
        if old is not None:
            self._ctx.remove_node(old)
        mesh = mu.create_sphere(self._draw_radius(), subdivisions=1, color=self._colors[b])
        tfs = np.tile(np.eye(4), (n_slots, 1, 1))
        tfs[:, :3, 3] = OFF_SCREEN
        pr_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=True, poses=tfs)
        self._bucket_nodes[b] = self._ctx.add_node(pr_mesh)
        self._bucket_max[b] = n_slots

    def _write_bucket_poses(self, positions: np.ndarray, bucket_idx: np.ndarray, env=None):
        positions = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
        bucket_idx = np.asarray(bucket_idx, dtype=np.int32).reshape(-1)
        n = min(positions.shape[0], bucket_idx.shape[0])

        counts = np.bincount(bucket_idx[:n], minlength=self._n_buckets)
        for b in range(self._n_buckets):
            needed = int(counts[b])
            if needed > self._bucket_max[b]:
                self._recreate_bucket_node(b, needed)

        with teleop_profile(env, "teleop_render_particle_bucket_write"):
            for b in range(self._n_buckets):
                node = self._bucket_nodes[b]
                if node is None:
                    continue
                n_slots = self._bucket_max[b]
                tfs = np.tile(np.eye(4), (n_slots, 1, 1))
                tfs[:, :3, 3] = OFF_SCREEN
                mask = bucket_idx[:n] == b
                count = int(mask.sum())
                if count > 0:
                    tfs[:count, :3, 3] = positions[:n][mask]
                for prim in node.mesh.primitives:
                    prim.poses = tfs
                buf_id = self._ctx._scene.get_buffer_id(node, "model")
                if buf_id != -1:
                    self._ctx.jit.update_buffer(buf_id, tfs.transpose((0, 2, 1)))

    def set_frame(
        self,
        positions: np.ndarray,
        scalars: np.ndarray | None = None,
        temps: np.ndarray | None = None,
        bucket_idx: np.ndarray | None = None,
        *,
        vmin: float | None = None,
        vmax: float | None = None,
    ):
        """Queue positions and scalar values for the next viewer update."""
        self._frame_pos = np.asarray(positions, dtype=np.float64)
        if bucket_idx is not None:
            self._bucket_idx = np.asarray(bucket_idx, dtype=np.int32)
        else:
            values = scalars if scalars is not None else temps
            if values is None:
                self._bucket_idx = np.full(self._frame_pos.shape[0], self._n_buckets // 2, dtype=np.int32)
            else:
                lo = self._temp_min if vmin is None else vmin
                hi = self._temp_max if vmax is None else vmax
                self._bucket_idx = _scalars_to_bucket_indices(values, lo, hi, self._n_buckets)

    def _scalar_field_from_env(self, env: "AgilityForgeEnv", pos: np.ndarray) -> tuple[np.ndarray, float, float]:
        from agforge.thermal_field import particle_q_ind, skin_weight

        cfg = env.cfg
        mode = getattr(cfg.general, "particle_color_mode", "temperature")
        solver = env.scene.sim.mpm_solver

        if mode == "temperature":
            temps = env.mpm_entity.get_particles_temp().detach().cpu().numpy().reshape(-1)
            return temps, cfg.general.particle_temp_min, cfg.general.particle_temp_max

        depths = env.mpm_entity.get_particles_induction_depth().detach().cpu().numpy().reshape(-1)
        skin_depth = float(cfg.skin_depth)

        if mode == "induction_depth":
            d_max = cfg.general.particle_depth_max
            if d_max is None:
                d_max = 3.0 * skin_depth
            return depths, 0.0, float(d_max)

        if mode == "skin_weight":
            weights = skin_weight(depths, skin_depth)
            return weights, 0.0, 1.0

        if mode == "q_ind":
            center, half_length, radius, q_peak, sd, active = solver.get_induction_uniforms_numpy(0)
            uniforms_unset = half_length <= 0.0 or radius <= 0.0 or q_peak <= 0.0
            if uniforms_unset:
                # GPU uniforms are zero until thermal is enabled / set_induction_params runs.
                half_length = cfg.robot.coil_length / 2.0
                radius = cfg.robot.coil_radius
                q_peak = cfg.heating_power
                center = np.array(
                    [float(cfg.robot.cylinder_pos[0]), 0.0, float(cfg.robot.cylinder_pos[2])],
                    dtype=np.float64,
                )
            if sd <= 0.0:
                sd = skin_depth
            q = particle_q_ind(
                pos,
                depths,
                coil_center_x=float(center[0]),
                half_length=half_length,
                radius=radius,
                q_peak=q_peak,
                skin_depth=sd,
                thermal_time_scale=float(cfg.mpm.thermal_time_scale),
            )
            if not uniforms_unset and not active:
                q = np.zeros_like(q)
            q = np.nan_to_num(q, nan=0.0, posinf=0.0, neginf=0.0)
            vmax = float(np.percentile(q, 99.5)) if q.size else 1.0
            return q, 0.0, max(vmax, 1e-12)

        temps = env.mpm_entity.get_particles_temp().detach().cpu().numpy().reshape(-1)
        return temps, cfg.general.particle_temp_min, cfg.general.particle_temp_max

    def sync_from_env(self, env: "AgilityForgeEnv"):
        """Read live particle positions and the configured scalar field from the simulation."""
        self._env = env
        with teleop_profile(env, "teleop_render_particle_gpu_pull"):
            pos = env.mpm_entity.get_particles_pos().detach().cpu().numpy().reshape(-1, 3)
            scalars, vmin, vmax = self._scalar_field_from_env(env, pos)
            if hasattr(env.mpm_entity, "get_particles_active"):
                active = env.mpm_entity.get_particles_active(envs_idx=0).detach().cpu().numpy().reshape(-1).astype(bool)
                if active.shape[0] == pos.shape[0]:
                    pos = pos[active]
                    scalars = scalars[active]
        self.set_frame(pos, scalars=scalars, vmin=vmin, vmax=vmax)

    @staticmethod
    def max_bucket_sizes_from_frames(
        temp_values: np.ndarray,
        offsets: np.ndarray,
        n_frames: int,
        temp_min: float,
        temp_max: float,
        n_buckets: int = N_BUCKETS,
    ) -> np.ndarray:
        """Pre-compute per-bucket capacity for replay (minimizes GPU instances)."""
        max_sizes = np.zeros(n_buckets, dtype=np.int32)
        for fi in range(n_frames):
            s, e = offsets[fi], offsets[fi + 1]
            if s >= e:
                continue
            bi = _temps_to_bucket_indices(temp_values[s:e], temp_min, temp_max, n_buckets)
            for b in range(n_buckets):
                count = int((bi == b).sum())
                if count > max_sizes[b]:
                    max_sizes[b] = count
        return max_sizes

    def detach(self):
        """Restore the original MPM update hook."""
        if self._original_update_mpm is not None:
            self._ctx.update_mpm = self._original_update_mpm
            self._original_update_mpm = None


def _pyrender_viewer(env: "AgilityForgeEnv"):
    vis = env.scene.visualizer
    if vis is None or vis.viewer is None:
        return None
    return vis.viewer._pyrender_viewer


def update_particle_color_display(
    env: "AgilityForgeEnv",
    *,
    flash: bool = False,
    renderer: TemperatureParticleRenderer | None = None,
    mesh_overlay=None,
    physics_mesher=None,
) -> None:
    """Update the minimal top-right status overlay (color mode + SDF mesh backend)."""
    from agforge.vis.status_overlay import update_viewer_status

    update_viewer_status(env, physics_mesher=physics_mesher)


def cycle_particle_color_mode(env: "AgilityForgeEnv", renderer: TemperatureParticleRenderer, step: int = 1) -> str:
    """Advance the live viewer scalar field and refresh particle colors."""
    modes = PARTICLE_COLOR_MODES
    current = getattr(env.cfg.general, "particle_color_mode", "temperature")
    try:
        idx = modes.index(current)
    except ValueError:
        idx = 0
    mode = modes[(idx + step) % len(modes)]
    env.cfg.general.particle_color_mode = mode
    renderer.sync_from_env(env)
    return mode


def register_particle_color_keybinds(
    env: "AgilityForgeEnv",
    renderer: TemperatureParticleRenderer | None,
    *,
    mesh_overlay=None,
    physics_mesher=None,
) -> None:
    """Bind G / Shift+G in the Genesis viewer to cycle particle color modes."""
    if renderer is None:
        return
    vis = env.scene.visualizer
    if vis is None or vis.viewer is None:
        return

    import genesis as gs
    from genesis.vis.keybindings import Key, KeyAction, KeyMod, Keybind

    viewer = vis.viewer

    def _show_mode(mode: str):
        label = PARTICLE_COLOR_MODE_LABELS.get(mode, mode)
        update_particle_color_display(env, physics_mesher=physics_mesher)
        gs.logger.info(f"Particle color mode: {label} ({mode})")

    def _refresh_viewer():
        if vis is not None:
            vis.update(force=False, auto=True)

    def _cycle_next():
        _show_mode(cycle_particle_color_mode(env, renderer, step=1))
        _refresh_viewer()

    def _cycle_prev():
        _show_mode(cycle_particle_color_mode(env, renderer, step=-1))
        _refresh_viewer()

    def _toggle_size():
        scale = renderer.cycle_render_scale(env)
        label = renderer.render_scale_label()
        gs.logger.info(f"Particle render size: {label} (scale={scale:.2f})")
        _refresh_viewer()

    viewer.register_keybinds(
        Keybind(
            "cycle_color_mode",
            Key.G,
            key_action=KeyAction.PRESS,
            callback=_cycle_next,
            allow_overload=True,
        ),
        Keybind(
            "prev_color_mode",
            Key.G,
            key_mods=(KeyMod.SHIFT,),
            key_action=KeyAction.PRESS,
            callback=_cycle_prev,
            allow_overload=True,
        ),
        Keybind(
            "toggle_particle_size",
            Key.B,
            key_action=KeyAction.PRESS,
            callback=_toggle_size,
            allow_overload=True,
        ),
        overwrite=False,
    )
    from agforge.vis.status_overlay import _refresh_keybind_help

    update_particle_color_display(env, physics_mesher=physics_mesher)
    _refresh_keybind_help(env)


def install_temperature_particle_renderer(
    env: "AgilityForgeEnv",
    *,
    temp_min: float = 293.0,
    temp_max: float = 1450.0,
    register_keybinds: bool = True,
) -> TemperatureParticleRenderer | None:
    """Enable temperature coloring when the Genesis viewer is active."""
    if not getattr(env.cfg.general, "show_viewer", False):
        return None
    if not getattr(env.cfg.general, "visualize_particle_temperature", True):
        return None
    if env.scene.visualizer is None:
        return None
    renderer = TemperatureParticleRenderer.from_env(env, temp_min=temp_min, temp_max=temp_max)
    if register_keybinds:
        register_particle_color_keybinds(env, renderer)
    return renderer
