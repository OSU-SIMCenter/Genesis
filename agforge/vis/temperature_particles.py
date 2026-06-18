"""Temperature-colored MPM particle rendering for the Genesis viewer.

Buckets particles into a small number of inferno-colored sphere instances so
`jit.update_buffer` can be called from inside `RasterizerContext.update()`.
Used by live teleop and episode replay.

Live teleop (all color modes including q_ind) reads from GPU-resident
``particles_render`` after ``update_render_fields`` — no per-frame particle
bundle pull. Episode replay still uses the CPU ``set_frame`` path.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

import genesis as gs

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


def _scalars_to_bucket_indices_torch(
    values: torch.Tensor, vmin: float, vmax: float, n_buckets: int = N_BUCKETS
) -> torch.Tensor:
    t = values.reshape(-1).float()
    t = torch.nan_to_num(t, nan=float(vmin), posinf=float(vmax), neginf=float(vmin))
    span = max(float(vmax) - float(vmin), 1e-12)
    tn = torch.clamp((t - vmin) / span, 0.0, 1.0)
    return torch.clamp((tn * n_buckets).long(), max=n_buckets - 1)


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
            mesh = mu.create_sphere(
                self._draw_radius(),
                subdivisions=self._sphere_subdivisions(),
                color=self._colors[b],
            )
            tfs = np.tile(np.eye(4), (n_slots, 1, 1))
            tfs[:, :3, 3] = OFF_SCREEN
            pr_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=True, poses=tfs)
            self._bucket_nodes.append(self._ctx.add_node(pr_mesh))
            self._bucket_pose_buffers[b] = tfs

        self._frame_pos: np.ndarray | None = None
        self._bucket_idx: np.ndarray | None = None
        self._env = None
        self._gpu_scalar_vmin: float | None = None
        self._gpu_scalar_vmax: float | None = None
        self._gpu_render_path_ok: bool | None = None
        self._sync_path_logged = False
        self._original_update_mpm = ctx.update_mpm
        self._bucket_pose_buffers: list[np.ndarray | None] = [None] * N_BUCKETS
        self._q_ind_vmax_cached: float | None = None
        self._q_ind_vmax_frame: int = 0
        self._render_hook_frame: int = 0
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

    def _sphere_subdivisions(self, env: "AgilityForgeEnv | None" = None) -> int:
        if env is not None:
            perf = getattr(env.cfg, "performance", None)
            if perf is not None:
                return max(0, int(getattr(perf, "particle_sphere_subdivisions", 1)))
        return 1

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
        self._rebuild_all_bucket_meshes(env)
        if self._frame_pos is not None and self._bucket_idx is not None:
            self._write_bucket_poses(self._frame_pos, self._bucket_idx, env=self._env)
        elif self._can_use_gpu_render_path(env):
            self._write_buckets_from_particles_render(env)
        return self._render_scale

    def _rebuild_all_bucket_meshes(self, env: "AgilityForgeEnv | None" = None):
        for b in range(self._n_buckets):
            n_slots = int(self._bucket_max[b])
            if n_slots > 0:
                self._recreate_bucket_node(b, n_slots, env=env)

    def _remove_default_nodes(self):
        for idx in self._ctx.rendered_envs_idx:
            key = (idx, self._mpm_entity.uid)
            if key in self._ctx.static_nodes:
                old_node = self._ctx.static_nodes.pop(key)
                self._ctx.remove_node(old_node)

    def _update_mpm_hook(self):
        if self._env is not None and self._can_use_gpu_render_path(self._env):
            self._write_buckets_from_particles_render(self._env)
            return
        if self._frame_pos is None or self._bucket_idx is None:
            return
        self._write_bucket_poses(self._frame_pos, self._bucket_idx, env=self._env)

    @staticmethod
    def _particles_render_has_thermal_scalars(solver) -> bool:
        pr = getattr(solver, "particles_render", None)
        if pr is None:
            return False
        try:
            pr.temp
            pr.depth
        except (AttributeError, KeyError, TypeError):
            return False
        return True

    def _is_simple_color(self, env: "AgilityForgeEnv") -> bool:
        return bool(getattr(env.cfg.general, "particle_simple_color", False))

    def _can_use_gpu_render_path(self, env: "AgilityForgeEnv") -> bool:
        """True when scalar coloring can use GPU-resident particles_render."""
        if self._gpu_render_path_ok is not None:
            return self._gpu_render_path_ok

        if self._is_simple_color(env):
            solver = env.scene.sim.mpm_solver
            if getattr(solver, "particles_render", None) is not None:
                self._log_gpu_render_path_once(env, True, mode="simple")
                return True

        solver = env.scene.sim.mpm_solver
        if not getattr(solver, "_enable_thermal", False):
            self._log_gpu_render_path_once(env, False, "solver._enable_thermal is False")
            return False
        if not self._particles_render_has_thermal_scalars(solver):
            self._log_gpu_render_path_once(env, False, "particles_render missing temp/depth")
            return False

        mode = getattr(env.cfg.general, "particle_color_mode", "temperature")
        self._log_gpu_render_path_once(env, True, mode=mode)
        return True

    def _log_gpu_render_path_once(
        self,
        env: "AgilityForgeEnv",
        ok: bool,
        reason: str = "",
        *,
        mode: str | None = None,
    ) -> None:
        if self._gpu_render_path_ok is not None:
            return
        self._gpu_render_path_ok = ok
        if ok:
            mode_label = mode or getattr(env.cfg.general, "particle_color_mode", "temperature")
            gs.logger.info(
                f"Particle render path: GPU-resident (mode={mode_label}, particles_render.temp/depth)"
            )
        else:
            gs.logger.warning(
                f"Particle render path: CPU bundle fallback ({reason}). "
                "If you just added particles_render.temp/depth, restart after Genesis recompiles kernels."
            )

    def _scalar_range_for_mode(self, env: "AgilityForgeEnv", mode: str) -> tuple[float, float]:
        cfg = env.cfg
        if mode == "temperature":
            return cfg.general.particle_temp_min, cfg.general.particle_temp_max
        if mode == "induction_depth":
            d_max = cfg.general.particle_depth_max
            if d_max is None:
                d_max = 3.0 * float(cfg.skin_depth)
            return 0.0, float(d_max)
        if mode == "skin_weight":
            return 0.0, 1.0
        if mode == "q_ind":
            return 0.0, 1.0
        return self._temp_min, self._temp_max

    def _entity_render_slice(
        self, env: "AgilityForgeEnv", *, need_scalars: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        solver = env.scene.sim.mpm_solver
        ps = self._mpm_entity.particle_start
        pe = ps + self._mpm_entity.n_particles
        entity = env.mpm_entity

        if hasattr(entity, "get_particles_render_bundle"):
            pos_t, active_t, temps_t, depths_t = entity.get_particles_render_bundle()
            pos = pos_t.reshape(-1, 3)[ps:pe]
            active = active_t.reshape(-1)[ps:pe].bool()
            temps = temps_t.reshape(-1)[ps:pe] if need_scalars else None
            depths = depths_t.reshape(-1)[ps:pe] if need_scalars else None
        else:
            pr = solver.particles_render
            pos = pr.pos.to_torch().reshape(-1, 3)[ps:pe]
            active = pr.active.to_torch().reshape(-1)[ps:pe].bool()
            temps = pr.temp.to_torch().reshape(-1)[ps:pe] if need_scalars else None
            depths = pr.depth.to_torch().reshape(-1)[ps:pe] if need_scalars else None

        offset = getattr(env.scene, "envs_offset", None)
        if offset is not None:
            env_offset = torch.as_tensor(offset[0], device=pos.device, dtype=pos.dtype)
            pos = pos + env_offset

        return pos, active, temps, depths

    def _subsample_particles(
        self,
        env: "AgilityForgeEnv",
        pos: torch.Tensor,
        scalars: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        frac = float(getattr(env.cfg.general, "particle_render_fraction", 1.0))
        if frac >= 1.0 or pos.shape[0] <= 1:
            return pos, scalars
        frac = max(0.05, min(1.0, frac))
        stride = max(1, int(round(1.0 / frac)))
        pos = pos[::stride]
        if scalars is not None:
            scalars = scalars[::stride]
        return pos, scalars

    def _gpu_scalar_tensor(
        self,
        env: "AgilityForgeEnv",
        pos: torch.Tensor,
        temps: torch.Tensor,
        depths: torch.Tensor,
    ) -> tuple[torch.Tensor, float, float]:
        mode = getattr(env.cfg.general, "particle_color_mode", "temperature")
        vmin, vmax = self._scalar_range_for_mode(env, mode)

        if mode == "temperature":
            return temps, vmin, vmax

        if mode == "induction_depth":
            return depths, vmin, vmax

        if mode == "skin_weight":
            skin_depth = float(env.cfg.skin_depth)
            if skin_depth <= 0.0:
                weights = torch.zeros_like(depths)
            else:
                weights = torch.exp(-2.0 * depths / skin_depth)
            return weights, vmin, vmax

        if mode == "q_ind":
            from agforge.thermal_field import particle_q_ind_torch

            cfg = env.cfg
            solver = env.scene.sim.mpm_solver
            center, half_length, radius, q_peak, sd, active = solver.get_induction_uniforms_numpy(0)
            uniforms_unset = half_length <= 0.0 or radius <= 0.0 or q_peak <= 0.0
            if uniforms_unset:
                half_length = cfg.robot.coil_length / 2.0
                radius = cfg.robot.coil_radius
                q_peak = cfg.heating_power
                center_x = float(cfg.robot.cylinder_pos[0])
            else:
                center_x = float(center[0])
            if sd <= 0.0:
                sd = float(cfg.skin_depth)
            q = particle_q_ind_torch(
                pos,
                depths,
                coil_center_x=center_x,
                half_length=half_length,
                radius=radius,
                q_peak=q_peak,
                skin_depth=sd,
                thermal_time_scale=float(cfg.mpm.thermal_time_scale),
            )
            if not uniforms_unset and not active:
                q = torch.zeros_like(q)
            q = torch.nan_to_num(q, nan=0.0, posinf=0.0, neginf=0.0)
            if q.numel() == 0:
                return q, 0.0, 1.0
            perf = getattr(env.cfg, "performance", None)
            interval = max(1, int(getattr(perf, "q_ind_vmax_refresh_interval", 15) if perf else 15))
            self._render_hook_frame += 1
            if (
                self._q_ind_vmax_cached is not None
                and (self._render_hook_frame - self._q_ind_vmax_frame) < interval
            ):
                vmax = self._q_ind_vmax_cached
            else:
                vmax = float(torch.quantile(q, 0.995).item())
                self._q_ind_vmax_cached = vmax
                self._q_ind_vmax_frame = self._render_hook_frame
            return q, 0.0, max(vmax, 1e-12)

        return temps, vmin, vmax

    def _write_buckets_from_particles_render(self, env: "AgilityForgeEnv"):
        """Bucket and upload poses from GPU-resident particles_render (no particle bundle pull)."""
        with teleop_profile(env, "teleop_render_particle_gpu_resident"):
            simple = self._is_simple_color(env)
            pos, active, temps, depths = self._entity_render_slice(
                env, need_scalars=not simple
            )

            if active.shape[0] == pos.shape[0]:
                pos = pos[active]
                if not simple and temps is not None:
                    temps = temps[active]
                    depths = depths[active] if depths is not None else None

            if simple:
                pos, _ = self._subsample_particles(env, pos, None)
                bucket_idx = torch.full(
                    (pos.shape[0],),
                    self._n_buckets // 2,
                    device=pos.device,
                    dtype=torch.long,
                )
            else:
                scalars, vmin, vmax = self._gpu_scalar_tensor(env, pos, temps, depths)
                if self._gpu_scalar_vmin is not None:
                    vmin = self._gpu_scalar_vmin
                if self._gpu_scalar_vmax is not None:
                    vmax = self._gpu_scalar_vmax
                pos, scalars = self._subsample_particles(env, pos, scalars)
                bucket_idx = _scalars_to_bucket_indices_torch(scalars, vmin, vmax, self._n_buckets)

        self._write_bucket_poses_torch(pos, bucket_idx, env=env)

    def _write_bucket_poses_torch(self, positions: torch.Tensor, bucket_idx: torch.Tensor, env=None):
        """Upload bucket poses with GPU sort + one GPU→CPU sync."""
        positions = positions.reshape(-1, 3)
        bucket_idx = bucket_idx.reshape(-1)
        n = min(positions.shape[0], bucket_idx.shape[0])
        if n == 0:
            return

        order = torch.argsort(bucket_idx[:n])
        sorted_pos = positions[:n][order]
        sorted_buckets = bucket_idx[:n][order]
        counts = torch.bincount(sorted_buckets, minlength=self._n_buckets)

        pos_np = sorted_pos.detach().cpu().numpy()
        counts_np = counts.detach().cpu().numpy().astype(np.int32)
        self._write_bucket_poses_sorted(pos_np, counts_np, env=env)

    def _write_bucket_poses_sorted(
        self, positions: np.ndarray, bucket_counts: np.ndarray, env=None
    ):
        """Fill preallocated pose buffers from GPU-sorted positions and per-bucket counts."""
        positions = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
        bucket_counts = np.asarray(bucket_counts, dtype=np.int32).reshape(-1)
        if bucket_counts.shape[0] != self._n_buckets:
            bucket_counts = np.pad(
                bucket_counts,
                (0, max(0, self._n_buckets - bucket_counts.shape[0])),
            )[: self._n_buckets]

        for b in range(self._n_buckets):
            needed = int(bucket_counts[b])
            if needed > self._bucket_max[b]:
                self._recreate_bucket_node(b, needed, env=env)

        with teleop_profile(env, "teleop_render_particle_bucket_write"):
            offset = 0
            for b in range(self._n_buckets):
                node = self._bucket_nodes[b]
                if node is None:
                    offset += int(bucket_counts[b])
                    continue
                n_slots = self._bucket_max[b]
                count = int(bucket_counts[b])
                tfs = self._bucket_pose_buffers[b]
                if tfs is None or tfs.shape[0] != n_slots:
                    tfs = np.tile(np.eye(4), (n_slots, 1, 1))
                    self._bucket_pose_buffers[b] = tfs
                tfs[:, :3, 3] = OFF_SCREEN
                if count > 0:
                    tfs[:count, :3, 3] = positions[offset : offset + count]
                offset += count
                for prim in node.mesh.primitives:
                    prim.poses = tfs
                buf_id = self._ctx._scene.get_buffer_id(node, "model")
                if buf_id != -1:
                    self._ctx.jit.update_buffer(buf_id, tfs.transpose((0, 2, 1)))

    def _recreate_bucket_node(self, b: int, n_slots: int, env=None):
        from genesis.ext import pyrender
        import genesis.utils.mesh as mu

        old = self._bucket_nodes[b]
        if old is not None:
            self._ctx.remove_node(old)
        mesh = mu.create_sphere(
            self._draw_radius(),
            subdivisions=self._sphere_subdivisions(env),
            color=self._colors[b],
        )
        tfs = np.tile(np.eye(4), (n_slots, 1, 1))
        tfs[:, :3, 3] = OFF_SCREEN
        pr_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=True, poses=tfs)
        self._bucket_nodes[b] = self._ctx.add_node(pr_mesh)
        self._bucket_max[b] = n_slots
        self._bucket_pose_buffers[b] = tfs

    def _write_bucket_poses(self, positions: np.ndarray, bucket_idx: np.ndarray, env=None):
        positions = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
        bucket_idx = np.asarray(bucket_idx, dtype=np.int32).reshape(-1)
        n = min(positions.shape[0], bucket_idx.shape[0])

        counts = np.bincount(bucket_idx[:n], minlength=self._n_buckets)
        for b in range(self._n_buckets):
            needed = int(counts[b])
            if needed > self._bucket_max[b]:
                self._recreate_bucket_node(b, needed, env=env)

        with teleop_profile(env, "teleop_render_particle_bucket_write"):
            for b in range(self._n_buckets):
                node = self._bucket_nodes[b]
                if node is None:
                    continue
                n_slots = self._bucket_max[b]
                tfs = self._bucket_pose_buffers[b]
                if tfs is None or tfs.shape[0] != n_slots:
                    tfs = np.tile(np.eye(4), (n_slots, 1, 1))
                    self._bucket_pose_buffers[b] = tfs
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
        self._gpu_scalar_vmin = vmin
        self._gpu_scalar_vmax = vmax

    def _scalar_field_from_env(
        self,
        env: "AgilityForgeEnv",
        pos: np.ndarray,
        *,
        temps: np.ndarray | None = None,
        depths: np.ndarray | None = None,
    ) -> tuple[np.ndarray, float, float]:
        from agforge.thermal_field import particle_q_ind, skin_weight

        cfg = env.cfg
        mode = getattr(cfg.general, "particle_color_mode", "temperature")
        solver = env.scene.sim.mpm_solver

        if mode == "temperature":
            if temps is None:
                temps = env.mpm_entity.get_particles_temp().detach().cpu().numpy().reshape(-1)
            return temps, cfg.general.particle_temp_min, cfg.general.particle_temp_max

        if depths is None:
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

        if temps is None:
            temps = env.mpm_entity.get_particles_temp().detach().cpu().numpy().reshape(-1)
        return temps, cfg.general.particle_temp_min, cfg.general.particle_temp_max

    @staticmethod
    def _pull_render_bundle_to_numpy(entity) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """One kernel launch + one GPU->CPU transfer for viewer particle sync."""
        pos_t, active_t, temps_t, depths_t = entity.get_particles_render_bundle()
        pos_t = pos_t.reshape(-1, 3)
        n = pos_t.shape[0]
        bundle = torch.cat(
            (
                pos_t,
                temps_t.reshape(n, 1),
                depths_t.reshape(n, 1),
                active_t.reshape(n, 1).to(dtype=torch.float32),
            ),
            dim=1,
        )
        arr = bundle.detach().cpu().numpy()
        pos = arr[:, :3]
        temps = arr[:, 3]
        depths = arr[:, 4]
        active = arr[:, 5].astype(bool)
        return pos, active, temps, depths

    def prepare_render_frame(self, env: "AgilityForgeEnv"):
        """Bind env and scalar range; bucket upload runs in the visualizer MPM hook."""
        self._env = env
        if self._can_use_gpu_render_path(env):
            mode = getattr(env.cfg.general, "particle_color_mode", "temperature")
            if mode == "q_ind":
                self._gpu_scalar_vmin = None
                self._gpu_scalar_vmax = None
            else:
                vmin, vmax = self._scalar_range_for_mode(env, mode)
                self._gpu_scalar_vmin = vmin
                self._gpu_scalar_vmax = vmax
            self._frame_pos = None
            self._bucket_idx = None
            if not self._sync_path_logged:
                self._sync_path_logged = True
                gs.logger.info("Particle render sync: GPU-resident (defer to visualizer hook)")
            return
        self._sync_from_env_cpu(env)

    def sync_from_env(self, env: "AgilityForgeEnv"):
        """Backward-compatible alias for :meth:`prepare_render_frame`."""
        self.prepare_render_frame(env)

    def _sync_from_env_cpu(self, env: "AgilityForgeEnv"):
        """CPU bundle pull fallback when particles_render thermal fields are unavailable."""
        self._env = env
        if not self._sync_path_logged:
            self._sync_path_logged = True
            gs.logger.info("Particle render sync: CPU bundle pull")
        with teleop_profile(env, "teleop_render_particle_gpu_pull"):
            entity = env.mpm_entity
            if hasattr(entity, "get_particles_render_bundle"):
                pos, active, temps, depths = self._pull_render_bundle_to_numpy(entity)
            else:
                pos = entity.get_particles_pos().detach().cpu().numpy().reshape(-1, 3)
                temps = None
                depths = None
                if hasattr(entity, "get_particles_active"):
                    active = entity.get_particles_active(envs_idx=0).detach().cpu().numpy().reshape(-1).astype(bool)
                else:
                    active = np.ones(pos.shape[0], dtype=bool)

            scalars, vmin, vmax = self._scalar_field_from_env(
                env,
                pos,
                temps=temps,
                depths=depths,
            )
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


def toggle_particle_simple_color(env: "AgilityForgeEnv", renderer: TemperatureParticleRenderer) -> bool:
    """Toggle plain-metal particles (skips scalar coloring work)."""
    general = env.cfg.general
    general.particle_simple_color = not bool(getattr(general, "particle_simple_color", False))
    renderer._gpu_render_path_ok = None
    renderer._q_ind_vmax_cached = None
    renderer.prepare_render_frame(env)
    return general.particle_simple_color


def toggle_viewer_fps_mode(env: "AgilityForgeEnv") -> int:
    """Toggle viewer redraw cap between quality and performance targets."""
    perf = env.cfg.performance
    quality = int(getattr(perf, "viewer_fps_quality_mode", 30))
    perf_fps = int(getattr(perf, "viewer_fps_perf_mode", 15))
    current = int(getattr(perf, "target_viewer_fps", quality) or quality)
    new_fps = quality if current <= perf_fps + 1 else perf_fps
    perf.target_viewer_fps = new_fps
    env.cfg.viewer.max_FPS = new_fps
    env.cfg.viewer.refresh_rate = new_fps
    return new_fps


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
    env.cfg.general.particle_simple_color = False
    renderer._gpu_render_path_ok = None
    renderer.prepare_render_frame(env)
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

    def _toggle_simple_color():
        enabled = toggle_particle_simple_color(env, renderer)
        label = "simple (mono)" if enabled else "fancy"
        gs.logger.info(f"Particle coloring: {label}")
        update_particle_color_display(env, physics_mesher=physics_mesher)
        _refresh_viewer()

    def _toggle_viewer_fps():
        fps = toggle_viewer_fps_mode(env)
        gs.logger.info(f"Viewer FPS cap: {fps}")
        update_particle_color_display(env, physics_mesher=physics_mesher)
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
        Keybind(
            "toggle_particle_simple_color",
            Key.C,
            key_mods=(KeyMod.SHIFT,),
            key_action=KeyAction.PRESS,
            callback=_toggle_simple_color,
            allow_overload=True,
        ),
        Keybind(
            "toggle_viewer_fps_mode",
            Key.F,
            key_action=KeyAction.PRESS,
            callback=_toggle_viewer_fps,
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
    renderer._can_use_gpu_render_path(env)
    if register_keybinds:
        register_particle_color_keybinds(env, renderer)
    return renderer
