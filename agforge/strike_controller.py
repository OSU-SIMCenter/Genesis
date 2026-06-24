import asyncio
import time
import enum
import zlib

import torch
import numpy as np
import genesis as gs
import contextlib

from agforge.reconstruction import SurfaceReconstructor
from agforge.physics_mesh import InductionPhysicsMesher
from agforge.recorder import AgForgeRecorder

class StrikeState(enum.Enum):
    IDLE = 0
    APPROACHING = 1
    HOLDING = 2
    PRESSING = 3
    RELEASE = 4

class SimulationStabilityError(Exception):
    """Raised when simulation state becomes unstable (NaNs, Exploding velocities)."""
    pass


def _per_particle_field_cpu(field: torch.Tensor, n_particles: int) -> torch.Tensor:
    """Collapse batched MPM particle fields to shape (n_particles,) on CPU."""
    t = field.detach().cpu()
    if t.numel() == 0:
        return t.reshape(0)
    if t.dim() == 1 and t.shape[0] == n_particles:
        return t
    if t.dim() >= 2 and t.shape[-1] == n_particles:
        return t.reshape(-1, n_particles)[0]
    if t.dim() >= 2 and t.shape[1] == n_particles:
        return t[0].reshape(n_particles)
    flat = t.reshape(-1)
    return flat[:n_particles]


# Configuration constants
MAX_CHECKPOINTS = 50  # Maximum number of checkpoints to retain
VERBOSE_LOGGING = True  # Enable per-frame logging during strike (set to False to disable)
LOG_EVERY_N_FRAMES = 3  # Log every Nth frame (1=every frame, 10=every 10th frame)

class StrikeController:
    """
    Encapsulates the core control logic, state machine, and physics interaction 
    for the Agility Forge simulation.
    """
    def __init__(self, env):
        self.env = env
        self.robot = env.robot
        
        # Robot State
        self.qpos = self.robot.entity.get_dofs_position()
        self.dof_limits = self.robot.entity.get_dofs_limit()
        
        # We need a lock because async socket tasks might access this concurrently 
        # with the simulation loop.
        self.lock = asyncio.Lock()
        
        # Strike Logic State
        self.strike_state = StrikeState.IDLE
        self.contact_L = False
        self.contact_R = False
        self.target_strain = 0.5
        self.stage_start_time = 0.0
        self.contact_width = 0.0
        self.stabilization_steps = 0
        self.strike_step_count = 0  # Count simulation steps during strike
        
        # Gripper limits (derived from config via XML logic in original, 
        # but we can query the robot or just use the config if available)
        # Assuming we can get them from the robot entity or similar.
        # For now, let's copy the logic if we can import RobotXMLGenerator or just store them.
        # We'll calculate them once.
        self._init_gripper_limits()

        # Surface Reconstruction
        recon_cfg = env.cfg.reconstruction
        self.reconstructor = SurfaceReconstructor(
            env,
            grid_res=recon_cfg.grid_res,
            backend=recon_cfg.backend,
            physics_grid_res=recon_cfg.physics_mesh_grid_res,
            mc_backend=getattr(recon_cfg, "mc_backend", "auto"),
        )
        if getattr(recon_cfg, "unified_mesh", True):
            # Single mesh build via physics_mesher; skip per-frame visual recon.
            self.reconstructor.recon_enabled = False
            gs.logger.info(
                f"Unified surface mesh enabled ({recon_cfg.physics_mesh_backend}, "
                f"grid={recon_cfg.physics_mesh_grid_res}, mc={getattr(recon_cfg, 'mc_backend', 'auto')})"
            )
        else:
            self.reconstructor.recon_enabled = recon_cfg.enabled
        # Note: Reconstruction init mostly happens on demand or at start

        self.physics_mesher = InductionPhysicsMesher(env, self.reconstructor, env.cfg)
        self._mesh_overlay = None
        self._pending_physics_rebuild = False
        self._pending_physics_upload_sdf = True
        
        # Data Recorder
        import os
        self.recorder = AgForgeRecorder(data_dir=os.path.join(os.path.dirname(__file__), "..", "data"))
        
        # Checkpointing
        self.checkpoints = []
        
        # Transformation constants for Unity (visualization) - Deprecated
        # self._init_transforms()
        
        # Optimization: Pre-allocated tensors
        self._vel_cmd = torch.zeros(4, device=self.env.device)
        self._last_vel_cmd = torch.zeros(4, device=self.env.device)
        self._dofs_idx_local = torch.tensor([0, 1, 2, 3], device=self.env.device)
        self._force_next_apply = True
        self._last_applied_qpos = None
        self._last_apply_strike_state = None

        # Normalized force for Unity pressure gauge (0-1, where 1 = 150kN)
        self.last_force_normalized = 0.0
        self._force_gauge_max = 100000.0  # Normalization ceiling in Newtons

        # Thermal state logic
        self.thermal_enabled = False
        self.heating_power = self.env.cfg.heating_power
        self.skin_depth = self.env.cfg.skin_depth
        self._cached_slider_x: float | None = None
        self._last_induction_key: tuple | None = None
        self.heater = None
        self._pending_sdf_upload = False
        self._temp_particle_renderer = None

        # Initialize billet to physical room temperature (293.0 K)
        try:
            if hasattr(self.env, 'mpm_entity') and hasattr(self.env.mpm_entity, 'get_particles_temp'):
                current_temps = self.env.mpm_entity.get_particles_temp()
                if current_temps is not None:
                    base_temps = torch.ones_like(current_temps) * 293.0
                    self.env.mpm_entity.set_particles_temp(base_temps)
                    import quadrants as qd  # type: ignore
                    qd.sync()
                    gs.logger.info("Initialized billet thermal baseline to 293.0 K")
        except Exception as e:
            gs.logger.warning(f"Could not initialize base temperatures: {e}")

        # Stability checking state
        self._physics_step_counter = 0
        self._stability_grace_steps = 0  # Suppress checks after undo/reset to let state settle

        # Diagnostic timing state
        self._diag_start_real_time = time.time()
        self._diag_last_real_time = self._diag_start_real_time
        self._diag_last_sim_time = 0.0
        self._diag_last_step = 0

        # Cached kNN bind for Unity vertex temperatures (SciPy cKDTree; rebuilt on geometry change).
        self._invalidate_unity_vertex_temp_cache()

        # Teleop IO / viewer throttling
        self._io_frame_counter = 0
        self._last_viewer_update_mono = 0.0
        self._io_mesh_cache: dict | None = None
        self._unity_knn_indices_t: torch.Tensor | None = None
        self._unity_knn_weights_t: torch.Tensor | None = None

    def _invalidate_unity_vertex_temp_cache(self) -> None:
        self._unity_knn_fingerprint_cache = None
        self._unity_knn_quick_stamp = None
        self._unity_knn_indices = None
        self._unity_knn_weights = None
        self._unity_knn_fade_alpha = None
        self._unity_knn_indices_t = None
        self._unity_knn_weights_t = None
        self._io_mesh_cache = None

    def _perf(self):
        return getattr(self.env.cfg, "performance", None)

    def _stability_check_interval(self) -> int:
        safety = getattr(self.env.cfg, "safety", None)
        base = max(1, int(getattr(safety, "check_interval", 1) if safety else 1))
        if safety is None:
            return base

        if self.strike_state == StrikeState.IDLE:
            if self.thermal_enabled:
                heating_iv = int(getattr(safety, "heating_idle_check_interval", base))
                return max(base, heating_iv)
            return base

        if self.strike_state == StrikeState.APPROACHING:
            return max(1, int(getattr(safety, "approaching_check_interval", base)))
        if self.strike_state in (StrikeState.PRESSING, StrikeState.HOLDING, StrikeState.RELEASE):
            return max(1, int(getattr(safety, "strike_check_interval", 1)))
        return base

    def _should_update_viewer(self) -> bool:
        perf = self._perf()
        cap = int(getattr(perf, "target_viewer_fps", 0) or 0) if perf else 0
        if cap <= 0:
            return True
        now = time.monotonic()
        min_dt = 1.0 / cap
        if now - self._last_viewer_update_mono < min_dt:
            return False
        self._last_viewer_update_mono = now
        return True

    def _should_compute_vertex_temps_io(self) -> bool:
        perf = self._perf()
        if self.strike_state != StrikeState.IDLE:
            interval = max(1, int(getattr(perf, "vertex_temp_io_interval_strike", 1) if perf else 1))
            return (self._io_frame_counter % interval) == 0
        if self.thermal_enabled:
            interval = max(
                1,
                int(getattr(perf, "vertex_temp_io_interval_heating_idle", 6) if perf else 6),
            )
        else:
            interval = max(1, int(getattr(perf, "vertex_temp_io_interval", 3) if perf else 3))
        return (self._io_frame_counter % interval) == 0

    def _should_refresh_mesh_io(self) -> bool:
        if self.strike_state != StrikeState.IDLE or not self.thermal_enabled:
            return True
        perf = self._perf()
        interval = max(1, int(getattr(perf, "mesh_io_interval_heating_idle", 1) if perf else 1))
        return (self._io_frame_counter % interval) == 0

    def _can_reuse_cached_vertex_temps(self, n_verts: int) -> bool:
        if self._io_mesh_cache is None or n_verts <= 0:
            return False
        prev = self._io_mesh_cache.get("vertices_temp")
        return prev is not None and len(prev) == n_verts

    def _unity_mesh_stamp(self) -> int:
        if self._uses_unified_surface_mesh():
            return int(getattr(self.physics_mesher, "version", 0))
        return int(getattr(self.reconstructor, "mesh_version", 0))

    def _invalidate_applied_qpos_cache(self) -> None:
        self._last_applied_qpos = None
        self._last_apply_strike_state = None
        self._invalidate_induction_params_cache()

    def _invalidate_induction_params_cache(self) -> None:
        self._cached_slider_x = None
        self._last_induction_key = None

    def _slider_x_for_induction(self) -> float:
        if self._cached_slider_x is not None:
            return self._cached_slider_x
        if self.qpos is None:
            return 0.0
        self._cached_slider_x = float(self.qpos[0, 0].item())
        return self._cached_slider_x

    def _maybe_set_induction_params(self, *, active: bool) -> None:
        cfg_robot = self.env.cfg.robot
        slider_x = self._slider_x_for_induction()
        key = (
            slider_x,
            float(self.heating_power),
            float(self.skin_depth),
            bool(active),
            float(cfg_robot.coil_length) / 2.0,
            float(cfg_robot.coil_radius),
            float(cfg_robot.coil_offset_x),
            float(cfg_robot.cylinder_pos[2]),
        )
        if key == self._last_induction_key:
            return
        coil_center = [slider_x + cfg_robot.coil_offset_x, 0.0, cfg_robot.cylinder_pos[2]]
        self.env.scene.sim.mpm_solver.set_induction_params(
            center=coil_center,
            half_length=key[4],
            radius=key[5],
            q_peak=key[1],
            skin_depth=key[2],
            active=key[3],
        )
        self._last_induction_key = key

    def _mark_qpos_applied(self, qpos: torch.Tensor) -> None:
        self._last_applied_qpos = qpos.clone()
        self._last_apply_strike_state = self.strike_state

    @staticmethod
    def _unity_knn_fingerprint(mapping_parts_np: np.ndarray, verts_np: np.ndarray) -> tuple:
        return (
            int(mapping_parts_np.shape[0]),
            int(verts_np.shape[0]),
            int(zlib.adler32(np.ascontiguousarray(verts_np, dtype=np.float32).tobytes())),
            int(zlib.adler32(np.ascontiguousarray(mapping_parts_np, dtype=np.float32).tobytes())),
        )

    def _ensure_unity_vertex_temp_cache(self, mapping_parts_np: np.ndarray, verts_np: np.ndarray) -> None:
        quick_stamp = (
            self._unity_mesh_stamp(),
            int(mapping_parts_np.shape[0]),
            int(verts_np.shape[0]),
        )
        if (
            self._unity_knn_quick_stamp == quick_stamp
            and self._unity_knn_fingerprint_cache is not None
            and self._unity_knn_indices is not None
        ):
            return

        fingerprint = self._unity_knn_fingerprint(mapping_parts_np, verts_np)
        if self._unity_knn_fingerprint_cache == fingerprint:
            self._unity_knn_quick_stamp = quick_stamp
            return

        k = min(3, int(mapping_parts_np.shape[0]))

        with self._profile("teleop_io_kdtree_build"):
            from scipy.spatial import cKDTree

            tree = cKDTree(mapping_parts_np, leafsize=32)
            dists, indices = tree.query(verts_np, k=k)
            if k == 1:
                dists = dists[:, np.newaxis]
                indices = indices[:, np.newaxis]
            dists = np.maximum(dists, 1e-6)
            weights = 1.0 / dists
            weights_norm = weights / weights.sum(axis=1, keepdims=True)
            self._unity_knn_indices = np.asarray(indices, dtype=np.int32)
            self._unity_knn_weights = np.asarray(weights_norm, dtype=np.float32)
            self._unity_knn_indices_t = None
            self._unity_knn_weights_t = None

        self._unity_knn_fingerprint_cache = fingerprint
        self._unity_knn_quick_stamp = quick_stamp

        if getattr(self.env.cfg, "thermal_visual_fade", True):
            L_v = self.env.cfg.robot.cylinder_height
            cylinder_center_x = self.env.cfg.robot.cylinder_pos[0]
            x_max_v = cylinder_center_x + (L_v / 2.0)
            x_clamp_v = x_max_v - 0.11 * L_v
            x_fade_v = x_max_v - 0.21 * L_v
            self._unity_knn_fade_alpha = np.clip(
                (mapping_parts_np[:, 0] - x_fade_v) / (x_clamp_v - x_fade_v),
                0.0,
                1.0,
            ).astype(np.float32)
        else:
            self._unity_knn_fade_alpha = None

    def _map_particle_temps_to_vertices(
        self,
        particles_temp: np.ndarray | torch.Tensor,
        mapping_parts_np: np.ndarray,
        verts_np: np.ndarray,
    ) -> np.ndarray:
        if mapping_parts_np.shape[0] < 3 or verts_np.shape[0] == 0:
            return np.zeros(verts_np.shape[0], dtype=np.float32)

        self._ensure_unity_vertex_temp_cache(mapping_parts_np, verts_np)

        with self._profile("teleop_io_kdtree_vertex_temps"):
            if isinstance(particles_temp, torch.Tensor):
                temps = particles_temp.reshape(-1)
                device = temps.device
                if self._unity_knn_indices_t is None or self._unity_knn_indices_t.device != device:
                    self._unity_knn_indices_t = torch.from_numpy(self._unity_knn_indices).to(device=device)
                    self._unity_knn_weights_t = torch.from_numpy(self._unity_knn_weights).to(device=device)
                if self._unity_knn_fade_alpha is not None:
                    fade = torch.from_numpy(self._unity_knn_fade_alpha).to(device=device, dtype=temps.dtype)
                    target_vis = torch.minimum(temps, torch.tensor(900.0, device=device, dtype=temps.dtype))
                    temps = temps * (1.0 - fade) + target_vis * fade
                neighbor_temps = temps[self._unity_knn_indices_t]
                vertices_temp = (neighbor_temps * self._unity_knn_weights_t).sum(dim=1)
                return vertices_temp.detach().cpu().numpy().astype(np.float32)

            temps = particles_temp.reshape(-1)
            if self._unity_knn_fade_alpha is not None:
                target_vis = np.minimum(temps, 900.0)
                temps = temps * (1.0 - self._unity_knn_fade_alpha) + target_vis * self._unity_knn_fade_alpha

            neighbor_temps = temps[self._unity_knn_indices]
            vertices_temp = (neighbor_temps * self._unity_knn_weights).sum(axis=1)
            return vertices_temp.astype(np.float32)

    def _uses_unified_surface_mesh(self) -> bool:
        return bool(getattr(self.env.cfg.reconstruction, "unified_mesh", True))

    def _needs_surface_rebuild(self) -> bool:
        """Whether geometry changed enough to rebuild the cached surface mesh."""
        return self._uses_unified_surface_mesh() or self.thermal_enabled

    def request_physics_rebuild(self, upload_sdf: bool = True) -> None:
        """Queue a physics mesh rebuild on the asyncio main thread."""
        self._pending_physics_rebuild = True
        self._pending_physics_upload_sdf = upload_sdf

    def ensure_physics_mesh(self) -> bool:
        """Build the physics mesh if missing and refresh the viewer overlay cache."""
        if len(self.physics_mesher.physics_mesh.vertices) < 4:
            self.request_physics_rebuild(upload_sdf=False)
            return False
        if self._mesh_overlay is not None:
            self._mesh_overlay.sync_from_controller(self)
        return True

    async def process_pending_physics_rebuild(self) -> bool:
        """Run a queued physics mesh rebuild (viewer keybinds must not call rebuild directly)."""
        if not self._pending_physics_rebuild:
            return False
        self._pending_physics_rebuild = False
        upload_sdf = self._pending_physics_upload_sdf
        async with self.lock:
            return self.rebuild_physics_induction(upload_sdf=upload_sdf)

    def rebuild_physics_induction(self, upload_sdf: bool = True) -> bool:
        """Rebuild surface mesh; in unified mode one build serves visual + induction SDF."""
        with self._profile("teleop_physics_rebuild"):
            ok = self.physics_mesher.rebuild()
            if upload_sdf and self.heater is not None:
                self.heater.recompute_and_upload()
        with self._profile("teleop_physics_rebuild_overlay_sync"):
            if self._mesh_overlay is not None:
                self._mesh_overlay.sync_from_controller(self)
            if self._temp_particle_renderer is not None:
                self._temp_particle_renderer.prepare_render_frame(self.env)
            from agforge.vis.temperature_particles import update_particle_color_display

            update_particle_color_display(self.env, physics_mesher=self.physics_mesher)
        return ok

    def cycle_physics_mesh_backend(self) -> str:
        """Swap the induction SDF backend; rebuild runs on the asyncio main thread."""
        self.physics_mesher.cycle_backend(step=1)
        gs.logger.info(f"Physics SDF backend: {self.physics_mesher.backend_label} (queued)")
        self.request_physics_rebuild(upload_sdf=self.heater is not None)
        return self.physics_mesher.backend.value

    def _init_gripper_limits(self):
        # We can re-use the XML generator logic or just hardcode if standard.
        # Since agforge_builder is where RobotXMLGenerator lives, we might need to import it
        # or just assume standard values if not critical. 
        # Better: Import it to be safe.
        from agforge.agforge_builder import RobotXMLGenerator
        xml_generator = RobotXMLGenerator(robot_cfg=self.env.cfg.robot)
        self.gripper_closed_pos = xml_generator.gripper_slide_range[1]
        self.gripper_open_pos = xml_generator.gripper_slide_range[0]


    async def set_thermal_state(self, enabled: bool):
        """Toggles induction heating."""
        async with self.lock:
            self.thermal_enabled = enabled
                
            if enabled:
                gs.logger.info(f"Thermal ACTIVATED (q_peak={self.heating_power:.3e} W/m³)")
            else:
                gs.logger.info("Thermal FROZEN")
                
            # Force server to broadcast the new thermal state back to Unity
            # This flushes any stale "heating=True" frames that might cause
            # Unity's SyncHeatVisual to erroneously flip the button back to ON.
            self.pending_mesh_send = True
            self.stabilization_steps = max(getattr(self, 'stabilization_steps', 0), 5)

    async def set_qpos(self, new_qpos):
        async with self.lock:
            # Only clamp slider/hinge, grippers managed by logic if striking
            new_qpos[:, :2] = torch.clamp(new_qpos[:, :2], self.dof_limits[0][:2], self.dof_limits[1][:2])
            self.qpos = new_qpos
            self._invalidate_applied_qpos_cache()

    async def get_qpos(self):
        async with self.lock:
            return self.qpos.clone()

    async def trigger_strike(self, force_param):
        async with self.lock:
            if self.strike_state != StrikeState.IDLE:
                gs.logger.warning(f"Strike requested but already in {self.strike_state.name}")
                return
            
            # SAVE CHECKPOINT BEFORE STRIKE STARTS
            # This ensures we can undo to the exact state before the attempt, 
            # preserving the user's intended position/angle.
            self.recorder.mark_strike_start()
            self._save_checkpoint_impl() 
            
            # Log initial state for debugging
            pos_L = self.robot.left_gripper.get_pos()
            pos_R = self.robot.right_gripper.get_pos()
            initial_width = torch.norm(pos_L - pos_R).item()
            
            gs.logger.info(f"Strike -> APPROACHING (target_strain={force_param:.4f})")
            gs.logger.info(f"  Initial: width={initial_width:.4f}, pos_L={pos_L[0].tolist()}, pos_R={pos_R[0].tolist()}")
            gs.logger.info(f"  Config: approach_spd={self.env.cfg.strike.approach_speed}, press_spd={self.env.cfg.strike.pressing_speed}, max_force={self.env.cfg.strike.max_force}")
            
            self.contact_L = False
            self.contact_R = False
            self.strike_state = StrikeState.APPROACHING
            self.stage_start_time = time.time()
            self.strike_start_time = time.time()  # Track total strike duration
            self.stabilization_steps = 0
            self.target_strain = force_param
            self.strike_step_count = 0
            self._invalidate_applied_qpos_cache()
            self.robot.set_control_mode("VELOCITY_CONTROL")
            gs.logger.info(f"  Control mode -> VELOCITY_CONTROL")
            
            # === FIX: Zero stale CPIC normals before first physics step ===
            # The CPIC normal field `mpm_rigid_normal` stores per-particle SDF
            # normals from the last physics step. After the user teleports the
            # robot (slider/hinge), these normals correspond to the OLD gripper
            # position. The directional guard in `mpm_surface_to_particle`:
            #     if sdf_normal.dot(mpm_rigid_normal[i_p, i_g, i_b]) >= 0:
            # prevents updating normals that have flipped due to the teleport,
            # leaving stale normals that cause incorrect CPIC separation
            # decisions in p2g/g2p → massive force spike → blow-up.
            #
            # Zeroing both fields lets the very first substep's preprocess
            # write fresh normals unconditionally (dot(new, zero) = 0 >= 0).
            coupler = self.env.scene.sim.coupler
            if hasattr(coupler, 'mpm_rigid_normal'):
                coupler.mpm_rigid_normal.fill(0)
            if hasattr(coupler, 'cpic_flag'):
                coupler.cpic_flag.fill(0)
            


    async def update_logic(self):
        """
        Executes one step of the strike state machine logic.
        This mirrors `update_strike_logic` from teleop_socket.py.
        """

        with self._profile("logic_step"):
            # --- APPROACHING STAGE ---
            if self.strike_state == StrikeState.APPROACHING:
                with self._profile("logic_update_state"):
                    approach_speed = self.env.cfg.strike.approach_speed
                    contact_threshold = self.env.cfg.strike.contact_force_threshold
                    approaching_timeout = self.env.cfg.strike.approaching_timeout
                    
                    if time.time() - self.stage_start_time > approaching_timeout:
                        gs.logger.warning(f"Strike APPROACHING timed out")
                        self.strike_state = StrikeState.RELEASE
                        self.stage_start_time = time.time()
                        self._stop_motors()
                        return
                
                    with self._profile("logic_get_resistance"):
                        # force_L, force_R are now TENSORS
                        force_L, force_R = self.robot.get_resistance_forces()
                    self._update_force_gauge(force_L, force_R)
                    
                    # Batch contact check sync: Combine predicates
                    # We need to know specific contacts to set velocity
                    # contacts = [L_hit, R_hit]
                    contacts_tensor = torch.stack([force_L > contact_threshold, force_R > contact_threshold])
                    
                    with self._profile("logic_tensor_sync"):
                        new_contact_L = contacts_tensor[0].item()
                        new_contact_R = contacts_tensor[1].item()

                    if not self.contact_L and new_contact_L:
                        self.contact_L = True
                        gs.logger.info(f"  Left gripper CONTACT (force={force_L.item():.4f})")
                        
                    if not self.contact_R and new_contact_R:
                        self.contact_R = True
                        gs.logger.info(f"  Right gripper CONTACT (force={force_R.item():.4f})")
                    
                    if VERBOSE_LOGGING and self.strike_step_count % LOG_EVERY_N_FRAMES == 0:
                        gs.logger.info(f"APPROACHING[{self.strike_step_count}]: F=[{force_L.item():.3f},{force_R.item():.3f}], contact=[{self.contact_L}, {self.contact_R}]")

                    self._vel_cmd.zero_()
                    self._vel_cmd[2] = 0.0 if self.contact_L else approach_speed
                    self._vel_cmd[3] = 0.0 if self.contact_R else approach_speed
                
                    with self._profile("logic_apply_vel"):
                        self._apply_vel_smart()
                
                    if self.contact_L and self.contact_R:
                        self.strike_state = StrikeState.PRESSING
                        self.stage_start_time = time.time()

                        with self._profile("logic_gripper_get_pos"):
                            pos_L = self.robot.left_gripper.get_pos()
                            pos_R = self.robot.right_gripper.get_pos()
                        self.contact_width_tensor = torch.norm(pos_L - pos_R)
                        with self._profile("logic_tensor_sync"):
                            self.contact_width = self.contact_width_tensor.item()
                        
                        gs.logger.info(f"Strike -> PRESSING (width={self.contact_width:.4f}, steps={self.strike_step_count})")
                    
            # --- PRESSING STAGE ---
            elif self.strike_state == StrikeState.PRESSING:
                with self._profile("logic_update_state"):
                    pressing_speed = self.env.cfg.strike.pressing_speed
                    target_strain = self.target_strain
                    max_force = self.env.cfg.strike.max_force
                    pressing_timeout = self.env.cfg.strike.pressing_timeout
                    force_balance_gain = self.env.cfg.strike.force_balance_gain

                    with self._profile("logic_get_resistance"):
                        force_L, force_R = self.robot.get_resistance_forces()
                    self._update_force_gauge(force_L, force_R)
                
                    with self._profile("logic_gripper_get_pos"):
                        pos_L = self.robot.left_gripper.get_pos()
                        pos_R = self.robot.right_gripper.get_pos()
                    current_width_tensor = torch.norm(pos_L - pos_R)
                    
                    if self.contact_width > 1e-6:
                         # contact_width_tensor saved in transition
                         # Use GPU tensor for reference width to avoid hybrid ops if possible, 
                         # though contact_width_tensor is already a 0-dim tensor.
                        current_strain_tensor = (self.contact_width_tensor - current_width_tensor) / self.contact_width_tensor
                    else:
                        current_strain_tensor = torch.tensor(0.0, device=self.env.device)
                    
                    # --- Logic Batching ---
                    # Conditions to check: Target Strain, Max Force
                    # We combine them into a single sync
                    
                    # Check 1: Strain (Tensor)
                    cond_strain = current_strain_tensor >= target_strain
                    
                    # Check 2: Max Force (Tensor)
                    cond_force = (force_L > max_force) | (force_R > max_force)
                    
                    # Check 3: Timeout (CPU - cheap)
                    elapsed_time = time.time() - self.stage_start_time
                    is_timeout = elapsed_time > pressing_timeout
                    
                    # Stack GPU conditions
                    stop_flags = torch.stack([cond_strain, cond_force])
                    
                    with self._profile("logic_tensor_sync"):
                        stop_any = stop_flags.any().item() or is_timeout

                    if stop_any:
                        with self._profile("logic_tensor_sync"):
                            if cond_strain.item():
                                stop_reason = "Target Strain"
                            elif cond_force.item():
                                stop_reason = "Max Force"
                            elif is_timeout:
                                stop_reason = "Timeout"
                            else:
                                stop_reason = "Unknown"
                            strain_val = current_strain_tensor.item()
                        gs.logger.info(f"Strike -> HOLDING ({stop_reason}, strain={strain_val:.4f}, steps={self.strike_step_count}, time={elapsed_time:.2f}s)")
                        self.strike_state = StrikeState.HOLDING
                        self.hold_steps_remaining = self.env.cfg.strike.hold_steps
                        self.stage_start_time = time.time()
                        self._stop_motors()
                        return

                    # Physics Update (Tensor Ops)
                    imbalance = force_L - force_R
                    
                    # --- ADVANCED PROTECTION: Feed Rate Modulation ---
                    # If force imbalance exceeds threshold, slow down the main pressing speed
                    # to allow the balance controller to catch up without fighting forward momentum.
                    SAFETY_THRESHOLD = 20000.0 # 20kN (~10% of max force)
                    adaptive_speed = pressing_speed
                    
                    imbalance_abs = torch.abs(imbalance)
                    if imbalance_abs > SAFETY_THRESHOLD:
                        # Linear decay: at 40kN imbalance, speed is 0.
                        # decay = 1.0 - (imbalance - Threshold) / Range
                        # Simple logic: If > 20kN, scale down.
                        # factor = 20000 / imbalance
                        scale_factor = SAFETY_THRESHOLD / (imbalance_abs + 1e-6)
                        adaptive_speed = pressing_speed * scale_factor
                        # gs.logger.debug(f"Protection Active: dF={imbalance.item():.0f}, speed={adaptive_speed:.2f}")

                    correction = imbalance * force_balance_gain
                    
                    v_L = torch.clamp(adaptive_speed - correction, min=0.0)
                    v_R = torch.clamp(adaptive_speed + correction, min=0.0)
                    
                    if VERBOSE_LOGGING and self.strike_step_count % LOG_EVERY_N_FRAMES == 0:
                        gs.logger.info(f"PRESSING[{self.strike_step_count}]: F=[{force_L.item():.3f},{force_R.item():.3f}] dF={imbalance.item():.4f}, v=[{v_L.item():.4f},{v_R.item():.4f}] corr={correction.item():.4f}")
                    
                    self._vel_cmd.zero_()
                    self._vel_cmd[2] = v_L
                    self._vel_cmd[3] = v_R
                
                with self._profile("logic_apply_vel"):
                    self._apply_vel_smart()

            # --- RELEASE STAGE ---
            elif self.strike_state == StrikeState.RELEASE:
                with self._profile("logic_update_state"):
                    release_speed = self.env.cfg.strike.pressing_speed
                    contact_threshold = self.env.cfg.strike.contact_force_threshold * 0.2
                    force_balance_gain = self.env.cfg.strike.force_balance_gain
                    release_timeout = self.env.cfg.strike.release_timeout
                    
                    if time.time() - self.stage_start_time > release_timeout:
                        gs.logger.warning(f"Strike RELEASE timed out - Forcing reset")
                        self._force_idle_reset()
                        await self.save_checkpoint()
                        return
                
                with self._profile("logic_get_resistance"):
                    force_L, force_R = self.robot.get_resistance_forces()
                self._update_force_gauge(force_L, force_R)

                with self._profile("logic_update_state"):
                    imbalance = force_L - force_R
                    correction = imbalance * force_balance_gain
                    
                    v_open = -release_speed
                    v_L = v_open - correction
                    v_R = v_open + correction
                    
                    if VERBOSE_LOGGING and self.strike_step_count % LOG_EVERY_N_FRAMES == 0:
                        gs.logger.info(f"RELEASE[{self.strike_step_count}]: F=[{force_L.item():.3f},{force_R.item():.3f}] dF={imbalance.item():.4f}")
                    
                    self._vel_cmd.zero_()
                    self._vel_cmd[2] = v_L
                    self._vel_cmd[3] = v_R
                
                with self._profile("logic_apply_vel"):
                    self._apply_vel_smart()
                
                with self._profile("logic_update_state"):
                    # Batch check for release completion
                    # cond: abs(force) < threshold
                    is_free = (torch.abs(force_L) < contact_threshold) & (torch.abs(force_R) < contact_threshold)

                    with self._profile("logic_tensor_sync"):
                        release_complete = is_free.item()

                    if release_complete:
                        with self._profile("logic_gripper_get_pos"):
                            pos_L = self.robot.left_gripper.get_pos()
                            pos_R = self.robot.right_gripper.get_pos()
                        with self._profile("logic_tensor_sync"):
                            final_width = torch.norm(pos_L - pos_R).item()
                        total_duration = time.time() - getattr(self, 'strike_start_time', self.stage_start_time)
                        
                        self._force_idle_reset()
                        # Stabilization after release
                        self.stabilization_steps = self.env.cfg.strike.post_release_steps
                        
                        gs.logger.info(f"Strike -> IDLE (Stabilizing for {self.stabilization_steps} steps)")
                        gs.logger.info(f"  Summary: total_time={total_duration:.2f}s, steps={self.strike_step_count}, final_width={final_width:.4f}")
                        
                        # RESTORED: Full reconstruction after strike to fix skinning drift
                        # This gives the user a "perfect" result after the action completes.
                        # DISABLED (User Request): Relying purely on edge splitting
                        # recon_start = time.time()
                        # self.reconstructor.create_reconstructed_mesh()
                        # self.reconstructor.init_skinning() # REQUIRED to sync weights with new mesh
                        # recon_time = (time.time() - recon_start) * 1000
                        # gs.logger.info(f"  Post-strike reconstruction: {recon_time:.1f}ms")

                        # Replace the pre-strike checkpoint (saved in trigger_strike)
                        # with this post-strike state so 1 undo = 1 strike reversal.
                        await self.save_checkpoint(replace_last=True)

                        # Flag to ensure updated mesh is sent to client
                        self.pending_mesh_send = True

                        # Rebuild surface mesh (unified visual+physics) and SDF after strike.
                        if self._needs_surface_rebuild():
                            upload_sdf = self.thermal_enabled and self.heater is not None
                            self.rebuild_physics_induction(upload_sdf=upload_sdf)
                        return

            # --- HOLDING STAGE ---
            elif self.strike_state == StrikeState.HOLDING:
                with self._profile("logic_get_resistance"):
                    force_L, force_R = self.robot.get_resistance_forces()
                self._update_force_gauge(force_L, force_R)
                
                with self._profile("logic_update_state"):
                    self.hold_steps_remaining -= 1
                    self._stop_motors()
                    
                    if VERBOSE_LOGGING and self.strike_step_count % LOG_EVERY_N_FRAMES == 0:
                        gs.logger.info(f"HOLDING[{self.strike_step_count}]: F=[{force_L.item():.3f},{force_R.item():.3f}] steps_left={self.hold_steps_remaining}")
                    
                    if self.hold_steps_remaining <= 0:
                        gs.logger.info(f"Strike -> RELEASE (Hold complete)")
                        self.strike_state = StrikeState.RELEASE
                        self.stage_start_time = time.time()
                return

    def _stop_motors(self):
        self._vel_cmd.zero_()
        self._apply_vel_smart()

    def _apply_vel_smart(self):
        """
        Applies velocity without checking for equality on CPU to avoid sync.
        Trusts the solver to handle updates efficiently.
        """
        self.robot.apply_velocity(self._vel_cmd, dofs_idx_local=self._dofs_idx_local)
        self._force_next_apply = False # Flag not really needed anymore but kept for compat checking

    def _force_idle_reset(self):
         self._stop_motors()
         self._force_next_apply = True
         self.last_force_normalized = 0.0  # Reset pressure gauge on idle
         
         current_qpos = self.qpos.clone()
         current_qpos[:, 2] = self.gripper_open_pos
         current_qpos[:, 3] = self.gripper_open_pos
         
         self.robot.set_control_mode("TELEPORT")
         self.qpos = current_qpos
         self.robot.apply_action(current_qpos)
         self._mark_qpos_applied(current_qpos)
         self._cached_slider_x = float(current_qpos[0, 0].item())
         self._last_induction_key = None

         self.strike_state = StrikeState.IDLE
         self.contact_L = False
         self.contact_R = False
         self.contact_width = 0.0

    def _update_force_gauge(self, force_L, force_R):
        """Update normalized force for Unity pressure gauge: avg(|F_L|, |F_R|) / max, clamped to [0, 1].
        During PRESSING, force can only increase (monotonic ramp-up) for a stable gauge reading."""
        with self._profile("logic_force_gauge"):
            avg_abs = (torch.abs(force_L) + torch.abs(force_R)) * 0.5
            new_val = min(1.0, avg_abs.item() / self._force_gauge_max)
            if self.strike_state in (StrikeState.PRESSING, StrikeState.HOLDING, StrikeState.RELEASE):
                self.last_force_normalized = max(self.last_force_normalized, new_val)
            else:
                self.last_force_normalized = new_val

    async def step_simulation(self):
        """
        Single atomic step of complexity:
        - Logic Update
        - Action Application
        - Force Clearing
        - Physics Step
        - Render Update
        - Reconstruction (if needed)
        """
        with self._profile("teleop_process_physics_rebuild"):
            await self.process_pending_physics_rebuild()

        # 1. Logic Update
        with self._profile("teleop_logic"):
            await self.update_logic()
        
        # Track steps during active strike
        if self.strike_state != StrikeState.IDLE:
            self.strike_step_count += 1

        # 2. Apply Actions (if not handled by strike logic, handle idle holding)
        with self._profile("teleop_apply_action"):
            if self.strike_state == StrikeState.IDLE or self.strike_state == StrikeState.HOLDING:
                with self._profile("teleop_apply_action_get_qpos"):
                    qpos = await self.get_qpos()
                state_changed = self._last_apply_strike_state != self.strike_state
                qpos_changed = (
                    self._last_applied_qpos is None
                    or self._last_applied_qpos.shape != qpos.shape
                    or not torch.equal(qpos, self._last_applied_qpos)
                )
                if state_changed or qpos_changed:
                    self.robot.apply_action(qpos)
                    self._mark_qpos_applied(qpos)
                    self._cached_slider_x = float(qpos[0, 0].item())
                    self._last_induction_key = None

        # 3. Clear Forces
        with self._profile("teleop_clear_forces"):
            if hasattr(self.env.scene.sim.coupler, 'clear_link_coupling_forces'):
                self.env.scene.sim.coupler.clear_link_coupling_forces()

        # 3b. Thermodynamics
        frozen_temps_tensor = None
        _is_striking = self.strike_state != StrikeState.IDLE
        tuner = getattr(self, "thermal_tuner", None)
        if tuner is not None:
            with self._profile("teleop_thermal_tuner"):
                tuner.on_pre_physics_step()
        if self.thermal_enabled:
            with self._profile("teleop_heating"):
                if self.heater is None:
                    with self._profile("teleop_heating_init"):
                        from agforge.thermal import InductionHeater
                        self.heater = InductionHeater(
                            solver=self.env.scene.sim.mpm_solver,
                            entity=self.env.mpm_entity,
                            physics_mesher=self.physics_mesher,
                            reconstructor=self.reconstructor,
                        )
                        # Precompute the initial physics mesh; defer SDF upload to next frame.
                        self.rebuild_physics_induction(upload_sdf=False)
                        self._pending_sdf_upload = True

                if self._pending_sdf_upload and self.heater is not None:
                    with self._profile("teleop_heating_sdf_upload"):
                        self.heater.recompute_and_upload()
                    self._pending_sdf_upload = False

                # Induction heat is deposited on the GPU inside the MPM P2G kernel. Here we
                # only publish the per-frame coil uniforms (center rides the sliding arm).
                # The coil is "powered" only when not striking.
                with self._profile("teleop_heating_set_params"):
                    self._maybe_set_induction_params(active=not _is_striking)
        else:
            # Thermal Freezing: Disable natural diffusion by snapshotting temperatures before physics step
            if hasattr(self.env.scene.sim.mpm_solver, 'particles') and hasattr(self.env.scene.sim.mpm_solver.particles, 'temp'):
                with self._profile("teleop_thermal_freeze_clone"):
                    frozen_temps_tensor = self.env.mpm_entity.get_particles_temp().clone()

        # 4. Physics Step
        physics_failed = False
        with self._profile("teleop_physics"):
            try:
                # Physics step
                with self._profile("sim_step"):
                    if hasattr(self.env.mpm_entity, 'clear_thermal_telemetry_buffers'):
                        self.env.mpm_entity.clear_thermal_telemetry_buffers()
                    self.env.scene.step(update_visualizer=False)
            except Exception as e:
                gs.logger.error(f"Physics step failed: {e}")
                physics_failed = True

        # Restore frozen temperatures to halt diffusion if thermal_enabled == False
        with self._profile("teleop_thermal_bcs"):
            if not self.thermal_enabled and frozen_temps_tensor is not None and not physics_failed:
                self.env.mpm_entity.set_particles_temp(frozen_temps_tensor)

            self._physics_step_counter += 1

            # --- DEBUG TEMP OVERRIDE ---
            # Uncomment the block below to manually test visual alignment of the induction coil and press geometry.
            # This completely overrides all thermal physics and boundary conditions to forcefully render
            # particles inside the geometrical bounds as brightly colored (orange/yellow) for alignment debugging.
            #
            # import torch
            # pos = self.env.mpm_entity.get_particles_pos()
            # slider_x = self.qpos[0, 0].item() if self.qpos is not None else 0.0
            # 
            # # ** 1. Coil Calculation (1300K) **
            # c_x = slider_x + self.env.cfg.robot.coil_offset_x
            # dx_coil = pos[:, :, 0] - c_x
            # coil_mask = torch.abs(dx_coil) <= (self.env.cfg.robot.coil_length / 2.0)
            # 
            # # ** 2. Press Calculation (1800K) **
            # # The gripper X-position tracks the slider exactly (no offset).
            # # The X half-length of the gripper box in the MJCF is defined as 0.5 * cylinder_radius.
            # dx_press = pos[:, :, 0] - slider_x
            # press_mask = torch.abs(dx_press) <= (self.env.cfg.robot.cylinder_radius * 0.5)
            # 
            # curr_temps = self.env.mpm_entity.get_particles_temp()
            # # Handle shape dynamically
            # if coil_mask.dim() < curr_temps.dim():
            #     coil_mask = coil_mask.unsqueeze(-1)
            #     press_mask = press_mask.unsqueeze(-1)
            #     
            # # Base temperature (293K)
            # debug_temps = torch.full_like(curr_temps, 293.0)
            # 
            # # Overlay Coil mask (1300K - glowing orange/red)
            # debug_temps = torch.where(coil_mask, torch.tensor(1300.0, device=pos.device, dtype=curr_temps.dtype), debug_temps)
            # 
            # # Overlay Press mask (1800K - bright yellow/white for contrast)
            # debug_temps = torch.where(press_mask, torch.tensor(1800.0, device=pos.device, dtype=curr_temps.dtype), debug_temps)
            # 
            # self.env.mpm_entity.set_particles_temp(debug_temps)
            # ---------------------------

        # --- GPU drain once before any post-physics GPU reads this step ---
        import quadrants as qd  # type: ignore

        safety = getattr(self.env.cfg, 'safety', None)
        _telemetry_interval = 10 if _is_striking else 20
        _run_telemetry = (
            self.thermal_enabled and self._physics_step_counter % _telemetry_interval == 0
        )

        if physics_failed:
            needs_check = True
        elif self._stability_grace_steps > 0:
            self._stability_grace_steps -= 1
            needs_check = False
        elif not safety or not getattr(safety, "enabled", True):
            needs_check = False
        else:
            needs_check = (self._physics_step_counter % self._stability_check_interval()) == 0

        _will_render = bool(self.env.scene.visualizer and self._should_update_viewer())
        _will_record = self.strike_state != StrikeState.IDLE
        if needs_check or _run_telemetry or _will_record or _will_render:
            with self._profile("teleop_gpu_drain"):
                qd.sync()

        # --- Thermal Telemetry ---
        # Log every 10th frame during strikes, and every 20th frame during idle/heating
        with self._profile("teleop_thermal_telemetry"):
            if _run_telemetry:
                try:
                    with self._profile("teleop_telemetry_gpu_pull"):
                        entity = self.env.mpm_entity
                        if hasattr(entity, "get_particles_thermal_telemetry_bundle"):
                            (
                                temps_after_physics,
                                dT_induction,
                                dT_conv,
                                dT_rad,
                                dT_bulk,
                                dT_diffusion,
                                dT_contact,
                                dT_adiabatic,
                            ) = entity.get_particles_thermal_telemetry_bundle()
                        else:
                            temps_after_physics = entity.get_particles_temp()
                            dT_induction = entity.get_particles_dT_induction()
                            dT_conv = entity.get_particles_dT_conv()
                            dT_rad = entity.get_particles_dT_rad()
                            dT_bulk = entity.get_particles_dT_bulk()
                            dT_diffusion = entity.get_particles_dT_diffusion()
                            dT_contact = entity.get_particles_dT_contact()
                            dT_adiabatic = entity.get_particles_dT_adiabatic()

                        t = temps_after_physics.float()

                        # Total thermal energy: E = Σ(m_real * Cp * T)
                        particle_mass_scaled = self.env.scene.sim.mpm_solver.particles_info[0].mass
                        particle_mass = particle_mass_scaled / self.env.scene.sim.mpm_solver._particle_volume_scale
                        from agforge.thermal import get_steel_cp_torch

                        cp_tensor = get_steel_cp_torch(t)

                        all_dT_names = [
                            "Induction",
                            "AirConv",
                            "Radiation",
                            "FixedEnd",
                            "Diffusion",
                            "Contact",
                            "Adiabatic",
                        ]
                        all_dT_tensors = [
                            dT_induction.float().reshape(-1),
                            dT_conv.float().reshape(-1),
                            dT_rad.float().reshape(-1),
                            dT_bulk.float().reshape(-1),
                            dT_diffusion.float().reshape(-1),
                            dT_contact.float().reshape(-1),
                            dT_adiabatic.float().reshape(-1),
                        ]

                    def W_str(watts):
                        if abs(watts) > 1e6:
                            return f"{watts/1e6:+.1f}MW"
                        if abs(watts) > 1e3:
                            return f"{watts/1e3:+.1f}kW"
                        return f"{watts:+.0f}W"

                    with self._profile("teleop_telemetry_aggregate"):
                        log_parts = []

                        if len(all_dT_tensors) > 0:
                            dTs = torch.stack(all_dT_tensors) # shape: [N_cats, N_particles]
                            mask = dTs.abs() > 1e-6

                            # 1. Parallel mathematical reductions
                            n_heated = mask.sum(dim=1)
                            energy_sums = (dTs * cp_tensor).sum(dim=1)

                            safe_n = n_heated.clamp(min=1).float()
                            means = (dTs * mask).sum(dim=1) / safe_n

                            mins = dTs.amin(dim=1)
                            maxes = dTs.amax(dim=1)
                            pks = torch.where(means < 0, mins, maxes)

                            # 2. Gather global temperature bounds (and energy sum)
                            global_tensor = torch.stack([t.mean(), t.min(), t.max(), t.std(), (t * cp_tensor).sum()])

                            # 3. Exactly ONE PCIe cross-bus sync point!
                            all_stats = torch.cat([global_tensor, n_heated.float(), means, pks, energy_sums]).cpu().numpy()

                            # 4. Telemetry string generation
                            avg_t, min_t, max_t, std_t, sum_t = all_stats[:5]
                            total_energy_kJ = (particle_mass * sum_t) / 1000.0

                            log_parts.append(f"♨️ THERMAL │ Avg: {avg_t:.1f}K Min: {min_t:.1f}K Max: {max_t:.1f}K σ: {std_t:.1f}K │ E: {total_energy_kJ:.1f}kJ")

                            num_cats = len(all_dT_names)
                            dt_sim = self.env.scene.sim.dt

                            n_arr = all_stats[5:5+num_cats]
                            mean_arr = all_stats[5+num_cats:5+2*num_cats]
                            pk_arr = all_stats[5+2*num_cats:5+3*num_cats]
                            sum_arr = all_stats[5+3*num_cats:5+4*num_cats]

                            for i, name in enumerate(all_dT_names):
                                if n_arr[i] > 0:
                                    energy_W = (particle_mass * sum_arr[i]) / dt_sim
                                    # Format strings exactly as before
                                    log_parts.append(f"\n  {name}: {mean_arr[i]:+.2f}K avg, {pk_arr[i]:+.2f}K pk ({W_str(energy_W)})")

                        if log_parts:
                            gs.logger.info("".join(log_parts))
                except Exception as e:
                    gs.logger.warning(f"Thermal telemetry failed: {e}")

        # --- Diagnostic Telemetry ---
        with self._profile("teleop_diagnostics"):
            # Log diagnostics every 50 frames
            if self._physics_step_counter > 0 and self._physics_step_counter % 50 == 0:
                current_time = time.time()
                
                # Incremental calculation
                elapsed_real_time = current_time - self._diag_last_real_time
                steps_taken = self._physics_step_counter - self._diag_last_step
                
                # Cumulative calculation
                cumulative_real_time = current_time - self._diag_start_real_time
                
                if elapsed_real_time > 0 and steps_taken > 0 and cumulative_real_time > 0:
                    dt = self.env.scene.sim.dt
                    substeps = getattr(self.env.scene.sim, '_substeps', 1)
                    
                    # Incremental metrics
                    steps_per_sec = steps_taken / elapsed_real_time
                    substeps_per_sec = steps_per_sec * substeps
                    sim_time_passed = steps_taken * dt
                    time_ratio = sim_time_passed / elapsed_real_time
                    
                    # Cumulative metrics
                    cumulative_sim_time = self._physics_step_counter * dt
                    cumulative_ratio = cumulative_sim_time / cumulative_real_time
                    
                    gs.logger.info(f"⏱️ DIAGNOSTICS (Incremental) │ Real: {elapsed_real_time:.2f}s │ Sim: {sim_time_passed:.4f}s │ Ratio: {time_ratio:,.2f}x │ Speed: {steps_per_sec:.1f} step/s ({substeps_per_sec:.1f} sub/s)")
                    gs.logger.info(f"📈 DIAGNOSTICS (Cumulative)  │ Real: {cumulative_real_time:.2f}s │ Sim: {cumulative_sim_time:.4f}s │ Ratio: {cumulative_ratio:,.2f}x │ dt: {dt}")
                    
                self._diag_last_real_time = current_time
                self._diag_last_step = self._physics_step_counter

        # 4b. Stability Check (separate from physics for accurate profiling)
        with self._profile("teleop_stability"):
            if needs_check:
                try:
                    if physics_failed:
                        raise SimulationStabilityError("Physics step threw an exception")
                    with self._profile("teleop_stability_check"):
                        self._check_stability()
                except SimulationStabilityError as e:
                    gs.logger.error(f"CRITICAL STABILITY FAILURE: {e}")
                    if safety and safety.auto_reset:
                        gs.logger.warning(">>> Auto-Undo Triggered by System Protection (Reverting to Checkpoint) <<<")
                        if len(self.checkpoints) > 0:
                            await self.load_checkpoint()
                        else:
                            gs.logger.warning("No checkpoint available for undo. Forcing full reset.")
                            await self.reset_simulation()
                        return
                    elif physics_failed:
                        return
                    else:
                        raise

        # 4c. Live unified surface mesh during strikes (high-res MC, same cadence as legacy visual recon).
        _live_stages = (StrikeState.PRESSING, StrikeState.HOLDING, StrikeState.RELEASE)
        if self._uses_unified_surface_mesh() and self.strike_state in _live_stages:
            with self._profile("teleop_unified_recon"):
                self.physics_mesher.update_live(should_reconstruct=True, is_deforming=True)

        # 5. Render Update — sync rigid/MPM visual state every step; heavy overlays only when capped.
        if self.env.scene.visualizer:
            self.env.scene.visualizer.update_visual_states(force_render=True)

        if self.env.scene.visualizer and _will_render:
            with self._profile("teleop_render"):
                if self._mesh_overlay is not None:
                    perf = self._perf()
                    stride = max(1, int(getattr(perf, "mesh_overlay_sync_stride", 1) if perf else 1))
                    _live_stages_overlay = (StrikeState.PRESSING, StrikeState.HOLDING, StrikeState.RELEASE)
                    in_live_strike = self.strike_state in _live_stages_overlay
                    if not in_live_strike or (self._physics_step_counter % stride == 0):
                        with self._profile("teleop_render_mesh_overlay_sync"):
                            self._mesh_overlay.sync_from_controller(self)
                if self._temp_particle_renderer is not None:
                    with self._profile("teleop_render_particle_sync"):
                        self._temp_particle_renderer.prepare_render_frame(self.env)
                with self._profile("teleop_render_visualizer_update"):
                    self.env.scene.visualizer.update(force=False, auto=True)

        # 6. Record Data Frame (only if actively striking)
        if self.strike_state != StrikeState.IDLE:
            with self._profile("teleop_record"):
                with self._profile("teleop_record_gpu_pull"):
                    entity = self.env.mpm_entity
                    if hasattr(entity, "get_particles_record_bundle"):
                        particles_pos, particles_vel, particles_temp_t, particles_F = entity.get_particles_record_bundle()
                        if particles_F.ndim == 4:
                            particles_F = particles_F[0]
                        particles_temp = particles_temp_t.detach().cpu().numpy().reshape(-1)
                        with self._profile("teleop_record_det_f"):
                            particles_detF = (
                                torch.linalg.det(particles_F.reshape(-1, 3, 3))
                                .detach()
                                .cpu()
                                .numpy()
                                .astype(np.float32)
                            )
                    elif hasattr(entity, "get_particles_temp"):
                        particles_pos = entity.get_particles_pos()
                        particles_vel = entity.get_particles_vel()
                        particles_temp = entity.get_particles_temp().cpu().numpy().reshape(-1)
                        particles_F = entity.get_particles_F()
                        if particles_F.ndim == 4:
                            particles_F = particles_F[0]
                        with self._profile("teleop_record_det_f"):
                            particles_detF = torch.linalg.det(particles_F).detach().cpu().numpy().astype(np.float32)
                    else:
                        particles_pos = entity.get_particles_pos()
                        particles_vel = entity.get_particles_vel()
                        particles_temp = np.zeros(particles_pos.shape[-2], dtype=np.float32)
                        particles_detF = np.ones(particles_pos.shape[-2], dtype=np.float32)

                    with self._profile("teleop_record_resistance_pull"):
                        force_L, force_R = self.robot.get_resistance_forces()

                    # Get base particle volume for absolute volume calculations
                    p_vol = 0.0
                    if hasattr(self.env.scene.sim, 'mpm_solver') and hasattr(self.env.scene.sim.mpm_solver, 'particle_volume_real'):
                        p_vol = self.env.scene.sim.mpm_solver.particle_volume_real

                with self._profile("teleop_record_append"):
                    self.recorder.record_frame(
                        particles_pos=particles_pos,
                        particles_vel=particles_vel,
                        particles_temp=particles_temp,
                        particles_detF=particles_detF,
                        particle_vol=p_vol,
                        qpos=self.env.robot.entity.get_qpos(),
                        force_L=force_L,
                        force_R=force_R,
                        dof_cmd=self._vel_cmd
                    )

    def _profile(self, name):
        # Fix for Pydantic model access
        teleop_opts = self.env.scene.profiling_options.configs.teleop
        opt_name = name.replace("teleop_", "")
        if getattr(teleop_opts, opt_name, False):
            return self.env.scene.profiling_options.profiler.time(name)
        return contextlib.suppress()

    def _compute_vertex_temps_for_io(self, cached: dict) -> np.ndarray:
        """kNN vertex temperatures using cached mesh + particle positions."""
        particles = cached["particles"]
        vertices_raw = cached["vertices"]
        with self._profile("teleop_io_vertex_temp_prep"):
            particles_temp_tensor = self.env.mpm_entity.get_particles_temp().reshape(-1)
            mapping_parts_np = self.reconstructor._cached_particles
            if mapping_parts_np is None:
                if isinstance(particles, torch.Tensor):
                    mapping_parts_np = particles.detach().cpu().numpy().squeeze()
                else:
                    mapping_parts_np = np.asarray(particles).squeeze()
            if mapping_parts_np.shape[0] != particles_temp_tensor.shape[0]:
                active_mask = self.env.mpm_entity.get_particles_active(envs_idx=0).squeeze(0)
                particles_temp = particles_temp_tensor[active_mask]
            else:
                particles_temp = particles_temp_tensor
            if isinstance(vertices_raw, torch.Tensor):
                verts_np = vertices_raw.detach().cpu().numpy()
            else:
                verts_np = np.asarray(vertices_raw)
        with self._profile("teleop_io_kdtree_map"):
            return self._map_particle_temps_to_vertices(
                particles_temp, mapping_parts_np, verts_np
            )

    async def update_and_get_recon_data(self, include_vertex_temps: bool | None = None):
        """Updates reconstruction and returns data for visualization/IO."""
        self._io_frame_counter += 1
        if include_vertex_temps is None:
            include_vertex_temps = bool(getattr(self, "thermal_enabled", False))

        want_vertex_temps = bool(include_vertex_temps) and self._should_compute_vertex_temps_io()
        if not self._should_refresh_mesh_io() and self._io_mesh_cache is not None:
            cached = self._io_mesh_cache
            if want_vertex_temps:
                vertices_temp = self._compute_vertex_temps_for_io(cached)
                cached["vertices_temp"] = vertices_temp
                cached["vertex_temp_frame"] = self._io_frame_counter
            elif include_vertex_temps:
                vertices_temp = cached.get("vertices_temp")
                if vertices_temp is None:
                    vertices_temp = cached["vertices_temp_empty"]
            else:
                vertices_temp = cached["vertices_temp_empty"]
            return (
                cached["vertices"],
                cached["triangles"],
                cached["particles"],
                vertices_temp,
            )

        with self._profile("teleop_recon"):
            allowed_stages = (StrikeState.PRESSING, StrikeState.HOLDING, StrikeState.RELEASE)
            should_reconstruct = self.strike_state in allowed_stages

            if should_reconstruct and not self._uses_unified_surface_mesh():
                with self._profile("teleop_recon_update"):
                    self.reconstructor.update(should_reconstruct, is_deforming=True)
            elif should_reconstruct and self._uses_unified_surface_mesh():
                # Mesh already updated in step_simulation via physics_mesher.update_live().
                pass
            if should_reconstruct:
                with self._profile("teleop_recon_get_particles"):
                    particles = self.reconstructor.get_active_particle_cache()
                    if particles is None:
                        particles = self.env.mpm_entity.get_particles_pos(envs_idx=0).squeeze(0)
            else:
                with self._profile("teleop_recon_get_particles"):
                    particles = self.env.mpm_entity.get_particles_pos(envs_idx=0).squeeze(0)

            with self._profile("teleop_recon_transform_particles"):
                points = self._apply_transformation(particles)
                
                # OPTIMIZATION: Use GPU tensor for vertices if available
                if hasattr(self.reconstructor, 'reconstructed_vertices_tensor') and self.reconstructor.reconstructed_vertices_tensor is not None:
                     vertices_raw = self.reconstructor.reconstructed_vertices_tensor
                else:
                     vertices_raw = self.reconstructor.reconstructed_mesh.vertices
                     
            with self._profile("teleop_recon_transform_vertices"):
                vertices = self._apply_transformation(vertices_raw)
            
            with self._profile("teleop_recon_faces_copy"):
                triangles = self.reconstructor.reconstructed_mesh.faces.copy()
            # Winding reversal is now handled by Unity after axis conversion

            n_verts = int(vertices_raw.shape[0]) if hasattr(vertices_raw, "shape") else 0
            if (
                include_vertex_temps
                and not want_vertex_temps
                and not self._can_reuse_cached_vertex_temps(n_verts)
            ):
                # Live MC during strikes changes vertex count most frames; throttling
                # would send zeros when the cached temp array length no longer matches.
                want_vertex_temps = True

            # Extract and spatial-interpolate thermal data to vertices
            if (
                want_vertex_temps
                and hasattr(self.env.scene.sim.mpm_solver, 'particles')
                and hasattr(self.env.scene.sim.mpm_solver.particles, 'temp')
            ):
                with self._profile("teleop_io_vertex_temp_prep"):
                    particles_temp_tensor = self.env.mpm_entity.get_particles_temp().reshape(-1)

                    mapping_parts_np = self.reconstructor._cached_particles
                    if mapping_parts_np is None:
                        if isinstance(particles, torch.Tensor):
                            mapping_parts_np = particles.detach().cpu().numpy().squeeze()
                        else:
                            mapping_parts_np = np.asarray(particles).squeeze()

                    if mapping_parts_np.shape[0] != particles_temp_tensor.shape[0]:
                        active_mask = self.env.mpm_entity.get_particles_active(envs_idx=0).squeeze(0)
                        particles_temp = particles_temp_tensor[active_mask]
                    else:
                        particles_temp = particles_temp_tensor

                    if isinstance(vertices_raw, torch.Tensor):
                        verts_np = vertices_raw.detach().cpu().numpy()
                    else:
                        verts_np = np.asarray(vertices_raw)

                with self._profile("teleop_io_kdtree_map"):
                    vertices_temp = self._map_particle_temps_to_vertices(
                        particles_temp, mapping_parts_np, verts_np
                    )
            else:
                vertices_temp = np.zeros(vertices_raw.shape[0] if hasattr(vertices_raw, 'shape') else 0, dtype=np.float32)
                if (
                    include_vertex_temps
                    and not want_vertex_temps
                    and self._io_mesh_cache is not None
                    and self._io_mesh_cache.get("vertices_temp") is not None
                ):
                    prev = self._io_mesh_cache["vertices_temp"]
                    if hasattr(vertices_raw, "shape") and len(prev) == vertices_raw.shape[0]:
                        vertices_temp = prev

            empty_temps = np.zeros(vertices_raw.shape[0] if hasattr(vertices_raw, 'shape') else 0, dtype=np.float32)
            # When vertex temps are throttled (want_vertex_temps=False), vertices_temp may
            # still hold the previous frame's kNN map — keep/send that instead of zeros.
            out_vertex_temps = vertices_temp if include_vertex_temps else empty_temps
            self._io_mesh_cache = {
                "vertices": vertices,
                "triangles": triangles,
                "particles": points,
                "vertices_temp": out_vertex_temps if include_vertex_temps else None,
                "vertices_temp_empty": empty_temps,
                "vertex_temp_frame": self._io_frame_counter,
            }
            
            return vertices, triangles, points, out_vertex_temps

    def _apply_transformation(self, points):
        """Return raw physics-space coordinates. Unity handles all visual transforms."""
        if isinstance(points, torch.Tensor):
            return points.clone()
        return points.copy()

    @staticmethod
    def _deep_clone_sim_state(state):
        """Recursively clone all tensors in a SimState to fully isolate GPU memory.
        
        .detach() (called by serializable()) does NOT copy data — it only
        disconnects from autograd. Without cloning, checkpoint tensors can alias
        the simulator's internal buffers, causing silent corruption as more
        simulation steps overwrite that memory.
        """
        for solver_state in state.solvers_state:
            if solver_state is None:
                continue
            for attr_name in list(vars(solver_state).keys()):
                val = getattr(solver_state, attr_name)
                if isinstance(val, torch.Tensor):
                    setattr(solver_state, attr_name, val.clone())
                elif isinstance(val, list):
                    # ToolSolverState has a list of entity states
                    for entity_state in val:
                        if entity_state is None:
                            continue
                        for eattr in list(vars(entity_state).keys()):
                            ev = getattr(entity_state, eattr)
                            if isinstance(ev, torch.Tensor):
                                setattr(entity_state, eattr, ev.clone())

    def _save_checkpoint_impl(self):
        with self._profile("teleop_checkpoint_save"):
            self._save_checkpoint_impl_inner()

    def _save_checkpoint_impl_inner(self):
        # Genesis SimState
        sim_state = self.env.scene.sim.get_state()
        sim_state.serializable() # Detach from autograd graph
        
        # CRITICAL: Deep-clone all tensors so the checkpoint is a fully
        # independent snapshot. Without this, .detach()'d tensors still alias
        # the simulator's GPU buffers and get silently corrupted by future
        # simulation steps. This is the root cause of the "5+ strikes" bug.
        self._deep_clone_sim_state(sim_state)
        
        # Clear the simulator's gradient-tracking cache
        if hasattr(self.env.scene.sim, '_queried_states'):
            self.env.scene.sim._queried_states.clear()

        ckpt = {
            'sim_state': sim_state,
            'strike_state': self.strike_state,
            'qpos': self.qpos.clone(),
            'recon_state': self.reconstructor.get_state(),
            'thermal_enabled': self.thermal_enabled,
            'heating_power': self.heating_power,
            'skin_depth': self.skin_depth
        }
            
        self.checkpoints.append(ckpt)
        if len(self.checkpoints) > MAX_CHECKPOINTS:
            self.checkpoints.pop(0)
        mesh_verts = len(self.reconstructor.reconstructed_mesh.vertices) if self.reconstructor.reconstructed_mesh.vertices is not None else 0
        gs.logger.info(f"Checkpoint saved (stack={len(self.checkpoints)}, mesh_verts={mesh_verts})")

    async def save_checkpoint(self, replace_last=False):
        async with self.lock:
            if replace_last and len(self.checkpoints) > 0:
                self.checkpoints.pop()
            self._save_checkpoint_impl()

    async def load_checkpoint(self):
        async with self.lock:
            if len(self.checkpoints) < 2:
                gs.logger.warning("No previous checkpoint to undo to")
                return

            self.checkpoints.pop()
            ckpt = self.checkpoints[-1]
            
            self.env.scene.sim.reset(ckpt['sim_state'])
            
            import quadrants as qd  # type: ignore
            qd.sync()
            if hasattr(self.env.scene.sim.mpm_solver, 'update_render_fields'):
                self.env.scene.sim.mpm_solver.update_render_fields()
            else:
                self.env.scene.visualizer.update_visual_states()
            
            self.strike_state = StrikeState.IDLE
            self.contact_L = False
            self.contact_R = False
            self.contact_width = 0.0
            
            # Use the checkpoint's own qpos — the authoritative robot position
            # at save time. Do NOT query the physics engine here
            strike_qpos = ckpt['qpos'].clone()
            
            # Preserve current Unity slider position and rotation so the induction coil
            # doesn't snap away from the Unity visual when we return to IDLE.
            current_slider_x = self.qpos[0, 0].item() if self.qpos is not None else None
            current_hinge_qpos = self.qpos[0, 1].item() if self.qpos is not None else None
            
            self.qpos = strike_qpos.clone()
            
            # Open the grippers (DOFs 2-3) — the checkpoint may have been taken
            # mid-strike when grippers were closed.
            self.qpos[:, 2] = self.gripper_open_pos
            self.qpos[:, 3] = self.gripper_open_pos
            
            # Restore the slider position and rotation for IDLE mode
            if current_slider_x is not None:
                self.qpos[0, 0] = current_slider_x
            if current_hinge_qpos is not None:
                self.qpos[0, 1] = current_hinge_qpos
            
            # Zero all DOF velocities so residual momentum from the pre-undo
            # state doesn't carry over and kick the robot on the first frame.
            self.robot.entity.zero_all_dofs_velocity()
            
            # Teleport ALL DOFs to the authoritative position.
            self.robot.set_control_mode("TELEPORT")
            self.robot.apply_action(self.qpos)
            self._mark_qpos_applied(self.qpos)
            self._cached_slider_x = float(self.qpos[0, 0].item())
            self._last_induction_key = None

            self._invalidate_unity_vertex_temp_cache()
            if self._uses_unified_surface_mesh():
                self.reconstructor.reset()
            elif 'recon_state' in ckpt:
                self.reconstructor.set_state(ckpt['recon_state'])
            else:
                self.reconstructor.reset()
                self.reconstructor.create_reconstructed_mesh()

            self.thermal_enabled = ckpt.get('thermal_enabled', False)
            self.heating_power = ckpt.get('heating_power', self.env.cfg.heating_power)
            self.skin_depth = ckpt.get('skin_depth', self.env.cfg.skin_depth)
            self._invalidate_induction_params_cache()

            # Particle positions changed on restore — rebuild surface mesh (+ SDF if heating).
            if self._needs_surface_rebuild():
                upload_sdf = self.thermal_enabled and self.heater is not None
                self.rebuild_physics_induction(upload_sdf=upload_sdf)

            # Trigger mesh data send to client (without physics step)
            self.pending_mesh_send = True
            
            # Ensure the simulation loop sends multiple consecutive frames so Unity
            # reliably picks up the undo result, even if the first frame is missed.
            self.stabilization_steps = max(self.stabilization_steps, 5)

            # Suppress stability checks for a grace period so residual elastic
            # energy in the restored state doesn't immediately trigger another undo.
            safety = getattr(self.env.cfg, 'safety', None)
            grace = safety.check_interval * 3 if safety else 30
            self._stability_grace_steps = grace
            
            mesh_verts = len(self.reconstructor.reconstructed_mesh.vertices) if self.reconstructor.reconstructed_mesh.vertices is not None else 0
            
            self.recorder.handle_undo()
            
            gs.logger.info(f"Undo complete (stack={len(self.checkpoints)}, mesh_verts={mesh_verts}, grace={grace})")

    def _check_stability(self):
        """
        Zero-Overhead Tier-1 Stability Intercept.
        Runs boolean threshold reductions purely on GPU to catch mid-blowups
        without stalling the simulation pipeline.
        """
        safety = getattr(self.env.cfg, 'safety', None)
        if not safety or not getattr(safety, "enabled", True):
            return

        entity = self.env.mpm_entity
        if hasattr(entity, "get_particles_stability_bundle"):
            pos, vels = entity.get_particles_stability_bundle()
        else:
            vels = entity.get_particles_vel()
            pos = entity.get_particles_pos()
        
        # Pull temperatures if available
        temp = None
        if hasattr(self.env.scene.sim.mpm_solver, 'particles') and hasattr(self.env.scene.sim.mpm_solver.particles, 'temp'):
            temp = self.env.scene.sim.mpm_solver.particles.temp.to_torch(device=pos.device)

        # 1. NaN checks
        has_nan = torch.isnan(vels).any() | torch.isnan(pos).any()
        if temp is not None: 
            has_nan |= torch.isnan(temp).any()

        # 2. Mid-Blowup Thresholds
        has_out_of_bounds = torch.tensor(False, device=pos.device)
        has_super_velocity = torch.tensor(False, device=pos.device)
        has_thermal_detonation = torch.tensor(False, device=pos.device)

        if getattr(safety, 'strict_tracing_enabled', False):
            # Compute physical boundaries (95% of config limits)
            upper_bounds = torch.tensor(self.env.cfg.mpm.upper_bound, device=pos.device) * 0.95
            lower_bounds = torch.tensor(self.env.cfg.mpm.lower_bound, device=pos.device) * 0.95
            
            # Use strict inequality on specific axes (clamp limits)
            has_out_of_bounds = (pos > upper_bounds).any() | (pos < lower_bounds).any()
            has_super_velocity = (vels.abs() > safety.max_particle_velocity).any()
            
            if temp is not None:
                has_thermal_detonation = (temp > safety.max_temperature).any() | (temp < safety.min_temperature).any()


        # 3. BULK REDUCTION -> Single PCIe Bus Sync
        is_critical = has_nan | has_out_of_bounds | has_super_velocity | has_thermal_detonation

        if is_critical.item():
            gs.logger.error("🚨 TIER-1 STABILITY INTERCEPT TRIGGERED! Running Forensic Trace... 🚨")
            
            # --- Field Isolation Sequence ---
            cause = []
            if has_nan.item(): cause.append("NaN Detected")
            if has_out_of_bounds.item(): cause.append("Grid Bounds Exceeded")
            if has_super_velocity.item(): cause.append(f"Supersonic Velocity (>{safety.max_particle_velocity}m/s)")
            if has_thermal_detonation.item(): cause.append("Thermal Detonation")
            
            gs.logger.error(f"-> Primary Diagnostic Triggers: {', '.join(cause)}")
            
            # --- Ground Zero Profiling ---
            # We download the fields to CPU to find the exact particle
            pos_cpu = pos.cpu()
            vels_cpu = vels.cpu()
            n_particles = pos_cpu.shape[1] if pos_cpu.dim() >= 2 else pos_cpu.shape[0]
            t_cpu = (
                _per_particle_field_cpu(temp, n_particles)
                if temp is not None
                else None
            )
            
            # Find the indices of the failing particles based on the triggers
            mask = torch.zeros(n_particles, dtype=torch.bool, device="cpu")
            
            if has_nan.item():
                if vels_cpu.dim() >= 2:
                    mask |= torch.isnan(vels_cpu).any(dim=-1).reshape(-1)[:n_particles]
                    mask |= torch.isnan(pos_cpu).any(dim=-1).reshape(-1)[:n_particles]
                else:
                    mask |= torch.isnan(vels_cpu).reshape(-1)[:n_particles]
                    mask |= torch.isnan(pos_cpu).reshape(-1)[:n_particles]
                if t_cpu is not None:
                    mask |= torch.isnan(t_cpu)
                
            if has_out_of_bounds.item():
                ub = upper_bounds.cpu()
                lb = lower_bounds.cpu()
                if pos_cpu.dim() >= 2:
                    mask |= (pos_cpu > ub).any(dim=-1).reshape(-1)[:n_particles]
                    mask |= (pos_cpu < lb).any(dim=-1).reshape(-1)[:n_particles]
                
            if has_super_velocity.item():
                if vels_cpu.dim() >= 2:
                    mask |= (vels_cpu.abs() > safety.max_particle_velocity).any(dim=-1).reshape(-1)[:n_particles]
                
            if has_thermal_detonation.item() and t_cpu is not None:
                mask |= (t_cpu > safety.max_temperature) | (t_cpu < safety.min_temperature)
                
            bad_indices = torch.where(mask)[0]
            
            if len(bad_indices) > 0:
                idx = bad_indices[0].item() # Take the first violating particle
                if pos_cpu.dim() >= 2:
                    p_pos = pos_cpu[0, idx].tolist()
                    p_vel = vels_cpu[0, idx].tolist()
                else:
                    p_pos = pos_cpu[idx].tolist()
                    p_vel = vels_cpu[idx].tolist()
                
                msg = f"\n[GROUND ZERO PROFILING]\n"
                msg += f"Particle Index: {idx}\n"
                msg += f"Position (X,Y,Z): [{p_pos[0]:.4f}, {p_pos[1]:.4f}, {p_pos[2]:.4f}]\n"
                msg += f"Velocity (X,Y,Z): [{p_vel[0]:.4f}, {p_vel[1]:.4f}, {p_vel[2]:.4f}]\n"
                
                if t_cpu is not None:
                    p_temp = t_cpu[idx].item()
                    msg += f"Temperature: {p_temp:.2f} K\n"

                # Pull the SVD Tensor from the solver
                try:
                    solver = self.env.scene.sim.mpm_solver
                    S_np = solver.particles.S.to_numpy()
                    C_np = solver.particles.C.to_numpy()
                    
                    if len(S_np.shape) == 5:
                        s_f0 = S_np[0, idx, 0].flatten().tolist()
                        s_f1 = S_np[1, idx, 0].flatten().tolist()
                        msg += f"S Tensor [Frame 0]: {s_f0}\n"
                        msg += f"S Tensor [Frame 1]: {s_f1}\n"
                        c_f0 = C_np[0, idx, 0].flatten().tolist()
                        msg += f"C Tensor [Frame 0]: {c_f0}\n"
                    elif len(S_np.shape) == 4:
                        s_f0 = S_np[idx, 0].flatten().tolist()
                        msg += f"S Tensor: {s_f0}\n"
                        c_f0 = C_np[idx, 0].flatten().tolist()
                        msg += f"C Tensor: {c_f0}\n"
                except Exception as e:
                    msg += f"Could not retrieve Tensors: {e}\n"
                    
                # Pull the CPIC flags to prove grid interference
                try:
                    cpic_np = self.env.scene.sim.coupler.cpic_flag.to_numpy()
                    if len(cpic_np.shape) == 5:
                        p_cpic = cpic_np[idx, :, :, :, 0].flatten().tolist()
                        msg += f"CPIC Interference Stencil [3x3x3]: {p_cpic}\n"
                    elif len(cpic_np.shape) == 4:
                        p_cpic = cpic_np[idx, :, :, :].flatten().tolist()
                        msg += f"CPIC Interference Stencil [3x3x3]: {p_cpic}\n"
                except Exception as e:
                    pass
                    
                gs.logger.error(msg)
            
            raise SimulationStabilityError(f"Diagnostic Triggers: {', '.join(cause)}")

    async def reset_simulation(self):
        async with self.lock:
            # Preserve current slider position and rotation
            current_slider_x = self.qpos[0, 0].item() if self.qpos is not None else None
            current_hinge_qpos = self.qpos[0, 1].item() if self.qpos is not None else None
            
            # Flush current episode if recording
            if getattr(self, 'recorder', None) and self.recorder.is_recording:
                self.recorder.flush_episode(success_flag=False, language_instruction="Episode interrupted by reset")
                
            self.env.reset()
            
            # Re-initialize room temperature (reset wipes custom states)
            try:
                if hasattr(self.env, 'mpm_entity') and hasattr(self.env.mpm_entity, 'get_particles_temp'):
                    current_temps = self.env.mpm_entity.get_particles_temp()
                    if current_temps is not None:
                        base_temps = torch.ones_like(current_temps) * 293.0
                        self.env.mpm_entity.set_particles_temp(base_temps)
                        gs.logger.info(f"Initialized room temp: {base_temps.shape}, mean: {base_temps.mean().item()} K")
            except Exception as e:
                gs.logger.error(f"Failed to set room temperature: {e}")

            import quadrants as qd  # type: ignore
            qd.sync()
            if hasattr(self.env.scene.sim.mpm_solver, 'update_render_fields'):
                self.env.scene.sim.mpm_solver.update_render_fields()
            else:
                self.env.scene.visualizer.update_visual_states()

            self.qpos = self.robot.entity.get_dofs_position()
            self.strike_state = StrikeState.IDLE
            self.contact_L = False
            self.contact_R = False
            self.checkpoints = []
            self.contact_width = 0.0
            
            # Zero residual velocities from previous strikes and ensure
            # grippers are open, then teleport ALL DOFs to the clean state.
            self.robot.entity.zero_all_dofs_velocity()
            self.qpos[:, 2] = self.gripper_open_pos
            self.qpos[:, 3] = self.gripper_open_pos
            
            # Restore the slider position and rotation so the coil doesn't snap away from the Unity visual
            if current_slider_x is not None:
                self.qpos[0, 0] = current_slider_x
            if current_hinge_qpos is not None:
                self.qpos[0, 1] = current_hinge_qpos
                
            self.robot.set_control_mode("TELEPORT")
            self.robot.apply_action(self.qpos)
            self._mark_qpos_applied(self.qpos)

            # --- CRITICAL FIX: Flush Ghost State ---
            # Run several frames of physics to fully resolve the teleportation
            # and flush out any residual collision/coupler forces from the broadphase.
            # If we don't do this, the warm-up sequence's massive contact force will persist
            # into the first strike attempt.
            for _ in range(5):
                self.env.scene.step(update_visualizer=False)
                if hasattr(self.env.scene.sim.coupler, 'clear_link_coupling_forces'):
                    self.env.scene.sim.coupler.clear_link_coupling_forces()

            self.reconstructor.reset()
            self._invalidate_unity_vertex_temp_cache()
            gs.logger.info("Initializing surface reconstruction...")
            recon_cfg = self.env.cfg.reconstruction
            if getattr(recon_cfg, "unified_mesh", True):
                if self.physics_mesher.rebuild():
                    gs.logger.info(
                        f"Surface mesh ready [{self.physics_mesher.backend_label}]: "
                        f"{len(self.physics_mesher.physics_mesh.vertices)} verts"
                    )
            else:
                self.reconstructor.create_reconstructed_mesh()
                if self.physics_mesher.rebuild():
                    gs.logger.info(
                        f"Physics mesh ready [{self.physics_mesher.backend_label}]: "
                        f"{len(self.physics_mesher.physics_mesh.vertices)} verts"
                    )
            if self._mesh_overlay is not None:
                self._mesh_overlay.sync_from_controller(self)

            from agforge.vis.temperature_particles import update_particle_color_display

            update_particle_color_display(self.env, physics_mesher=self.physics_mesher)

            self._save_checkpoint_impl()
            
            # Trigger mesh data send to client (without physics step)
            self.pending_mesh_send = True
            self.stabilization_steps = max(self.stabilization_steps, 5)

            # Suppress stability checks while the freshly reset state settles
            safety = getattr(self.env.cfg, 'safety', None)
            self._stability_grace_steps = safety.check_interval * 3 if safety else 30
            
            self.recorder.start_new_episode("continuous_forge")
            gs.logger.info("Simulation reset")
