import asyncio
import time
import enum
import torch
import numpy as np
import genesis as gs
import contextlib

from agforge.reconstruction import SurfaceReconstructor

class StrikeState(enum.Enum):
    IDLE = 0
    APPROACHING = 1
    HOLDING = 2
    PRESSING = 3
    RELEASE = 4

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
        self.reconstructor = SurfaceReconstructor(env)
        # Note: Reconstruction init mostly happens on demand or at start
        
        # Checkpointing
        self.checkpoints = []
        
        # Transformation constants for Unity (visualization)
        self._init_transforms()
        
        # Optimization: Pre-allocated tensors
        self._vel_cmd = torch.zeros(4, device=self.env.device)
        self._last_vel_cmd = torch.zeros(4, device=self.env.device)
        self._dofs_idx_local = torch.tensor([0, 1, 2, 3], device=self.env.device)
        self._force_next_apply = True

    def _init_gripper_limits(self):
        # We can re-use the XML generator logic or just hardcode if standard.
        # Since agforge_builder is where RobotXMLGenerator lives, we might need to import it
        # or just assume standard values if not critical. 
        # Better: Import it to be safe.
        from agforge.agforge_builder import RobotXMLGenerator
        xml_generator = RobotXMLGenerator(robot_cfg=self.env.cfg.robot)
        self.gripper_closed_pos = xml_generator.gripper_slide_range[1]
        self.gripper_open_pos = xml_generator.gripper_slide_range[0]

    def _init_transforms(self):
        # Constants from teleop_socket.py
        TRANSFORM_SCALE = 31.275
        TRANSFORM_HEIGHT_FACTOR = 0.375
        
        self.unity_translation = -(self.env.cfg.robot.cylinder_pos + np.array([-TRANSFORM_HEIGHT_FACTOR * self.env.cfg.robot.cylinder_height, 0, 0]))
        self.unity_scale = np.array((TRANSFORM_SCALE,) * 3)
        self.unity_translation_tensor = torch.tensor(self.unity_translation, dtype=torch.float32, device=self.env.device).view(1, 3)
        self.unity_scale_tensor = torch.tensor((TRANSFORM_SCALE,) * 3, dtype=torch.float32, device=self.env.device).view(1, 3)

    async def set_qpos(self, new_qpos):
        async with self.lock:
            # Only clamp slider/hinge, grippers managed by logic if striking
            new_qpos[:, :2] = torch.clamp(new_qpos[:, :2], self.dof_limits[0][:2], self.dof_limits[1][:2])
            self.qpos = new_qpos

    async def get_qpos(self):
        async with self.lock:
            return self.qpos.clone()

    async def trigger_strike(self, force_param):
        async with self.lock:
            if self.strike_state != StrikeState.IDLE:
                gs.logger.warning(f"Strike requested but already in {self.strike_state.name}")
                return
            
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
            self.robot.set_control_mode("VELOCITY_CONTROL")
            gs.logger.info(f"  Control mode -> VELOCITY_CONTROL")

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
                    force_L, force_R = self.robot.get_resistance_forces()
                
                with self._profile("logic_update_state"):
                    if not self.contact_L and force_L > contact_threshold:
                        self.contact_L = True
                        gs.logger.info(f"  Left gripper CONTACT (force={force_L:.4f})")
                        
                    if not self.contact_R and force_R > contact_threshold:
                        self.contact_R = True
                        gs.logger.info(f"  Right gripper CONTACT (force={force_R:.4f})")
                    
                    if VERBOSE_LOGGING and self.strike_step_count % LOG_EVERY_N_FRAMES == 0:
                        gs.logger.info(f"APPROACHING[{self.strike_step_count}]: F=[{force_L:.3f},{force_R:.3f}], contact=[{self.contact_L}, {self.contact_R}]")

                    self._vel_cmd.zero_()
                    self._vel_cmd[2] = 0.0 if self.contact_L else approach_speed
                    self._vel_cmd[3] = 0.0 if self.contact_R else approach_speed
                
                with self._profile("logic_apply_vel"):
                    self._apply_vel_dedup()
                
                with self._profile("logic_update_state"):
                    if self.contact_L and self.contact_R:
                        self.strike_state = StrikeState.PRESSING
                        self.stage_start_time = time.time()
                        
                        pos_L = self.robot.left_gripper.get_pos()
                        pos_R = self.robot.right_gripper.get_pos()
                        self.contact_width = torch.norm(pos_L - pos_R).item()
                        
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
                
                with self._profile("logic_update_state"):
                    pos_L = self.robot.left_gripper.get_pos()
                    pos_R = self.robot.right_gripper.get_pos()
                    current_width = torch.norm(pos_L - pos_R).item()
                    
                    if self.contact_width > 1e-6:
                        current_strain = (self.contact_width - current_width) / self.contact_width
                    else:
                        current_strain = 0.0
                    
                    # Warn about unusual conditions
                    if current_strain < -0.01:
                        gs.logger.warning(f"Negative strain detected ({current_strain:.4f}) - grippers moving apart?")
                    if force_L < 0 or force_R < 0:
                        gs.logger.warning(f"Negative force detected: F_L={force_L:.4f}, F_R={force_R:.4f}")
                    
                    elapsed_time = time.time() - self.stage_start_time

                    stop_reason = None
                    if current_strain >= target_strain:
                        stop_reason = "Target Strain"
                    elif force_L > max_force or force_R > max_force:
                        stop_reason = "Max Force"
                    elif elapsed_time > pressing_timeout:
                        stop_reason = "Timeout"
                    
                    if stop_reason:
                        gs.logger.info(f"Strike -> RELEASE ({stop_reason}, strain={current_strain:.4f}, steps={self.strike_step_count}, time={elapsed_time:.2f}s)")
                        self.strike_state = StrikeState.RELEASE
                        self.stage_start_time = time.time()
                        self._stop_motors()
                        return

                    imbalance = force_L - force_R
                    correction = imbalance * force_balance_gain
                    
                    v_L = max(0.0, pressing_speed - correction)
                    v_R = max(0.0, pressing_speed + correction)
                    
                    if VERBOSE_LOGGING and self.strike_step_count % LOG_EVERY_N_FRAMES == 0:
                        gs.logger.info(f"PRESSING[{self.strike_step_count}]: F=[{force_L:.3f},{force_R:.3f}] dF={imbalance:.4f}, v=[{v_L:.4f},{v_R:.4f}] corr={correction:.4f}, strain={current_strain:.3f}/{target_strain:.3f}")
                    
                    self._vel_cmd.zero_()
                    self._vel_cmd[2] = v_L
                    self._vel_cmd[3] = v_R
                
                with self._profile("logic_apply_vel"):
                    self._apply_vel_dedup()

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

                with self._profile("logic_update_state"):
                    imbalance = force_L - force_R
                    correction = imbalance * force_balance_gain
                    
                    v_open = -release_speed
                    v_L = v_open - correction
                    v_R = v_open + correction
                    
                    if VERBOSE_LOGGING and self.strike_step_count % LOG_EVERY_N_FRAMES == 0:
                        gs.logger.info(f"RELEASE[{self.strike_step_count}]: F=[{force_L:.3f},{force_R:.3f}] dF={imbalance:.4f}, thresh={contact_threshold:.4f}, v=[{v_L:.4f},{v_R:.4f}] corr={correction:.4f}")
                    
                    self._vel_cmd.zero_()
                    self._vel_cmd[2] = v_L
                    self._vel_cmd[3] = v_R
                
                with self._profile("logic_apply_vel"):
                    self._apply_vel_dedup()
                
                with self._profile("logic_update_state"):
                    if abs(force_L) < contact_threshold and abs(force_R) < contact_threshold:
                        # Calculate final stats
                        pos_L = self.robot.left_gripper.get_pos()
                        pos_R = self.robot.right_gripper.get_pos()
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

                        # Save checkpoint
                        await self.save_checkpoint()
                        
                        # Flag to ensure updated mesh is sent to client
                        self.pending_mesh_send = True
                        return

            # --- HOLDING STAGE ---
            elif self.strike_state == StrikeState.HOLDING:
                pass

    def _stop_motors(self):
        self._vel_cmd.zero_()
        self._apply_vel_dedup()

    def _apply_vel_dedup(self):
        """
        Applies velocity only if the command has changed from the last step.
        Reduces overhead of repeated identical calls (e.g. during APPROACHING).
        """
        if not self._force_next_apply and torch.equal(self._vel_cmd, self._last_vel_cmd):
            return

        self.robot.apply_velocity(self._vel_cmd, dofs_idx_local=self._dofs_idx_local)
        self._last_vel_cmd.copy_(self._vel_cmd)
        self._force_next_apply = False

    def _force_idle_reset(self):
         self._stop_motors()
         self._force_next_apply = True
         
         current_qpos = self.qpos.clone()
         current_qpos[:, 2] = self.gripper_open_pos
         current_qpos[:, 3] = self.gripper_open_pos
         
         self.robot.set_control_mode("TELEPORT")
         self.qpos = current_qpos
         self.robot.apply_action(current_qpos)
         
         self.strike_state = StrikeState.IDLE
         self.contact_L = False
         self.contact_R = False
         self.contact_width = 0.0

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
        
        # 1. Logic Update
        with self._profile("teleop_logic"):
            await self.update_logic()
        
        # Track steps during active strike
        if self.strike_state != StrikeState.IDLE:
            self.strike_step_count += 1

        # 2. Apply Actions (if not handled by strike logic, handle idle holding)
        if self.strike_state == StrikeState.IDLE or self.strike_state == StrikeState.HOLDING:
            qpos = await self.get_qpos()
            self.robot.apply_action(qpos)

        # 3. Clear Forces
        if hasattr(self.env.scene.sim.coupler, 'clear_link_coupling_forces'):
            self.env.scene.sim.coupler.clear_link_coupling_forces()

        # 4. Physics Step
        with self._profile("teleop_physics"):
            try:
                self.env.scene.step(update_visualizer=False)
            except Exception as e:
                gs.logger.warning(f"Physics step skipped due to instability: {e}")

        # 5. Render Update
        if self.env.scene.visualizer:
            with self._profile("teleop_render"):
                self.env.scene.visualizer.update(force=False, auto=True)

    def _profile(self, name):
        # Fix for Pydantic model access
        teleop_opts = self.env.scene.profiling_options.configs.teleop
        opt_name = name.replace("teleop_", "")
        if getattr(teleop_opts, opt_name, False):
            return self.env.scene.profiling_options.profiler.time(name)
        return contextlib.suppress()

    async def update_and_get_recon_data(self):
        """Updates reconstruction and returns data for visualization/IO."""
        with self._profile("teleop_recon"):
            allowed_stages = (StrikeState.PRESSING, StrikeState.RELEASE)
            should_reconstruct = self.strike_state in allowed_stages
            
            if should_reconstruct:
                with self._profile("teleop_recon_update"):
                    self.reconstructor.update(should_reconstruct)
                with self._profile("teleop_recon_get_particles"):
                    particles = self.reconstructor.get_active_particle_cache()
                    if particles is None:
                         particles = self.env.mpm_entity.get_particles_pos(envs_idx=0).squeeze(0)
            else:
                with self._profile("teleop_recon_get_particles"):
                    particles = self.env.mpm_entity.get_particles_pos(envs_idx=0).squeeze(0)

            with self._profile("teleop_recon_transform"):
                points = self._apply_transformation(particles)
                vertices = self._apply_transformation(self.reconstructor.reconstructed_mesh.vertices)
            
            triangles = self.reconstructor.reconstructed_mesh.faces
            
            return vertices, triangles, points

    def _apply_transformation(self, points):
        """Transform points to Unity space."""
        if isinstance(points, torch.Tensor):
            points = points + self.unity_translation_tensor
            points = points * self.unity_scale_tensor
            points[:, 0] *= -1 
        else: 
            points = points + self.unity_translation.reshape(1, 3)
            points = points * self.unity_scale.reshape(1, 3)
            points[:, 0] *= -1
        return points

    def _save_checkpoint_impl(self):
        # Genesis SimState
        sim_state = self.env.scene.sim.get_state()
        sim_state.serializable() # Ensure CPU friendly if needed
        
        # Clear specific internal cache if needed (from original code)
        if hasattr(self.env.scene.sim, '_queried_states'):
            self.env.scene.sim._queried_states.clear()

        ckpt = {
            'sim_state': sim_state,
            'strike_state': self.strike_state,
            'qpos': self.qpos.clone(),
            'recon_state': self.reconstructor.get_state()
        }
        self.checkpoints.append(ckpt)
        if len(self.checkpoints) > MAX_CHECKPOINTS:
            self.checkpoints.pop(0)
        mesh_verts = len(self.reconstructor.reconstructed_mesh.vertices) if self.reconstructor.reconstructed_mesh.vertices is not None else 0
        gs.logger.info(f"Checkpoint saved (stack={len(self.checkpoints)}, mesh_verts={mesh_verts})")

    async def save_checkpoint(self):
        async with self.lock:
            self._save_checkpoint_impl()

    async def load_checkpoint(self):
        async with self.lock:
            if len(self.checkpoints) < 2:
                gs.logger.warning("No previous checkpoint to undo to")
                return

            self.checkpoints.pop()
            ckpt = self.checkpoints[-1]
            
            self.env.scene.sim.reset(ckpt['sim_state'])
            
            import gstaichi as ti
            ti.sync()
            if hasattr(self.env.scene.sim.mpm_solver, 'update_render_fields'):
                self.env.scene.sim.mpm_solver.update_render_fields()
            else:
                self.env.scene.visualizer.update_visual_states()
            
            self.strike_state = StrikeState.IDLE
            self.contact_L = False
            self.contact_R = False
            self.contact_width = 0.0
            
            self.qpos = ckpt['qpos'].clone()
            self.qpos[:, 2] = self.gripper_open_pos
            self.qpos[:, 3] = self.gripper_open_pos
            
            self.robot.set_control_mode("TELEPORT")
            self.robot.apply_action(self.qpos)
            self.robot.apply_action(self.qpos) # Apply a few times to settle
            
            if 'recon_state' in ckpt:
                self.reconstructor.set_state(ckpt['recon_state'])
            else:
                self.reconstructor.reset()
                self.reconstructor.create_reconstructed_mesh()

            # Trigger mesh data send to client (without physics step)
            self.pending_mesh_send = True
            
            mesh_verts = len(self.reconstructor.reconstructed_mesh.vertices) if self.reconstructor.reconstructed_mesh.vertices is not None else 0
            gs.logger.info(f"Undo complete (stack={len(self.checkpoints)}, mesh_verts={mesh_verts})")

    async def reset_simulation(self):
        async with self.lock:
            self.env.reset()
            
            import gstaichi as ti
            ti.sync()
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
            self.reconstructor.reset()
            
            gs.logger.info("Initializing surface skinning...")
            self.reconstructor.create_reconstructed_mesh()
            self.reconstructor.init_skinning()
            
            self._save_checkpoint_impl()
            
            # Trigger mesh data send to client (without physics step)
            self.pending_mesh_send = True
            
            gs.logger.info("Simulation reset")
