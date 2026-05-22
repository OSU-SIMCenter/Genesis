import asyncio
import time
import enum
import torch
import numpy as np
import genesis as gs
import contextlib

from agforge.reconstruction import SurfaceReconstructor
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
            backend=recon_cfg.backend
        )
        self.reconstructor.recon_enabled = recon_cfg.enabled
        # Note: Reconstruction init mostly happens on demand or at start
        
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

        # Thermal state logic
        self.thermal_enabled = False
        self.heating_power = self.env.cfg.heating_power
        self.skin_depth = self.env.cfg.skin_depth
        self.heater = None

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
                gs.logger.info(f"Thermal ACTIVATED (power={self.heating_power}W)")
            else:
                gs.logger.info("Thermal FROZEN")

    def _apply_fixed_end_heat_sink(self):
        """Apply Dirichlet BC at the fixed end to model bulk billet conduction.

        Particles within the fixed region have their temperature lerped back
        toward ambient. The lerp strength increases toward the boundary:
        - At the inner edge (closest to free end): gentle pull (alpha ~ 0)
        - At the far end of the fixed region: full clamp (alpha ~ 1)

        This models heat conduction into the infinite thermal mass of the
        remaining billet beyond the simulated section.
        """
        T_ambient = 293.0

        pos = self.env.mpm_entity.get_particles_pos()    # [1, N, 3]
        temps = self.env.mpm_entity.get_particles_temp()  # [1, N]

        x_pos = pos[0, :, 0]  # [N] particle X positions

        # Parametrically lock the bounds to the initial cylinder dimensions
        # This prevents the clamp zone from dragging/shrinking when the hammer squishes the billet, 
        # and ignores the artificially oversized 'fixed_region_bounds' box.
        L = self.env.cfg.robot.cylinder_height
        cylinder_center_x = self.env.cfg.robot.cylinder_pos[0]
        x_max = cylinder_center_x + (L / 2.0)

        # 0% to 11% from fixed end -> 100% clamped.
        # 11% to 21% from fixed end -> Smooth linear fade to 0%.
        x_clamp = x_max - 0.11 * L
        x_fade = x_max - 0.21 * L
        
        # Calculate alpha:
        # > x_clamp : alpha = 1.0
        # < x_fade  : alpha = 0.0
        alpha = ((x_pos - x_fade) / (x_clamp - x_fade)).clamp(0.0, 1.0)

        # We need a true Dirichlet constraint: at alpha=1, temperature MUST be locked to a heat sink target.
        # We decouple the physical drain from visuals: we clamp to 1/3 of the ACTIVE ZONE average temp.
        active_mask = x_pos < x_fade
        active_temps = temps[0][active_mask]
        
        if active_temps.numel() > 0:
            active_avg = active_temps.mean().item()
        else:
            active_avg = T_ambient
            
        # Delta-T formulation: anchors at room temp, scales the gradient gap smoothly
        target_calc = T_ambient + (active_avg - T_ambient) * (1.0 / 2.0)
        
        # Floor safely to ambient. Note: we no longer cap the physical boundaries to 900K! 
        # The metal can reach 1200K natively; visualization limits are handled separately.
        target_clamp_temp = max(T_ambient, target_calc)

        new_temps = temps.clone()
        new_temps[0] = temps[0] * (1.0 - alpha) + target_clamp_temp * alpha
        self.env.mpm_entity.set_particles_temp(new_temps)
        
        # Return dT for telemetry
        return new_temps - temps

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
                        # force_L, force_R are now TENSORS
                        force_L, force_R = self.robot.get_resistance_forces()
                    
                    # Batch contact check sync: Combine predicates
                    # We need to know specific contacts to set velocity
                    # contacts = [L_hit, R_hit]
                    contacts_tensor = torch.stack([force_L > contact_threshold, force_R > contact_threshold])
                    
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
                        
                        pos_L = self.robot.left_gripper.get_pos()
                        pos_R = self.robot.right_gripper.get_pos()
                        # Keep as tensor for GPU strain calc
                        self.contact_width_tensor = torch.norm(pos_L - pos_R)
                        self.contact_width = self.contact_width_tensor.item() # Sync once for logging/logic
                        
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
                
                    pos_L = self.robot.left_gripper.get_pos()
                    pos_R = self.robot.right_gripper.get_pos()
                    # GPU calculation
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
                    
                    # Single Sync: Check if any stop condition is met
                    # We pull as int to see WHICH flag triggered if we wanted, or just any().item()
                    stop_any = stop_flags.any().item() or is_timeout
                    
                    if stop_any:
                        # Determine reason (Now we can sync values for logging since we are stopping)
                        if cond_strain.item(): stop_reason = "Target Strain"
                        elif cond_force.item(): stop_reason = "Max Force"
                        elif is_timeout: stop_reason = "Timeout"
                        else: stop_reason = "Unknown"
                        
                        strain_val = current_strain_tensor.item()
                        gs.logger.info(f"Strike -> RELEASE ({stop_reason}, strain={strain_val:.4f}, steps={self.strike_step_count}, time={elapsed_time:.2f}s)")
                        self.strike_state = StrikeState.RELEASE
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
                    
                    if is_free.item(): # 1 Sync
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

                        # Replace the pre-strike checkpoint (saved in trigger_strike)
                        # with this post-strike state so 1 undo = 1 strike reversal.
                        await self.save_checkpoint(replace_last=True)
                        
                        # Flag to ensure updated mesh is sent to client
                        self.pending_mesh_send = True
                        return

            # --- HOLDING STAGE ---
            elif self.strike_state == StrikeState.HOLDING:
                pass

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
        with self._profile("teleop_apply_action"):
            if self.strike_state == StrikeState.IDLE or self.strike_state == StrikeState.HOLDING:
                qpos = await self.get_qpos()
                self.robot.apply_action(qpos)

        # 3. Clear Forces
        with self._profile("teleop_clear_forces"):
            if hasattr(self.env.scene.sim.coupler, 'clear_link_coupling_forces'):
                self.env.scene.sim.coupler.clear_link_coupling_forces()

        # 3b. Thermodynamics
        frozen_temps_tensor = None
        _temps_before_heating = None
        _temps_after_heating = None
        _is_striking = self.strike_state not in (StrikeState.IDLE, StrikeState.HOLDING)
        if self.thermal_enabled:
            with self._profile("teleop_heating"):
                if self.heater is None:
                    from agforge.thermal import InductionHeater
                    self.heater = InductionHeater(
                        solver=self.env.scene.sim.mpm_solver, 
                        entity=self.env.mpm_entity, 
                        reconstructor=self.reconstructor
                    )
                
                if not _is_striking:
                    # Only do induction heating + snapshots when NOT in a strike
                    # (avoids 2 extra GPU syncs that serialize the pipeline during fast strike loops)
                    _temps_before_heating = self.env.mpm_entity.get_particles_temp().clone()
                    
                    # Ride the sidecar! Calculate absolute physics position dynamically from the sliding arm
                    current_slider_x = self.qpos[0, 0].item() if self.qpos is not None else 0.0
                    dynamic_coil_x = current_slider_x + self.env.cfg.robot.coil_offset_x
                    
                    # Hardcode Y to 0 and Z to the cylinder's world center
                    coil_center = [dynamic_coil_x, 0.0, self.env.cfg.robot.cylinder_pos[2]]
                    
                    # Use real-time physical dt for induction heating calculation
                    thermal_dt = self.env.scene.sim.dt * self.env.scene.sim.mpm_solver._thermal_time_scale

                    self.heater.step_heat(
                        thermal_dt, 
                        self.heating_power, 
                        self.skin_depth, 
                        coil_center, 
                        self.env.cfg.robot.coil_length / 2.0,
                        profile_ctx=self._profile
                    )
                    
                    # Snapshot after induction heating (before engine physics)
                    _temps_after_heating = self.env.mpm_entity.get_particles_temp().clone()
                else:
                    # During strikes: just snapshot pre-physics temps (1 GPU sync instead of 3)
                    _temps_after_heating = self.env.mpm_entity.get_particles_temp().clone()
        else:
            # Thermal Freezing: Disable natural diffusion by snapshotting temperatures before physics step
            if hasattr(self.env.scene.sim.mpm_solver, 'particles') and hasattr(self.env.scene.sim.mpm_solver.particles, 'temp'):
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

            # --- Fixed-End Heat Sink (Dirichlet BC) ---
            # Model heat conduction into the bulk billet by clamping particles
            # near the held end toward ambient temperature.
            self._dt_heat_sink = None
            if self.thermal_enabled and not physics_failed:
                self._dt_heat_sink = self._apply_fixed_end_heat_sink()

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

        # --- Thermal Telemetry ---
        # Log every 5th frame during strikes, and every 10th frame during idle/heating
        _telemetry_interval = 5 if _is_striking else 10
        with self._profile("teleop_thermal_telemetry"):
            if self.thermal_enabled and self._physics_step_counter % _telemetry_interval == 0:
                try:
                    temps_after_physics = self.env.mpm_entity.get_particles_temp()
                    t = temps_after_physics.float()
                    
                    # Total thermal energy: E = Σ(m_real * Cp * T)
                    particle_mass_scaled = self.env.scene.sim.mpm_solver.particles_info[0].mass
                    particle_mass = particle_mass_scaled / self.env.scene.sim.mpm_solver._particle_volume_scale
                    import torch
                    from agforge.thermal import get_steel_cp_numpy
                    cp_tensor = torch.tensor(get_steel_cp_numpy(t.cpu().numpy()), device=t.device)
                    n_particles = t.numel()
                    
                    import torch
                    all_dT_names = []
                    all_dT_tensors = []

                    if _temps_before_heating is not None and _temps_after_heating is not None:
                        all_dT_names.append("Induction")
                        all_dT_tensors.append((_temps_after_heating - _temps_before_heating).float().squeeze(-1).view(-1))

                    # --- Engine Mechanisms ---
                    all_dT_names.extend(["Convection", "Radiation", "Contact", "Adiabatic"])
                    all_dT_tensors.extend([
                        self.env.mpm_entity.get_particles_dT_conv().float().squeeze(-1).view(-1),
                        self.env.mpm_entity.get_particles_dT_rad().float().squeeze(-1).view(-1),
                        self.env.mpm_entity.get_particles_dT_contact().float().squeeze(-1).view(-1),
                        self.env.mpm_entity.get_particles_dT_adiabatic().float().squeeze(-1).view(-1)
                    ])
                    
                    if self._dt_heat_sink is not None:
                        all_dT_names.append("HeatSink")
                        all_dT_tensors.append(self._dt_heat_sink.float().squeeze(-1).view(-1))

                    def W_str(watts):
                        if abs(watts) > 1e6:
                            return f"{watts/1e6:+.1f}MW"
                        if abs(watts) > 1e3:
                            return f"{watts/1e3:+.1f}kW"
                        return f"{watts:+.0f}W"

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
        # - Always fires on physics exceptions (regardless of grace period)
        # - During active strikes: checks every step for fast detection
        # - During idle: checks every check_interval steps to reduce GPU sync overhead
        # - Suppressed during grace period after undo/reset (prevents cascading auto-undos
        #   from residual elastic energy in the restored state)
        safety = getattr(self.env.cfg, 'safety', None)

        with self._profile("teleop_stability"):
            if physics_failed:
                needs_check = True
            elif self._stability_grace_steps > 0:
                self._stability_grace_steps -= 1
                needs_check = False
            else:
                needs_check = False

            if needs_check:
                try:
                    if physics_failed:
                        raise SimulationStabilityError("Physics step threw an exception")
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

        # 5. Render Update
        if self.env.scene.visualizer:
            with self._profile("teleop_render"):
                self.env.scene.visualizer.update(force=False, auto=True)

        # 6. Record Data Frame (only if actively striking)
        if self.strike_state != StrikeState.IDLE:
            with self._profile("teleop_record"):
                particles_pos = self.env.mpm_entity.get_particles_pos()
                particles_vel = self.env.mpm_entity.get_particles_vel()
                
                # Check for thermal state
                if hasattr(self.env.scene.sim.mpm_solver, 'particles') and hasattr(self.env.scene.sim.mpm_solver.particles, 'temp'):
                    particles_temp = self.env.scene.sim.mpm_solver.particles.temp.to_numpy()[0, :, 0]
                else:
                    particles_temp = np.zeros(particles_pos.shape[1], dtype=np.float32)
                    
                force_L, force_R = self.robot.get_resistance_forces()
                self.recorder.record_frame(
                    particles_pos=particles_pos,
                    particles_vel=particles_vel,
                    particles_temp=particles_temp,
                    qpos=self.qpos,
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

    async def update_and_get_recon_data(self):
        """Updates reconstruction and returns data for visualization/IO."""
        with self._profile("teleop_recon"):
            allowed_stages = (StrikeState.PRESSING, StrikeState.RELEASE)
            should_reconstruct = self.strike_state in allowed_stages
            
            if should_reconstruct:
                with self._profile("teleop_recon_update"):
                    self.reconstructor.update(should_reconstruct, is_deforming=True)
                with self._profile("teleop_recon_get_particles"):
                    particles = self.reconstructor.get_active_particle_cache()
                    if particles is None:
                        particles = self.env.mpm_entity.get_particles_pos(envs_idx=0).squeeze(0)
            else:
                with self._profile("teleop_recon_get_particles"):
                    particles = self.env.mpm_entity.get_particles_pos(envs_idx=0).squeeze(0)

            with self._profile("teleop_recon_transform"):
                points = self._apply_transformation(particles)
                
                # OPTIMIZATION: Use GPU tensor for vertices if available
                if hasattr(self.reconstructor, 'reconstructed_vertices_tensor') and self.reconstructor.reconstructed_vertices_tensor is not None:
                     vertices_raw = self.reconstructor.reconstructed_vertices_tensor
                else:
                     vertices_raw = self.reconstructor.reconstructed_mesh.vertices
                     
                vertices = self._apply_transformation(vertices_raw)
            
            triangles = self.reconstructor.reconstructed_mesh.faces.copy()
            # Winding reversal is now handled by Unity after axis conversion
            
            # Extract and spatial-interpolate thermal data to vertices
            if hasattr(self.env.scene.sim.mpm_solver, 'particles') and hasattr(self.env.scene.sim.mpm_solver.particles, 'temp'):
                particles_temp = self.env.scene.sim.mpm_solver.particles.temp.to_numpy()[0, :, 0]
                
                import scipy.spatial
                parts_np = particles.cpu().numpy().squeeze() if isinstance(particles, torch.Tensor) else np.asarray(particles).squeeze()
                
                # --- VISUAL STATE OVERRIDE ---
                # We fade the visual temperatures of the 11-21% clamp zone down to 900K *before* the KD-Tree
                # so the renderer keeps the dummy handle black, without capping the actual physics engine.
                L_v = self.env.cfg.robot.cylinder_height
                cylinder_center_x = self.env.cfg.robot.cylinder_pos[0]
                x_max_v = cylinder_center_x + (L_v / 2.0)
                
                x_clamp_v = x_max_v - 0.11 * L_v
                x_fade_v = x_max_v - 0.21 * L_v
                
                alpha_vis = np.clip((parts_np[:, 0] - x_fade_v) / (x_clamp_v - x_fade_v), 0.0, 1.0)
                target_vis = np.minimum(particles_temp, 900.0)
                particles_temp = particles_temp * (1.0 - alpha_vis) + target_vis * alpha_vis
                
                if isinstance(vertices_raw, torch.Tensor):
                    verts_np = vertices_raw.cpu().numpy()
                else:
                    verts_np = vertices_raw
                    
                # Guard against degenerate mesh/particles
                if parts_np.shape[0] >= 3 and verts_np.shape[0] > 0:
                    tree = scipy.spatial.cKDTree(parts_np)
                    dists, indices = tree.query(verts_np, k=3)
                    
                    # Inverse distance weighting
                    dists = np.maximum(dists, 1e-6)  # Avoid div by zero
                    weights = 1.0 / dists
                    weight_sums = weights.sum(axis=1)
                    neighbor_temps = particles_temp[indices]
                    
                    vertices_temp = (neighbor_temps * weights).sum(axis=1) / weight_sums
                    vertices_temp = vertices_temp.astype(np.float32)
                else:
                    vertices_temp = np.zeros(verts_np.shape[0], dtype=np.float32)
            else:
                vertices_temp = np.zeros(vertices_raw.shape[0] if hasattr(vertices_raw, 'shape') else 0, dtype=np.float32)
            
            return vertices, triangles, points, vertices_temp

    def _apply_transformation(self, points):
        """Return raw physics-space coordinates. Unity handles all visual transforms."""
        if isinstance(points, torch.Tensor):
            return points.clone()
        return points.copy()

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
                
            self.thermal_enabled = ckpt.get('thermal_enabled', False)
            self.heating_power = ckpt.get('heating_power', 2000.0)
            self.skin_depth = ckpt.get('skin_depth', 0.02)

            # Trigger mesh data send to client (without physics step)
            self.pending_mesh_send = True

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
        if not safety:
            return

        vels = self.env.mpm_entity.get_particles_vel()
        pos = self.env.mpm_entity.get_particles_pos()
        
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
            t_cpu = temp.cpu() if temp is not None else None
            
            # Find the indices of the failing particles based on the triggers
            mask = torch.zeros(pos_cpu.shape[1], dtype=torch.bool, device='cpu')
            
            if has_nan.item():
                mask |= torch.isnan(vels_cpu).any(dim=-1).squeeze(0)
                mask |= torch.isnan(pos_cpu).any(dim=-1).squeeze(0)
                if temp is not None: mask |= torch.isnan(temp.cpu()).squeeze(-1).squeeze(0)
                
            if has_out_of_bounds.item():
                ub = upper_bounds.cpu()
                lb = lower_bounds.cpu()
                mask |= (pos_cpu > ub).any(dim=-1).squeeze(0) | (pos_cpu < lb).any(dim=-1).squeeze(0)
                
            if has_super_velocity.item():
                mask |= (vels_cpu.abs() > safety.max_particle_velocity).any(dim=-1).squeeze(0)
                
            if has_thermal_detonation.item() and t_cpu is not None:
                mask |= (t_cpu > safety.max_temperature).squeeze(-1).squeeze(0) | (t_cpu < safety.min_temperature).squeeze(-1).squeeze(0)
                
            bad_indices = torch.where(mask)[0]
            
            if len(bad_indices) > 0:
                idx = bad_indices[0].item() # Take the first violating particle
                p_pos = pos_cpu[0, idx].tolist()
                p_vel = vels_cpu[0, idx].tolist()
                
                msg = f"\n[GROUND ZERO PROFILING]\n"
                msg += f"Particle Index: {idx}\n"
                msg += f"Position (X,Y,Z): [{p_pos[0]:.4f}, {p_pos[1]:.4f}, {p_pos[2]:.4f}]\n"
                msg += f"Velocity (X,Y,Z): [{p_vel[0]:.4f}, {p_vel[1]:.4f}, {p_vel[2]:.4f}]\n"
                
                if temp is not None:
                    p_temp = t_cpu[0, idx].item()
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
            self.reconstructor.reset()
            gs.logger.info("Initializing surface reconstruction...")
            self.reconstructor.create_reconstructed_mesh()
            
            self._save_checkpoint_impl()
            
            # Trigger mesh data send to client (without physics step)
            self.pending_mesh_send = True

            # Suppress stability checks while the freshly reset state settles
            safety = getattr(self.env.cfg, 'safety', None)
            self._stability_grace_steps = safety.check_interval * 3 if safety else 30
            
            self.recorder.start_new_episode("continuous_forge")
            gs.logger.info("Simulation reset")
