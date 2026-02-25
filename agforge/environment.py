import torch
import numpy as np
import genesis as gs
import sys
import os
from agforge.options import (
    AgilityForgeOptions,
    RobotOptions,
    GENERATED_ROBOT_XML_PATH,
)
from agforge.materials import JohnsonCookPlasticity

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except AttributeError:
        base_path = os.path.dirname(__file__)

    return os.path.join(base_path, relative_path)

class AgilityForgeManipulator:
    """Encapsulates the robot's properties and provides a clean action interface."""
    def __init__(self, scene: gs.Scene, robot_cfg: RobotOptions):
        self.scene = scene
        self.device = gs.device
        self.robot_cfg = robot_cfg
        morph = gs.morphs.MJCF(file=GENERATED_ROBOT_XML_PATH)
        material = gs.materials.Rigid(
            coup_softness=float(0.001e-1),
            gravity_compensation=1.0,
        )
        self.entity = scene.add_entity(morph=morph, material=material)
        self.ee_link = self.entity.get_link("clamp_bar")
        self.left_gripper = self.entity.get_link("left_gripper")
        self.right_gripper = self.entity.get_link("right_gripper")
        
        # Cache indices for fast access
        self._left_gripper_link_idx = self.entity.get_link("left_gripper").idx
        self._right_gripper_link_idx = self.entity.get_link("right_gripper").idx
        self._coupler = None # Will cache on first access or setup
        
        # Calculate indices relative to the entity start for force lookup
        self.left_gripper_idx = self.left_gripper.idx - self.entity.link_start
        self.right_gripper_idx = self.right_gripper.idx - self.entity.link_start
        
        self.default_joint_angles = torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=self.device)
        self.control_mode = "PD_CONTROL"  # Initial control mode
        
        # Pre-allocated tensors for resistance force calculation (Issue #16 optimization)
        self._squeeze_dir_L = torch.tensor([0.0, 1.0, 0.0], device=self.device)
        self._squeeze_dir_R = torch.tensor([0.0, -1.0, 0.0], device=self.device)

    def set_pd_gains(self):
        """Sets the Proportional-Derivative controller gains for the robot's joints."""
        self.entity.set_dofs_kp(torch.full((4,), self.robot_cfg.kp, dtype=torch.float32, device=self.device))
        self.entity.set_dofs_kv(torch.full((4,), self.robot_cfg.kv, dtype=torch.float32, device=self.device))

    def set_clamp_force_range(self):
        """Sets the force range for the gripper joints."""
        max_force = self.robot_cfg.clamp_force
        # The gripper DOFs are the last two
        dofs_idx = torch.tensor([2, 3], dtype=torch.int32, device=self.device)
        lower = torch.full((2,), -max_force, dtype=torch.float32, device=self.device)
        upper = torch.full((2,), max_force, dtype=torch.float32, device=self.device)
        self.entity.set_dofs_force_range(lower, upper, dofs_idx_local=dofs_idx)

    def set_control_mode(self, mode: str):
        """Sets the control mode for the manipulator."""
        if mode not in ["PD_CONTROL", "TELEPORT", "VELOCITY_CONTROL"]:
            raise ValueError("Invalid control mode. Must be 'PD_CONTROL', 'TELEPORT', or 'VELOCITY_CONTROL'.")
        self.control_mode = mode

    def reset(self, envs_idx: torch.Tensor):
        """Resets the robot's joint positions to default for specified environments."""
        if len(envs_idx) > 0:
            qpos_to_set = self.default_joint_angles.expand(len(envs_idx), -1)
            self.entity.set_qpos(qpos_to_set, envs_idx=envs_idx)

    def apply_action(self, action: torch.Tensor, dofs_idx_local=None):
        """Applies an action based on the current control mode."""
        if self.control_mode == "PD_CONTROL":
            self.entity.control_dofs_position(position=action, dofs_idx_local=dofs_idx_local)
        elif self.control_mode == "TELEPORT":
            self.entity.control_dofs_position(position=action, dofs_idx_local=dofs_idx_local)
            self.entity.set_dofs_position(action, dofs_idx_local=dofs_idx_local)
        elif self.control_mode == "VELOCITY_CONTROL":
            self.apply_velocity(velocity=action, dofs_idx_local=dofs_idx_local)

    def apply_velocity(self, velocity: torch.Tensor, dofs_idx_local=None):
        """
        Applies velocity control by setting both the physical velocity state and the PD controller target.
        This provides 'stiff' velocity tracking suitable for the approaching stage.
        """
        self.entity.set_dofs_velocity(velocity, dofs_idx_local=dofs_idx_local)
        self.entity.control_dofs_velocity(velocity, dofs_idx_local=dofs_idx_local)

    def get_gripper_net_contact_force(self):
        """
        Returns the net external contact force on the left and right grippers.
        Returns:
            torch.Tensor: Shape (n_envs, 2, 3) where dim 1 is [Left, Right]
        """
        # (n_envs, n_links, 3)
        net_forces = self.entity.get_links_net_contact_force()
        # Select gripper links
        # Assuming single env for now or preserving batch dim
        if net_forces.dim() == 2: # (n_links, 3) - happens if n_envs=0 or something weird, but likely (n_envs, n_links, 3)
             return torch.stack([net_forces[self.left_gripper_idx], net_forces[self.right_gripper_idx]], dim=0).unsqueeze(0)
             
        # (n_envs, n_links, 3)
        # (n_envs, n_links, 3)
        rigid_forces = torch.stack([
            net_forces[:, self.left_gripper_idx, :],
            net_forces[:, self.right_gripper_idx, :]
        ], dim=1)
        
        # Add MPM forces if available (from LegacyCoupler accumulator)
        coupler = self.scene.sim.coupler
        if hasattr(coupler, 'link_coupling_forces'):
             # Get global indices
             idx_L = self._left_gripper_link_idx
             idx_R = self._right_gripper_link_idx
             
             # Optimization: Direct GPU access to Taichi field
             # shape: (n_links, n_envs, 3) or similar? 
             # Check legacy_coupler.py: shape=(rigid_solver.n_links_, sim._B)
             # Wait, field is Vector(3). shape is (n_links, n_envs).
             # .to_torch() will return shape (n_links, n_envs, 3)
             
             # We want to avoid pulling the WHOLE field if possible, but to_torch usually pulls all.
             # However, pulling ~100 links (tiny) on GPU is much faster than CPU sync.
             
             all_forces = coupler.link_coupling_forces.to_torch(device=self.device)
             
             # Select specific links for env 0
             # (n_links, n_envs, 3) -> (n_links, 3) for env 0
             forces_L = all_forces[idx_L, 0]
             forces_R = all_forces[idx_R, 0]
             
             # Stack [L, R] -> (2, 3) -> unsqueeze -> (1, 2, 3)
             mpm_stack = torch.stack([forces_L, forces_R]).unsqueeze(0)
             
             # Expand to match batch size if needed
             if rigid_forces.shape[0] > 1:
                  mpm_stack = mpm_stack.expand(rigid_forces.shape[0], -1, -1)
             
             # Normalize accumulated forces by substeps
             if hasattr(self.scene.sim, '_substeps'):
                 substeps = self.scene.sim._substeps
             else:
                 substeps = 1 # Fallback
                 
             return rigid_forces + (mpm_stack / substeps)
             
        return rigid_forces

    def _rotate_vector_by_quat(self, vector, quat):
        """
        Rotates a vector by a quaternion.
        vector: (3,) or (N, 3)
        quat: (4,) [w, x, y, z] or (N, 4)  # Genesis uses wxyz format
        """
        # Standard implementation: v + 2*cross(q_xyz, cross(q_xyz, v) + q_w*v)
        # Ensure dimensions match
        if vector.dim() == 1: vector = vector.unsqueeze(0)
        if quat.dim() == 1: quat = quat.unsqueeze(0)
        
        # Genesis quaternion format is [w, x, y, z]
        q_w = quat[:, 0].unsqueeze(-1)
        q_xyz = quat[:, 1:4]
        
        t = 2.0 * torch.cross(q_xyz, vector, dim=-1)
        return vector + q_w * t + torch.cross(q_xyz, t, dim=-1)

    def get_resistance_forces(self):
        """
        Returns the scalar resistance force opposing the squeeze motion.
        Projects the net contact force onto the INVERSE squeeze direction.
        Positive value means resistance (pushing back against squeeze).
        
        Returns:
            tuple[torch.Tensor, torch.Tensor]: (Force_L, Force_R) resistance forces for each gripper
        """
        # Get orientation of clamp_bar (parent of grippers)
        # quat is (n_envs, 4) or (4,)
        quat = self.ee_link.get_quat() 
        if quat.dim() == 1: quat = quat.unsqueeze(0)

        # Local squeeze directions (Pushing IN) - use pre-allocated tensors
        # Left slides +Y: (0, 1, 0)
        # Right slides -Y: (0, -1, 0)
        local_squeeze_L = self._squeeze_dir_L.expand(quat.shape[0], 3)
        local_squeeze_R = self._squeeze_dir_R.expand(quat.shape[0], 3)
        
        # Global squeeze directions
        global_squeeze_L = self._rotate_vector_by_quat(local_squeeze_L, quat)
        global_squeeze_R = self._rotate_vector_by_quat(local_squeeze_R, quat)
        
        # Get Contact Forces (F_contact)
        # shape: (n_envs, 2, 3)
        contact_forces = self.get_gripper_net_contact_force()
        force_L = contact_forces[:, 0, :]
        force_R = contact_forces[:, 1, :]
        
        # Resistance is Component of Force opposing Motion
        # F_resist = Dot(F_contact, -v_squeeze)
        #          = - Dot(F_contact, v_squeeze)
        
        resist_L = -torch.sum(force_L * global_squeeze_L, dim=-1)
        resist_R = -torch.sum(force_R * global_squeeze_R, dim=-1)
        
        # Return tuple of tensors for single environment (compatibility with teleop_socket)
        # Assuming batch size 1 for teleop
        return resist_L[0], resist_R[0]

    @property
    def ee_pose(self) -> torch.Tensor:
        """Returns the end-effector pose as (position, quaternion)."""
        pos, quat = self.ee_link.get_pos(), self.ee_link.get_quat()
        return torch.cat([pos, quat], dim=-1)

class AgilityForgeEnv:
    """An RL environment for the robotic forging task, configured parametrically."""
    def __init__(self, cfg: AgilityForgeOptions):
        self.cfg = cfg
        self.device = gs.device

        # Cache constant tensors on device for performance (must be done before usage)
        self.action_lower_bounds = self.cfg.env.action_lower_bounds.to(dtype=torch.float32, device=self.device)
        self.action_upper_bounds = self.cfg.env.action_upper_bounds.to(dtype=torch.float32, device=self.device)
        self.fixed_region_bounds = self.cfg.env.fixed_region_bounds.to(dtype=torch.float32, device=self.device)
        self.target_shape_bounds = self.cfg.env.target_shape_bounds.to(dtype=torch.float32, device=self.device)

        self._setup_scene()
        self.robot = AgilityForgeManipulator(scene=self.scene, robot_cfg=self.cfg.robot)
        self._setup_entities()
        self._add_visual_guides()
        
        if self.cfg.general.record:
            self._setup_recording_camera()
        
        self.scene.build(n_envs=self.cfg.env.num_envs, env_spacing=(0.75, 0.5))

        self.robot.set_pd_gains()
        self.robot.set_clamp_force_range()
        self.initial_robot_qpos = self.robot.entity.get_qpos().clone()
        self.initial_mpm_pos = self.mpm_entity.get_state().pos.clone()
        self._set_fixed_particles()
        
        num_particles = self.initial_mpm_pos.shape[1]
        self.num_obs = 3 + num_particles * 9
        self.num_actions = self.cfg.env.num_actions
        self.num_envs = self.cfg.env.num_envs
        self.max_episode_length = self.cfg.env.max_episode_length
        self.target_mpm_pos = torch.zeros_like(self.initial_mpm_pos)
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.int32)
        self.reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.bool)
        self.extras = {}

    def _setup_scene(self):
        """Configures and initializes the simulation scene from config objects."""
        viewer_options = self.cfg.viewer if self.cfg.general.show_viewer else gs.options.ViewerOptions()
        
        self.scene = gs.Scene(
            sim_options=self.cfg.sim,
            viewer_options=viewer_options,
            rigid_options=gs.options.RigidOptions(dt=self.cfg.sim.dt),
            mpm_options=self.cfg.mpm,
            vis_options=self.cfg.vis,
            profiling_options=self.cfg.profiling,
            show_viewer=self.cfg.general.show_viewer,
        )
        self.scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", pos=(0, 0, 0.01), fixed=True))

    def _setup_entities(self):
        """Creates the MPM object and the target shape visual guide."""
        self.mpm_entity = self.scene.add_entity(
            material=gs.materials.MPM.ElastoPlastic(
                E=self.cfg.mat.E, nu=self.cfg.mat.nu, rho=self.cfg.mat.rho,
                von_mises_yield_stress=self.cfg.mat.von_mises_yield_stress,
                sampler=resource_path("pbs_samples/cylinder.ptc"),
            ) if not getattr(self.cfg.mat, 'use_johnson_cook', False) else JohnsonCookPlasticity(
                E=self.cfg.mat.E, nu=self.cfg.mat.nu, rho=self.cfg.mat.rho,
                # J-C Params
                A=self.cfg.mat.jc_A, B=self.cfg.mat.jc_B, n=self.cfg.mat.jc_n,
                C=self.cfg.mat.jc_C, eps0=self.cfg.mat.jc_eps0,
                sampler=resource_path("pbs_samples/cylinder.ptc"),
            ),
            morph=gs.morphs.Cylinder(
                radius=self.cfg.robot.cylinder_radius, height=self.cfg.robot.cylinder_height, pos=self.cfg.robot.cylinder_pos, euler=self.cfg.robot.cylinder_euler
            ),
            surface=gs.surfaces.Metal(color=(0.8, 0.4, 0.0), vis_mode="particle"),
        )
        self.scene.add_entity(
            morph=gs.morphs.Box(size=(self.cfg.robot.target_shape_bounds[1] - self.cfg.robot.target_shape_bounds[0]).cpu(), pos=((self.cfg.robot.target_shape_bounds[0] + self.cfg.robot.target_shape_bounds[1]) / 2).cpu(), fixed=True, collision=False),
            surface=gs.surfaces.Default(color=(1.0, 0.0, 0.0, 0.4)),
        )

    def _add_visual_guides(self):
        """Adds a non-physical box to visualize the fixed particle region."""
        bounds = self.cfg.env.fixed_region_bounds
        box_size = (bounds[:, 1] - bounds[:, 0]).cpu()
        box_pos = bounds.mean(dim=1).cpu()
        self.scene.add_entity(
            morph=gs.morphs.Box(size=box_size, pos=box_pos, fixed=True, collision=False),
            surface=gs.surfaces.Default(color=(0.1, 0.1, 0.6, 0.4)),
        )

    def _set_fixed_particles(self):
        """Identifies and sets the fixed MPM particles based on the defined region."""
        bounds = self.fixed_region_bounds
        mpm_pos_env0 = self.initial_mpm_pos[0]
        is_in_box = ((mpm_pos_env0 >= bounds[:, 0]) & (mpm_pos_env0 <= bounds[:, 1])).all(dim=1)
        self.mpm_entity.set_free(~is_in_box)

    def _setup_recording_camera(self):
        """Initializes the camera used for video recording."""
        self.recording_camera = self.scene.add_camera(
            res=self.cfg.viewer.res,
            pos=self.cfg.viewer.camera_pos,
            lookat=self.cfg.viewer.camera_lookat,
            fov=45,
            GUI=False
        )
        self.scene.add_camera(GUI=True)

    def start_recording(self):
        if self.cfg.general.record:
            self.recording_camera.start_recording()

    def stop_recording(self, filename="agforge_demo.mp4", fps=30):
        if self.cfg.general.record:
            self.recording_camera.stop_recording(save_to_filename=filename, fps=fps)

    def reset_idx(self, envs_idx: torch.Tensor):
        """Resets specified environments to their initial state."""
        if len(envs_idx) == 0: return
        
        self.episode_length_buf[envs_idx] = 0
        self.robot.reset(envs_idx)
        self.scene.reset(envs_idx=envs_idx)

        init_bounds = torch.stack([self.initial_mpm_pos[0].min(dim=0).values, self.initial_mpm_pos[0].max(dim=0).values])
        target_bounds = self.target_shape_bounds
        
        eps = 1e-8
        scale = (target_bounds[1] - target_bounds[0]) / (init_bounds[1] - init_bounds[0] + eps)
        offset = target_bounds[0] - init_bounds[0] * scale
        
        self.target_mpm_pos = self.initial_mpm_pos * scale + offset

    def reset(self) -> tuple[torch.Tensor, None]:
        """Resets all environments and returns the initial observation."""
        all_envs = torch.arange(self.num_envs, device=self.device)
        self.reset_idx(all_envs)
        obs, _ = self.get_observations()
        return obs, None

    def _step_sim(self, num_steps: int):
        """Steps the simulation for a given number of steps, with optional recording."""
        if self.cfg.general.record:
            for _ in range(num_steps):
                self.scene.step()
                self.recording_camera.render(rgb=True)
        else:
            for _ in range(num_steps):
                self.scene.step()

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Applies an action and steps the simulation forward."""
        self.episode_length_buf += 1
        actions = torch.clamp(
            actions, self.action_lower_bounds, self.action_upper_bounds
        )

        qpos = self.robot.entity.get_qpos()
        
        # Teleport to the starting position for the clamp
        self.robot.set_control_mode("TELEPORT")
        qpos[:, 0:2] = actions[:, 0:2]
        self.robot.apply_action(qpos[:, 0:2], dofs_idx_local=[0, 1])
        self._step_sim(1)  # One step to ensure the state is updated

        # Switch to PD control for the clamping action
        self.robot.set_control_mode("PD_CONTROL")
        qpos[:, 2:4] = actions[:, 2].unsqueeze(-1)
        self.robot.apply_action(qpos[:, 2:4], dofs_idx_local=[2, 3])
        self._step_sim(self.cfg.env.action_duration_steps)

        # Reset for the next action
        qpos_prev = self.robot.entity.get_qpos()
        reset_qpos = self.initial_robot_qpos.clone()
        reset_qpos[:, 0:2] = qpos_prev[:, 0:2]
        self.robot.apply_action(reset_qpos)
        self._step_sim(self.cfg.env.reset_duration_steps)

        self.reset_buf = (self.episode_length_buf >= self.max_episode_length)
        if self.reset_buf.any():
            self.reset_idx(self.reset_buf.nonzero(as_tuple=False).squeeze(-1))

        rewards = self._reward_mpm_deformation()
        obs, self.extras = self.get_observations()
        return obs, rewards, self.reset_buf, self.extras

    def get_observations(self) -> tuple[torch.Tensor, dict]:
        """Constructs the observation tensor from the current simulation state."""
        mpm_pos = self.mpm_entity.get_state().pos
        mpm_obs = torch.cat([mpm_pos, self.target_mpm_pos, mpm_pos - self.target_mpm_pos], dim=-1).flatten(start_dim=1)
        obs = torch.cat([self.robot.ee_pose[:, :3], mpm_obs], dim=-1)
        
        self.extras['observations'] = {'critic': obs}
        return obs, self.extras
    
    def get_privileged_observations(self) -> None:
        return None

    def _reward_mpm_deformation(self) -> torch.Tensor:
        """Calculates reward based on the mean distance to the target shape."""
        dist = torch.norm(self.mpm_entity.get_state().pos - self.target_mpm_pos, p=2, dim=-1).mean(dim=-1)
        return torch.exp(-dist)