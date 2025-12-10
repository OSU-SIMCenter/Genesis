import torch
import genesis as gs
from options import (
    AgilityForgeOptions,
    RobotOptions,
    GENERATED_ROBOT_XML_PATH,
)

class AgilityForgeManipulator:
    """Encapsulates the robot's properties and provides a clean action interface."""
    def __init__(self, scene: gs.Scene, robot_cfg: RobotOptions):
        self.device = gs.device
        self.robot_cfg = robot_cfg
        morph = gs.morphs.MJCF(file=GENERATED_ROBOT_XML_PATH)
        material = gs.materials.Rigid(
            coup_softness=0.002,
            gravity_compensation=1.0,
        )
        self.entity = scene.add_entity(morph=morph, material=material)
        self.ee_link = self.entity.get_link("clamp_bar")
        self.default_joint_angles = torch.tensor([0.0, 0.0, 0.0, 0.0], device=self.device)
        self.control_mode = "PD_CONTROL"  # Initial control mode

    def set_pd_gains(self):
        """Sets the Proportional-Derivative controller gains for the robot's joints."""
        self.entity.set_dofs_kp(torch.full((4,), self.robot_cfg.kp, device=self.device))
        self.entity.set_dofs_kv(torch.full((4,), self.robot_cfg.kv, device=self.device))

    def set_clamp_force_range(self):
        """Sets the force range for the gripper joints."""
        max_force = self.robot_cfg.clamp_force
        # The gripper DOFs are the last two
        dofs_idx = torch.tensor([2, 3], device=self.device)
        lower = torch.full((2,), -max_force, device=self.device)
        upper = torch.full((2,), max_force, device=self.device)
        self.entity.set_dofs_force_range(lower, upper, dofs_idx_local=dofs_idx)

    def set_control_mode(self, mode: str):
        """Sets the control mode for the manipulator."""
        if mode not in ["PD_CONTROL", "TELEPORT"]:
            raise ValueError("Invalid control mode. Must be 'PD_CONTROL' or 'TELEPORT'.")
        self.control_mode = mode

    def reset(self, envs_idx: torch.Tensor):
        """Resets the robot's joint positions to default for specified environments."""
        if len(envs_idx) > 0:
            qpos_to_set = self.default_joint_angles.expand(len(envs_idx), -1)
            self.entity.set_qpos(qpos_to_set, envs_idx=envs_idx)

    def apply_action(self, position: torch.Tensor, dofs_idx_local=None):
        """Applies an action based on the current control mode."""
        if self.control_mode == "PD_CONTROL":
            self.entity.control_dofs_position(position=position, dofs_idx_local=dofs_idx_local)
        elif self.control_mode == "TELEPORT":
            self.entity.control_dofs_position(position=position, dofs_idx_local=dofs_idx_local)
            self.entity.set_dofs_position(position, dofs_idx_local=dofs_idx_local)

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
                von_mises_yield_stress=self.cfg.mat.von_mises_yield_stress
            ),
            morph=gs.morphs.Cylinder(
                radius=self.cfg.robot.cylinder_radius, height=self.cfg.robot.cylinder_height, pos=self.cfg.robot.cylinder_pos, euler=self.cfg.robot.cylinder_euler
            ),
            surface=gs.surfaces.Metal(color=(0.8, 0.4, 0.0), vis_mode="particle"),
        )
        self.scene.add_entity(
            morph=gs.morphs.Box(size=self.cfg.robot.target_shape_bounds[1] - self.cfg.robot.target_shape_bounds[0], pos=(self.cfg.robot.target_shape_bounds[0] + self.cfg.robot.target_shape_bounds[1]) / 2, fixed=True, collision=False),
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
        bounds = self.cfg.env.fixed_region_bounds.to(self.device)
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
        target_bounds = self.cfg.env.target_shape_bounds.to(self.device)
        
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
            actions, self.cfg.env.action_lower_bounds.to(self.device), self.cfg.env.action_upper_bounds.to(self.device)
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