import torch
import genesis as gs
from config import SimConfig, EnvConfig, GeneralConfig, RobotConfig, MaterialConfig, GENERATED_ROBOT_XML_PATH
from config import CYLINDER_RADIUS, CYLINDER_HEIGHT, CYLINDER_POS, CYLINDER_EULER
from config import TARGET_GUIDE_BOX_SIZE, TARGET_GUIDE_BOX_POS

class AgilityForgeManipulator:
    """Encapsulates the robot's properties and provides a clean action interface."""
    def __init__(self, scene: gs.Scene, robot_cfg: RobotConfig):
        self.device = gs.device
        self.robot_cfg = robot_cfg
        morph = gs.morphs.MJCF(file=GENERATED_ROBOT_XML_PATH)
        material = gs.materials.Rigid(
            coup_softness=100.,
            gravity_compensation=1.0,
        )
        self.entity = scene.add_entity(morph=morph, material=material)
        self.ee_link = self.entity.get_link("clamp_bar")
        self.default_joint_angles = torch.tensor([0.0, 0.0, 0.0, 0.0], device=self.device)

    def set_pd_gains(self):
        """Sets the Proportional-Derivative controller gains for the robot's joints."""
        self.entity.set_dofs_kp(torch.full((4,), self.robot_cfg.kp, device=self.device))
        self.entity.set_dofs_kv(torch.full((4,), self.robot_cfg.kv, device=self.device))

    def reset(self, envs_idx: torch.Tensor):
        """Resets the robot's joint positions to default for specified environments."""
        if len(envs_idx) > 0:
            qpos_to_set = self.default_joint_angles.expand(len(envs_idx), -1)
            self.entity.set_qpos(qpos_to_set, envs_idx=envs_idx)

    def apply_action(self, position: torch.Tensor):
        """Applies a target joint position to the PD controller."""
        self.entity.control_dofs_position(position=position)

    @property
    def ee_pose(self) -> torch.Tensor:
        """Returns the end-effector pose as (position, quaternion)."""
        pos, quat = self.ee_link.get_pos(), self.ee_link.get_quat()
        return torch.cat([pos, quat], dim=-1)

class AgilityForgeEnv:
    """An RL environment for the robotic forging task, configured parametrically."""
    def __init__(self, cfg):
        self.cfg = cfg
        self.sim_cfg = cfg.sim
        self.env_cfg = cfg.env
        self.general_cfg = cfg.general
        self.robot_cfg = cfg.robot
        self.mat_cfg = cfg.mat
        self.vis_cfg = cfg.vis
        self.profiling_cfg = cfg.profiling
        self.device = gs.device

        self._setup_scene()
        self.robot = AgilityForgeManipulator(scene=self.scene, robot_cfg=self.robot_cfg)
        self._setup_entities()
        self._add_visual_guides()
        
        if self.general_cfg.record:
            self._setup_recording_camera()
        
        self.scene.build(n_envs=self.env_cfg.num_envs, env_spacing=(0.75, 0.5))

        self.robot.set_pd_gains()
        self.initial_robot_qpos = self.robot.entity.get_qpos().clone()
        self.initial_mpm_pos = self.mpm_entity.get_state().pos.clone()
        self._set_fixed_particles()
        
        num_particles = self.initial_mpm_pos.shape[1]
        self.num_obs = 3 + num_particles * 9
        self.num_actions = self.env_cfg.num_actions
        self.num_envs = self.env_cfg.num_envs
        self.max_episode_length = self.env_cfg.max_episode_length
        self.target_mpm_pos = torch.empty_like(self.initial_mpm_pos)
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.int32)
        self.reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.bool)
        self.extras = {}

    def _setup_scene(self):
        """Configures and initializes the simulation scene from config objects."""
        viewer_options = gs.options.ViewerOptions(
            camera_pos=self.general_cfg.camera_pos,
            camera_lookat=self.general_cfg.camera_lookat,
            max_FPS=60,
            res=self.vis_cfg.camera_res,
        ) if self.general_cfg.show_viewer else gs.options.ViewerOptions()
        
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.sim_cfg.dt, substeps=self.sim_cfg.substeps, requires_grad=False),
            viewer_options=viewer_options,
            rigid_options=gs.options.RigidOptions(dt=self.sim_cfg.dt),
            mpm_options=gs.options.MPMOptions(
                lower_bound=self.sim_cfg.lower_bound, upper_bound=self.sim_cfg.upper_bound,
                gravity=self.sim_cfg.gravity, grid_density=self.sim_cfg.grid_density, particle_size=self.sim_cfg.particle_size
            ),
            vis_options=gs.options.VisOptions(
                show_world_frame=self.vis_cfg.show_world_frame,
                visualize_mpm_boundary=self.vis_cfg.visualize_mpm_boundary,
                visualize_mpm_grid=self.vis_cfg.visualize_mpm_grid,
                render_particle_as=self.vis_cfg.render_particle_as,
                shadow=self.vis_cfg.shadow,
                plane_reflection=self.vis_cfg.plane_reflection,
                particle_render_fraction=self.vis_cfg.particle_render_fraction,
            ),
            profiling_options=self.profiling_cfg.profiling_options,
            show_viewer=self.general_cfg.show_viewer,
        )
        self.scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", pos=(0, 0, 0.01), fixed=True))

    def _setup_entities(self):
        """Creates the MPM object and the target shape visual guide."""
        self.mpm_entity = self.scene.add_entity(
            material=gs.materials.MPM.ElastoPlastic(
                E=self.mat_cfg.E, nu=self.mat_cfg.nu, rho=self.mat_cfg.rho,
                von_mises_yield_stress=self.mat_cfg.von_mises_yield_stress
            ),
            morph=gs.morphs.Cylinder(
                radius=CYLINDER_RADIUS, height=CYLINDER_HEIGHT, pos=CYLINDER_POS, euler=CYLINDER_EULER
            ),
            surface=gs.surfaces.Metal(color=(0.8, 0.4, 0.0), vis_mode="particle"),
        )
        self.scene.add_entity(
            morph=gs.morphs.Box(size=TARGET_GUIDE_BOX_SIZE, pos=TARGET_GUIDE_BOX_POS, fixed=True, collision=False),
            surface=gs.surfaces.Default(color=(1.0, 0.0, 0.0, 0.4)),
        )

    def _add_visual_guides(self):
        """Adds a non-physical box to visualize the fixed particle region."""
        bounds = self.env_cfg.fixed_region_bounds
        box_size = (bounds[:, 1] - bounds[:, 0]).cpu()
        box_pos = bounds.mean(dim=1).cpu()
        self.scene.add_entity(
            morph=gs.morphs.Box(size=box_size, pos=box_pos, fixed=True, collision=False),
            surface=gs.surfaces.Default(color=(0.1, 0.1, 0.6, 0.4)),
        )

    def _set_fixed_particles(self):
        """Identifies and sets the fixed MPM particles based on the defined region."""
        bounds = self.env_cfg.fixed_region_bounds.to(self.device)
        mpm_pos_env0 = self.initial_mpm_pos[0]
        is_in_box = ((mpm_pos_env0 >= bounds[:, 0]) & (mpm_pos_env0 <= bounds[:, 1])).all(dim=1)
        self.mpm_entity.set_free(~is_in_box)

    def _setup_recording_camera(self):
        """Initializes the camera used for video recording."""
        self.recording_camera = self.scene.add_camera(
            res=self.vis_cfg.camera_res, pos=self.general_cfg.camera_pos,
            lookat=self.general_cfg.camera_lookat, fov=45, GUI=False
        )
        self.scene.add_camera(GUI=True)

    def start_recording(self):
        if self.general_cfg.record:
            self.recording_camera.start_recording()

    def stop_recording(self, filename="agforge_demo.mp4", fps=30):
        if self.general_cfg.record:
            self.recording_camera.stop_recording(save_to_filename=filename, fps=fps)

    def reset_idx(self, envs_idx: torch.Tensor):
        """Resets specified environments to their initial state."""
        if len(envs_idx) == 0: return
        
        self.episode_length_buf[envs_idx] = 0
        self.robot.reset(envs_idx)
        self.scene.reset(envs_idx=envs_idx)

        init_bounds = torch.stack([self.initial_mpm_pos[0].min(dim=0).values, self.initial_mpm_pos[0].max(dim=0).values])
        target_bounds = self.env_cfg.target_shape_bounds.to(self.device)
        
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
        if self.general_cfg.record:
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
            actions, self.env_cfg.action_lower_bounds.to(self.device), self.env_cfg.action_upper_bounds.to(self.device)
        )

        qpos = self.robot.entity.get_qpos()
        
        qpos[:, 0:2] = actions[:, 0:2]
        self.robot.apply_action(qpos)
        self._step_sim(self.env_cfg.action_duration_steps)

        qpos[:, 2:4] = actions[:, 2].unsqueeze(-1)
        self.robot.apply_action(qpos)
        self._step_sim(self.env_cfg.action_duration_steps)

        qpos_prev = self.robot.entity.get_qpos()
        reset_qpos = self.initial_robot_qpos.clone()
        reset_qpos[:, 0:2] = qpos_prev[:, 0:2]
        self.robot.apply_action(reset_qpos)
        self._step_sim(self.env_cfg.reset_duration_steps)

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