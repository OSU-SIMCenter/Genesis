import torch
import math
import genesis as gs
from rsl_rl.runners import OnPolicyRunner

class AgilityForgeManipulator:
    """A helper class that encapsulates the robot's properties and actions."""
    def __init__(self, scene: gs.Scene):
        self.device = gs.device
        
        morph = gs.morphs.MJCF(file="xml/agforge_demo.xml")
        self.entity = scene.add_entity(morph=morph, material=gs.materials.Rigid(gravity_compensation=1.0))
        self.ee_link = self.entity.get_link("clamp_bar")
        
        # Default joint angles stored as a reusable tensor
        self.default_joint_angles = torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=self.device)

    def set_pd_gains(self, kp: float = 4500., kv: float = 450.):
        """Sets the Proportional-Derivative controller gains for the robot's joints."""
        self.entity.set_dofs_kp(torch.full((4,), kp, device=self.device))
        self.entity.set_dofs_kv(torch.full((4,), kv, device=self.device))

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
    """
    An RL environment for a robotic forging task.
    The goal is to deform a cylindrical MPM object into a target rectangular shape.
    """
    def __init__(self, num_envs: int, show_viewer: bool = False, record: bool = False):
        self.num_envs = num_envs
        self.device = gs.device
        self.ctrl_dt = 0.005
        self.max_episode_length = math.ceil(100. / self.ctrl_dt)
        self.action_duration_steps = 40  # Sim steps to hold an action
        self.reset_duration_steps = 23   # Sim steps to reset joints

        # Action space definition
        self.num_actions = 3 # (slider_pos, hinge_angle, gripper_opening)
        self.action_lower_bounds = torch.tensor([-0.18, -40.0, 0.95], device=self.device)
        self.action_upper_bounds = torch.tensor([-0.12, 40.0, 0.1], device=self.device)
        
        # Fixed region bounds for MPM particles
        self.fixed_region_bounds = torch.tensor([[0.15, 0.25], [-0.1, 0.1], [0.2, 0.4]], device=self.device)
        
        # Target shape bounds for transformation
        self.target_shape_bounds = torch.tensor([[-0.01, 0.1, -0.01], [0.01, 0.5, 0.01]], device=self.device)

        self._setup_scene(show_viewer)
        self.robot = AgilityForgeManipulator(scene=self.scene)
        self._setup_entities()

        # Add visual guides before building scene
        self._add_visual_guides()
        
        # Add recording camera
        if record:
            self.recording_camera = self.scene.add_camera(
                res=(1280, 720),
                pos=(-0.9, 0.05, 0.87),
                lookat=(0.62, 0.58, 0.017),
                fov=45,
                GUI=False,
            )
            self.scene.add_camera(GUI=True)
        self.record = record
        
        self.scene.build(n_envs=num_envs, env_spacing=(0.75, 0.5))

        # Final Initializations
        self.robot.set_pd_gains()
        self.initial_robot_qpos = self.robot.entity.get_qpos().clone()
        self.initial_mpm_pos = self.mpm_entity.get_state().pos.clone()
        self._set_fixed_particles()
        num_particles = self.initial_mpm_pos.shape[1]
        self.num_obs = 3 + num_particles * 9
        self.target_mpm_pos = torch.empty_like(self.initial_mpm_pos)
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.int32)
        self.reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.bool)
        self.extras = {}

    def start_recording(self):
        """Start recording video from the camera."""
        self.recording_camera.start_recording()

    def stop_recording(self, filename="agforge_demo.mp4", fps=30):
        """Stop recording and save video."""
        self.recording_camera.stop_recording(save_to_filename=filename, fps=fps)

    def _setup_scene(self, show_viewer: bool):
        """Configures and initializes the simulation scene."""
        viewer_options = gs.options.ViewerOptions(camera_pos=(-1.1, -0.4, 0.9), camera_lookat=(0.5, 0.15, 0.0), max_FPS=30) if show_viewer else gs.options.ViewerOptions()
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.ctrl_dt, substeps=256, requires_grad=False),
            viewer_options=viewer_options,
            rigid_options=gs.options.RigidOptions(dt=self.ctrl_dt),
            mpm_options=gs.options.MPMOptions(lower_bound=(-0.4, -0.15, 0.15), upper_bound=(0.25, 0.15, 0.45), gravity=(0, 0, 0), grid_density=73),
            vis_options=gs.options.VisOptions(show_world_frame=False, visualize_mpm_boundary=True),
            show_viewer=show_viewer,
        )
        self.scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", pos=(0, 0, 0.01), fixed=True))

    def _setup_entities(self):
        """Creates the MPM object and non-physical visual guides."""
        self.mpm_entity = self.scene.add_entity(
            material=gs.materials.MPM.ElastoPlastic(E=5.e5, nu=0.3, rho=100., von_mises_yield_stress=480.),
            morph=gs.morphs.Cylinder(radius=0.05, height=0.4, pos=(0.0, 0.0, 0.3), euler=(0.0, 90.0, 0.0)),
            surface=gs.surfaces.Metal(color=(0.8, 0.4, 0.0), vis_mode="particle"),
        )
        # Visual guide for the target shape (non-physical)
        self.scene.add_entity(
            morph=gs.morphs.Box(size=(0.3, 0.06, 0.06), pos=(-0.15, 0.0, 0.3), fixed=True, collision=False),
            surface=gs.surfaces.Default(color=(1.0, 0.0, 0.0, 0.4)), # Red, semi-transparent
        )

    def _add_visual_guides(self):
        """Adds visual guide entities before scene is built."""
        # Visual guide for the fixed region (non-physical)
        self.scene.add_entity(
            morph=gs.morphs.Box(
                size=(self.fixed_region_bounds[:, 1] - self.fixed_region_bounds[:, 0]).cpu(),
                pos=self.fixed_region_bounds.mean(dim=1).cpu(),
                fixed=True,
                collision=False
            ),
            surface=gs.surfaces.Default(color=(0.1, 0.1, 0.6, 0.4)),  # Blue, semi-transparent
        )
    
    def _set_fixed_particles(self):
        """Sets the fixed particle mask after scene is built."""
        # Vectorized creation of the mask for fixed particles
        mpm_pos_env0 = self.initial_mpm_pos[0]
        is_in_box = ((mpm_pos_env0 >= self.fixed_region_bounds[:, 0]) & (mpm_pos_env0 <= self.fixed_region_bounds[:, 1])).all(dim=1)
        self.mpm_entity.set_free(~is_in_box)

    def reset_idx(self, envs_idx: torch.Tensor):
        """Resets specified environments to their initial state."""
        if len(envs_idx) == 0: return
            
        self.episode_length_buf[envs_idx] = 0
        self.robot.reset(envs_idx)
        self.scene.reset(envs_idx=envs_idx)

        # Linearly transform initial particle positions to the target shape's bounding box.
        init_bounds = torch.stack([self.initial_mpm_pos[0].min(dim=0).values, self.initial_mpm_pos[0].max(dim=0).values])
        
        eps = 1e-8
        scale = (self.target_shape_bounds[1] - self.target_shape_bounds[0]) / (init_bounds[1] - init_bounds[0] + eps)
        offset = self.target_shape_bounds[0] - init_bounds[0] * scale
        
        # Apply transformation to all envs via broadcasting
        self.target_mpm_pos = self.initial_mpm_pos * scale + offset

    def reset(self) -> tuple[torch.Tensor, None]:
        """Resets all environments and returns the initial observation."""
        all_envs = torch.arange(self.num_envs, device=self.device)
        self.reset_idx(all_envs)
        obs, _ = self.get_observations()
        return obs, None

    def step_action(self, num_action_steps):
        if self.record:
            for _ in range(num_action_steps):
                self.scene.step()
                self.recording_camera.render(rgb=True)
        else:
            for _ in range(num_action_steps): self.scene.step()

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Applies an action and steps the simulation forward."""
        self.episode_length_buf += 1
        actions = torch.clamp(actions, self.action_lower_bounds, self.action_upper_bounds)

        # Phase 1: Move slider and hinge
        qpos = self.robot.entity.get_qpos()
        qpos[:, 0:2] = actions[:, 0:2]
        self.robot.apply_action(qpos)
        self.step_action(self.action_duration_steps)

        # Phase 2: Close/open gripper
        qpos[:, 2:4] = actions[:, 2].unsqueeze(-1)
        self.robot.apply_action(qpos)
        self.step_action(self.action_duration_steps)

        # Phase 3: Reset gripper, keeping slider and hinge positions
        qpos_prev = self.robot.entity.get_qpos()
        reset_qpos = self.initial_robot_qpos.clone()
        reset_qpos[:, 0:2] = qpos_prev[:, 0:2]
        self.robot.apply_action(reset_qpos)
        self.step_action(self.reset_duration_steps)

        # Check for episode termination
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
        return torch.exp(-dist) # Exponential reward to heavily penalize large distances

def main():
    # --- Hyperparameters ---
    NUM_ENVS = 10
    MAX_ITERATIONS = 1
    SHOW_VIEWER = False
    RECORD = False

    gs.init(logging_level="info", precision="32", backend=gs.gpu)
    env = AgilityForgeEnv(num_envs=NUM_ENVS, show_viewer=SHOW_VIEWER, record=RECORD)

    # Start recording
    if RECORD: env.start_recording()

    # Configuration for the PPO algorithm from rsl_rl
    train_cfg = {
        "algorithm": {"class_name": "PPO", "gamma": 0.99, "lam": 0.95, "learning_rate": 5e-4, "entropy_coef": 0.1},
        "policy": {"class_name": "ActorCritic", "actor_hidden_dims": [256, 128], "critic_hidden_dims": [256, 128]},
        "runner": {"max_iterations": MAX_ITERATIONS, "run_name": "agforge_demo"},
        "runner_class_name": "OnPolicyRunner",
        "num_steps_per_env": 1,
        "empirical_normalization": False,
        "save_interval": 50,
    }

    log_dir = "logs/agforge_demo"
    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    runner.learn(num_learning_iterations=MAX_ITERATIONS, init_at_random_ep_len=True)
    
    # Stop recording and save video
    if RECORD: env.stop_recording()

if __name__ == "__main__":
    main()
