import torch
import math
import argparse
import genesis as gs
from rsl_rl.runners import OnPolicyRunner
# Import necessary libraries
from config import SimConfig, EnvConfig, SacConfig, AdamConfig, GeneralConfig

# ==================================================================================================
# Robot Manipulator Class
# ==================================================================================================
class AgilityForgeManipulator:
    """
    A helper class that encapsulates the robot's properties and actions, providing a clean interface
    for controlling the robot within the simulation.
    """
    def __init__(self, scene: gs.Scene):
        """
        Initializes the manipulator, loading the robot model and identifying key links.
        
        Parameters:
            scene (gs.Scene): The simulation scene to which the robot will be added.
        """
        self.device = gs.device
        
        # Load the robot from an MJCF file
        morph = gs.morphs.MJCF(file="xml/agforge_demo.xml")
        self.entity = scene.add_entity(morph=morph, material=gs.materials.Rigid(gravity_compensation=1.0))
        self.ee_link = self.entity.get_link("clamp_bar")  # End-effector link
        
        # Default joint angles for resetting the robot's pose
        self.default_joint_angles = torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=self.device)

    def set_pd_gains(self, kp: float = 4500., kv: float = 450.):
        """
        Sets the Proportional-Derivative (PD) controller gains for the robot's joints.
        
        Parameters:
            kp (float): Proportional gain.
            kv (float): Derivative gain.
        """
        self.entity.set_dofs_kp(torch.full((4,), kp, device=self.device))
        self.entity.set_dofs_kv(torch.full((4,), kv, device=self.device))

    def reset(self, envs_idx: torch.Tensor):
        """
        Resets the robot's joint positions to their default values for specified environments.
        
        Parameters:
            envs_idx (torch.Tensor): Indices of the environments to reset.
        """
        if len(envs_idx) > 0:
            qpos_to_set = self.default_joint_angles.expand(len(envs_idx), -1)
            self.entity.set_qpos(qpos_to_set, envs_idx=envs_idx)

    def apply_action(self, position: torch.Tensor):
        """
        Applies a target joint position to the PD controller.
        
        Parameters:
            position (torch.Tensor): The target joint positions.
        """
        self.entity.control_dofs_position(position=position)

    @property
    def ee_pose(self) -> torch.Tensor:
        """
        Returns the end-effector pose as a tensor containing position and quaternion.
        """
        pos, quat = self.ee_link.get_pos(), self.ee_link.get_quat()
        return torch.cat([pos, quat], dim=-1)

# ==================================================================================================
# RL Environment Class
# ==================================================================================================
class AgilityForgeEnv:
    """
    An RL environment for a robotic forging task. The goal is to deform a cylindrical
    MPM object into a target rectangular shape using a robotic manipulator.
    """
    def __init__(self, sim_cfg: SimConfig, env_cfg: EnvConfig, general_cfg: GeneralConfig, device: str):
        """
        Initializes the environment, including the simulation scene, robot, and MPM object.
        
        Parameters:
            sim_cfg (SimConfig): Simulation configuration.
            env_cfg (EnvConfig): Environment configuration.
            general_cfg (GeneralConfig): General configuration (e.g., for visualization).
            device (str): The device to run the simulation on.
        """
        self.sim_cfg = sim_cfg
        self.env_cfg = env_cfg
        self.general_cfg = general_cfg
        self.device = device
        self.env_cfg.fixed_region_bounds = self.env_cfg.fixed_region_bounds.to(self.device)
        self.env_cfg.target_shape_bounds = self.env_cfg.target_shape_bounds.to(self.device)
        self.env_cfg.action_lower_bounds = self.env_cfg.action_lower_bounds.to(self.device)
        self.env_cfg.action_upper_bounds = self.env_cfg.action_upper_bounds.to(self.device)

        # Setup the scene, entities, and visual guides
        self._setup_scene()
        self.robot = AgilityForgeManipulator(scene=self.scene)
        self._setup_entities()
        self._add_visual_guides()
        
        # Setup recording if enabled
        if self.general_cfg.record:
            self.recording_camera = self.scene.add_camera(
                res=(1280, 720), pos=(-0.9, 0.05, 0.87),
                lookat=(0.62, 0.58, 0.017), fov=45, GUI=False,
            )
            self.scene.add_camera(GUI=True)
        
        # Build the scene for the specified number of environments
        self.scene.build(n_envs=self.env_cfg.num_envs, env_spacing=(0.75, 0.5))

        # Finalize initialization of robot and environment state
        self.robot.set_pd_gains()
        self.initial_robot_qpos = self.robot.entity.get_qpos().clone()
        self.initial_mpm_pos = self.mpm_entity.get_state().pos.clone()
        self._set_fixed_particles()
        num_particles = self.initial_mpm_pos.shape[1]
        self.num_obs = 3 + num_particles * 9
        self.num_actions = 3
        self.num_envs = self.env_cfg.num_envs
        self.max_episode_length = self.env_cfg.max_episode_length
        self.target_mpm_pos = torch.empty_like(self.initial_mpm_pos)
        self.episode_length_buf = torch.zeros(self.env_cfg.num_envs, device=self.device, dtype=torch.int32)
        self.reset_buf = torch.ones(self.env_cfg.num_envs, device=self.device, dtype=torch.bool)
        self.extras = {}

    def start_recording(self):
        """Starts video recording."""
        self.recording_camera.start_recording()

    def stop_recording(self, filename="agforge_demo.mp4", fps=30):
        """Stops video recording and saves the file."""
        self.recording_camera.stop_recording(save_to_filename=filename, fps=fps)

    def _setup_scene(self):
        """Configures and initializes the simulation scene."""
        viewer_options = gs.options.ViewerOptions(
            camera_pos=(-1.1, -0.4, 0.9), camera_lookat=(0.5, 0.15, 0.0), max_FPS=30
        ) if self.general_cfg.show_viewer else gs.options.ViewerOptions()
        
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.sim_cfg.dt, substeps=self.sim_cfg.substeps, requires_grad=False),
            viewer_options=viewer_options,
            rigid_options=gs.options.RigidOptions(dt=self.sim_cfg.dt),
            mpm_options=gs.options.MPMOptions(
                lower_bound=self.sim_cfg.lower_bound, upper_bound=self.sim_cfg.upper_bound,
                gravity=self.sim_cfg.gravity, grid_density=self.sim_cfg.grid_density
            ),
            vis_options=gs.options.VisOptions(show_world_frame=False, visualize_mpm_boundary=True),
            show_viewer=self.general_cfg.show_viewer,
        )
        self.scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", pos=(0, 0, 0.01), fixed=True))

    def _setup_entities(self):
        """Creates the MPM object and non-physical visual guides."""
        self.mpm_entity = self.scene.add_entity(
            material=gs.materials.MPM.ElastoPlastic(E=5.e5, nu=0.3, rho=100., von_mises_yield_stress=480.),
            morph=gs.morphs.Cylinder(radius=0.05, height=0.4, pos=(0.0, 0.0, 0.3), euler=(0.0, 90.0, 0.0)),
            surface=gs.surfaces.Metal(color=(0.8, 0.4, 0.0), vis_mode="particle"),
        )
        self.scene.add_entity(
            morph=gs.morphs.Box(size=(0.3, 0.06, 0.06), pos=(-0.15, 0.0, 0.3), fixed=True, collision=False),
            surface=gs.surfaces.Default(color=(1.0, 0.0, 0.0, 0.4)),
        )

    def _add_visual_guides(self):
        """Adds visual guides for the fixed region before the scene is built."""
        self.scene.add_entity(
            morph=gs.morphs.Box(
                size=(self.env_cfg.fixed_region_bounds[:, 1] - self.env_cfg.fixed_region_bounds[:, 0]).cpu(),
                pos=self.env_cfg.fixed_region_bounds.mean(dim=1).cpu(),
                fixed=True,
                collision=False
            ),
            surface=gs.surfaces.Default(color=(0.1, 0.1, 0.6, 0.4)),
        )
    
    def _set_fixed_particles(self):
        """Sets the fixed particle mask after the scene is built."""
        mpm_pos_env0 = self.initial_mpm_pos[0]
        is_in_box = ((mpm_pos_env0 >= self.env_cfg.fixed_region_bounds[:, 0]) & (mpm_pos_env0 <= self.env_cfg.fixed_region_bounds[:, 1])).all(dim=1)
        self.mpm_entity.set_free(~is_in_box)

    def reset_idx(self, envs_idx: torch.Tensor):
        """Resets specified environments to their initial state."""
        if len(envs_idx) == 0: return
            
        self.episode_length_buf[envs_idx] = 0
        self.robot.reset(envs_idx)
        self.scene.reset(envs_idx=envs_idx)

        init_bounds = torch.stack([self.initial_mpm_pos[0].min(dim=0).values, self.initial_mpm_pos[0].max(dim=0).values])
        
        eps = 1e-8
        scale = (self.env_cfg.target_shape_bounds[1] - self.env_cfg.target_shape_bounds[0]) / (init_bounds[1] - init_bounds[0] + eps)
        offset = self.env_cfg.target_shape_bounds[0] - init_bounds[0] * scale
        
        self.target_mpm_pos = self.initial_mpm_pos * scale + offset

    def reset(self) -> tuple[torch.Tensor, None]:
        """Resets all environments and returns the initial observation."""
        all_envs = torch.arange(self.env_cfg.num_envs, device=self.device)
        self.reset_idx(all_envs)
        obs, _ = self.get_observations()
        return obs, None

    def step_action(self, num_action_steps):
        """Steps the simulation for a given number of steps, with optional recording."""
        if self.general_cfg.record:
            for _ in range(num_action_steps):
                self.scene.step()
                self.recording_camera.render(rgb=True)
        else:
            for _ in range(num_action_steps): self.scene.step()

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Applies an action, steps the simulation, and returns the results."""
        self.episode_length_buf += 1
        actions = torch.clamp(actions, self.env_cfg.action_lower_bounds, self.env_cfg.action_upper_bounds)

        qpos = self.robot.entity.get_qpos()
        qpos[:, 0:2] = actions[:, 0:2]
        self.robot.apply_action(qpos)
        self.step_action(self.env_cfg.action_duration_steps)

        qpos[:, 2:4] = actions[:, 2].unsqueeze(-1)
        self.robot.apply_action(qpos)
        self.step_action(self.env_cfg.action_duration_steps)

        qpos_prev = self.robot.entity.get_qpos()
        reset_qpos = self.initial_robot_qpos.clone()
        reset_qpos[:, 0:2] = qpos_prev[:, 0:2]
        self.robot.apply_action(reset_qpos)
        self.step_action(self.env_cfg.reset_duration_steps)

        self.reset_buf = (self.episode_length_buf >= self.env_cfg.max_episode_length)
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
        """Returns privileged observations (not used in this environment)."""
        return None

    def _reward_mpm_deformation(self) -> torch.Tensor:
        """Calculates reward based on the mean distance to the target shape."""
        dist = torch.norm(self.mpm_entity.get_state().pos - self.target_mpm_pos, p=2, dim=-1).mean(dim=-1)
        return torch.exp(-dist)

# ==================================================================================================
# Optimizer-specific Trainer Classes
# ==================================================================================================
class SACTrainer:
    """A trainer for the Soft Actor-Critic (SAC) reinforcement learning algorithm."""
    def __init__(self, env: AgilityForgeEnv, sac_cfg: SacConfig, general_cfg: GeneralConfig):
        """
        Initializes the SAC trainer.
        
        Parameters:
            env (AgilityForgeEnv): The environment to train on.
            sac_cfg (SacConfig): Configuration for the SAC algorithm.
            general_cfg (GeneralConfig): General configuration.
        """
        self.env = env
        self.sac_cfg = sac_cfg
        self.general_cfg = general_cfg

    def train(self):
        """Trains the SAC agent using the OnPolicyRunner."""
        train_cfg = {
            "algorithm": {"class_name": self.sac_cfg.class_name, "gamma": self.sac_cfg.gamma, "lam": self.sac_cfg.lam, "learning_rate": self.sac_cfg.learning_rate, "entropy_coef": self.sac_cfg.entropy_coef},
            "policy": {"class_name": "ActorCritic", "actor_hidden_dims": self.sac_cfg.actor_hidden_dims, "critic_hidden_dims": self.sac_cfg.critic_hidden_dims},
            "runner": {"max_iterations": self.sac_cfg.max_iterations, "run_name": self.sac_cfg.run_name},
            "runner_class_name": self.sac_cfg.runner_class_name,
            "num_steps_per_env": self.sac_cfg.num_steps_per_env,
            "empirical_normalization": self.sac_cfg.empirical_normalization,
            "save_interval": self.sac_cfg.save_interval,
        }
        runner = OnPolicyRunner(self.env, train_cfg, self.general_cfg.log_dir, device=gs.device)
        runner.learn(num_learning_iterations=self.sac_cfg.max_iterations, init_at_random_ep_len=True)

class AdamOptimizer:
    """A trainer for gradient-based optimization using the Adam optimizer."""
    def __init__(self, env: AgilityForgeEnv, adam_cfg: AdamConfig):
        """
        Initializes the Adam optimizer trainer.
        
        Parameters:
            env (AgilityForgeEnv): The environment to train on.
            adam_cfg (AdamConfig): Configuration for the Adam optimizer.
        """
        self.env = env
        self.adam_cfg = adam_cfg
        # Actions are treated as learnable parameters
        self.actions = torch.zeros((self.env.env_cfg.max_episode_length, self.env.env_cfg.num_envs, 3), requires_grad=True, device=self.env.device)
        self.optimizer = torch.optim.Adam([self.actions], lr=self.adam_cfg.learning_rate)

    def train(self):
        """Performs gradient-based optimization using Adam."""
        for i in range(self.adam_cfg.max_iterations):
            self.optimizer.zero_grad()
            self.env.reset()
            loss = 0
            for t in range(self.env.env_cfg.max_episode_length):
                _, rewards, _, _ = self.env.step(self.actions[t])
                loss -= rewards.mean()  # We want to maximize reward, so we minimize negative reward
            loss.backward()
            self.optimizer.step()
            print(f"Iteration {i+1}/{self.adam_cfg.max_iterations}, Loss: {loss.item()}")

# ==================================================================================================
# Main Execution Block
# ==================================================================================================
def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Train a robotic forging task with a selectable optimizer.")
    parser.add_argument("--optimizer", type=str, default="sac", choices=["sac", "adam"],
                        help="The optimizer to use for training (e.g., 'sac' or 'adam').")
    return parser.parse_args()

if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_args()
    
    # Load configurations
    sim_cfg = SimConfig()
    env_cfg = EnvConfig()
    general_cfg = GeneralConfig()

    # Initialize Genesis
    gs.init(backend=gs.gpu)

    # Initialize the environment
    env = AgilityForgeEnv(sim_cfg, env_cfg, general_cfg, device=gs.device)

    # Select and initialize the trainer based on the chosen optimizer
    if args.optimizer == "sac":
        sac_cfg = SacConfig()
        trainer = SACTrainer(env, sac_cfg, general_cfg)
    elif args.optimizer == "adam":
        adam_cfg = AdamConfig()
        trainer = AdamOptimizer(env, adam_cfg)
    
    # Start recording if enabled
    if general_cfg.record:
        env.start_recording()

    # Start the training process
    trainer.train()

    # Stop recording if enabled
    if general_cfg.record:
        env.stop_recording()