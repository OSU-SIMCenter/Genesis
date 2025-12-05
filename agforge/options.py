import torch
import numpy as np
from pint import UnitRegistry
from typing import Tuple, List

import genesis as gs
from genesis.options.options import Options
from genesis.options import ProfilingOptions, SimOptions, MPMOptions, VisOptions, ViewerOptions

ureg = UnitRegistry()

GENERATED_ROBOT_XML_PATH = "genesis/assets/xml/agforge_demo.xml"

def convert_to_robot_time_units(quantity: ureg.Quantity, time_unit_str: str) -> float:
    """Convert a quantity to use a specified time unit instead of seconds."""
    return quantity.to(str(quantity.to_base_units().units).replace('second', time_unit_str)).magnitude

class MaterialOptions(Options):
    """Parameters for the elasto-plastic material."""
    E: float = 200.e9 * 0.25
    nu: float = 0.28
    rho: float = 8000.
    von_mises_yield_stress: float = 190.e6 * 0.1

class EnvOptions(Options):
    """Parameters related to the RL environment and task."""
    num_envs: int = 1
    max_episode_length: int
    action_duration_steps: int
    reset_duration_steps: int
    num_actions: int = 1
    action_lower_bounds: torch.Tensor
    action_upper_bounds: torch.Tensor
    fixed_region_bounds: torch.Tensor
    target_shape_bounds: torch.Tensor
    class Config:
        arbitrary_types_allowed = True


class SacOptions(Options):
    """Hyperparameters for the SAC (PPO) reinforcement learning algorithm."""
    class_name: str = "PPO"
    gamma: float = 0.99
    lam: float = 0.95
    learning_rate: float = 5e-4
    entropy_coef: float = 0.1
    actor_hidden_dims: List[int] = [256, 128]
    critic_hidden_dims: List[int] = [256, 128]
    max_iterations: int = 1000
    run_name: str = "agforge_parametric"
    runner_class_name: str = "OnPolicyRunner"
    num_steps_per_env: int = 1
    save_interval: int = 50
    empirical_normalization: bool = False

class AdamOptions(Options):
    """Hyperparameters for the Adam gradient-based optimizer."""
    learning_rate: float = 1e-3
    max_iterations: int = 1000

class GeneralOptions(Options):
    """General settings for visualization, logging, and recording."""
    show_viewer: bool = True
    record: bool = False
    log_dir: str = "logs/agforge_parametric"

class RobotOptions(Options):
    """Parameters for the robot arm in the MuJoCo XML file."""
    time_unit_str: str = "rtu"
    robot_time_to_seconds: float = 1.

    # Declare fields for pydantic
    cylinder_diameter: float = None
    cylinder_radius: float = None
    cylinder_height: float = None
    cylinder_pos: object = None
    cylinder_euler: tuple = None
    base_grid_density: int = None
    mpm_lower_bound: tuple = None
    mpm_upper_bound: tuple = None
    fixed_region_bounds: object = None
    target_shape_bounds: object = None
    action_lower_bounds: object = None
    action_upper_bounds: object = None
    _kp: object = None
    _kv: object = None
    clamp_force: float = 196200.0

    class Config:
        arbitrary_types_allowed = True
    
    def model_post_init(self, __context: any) -> None:
        # Temporarily get values needed for calculations
        ureg.define(f"{self.time_unit_str} = {self.robot_time_to_seconds} * second")

        # --- Perform all calculations first ---
        self.cylinder_diameter = (1.0 * ureg.inch).to(ureg.meter).magnitude
        self.cylinder_radius = self.cylinder_diameter / 2
        self.cylinder_height = 6 * self.cylinder_radius
        self.cylinder_pos = np.array([0.0, 0.0, 6 * self.cylinder_radius])
        self.cylinder_euler = (0.0, 90.0, 0.0)

        self.base_grid_density = int(7 / self.cylinder_diameter)
        dx = 1.0 / self.base_grid_density
        mpm_solver_padding = 3 * dx
        mpm_x_padding_lower = self.cylinder_height * 0.85
        mpm_x_padding_upper = self.cylinder_height * 0.52
        mpm_yz_padding = self.cylinder_radius * 1.6
        mpm_lower_offset = np.array([mpm_x_padding_lower, mpm_yz_padding, mpm_yz_padding]) + mpm_solver_padding
        mpm_upper_offset = np.array([mpm_x_padding_upper, mpm_yz_padding, mpm_yz_padding]) + mpm_solver_padding
        self.mpm_lower_bound = tuple(self.cylinder_pos - mpm_lower_offset)
        self.mpm_upper_bound = tuple(self.cylinder_pos + mpm_upper_offset)

        fixed_region_size = np.array([0.25 * self.cylinder_height, 4 * self.cylinder_radius, 4 * self.cylinder_radius])
        fixed_region_center = self.cylinder_pos + np.array([0.5 * self.cylinder_height, 0, 0])
        self.fixed_region_bounds = torch.tensor(np.array([
            fixed_region_center - fixed_region_size / 2,
            fixed_region_center + fixed_region_size / 2,
        ])).T

        target_guide_box_size = np.array([0.75 * self.cylinder_height, 1.2 * self.cylinder_radius, 1.2 * self.cylinder_radius])
        target_guide_box_pos = self.cylinder_pos + np.array([-0.375 * self.cylinder_height, 0, 0])
        self.target_shape_bounds = torch.tensor(np.array([
            target_guide_box_pos - target_guide_box_size / 2,
            target_guide_box_pos + target_guide_box_size / 2,
        ]))

        action_x_center = self.cylinder_pos[0] - self.cylinder_height * 0.75
        action_x_width = 0.06
        action_hinge_angle_limit_deg = 40.0
        action_gripper_open_val = 20 * self.cylinder_radius
        action_gripper_closed_val = 2 * self.cylinder_radius
        self.action_lower_bounds = torch.tensor([
            action_x_center - action_x_width / 2,
            -action_hinge_angle_limit_deg,
            action_gripper_closed_val,
        ])
        self.action_upper_bounds = torch.tensor([
            action_x_center + action_x_width / 2,
            action_hinge_angle_limit_deg,
            action_gripper_open_val,
        ])

        
        kp_val = 0.2
        kv_val = 2. * ((kp_val * 10.) ** 0.5)
        self._kp = kp_val * ureg.newton * ureg.meter
        self._kv = kv_val * ureg.newton * ureg.meter * ureg.second

    @property
    def kp(self) -> float:
        """Get stiffness in robot units (N·m)"""
        return convert_to_robot_time_units(self._kp, self.time_unit_str)
        
    @property
    def kv(self) -> float:
        """Get damping with time unit converted to the specified time unit (N·m·rtu)"""
        return convert_to_robot_time_units(self._kv, self.time_unit_str)


class AgilityForgeOptions(Options):
    """Aggregated configuration for the AgilityForge environment."""
    mat: MaterialOptions = MaterialOptions()
    sac: SacOptions = SacOptions()
    adam: AdamOptions = AdamOptions()
    performance_mode: bool = True

    # Declare fields for pydantic
    sim: object = None
    robot: object = None
    mpm: object = None
    env: object = None
    general: object = None
    profiling: object = None
    vis: object = None
    viewer: object = None

    class Config:
        arbitrary_types_allowed = True

    def model_post_init(self, __context: any) -> None:
        # --- Perform all calculations first ---
        self.sim = SimOptions(
            dt=1.4e-6 * 16,
            substeps=16,
            gravity=(0, 0, 0),
        )
        self.robot = RobotOptions(robot_time_to_seconds=0.1 * self.sim.substeps / self.sim.dt)
        self.mpm = MPMOptions(
            grid_density=self.robot.base_grid_density,
            particle_size=0.8 * 0.01 * 64.0 / self.robot.base_grid_density,
            lower_bound=self.robot.mpm_lower_bound,
            upper_bound=self.robot.mpm_upper_bound,
        )
        self.env = EnvOptions(
            num_envs=1,
            max_episode_length=int(100. / self.sim.dt),
            action_duration_steps=40,
            reset_duration_steps=20,
            num_actions=3,
            action_lower_bounds=self.robot.action_lower_bounds,
            action_upper_bounds=self.robot.action_upper_bounds,
            fixed_region_bounds=self.robot.fixed_region_bounds,
            target_shape_bounds=self.robot.target_shape_bounds,
        )
        self.general = GeneralOptions(
            show_viewer=True,
            record=False,
        )
        self.profiling = ProfilingOptions(enabled=True, show_FPS=False)

        camera_lookat = tuple(self.robot.cylinder_pos)
        camera_pos_offset = np.array([-2.75 * self.robot.cylinder_height, -8.0 * self.robot.cylinder_radius, 3.0 * self.robot.cylinder_height])
        camera_pos = tuple(self.robot.cylinder_pos + camera_pos_offset)

        self.viewer = ViewerOptions(
            camera_pos=camera_pos,
            camera_lookat=camera_lookat,
            res=(1280, 720),
        )

        self.vis = VisOptions(
            particle_render_fraction=0.3,
            show_world_frame=False,
            visualize_mpm_boundary=False,
            visualize_mpm_grid=False,
            render_particle_as="sphere",
            shadow=False,
            plane_reflection=False,
        )

        if not self.performance_mode:
            self.vis.show_world_frame = True
            self.vis.visualize_mpm_boundary = True
            self.vis.visualize_mpm_grid = True
            self.vis.shadow = True
            self.vis.plane_reflection = True
            self.vis.particle_render_fraction = 1.0


class TrainingOptions(AgilityForgeOptions):
    """Aggregated configuration for training."""

class TeleopOptions(AgilityForgeOptions):
    """Aggregated configuration for teleoperation."""
    _slider_speed: float = 0.0034
    _hinge_speed: float = 0.08
    _gripper_speed: float = 0.002

    @property
    def slider_speed(self) -> float:
        return self._slider_speed

    @property
    def hinge_speed(self) -> float:
        return self._hinge_speed

    @property
    def gripper_speed(self) -> float:
        return self._gripper_speed
