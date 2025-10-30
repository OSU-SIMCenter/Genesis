import torch
import numpy as np
from pint import UnitRegistry
from typing import Tuple, List

import genesis as gs
from genesis.options import Options, ProfilingOptions, SimOptions, MPMOptions, VisOptions, ViewerOptions

ureg = UnitRegistry()

GENERATED_ROBOT_XML_PATH = "genesis/assets/xml/agforge_demo.xml"

# --------------------------------------------------------------------------
# Base Parametric Relationships
# --------------------------------------------------------------------------
CYLINDER_DIAMETER = (1.0 * ureg.inch).to(ureg.meter).magnitude
CYLINDER_RADIUS = CYLINDER_DIAMETER / 2
CYLINDER_HEIGHT = 6 * CYLINDER_RADIUS
CYLINDER_POS = np.array([0.0, 0.0, 6 * CYLINDER_RADIUS])
CYLINDER_EULER = (0.0, 90.0, 0.0) # Orients the cylinder along the X-axis

# --- MPM Boundary Calculation with Solver Padding ---
BASE_GRID_DENSITY = int(7 / CYLINDER_DIAMETER)
DX = 1.0 / BASE_GRID_DENSITY  # Cell size
MPM_SOLVER_PADDING = 3 * DX   # Internal padding used by the MPM solver

# Asymmetrical user-defined padding
MPM_X_PADDING_LOWER = CYLINDER_HEIGHT * 0.85
MPM_X_PADDING_UPPER = CYLINDER_HEIGHT * 0.52
MPM_YZ_PADDING = CYLINDER_RADIUS * 1.6

# Combine user padding with solver padding
MPM_LOWER_OFFSET = np.array([MPM_X_PADDING_LOWER, MPM_YZ_PADDING, MPM_YZ_PADDING]) + MPM_SOLVER_PADDING
MPM_UPPER_OFFSET = np.array([MPM_X_PADDING_UPPER, MPM_YZ_PADDING, MPM_YZ_PADDING]) + MPM_SOLVER_PADDING
MPM_LOWER_BOUND = tuple(CYLINDER_POS - MPM_LOWER_OFFSET)
MPM_UPPER_BOUND = tuple(CYLINDER_POS + MPM_UPPER_OFFSET)

# Defines the region where particles are held stationary
FIXED_REGION_SIZE = np.array([0.25 * CYLINDER_HEIGHT, 4 * CYLINDER_RADIUS, 4 * CYLINDER_RADIUS])
FIXED_REGION_CENTER = CYLINDER_POS + np.array([0.5 * CYLINDER_HEIGHT, 0, 0])
FIXED_REGION_BOUNDS = torch.tensor(np.array([
    FIXED_REGION_CENTER - FIXED_REGION_SIZE / 2,
    FIXED_REGION_CENTER + FIXED_REGION_SIZE / 2,
])).T # Transpose is necessary to get the shape [dim, min/max]

# Defines the target shape for the reward function and visualization
TARGET_GUIDE_BOX_SIZE = np.array([0.75 * CYLINDER_HEIGHT, 1.2 * CYLINDER_RADIUS, 1.2 * CYLINDER_RADIUS])
TARGET_GUIDE_BOX_POS = CYLINDER_POS + np.array([-0.375 * CYLINDER_HEIGHT, 0, 0])
TARGET_SHAPE_BOUNDS = torch.tensor(np.array([
    TARGET_GUIDE_BOX_POS - TARGET_GUIDE_BOX_SIZE / 2,
    TARGET_GUIDE_BOX_POS + TARGET_GUIDE_BOX_SIZE / 2,
]))

# Defines the robot's action space limits
ACTION_X_CENTER = CYLINDER_POS[0] - CYLINDER_HEIGHT * 0.75
ACTION_X_WIDTH = 0.06
ACTION_HINGE_ANGLE_LIMIT_DEG = 40.0
ACTION_GRIPPER_OPEN_VAL = 20 * CYLINDER_RADIUS   # Gripper open position
ACTION_GRIPPER_CLOSED_VAL = 2 * CYLINDER_RADIUS  # Gripper closed position
ACTION_LOWER_BOUNDS = torch.tensor([
    ACTION_X_CENTER - ACTION_X_WIDTH / 2,
    -ACTION_HINGE_ANGLE_LIMIT_DEG,
    ACTION_GRIPPER_OPEN_VAL,
])
ACTION_UPPER_BOUNDS = torch.tensor([
    ACTION_X_CENTER + ACTION_X_WIDTH / 2,
    ACTION_HINGE_ANGLE_LIMIT_DEG,
    ACTION_GRIPPER_CLOSED_VAL,
])

# Defines the camera's viewpoint relative to the main object
CAMERA_LOOKAT = tuple(CYLINDER_POS)
CAMERA_POS_OFFSET = np.array([-2.75 * CYLINDER_HEIGHT, -8.0 * CYLINDER_RADIUS, 3.0 * CYLINDER_HEIGHT])
CAMERA_POS = tuple(CYLINDER_POS + CAMERA_POS_OFFSET)

# --------------------------------------------------------------------------
# Configuration Options
# --------------------------------------------------------------------------

class MaterialOptions(Options):
    """Parameters for the elasto-plastic material."""
    E: float = 200.e9 * 0.25
    nu: float = 0.28
    rho: float = 8000.
    von_mises_yield_stress: float = 190.e6 * 0.1

class EnvOptions(Options):
    """Parameters related to the RL environment and task."""
    num_envs: int = 1
    max_episode_length: int = int(100. / (1.5e-6 * 64)) # SimConfig.dt
    action_duration_steps: int = 40
    reset_duration_steps: int = 23
    num_actions: int = 3
    action_lower_bounds: torch.Tensor = ACTION_LOWER_BOUNDS
    action_upper_bounds: torch.Tensor = ACTION_UPPER_BOUNDS
    fixed_region_bounds: torch.Tensor = FIXED_REGION_BOUNDS
    target_shape_bounds: torch.Tensor = TARGET_SHAPE_BOUNDS
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
    camera_pos: Tuple[float, float, float] = CAMERA_POS
    camera_lookat: Tuple[float, float, float] = CAMERA_LOOKAT

def convert_to_robot_time_units(quantity: ureg.Quantity, time_unit_str: str) -> float:
    """Convert a quantity to use a specified time unit instead of seconds."""
    return quantity.to(str(quantity.to_base_units().units).replace('second', time_unit_str)).magnitude

KP = 0.2
KV = 2. * ((KP * 10.) ** 0.5)

class RobotOptions(Options):
    """Parameters for the robot arm in the MuJoCo XML file."""
    time_unit_str: str = "rtu"
    robot_time_to_seconds: float = 1.
    _kp: ureg.Quantity = KP * ureg.newton * ureg.meter
    _kv: ureg.Quantity = KV * ureg.newton * ureg.meter * ureg.second

    def __init__(self, **data):
        super().__init__(**data)
        ureg.define(f"{self.time_unit_str} = {self.robot_time_to_seconds} * second")

    @property
    def kp(self) -> float:
        """Get stiffness in robot units (N·m)"""
        return convert_to_robot_time_units(self._kp, self.time_unit_str)
        
    @property
    def kv(self) -> float:
        """Get damping with time unit converted to the specified time unit (N·m·rtu)"""
        return convert_to_robot_time_units(self._kv, self.time_unit_str)

    class Config:
        arbitrary_types_allowed = True


class AgilityForgeOptions(Options):
    """Aggregated configuration for the AgilityForge environment."""
    sim: SimOptions = SimOptions(
        dt=1.5e-6 * 64,
        substeps=64,
        gravity=(0, 0, 0),
    )
    mpm: MPMOptions = MPMOptions(
        grid_density=BASE_GRID_DENSITY,
        particle_size=0.8 * 0.01 * 64.0 / BASE_GRID_DENSITY,
        lower_bound=MPM_LOWER_BOUND,
        upper_bound=MPM_UPPER_BOUND,
    )
    env: EnvOptions = EnvOptions()
    general: GeneralOptions = GeneralOptions()
    vis: VisOptions = VisOptions(
        performance_mode=True,
        particle_render_fraction=0.5,
        camera_res=(1280, 720),
        show_world_frame=False,
        visualize_mpm_boundary=False,
        visualize_mpm_grid=False,
        render_particle_as="sphere",
        shadow=False,
        plane_reflection=False,
    )
    viewer: ViewerOptions = ViewerOptions(
        camera_pos=CAMERA_POS,
        camera_lookat=CAMERA_LOOKAT,
    )
    robot: RobotOptions = RobotOptions()
    mat: MaterialOptions = MaterialOptions()
    sac: SacOptions = SacOptions()
    adam: AdamOptions = AdamOptions()
    profiling: ProfilingOptions = ProfilingOptions()
    performance_mode: bool = True

    def __init__(self, **data):
        super().__init__(**data)
        if not self.performance_mode:
            self.vis.show_world_frame = True
            self.vis.visualize_mpm_boundary = True
            self.vis.visualize_mpm_grid = True
            self.vis.shadow = True
            self.vis.plane_reflection = True
            self.vis.particle_render_fraction = 1.0
            self.vis.camera_res = (1280, 720)


class TrainingOptions(AgilityForgeOptions):
    """Aggregated configuration for training."""
    pass


class TeleopOptions(AgilityForgeOptions):
    """Aggregated configuration for teleoperation."""
    sim: SimOptions = SimOptions(
        dt=1.4e-6 * 16,
        substeps=16,
        gravity=(0, 0, 0),
    )
    env: EnvOptions = EnvOptions(num_envs=1)
    general: GeneralOptions = GeneralOptions(show_viewer=True, record=False)
    robot: RobotOptions = RobotOptions(
        robot_time_to_seconds=0.03 / 1.4e-6
    )
    profiling: ProfilingOptions = ProfilingOptions(
        enabled=True,
        profiling_options=ProfilingOptions(enabled=True, show_FPS=False)
    )

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
