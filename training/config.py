import torch
import numpy as np
from pint import UnitRegistry
from dataclasses import dataclass, field
from typing import Tuple, List

ureg = UnitRegistry()

GENERATED_ROBOT_XML_PATH = "genesis/assets/xml/agforge_demo.xml"

# --------------------------------------------------------------------------
# Base Parametric Relationships
# --------------------------------------------------------------------------
CYLINDER_DIAMETER = (1.0 * ureg.inch).to(ureg.meter).magnitude
CYLINDER_RADIUS = CYLINDER_DIAMETER / 2
CYLINDER_HEIGHT = 8 * CYLINDER_RADIUS
CYLINDER_POS = np.array([0.0, 0.0, 6 * CYLINDER_RADIUS])
CYLINDER_EULER = (0.0, 90.0, 0.0) # Orients the cylinder along the X-axis

# --- MPM Boundary Calculation with Solver Padding ---
BASE_GRID_DENSITY = int(10 / CYLINDER_DIAMETER)
DX = 1.0 / BASE_GRID_DENSITY  # Cell size
MPM_SOLVER_PADDING = 3 * DX   # Internal padding used by the MPM solver

# Asymmetrical user-defined padding
MPM_X_PADDING_LOWER = CYLINDER_HEIGHT * 0.85
MPM_X_PADDING_UPPER = CYLINDER_HEIGHT * 0.6
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
ACTION_GRIPPER_OPEN_VAL = 19 * CYLINDER_RADIUS   # Gripper open position
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
# Configuration Dataclasses
# --------------------------------------------------------------------------
SUBSTEPS = 64

@dataclass
class SimConfig:
    """Parameters related to the core physics simulation."""
    dt: float = 1.5e-6 * SUBSTEPS
    substeps: int = SUBSTEPS
    gravity: Tuple[float, float, float] = (0, 0, 0)
    grid_density: int = BASE_GRID_DENSITY
    lower_bound: Tuple[float, float, float] = MPM_LOWER_BOUND
    upper_bound: Tuple[float, float, float] = MPM_UPPER_BOUND

@dataclass
class EnvConfig:
    """Parameters related to the RL environment and task."""
    num_envs: int = 1
    max_episode_length: int = int(100. / SimConfig.dt)
    action_duration_steps: int = 40
    reset_duration_steps: int = 23
    num_actions: int = 3
    action_lower_bounds: torch.Tensor = ACTION_LOWER_BOUNDS
    action_upper_bounds: torch.Tensor = ACTION_UPPER_BOUNDS
    fixed_region_bounds: torch.Tensor = FIXED_REGION_BOUNDS
    target_shape_bounds: torch.Tensor = TARGET_SHAPE_BOUNDS

@dataclass
class SacConfig:
    """Hyperparameters for the SAC (PPO) reinforcement learning algorithm."""
    class_name: str = "PPO"
    gamma: float = 0.99
    lam: float = 0.95
    learning_rate: float = 5e-4
    entropy_coef: float = 0.1
    actor_hidden_dims: List[int] = field(default_factory=lambda: [256, 128])
    critic_hidden_dims: List[int] = field(default_factory=lambda: [256, 128])
    max_iterations: int = 1000
    run_name: str = "agforge_parametric"
    runner_class_name: str = "OnPolicyRunner"
    num_steps_per_env: int = 1
    save_interval: int = 50
    empirical_normalization: bool = False

@dataclass
class AdamConfig:
    """Hyperparameters for the Adam gradient-based optimizer."""
    learning_rate: float = 1e-3
    max_iterations: int = 1000

@dataclass
class GeneralConfig:
    """General settings for visualization, logging, and recording."""
    show_viewer: bool = True
    record: bool = False
    log_dir: str = "logs/agforge_parametric"
    camera_pos: Tuple[float, float, float] = CAMERA_POS
    camera_lookat: Tuple[float, float, float] = CAMERA_LOOKAT

ureg.define("robot_time_unit = 3e2 * second = rtu")  # Define custom time unit with shorthand

def convert_to_robot_time_units(quantity: ureg.Quantity) -> float:
    """Convert a quantity to use robot_time_unit instead of second for time components."""
    return quantity.to(str(quantity.to_base_units().units).replace('second', 'robot_time_unit')).magnitude

@dataclass
class RobotConfig:
    """Parameters for the robot arm in the MuJoCo XML file."""
    _kp: ureg.Quantity = 1000.0 * ureg.newton * ureg.meter  # Stiffness in SI units (N·m)
    _kv: ureg.Quantity = 50.0 * ureg.newton * ureg.meter * ureg.second  # Damping in SI units (N·m·s)
    
    @property
    def kp(self) -> float:
        """Get stiffness in robot units (N·m)"""
        return convert_to_robot_time_units(self._kp)
        
    @property
    def kv(self) -> float:
        """Get damping with time unit converted to robot_time_unit (N·m·rtu)"""
        return convert_to_robot_time_units(self._kv)