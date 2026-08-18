import torch
import numpy as np
from pint import UnitRegistry
from typing import Optional, Tuple, List

import genesis as gs
from genesis.options.options import Options
from genesis.options import ProfilingOptions, SimOptions, MPMOptions, VisOptions, ViewerOptions

ureg = UnitRegistry()

import os
import sys

# Determine path for generated assets relative to the application/script
if getattr(sys, 'frozen', False):
    # If compiled with PyInstaller
    _base_dir = os.path.dirname(sys.executable)
else:
    # If running as script
    _base_dir = os.path.dirname(os.path.abspath(__file__))

GENERATED_ROBOT_XML_PATH = os.path.join(_base_dir, "agforge_demo.xml")

def convert_to_robot_time_units(quantity: ureg.Quantity, time_unit_str: str) -> float:
    """Convert a quantity to use a specified time unit instead of seconds."""
    return quantity.to(str(quantity.to_base_units().units).replace('second', time_unit_str)).magnitude

class MaterialOptions(Options):
    """Parameters for the elasto-plastic material."""
    E: float = 200.e9 * 0.25
    nu: float = 0.28
    rho: float = 8000.
    von_mises_yield_stress: float = 190.e6 * 0.1

    # Johnson-Cook Parameters (Hot Steel ~1200C)
    use_johnson_cook: bool = True
    jc_A: float = 40.e6   # ~40 MPa (Very Hot)
    jc_B: float = 100.e6  # Reduced hardening
    jc_n: float = 0.26
    jc_C: float = 0.014
    jc_eps0: float = 1.0
    # These three were hardcoded at the JohnsonCookPlasticity call site in environment.py, which
    # silently overrode anything set here. Defaults below are exactly what that call site passed,
    # so behaviour is unchanged -- they exist so a card whose reference temperature is its
    # CALIBRATION temperature (a 316L card calibrated at 1000 C has jc_T_ref = 1273.15, not room
    # temperature) can actually be applied.
    jc_T_ref: float = 293.15
    jc_T_melt: float = 1793.0
    jc_m: float = 1.03

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
    show_target_bounds: bool = False
    particle_sampler: str = "default"
    class Config:
        arbitrary_types_allowed = True


class RLOptions(Options):
    """Hyperparameters for reinforcement learning (PPO algorithm)."""
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
    verbose: bool = True # Enable detailed logging
    log_dir: str = "logs/agforge_parametric"
    # Genesis viewer: color MPM particles by a scalar field (inferno buckets).
    visualize_particle_temperature: bool = True
    # "temperature" | "induction_depth" | "skin_weight" | "q_ind"
    particle_color_mode: str = "temperature"
    particle_temp_min: float = 293.0
    particle_temp_max: float = 1450.0
    # Depth color scale for induction_depth mode [m]; None → 3 * skin_depth
    particle_depth_max: Optional[float] = None
    # Viewer-only draw radius multiplier when toggled with [B] (physics unchanged).
    particle_render_scale_small: float = 0.3
    # Runtime toggle: plain metal spheres (skip scalar coloring / bucketing work).
    particle_simple_color: bool = False
    # Number of inferno color buckets for particle temperature visualization.
    particle_color_buckets: int = 32
    # Fraction of active particles drawn in the viewer (physics unchanged).
    particle_render_fraction: float = 1.0
    # Live thermal tuning HUD + keybinds in the Genesis viewer (adds GPU telemetry overhead).
    interactive_thermal_tuner: bool = False

class ReconstructionOptions(Options):
    """Parameters for surface reconstruction."""
    grid_res: int = 64
    backend: str = "hybrid"  # 'hybrid' or 'splashsurf'
    enabled: bool = True
    # When True, one mesh build (physics_mesh_backend) serves visual + induction SDF.
    # When False, visual uses live reconstructor (grid_res) and physics_mesh_backend separately.
    unified_mesh: bool = True
    # Surface mesh for physics SDF (and visual when unified_mesh): hybrid_low | hybrid_high | splashsurf
    physics_mesh_backend: str = "hybrid_high"
    physics_mesh_grid_res: int = 96  # hybrid_high marching-cubes grid resolution
    # Isosurface extractor: auto (CUDA→Warp, else PyVista), warp, or pyvista
    mc_backend: str = "auto"

class RobotOptions(Options):
    """Parameters for the robot arm in the MuJoCo XML file."""
    time_unit_str: str = "rtu"
    robot_time_to_seconds: float = 1.

    # Induction Coil physical placeholders (Unity Sync)
    coil_offset_x: float = -0.129891413331031800
    coil_length: float = 0.037
    # Effective solenoid radius = multiplier × billet radius. For close-coupled forging
    # coils the bore is typically ~110–125% of workpiece OD (ASM / induction heating practice).
    coil_radius_multiplier: float = 2.0
    coil_radius: Optional[float] = None  # computed in model_post_init
    coup_softness: float = 5e-4

    # Declare fields for pydantic — Optional because they are computed in model_post_init
    # (cylinder_diameter/cylinder_height may also be pre-set by the caller, see below)
    cylinder_diameter: Optional[float] = None
    cylinder_radius: Optional[float] = None
    cylinder_height: Optional[float] = None
    cylinder_pos: object = None
    cylinder_euler: Optional[tuple] = None
    # Optional jaw axial (X) full-width override [m]; None = the default 0.5*radius half-extent.
    gripper_axial_width: Optional[float] = None
    base_grid_density: Optional[int] = None
    mpm_lower_bound: Optional[tuple] = None
    mpm_upper_bound: Optional[tuple] = None
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
        # Keep pre-set stock dimensions if the caller provided them (e.g. an
        # external driver like the forge_common genesis adapter); otherwise
        # fall back to the usual defaults.
        if self.cylinder_diameter is None:
            self.cylinder_diameter = (1.0 * ureg.inch).to(ureg.meter).magnitude
        self.cylinder_radius = self.cylinder_diameter / 2
        self.coil_radius = self.coil_radius_multiplier * self.cylinder_radius
        if self.cylinder_height is None:
            self.cylinder_height = 8 * self.cylinder_radius
        self.cylinder_pos = np.array([0.0, 0.0, 6 * self.cylinder_radius])
        self.cylinder_euler = (0.0, 90.0, 0.0)

        # Cells across one billet diameter. 7 is the long-standing default; the die's
        # contact tip is ~13.5 mm, so at 7 cells/diameter (dx = 5.71 mm on a 40 mm bar)
        # the contact patch spans only ~2.4 cells. Overridable for resolution sweeps:
        #   AGF_CELLS_PER_DIAMETER=10 <cmd>
        _cpd = float(os.environ.get("AGF_CELLS_PER_DIAMETER", 7))
        self.base_grid_density = int(_cpd / self.cylinder_diameter)
        dx = 1.0 / self.base_grid_density
        mpm_solver_padding = 3 * dx
        # Headroom for the billet to ELONGATE into, as a multiple of its own length. The billet
        # is pinned at +x by `fixed_region_bounds` and grows in -x, so its usable room is
        # (mpm_x_padding_lower - cylinder_height/2).
        #
        # 0.85 gives 0.85*59 - 29.5 = 20.65 mm. The real 17-hit sequence needs 34.4 mm (59 ->
        # 93.4 mm), so the bar runs into the domain and STOPS: measured x_max freezes at 77.8 mm
        # from hit 13 onward while the real part keeps growing, and surface error doubles at that
        # exact hit. Contact-method comparisons past hit ~13 are partly measuring that wall.
        #
        # Default left at the historical 0.85 so other work in this tree is unaffected; opt in
        # for the full real replay. 1.3 leaves ~47 mm of headroom against the 34.4 mm needed
        # (minimum that fits is 1.083). Costs x grid cells only: 28 -> ~33.
        #   AGF_MPM_X_PAD_LOWER=1.3 <cmd>
        mpm_x_padding_lower = self.cylinder_height * float(
            os.environ.get("AGF_MPM_X_PAD_LOWER", "0.85"))
        mpm_x_padding_upper = self.cylinder_height * 0.52
        mpm_yz_padding = self.cylinder_radius * 1.6
        mpm_lower_offset = np.array([mpm_x_padding_lower, mpm_yz_padding, mpm_yz_padding]) + mpm_solver_padding
        mpm_upper_offset = np.array([mpm_x_padding_upper, mpm_yz_padding, mpm_yz_padding]) + mpm_solver_padding
        self.mpm_lower_bound = tuple(self.cylinder_pos - mpm_lower_offset)
        self.mpm_upper_bound = tuple(self.cylinder_pos + mpm_upper_offset)

        fixed_region_size = np.array([0.35 * self.cylinder_height, 4 * self.cylinder_radius, 4 * self.cylinder_radius])
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
    sac: RLOptions = RLOptions()  # Note: Still named 'sac' for backwards compatibility
    adam: AdamOptions = AdamOptions()
    reconstruction: ReconstructionOptions = ReconstructionOptions()
    performance_mode: bool = True

    # Optional stock-size overrides [m] for external drivers (e.g. the
    # forge_common genesis adapter). None = RobotOptions defaults (1in x 8r).
    stock_diameter: Optional[float] = None
    stock_length: Optional[float] = None
    gripper_axial_width: Optional[float] = None

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
        import math
        
        # 1. Derive timestep from CFL condition
        # CFL limit: substep_dt <= dx / c, where c = sqrt(E/rho) is the speed of sound.
        # We use a 0.95 safety margin (5% below the theoretical CFL limit).
        temp_robot = RobotOptions(robot_time_to_seconds=1.0,
                                  cylinder_diameter=self.stock_diameter,
                                  cylinder_height=self.stock_length)
        dx = 1.0 / temp_robot.base_grid_density
        c = math.sqrt(self.mat.E / self.mat.rho)
        dt_cfl = dx / c  # Theoretical max substep_dt

        # THE temporal-refinement knob. substep_dt is what actually integrates, so halving
        # this halves the integration step with dx and particle count untouched --
        # unlike base_grid_density, which moves grid, dt and particle count together.
        # (`substeps` below scales macro_dt / control-loop rate, NOT the integration step.)
        #   AGF_CFL_SAFETY=0.45 <cmd>
        cfl_safety = float(os.environ.get("AGF_CFL_SAFETY", 0.90))  # NB: 0.90 = 10% margin,
                                                  # the original '5% safety margin' comment was wrong
        substeps = 8                              # Fixed substep count for real-time teleop
        substep_dt = dt_cfl * cfl_safety          # Safe substep timestep
        macro_dt = substep_dt * substeps           # Macro timestep = substep_dt × substeps
        
        # CFL validation: guard against future material/grid changes silently breaking stability
        assert substep_dt < dt_cfl, (
            f"CFL violation: substep_dt={substep_dt:.3e} >= dt_CFL={dt_cfl:.3e} "
            f"(E={self.mat.E:.1e}, rho={self.mat.rho:.0f}, dx={dx:.4e}, c={c:.1f})"
        )
        
        # 2. Derive thermal_time_scale from thermal CFL condition
        # Thermal diffusion stability (3D explicit FTCS): substep_dt <= dx² / (6 · α_scaled)
        # where α_scaled = α_base · S_T.  Rearranging: S_T <= dx² / (6 · α_base · substep_dt)
        #
        # Use worst-case (highest) thermal diffusivity — room-temp AISI 4340 steel:
        #   k(293K) = 44 W/m·K,  ρ = 7850 kg/m³,  Cp(293K) = 450 J/kg·K
        #   α_worst = k / (ρ · Cp) ≈ 1.245e-5 m²/s
        # At forging temps (1000K+), α drops to ~4.6e-6, so room temp is the binding limit.
        alpha_worst = 44.0 / (7850.0 * 450.0)  # ~1.245e-5 m²/s
        S_T_max = dx**2 / (6.0 * alpha_worst * substep_dt)
        
        # Fraction of the explicit-diffusion CFL limit used as the thermal time-scale. Higher =
        # faster wall-clock thermal evolution (heating AND cooling together, so the balance is
        # preserved); explicit FTCS develops checkerboard oscillations above ~60-70% of CFL, keep <=~0.4.
        thermal_cfl_fraction = 0.25
        thermal_time_scale = S_T_max * thermal_cfl_fraction
        
        # Thermal CFL validation
        alpha_scaled = alpha_worst * thermal_time_scale
        dt_thermal_cfl = dx**2 / (6.0 * alpha_scaled)
        assert substep_dt < dt_thermal_cfl, (
            f"Thermal CFL violation: substep_dt={substep_dt:.3e} >= dt_thermal={dt_thermal_cfl:.3e} "
            f"(α_worst={alpha_worst:.3e}, S_T={thermal_time_scale:.0f}, dx={dx:.4e})"
        )
        
        self.sim = SimOptions(
            dt=macro_dt,
            substeps=substeps,
            gravity=(0, 0, 0),
            check_bounds=not self.performance_mode,
        )
        # robot_time_to_seconds is derived from dt, and it feeds convert_to_robot_time_units
        # for the PD gains _kp/_kv. That means changing cfl_safety silently changes the
        # CONTROLLER as well as the integrator, so a timestep sweep is not single-variable.
        # Pin this to hold the control problem fixed while refining dt:
        #   AGF_ROBOT_TIME_TO_SECONDS=48611.1 <cmd>
        _rtu = 0.1 * self.sim.substeps / self.sim.dt
        _rtu = float(os.environ.get("AGF_ROBOT_TIME_TO_SECONDS", _rtu))
        self.robot = RobotOptions(robot_time_to_seconds=_rtu,
                                  cylinder_diameter=self.stock_diameter,
                                  cylinder_height=self.stock_length,
                                  gripper_axial_width=self.gripper_axial_width)
        self.mpm = MPMOptions(
            grid_density=self.robot.base_grid_density,
            # PPC = divisor³ : 2.0 -> 8 PPC, 3.0 -> 27 PPC. Pinned to dx by default, which
            # is why a grid sweep alone cannot separate resolution from particle density.
            #   AGF_PPC_DIVISOR=3.0 <cmd>
            particle_size=dx / float(os.environ.get("AGF_PPC_DIVISOR", 2.0)),
            lower_bound=self.robot.mpm_lower_bound,
            upper_bound=self.robot.mpm_upper_bound,
            # CPIC resolves rigid contact INSIDE g2p (base_mpm_solver.g2p): it corrects
            # `grid_vel` before that value accumulates into new_vel AND new_C, so the
            # correction reaches the affine field and hence F. That is the same pathway the
            # 'particle' contact mode uses -- which is why the coupler forbids both at once.
            # Toggle to isolate that pathway's effect on volume conservation:
            #   AGF_ENABLE_CPIC=0 <cmd>
            enable_CPIC=bool(int(os.environ.get("AGF_ENABLE_CPIC", "1"))),
            enable_thermal=True,
            # AGF_BILLET_TEMP_K. NOTE this is a MECHANICAL setting, not only a thermal one:
            # the flow stress is temperature-coupled through the Johnson-Cook melting term in
            # materials.py, so with jc_T_ref at its 293.15 default any temperature above ~293 K
            # softens the material. The real billet is ~960 C at blow 1 (measured), i.e. 1233 K,
            # against the 293.0 K default this has always run at. Default unchanged.
            default_initial_temperature=float(os.environ.get("AGF_BILLET_TEMP_K", 293.0)),
            thermal_time_scale=thermal_time_scale,
            # Fixed-end (truncated-domain) BC: the held end conducts into the unsimulated
            # rod (Robin BC on the cut plane) instead of being exposed to air.
            enable_fixed_end_bc=True,
            thermal_contact_conductivity=3000.0,  # W/(m²K); reduced from 5000 — less aggressive die chill on light contact
            fixed_end_x_cut=float(self.robot.cylinder_pos[0] + self.robot.cylinder_height / 2.0),
            fixed_end_conduction_length=0.08,  # L_eff [m]; larger = weaker held-end conduction sink
            fixed_end_ambient=293.0,
            fixed_end_blend=0.0,
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
            particle_render_fraction=1.0,
            show_world_frame=False,
            visualize_mpm_boundary=False,
            visualize_mpm_grid=False,
            render_particle_as="sphere",
            shadow=False,
            plane_reflection=False,
        )

        if not self.performance_mode:
            self.vis.show_world_frame = True
            self.vis.visualize_mpm_grid = True
            self.vis.shadow = True
            self.vis.plane_reflection = True
            self.vis.particle_render_fraction = 1.0


class TrainingOptions(AgilityForgeOptions):
    """Aggregated configuration for training."""

class StrikeOptions(Options):
    """Parameters for the approaching and pressing stage."""
    # Approach speed is calculated parametrically in TeleopOptions.model_post_init based on CFL
    approach_speed: float = 0.0 # Placeholder
    contact_force_threshold: float = 150.0 # Force threshold to detect contact
    
    target_strain: float = 0.5 # 50% reduction
    # Overridable for stability/fidelity sweeps without editing code:
    #   AGF_PRESSING_SPEED=5.0 <cmd>
    # Real presses run 0.02-0.5 m/s; 25.0 is a teleop-era literal (not CFL-derived --
    # approach_speed is, this is not). At 25.0 the two jaws close at 50 m/s against a
    # 100 m/s max_particle_velocity abort, i.e. only 2x headroom.
    pressing_speed: float = float(os.environ.get("AGF_PRESSING_SPEED", 25.0)) # m/s
    
    # Force Balance Control
    # 5e-5 was robust. 1.5e-4 is peak performance but near instability (2e-4).
    # User selected 1.5e-4 for maximum benchmark results.
    force_balance_gain: float = 1.5e-4 
    
    # Safety Limits
    max_force_imbalance: float = 20000.0 # 20 kN% compression
    # AGF_MAX_FORCE raises/disables the press control stop (strike_controller.py:663 ends
    # PRESSING when force_L or force_R exceeds this). Default is unchanged at 200 kN.
    # Measured 2026-08-13: this stop FIRES on 7 of 17 hits for g1_grid_prod and 14 of 17
    # for p3_pg2p_pos, and trip-count correlates -0.994 with elongation shortfall across
    # arms -- so arms may be partly ranked by how often they trip it. Set high to test.
    max_force: float = float(os.environ.get("AGF_MAX_FORCE", 200000.0)) # 20 tons (200kN)
    # AGF_PRESSING_TIMEOUT. This is WALL-CLOCK, not sim time, so it binds harder as the press
    # is slowed or the grid refined. It already fired once at 25 m/s, and it becomes the BINDING
    # stop as soon as AGF_MAX_FORCE removes the force stop -- swapping one arm-biased truncation
    # for another. Raise it whenever max_force is raised. Default unchanged at 30 s.
    pressing_timeout: float = float(os.environ.get("AGF_PRESSING_TIMEOUT", 30.0)) # seconds
    approaching_timeout: float = 30.0 # seconds (Increased to avoid timeout)
    release_timeout: float = 10.0 # seconds
    hold_steps: int = 15 # Number of steps to hold position after reaching target before releasing
    post_release_steps: int = 10 # steps

class SafetyOptions(Options):
    """Parameters for simulation stability checks."""
    enabled: bool = True
    strict_tracing_enabled: bool = True # Gating flag for heavy threshold bounds checking
    
    # Thresholds for early detection (Mid-blowup catching)
    # Overridable: AGF_MAX_PARTICLE_VELOCITY=400 <cmd>
    # NB this is a heuristic tripwire, not the numerical limit: the CFL velocity bound is
    # dx/substep_dt ~= 2778 m/s and the material sound speed is 2500 m/s, so 100 m/s is 4%
    # of sonic despite the 'Supersonic catch' label.
    max_particle_velocity: float = float(os.environ.get("AGF_MAX_PARTICLE_VELOCITY", 100.0)) # m/s
    max_temperature: float = 4000.0      # Thermal runaway threshold
    min_temperature: float = 0.0         # Thermal collapse threshold
    
    check_nan: bool = True
    auto_reset: bool = True
    check_interval: int = 10  # IDLE (no heating): check every N physics steps
    # When heating with no active strike, use this longer interval.
    heating_idle_check_interval: int = 25
    # During strike phases (still catches runaway physics; APPROACHING uses approaching_check_interval).
    strike_check_interval: int = 3
    approaching_check_interval: int = 5


class TeleopPerformanceOptions(Options):
    """Runtime tuning for teleop loop throughput (viewer + Unity IO)."""
    # Sim / websocket loop cap (physics may step slower when viewer is expensive).
    target_physics_fps: int = 60
    # Cap Genesis viewer redraws independently of physics (0 = every physics step).
    target_viewer_fps: int = 10
    # Scale viewer resolution (1.0 = full; 0.75 → 960×540 from 1280×720).
    viewer_res_scale: float = 1.0
    # Force OpenGL backend: "egl", "glx", or "osmesa". None = Genesis auto-fallback.
    # On WSL, unset + wsl_prefer_glx=True sets "glx" before viewer init.
    opengl_platform: Optional[str] = None
    wsl_prefer_glx: bool = True
    # WSLg: Mesa defaults to llvmpipe without /dev/dri; D3D12 uses /dev/dxg for GPU GL.
    wsl_use_d3d12: bool = True
    # Only sleep when ahead of this loop rate (False = legacy fixed sleep every iteration).
    smart_physics_pacing: bool = True
    # Refresh q_ind color vmax every N viewer frames (avoids per-frame quantile).
    q_ind_vmax_refresh_interval: int = 15
    # During live strike MC, sync viewer mesh overlay every N physics steps (1 = every step).
    mesh_overlay_sync_stride: int = 2
    # pyrender sphere subdivisions for particle buckets (0 = cheaper, 1 = smoother).
    particle_sphere_subdivisions: int = 1
    # Viewer FPS caps cycled by [T] (10 → 20 → 30).
    viewer_fps_cycle: List[int] = [10, 20, 30]

    # Unity IO: vertex temperature kNN map every N websocket frames (idle, no heating).
    vertex_temp_io_interval: int = 3
    # Heating+idle: less frequent vertex temp kNN (temps change slowly).
    vertex_temp_io_interval_heating_idle: int = 6
    # During strikes: every N websocket frames (still responsive for Unity).
    vertex_temp_io_interval_strike: int = 2
    # While heating+idle, reuse the last mesh snapshot every N websocket frames.
    mesh_io_interval_heating_idle: int = 3
    # Log OpenGL platform/renderer after viewer init (helps spot llvmpipe on WSL).
    log_opengl_info: bool = True


class AdaptiveControlConfig(Options):
    """Configuration for adaptive control gains."""
    base_kp: float = 5000.0
    base_kv: float = 200.0
    mass_scale_factor: float = 1.0

class TeleopOptions(AgilityForgeOptions):
    """Aggregated configuration for teleoperation."""
    strike: StrikeOptions = StrikeOptions()
    adaptive_control: AdaptiveControlConfig = AdaptiveControlConfig()
    safety: SafetyOptions = SafetyOptions()
    performance: TeleopPerformanceOptions = TeleopPerformanceOptions()
    print_profiling_on_exit: bool = True  # Print profiler visualizations on shutdown
    
    # Induction Heater physical configurations
    # NOTE: heating_power is now a PEAK VOLUMETRIC POWER DENSITY [W/m^3], i.e. the coil field
    # intensity at the surface directly under the coil center. It is geometry-independent: a
    # partially- and a fully-inserted billet heat at the same surface rate (no total-power
    # funneling). Deposition profile: q_peak * exp(-2*depth/skin_depth) * f_axial(x).
    # Physics-tuned so forging temp is reachable: radiation (~T^4) and held-end conduction cap the
    # steady-state ceiling, so q_peak must be high enough to beat them at ~1200C. Fine-tune the
    # exact value to the observed idle-heating plateau.
    heating_power: float = 2.5e8  # Peak volumetric power density [W/m^3]
    skin_depth: Optional[float] = None # Calculated parametrically based on cylinder radius
    thermal_visual_fade: bool = True  # Display-only: fade held-end color to <=900K at the seam (physics unchanged)
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

    def model_post_init(self, __context: any) -> None:
        # Initialize base options (Sim, Robot, MPM, etc.)
        super().model_post_init(__context)

        self.general.interactive_thermal_tuner = True

        perf = self.performance
        if perf.viewer_res_scale not in (None, 1.0):
            scale = float(perf.viewer_res_scale)
            w, h = self.viewer.res if self.viewer.res is not None else (1280, 720)
            self.viewer.res = (
                max(320, int(w * scale)),
                max(240, int(h * scale)),
            )
        if perf.target_viewer_fps:
            self.viewer.max_FPS = int(perf.target_viewer_fps)
            self.viewer.refresh_rate = int(perf.target_viewer_fps)
        
        # Parametric Approach Speed Calculation
        # User requested max safe ratio = 0.35
        # v_approach = ratio * (dx / dt)
        
        # Overridable so the approach-speed hypothesis can be tested:
        #   AGF_APPROACH_CFL_RATIO=0.05 <cmd>
        # 0.35 * (dx/dt) = ~243 m/s per jaw, against a 100 m/s max_particle_velocity abort.
        # Derived for GRID stability; per-particle contact samplers see the raw jaw velocity.
        target_cfl_ratio = float(os.environ.get("AGF_APPROACH_CFL_RATIO", 0.35))
        dx = 1.0 / self.robot.base_grid_density
        dt = self.sim.dt
        
        # Calculate optimal safe speed
        parametric_speed = target_cfl_ratio * (dx / dt)
        
        # Override default
        self.strike.approach_speed = parametric_speed
        
        # Skin depth = hot reference depth of a well-designed MF through-heating coil. Eddy-current
        # power deposits as exp(-2d/delta) with delta = sqrt(rho_e/(pi*f*mu)); for hot (above-Curie,
        # non-magnetic) steel at ~6-7 kHz this gives delta ~= R/2, i.e. the diameter/skin-depth ratio
        # d/delta ~= 4 that Rudnev cites as the efficiency knee for billet through-heating.
        self.skin_depth = self.robot.cylinder_radius / 2.0

