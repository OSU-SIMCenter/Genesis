import torch
import numpy as np
from pint import UnitRegistry
from typing import Literal, Optional, Tuple, List

import genesis as gs
from genesis.options.options import Options
from genesis.options import ProfilingOptions, SimOptions, MPMOptions, VisOptions, ViewerOptions

from agforge.material_properties import ACTIVE_MATERIAL

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
    """Parameters for the elasto-plastic material: 316L at forging temperature.

    Every value here is now sourced. See ``agforge/material_properties_mechanical.py``
    for the derivations and ``docs/316L_MECHANICAL_PROPERTIES.md`` for the sources,
    confidence levels and error budget.

    Operating point: **1000 C, 1 /s**, isothermal.
    """
    #: [BAM2023] dynamic resonance to 900 C, extrapolated 100 C further; Andrews
    #: independently gives 118.7 GPa at 1270 K. Was 50 GPa (200e9*0.25), a
    #: numerical choice that was 2.4x low.
    #:
    #: STABILITY: this is NOT a CFL hazard. substep_dt is DERIVED from
    #: dx/sqrt(E/rho) at a fixed 0.90 safety ratio in model_post_init below, so
    #: the timestep self-adjusts and the assertions still hold. What it costs is
    #: wall-clock: c goes 2500 -> 4070 m/s, so 1.63x more substeps buy the same
    #: physical time. (Not 1.56x as first quoted - that ignored the density drop;
    #: the ratio is sqrt((E/rho)_new / (E/rho)_old), and rho fell too.)
    E: float = 121.5e9
    #: Rises with temperature for austenitics [ISIJ1993]. MODERATE confidence -
    #: the published elevated-temperature values scatter non-monotonically, so
    #: this is derived rather than read off a table.
    #:
    #: ✅ SOURCED as of 2026-08-07. nu = E/(2G) - 1 applied to [BAM2023]'s own E
    #: and G, fitted across the whole table to average out the whole-GPa
    #: rounding, gives 0.383 at 1000 C; a direct point evaluation of the same
    #: two functions gives 0.381. This equals
    #: material_properties_mechanical.poisson_ratio(1273.15) -- the two are kept
    #: in step deliberately, and a test pins that.
    #:
    #: ⚠️ WAS 0.329, from an interpolation invented for this repo, held down by
    #: STABILITY rather than by data. The note that used to sit here read "THIS
    #: VALUE IS NOT SOURCED, AND IS HELD DOWN BY STABILITY, NOT BY DATA" -- that
    #: was accurate, and the block below records the measurements behind it.
    #: [BAM2023]'s own E and G - the same dataset that gives us E - imply
    #: nu = E/(2G) - 1, the standard identity, which fits to 0.382 at 1000 C
    #: (0.331 at room temperature, where this interpolation says 0.28).
    #:
    #: That sourced value CANNOT be adopted as things stand: it was measured to
    #: break the sim. nu drives the bulk modulus, and substep_dt below is derived
    #: from sqrt(E/rho) - which does not depend on nu at all - so raising nu
    #: raises the true wave speed while the timestep stays put. Empirically:
    #:   nu 0.329 -> ratio 1.094 vs the P-wave CFL limit -> stable (89 steps)
    #:   nu 0.34  -> ratio 1.117 -> stable (89 steps)
    #:   nu 0.35  -> ratio 1.140 -> press dies at 33 steps
    #:   nu 0.382 -> ratio 1.240 -> peak force 2.8e11 N
    #: Fixing this properly means deriving substep_dt from the P-wave speed; see
    #: cfl_use_pwave below.
    nu: float = 0.383

    #: Derive substep_dt from the P-wave speed sqrt((K + 4mu/3)/rho) instead of
    #: the thin-rod bar wave sqrt(E/rho). The P-wave is the fastest elastic wave
    #: in a bulk solid, so this is the criterion that actually governs stability;
    #: sqrt(E/rho) understates it by 38% at the shipped card (4070 vs 5620 m/s),
    #: which is why the nominal "0.90 safety factor" was really ~1.09x the true
    #: limit before this was turned on. The gap was 22% at the old nu = 0.329;
    #: raising nu raises the bulk modulus, so the P-wave got faster too.
    #:
    #: ✅ DEFAULT ON as of 2026-08-07. It is the criterion that actually governs
    #: stability, and it is what makes the sourced nu above usable. It only ever
    #: makes the timestep SMALLER, so it cannot destabilise a configuration that
    #: already runs.
    #:
    #: It is not free, and both costs are real:
    #:   - it shrinks substep_dt 1.208e-06 -> 8.751e-07 s => 38% more substeps
    #:     for the same sim time (it was 22% at the old nu = 0.329; the two
    #:     changes compound, so measure wall-clock rather than assuming 21%)
    #:   - it changes every trajectory on this branch, which other workstreams
    #:     are tuning against, so it needs coordinating rather than assuming
    #:
    #: Turning it on is what unblocks raising nu to the sourced ~0.382, and is
    #: the principled version of the "halve dt / AGF_CFL_SAFETY=0.45" workaround
    #: that makes long hit sequences complete elsewhere in the project.
    #:
    #: ✅ CONFIRMED by experiment (single press, GPU), which is the direct test
    #: of the diagnosis above:
    #:   nu 0.382, cfl_use_pwave=False -> ratio 1.240 -> 2.8e11 N, blows up
    #:   nu 0.382, cfl_use_pwave=True  -> ratio 0.900 -> STABLE, 207 kN, 83 steps
    #:   nu 0.329, cfl_use_pwave=True  -> ratio 0.900 -> STABLE, 201 kN, 83 steps
    #: Note the last two agree to ~3%, so once the timestep is right the choice
    #: of nu barely moves the answer - the instability was the timestep, not the
    #: material. substep_dt went 1.208e-06 -> 9.94e-07 at the nu = 0.329 those
    #: runs used; at the now-shipped nu = 0.383 it is 8.751e-07.
    cfl_use_pwave: bool = True
    #: [NIST2021] SRM 1155a, D(T) = 8052 - 0.564 T, at 1273.15 K. Was 8000
    #: (a room-temperature figure). Inter-lab spread puts the band at 7330-7570.
    rho: float = 7334.
    #: Only used when use_johnson_cook is False. Set to the 316L peak flow stress
    #: at the operating point so the two paths no longer disagree by ~11x.
    von_mises_yield_stress: float = 213.4e6

    # ---------------------------------------------------------------- #
    # Johnson-Cook parameters for 316L, calibrated AT THE OPERATING POINT
    # (1000 C, 1 /s) rather than globally.
    #
    # This block previously descended, essentially unchanged, from the canonical
    # Johnson & Cook (1983) set for AISI 4340 (A=792, B=510, n=0.26, C=0.014,
    # m=1.03, T_melt=1793 K) - only A and B had ever been hand-scaled. The billet
    # is 316L. All five are now derived from measured 316L data.
    #
    # WHY "at the operating point" and not a global fit: a joint 5-parameter fit
    # over 800-1000 C x 0.1-100 /s was tried and REJECTED - it reached 24.7%
    # worst-case error and drove m to its bound. That is not a tuning failure, it
    # is the known structural one: JC multiplies three uncoupled terms and cannot
    # represent the coupled, softening response of 316L in the DRX regime
    # (48% AARE vs 7.7% for Arrhenius - see the docs). Calibrating locally makes
    # the model exact where the sim actually runs.
    # ---------------------------------------------------------------- #
    use_johnson_cook: bool = True

    #: Use the strain-compensated hyperbolic-sine Arrhenius flow rule instead of
    #: Johnson-Cook. Takes precedence over use_johnson_cook when True.
    #:
    #: This is the model the literature endorses for 316L in the DRX regime
    #: (~7.7% AARE vs JC's ~48%), and unlike the JC path here it is genuinely
    #: rate- AND temperature-coupled: it strengthens on cooling, it reads the
    #: plastic strain rate, and it reproduces the DRX peak-then-soften curve.
    #:
    #: DEFAULT OFF, deliberately. The fit is a HOT-WORKING model valid over
    #: 800-1000 C, but the billet currently starts at room temperature
    #: (default_initial_temperature = 293.0 below). Arrhenius extrapolated to
    #: 293 K returns ~3.8 GPa - about 18x the forging-temperature flow stress -
    #: so ArrheniusPlasticity clamps temperature into the fit window as a
    #: numerical guard. At 293 K that clamp yields the 800 C value (~394 MPa at
    #: eps_p = 0.2), roughly 1.85x the Johnson-Cook card. Turning this on before
    #: the billet actually runs hot therefore changes the answer without making
    #: it more true. Switch it on together with a forging-temperature billet.
    use_arrhenius: bool = False

    #: Prescribed process strain rate [1/s] for the Arrhenius flow rule. None
    #: (or 0) derives the rate from the solver timestep, which at the sim's
    #: 25 m/s press speed means ~656 /s -- roughly 3e4 above the fit domain,
    #: and NOT caught by the rate clamp. Set this to the real process rate
    #: (0.41 /s for the first blow of the 2026-06-15 T4 dataset) to make the
    #: rate coupling physical. See ArrheniusPlasticity.process_strain_rate.
    arrhenius_process_strain_rate: Optional[float] = None

    #: Derived in material_properties_mechanical.isothermal_card(1.0). A is PINNED,
    #: not fitted: the residual is nearly flat in it, but the solver evaluates
    #: A + B eps^n from eps_p = 0, so A is the initial yield stress the sim sees.
    #: Was 40 MPa.
    jc_A: float = 100.3e6
    jc_B: float = 195.0e6   # was 100 MPa
    jc_n: float = 0.417     # was 0.26, which was 4340's exponent
    #: Derived from the validated Arrhenius model at 1000 C over the 0.1-10 /s
    #: band: C = (sigma_hi/sigma_lo - 1)/ln(rate ratio) gives 0.114 and 0.125
    #: either side of nominal. Was 0.014 - AISI 4340's ROOM-TEMPERATURE value,
    #: ~9x too low, because rate sensitivity rises steeply with temperature.
    jc_C: float = 0.120
    #: Reference rate = the nominal forging rate, so the rate term is exactly 1.0
    #: at nominal (where A and B are calibrated) and only corrects DEVIATION from
    #: it. Change this together with the rate used for jc_A/jc_B/jc_n.
    jc_eps0: float = 1.0

    # ---- thermal softening -------------------------------------------------
    # These used to sit hardcoded at the JohnsonCookPlasticity call site in
    # environment.py rather than here, which is why the 316L conversion missed
    # them and left 4340's values in place.
    #
    # T_ref is the FORGING temperature, not room temperature, because jc_A/jc_B
    # are already the 1000 C values. That makes T* = 0 and the softening factor
    # exactly 1.0 at the operating point, so the calibrated card is reproduced
    # exactly instead of being softened a second time.
    #
    # LIMITATION, and the reason to replace this model: the kernel clamps T* to
    # [0, 1], so BELOW 1000 C the factor pins at 1.0 and the billet gets NO
    # stronger as it cools. Real 316L stiffens ~43% for a 100 C drop. That is
    # inert today (thermal_enabled = False) but it is wrong the moment thermal
    # is switched on, and JC cannot be patched into correctness here - see the
    # Arrhenius recommendation in the docs.
    jc_T_ref: float = 1273.15
    #: Read from the material module so it cannot drift. 316L solidus = 1675 K;
    #: the old hardcoded 1793 K was AISI 4340's melting point.
    jc_T_melt: float = ACTIVE_MATERIAL.t_melt_k
    #: Governs softening between the forging temperature and the solidus. Linear
    #: decay to zero at melting. No 316L flow-stress data above 1000 C was found
    #: to fit this against, so it is a physically-reasonable choice rather than a
    #: measured one - the one value in this block that is not sourced.
    jc_m: float = 1.0

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

    # ---------------------------------------------------------------- as-built
    # These defaults mirror the real Agility Forge rather than a generic billet.
    # Provenance and confidence for every number: docs/AS_BUILT_AGILITY_FORGE.md
    #
    # The billet diameter used to be hardcoded to 1.0 inch inside model_post_init,
    # where it could not be configured at all. Both Colton Wright datasets
    # (2026-06-15 and 2026-07-17) are 1.5" 316L, so 1.0" matched no real run.
    billet_diameter_in: float = 1.5
    #: Simulated billet length [m]. None -> 8 * radius, the legacy self-similar rule,
    #: which at 1.5" gives 152.4 mm — enough to span the coil (76 mm) plus the 50 mm
    #: protrusion Colton describes. The real rod continues past the cut plane into the
    #: chuck; that is what enable_fixed_end_bc models, so this is the SIMULATED
    #: portion, not the physical rod length.
    billet_length_m: Optional[float] = None
    #: Grid cells across the billet diameter. Drives base_grid_density.
    grid_cells_across_billet: int = 7

    # ------------------------------------------------------------------ dies
    # The die footprint used to be derived from the BILLET radius
    # (0.5r x 1.2r, i.e. 19.05 x 45.72 mm), which is a box scaled off the
    # workpiece rather than a tool. The real dies are forge_common's
    # `Tool2_clip.obj` -- and the 2026-06-15 T4 dataset names them directly:
    # every action in the embedded plan carries `anvil_tool: 2, hammer_tool: 2`.
    #
    # Raw mesh extents are 15.82 x 54.71 x 12.70 mm, but the tool is NOT a
    # uniform block: it tapers from a flat mounting shank (15.82 mm axial)
    # through a rounded region to a narrower contact tip (9.49 mm axial).
    # forge_common's `press_tool.die_contact_axial_width_mm()` measures the
    # EFFECTIVE contact band at 25% of tool height above the tip -- where the
    # lateral extent reaches its full plateau -- giving 13.48 mm. That is the
    # number a flat-plane die approximation should carry; note it deliberately
    # differs from `real_scale.REAL_PRESS_AXIAL_BAND_MM = 15.8187`, which is
    # the whole-tool bounding box and overstates the contact band.
    #
    # Corroborated independently: inverting the measured force curve of the
    # first blow in the T4 dataset (66.5 kN, gap 38.41 -> 31.06 mm) for die
    # length gives 8-16 mm depending on the assumed constraint factor, which
    # brackets the 9.49 (tip) / 13.48 (effective) / 15.82 (shank) mesh values.
    #
    # Only the CONTACT FOOTPRINT is set here. Die thickness along the approach
    # axis stays on the legacy billet-radius rule because it sets the closed
    # gap via the slide range, and changing it would alter the kinematics
    # rather than the contact area. A box still cannot represent the taper --
    # replacing it with the actual mesh is a separate, larger change.
    #: Die contact width along the bar axis [m]. Sets contact area, hence force.
    die_axial_width_m: float = 0.01348
    #: Die span across the bar [m]. Exceeds the 38.1 mm billet, so it does not
    #: limit contact; carried for geometric fidelity.
    die_lateral_span_m: float = 0.0547116

    # Induction coil. Colton's email says "the coil is ~3\" in length"; a direct
    # count off the tape-measure photos IMG_9854/9855/9856 gives 4 turns (possibly
    # 3.5) spanning ~3.5", i.e. a pitch of ~22 mm. That pitch is corroborated by the
    # coil-shadow spacing in the thermal frames (~24 mm) and refutes an earlier
    # photo-derived reading of ~17 mm / ~5 turns.
    #
    # 3.5" is the extent of the CURRENT SHEET (turns x pitch), which is what the
    # finite-solenoid f_axial actually models; Colton's "~3" is approximate and may
    # describe the helical body without leads. The difference is not cosmetic:
    # 3.0" -> 3.5" raises the effective heated length (integral of f_axial along the
    # rod) by 15%, and fitted q_peak scales as ~1/L_eff, so it moves the fit by 15%.
    # Treat this as the dominant remaining GEOMETRIC uncertainty in the fit.
    coil_offset_x: float = -0.129891413331031800
    coil_length: float = 0.0889  # 3.5 inch — see above; was 0.0762 (3")
    # Effective solenoid radius = multiplier × billet radius. For close-coupled forging
    # coils the bore is typically ~110–125% of workpiece OD (ASM / induction heating practice).
    #
    # This was 2.0, which puts the BORE at 200% of workpiece OD — contradicting the
    # comment directly above it. Measured off IMG_9856 (tape held across the bore):
    # coil OD ~58–61 mm, tube OD ~1/4–3/8", giving a bore of ~45 mm and a current-path
    # (tube centreline) radius of ~26 mm against the 19.05 mm billet radius. Biot–Savart
    # wants the current-path radius, hence 26.0/19.05 = 1.365. Bore/OD works out at
    # ~1.18, inside the ASM range.
    #
    # Handheld photos with the tape at a different depth than the coil: treat as ±10–15%.
    #
    # A later direct reading off the same photos put the coil OD at ~2.5" (63.5 mm)
    # rather than the 58–61 mm measured here, which would imply a multiplier of
    # ~1.42–1.50. Deliberately NOT changed: sweeping the current-path radius from
    # 26.0 to 28.6 mm moves the effective heated length by <1%, so it is far below
    # the coil-LENGTH uncertainty above and not worth churning a committed number.
    # The two readings agree to within 5–9%, which is inside the ±10–15% band.
    coil_radius_multiplier: float = 1.365
    coil_radius: Optional[float] = None  # computed in model_post_init
    coup_softness: float = 5e-4

    # Declare fields for pydantic — Optional because they are computed in model_post_init
    cylinder_diameter: Optional[float] = None
    cylinder_radius: Optional[float] = None
    cylinder_height: Optional[float] = None
    cylinder_pos: object = None
    cylinder_euler: Optional[tuple] = None
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
        self.cylinder_diameter = (self.billet_diameter_in * ureg.inch).to(ureg.meter).magnitude
        self.cylinder_radius = self.cylinder_diameter / 2
        self.coil_radius = self.coil_radius_multiplier * self.cylinder_radius
        self.cylinder_height = (
            self.billet_length_m
            if self.billet_length_m is not None
            else 8 * self.cylinder_radius
        )
        self.cylinder_pos = np.array([0.0, 0.0, 6 * self.cylinder_radius])
        self.cylinder_euler = (0.0, 90.0, 0.0)

        # Self-similar grid: a fixed number of cells across the billet regardless of
        # its size. At 1.5" this gives dx = 5.46 mm.
        #
        # 🚨 THIS IS NOW KNOWN TO BE INADEQUATE FOR INDUCTION. The previous note here
        # reasoned that at an assumed 3 kHz the skin depth was ~10.2 mm against a
        # 19.05 mm radius (d/delta ~= 3.7, Rudnev's through-heating knee), so deposition
        # was near-volumetric and ~3.5 cells surface-to-axis was "coarse but not obviously
        # inadequate" — while explicitly warning it "would be inadequate at a much higher
        # frequency". Colton confirmed 2026-08-04 that the real drive is ~250 kHz, 83x
        # higher, giving delta ~= 1.12 mm: the skin is ~4.9x FINER than one cell here.
        # The induction deposition profile is therefore unresolved at this resolution.
        # Do not read induction results at the default grid as converged.
        self.base_grid_density = int(self.grid_cells_across_billet / self.cylinder_diameter)
        dx = 1.0 / self.base_grid_density
        mpm_solver_padding = 3 * dx
        mpm_x_padding_lower = self.cylinder_height * 0.85
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

    #: How thermal_time_scale (S_T) is chosen. See the derivation in
    #: model_post_init for the measured evidence behind both settings.
    #:   'cfl'        - 25% of the explicit-diffusion stability ceiling. A purely
    #:                  NUMERICAL choice, unrelated to how fast the press moves.
    #:                  What the induction-calibration workstream is tuned against.
    #:   'mechanical' - S_T = pressing_speed / real_die_speed, so the thermal and
    #:                  mechanical clocks agree. Required for any coupled
    #:                  thermo-mechanical run (die chill, transfer cooling).
    thermal_time_scale_mode: Literal["cfl", "mechanical"] = "cfl"

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
        #
        # 🚨 THE 0.90 BELOW IS NOT THE REAL SAFETY MARGIN. sqrt(E/rho) is the thin-rod
        # bar-wave speed; the fastest elastic wave in a bulk solid is the P-wave,
        #     c_p = sqrt((K + 4*mu/3)/rho),  K = E/(3(1-2nu)),  mu = E/(2(1+nu))
        # which at the card these measurements used (E=121.5 GPa, nu=0.329,
        # rho=7334) is 4945 m/s against sqrt(E/rho) = 4070 m/s, i.e. 21% faster.
        # So the bar-wave substep_dt is about 1.09x the STRICT P-wave CFL limit,
        # not 0.90x of it. At the now-shipped nu = 0.383 the P-wave is 5620 m/s
        # and the bar wave would be 1.24x the limit -- which is why nu could not
        # be raised until cfl_use_pwave shipped.
        #
        # Measured stability cliff (single press, GPU, one variable moved - nu):
        #     ratio 1.094 (nu 0.329) stable | 1.117 (nu 0.34) stable
        #     ratio 1.140 (nu 0.35)  BREAKS | 1.240 (nu 0.382) peak force 2.8e11 N
        # So the sim runs roughly 2% below the cliff. This is very likely the same
        # margin behind the intermittent long-sequence failures elsewhere in the
        # project, where halving dt (or AGF_CFL_SAFETY=0.45) is what makes the
        # 17-hit sequence complete - both put the ratio near 0.55.
        #
        # Note nu does NOT appear in the bar-wave branch, so raising it raises c_p
        # without shrinking substep_dt. That is exactly why the sourced nu was
        # unusable until the timestep came from c_p. ✅ Both now ship: the cost is
        # 38% more substeps, and it changes stability for every consumer of this
        # branch, so it is a coordination point rather than a silent default.
        temp_robot = RobotOptions(robot_time_to_seconds=1.0)
        dx = 1.0 / temp_robot.base_grid_density

        if getattr(self.mat, 'cfl_use_pwave', False):
            # Correct criterion: the fastest elastic wave in a bulk solid.
            mu_lame = self.mat.E / (2.0 * (1.0 + self.mat.nu))
            k_bulk = self.mat.E / (3.0 * (1.0 - 2.0 * self.mat.nu))
            c = math.sqrt((k_bulk + 4.0 * mu_lame / 3.0) / self.mat.rho)
        else:
            # Historical: the thin-rod bar-wave speed. Understates the true limit
            # by 38% at the shipped card (22% at the old nu = 0.329) - see above.
            c = math.sqrt(self.mat.E / self.mat.rho)
        dt_cfl = dx / c  # Theoretical max substep_dt

        cfl_safety = 0.90                         # 5% safety margin
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
        # Use worst-case (highest) thermal diffusivity over the operating range.
        #
        # NOTE — this flipped when the billet material changed from AISI 4340 to 316L.
        # For 4340, α was highest at ROOM temperature (1.245e-5) and fell to ~4.6e-6 when
        # hot, so room temp was the binding limit. 316L is the other way round: its
        # conductivity RISES with temperature, so α climbs from ~3.7e-6 (293 K) to
        # ~6.0e-6 (1450 K) and the binding limit is now the FORGING end, not the cold end.
        #
        # Net effect: 316L is roughly half as diffusive as 4340 at its worst, so the
        # thermal CFL is ~2x more permissive than it used to be.
        alpha_worst = ACTIVE_MATERIAL.alpha_worst()  # 316L: ~6.05e-6 m²/s at ~1450 K
        S_T_max = dx**2 / (6.0 * alpha_worst * substep_dt)
        
        # Fraction of the explicit-diffusion CFL limit used as the thermal time-scale. Higher =
        # faster wall-clock thermal evolution (heating AND cooling together, so the balance is
        # preserved); explicit FTCS develops checkerboard oscillations above ~60-70% of CFL, keep <=~0.4.
        thermal_cfl_fraction = 0.25
        thermal_time_scale = S_T_max * thermal_cfl_fraction

        # ---------------------------------------------------------------------
        # S_T IS A NUMERICAL QUANTITY, NOT A PHYSICAL ONE.
        #
        # The value above is 25% of a stability ceiling. Nothing ties it to how
        # fast the press actually moves. Measured on GPU at the shipped card:
        #     macro_dt = 9.666e-06 s and S_T = 171653, so ONE macro step advances
        #     1.659 s of THERMAL time, while the mechanics advance 9.666e-06 s of
        #     sim time = 17.1 ms of real time at the 1773x press acceleration.
        # The two clocks disagree by ~97x. Over a 74-step press that is 123 s of
        # cooling applied to a blow that really takes 0.505 s.
        #
        # This is LATENT, not active, in the default forging path. StrikeController
        # runs with thermal_enabled = False, which snapshots particle temperatures
        # before each physics step and restores them after ("Thermal Freezing"), so
        # a press is exactly isothermal. Verified: a 74-step press starting at
        # 1273.15 K ends at 1273.15 K with std 0.0, at BOTH time scales.
        #
        # It becomes ACTIVE the moment thermal physics runs during a strike, which
        # is required for die chill. Measured idle, thermal live, 100 steps from a
        # uniform 1273.15 K billet:
        #     S_T = 171653 -> mean 965.8 K, min 473.4 K   (166 s of thermal time)
        #     S_T =   1773 -> mean 1267.4 K, min 1232.7 K (1.7 s of thermal time)
        #
        # The first of those IS the 612-1269 K spread previously attributed to the
        # fixed-end BC. That attribution was WRONG. The dominant term is surface
        # radiation/convection amplified by S_T; the fixed-end cut face only
        # supplies the single coldest cell at the +x end.
        #
        # 'mechanical' is strictly SMALLER than the CFL value, so it can only be
        # more stable. The default stays 'cfl' because the induction-calibration
        # workstream on this branch is tuned against it.
        mode = getattr(self, 'thermal_time_scale_mode', 'cfl')
        if mode == 'mechanical':
            strike_cfg = getattr(self, 'strike', None) or StrikeOptions()
            mech_accel = strike_cfg.pressing_speed / strike_cfg.real_die_speed
            assert mech_accel <= thermal_time_scale, (
                f"thermal_time_scale_mode='mechanical' asks for S_T={mech_accel:.0f}, "
                f"above the explicit-diffusion CFL ceiling {thermal_time_scale:.0f}"
            )
            thermal_time_scale = mech_accel
        # ---------------------------------------------------------------------
        
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
        self.robot = RobotOptions(robot_time_to_seconds=0.1 * self.sim.substeps / self.sim.dt)
        self.mpm = MPMOptions(
            grid_density=self.robot.base_grid_density,
            particle_size=dx / 2.0,  # 8 particles per cell (2³ = 8 PPC)
            lower_bound=self.robot.mpm_lower_bound,
            upper_bound=self.robot.mpm_upper_bound,
            enable_CPIC=True,  # Improved rigid-MPM contact accuracy
            enable_thermal=True,
            default_initial_temperature=293.0,
            thermal_time_scale=thermal_time_scale,
            # Fixed-end (truncated-domain) BC: the held end conducts into the unsimulated
            # rod (Robin BC on the cut plane) instead of being exposed to air.
            enable_fixed_end_bc=True,
            thermal_contact_conductivity=3000.0,  # W/(m²K); reduced from 5000 — less aggressive die chill on light contact
            # Surface emissivity, previously left at the engine default and assumed 0.80
            # in calibration.json. Balat-Pichelin et al. measure as-received 316L
            # oxidising in air at only ~0.25 climbing to ~0.70 across 1100–1500 K, so
            # 0.80 overstated radiative loss roughly 2x over the range the 07-17 run
            # actually reached. 0.40 is representative near the hot end, where the T^4
            # term dominates and the calibration constraint is strongest.
            #
            # ⚠️ The real emissivity is NOT constant — it climbs as the oxide forms, and
            # a scalar cannot express that. Making the radiation term take epsilon(T)
            # from material_properties.emissivity_316l is the proper fix; it needs a
            # kernel change. Until then this is a deliberate mid-range compromise.
            emissivity=0.40,
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
    pressing_speed: float = 25.0 # m/s

    #: Real die closing speed [m/s], measured from blow #1 of the 2026-06-15 T4
    #: dataset: 7.12 mm of bite in 0.505 s from contact to peak force.
    #:
    #: The sim presses at pressing_speed = 25 m/s, i.e. 1773x faster. That is a
    #: defensible quasi-static proxy for the MECHANICS (Johnson-Cook's rate term
    #: is inert, and the inertial stress rho*v^2 ~ 4.6 MPa is only ~2% of flow
    #: stress) - but it is the reference any THERMAL time scaling has to agree
    #: with, which is what thermal_time_scale_mode uses it for.
    real_die_speed: float = 0.0141
    
    # Force Balance Control
    # 5e-5 was robust. 1.5e-4 is peak performance but near instability (2e-4).
    # User selected 1.5e-4 for maximum benchmark results.
    force_balance_gain: float = 1.5e-4 
    
    # Safety Limits
    max_force_imbalance: float = 20000.0 # 20 kN% compression
    max_force: float = 200000.0 # 20 tons (200kN)
    pressing_timeout: float = 30.0 # seconds (Increased to avoid timeout)
    approaching_timeout: float = 30.0 # seconds (Increased to avoid timeout)
    release_timeout: float = 10.0 # seconds
    hold_steps: int = 15 # Number of steps to hold position after reaching target before releasing
    post_release_steps: int = 10 # steps

class SafetyOptions(Options):
    """Parameters for simulation stability checks."""
    enabled: bool = True
    strict_tracing_enabled: bool = True # Gating flag for heavy threshold bounds checking
    
    # Thresholds for early detection (Mid-blowup catching)
    max_particle_velocity: float = 100.0 # m/s (Supersonic catch)
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
    skin_depth: Optional[float] = None  # Derived from material + coil_frequency_hz below

    # Induction coil drive frequency [Hz].
    # ANSWERED by Colton Wright 2026-08-04: "coil frequency is adjusted by Ambrell's
    # controller dynamically and it tends to sit around 250kHz for a 1.5" piece of 316L."
    # This replaces an inferred 3 kHz, which was wrong by 83x.
    #
    # 🚨 The consequence is not a detail. delta = sqrt(rho_e/(pi*f*mu0)) scales as
    # 1/sqrt(f), so delta collapses 10.22 mm -> 1.12 mm (at 1273 K) and R/delta goes
    # 1.9 -> 17. This is the THIN-SKIN regime, not the through-heating regime the old
    # comment assumed: only ~5% of the cross-section receives power directly, and the
    # rest heats by conduction (skin->bulk equilibration is ~0.2 s).
    #
    # 🚨 AND THE GRID CANNOT RESOLVE IT. dx = 5.46 mm at the default 7 cells across the
    # billet, so the skin is ~4.9x FINER THAN ONE CELL. The comment on base_grid_density
    # above warned this scheme "would be inadequate at a much higher frequency" — that
    # condition is now met, 83x over. Deposition is effectively a surface flux and
    # arguably wants a surface boundary condition rather than a volumetric source.
    # Treat any induction result at the default resolution as unresolved until this is
    # addressed. See docs/DATASET_20260717_STRUCTURE.md.
    coil_frequency_hz: float = 250000.0
    # Temperature at which the material's electrical resistivity is evaluated for the skin
    # depth. Resistivity rises with temperature, so a hot reference gives a deeper delta.
    induction_reference_temp_k: float = 1273.15
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
        
        target_cfl_ratio = 0.35
        dx = 1.0 / self.robot.base_grid_density
        dt = self.sim.dt
        
        # Calculate optimal safe speed
        parametric_speed = target_cfl_ratio * (dx / dt)
        
        # Override default
        self.strike.approach_speed = parametric_speed
        
        # EM skin depth delta = sqrt(rho_e / (pi * f * mu0 * mu_r)) — a MATERIAL + FREQUENCY
        # property. It does NOT depend on billet radius.
        #
        # This used to read `self.skin_depth = self.robot.cylinder_radius / 2.0`, justified as
        # the d/delta ~= 4 through-heating efficiency knee Rudnev cites. That is a COIL DESIGN
        # rule (choose f to suit the billet), not a material property, and tying delta to R had
        # a silent failure mode: changing billet size rewrote the implied drive frequency
        # without saying so. At the 1-inch sim billet, delta = R/2 implies ~6.6 kHz for hot
        # 316L — which is what the old comment asserted. At Colton's 38.1 mm rod the identical
        # rule implies ~2.9 kHz. Same line of code, different physics, no warning.
        #
        # Deriving delta from the material and an explicit frequency makes the assumption
        # visible and overridable. See agforge/material_properties.py.
        #
        # ⚠️ coil_frequency_hz is REPORTED, not measured — Colton gave ~250 kHz (see its
        # declaration above), but it is a dynamically tracked resonance, not a setpoint, so
        # the true value drifts with load during a run. q_peak scales roughly as 1/delta, so
        # a wrong frequency trades off directly against a wrong q_peak: total absorbed power
        # stays well constrained by the measured heating curve, but the surface-intensity /
        # penetration-depth split does not.
        self.skin_depth = ACTIVE_MATERIAL.skin_depth_m(
            self.coil_frequency_hz, temp_k=self.induction_reference_temp_k
        )

