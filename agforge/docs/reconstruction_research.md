# Research Report: Surface Reconstruction Artifacts in Hybrid MPM

## 1. Executive Summary
The current **Hybrid GPU-Splatting / CPU-Meshing** reconstruction engine provides high performance ($O(1)$ scaling w.r.t particles) but exhibits two distinct surface artifacts:
1.  **"Wavy" Patterns:** Moiré-like aliasing caused by the interaction between the moving particle distribution and the fixed density grid.
2.  **"Bumpy" Surface:** Isotropic kernel limitations leading to "pinching" between particles when material is deformed (anisotropic stretching).

This report details the technical root causes, analyzes the current implementation, and proposes a roadmap for mitigation ranging from immediate fixes to advanced research directions.

## 2. Current Implementation Analysis

### **Pipeline Architecture**
The reconstruction is handled in `agforge/reconstruction.py`. It uses a Grid-Based Implicit Surface approach:

```python
# Core Algorithm (Simplified)
# 1. GPU Density Computation (Taichi)
@ti.kernel
def compute_density(particles_pos):
    for p in particles:
        # Splat particle mass onto grid cells using cubic spline kernel
        # Kernel: (1 - r^2)^3 where r = distance / influence_radius
        grid_density[cell] += kernel(dist)

# 2. CPU Meshing (Skimage)
def create_mesh(density_field):
    # Marching Cubes extracts isosurface at density = 0.5
    verts, faces, _, _ = marching_cubes(density_field, level=0.5)
    return verts, faces
```

### **Key Parameters**
*   **Grid Resolution:** Configurable (`64^3` default for Teleop, `128^3` for High-Quality).
*   **Influence Radius ($R$):** $2.5 \times r_{particle}$.
*   **Kernel Function:** Isotropic Cubic Spline $W(r) = (1 - r^2)^3$.
*   **Iso-value:** Fixed at $0.5$.

### **Performance Profile**
*   **Time:** ~6ms (64^3) / ~30ms (128^3).
*   **Scaling:** Constant time w.r.t particle count ($>$ 140k).
*   **Bottleneck:** Memory transfer (GPU $\to$ CPU) and `marching_cubes` execution on CPU.

## 3. Artifact Analysis

### **A. The "Wavy" Pattern (Aliasing)**
*   **Observation:** Surfaces appear to ripple or shimmer during slow deformation.
*   **Cause:** **Grid Aliasing**. As a particle moves across a grid cell boundary, its contribution to the discrete density nodes shifts. Because the grid resolution is finite (and relatively coarse at $64^3$), this shift causes the reconstructed isosurface to jump or oscillate between grid points.
*   **Analogy:** Similar to Moiré patterns when viewing a screen through a mesh.

### **B. The "Bumpy" Surface (Kernel Isotropy)**
*   **Observation:** Deformed surfaces look like "bunches of grapes" or have regular bumps.
*   **Cause:** **Lack of Anisotropy**. 
    *   The kernel is perfectly spherical.
    *   When the material is compressed or stretched, the physical distribution of particles becomes anisotropic (closer in compression axis, further in stretch axis).
    *   The spherical kernel cannot stretch to bridge the gap in the elongated direction, leading to density dips between particles and resulting in a bumpy isosurface.

## 4. Proposed Solutions & Roadmap

### **Phase 1: Immediate Mitigations (Low Cost)**
*Implementation Time: < 1 day*

1.  **Laplacian Mesh Smoothing (Recommended Fix for Bumps)**
    *   **Technique:** Apply iterative smoothing to the generated mesh vertices to relax high-frequency noise.
    *   **Implementation:** Use `trimesh.smooth.laplacian(mesh, iterations=3)`.
    *   **Constraint:** May slightly shrink the mesh volume.

2.  **Influence Radius Tuning**
    *   **Technique:** Increase $R$ from $2.5x$ to $3.0x$.
    *   **Effect:** Blurs the density field more, merging individual particle "blobs" better.
    *   **Constraint:** Loss of sharp details/corners.

### **Phase 2: Temporal Stability (Medium Cost)**
*Implementation Time: 2-3 days*

1.  **Temporal Density Blending**
    *   **Technique:** $D_{t}(x) = \alpha D_{current}(x) + (1-\alpha) D_{t-1}(x)$.
    *   **Effect:** Significantly reduces "wavy" aliasing artifacts by damping frame-to-frame density jitter.
    *   **Constraint:** Introduces "ghosting" or lag during fast motion. Requires double buffering the density grid on GPU.

### **Phase 3: Advanced High-Fidelity (Research/High Cost)**
*Implementation Time: 1-2 weeks*

1.  **Anisotropic Kernels (The "Gold Standard")**
    *   **Technique:** Compute the covariance matrix ($\Sigma$) of each particle's neighborhood (PCA). Deform the kernel shape ($W$) to match the local particle distribution: $W(r, \Sigma)$.
    *   **Effect:** Produces perfectly smooth, flat surfaces even under extreme deformation. Preserves sharp features.
    *   **Technical Challenge:** Requires computing SVD/Eigendecomposition per particle (expensive) or approximating it (complex). Hard to optimize for real-time.

2.  **Dual Contouring / Manifold Dual Contouring**
    *   **Technique:** Replaces Marching Cubes. Uses Hermite data (normals) to position vertices more accurately within cells.
    *   **Effect:** Reduces grid aliasing (sharp edges are preserved better).
    *   **Technical Challenge:** Significantly more complex meshing algorithm.

## 5. Implementation Guide (Phase 1)

**Recommended Action:** Add post-process smoothing to `reconstruction.py`.

```python
# Add to create_reconstructed_mesh:
try:
    # ... existing marching cubes code ...
    
    self.reconstructed_mesh = trimesh.Trimesh(...)
    
    # [NEW] Apply Laplacian Smoothing
    trimesh.smoothing.filter_laplacian(
        self.reconstructed_mesh, 
        lamb=0.5, 
        iterations=3
    )

except Exception as e:
    ...
```

## 6. Resources for Further Research
*   **Paper:** ["Reconstructing Surfaces of Particle-Based Fluids using Anisotropic Kernels"](https://dl.acm.org/doi/10.1145/2461912.2461944) (Yu & Turk, 2013).
*   **Paper:** ["Screen Space Fluid Rendering with Curvature Flow"](https://dl.acm.org/doi/10.1145/1507149.1507153) (van der Laan et al., 2009) - (If moving to pure rendering approach).


## Appendix: Source Code

### agforge/reconstruction.py
```python
import time
import torch
import numpy as np
import trimesh
import gstaichi as ti
import genesis as gs
import genesis.utils.particle as pu
from skimage.measure import marching_cubes
from enum import Enum

# Compatibility Enum
class SamplingMethod(Enum):
    RANDOM = "random"
    VOXEL_STRATIFIED = "voxel_stratified"
    FPS = "fps"
    HALTON_LLOYD = "halton_lloyd"

@ti.data_oriented
class SurfaceReconstructor:
    def __init__(self, env, grid_res=128, backend='hybrid'):
        self.env = env
        self.reconstructed_mesh = trimesh.Trimesh()
        self.recon_enabled = True
        self.recon_frame_interval = 3  # Reconstruct every 3 frames
        self.frame_counter = 0
        
        # Compatibility / Configuration
        self.sampling_method = SamplingMethod.VOXEL_STRATIFIED
        self.recon_particle_fraction = 1.0
        self.skinning_enabled = False # Dummy flag
        self.bind_indices = None
        self.bind_weights = None
        
        # Backend Configuration
        self.backend = backend # 'hybrid' or 'splashsurf'
        
        # Caching for Visualization
        self._cached_particles = None
        
        # Grid Configuration
        self.grid_res = grid_res
        self.density = ti.field(dtype=float, shape=(self.grid_res, self.grid_res, self.grid_res))
        
        # Constants
        self.particle_radius = 0.005 
        try:
            self.particle_radius = self.env.scene.sim.mpm_solver.particle_size / 2.0
            gs.logger.info(f"Reconstruction: Inferred particle radius: {self.particle_radius}")
        except Exception:
            pass
        self.influence_radius = self.particle_radius * 2.5

    def get_state(self):
        return {
            'mesh': self.reconstructed_mesh.copy(),
            'recon_enabled': self.recon_enabled,
            'frame_counter': self.frame_counter,
        }

    def set_state(self, state):
        if state is None: return
        self.reconstructed_mesh = state['mesh'].copy()
        self.recon_enabled = state.get('recon_enabled', True)
        self.frame_counter = state.get('frame_counter', 0)

    def reset(self):
        self.reconstructed_mesh = trimesh.Trimesh()
        self.frame_counter = 0
        self._cached_particles = None
        self.skinning_enabled = False

    # --- Compatibility Interface ---
    def init_skinning(self):
        """No-op compatibility method."""
        pass
        
    def update_skinning(self):
        """No-op compatibility method."""
        pass
        
    def subdivide_long_edges(self) -> bool:
        """No-op compatibility method."""
        return False
        
    def get_active_particle_cache(self):
        """Returns cached particles for visualization."""
        if self._cached_particles is None:
             # Try to fetch if cache is empty
             return self._get_active_particles(use_cache=False)
        return self._cached_particles
        
    def _get_active_particles(self, use_cache: bool = True, apply_subsampling: bool = True):
        """Fetches active particles from MPM entity."""
        if use_cache and self._cached_particles is not None:
            return self._cached_particles
            
        try:
            mpm_entity = self.env.mpm_entity
            parts = mpm_entity.get_particles_pos(envs_idx=0).squeeze(0)
            active = mpm_entity.get_particles_active(envs_idx=0).squeeze(0)
            parts = parts[active]
            
            p_numpy = parts.cpu().numpy()
            self._cached_particles = p_numpy
            return p_numpy
        except Exception:
            return None

    @ti.kernel
    def _compute_density_kernel(
        self, 
        particles_pos: ti.types.ndarray(),
        active_mask: ti.types.ndarray(),
        n_particles: int,
        lower_bound: ti.types.vector(3, float),
        dx: float,
        influence_radius: float
    ):
        for I in ti.grouped(self.density):
            self.density[I] = 0.0
            
        for i in range(n_particles):
            if active_mask[i]:
                pos = ti.Vector([particles_pos[i, 0], particles_pos[i, 1], particles_pos[i, 2]])
                grid_pos = (pos - lower_bound) / dx
                rad_cells = influence_radius / dx
                
                base_idx = ti.cast(ti.floor(grid_pos - rad_cells), ti.int32)
                end_idx = ti.cast(ti.ceil(grid_pos + rad_cells), ti.int32)
                
                for ix in range(base_idx[0], end_idx[0] + 1):
                    for iy in range(base_idx[1], end_idx[1] + 1):
                        for iz in range(base_idx[2], end_idx[2] + 1):
                            if (0 <= ix < self.grid_res and 
                                0 <= iy < self.grid_res and 
                                0 <= iz < self.grid_res):
                                cell_center = lower_bound + ti.Vector([ix, iy, iz]) * dx
                                dist_sq = (pos - cell_center).norm_sqr()
                                if dist_sq < influence_radius**2:
                                    r2 = dist_sq / (influence_radius**2)
                                    val = (1.0 - r2)**3
                                    self.density[ix, iy, iz] += val

    def update(self, should_reconstruct: bool):
        if not self.recon_enabled:
            return
        self.frame_counter += 1
        if not should_reconstruct and (self.frame_counter % self.recon_frame_interval != 0):
            return
        self.create_reconstructed_mesh()

    def create_reconstructed_mesh(self):
        # Legacy SplashSurf Path
        if self.backend == 'splashsurf':
            self._create_splashsurf_mesh()
            return

        # Hybrid Path
        try:
            mpm_entity = self.env.mpm_entity
            particles_pos = mpm_entity.get_particles_pos(envs_idx=0).squeeze(0)
            particles_active = mpm_entity.get_particles_active(envs_idx=0).squeeze(0)
            
            n_particles = particles_pos.shape[0]
            if n_particles == 0:
                return

            # Update cache for visualization
            self._cached_particles = particles_pos[particles_active].cpu().numpy()

            min_bound = particles_pos.min(dim=0).values.cpu().numpy()
            max_bound = particles_pos.max(dim=0).values.cpu().numpy()
            
            padding = self.influence_radius * 2.0
            min_bound -= padding
            max_bound += padding
            
            extent = max_bound - min_bound
            max_extent = np.max(extent)
            if max_extent < 1e-4: max_extent = 1.0
                
            dx = max_extent / (self.grid_res - 1)
            lower_bound_ti = ti.Vector([min_bound[0], min_bound[1], min_bound[2]])
            
            self._compute_density_kernel(
                particles_pos, 
                particles_active, 
                n_particles, 
                lower_bound_ti, 
                float(dx), 
                float(self.influence_radius)
            )
            
            density_cpu = self.density.to_numpy()
            thresh = 0.5 
            
            verts, faces, normals, values = marching_cubes(
                density_cpu, 
                level=thresh,
                spacing=(dx, dx, dx),
                allow_degenerate=False
            )
            verts += min_bound
            
            self.reconstructed_mesh = trimesh.Trimesh(
                vertices=verts, 
                faces=faces, 
                vertex_normals=normals,
                process=False
            )
            
        except Exception as e:
            gs.logger.debug(f"Reconstruction skip: {e}")

    def _create_splashsurf_mesh(self):
        """Legacy SplashSurf reconstruction."""
        try:
            particles = self._get_active_particles(use_cache=False)
            if particles is None or len(particles) == 0:
                return
                
            # Use ~1.0x particle radius for reconstruction
            # SplashSurf handles neighborhood search internally
            self.reconstructed_mesh = pu.particles_to_mesh(
                positions=particles,
                radius=self.particle_radius * 1.5,
                backend='splashsurf'
            )
        except Exception as e:
            gs.logger.warning(f"SplashSurf failed: {e}")
```

### agforge/strike_controller.py
```python
import asyncio
import time
import enum
import torch
import numpy as np
import genesis as gs
import contextlib

from agforge.reconstruction import SurfaceReconstructor

class StrikeState(enum.Enum):
    IDLE = 0
    APPROACHING = 1
    HOLDING = 2
    PRESSING = 3
    RELEASE = 4

# Configuration constants
MAX_CHECKPOINTS = 50  # Maximum number of checkpoints to retain
VERBOSE_LOGGING = True  # Enable per-frame logging during strike (set to False to disable)
LOG_EVERY_N_FRAMES = 3  # Log every Nth frame (1=every frame, 10=every 10th frame)

class StrikeController:
    """
    Encapsulates the core control logic, state machine, and physics interaction 
    for the Agility Forge simulation.
    """
    def __init__(self, env):
        self.env = env
        self.robot = env.robot
        
        # Robot State
        self.qpos = self.robot.entity.get_dofs_position()
        self.dof_limits = self.robot.entity.get_dofs_limit()
        
        # We need a lock because async socket tasks might access this concurrently 
        # with the simulation loop.
        self.lock = asyncio.Lock()
        
        # Strike Logic State
        self.strike_state = StrikeState.IDLE
        self.contact_L = False
        self.contact_R = False
        self.target_strain = 0.5
        self.stage_start_time = 0.0
        self.contact_width = 0.0
        self.stabilization_steps = 0
        self.strike_step_count = 0  # Count simulation steps during strike
        
        # Gripper limits (derived from config via XML logic in original, 
        # but we can query the robot or just use the config if available)
        # Assuming we can get them from the robot entity or similar.
        # For now, let's copy the logic if we can import RobotXMLGenerator or just store them.
        # We'll calculate them once.
        self._init_gripper_limits()

        # Surface Reconstruction
        recon_cfg = env.cfg.reconstruction
        self.reconstructor = SurfaceReconstructor(
            env, 
            grid_res=recon_cfg.grid_res, 
            backend=recon_cfg.backend
        )
        self.reconstructor.recon_enabled = recon_cfg.enabled
        # Note: Reconstruction init mostly happens on demand or at start
        
        # Checkpointing
        self.checkpoints = []
        
        # Transformation constants for Unity (visualization)
        self._init_transforms()
        
        # Optimization: Pre-allocated tensors
        self._vel_cmd = torch.zeros(4, device=self.env.device)
        self._last_vel_cmd = torch.zeros(4, device=self.env.device)
        self._dofs_idx_local = torch.tensor([0, 1, 2, 3], device=self.env.device)
        self._force_next_apply = True

    def _init_gripper_limits(self):
        # We can re-use the XML generator logic or just hardcode if standard.
        # Since agforge_builder is where RobotXMLGenerator lives, we might need to import it
        # or just assume standard values if not critical. 
        # Better: Import it to be safe.
        from agforge.agforge_builder import RobotXMLGenerator
        xml_generator = RobotXMLGenerator(robot_cfg=self.env.cfg.robot)
        self.gripper_closed_pos = xml_generator.gripper_slide_range[1]
        self.gripper_open_pos = xml_generator.gripper_slide_range[0]

    def _init_transforms(self):
        # Constants from teleop_socket.py
        TRANSFORM_SCALE = 31.275
        TRANSFORM_HEIGHT_FACTOR = 0.375
        
        self.unity_translation = -(self.env.cfg.robot.cylinder_pos + np.array([-TRANSFORM_HEIGHT_FACTOR * self.env.cfg.robot.cylinder_height, 0, 0]))
        self.unity_scale = np.array((TRANSFORM_SCALE,) * 3)
        self.unity_translation_tensor = torch.tensor(self.unity_translation, dtype=torch.float32, device=self.env.device).view(1, 3)
        self.unity_scale_tensor = torch.tensor((TRANSFORM_SCALE,) * 3, dtype=torch.float32, device=self.env.device).view(1, 3)

    async def set_qpos(self, new_qpos):
        async with self.lock:
            # Only clamp slider/hinge, grippers managed by logic if striking
            new_qpos[:, :2] = torch.clamp(new_qpos[:, :2], self.dof_limits[0][:2], self.dof_limits[1][:2])
            self.qpos = new_qpos

    async def get_qpos(self):
        async with self.lock:
            return self.qpos.clone()

    async def trigger_strike(self, force_param):
        async with self.lock:
            if self.strike_state != StrikeState.IDLE:
                gs.logger.warning(f"Strike requested but already in {self.strike_state.name}")
                return
            
            # Log initial state for debugging
            pos_L = self.robot.left_gripper.get_pos()
            pos_R = self.robot.right_gripper.get_pos()
            initial_width = torch.norm(pos_L - pos_R).item()
            
            gs.logger.info(f"Strike -> APPROACHING (target_strain={force_param:.4f})")
            gs.logger.info(f"  Initial: width={initial_width:.4f}, pos_L={pos_L[0].tolist()}, pos_R={pos_R[0].tolist()}")
            gs.logger.info(f"  Config: approach_spd={self.env.cfg.strike.approach_speed}, press_spd={self.env.cfg.strike.pressing_speed}, max_force={self.env.cfg.strike.max_force}")
            
            self.contact_L = False
            self.contact_R = False
            self.strike_state = StrikeState.APPROACHING
            self.stage_start_time = time.time()
            self.strike_start_time = time.time()  # Track total strike duration
            self.stabilization_steps = 0
            self.target_strain = force_param
            self.strike_step_count = 0
            self.robot.set_control_mode("VELOCITY_CONTROL")
            gs.logger.info(f"  Control mode -> VELOCITY_CONTROL")

    async def update_logic(self):
        """
        Executes one step of the strike state machine logic.
        This mirrors `update_strike_logic` from teleop_socket.py.
        """

        with self._profile("logic_step"):
            # --- APPROACHING STAGE ---
            if self.strike_state == StrikeState.APPROACHING:
                with self._profile("logic_update_state"):
                    approach_speed = self.env.cfg.strike.approach_speed
                    contact_threshold = self.env.cfg.strike.contact_force_threshold
                    approaching_timeout = self.env.cfg.strike.approaching_timeout
                    
                    if time.time() - self.stage_start_time > approaching_timeout:
                        gs.logger.warning(f"Strike APPROACHING timed out")
                        self.strike_state = StrikeState.RELEASE
                        self.stage_start_time = time.time()
                        self._stop_motors()
                        return
                
                with self._profile("logic_get_resistance"):
                    force_L, force_R = self.robot.get_resistance_forces()
                
                with self._profile("logic_update_state"):
                    if not self.contact_L and force_L > contact_threshold:
                        self.contact_L = True
                        gs.logger.info(f"  Left gripper CONTACT (force={force_L:.4f})")
                        
                    if not self.contact_R and force_R > contact_threshold:
                        self.contact_R = True
                        gs.logger.info(f"  Right gripper CONTACT (force={force_R:.4f})")
                    
                    if VERBOSE_LOGGING and self.strike_step_count % LOG_EVERY_N_FRAMES == 0:
                        gs.logger.info(f"APPROACHING[{self.strike_step_count}]: F=[{force_L:.3f},{force_R:.3f}], contact=[{self.contact_L}, {self.contact_R}]")

                    self._vel_cmd.zero_()
                    self._vel_cmd[2] = 0.0 if self.contact_L else approach_speed
                    self._vel_cmd[3] = 0.0 if self.contact_R else approach_speed
                
                with self._profile("logic_apply_vel"):
                    self._apply_vel_dedup()
                
                with self._profile("logic_update_state"):
                    if self.contact_L and self.contact_R:
                        self.strike_state = StrikeState.PRESSING
                        self.stage_start_time = time.time()
                        
                        pos_L = self.robot.left_gripper.get_pos()
                        pos_R = self.robot.right_gripper.get_pos()
                        self.contact_width = torch.norm(pos_L - pos_R).item()
                        
                        gs.logger.info(f"Strike -> PRESSING (width={self.contact_width:.4f}, steps={self.strike_step_count})")
                    
            # --- PRESSING STAGE ---
            elif self.strike_state == StrikeState.PRESSING:
                with self._profile("logic_update_state"):
                    pressing_speed = self.env.cfg.strike.pressing_speed
                    target_strain = self.target_strain
                    max_force = self.env.cfg.strike.max_force
                    pressing_timeout = self.env.cfg.strike.pressing_timeout
                    force_balance_gain = self.env.cfg.strike.force_balance_gain

                with self._profile("logic_get_resistance"):
                    force_L, force_R = self.robot.get_resistance_forces()
                
                with self._profile("logic_update_state"):
                    pos_L = self.robot.left_gripper.get_pos()
                    pos_R = self.robot.right_gripper.get_pos()
                    current_width = torch.norm(pos_L - pos_R).item()
                    
                    if self.contact_width > 1e-6:
                        current_strain = (self.contact_width - current_width) / self.contact_width
                    else:
                        current_strain = 0.0
                    
                    # Warn about unusual conditions
                    if current_strain < -0.01:
                        gs.logger.warning(f"Negative strain detected ({current_strain:.4f}) - grippers moving apart?")
                    if force_L < 0 or force_R < 0:
                        gs.logger.warning(f"Negative force detected: F_L={force_L:.4f}, F_R={force_R:.4f}")
                    
                    elapsed_time = time.time() - self.stage_start_time

                    stop_reason = None
                    if current_strain >= target_strain:
                        stop_reason = "Target Strain"
                    elif force_L > max_force or force_R > max_force:
                        stop_reason = "Max Force"
                    elif elapsed_time > pressing_timeout:
                        stop_reason = "Timeout"
                    
                    if stop_reason:
                        gs.logger.info(f"Strike -> RELEASE ({stop_reason}, strain={current_strain:.4f}, steps={self.strike_step_count}, time={elapsed_time:.2f}s)")
                        self.strike_state = StrikeState.RELEASE
                        self.stage_start_time = time.time()
                        self._stop_motors()
                        return

                    imbalance = force_L - force_R
                    correction = imbalance * force_balance_gain
                    
                    v_L = max(0.0, pressing_speed - correction)
                    v_R = max(0.0, pressing_speed + correction)
                    
                    if VERBOSE_LOGGING and self.strike_step_count % LOG_EVERY_N_FRAMES == 0:
                        gs.logger.info(f"PRESSING[{self.strike_step_count}]: F=[{force_L:.3f},{force_R:.3f}] dF={imbalance:.4f}, v=[{v_L:.4f},{v_R:.4f}] corr={correction:.4f}, strain={current_strain:.3f}/{target_strain:.3f}")
                    
                    self._vel_cmd.zero_()
                    self._vel_cmd[2] = v_L
                    self._vel_cmd[3] = v_R
                
                with self._profile("logic_apply_vel"):
                    self._apply_vel_dedup()

            # --- RELEASE STAGE ---
            elif self.strike_state == StrikeState.RELEASE:
                with self._profile("logic_update_state"):
                    release_speed = self.env.cfg.strike.pressing_speed
                    contact_threshold = self.env.cfg.strike.contact_force_threshold * 0.2
                    force_balance_gain = self.env.cfg.strike.force_balance_gain
                    release_timeout = self.env.cfg.strike.release_timeout
                    
                    if time.time() - self.stage_start_time > release_timeout:
                        gs.logger.warning(f"Strike RELEASE timed out - Forcing reset")
                        self._force_idle_reset()
                        await self.save_checkpoint()
                        return
                
                with self._profile("logic_get_resistance"):
                    force_L, force_R = self.robot.get_resistance_forces()

                with self._profile("logic_update_state"):
                    imbalance = force_L - force_R
                    correction = imbalance * force_balance_gain
                    
                    v_open = -release_speed
                    v_L = v_open - correction
                    v_R = v_open + correction
                    
                    if VERBOSE_LOGGING and self.strike_step_count % LOG_EVERY_N_FRAMES == 0:
                        gs.logger.info(f"RELEASE[{self.strike_step_count}]: F=[{force_L:.3f},{force_R:.3f}] dF={imbalance:.4f}, thresh={contact_threshold:.4f}, v=[{v_L:.4f},{v_R:.4f}] corr={correction:.4f}")
                    
                    self._vel_cmd.zero_()
                    self._vel_cmd[2] = v_L
                    self._vel_cmd[3] = v_R
                
                with self._profile("logic_apply_vel"):
                    self._apply_vel_dedup()
                
                with self._profile("logic_update_state"):
                    if abs(force_L) < contact_threshold and abs(force_R) < contact_threshold:
                        # Calculate final stats
                        pos_L = self.robot.left_gripper.get_pos()
                        pos_R = self.robot.right_gripper.get_pos()
                        final_width = torch.norm(pos_L - pos_R).item()
                        total_duration = time.time() - getattr(self, 'strike_start_time', self.stage_start_time)
                        
                        self._force_idle_reset()
                        # Stabilization after release
                        self.stabilization_steps = self.env.cfg.strike.post_release_steps
                        
                        gs.logger.info(f"Strike -> IDLE (Stabilizing for {self.stabilization_steps} steps)")
                        gs.logger.info(f"  Summary: total_time={total_duration:.2f}s, steps={self.strike_step_count}, final_width={final_width:.4f}")
                        
                        # RESTORED: Full reconstruction after strike to fix skinning drift
                        # This gives the user a "perfect" result after the action completes.
                        # DISABLED (User Request): Relying purely on edge splitting
                        # recon_start = time.time()
                        # self.reconstructor.create_reconstructed_mesh()
                        # self.reconstructor.init_skinning() # REQUIRED to sync weights with new mesh
                        # recon_time = (time.time() - recon_start) * 1000
                        # gs.logger.info(f"  Post-strike reconstruction: {recon_time:.1f}ms")

                        # Save checkpoint
                        await self.save_checkpoint()
                        
                        # Flag to ensure updated mesh is sent to client
                        self.pending_mesh_send = True
                        return

            # --- HOLDING STAGE ---
            elif self.strike_state == StrikeState.HOLDING:
                pass

    def _stop_motors(self):
        self._vel_cmd.zero_()
        self._apply_vel_dedup()

    def _apply_vel_dedup(self):
        """
        Applies velocity only if the command has changed from the last step.
        Reduces overhead of repeated identical calls (e.g. during APPROACHING).
        """
        if not self._force_next_apply and torch.equal(self._vel_cmd, self._last_vel_cmd):
            return

        self.robot.apply_velocity(self._vel_cmd, dofs_idx_local=self._dofs_idx_local)
        self._last_vel_cmd.copy_(self._vel_cmd)
        self._force_next_apply = False

    def _force_idle_reset(self):
         self._stop_motors()
         self._force_next_apply = True
         
         current_qpos = self.qpos.clone()
         current_qpos[:, 2] = self.gripper_open_pos
         current_qpos[:, 3] = self.gripper_open_pos
         
         self.robot.set_control_mode("TELEPORT")
         self.qpos = current_qpos
         self.robot.apply_action(current_qpos)
         
         self.strike_state = StrikeState.IDLE
         self.contact_L = False
         self.contact_R = False
         self.contact_width = 0.0

    async def step_simulation(self):
        """
        Single atomic step of complexity:
        - Logic Update
        - Action Application
        - Force Clearing
        - Physics Step
        - Render Update
        - Reconstruction (if needed)
        """
        
        # 1. Logic Update
        with self._profile("teleop_logic"):
            await self.update_logic()
        
        # Track steps during active strike
        if self.strike_state != StrikeState.IDLE:
            self.strike_step_count += 1

        # 2. Apply Actions (if not handled by strike logic, handle idle holding)
        if self.strike_state == StrikeState.IDLE or self.strike_state == StrikeState.HOLDING:
            qpos = await self.get_qpos()
            self.robot.apply_action(qpos)

        # 3. Clear Forces
        if hasattr(self.env.scene.sim.coupler, 'clear_link_coupling_forces'):
            self.env.scene.sim.coupler.clear_link_coupling_forces()

        # 4. Physics Step
        with self._profile("teleop_physics"):
            try:
                self.env.scene.step(update_visualizer=False)
            except Exception as e:
                gs.logger.warning(f"Physics step skipped due to instability: {e}")

        # 5. Render Update
        if self.env.scene.visualizer:
            with self._profile("teleop_render"):
                self.env.scene.visualizer.update(force=False, auto=True)

    def _profile(self, name):
        # Fix for Pydantic model access
        teleop_opts = self.env.scene.profiling_options.configs.teleop
        opt_name = name.replace("teleop_", "")
        if getattr(teleop_opts, opt_name, False):
            return self.env.scene.profiling_options.profiler.time(name)
        return contextlib.suppress()

    async def update_and_get_recon_data(self):
        """Updates reconstruction and returns data for visualization/IO."""
        with self._profile("teleop_recon"):
            allowed_stages = (StrikeState.PRESSING, StrikeState.RELEASE)
            should_reconstruct = self.strike_state in allowed_stages
            
            if should_reconstruct:
                with self._profile("teleop_recon_update"):
                    self.reconstructor.update(should_reconstruct)
                with self._profile("teleop_recon_get_particles"):
                    particles = self.reconstructor.get_active_particle_cache()
                    if particles is None:
                         particles = self.env.mpm_entity.get_particles_pos(envs_idx=0).squeeze(0)
            else:
                with self._profile("teleop_recon_get_particles"):
                    particles = self.env.mpm_entity.get_particles_pos(envs_idx=0).squeeze(0)

            with self._profile("teleop_recon_transform"):
                points = self._apply_transformation(particles)
                vertices = self._apply_transformation(self.reconstructor.reconstructed_mesh.vertices)
            
            triangles = self.reconstructor.reconstructed_mesh.faces.copy()
            # Since we flip X axis in transformation, we must flip triangle winding to keep normals correct
            # Swap vertex 1 and 2 of each triangle
            triangles[:, [1, 2]] = triangles[:, [2, 1]]
            
            return vertices, triangles, points

    def _apply_transformation(self, points):
        """Transform points to Unity space."""
        if isinstance(points, torch.Tensor):
            points = points + self.unity_translation_tensor
            points = points * self.unity_scale_tensor
            points[:, 0] *= -1 
        else: 
            points = points + self.unity_translation.reshape(1, 3)
            points = points * self.unity_scale.reshape(1, 3)
            points[:, 0] *= -1
        return points

    def _save_checkpoint_impl(self):
        # Genesis SimState
        sim_state = self.env.scene.sim.get_state()
        sim_state.serializable() # Ensure CPU friendly if needed
        
        # Clear specific internal cache if needed (from original code)
        if hasattr(self.env.scene.sim, '_queried_states'):
            self.env.scene.sim._queried_states.clear()

        ckpt = {
            'sim_state': sim_state,
            'strike_state': self.strike_state,
            'qpos': self.qpos.clone(),
            'recon_state': self.reconstructor.get_state()
        }
        self.checkpoints.append(ckpt)
        if len(self.checkpoints) > MAX_CHECKPOINTS:
            self.checkpoints.pop(0)
        mesh_verts = len(self.reconstructor.reconstructed_mesh.vertices) if self.reconstructor.reconstructed_mesh.vertices is not None else 0
        gs.logger.info(f"Checkpoint saved (stack={len(self.checkpoints)}, mesh_verts={mesh_verts})")

    async def save_checkpoint(self):
        async with self.lock:
            self._save_checkpoint_impl()

    async def load_checkpoint(self):
        async with self.lock:
            if len(self.checkpoints) < 2:
                gs.logger.warning("No previous checkpoint to undo to")
                return

            self.checkpoints.pop()
            ckpt = self.checkpoints[-1]
            
            self.env.scene.sim.reset(ckpt['sim_state'])
            
            import gstaichi as ti
            ti.sync()
            if hasattr(self.env.scene.sim.mpm_solver, 'update_render_fields'):
                self.env.scene.sim.mpm_solver.update_render_fields()
            else:
                self.env.scene.visualizer.update_visual_states()
            
            self.strike_state = StrikeState.IDLE
            self.contact_L = False
            self.contact_R = False
            self.contact_width = 0.0
            
            self.qpos = ckpt['qpos'].clone()
            self.qpos[:, 2] = self.gripper_open_pos
            self.qpos[:, 3] = self.gripper_open_pos
            
            self.robot.set_control_mode("TELEPORT")
            self.robot.apply_action(self.qpos)
            self.robot.apply_action(self.qpos) # Apply a few times to settle
            
            if 'recon_state' in ckpt:
                self.reconstructor.set_state(ckpt['recon_state'])
            else:
                self.reconstructor.reset()
                self.reconstructor.create_reconstructed_mesh()

            # Trigger mesh data send to client (without physics step)
            self.pending_mesh_send = True
            
            mesh_verts = len(self.reconstructor.reconstructed_mesh.vertices) if self.reconstructor.reconstructed_mesh.vertices is not None else 0
            gs.logger.info(f"Undo complete (stack={len(self.checkpoints)}, mesh_verts={mesh_verts})")

    async def reset_simulation(self):
        async with self.lock:
            self.env.reset()
            
            import gstaichi as ti
            ti.sync()
            if hasattr(self.env.scene.sim.mpm_solver, 'update_render_fields'):
                self.env.scene.sim.mpm_solver.update_render_fields()
            else:
                self.env.scene.visualizer.update_visual_states()

            self.qpos = self.robot.entity.get_dofs_position()
            self.strike_state = StrikeState.IDLE
            self.contact_L = False
            self.contact_R = False
            self.checkpoints = []
            self.contact_width = 0.0
            self.reconstructor.reset()
            
            gs.logger.info("Initializing surface skinning...")
            self.reconstructor.create_reconstructed_mesh()
            self.reconstructor.init_skinning()
            
            self._save_checkpoint_impl()
            
            # Trigger mesh data send to client (without physics step)
            self.pending_mesh_send = True
            
            gs.logger.info("Simulation reset")

```

### agforge/options.py
```python
import torch
import numpy as np
from pint import UnitRegistry
from typing import Tuple, List

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
    log_dir: str = "logs/agforge_parametric"
    
class ReconstructionOptions(Options):
    """Parameters for surface reconstruction."""
    grid_res: int = 64
    backend: str = "hybrid"  # 'hybrid' or 'splashsurf'
    enabled: bool = True

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
            dt=1.4e-6 * 8,
            substeps=8,
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
            self.vis.visualize_mpm_boundary = True
            self.vis.visualize_mpm_grid = True
            self.vis.shadow = True
            self.vis.plane_reflection = True
            self.vis.particle_render_fraction = 1.0


class TrainingOptions(AgilityForgeOptions):
    """Aggregated configuration for training."""

class StrikeOptions(Options):
    """Parameters for the approaching and pressing stage."""
    approach_speed: float = 4.e1
    contact_force_threshold: float = 150.0 # Force threshold to detect contact
    
    # Pressing Stage
    pressing_speed: float = 1.e1 # m/s
    force_balance_gain: float = 1.e-4 # (m/s) / N. Start with 0.0 for constant velocity testing as requested.
    target_strain: float = 0.1 # 10% compression
    max_force: float = 50000.0 # N
    pressing_timeout: float = 15.0 # seconds
    approaching_timeout: float = 10.0 # seconds
    release_timeout: float = 10.0 # seconds
    post_release_steps: int = 10 # steps
    
class AdaptiveControlConfig(Options):
    """Configuration for adaptive control gains."""
    base_kp: float = 5000.0
    base_kv: float = 200.0
    mass_scale_factor: float = 1.0

class TeleopOptions(AgilityForgeOptions):
    """Aggregated configuration for teleoperation."""
    strike: StrikeOptions = StrikeOptions()
    adaptive_control: AdaptiveControlConfig = AdaptiveControlConfig()
    print_profiling_on_exit: bool = True  # Print profiler visualizations on shutdown
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

```
