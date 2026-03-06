# Genesis MPM Thermo-Mechanical Forge Scaling Research
    
## 1. System Overview
The Genesis Material Point Method (MPM) engine has been extended to simulate fully coupled thermo-mechanical interactions. Our simulation tracks **Temperature ($T$)** as a fundamental property of every particle and grid cell alongside stress and velocity.

### Implemented Physics
1.  **Adiabatic Heating (Mechanical $\rightarrow$ Thermal):** We implemented the **Johnson-Cook Elasto-Plasticity** model. During deformation, mechanical plastic work ($W_p$) is converted entirely into internal thermal energy ($\Delta T = W_p / (\rho \cdot C_p)$). As the temperature rises, the yield strength of the material drops exponentially (thermal softening).
2.  **Newton's Law of Cooling (Convection):** We detect boundary cells with empty neighborhoods and apply baseline convective cooling against the `thermal_air_conductivity` ($h_{air}$).
3.  **SDF Contact Conduction:** We perform bounding checks against the rigid solver's Signed Distance Field (SDF). When MPM grid cells contact heavy solid tools (anvils/hammers), they dump heat rapidly using `thermal_contact_conductivity` ($h_{contact}$), overriding air convection.
4.  **Mass-Conservative Volume-Fraction Laplacian:** To diffuse heat internally without destroying energy across fractional boundary cells, we implemented a conditionally stable Laplacian flux operator bounded by $min(M_{neighbor}, M_{center})$. The energy conservation mathematical drift is floating-point perfect.

## 2. The Core Bottleneck: High-Speed Elasto-Plasticity vs. Real-World Thermodynamics
We are currently simulating **hot steel forging**. This introduces a catastrophic timing mismatch.

To simulate the violent rigid-to-deformable high-speed impacts of a forging hammer without mathematical explosions (NaNs), the MPM solver is strictly bound by a microscopic **CFL (Courant-Friedrichs-Lewy) timestep limit** (e.g., $dt = 1.12 \times 10^{-5}$ seconds).

**The Timeline Mismatch:**
*   Mechanical deformation during a hammer strike happens in fractions of a second.
*   Thermal radiation, convection, and conduction cooling cycles for large steel billets happen over *minutes*.
*   Because the computational limits restrict our total simulated elapsed time to a few milliseconds, a purely physical reading of steel thermal parameters results in virtually **zero temperature change** during the simulation. 

## 3. The Objective: Optimal Parameter Tuning for Accuracy, Convergence, and Performance
To allow the thermal fields to interact simultaneously with the mechanical fields during the forging process, we must artificially fast-forward the "thermal mathematical clock" while executing slow-motion physics. 

We need to research State-of-the-Art (SOTA) scaling methodologies to inflate the thermal parameters:
*   $C_p$ (Specific Heat Capacity)
*   $\alpha$ (Thermal Diffusivity)
*   $h_{air}$ (Air Convection)
*   $h_{contact}$ (Contact Conduction)
*   $jc\_m$ (Johnson-Cook Thermal Softening Exponent)

The research must evaluate how to scale these specific values mathematically such that we achieve:
1.  **Accuracy:** The mathematical laws of thermodynamics and volumetric energy conservation remain physically true to steel, producing an exact but vastly accelerated cooling/heating gradient. 
2.  **Convergence / Stability:** Ensuring the artificially inflated thermal fluxes reliably compute without exceeding explicitly bounding limits ($\Delta t_{max} = dx^2 / (6\alpha)$), creating massive force/stress shockwaves in the return mapping algorithms, or destroying the MPM rigid-body collision solver geometry.
3.  **Performance / Speed:** Ensuring the chosen scaling formula can run smoothly, fast enough inside the Taichi `mpm_grid_op` nested GPU loops to be fully viable for iterative machine learning and continuous testing.

## 4. Source Code Appendices (Genesis Engine)
Below are the three core Python implementation files that represent the underlying thermo-mechanical logic: `base_mpm_solver.py`, `legacy_coupler.py`, and `elasto_plastic.py`.

```python
from typing import TYPE_CHECKING

import numpy as np
import gstaichi as ti
import torch

import genesis as gs
import genesis.utils.array_class as array_class
import genesis.utils.geom as gu
import genesis.utils.sdf as sdf
from genesis.engine.boundaries import CubeBoundary
from genesis.engine.entities import MPMEntity
from genesis.engine.states.solvers import MPMSolverState
from genesis.options.solvers import MPMOptions
from genesis.utils.misc import DeprecationError
import contextlib

from .base_solver import Solver

if TYPE_CHECKING:
    from genesis.engine.scene import Scene
    from genesis.engine.solvers.base_solver import Solver
    from genesis.engine.simulator import Simulator


@ti.data_oriented
class BaseMPMSolver(Solver):
    # ------------------------------------------------------------------------------------
    # --------------------------------- Initialization -----------------------------------
    # ------------------------------------------------------------------------------------

    def __init__(self, scene: "Scene", sim: "Simulator", options: "MPMOptions"):
        super().__init__(scene, sim, options)

        # options
        self._grid_density = options.grid_density
        self._particle_size = options.particle_size
        self._upper_bound = np.array(options.upper_bound)
        self._lower_bound = np.array(options.lower_bound)
        self._enable_CPIC = options.enable_CPIC
        self._constraints_initialized = False

        # Thermal config
        self._enable_thermal = options.enable_thermal
        self._default_initial_temperature = options.default_initial_temperature
        self._default_heat_capacity = options.default_heat_capacity
        self._h_contact = options.thermal_contact_conductivity
        self._h_air = options.thermal_air_conductivity
        self._alpha_thermal = options.default_thermal_diffusivity

        self._n_vvert_supports = self.scene.vis_options.n_support_neighbors

        # `_particle_volume_scale` is used to avoid potential numerical instability, as the actual `_particle_volume` may be very small.
        # Note that the magnitude of `_particle_volume` doesn't affect MPM simulation itself, but it is used to compute particle
        # mass. We need to account for this scale when handling coupling.
        self._particle_volume_real = float(self._particle_size**3)
        self._particle_volume_scale = 1e3
        self._particle_volume = self._particle_volume_real * self._particle_volume_scale

        # Other derived parameters
        self._dx = float(1.0 / self._grid_density)
        self._inv_dx = float(self._grid_density)
        self._lower_bound_cell = np.round(self._grid_density * self._lower_bound).astype(gs.np_int)
        self._upper_bound_cell = np.round(self._grid_density * self._upper_bound).astype(gs.np_int)
        self._grid_res = self._upper_bound_cell - self._lower_bound_cell + 1  # +1 to include both corner
        gs.logger.info(f"Grid resolution: {self._grid_res} {self._grid_res.prod()}")
        self._grid_offset = ti.Vector(self._lower_bound_cell)
        if np.prod(self._grid_res) > 1e9:
            gs.raise_exception(
                "Grid size larger than 1e9 not supported by MPM solver. Please reduce 'grid_density', or set tighter "
                "boundaries via 'lower_bound' / 'upper_bound'."
            )

        # materials
        self._materials = list()
        self._materials_idx = list()
        self._materials_update_F_S_Jp = list()
        self._materials_update_stress = list()

        # boundary
        self.setup_boundary()

    def setup_boundary(self):
        # safety padding
        self.boundary_padding = 3 * self._dx
        self.boundary = CubeBoundary(
            lower=self._lower_bound + self.boundary_padding,
            upper=self._upper_bound - self.boundary_padding,
        )

    def _make_particle_state_template(self):
        template = {
            "pos": gs.ti_vec3,  # position
            "vel": gs.ti_vec3,  # velocity
            "C": gs.ti_mat3,  # affine velocity field
            "F": gs.ti_mat3,  # deformation gradient
            "F_tmp": gs.ti_mat3,  # temp deformation gradient
            "U": gs.ti_mat3,  # SVD
            "V": gs.ti_mat3,  # SVD
            "S": gs.ti_mat3,  # SVD
            "actu": gs.ti_float,  # actuation
            "Jp": gs.ti_float,  # volume ratio
        }
        if self._enable_thermal:
            template.update({
                "temp": gs.ti_float,
                "plastic_strain": gs.ti_float,
                "plastic_work": gs.ti_float,
            })
        return template

    def _make_particle_state_ng_template(self):
        return {
            "active": gs.ti_bool,
        }

    def _make_particle_info_template(self):
        return {
            "material_idx": gs.ti_int,
            "mass": gs.ti_float,
            "default_Jp": gs.ti_float,
            "free": gs.ti_bool,
            # for muscle
            "muscle_group": gs.ti_int,
            "muscle_direction": gs.ti_vec3,
        }

    def _make_particle_state_render_template(self):
        return {
            "pos": gs.ti_vec3,
            "vel": gs.ti_vec3,
            "active": gs.ti_bool,
        }

    def init_particle_fields(self):
        # dynamic particle state
        struct_particle_state = ti.types.struct(**self._make_particle_state_template())
        self._zero_particle_state = struct_particle_state.filled_with_scalar(0.0)

        # dynamic particle state without gradient
        struct_particle_state_ng = ti.types.struct(**self._make_particle_state_ng_template())

        # static particle info
        struct_particle_info = ti.types.struct(**self._make_particle_info_template())

        # single frame particle state for rendering
        struct_particle_state_render = ti.types.struct(**self._make_particle_state_render_template())

        # construct fields
        self.particles = struct_particle_state.field(
            shape=(self._sim.substeps_local + 1, self._n_particles, self._B),
            needs_grad=True,
            layout=ti.Layout.SOA,
        )
        self.particles_ng = struct_particle_state_ng.field(
            shape=(self._sim.substeps_local + 1, self._n_particles, self._B),
            needs_grad=False,
            layout=ti.Layout.SOA,
        )
        self.particles_info = struct_particle_info.field(
            shape=self._n_particles, needs_grad=False, layout=ti.Layout.SOA
        )
        self.particles_render = struct_particle_state_render.field(
            shape=(self._n_particles, self._B), needs_grad=False, layout=ti.Layout.SOA
        )
        if self._enable_thermal:
            self.particles.temp.fill(self._default_initial_temperature)

    def _make_grid_cell_state_template(self):
        template = {
            "mass": gs.ti_float,  # mass
            "vel_in": gs.ti_vec3,  # input momentum/velocity
            "vel_out": gs.ti_vec3,  # output momentum/velocity
        }
        if self._enable_thermal:
            template.update({
                "temp": gs.ti_float,
                "temp_diffused": gs.ti_float,
                "mass_thermal": gs.ti_float,
            })
        return template

    def init_grid_fields(self):
        grid_cell_state = ti.types.struct(**self._make_grid_cell_state_template())
        self._zero_grid_cell_state = grid_cell_state.filled_with_scalar(0.0)
        self.grid = grid_cell_state.field(
            shape=(self._sim.substeps_local + 1, *self._grid_res, self._B),
            needs_grad=True,
            layout=ti.Layout.SOA,
        )

    def init_vvert_fields(self):
        struct_vvert_info = ti.types.struct(
            support_idxs=ti.types.vector(self._n_vvert_supports, gs.ti_int),
            support_weights=ti.types.vector(self._n_vvert_supports, gs.ti_float),
        )
        self.vverts_info = struct_vvert_info.field(shape=(max(1, self._n_vverts),), layout=ti.Layout.SOA)

        struct_vvert_state_render = ti.types.struct(
            pos=gs.ti_vec3,
            active=gs.ti_bool,
        )
        self.vverts_render = struct_vvert_state_render.field(
            shape=(max(1, self._n_vverts), self._B), layout=ti.Layout.SOA
        )

    def init_ckpt(self):
        self._ckpt = dict()

    def init_constraints(self):
        """Lazy initialization of particle constraint fields."""
        # Memory check: ensure index fits in int32
        if self._n_particles * self._B * 3 > np.iinfo(np.int32).max:
            gs.raise_exception(
                f"Particle constraint shape (n_envs={self._B}, n_particles={self._n_particles}, 3) is too large. "
                "Consider reducing n_envs or n_particles."
            )

        self._constraints_initialized = True

        particle_constraint_info = ti.types.struct(
            is_constrained=gs.ti_bool,  # whether particle is constrained
            target_pos=gs.ti_vec3,  # target position for the constraint
            stiffness=gs.ti_float,  # spring stiffness
            link_idx=gs.ti_int,  # index of the rigid link (-1 if not linked)
            link_local_pos=gs.ti_vec3,  # offset from link origin in link's local frame
        )

        self.particle_constraints = particle_constraint_info.field(
            shape=(self._n_particles, self._B), needs_grad=False, layout=ti.Layout.AOS
        )

        self.particle_constraints.is_constrained.fill(False)
        self.particle_constraints.link_idx.fill(-1)

    def reset_grad(self):
        self.particles.grad.fill(0.0)
        self.grid.grad.fill(0.0)

        for entity in self._entities:
            entity.reset_grad()

    def build(self):
        super().build()

        # particles and entities
        self._B = self._sim._B
        self._n_particles = self.n_particles
        self._n_vverts = self.n_vverts
        self._n_vfaces = self.n_vfaces

        self._coupler = self.sim._coupler

        if self.is_active:
            if self._enable_CPIC and self._sim.requires_grad:
                gs.raise_exception(
                    "CPIC is not supported in differentiable mode yet. Submit a feature request if you need it."
                )

            self.init_particle_fields()
            self.init_grid_fields()
            self.init_vvert_fields()
            self.init_ckpt()

            for entity in self._entities:
                entity._add_to_solver()

            # See: https://github.com/taichi-dev/taichi_elements/blob/d19678869a28b09a32ef415b162e35dc929b792d/engine/mpm_solver.py#L84
            suggested_dt = 2e-2 * self._dx
            if self.substep_dt > suggested_dt:
                gs.logger.warning(
                    f"Current `substep_dt` ({self.substep_dt:.6g}) is greater than suggested_dt ({suggested_dt:.6g}, "
                    "calculated based on `grid_density`). Simulation might be unstable."
                )
                
            if self._enable_thermal:
                dt_cfl = self._dx ** 2 / (6.0 * self._alpha_thermal)
                if self.substep_dt > dt_cfl:
                    gs.logger.warning(
                        f"Current `substep_dt` ({self.substep_dt:.6g}) exceeds the thermal diffusion CFL limit "
                        f"({dt_cfl:.6g}) for alpha={self._alpha_thermal}. Heat diffusion may mathematically explode."
                    )

        # Overwrite gravity because only field is supported for now
        if self._gravity is not None:
            gravity = self._gravity.to_numpy()
            self._gravity = ti.field(dtype=gs.ti_vec3, shape=(self._B,))
            self._gravity.from_numpy(gravity)

    # ------------------------------------------------------------------------------------
    # -------------------------------------- misc ----------------------------------------
    # ------------------------------------------------------------------------------------

    @property
    def is_active(self):
        return self.n_particles > 0

    def add_entity(self, idx, material, morph, surface):
        self.add_material(material)

        # create entity
        entity = MPMEntity(
            scene=self._scene,
            solver=self,
            material=material,
            morph=morph,
            surface=surface,
            particle_size=self._particle_size,
            idx=idx,
            particle_start=self.n_particles,
            vvert_start=self.n_vverts,
            vface_start=self.n_vfaces,
        )
        self._entities.append(entity)

        return entity

    def add_material(self, material):
        # Register material update methods if and only if the provided material is not already registered
        for material_i in self._materials:
            if material == material_i:
                material._idx = material_i._idx
                break
        else:
            material._idx = len(self._materials_idx)
            self._materials_idx.append(material._idx)
            self._materials_update_F_S_Jp.append(material.update_F_S_Jp)
            self._materials_update_stress.append(material.update_stress)
        self._materials.append(material)

    @ti.func
    def stencil_range(self):
        return ti.ndrange(3, 3, 3)

    @ti.func
    def get_particle_thermal_state(self, f: ti.i32, i_p: ti.i32, i_b: ti.i32):
        temp = gs.ti_float(293.15)
        plastic_strain = gs.ti_float(0.0)
        plastic_work = gs.ti_float(0.0)
        if ti.static(self._enable_thermal):
            temp = self.particles[f, i_p, i_b].temp
            plastic_strain = self.particles[f, i_p, i_b].plastic_strain
            plastic_work = self.particles[f, i_p, i_b].plastic_work
        return temp, plastic_strain, plastic_work

    @ti.func
    def p2g_post_constitutive(self, f, i_p, i_b, delta_gamma, effective_yield):
        if ti.static(self._enable_thermal):
            fraction = 0.9
            rho = self.particles_info[i_p].mass / ti.max(self._particle_volume, gs.EPS)
            Cp = self._default_heat_capacity
            if delta_gamma > 0.0:
                vol_work = effective_yield * delta_gamma
                self.particles[f, i_p, i_b].plastic_strain += delta_gamma
                self.particles[f, i_p, i_b].plastic_work += vol_work
                dT = fraction * vol_work / (rho * Cp)
                self.particles[f + 1, i_p, i_b].temp = self.particles[f, i_p, i_b].temp + dT
            else:
                self.particles[f + 1, i_p, i_b].temp = self.particles[f, i_p, i_b].temp

    @ti.func
    def p2g_transfer_extra_fields(self, f, i_p, idx: ti.template(), i_b, weight):
        if ti.static(self._enable_thermal):
            mass = self.particles_info[i_p].mass
            self.grid[f, idx, i_b].mass_thermal += weight * mass
            self.grid[f, idx, i_b].temp += weight * mass * self.particles[f + 1, i_p, i_b].temp

    @ti.func
    def g2p_prologue(self, f, i_p, i_b):
        if ti.static(self._enable_thermal):
            self.particles[f + 1, i_p, i_b].temp = 0.0

    @ti.func
    def g2p_transfer_extra_fields(self, f, i_p, i_b, weight, grid_index: ti.template()):
        if ti.static(self._enable_thermal):
            self.particles[f + 1, i_p, i_b].temp += weight * self.grid[f, grid_index, i_b].temp_diffused

    # ------------------------------------------------------------------------------------
    # ----------------------------------- simulation -------------------------------------
    # ------------------------------------------------------------------------------------

    @ti.kernel
    def compute_F_tmp(self, f: ti.i32):
        for i_p, i_b in ti.ndrange(self._n_particles, self._B):
            if self.particles_ng[f, i_p, i_b].active:
                self.particles[f, i_p, i_b].F_tmp = (
                    ti.Matrix.identity(gs.ti_float, 3) + self.substep_dt * self.particles[f, i_p, i_b].C
                ) @ self.particles[f, i_p, i_b].F

    @ti.kernel
    def svd(self, f: ti.i32):
        for i_p, i_b in ti.ndrange(self._n_particles, self._B):
            if self.particles_ng[f, i_p, i_b].active:
                self.particles[f, i_p, i_b].U, self.particles[f, i_p, i_b].S, self.particles[f, i_p, i_b].V = ti.svd(
                    self.particles[f, i_p, i_b].F_tmp, gs.ti_float
                )

    @ti.kernel
    def svd_grad(self, f: ti.i32):
        for i_p, i_b in ti.ndrange(self._n_particles, self._B):
            if self.particles_ng[f, i_p, i_b].active:
                self.particles.grad[f, i_p, i_b].F_tmp += backward_svd(
                    self.particles.grad[f, i_p, i_b].U,
                    self.particles.grad[f, i_p, i_b].S,
                    self.particles.grad[f, i_p, i_b].V,
                    self.particles[f, i_p, i_b].U,
                    self.particles[f, i_p, i_b].S,
                    self.particles[f, i_p, i_b].V,
                )

    @ti.func
    def p2g_helper(
        self,
        f: ti.i32,
        i_p: ti.i32,
        i_b: ti.i32,
        geoms_state: array_class.GeomsState,
        geoms_info: array_class.GeomsInfo,
        links_state: array_class.LinksState,
        rigid_global_info: array_class.RigidGlobalInfo,
        sdf_info: array_class.SDFInfo,
        collider_static_config: ti.template(),
    ):
        # A. update F (deformation gradient), S (Sigma from SVD(F), essentially represents volume) and Jp
        # (volume compression ratio) based on material type
        J = self.particles[f, i_p, i_b].S.determinant()
        F_new = ti.Matrix.zero(gs.ti_float, 3, 3)
        S_new = ti.Matrix.zero(gs.ti_float, 3, 3)
        Jp_new = gs.ti_float(1.0)
        p_temp, _, _ = self.get_particle_thermal_state(f, i_p, i_b)
        
        for material_idx in ti.static(self._materials_idx):
            if self.particles_info[i_p].material_idx == material_idx:
                F_new, S_new, Jp_new, delta_gamma, effective_yield = self._materials_update_F_S_Jp[material_idx](
                    J=J,
                    F_tmp=self.particles[f, i_p, i_b].F_tmp,
                    U=self.particles[f, i_p, i_b].U,
                    S=self.particles[f, i_p, i_b].S,
                    V=self.particles[f, i_p, i_b].V,
                    Jp=self.particles[f, i_p, i_b].Jp,
                    temp=p_temp,
                )
                self.p2g_post_constitutive(f, i_p, i_b, delta_gamma, effective_yield)
                
        self.particles[f + 1, i_p, i_b].F = F_new
        self.particles[f + 1, i_p, i_b].Jp = Jp_new

        # B. compute stress
        # NOTE:
        # 1. Here we pass in both F_tmp and the updated F_new because in the official taichi example, F_new is
        # used for stress computation. However, although this works for both elastic and elasto-plastic
        # materials, it is mathematically incorrect for liquid material with non-zero viscosity (mu). In the
        # latter case, stress computation needs to be based on the F_tmp (deformation gradient before resetting
        # to identity).
        # 2. Jp is only used by Snow material, and it uses Jp from the previous frame, not the updated one.
        stress = ti.Matrix.zero(gs.ti_float, 3, 3)
        for material_idx in ti.static(self._materials_idx):
            if self.particles_info[i_p].material_idx == material_idx:
                stress = self._materials_update_stress[material_idx](
                    U=self.particles[f, i_p, i_b].U,
                    S=S_new,
                    V=self.particles[f, i_p, i_b].V,
                    F_tmp=self.particles[f, i_p, i_b].F_tmp,
                    F_new=F_new,
                    J=J,
                    Jp=self.particles[f, i_p, i_b].Jp,
                    actu=self.particles[f, i_p, i_b].actu,
                    m_dir=self.particles_info[i_p].muscle_direction,
                )
        stress = (-self.substep_dt * self._particle_volume * 4 * self._inv_dx * self._inv_dx) * stress
        affine = stress + self.particles_info[i_p].mass * self.particles[f, i_p, i_b].C

        # C. project onto grid
        base = ti.floor(self.particles[f, i_p, i_b].pos * self._inv_dx - 0.5).cast(gs.ti_int)
        fx = self.particles[f, i_p, i_b].pos * self._inv_dx - base.cast(gs.ti_float)
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
        for offset in ti.static(ti.grouped(self.stencil_range())):
            dpos = (offset.cast(gs.ti_float) - fx) * self._dx
            weight = gs.ti_float(1.0)
            for d in ti.static(range(3)):
                weight *= w[offset[d]][d]

            sep_geom_idx = -1
            if ti.static(self._enable_CPIC and self.sim.rigid_solver.is_active):
                # check if particle and cell center are at different side of any thin object
                cell_pos = (base + offset) * self._dx

                for i_g in range(self.sim.rigid_solver.n_geoms):
                    if geoms_info.needs_coup[i_g]:
                        sdf_normal_particle = self._coupler.mpm_rigid_normal[i_p, i_g, i_b]
                        sdf_normal_cell = sdf.sdf_func_normal_world(
                            geoms_state=geoms_state,
                            geoms_info=geoms_info,
                            rigid_global_info=rigid_global_info,
                            collider_static_config=collider_static_config,
                            sdf_info=sdf_info,
                            pos_world=cell_pos,
                            geom_idx=i_g,
                            batch_idx=i_b,
                        )

                        if sdf_normal_particle.dot(sdf_normal_cell) < 0:  # separated by geom i_g
                            sep_geom_idx = i_g
                            break
                self._coupler.cpic_flag[i_p, offset[0], offset[1], offset[2], i_b] = sep_geom_idx
            if sep_geom_idx == -1:
                self.grid[f, base - self._grid_offset + offset, i_b].vel_in += weight * (
                    self.particles_info[i_p].mass * self.particles[f, i_p, i_b].vel + affine @ dpos
                )
                self.grid[f, base - self._grid_offset + offset, i_b].mass += (
                    weight * self.particles_info[i_p].mass
                )
                self.p2g_transfer_extra_fields(f, i_p, base - self._grid_offset + offset, i_b, weight)

            if not self.particles_info[i_p].free:  # non-free particles behave as boundary conditions
                self.grid[f, base - self._grid_offset + offset, i_b].vel_in = ti.Vector.zero(gs.ti_float, 3)

    @ti.kernel
    def p2g(
        self,
        f: ti.i32,
        geoms_state: array_class.GeomsState,
        geoms_info: array_class.GeomsInfo,
        links_state: array_class.LinksState,
        rigid_global_info: array_class.RigidGlobalInfo,
        sdf_info: array_class.SDFInfo,
        collider_static_config: ti.template(),
    ):
        for i_p, i_b in ti.ndrange(self._n_particles, self._B):
            if self.particles_ng[f, i_p, i_b].active:
                self.p2g_helper(f, i_p, i_b, geoms_state, geoms_info, links_state, rigid_global_info, sdf_info, collider_static_config)

    @ti.func
    def g2p_helper(
        self,
        f: ti.i32,
        i_p: ti.i32,
        i_b: ti.i32,
        geoms_info: array_class.GeomsInfo,
        links_state: array_class.LinksState,
        rigid_global_info: array_class.RigidGlobalInfo,
    ):
        self.g2p_prologue(f, i_p, i_b)
        base = ti.floor(self.particles[f, i_p, i_b].pos * self._inv_dx - 0.5).cast(gs.ti_int)
        fx = self.particles[f, i_p, i_b].pos * self._inv_dx - base.cast(gs.ti_float)
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
        new_vel = ti.Vector.zero(gs.ti_float, 3)
        new_C = ti.Matrix.zero(gs.ti_float, 3, 3)
        for offset in ti.static(ti.grouped(self.stencil_range())):
            dpos = offset.cast(gs.ti_float) - fx
            grid_vel = self.grid[f, base - self._grid_offset + offset, i_b].vel_out
            weight = gs.ti_float(1.0)
            for d in ti.static(range(3)):
                weight *= w[offset[d]][d]

            if ti.static(self._enable_CPIC and self.sim.rigid_solver.is_active):
                sep_geom_idx = self._coupler.cpic_flag[i_p, offset[0], offset[1], offset[2], i_b]
                if sep_geom_idx != -1:
                    grid_vel = self.sim.coupler._func_collide_in_rigid_geom(
                        self.particles[f, i_p, i_b].pos,
                        self.particles[f, i_p, i_b].vel,
                        self.particles_info[i_p].mass * weight / self._particle_volume_scale,
                        self._coupler.mpm_rigid_normal[i_p, sep_geom_idx, i_b],
                        1.0,
                        sep_geom_idx,
                        i_b,
                        geoms_info=geoms_info,
                        links_state=links_state,
                        rigid_global_info=rigid_global_info,
                    )

            new_vel += weight * grid_vel
            new_C += 4 * self._inv_dx * weight * grid_vel.outer_product(dpos)
            self.g2p_transfer_extra_fields(f, i_p, i_b, weight, base - self._grid_offset + offset)

        # compute actual new_pos with new_vel
        new_pos = self.particles[f, i_p, i_b].pos + self.substep_dt * new_vel

        # impose boundary for safety, in case simulation explodes and tries to access illegal cell address
        new_pos, new_vel = self.boundary.impose_pos_vel(new_pos, new_vel)

        # advect to next frame
        self.particles[f + 1, i_p, i_b].vel = new_vel
        self.particles[f + 1, i_p, i_b].C = new_C
        self.particles[f + 1, i_p, i_b].pos = new_pos
    @ti.kernel
    def g2p(
        self,
        f: ti.i32,
        geoms_info: array_class.GeomsInfo,
        links_state: array_class.LinksState,
        rigid_global_info: array_class.RigidGlobalInfo,
    ):
        for i_p, i_b in ti.ndrange(self._n_particles, self._B):
            if self.particles_ng[f, i_p, i_b].active:
                self.g2p_helper(f, i_p, i_b, geoms_info, links_state, rigid_global_info)
            else:
                self.copy_frame_helper(f, f + 1, i_p, i_b)

            self.particles_ng[f + 1, i_p, i_b].active = self.particles_ng[f, i_p, i_b].active

    @ti.kernel
    def _is_state_valid(self, f: ti.i32) -> ti.i32:
        is_success = True
        for i_p, i_b, i_3 in ti.ndrange(self._n_particles, self._B, 3):
            if ti.math.isnan(self.particles[f, i_p, i_b].pos[i_3]):
                is_success = False
        return is_success

    # ------------------------------------------------------------------------------------
    # ------------------------------------ stepping --------------------------------------
    # ------------------------------------------------------------------------------------

    def process_input(self, in_backward=False):
        for entity in self._entities:
            entity.process_input(in_backward=in_backward)

    def process_input_grad(self):
        for entity in self._entities[::-1]:
            entity.process_input_grad()

    def substep_pre_coupling(self, f):
        profiler = self.sim.scene.profiling_options.profiler
        with profiler.time("mpm_pre_couple_ops") if True else contextlib.suppress():
            with profiler.time("mpm_reset_grid_grad") if True else contextlib.suppress():
                self.reset_grid_and_grad(f)
            with profiler.time("mpm_compute_F_tmp") if True else contextlib.suppress():
                self.compute_F_tmp(f)
            with profiler.time("mpm_svd") if True else contextlib.suppress():
                self.svd(f)
            with profiler.time("mpm_p2g") if True else contextlib.suppress():
                self.p2g(
                    f,
                    self.sim.coupler.rigid_solver.geoms_state,
                    self.sim.coupler.rigid_solver.geoms_info,
                    self.sim.coupler.rigid_solver.links_state,
                    self.sim.coupler.rigid_solver._rigid_global_info,
                    self.sim.coupler.rigid_solver.sdf._sdf_info,
                    self.sim.coupler.rigid_solver.collider._collider_static_config,
                )

    def substep_pre_coupling_grad(self, f):
        self.p2g.grad(
            f,
            self.sim.coupler.rigid_solver.geoms_state,
            self.sim.coupler.rigid_solver.geoms_info,
            self.sim.coupler.rigid_solver.links_state,
            self.sim.coupler.rigid_solver._rigid_global_info,
            self.sim.coupler.rigid_solver.sdf._sdf_info,
            self.sim.coupler.rigid_solver.collider._collider_static_config,
        )
        self.svd_grad(f)
        self.compute_F_tmp.grad(f)

    def substep_post_coupling(self, f):
        profiler = self.sim.scene.profiling_options.profiler
        with profiler.time("mpm_post_couple") if True else contextlib.suppress():
            with profiler.time("mpm_g2p") if self.sim.scene.profiling_options.configs.simulator.mpm_g2p else contextlib.suppress():
                self.g2p(
                    f,
                    self.sim.coupler.rigid_solver.geoms_info,
                    self.sim.coupler.rigid_solver.links_state,
                    self.sim.coupler.rigid_solver._rigid_global_info,
                )

            # Apply particle constraints after g2p
            if self._constraints_initialized:
                with profiler.time("mpm_apply_constraints") if True else contextlib.suppress():
                    self.apply_particle_constraints(f, self.sim.coupler.rigid_solver.links_state)

            # FIXME: Use existing errno mechanism for this.
            if self.sim.options.check_bounds:
                with profiler.time("mpm_check_valid") if True else contextlib.suppress():
                    is_valid = True
                    with profiler.time("mpm_check_valid_sync") if True else contextlib.suppress():
                        is_valid = self._is_state_valid(f)
                    
                    if not is_valid:
                        gs.raise_exception(
                            "NaN detected in MPM states. Try reducing the time step size or adjusting simulation parameters."
                        )

    def substep_post_coupling_grad(self, f):
        self.g2p.grad(
            f,
            self.sim.coupler.rigid_solver.geoms_info,
            self.sim.coupler.rigid_solver.links_state,
            self.sim.coupler.rigid_solver._rigid_global_info,
        )

    @ti.func
    def copy_frame_helper(self, source: ti.i32, target: ti.i32, i_p: ti.i32, i_b: ti.i32):
        self.particles[target, i_p, i_b].pos = self.particles[source, i_p, i_b].pos
        self.particles[target, i_p, i_b].vel = self.particles[source, i_p, i_b].vel
        self.particles[target, i_p, i_b].F = self.particles[source, i_p, i_b].F
        self.particles[target, i_p, i_b].C = self.particles[source, i_p, i_b].C
        self.particles[target, i_p, i_b].Jp = self.particles[source, i_p, i_b].Jp
        if ti.static(self._enable_thermal):
            self.particles[target, i_p, i_b].temp = self.particles[source, i_p, i_b].temp
            self.particles[target, i_p, i_b].plastic_strain = self.particles[source, i_p, i_b].plastic_strain
            self.particles[target, i_p, i_b].plastic_work = self.particles[source, i_p, i_b].plastic_work
        self.particles_ng[target, i_p, i_b].active = self.particles_ng[source, i_p, i_b].active

    @ti.kernel
    def copy_frame(self, source: ti.i32, target: ti.i32):
        for i_p, i_b in ti.ndrange(self._n_particles, self._B):
            self.copy_frame_helper(source, target, i_p, i_b)

    @ti.func
    def copy_grad_helper(self, source: ti.i32, target: ti.i32, i_p: ti.i32, i_b: ti.i32):
        self.particles.grad[target, i_p, i_b].pos = self.particles.grad[source, i_p, i_b].pos
        self.particles.grad[target, i_p, i_b].vel = self.particles.grad[source, i_p, i_b].vel
        self.particles.grad[target, i_p, i_b].F = self.particles.grad[source, i_p, i_b].F
        self.particles.grad[target, i_p, i_b].C = self.particles.grad[source, i_p, i_b].C
        self.particles.grad[target, i_p, i_b].Jp = self.particles.grad[source, i_p, i_b].Jp
        if ti.static(self._enable_thermal):
            self.particles.grad[target, i_p, i_b].temp = self.particles.grad[source, i_p, i_b].temp
            self.particles.grad[target, i_p, i_b].plastic_strain = self.particles.grad[source, i_p, i_b].plastic_strain
            self.particles.grad[target, i_p, i_b].plastic_work = self.particles.grad[source, i_p, i_b].plastic_work
        self.particles_ng[target, i_p, i_b].active = self.particles_ng[source, i_p, i_b].active

    @ti.kernel
    def copy_grad(self, source: ti.i32, target: ti.i32):
        for i_p, i_b in ti.ndrange(self._n_particles, self._B):
            self.copy_grad_helper(source, target, i_p, i_b)

    @ti.func
    def reset_grid_helper(self, f: ti.i32, i: ti.i32, j: ti.i32, k: ti.i32, i_b: ti.i32):
        self.grid[f, i, j, k, i_b] = self._zero_grid_cell_state

    @ti.func
    def reset_grid_grad_helper(self, f: ti.i32, i: ti.i32, j: ti.i32, k: ti.i32, i_b: ti.i32):
        self.grid.grad[f, i, j, k, i_b] = self._zero_grid_cell_state

    @ti.kernel
    def reset_grid_and_grad(self, f: ti.i32):
        # Zero out the grid at frame f for *all* grid cells and *all* batch indices
        for i, j, k, i_b in ti.ndrange(*self._grid_res, self._B):
            self.reset_grid_helper(f, i, j, k, i_b)
            self.reset_grid_grad_helper(f, i, j, k, i_b)

    @ti.kernel
    def reset_grad_till_frame(self, f: ti.i32):
        # Zero out particle grads in frames [0, f-1], for all particles, all batch indices
        for i_f, i_p, i_b in ti.ndrange(f, self._n_particles, self._B):
            self.particles.grad[i_f, i_p, i_b].pos = ti.Vector.zero(gs.ti_float, 3)
            self.particles.grad[i_f, i_p, i_b].vel = ti.Vector.zero(gs.ti_float, 3)
            self.particles.grad[i_f, i_p, i_b].C = ti.Matrix.zero(gs.ti_float, 3, 3)
            self.particles.grad[i_f, i_p, i_b].F = ti.Matrix.zero(gs.ti_float, 3, 3)
            self.particles.grad[i_f, i_p, i_b].F_tmp = ti.Matrix.zero(gs.ti_float, 3, 3)
            self.particles.grad[i_f, i_p, i_b].Jp = gs.ti_float(0.0)
            self.particles.grad[i_f, i_p, i_b].U = ti.Matrix.zero(gs.ti_float, 3, 3)
            self.particles.grad[i_f, i_p, i_b].V = ti.Matrix.zero(gs.ti_float, 3, 3)
            self.particles.grad[i_f, i_p, i_b].S = ti.Matrix.zero(gs.ti_float, 3, 3)

    # ------------------------------------------------------------------------------------
    # ------------------------------------ gradient --------------------------------------
    # ------------------------------------------------------------------------------------

    def collect_output_grads(self):
        """
        Collect gradients from downstream queried states.
        """
        for entity in self._entities:
            entity.collect_output_grads()

    def add_grad_from_state(self, state):
        if self.is_active:
            if state.pos.grad is not None:
                state.pos.assert_contiguous()
                self.add_grad_from_pos(self._sim.cur_substep_local, state.pos.grad)

            if state.vel.grad is not None:
                state.vel.assert_contiguous()
                self.add_grad_from_vel(self._sim.cur_substep_local, state.vel.grad)

            if state.C.grad is not None:
                state.C.assert_contiguous()
                self.add_grad_from_C(self._sim.cur_substep_local, state.C.grad)

            if state.F.grad is not None:
                state.F.assert_contiguous()
                self.add_grad_from_F(self._sim.cur_substep_local, state.F.grad)

            if state.Jp.grad is not None:
                state.Jp.assert_contiguous()
                self.add_grad_from_Jp(self._sim.cur_substep_local, state.Jp.grad)

    @ti.kernel
    def add_grad_from_pos(self, f: ti.i32, pos_grad: ti.types.ndarray()):
        # pos_grad shape: [B, n_particles, 3]
        for i_p, i_b in ti.ndrange(self._n_particles, self._B):
            for j in ti.static(range(3)):
                self.particles.grad[f, i_p, i_b].pos[j] += pos_grad[i_b, i_p, j]

    @ti.kernel
    def add_grad_from_vel(self, f: ti.i32, vel_grad: ti.types.ndarray()):
        # vel_grad shape: [B, n_particles, 3]
        for i_p, i_b in ti.ndrange(self._n_particles, self._B):
            for j in ti.static(range(3)):
                self.particles.grad[f, i_p, i_b].vel[j] += vel_grad[i_b, i_p, j]

    @ti.kernel
    def add_grad_from_C(self, f: ti.i32, C_grad: ti.types.ndarray()):
        # C_grad shape: [B, n_particles, 3, 3]
        for i_p, i_b in ti.ndrange(self._n_particles, self._B):
            for j in ti.static(range(3)):
                for k in ti.static(range(3)):
                    self.particles.grad[f, i_p, i_b].C[j, k] += C_grad[i_b, i_p, j, k]

    @ti.kernel
    def add_grad_from_F(self, f: ti.i32, F_grad: ti.types.ndarray()):
        # F_grad shape: [B, n_particles, 3, 3]
        for i_p, i_b in ti.ndrange(self._n_particles, self._B):
            for j in ti.static(range(3)):
                for k in ti.static(range(3)):
                    self.particles.grad[f, i_p, i_b].F[j, k] += F_grad[i_b, i_p, j, k]

    @ti.kernel
    def add_grad_from_Jp(self, f: ti.i32, Jp_grad: ti.types.ndarray()):
        # Jp_grad shape: [B, n_particles]
        for i_p, i_b in ti.ndrange(self._n_particles, self._B):
            self.particles.grad[f, i_p, i_b].Jp += Jp_grad[i_b, i_p]

    # ------------------------------------------------------------------------------------
    # --------------------------------------- io -----------------------------------------
    # ------------------------------------------------------------------------------------

    def save_ckpt(self, ckpt_name):
        if self._sim.requires_grad:
            if ckpt_name not in self._ckpt:
                self._ckpt[ckpt_name] = dict()
                self._ckpt[ckpt_name]["pos"] = torch.zeros((self._B, self._n_particles, 3), dtype=gs.tc_float)
                self._ckpt[ckpt_name]["vel"] = torch.zeros((self._B, self._n_particles, 3), dtype=gs.tc_float)
                self._ckpt[ckpt_name]["C"] = torch.zeros((self._B, self._n_particles, 3, 3), dtype=gs.tc_float)
                self._ckpt[ckpt_name]["F"] = torch.zeros((self._B, self._n_particles, 3, 3), dtype=gs.tc_float)
                self._ckpt[ckpt_name]["Jp"] = torch.zeros((self._B, self._n_particles), dtype=gs.tc_float)
                self._ckpt[ckpt_name]["active"] = torch.zeros((self._B, self._n_particles), dtype=gs.tc_bool)

            self._kernel_get_state(
                0,
                self._ckpt[ckpt_name]["pos"],
                self._ckpt[ckpt_name]["vel"],
                self._ckpt[ckpt_name]["C"],
                self._ckpt[ckpt_name]["F"],
                self._ckpt[ckpt_name]["Jp"],
                self._ckpt[ckpt_name]["active"],
            )

            for entity in self._entities:
                entity.save_ckpt(ckpt_name)

        # restart from frame 0 in memory
        self.copy_frame(self._sim.substeps_local, 0)

    def load_ckpt(self, ckpt_name):
        self.copy_frame(0, self._sim.substeps_local)
        self.copy_grad(0, self._sim.substeps_local)

        if self._sim.requires_grad:
            self.reset_grad_till_frame(self._sim.substeps_local)

            self._kernel_set_state(
                0,
                self._ckpt[ckpt_name]["pos"],
                self._ckpt[ckpt_name]["vel"],
                self._ckpt[ckpt_name]["C"],
                self._ckpt[ckpt_name]["F"],
                self._ckpt[ckpt_name]["Jp"],
                self._ckpt[ckpt_name]["active"],
            )

            for entity in self._entities:
                entity.load_ckpt(ckpt_name=ckpt_name)

    def set_state(self, f, state, envs_idx=None):
        if self.is_active:
            self._kernel_set_state(f, state.pos, state.vel, state.C, state.F, state.Jp, state.active)

    @ti.kernel
    def _kernel_set_state(
        self,
        f: ti.i32,
        pos: ti.types.ndarray(),  # shape [B, n_particles, 3]
        vel: ti.types.ndarray(),  # shape [B, n_particles, 3]
        C: ti.types.ndarray(),  # shape [B, n_particles, 3, 3]
        F: ti.types.ndarray(),  # shape [B, n_particles, 3, 3]
        Jp: ti.types.ndarray(),  # shape [B, n_particles]
        active: ti.types.ndarray(),  # shape [B, n_particles]
    ):
        for i_p, i_b in ti.ndrange(self._n_particles, self._B):
            # Write pos, vel
            for j in ti.static(range(3)):
                self.particles[f, i_p, i_b].pos[j] = pos[i_b, i_p, j]
                self.particles[f, i_p, i_b].vel[j] = vel[i_b, i_p, j]
                # Write C, F
                for k in ti.static(range(3)):
                    self.particles[f, i_p, i_b].C[j, k] = C[i_b, i_p, j, k]
                    self.particles[f, i_p, i_b].F[j, k] = F[i_b, i_p, j, k]
            # Write Jp, active
            self.particles[f, i_p, i_b].Jp = Jp[i_b, i_p]
            self.particles_ng[f, i_p, i_b].active = active[i_b, i_p]

    def get_state(self, f):
        if not self.is_active:
            return None

        state = MPMSolverState(self._scene)
        self._kernel_get_state(f, state.pos, state.vel, state.C, state.F, state.Jp, state.active)
        return state

    @ti.kernel
    def _kernel_get_state(
        self,
        f: ti.i32,
        pos: ti.types.ndarray(),  # shape [B, n_particles, 3]
        vel: ti.types.ndarray(),  # shape [B, n_particles, 3]
        C: ti.types.ndarray(),  # shape [B, n_particles, 3, 3]
        F: ti.types.ndarray(),  # shape [B, n_particles, 3, 3]
        Jp: ti.types.ndarray(),  # shape [B, n_particles]
        active: ti.types.ndarray(),  # shape [B, n_particles]
    ):
        for i_p, i_b in ti.ndrange(self._n_particles, self._B):
            for j in ti.static(range(3)):
                pos[i_b, i_p, j] = self.particles[f, i_p, i_b].pos[j]
                vel[i_b, i_p, j] = self.particles[f, i_p, i_b].vel[j]
                for k in ti.static(range(3)):
                    C[i_b, i_p, j, k] = self.particles[f, i_p, i_b].C[j, k]
                    F[i_b, i_p, j, k] = self.particles[f, i_p, i_b].F[j, k]
            Jp[i_b, i_p] = self.particles[f, i_p, i_b].Jp
            active[i_b, i_p] = ti.cast(self.particles_ng[f, i_p, i_b].active, gs.ti_bool)

    def update_render_fields(self):
        self._kernel_update_render_fields(self.sim.cur_substep_local)

    @ti.kernel
    def _kernel_update_render_fields(self, f: ti.i32):
        for i_p, i_b in ti.ndrange(self._n_particles, self._B):
            if self.particles_ng[f, i_p, i_b].active:
                self.particles_render[i_p, i_b].pos = self.particles[f, i_p, i_b].pos
                self.particles_render[i_p, i_b].vel = self.particles[f, i_p, i_b].vel
            else:
                self.particles_render[i_p, i_b].pos = gu.ti_nowhere()
            self.particles_render[i_p, i_b].active = self.particles_ng[f, i_p, i_b].active

        for i_v, i_b in ti.ndrange(self._n_vverts, self._B):
            vvert_pos = ti.Vector.zero(gs.ti_float, 3)
            for j in range(self._n_vvert_supports):
                vvert_pos += (
                    self.particles[f, self.vverts_info.support_idxs[i_v][j], i_b].pos
                    * self.vverts_info.support_weights[i_v][j]
                )
            self.vverts_render[i_v, i_b].pos = vvert_pos
            self.vverts_render[i_v, i_b].active = self.particles_render[
                self.vverts_info.support_idxs[i_v][0], i_b
            ].active

    @ti.kernel
    def _kernel_add_particles(
        self,
        f: ti.i32,
        active: ti.i32,
        particle_start: ti.i32,
        n_particles: ti.i32,
        material_idx: ti.i32,
        mat_default_Jp: ti.f32,
        mat_rho: ti.f32,
        pos: ti.types.ndarray(),  # shape [n_particles, 3]
    ):
        for i_p_ in range(n_particles):
            i_p = i_p_ + particle_start

            self.particles_info[i_p].material_idx = material_idx
            self.particles_info[i_p].default_Jp = mat_default_Jp
            self.particles_info[i_p].mass = self._particle_volume * mat_rho
            self.particles_info[i_p].free = True
            self.particles_info[i_p].muscle_group = 0
            self.particles_info[i_p].muscle_direction = ti.Vector([0.0, 0.0, 1.0], dt=gs.ti_float)

        for i_p_, i_b in ti.ndrange(n_particles, self._B):
            i_p = i_p_ + particle_start

            self.particles_ng[f, i_p, i_b].active = ti.cast(active, gs.ti_bool)
            for i in ti.static(range(3)):
                self.particles[f, i_p, i_b].pos[i] = pos[i_p_, i]

            self.particles[f, i_p, i_b].vel = ti.Vector.zero(gs.ti_float, 3)
            self.particles[f, i_p, i_b].F = ti.Matrix.identity(gs.ti_float, 3)
            self.particles[f, i_p, i_b].C = ti.Matrix.zero(gs.ti_float, 3, 3)
            self.particles[f, i_p, i_b].Jp = mat_default_Jp
            self.particles[f, i_p, i_b].actu = gs.ti_float(0.0)
            if ti.static(self._enable_thermal):
                self.particles[f, i_p, i_b].temp = self._default_initial_temperature
                self.particles[f, i_p, i_b].plastic_strain = 0.0
                self.particles[f, i_p, i_b].plastic_work = 0.0

    @ti.kernel
    def _kernel_set_particles_pos(
        self,
        f: ti.i32,
        particles_idx: ti.types.ndarray(),
        envs_idx: ti.types.ndarray(),
        poss: ti.types.ndarray(),
    ):
        for i_p_, i_b_ in ti.ndrange(particles_idx.shape[1], envs_idx.shape[0]):
            i_p = particles_idx[i_b_, i_p_]
            i_b = envs_idx[i_b_]

            for i in ti.static(range(3)):
                self.particles[f, i_p, i_b].pos[i] = poss[i_b_, i_p_, i]

            # Reset these attributes whenever overwritting particle positions manually
            self.particles[f, i_p, i_b].vel.fill(0.0)
            self.particles[f, i_p, i_b].F = ti.Matrix.identity(gs.ti_float, 3)
            self.particles[f, i_p, i_b].C.fill(0.0)
            self.particles[f, i_p, i_b].Jp = self.particles_info[i_p].default_Jp
            if ti.static(self._enable_thermal):
                self.particles[f, i_p, i_b].temp = self._default_initial_temperature
                self.particles[f, i_p, i_b].plastic_strain = 0.0
                self.particles[f, i_p, i_b].plastic_work = 0.0

    @ti.kernel
    def _kernel_set_particles_pos_grad(
        self,
        f: ti.i32,
        particle_start: ti.i32,
        n_particles: ti.i32,
        poss_grad: ti.types.ndarray(),  # shape [B, n_particles, 3]
    ):
        for i_p_, i_b in ti.ndrange(n_particles, self._B):
            i_p = i_p_ + particle_start
            for i in ti.static(range(3)):
                poss_grad[i_b, i_p_, i] = self.particles.grad[f, i_p, i_b].pos[i]

    @ti.kernel
    def _kernel_get_particles_pos(
        self,
        f: ti.i32,
        particle_start: ti.i32,
        n_particles: ti.i32,
        envs_idx: ti.types.ndarray(),
        poss: ti.types.ndarray(),
    ):
        for i_p_, i_b_ in ti.ndrange(n_particles, envs_idx.shape[0]):
            i_p = i_p_ + particle_start
            i_b = envs_idx[i_b_]
            for i in ti.static(range(3)):
                poss[i_b_, i_p_, i] = self.particles[f, i_p, i_b].pos[i]

    @ti.kernel
    def _kernel_set_particles_vel(
        self,
        f: ti.i32,
        particles_idx: ti.types.ndarray(),
        envs_idx: ti.types.ndarray(),
        vels: ti.types.ndarray(),  # shape [B, n_particles, 3]
    ):
        for i_p_, i_b_ in ti.ndrange(particles_idx.shape[1], envs_idx.shape[0]):
            i_p = particles_idx[i_b_, i_p_]
            i_b = envs_idx[i_b_]
            for i in ti.static(range(3)):
                self.particles[f, i_p, i_b].vel[i] = vels[i_b_, i_p_, i]

    @ti.kernel
    def _kernel_set_particles_vel_grad(
        self,
        f: ti.i32,
        particle_start: ti.i32,
        n_particles: ti.i32,
        vels_grad: ti.types.ndarray(),  # shape [B, n_particles, 3]
    ):
        for i_p_, i_b in ti.ndrange(n_particles, self._B):
            i_p = i_p_ + particle_start
            for i in ti.static(range(3)):
                vels_grad[i_b, i_p_, i] = self.particles.grad[f, i_p, i_b].vel[i]

    @ti.kernel
    def _kernel_get_particles_vel(
        self,
        f: ti.i32,
        particle_start: ti.i32,
        n_particles: ti.i32,
        envs_idx: ti.types.ndarray(),
        vels: ti.types.ndarray(),
    ):
        for i_p_, i_b_ in ti.ndrange(n_particles, envs_idx.shape[0]):
            i_p = i_p_ + particle_start
            i_b = envs_idx[i_b_]
            for i in ti.static(range(3)):
                vels[i_b_, i_p_, i] = self.particles[f, i_p, i_b].vel[i]

    @ti.kernel
    def _kernel_set_particles_active(
        self,
        f: ti.i32,
        particles_idx: ti.types.ndarray(),
        envs_idx: ti.types.ndarray(),
        actives: ti.types.ndarray(),  # shape [B, n_particles]
    ):
        for i_p_, i_b_ in ti.ndrange(particles_idx.shape[1], envs_idx.shape[0]):
            i_p = particles_idx[i_b_, i_p_]
            i_b = envs_idx[i_b_]
            self.particles_ng[f, i_p, i_b].active = ti.cast(actives[i_b_, i_p_], gs.ti_bool)

    @ti.kernel
    def _kernel_get_particles_active(
        self,
        f: ti.i32,
        particle_start: ti.i32,
        n_particles: ti.i32,
        envs_idx: ti.types.ndarray(),
        actives: ti.types.ndarray(),  # shape [B, n_particles]
    ):
        for i_p_, i_b_ in ti.ndrange(n_particles, envs_idx.shape[0]):
            i_p = i_p_ + particle_start
            i_b = envs_idx[i_b_]
            actives[i_b_, i_p_] = self.particles_ng[f, i_p, i_b].active

    @ti.kernel
    def _kernel_set_particles_actu(
        self,
        f: ti.i32,
        n_groups: ti.i32,
        particles_idx: ti.types.ndarray(),
        envs_idx: ti.types.ndarray(),
        actus: ti.types.ndarray(),  # shape [B, n_particles, n_groups]
    ):
        for i_p_, i_g, i_b_ in ti.ndrange(particles_idx.shape[1], n_groups, envs_idx.shape[0]):
            i_p = particles_idx[i_b_, i_p_]
            i_b = envs_idx[i_b_]
            if self.particles_info[i_p].muscle_group == i_g:
                self.particles[f, i_p, i_b].actu = actus[i_b_, i_p_, i_g]

    @ti.kernel
    def _kernel_set_particles_actu_grad(
        self,
        f: ti.i32,
        particle_start: ti.i32,
        n_particles: ti.i32,
        envs_idx: ti.types.ndarray(),
        actus_grad: ti.types.ndarray(),  # shape [B, n_particles]
    ):
        for i_p_, i_g, i_b_ in ti.ndrange(n_particles, envs_idx.shape[0]):
            i_p = i_p_ + particle_start
            i_b = envs_idx[i_b_]
            actus_grad[i_b_, i_p_] = self.particles.grad[f, i_p, i_b].actu

    @ti.kernel
    def _kernel_get_particles_actu(
        self,
        f: ti.i32,
        particle_start: ti.i32,
        n_particles: ti.i32,
        envs_idx: ti.types.ndarray(),
        actus: ti.types.ndarray(),  # shape [B, n_particles]
    ):
        for i_p_, i_b_ in ti.ndrange(n_particles, envs_idx.shape[0]):
            i_p = i_p_ + particle_start
            i_b = envs_idx[i_b_]
            actus[i_b_, i_p_] = self.particles[f, i_p, i_b].actu

    @ti.kernel
    def _kernel_set_particles_muscle_group(self, particles_idx: ti.types.ndarray(), muscle_group: ti.types.ndarray()):
        for i_p_ in range(particles_idx.shape[0]):
            i_p = particles_idx[i_p_]
            self.particles_info[i_p].muscle_group = muscle_group[i_p_]

    @ti.kernel
    def _kernel_get_particles_muscle_group(
        self, particle_start: ti.i32, n_particles: ti.i32, muscle_group: ti.types.ndarray()
    ):
        for i_p_ in range(n_particles):
            i_p = i_p_ + particle_start
            muscle_group[i_p_] = self.particles_info[i_p].muscle_group

    @ti.kernel
    def _kernel_set_particles_muscle_direction(
        self, particles_idx: ti.types.ndarray(), muscle_direction: ti.types.ndarray()
    ):
        for i_p_ in range(particles_idx.shape[0]):
            i_p = particles_idx[i_p_]
            for i in ti.static(range(3)):
                self.particles_info[i_p].muscle_direction[i] = muscle_direction[i_p_, i]

    @ti.kernel
    def _kernel_set_particles_free(self, particles_idx: ti.types.ndarray(), free: ti.types.ndarray()):
        for i_p_ in range(particles_idx.shape[0]):
            i_p = particles_idx[i_p_]
            self.particles_info[i_p].free = free[i_p_]

    @ti.kernel
    def _kernel_get_particles_free(self, particle_start: ti.i32, n_particles: ti.i32, free: ti.types.ndarray()):
        for i_p_ in range(n_particles):
            i_p = i_p_ + particle_start
            free[i_p_] = self.particles_info[i_p].free

    @ti.kernel
    def _kernel_get_mass(
        self, particle_start: ti.i32, n_particles: ti.i32, mass: ti.types.ndarray(), envs_idx: ti.types.ndarray()
    ):
        total_mass = gs.ti_float(0.0)
        for i_p_ in range(n_particles):
            i_p = i_p_ + particle_start
            total_mass += self.particles_info[i_p].mass
        total_mass = total_mass / self._particle_volume_scale
        for i_b_ in range(envs_idx.shape[0]):
            mass[i_b_] = total_mass

    # ------------------------------------------------------------------------------------
    # -------------------------------- particle constraints ------------------------------
    # ------------------------------------------------------------------------------------

    @ti.kernel
    def _kernel_set_particle_constraints(
        self,
        f: ti.i32,
        particles_mask: ti.types.ndarray(),  # shape [n_envs, n_particles] boolean mask
        particle_start: ti.i32,
        stiffness: ti.f32,
        link_idx: ti.i32,
        link_pos: ti.types.ndarray(),  # shape [n_envs, 3]
        link_quat: ti.types.ndarray(),  # shape [n_envs, 4]
    ):
        for i_p_local, i_b in ti.ndrange(particles_mask.shape[1], particles_mask.shape[0]):
            if particles_mask[i_b, i_p_local]:
                i_p = i_p_local + particle_start

                # Get current particle position
                pos = self.particles[f, i_p, i_b].pos

                # Get link transform
                l_pos = ti.Vector([link_pos[i_b, 0], link_pos[i_b, 1], link_pos[i_b, 2]], dt=gs.ti_float)
                l_quat = ti.Vector(
                    [link_quat[i_b, 0], link_quat[i_b, 1], link_quat[i_b, 2], link_quat[i_b, 3]], dt=gs.ti_float
                )

                # Compute offset in link's local frame
                local_pos = gu.ti_inv_transform_by_trans_quat(pos, l_pos, l_quat)

                # Store constraint info
                self.particle_constraints[i_p, i_b].is_constrained = True
                self.particle_constraints[i_p, i_b].target_pos = pos  # initial target is current position
                self.particle_constraints[i_p, i_b].stiffness = stiffness
                self.particle_constraints[i_p, i_b].link_idx = link_idx
                self.particle_constraints[i_p, i_b].link_local_pos = local_pos

    @ti.kernel
    def _kernel_remove_particle_constraints(
        self,
        particles_mask: ti.types.ndarray(),  # shape [n_envs, n_particles] boolean mask
        particle_start: ti.i32,
    ):
        for i_p_local, i_b in ti.ndrange(particles_mask.shape[1], particles_mask.shape[0]):
            if particles_mask[i_b, i_p_local]:
                i_p = i_p_local + particle_start
                self.particle_constraints[i_p, i_b].is_constrained = False
                self.particle_constraints[i_p, i_b].link_idx = -1

    @ti.kernel
    def apply_particle_constraints(
        self,
        f: ti.i32,
        links_state: array_class.LinksState,
    ):
        for i_p, i_b in ti.ndrange(self._n_particles, self._B):
            if self.particle_constraints[i_p, i_b].is_constrained:
                # Update target position from link pose
                i_l = self.particle_constraints[i_p, i_b].link_idx
                if i_l >= 0:
                    link_pos = links_state.pos[i_l, i_b]
                    link_quat = links_state.quat[i_l, i_b]
                    local_pos = self.particle_constraints[i_p, i_b].link_local_pos
                    target = gu.ti_transform_by_trans_quat(local_pos, link_pos, link_quat)
                    self.particle_constraints[i_p, i_b].target_pos = target

                # Apply spring force to velocity
                target_pos = self.particle_constraints[i_p, i_b].target_pos
                stiffness = self.particle_constraints[i_p, i_b].stiffness
                mass = self.particles_info[i_p].mass / self._particle_volume_scale

                pos = self.particles[f + 1, i_p, i_b].pos
                vel = self.particles[f + 1, i_p, i_b].vel

                pos_error = pos - target_pos
                spring_force = -stiffness * pos_error
                damping_force = -2.0 * ti.math.sqrt(stiffness * mass) * vel

                dv = self.substep_dt * (spring_force + damping_force) / mass
                self.particles[f + 1, i_p, i_b].vel = vel + dv

    # ------------------------------------------------------------------------------------
    # ----------------------------------- properties -------------------------------------
    # ------------------------------------------------------------------------------------

    @property
    def n_particles(self):
        if self.is_built:
            return self._n_particles
        return sum(entity.n_particles for entity in self._entities)

    @property
    def n_vverts(self):
        if self.is_built:
            return self._n_vverts
        return sum(entity.n_vverts for entity in self._entities)

    @property
    def n_vfaces(self):
        if self.is_built:
            return self._n_vfaces
        return sum(entity.n_vfaces for entity in self._entities)

    @property
    def grid_density(self):
        return self._grid_density

    @property
    def particle_size(self):
        return self._particle_size

    @property
    def particle_radius(self):
        return self._particle_size / 2.0

    @property
    def upper_bound(self):
        return self._upper_bound

    @property
    def lower_bound(self):
        return self._lower_bound

    @property
    def leaf_block_size(self):
        raise DeprecationError("This property has been removed.")

    @property
    def use_sparse_grid(self):
        return DeprecationError("This property has been removed.")

    @property
    def dx(self):
        return self._dx

    @property
    def inv_dx(self):
        return self._inv_dx

    @property
    def particle_volume_real(self):
        return self._particle_volume_real

    @property
    def particle_volume(self):
        return self._particle_volume

    @property
    def particle_volume_scale(self):
        return self._particle_volume_scale

    @property
    def is_built(self):
        return self._scene._is_built

    @property
    def lower_bound_cell(self):
        return self._lower_bound_cell

    @property
    def upper_bound_cell(self):
        return self._upper_bound_cell

    @property
    def grid_res(self):
        return self._grid_res

    @property
    def grid_offset(self):
        return self._grid_offset

    @property
    def enable_CPIC(self):
        return self._enable_CPIC

    @property
    def enable_particle_constraints(self):
        return self._enable_particle_constraints


@ti.func
def signmax(a, eps):
    sign = ti.select(a >= 0, 1.0, -1.0)
    return sign * ti.max(ti.abs(a), eps)


@ti.func
def backward_svd(grad_U, grad_S, grad_V, U, S, V):
    # https://github.com/pytorch/pytorch/blob/ab0a04dc9c8b84d4a03412f1c21a6c4a2cefd36c/tools/autograd/templates/Functions.cpp
    vt = V.transpose()
    ut = U.transpose()
    S_term = U @ grad_S @ vt

    s = ti.Vector.zero(gs.ti_float, 3)
    s = ti.Vector([S[0, 0], S[1, 1], S[2, 2]]) ** 2
    F = ti.Matrix.zero(gs.ti_float, 3, 3)
    for i, j in ti.static(ti.ndrange(3, 3)):
        if i == j:
            F[i, j] = 0.0
        else:
            F[i, j] = 1.0 / signmax(s[j] - s[i], 1e-6)
    u_term = U @ ((F * (ut @ grad_U - grad_U.transpose() @ U)) @ S) @ vt
    v_term = U @ (S @ ((F * (vt @ grad_V - grad_V.transpose() @ V)) @ vt))
    return u_term + v_term + S_term
from typing import TYPE_CHECKING

import numpy as np
import gstaichi as ti

import genesis as gs
import genesis.utils.sdf as sdf

from genesis.options.solvers import LegacyCouplerOptions
from genesis.repr_base import RBC
from genesis.utils import array_class
from genesis.utils.array_class import LinksState
from genesis.utils.geom import ti_inv_transform_by_trans_quat, ti_transform_by_trans_quat

if TYPE_CHECKING:
    from genesis.engine.simulator import Simulator

CLAMPED_INV_DT = 50.0


@ti.data_oriented
class LegacyCoupler(RBC):
    """
    This class handles all the coupling between different solvers. LegacyCoupler will be deprecated in the future.
    """

    # ------------------------------------------------------------------------------------
    # --------------------------------- Initialization -----------------------------------
    # ------------------------------------------------------------------------------------

    def __init__(
        self,
        simulator: "Simulator",
        options: "LegacyCouplerOptions",
    ) -> None:
        self.sim = simulator
        self.options = options

        self.tool_solver = self.sim.tool_solver
        self.rigid_solver = self.sim.rigid_solver
        self.mpm_solver = self.sim.mpm_solver
        self.sph_solver = self.sim.sph_solver
        self.pbd_solver = self.sim.pbd_solver
        self.fem_solver = self.sim.fem_solver
        self.sf_solver = self.sim.sf_solver

    def build(self) -> None:
        self._rigid_mpm = self.rigid_solver.is_active and self.mpm_solver.is_active and self.options.rigid_mpm
        self._rigid_sph = self.rigid_solver.is_active and self.sph_solver.is_active and self.options.rigid_sph
        self._rigid_pbd = self.rigid_solver.is_active and self.pbd_solver.is_active and self.options.rigid_pbd
        self._rigid_fem = self.rigid_solver.is_active and self.fem_solver.is_active and self.options.rigid_fem
        self._mpm_sph = self.mpm_solver.is_active and self.sph_solver.is_active and self.options.mpm_sph
        self._mpm_pbd = self.mpm_solver.is_active and self.pbd_solver.is_active and self.options.mpm_pbd
        self._fem_mpm = self.fem_solver.is_active and self.mpm_solver.is_active and self.options.fem_mpm
        self._fem_sph = self.fem_solver.is_active and self.sph_solver.is_active and self.options.fem_sph

        if self._rigid_mpm and self.mpm_solver.enable_CPIC:
            # this field stores the geom index of the thin shell rigid object (if any) that separates particle and its surrounding grid cell
            self.cpic_flag = ti.field(gs.ti_int, shape=(self.mpm_solver.n_particles, 3, 3, 3, self.mpm_solver._B))
            self.mpm_rigid_normal = ti.Vector.field(
                3,
                dtype=gs.ti_float,
                shape=(self.mpm_solver.n_particles, self.rigid_solver.n_geoms_, self.mpm_solver._B),
            )

        if self._rigid_sph:
            self.sph_rigid_normal = ti.Vector.field(
                3,
                dtype=gs.ti_float,
                shape=(self.sph_solver.n_particles, self.rigid_solver.n_geoms_, self.sph_solver._B),
            )
            self.sph_rigid_normal_reordered = ti.Vector.field(
                3,
                dtype=gs.ti_float,
                shape=(self.sph_solver.n_particles, self.rigid_solver.n_geoms_, self.sph_solver._B),
            )

        if self._rigid_pbd:
            self.pbd_rigid_normal_reordered = ti.Vector.field(
                3, dtype=gs.ti_float, shape=(self.pbd_solver.n_particles, self.pbd_solver._B, self.rigid_solver.n_geoms)
            )

            struct_particle_attach_info = ti.types.struct(
                link_idx=gs.ti_int,
                local_pos=gs.ti_vec3,
            )

            self.particle_attach_info = struct_particle_attach_info.field(
                shape=(self.pbd_solver._n_particles, self.pbd_solver._B), layout=ti.Layout.SOA
            )
            self.particle_attach_info.link_idx.fill(-1)
            self.particle_attach_info.local_pos.fill(0.0)

        if self._mpm_sph:
            self.mpm_sph_stencil_size = int(np.floor(self.mpm_solver.dx / self.sph_solver.hash_grid_cell_size) + 2)

        if self._mpm_pbd:
            self.mpm_pbd_stencil_size = int(np.floor(self.mpm_solver.dx / self.pbd_solver.hash_grid_cell_size) + 2)

        ## DEBUG
        self._dx = 1 / 1024
        self._stencil_size = int(np.floor(self._dx / self.sph_solver.hash_grid_cell_size) + 2)

        if self._rigid_mpm: # Always initialize if rigid_mpm is active, not just for CPIC
             self.link_coupling_forces = ti.Vector.field(
                3,
                dtype=gs.ti_float,
                shape=(self.rigid_solver.n_links_, self.sim._B),
            )

        self.reset(envs_idx=self.sim.scene._envs_idx)

    def reset(self, envs_idx=None) -> None:
        if self._rigid_mpm and self.mpm_solver.enable_CPIC:
            if envs_idx is None:
                self.mpm_rigid_normal.fill(0)
            else:
                self._kernel_reset_mpm(envs_idx)
        
        if self._rigid_mpm:
             if envs_idx is None:
                 self.link_coupling_forces.fill(0)
             else:
                 self._kernel_reset_link_coupling_forces(envs_idx)

        if self._rigid_sph:
            if envs_idx is None:
                self.sph_rigid_normal.fill(0)
            else:
                self._kernel_reset_sph(envs_idx)

    @ti.kernel
    def _kernel_reset_mpm(self, envs_idx: ti.types.ndarray()):
        for i_p, i_g, i_b_ in ti.ndrange(self.mpm_solver.n_particles, self.rigid_solver.n_geoms, envs_idx.shape[0]):
            self.mpm_rigid_normal[i_p, i_g, envs_idx[i_b_]] = 0.0

    @ti.kernel
    def _kernel_reset_sph(self, envs_idx: ti.types.ndarray()):
        for i_p, i_g, i_b_ in ti.ndrange(self.sph_solver.n_particles, self.rigid_solver.n_geoms, envs_idx.shape[0]):
            self.sph_rigid_normal[i_p, i_g, envs_idx[i_b_]] = 0.0

    @ti.kernel
    def _kernel_reset_link_coupling_forces(self, envs_idx: ti.types.ndarray()):
        for i_l, i_b_ in ti.ndrange(self.rigid_solver.n_links_, envs_idx.shape[0]):
            self.link_coupling_forces[i_l, envs_idx[i_b_]].fill(0)

    def get_link_coupling_forces(self, link_idx=None, envs_idx=None):
        if not hasattr(self, 'link_coupling_forces'):
            return None
            
        forces = self.link_coupling_forces.to_numpy()  # Shape: (n_links_, n_envs, 3)
        
        if link_idx is not None:
            forces = forces[link_idx]
            if envs_idx is not None:
                forces = forces[envs_idx]
        elif envs_idx is not None:
            forces = forces[:, envs_idx, :]
            
        return forces
    
    def clear_link_coupling_forces(self):
        if hasattr(self, 'link_coupling_forces'):
            self.link_coupling_forces.fill(0)

    @ti.func
    def _func_collide_with_rigid(
        self,
        f,
        pos_world,
        vel,
        mass,
        i_b,
        geoms_state: array_class.GeomsState,
        geoms_info: array_class.GeomsInfo,
        links_state: array_class.LinksState,
        rigid_global_info: array_class.RigidGlobalInfo,
        sdf_info: array_class.SDFInfo,
        collider_static_config: ti.template(),
    ):
        for i_g in range(self.rigid_solver.n_geoms):
            if geoms_info.needs_coup[i_g]:
                vel = self._func_collide_with_rigid_geom(
                    pos_world,
                    vel,
                    mass,
                    i_g,
                    i_b,
                    geoms_state=geoms_state,
                    geoms_info=geoms_info,
                    links_state=links_state,
                    rigid_global_info=rigid_global_info,
                    sdf_info=sdf_info,
                    collider_static_config=collider_static_config,
                )
        return vel

    @ti.func
    def _func_collide_with_rigid_geom(
        self,
        pos_world,
        vel,
        mass,
        geom_idx,
        batch_idx,
        geoms_state: array_class.GeomsState,
        geoms_info: array_class.GeomsInfo,
        links_state: array_class.LinksState,
        rigid_global_info: array_class.RigidGlobalInfo,
        sdf_info: array_class.SDFInfo,
        collider_static_config: ti.template(),
    ):
        signed_dist = sdf.sdf_func_world(
            geoms_state=geoms_state,
            geoms_info=geoms_info,
            sdf_info=sdf_info,
            pos_world=pos_world,
            geom_idx=geom_idx,
            batch_idx=batch_idx,
        )

        # bigger coup_softness implies that the coupling influence extends further away from the object.
        influence = ti.min(ti.exp(-signed_dist / max(1e-10, geoms_info.coup_softness[geom_idx])), 1)

        if influence > 0.1:
            normal_rigid = sdf.sdf_func_normal_world(
                geoms_state=geoms_state,
                geoms_info=geoms_info,
                rigid_global_info=rigid_global_info,
                collider_static_config=collider_static_config,
                sdf_info=sdf_info,
                pos_world=pos_world,
                geom_idx=geom_idx,
                batch_idx=batch_idx,
            )
            vel = self._func_collide_in_rigid_geom(
                pos_world,
                vel,
                mass,
                normal_rigid,
                influence,
                geom_idx,
                batch_idx,
                geoms_info,
                links_state,
                rigid_global_info,
            )

        return vel

    @ti.func
    def _func_collide_with_rigid_geom_robust(
        self,
        pos_world,
        vel,
        mass,
        normal_prev,
        geom_idx,
        batch_idx,
        geoms_state: array_class.GeomsState,
        geoms_info: array_class.GeomsInfo,
        links_state: array_class.LinksState,
        rigid_global_info: array_class.RigidGlobalInfo,
        sdf_info: array_class.SDFInfo,
        collider_static_config: ti.template(),
    ):
        """
        Similar to _func_collide_with_rigid_geom, but additionally handles potential side flip due to penetration.
        """
        signed_dist = sdf.sdf_func_world(
            geoms_state=geoms_state,
            geoms_info=geoms_info,
            sdf_info=sdf_info,
            pos_world=pos_world,
            geom_idx=geom_idx,
            batch_idx=batch_idx,
        )
        normal_rigid = sdf.sdf_func_normal_world(
            geoms_state=geoms_state,
            geoms_info=geoms_info,
            rigid_global_info=rigid_global_info,
            collider_static_config=collider_static_config,
            sdf_info=sdf_info,
            pos_world=pos_world,
            geom_idx=geom_idx,
            batch_idx=batch_idx,
        )

        # bigger coup_softness implies that the coupling influence extends further away from the object.
        influence = ti.min(ti.exp(-signed_dist / max(1e-10, geoms_info.coup_softness[geom_idx])), 1)

        # if normal_rigid.dot(normal_prev) < 0: # side flip due to penetration
        #     influence = 1.0
        #     normal_rigid = normal_prev
        if influence > 0.1:
            vel = self._func_collide_in_rigid_geom(
                pos_world,
                vel,
                mass,
                normal_rigid,
                influence,
                geom_idx,
                batch_idx,
                geoms_info,
                links_state,
                rigid_global_info,
            )

        # attraction force
        # if 0.001 < signed_dist < 0.01:
        #     vel = vel - normal_rigid * 0.1 * signed_dist

        return vel, normal_rigid

    @ti.func
    def _func_collide_in_rigid_geom(
        self,
        pos_world,
        vel,
        mass,
        normal_rigid,
        influence,
        geom_idx,
        i_b,
        geoms_info: array_class.GeomsInfo,
        links_state: array_class.LinksState,
        rigid_global_info: array_class.RigidGlobalInfo,
    ):
        """
        Resolves collision when a particle is already in collision with a rigid object.
        This function assumes known normal_rigid and influence.
        """
        vel_rigid = self.rigid_solver._func_vel_at_point(
            pos_world=pos_world,
            link_idx=geoms_info.link_idx[geom_idx],
            i_b=i_b,
            links_state=links_state,
        )

        # v w.r.t rigid
        rvel = vel - vel_rigid
        rvel_normal_magnitude = rvel.dot(normal_rigid)  # negative if inward

        if rvel_normal_magnitude < 0:  # colliding
            #################### rigid -> particle ####################
            # tangential component
            rvel_tan = rvel - rvel_normal_magnitude * normal_rigid
            rvel_tan_norm = rvel_tan.norm(gs.EPS)

            # tangential component after friction
            rvel_tan = (
                rvel_tan
                / rvel_tan_norm
                * ti.max(0, rvel_tan_norm + rvel_normal_magnitude * geoms_info.coup_friction[geom_idx])
            )

            # normal component after collision
            rvel_normal = -normal_rigid * rvel_normal_magnitude * geoms_info.coup_restitution[geom_idx]

            # normal + tangential component
            rvel_new = rvel_tan + rvel_normal

            # apply influence
            vel_old = vel
            vel = vel_rigid + rvel_new * influence + rvel * (1 - influence)

            #################### particle -> rigid ####################
            # Compute delta momentum and apply to rigid body.
            delta_mv = mass * (vel - vel_old)
            force = -delta_mv / rigid_global_info.substep_dt[None]
            self.rigid_solver._func_apply_coupling_force(
                pos_world,
                force,
                geoms_info.link_idx[geom_idx],
                i_b,
                links_state,
            )

            # Store the coupling force for contact detection
            if ti.static(hasattr(self, 'link_coupling_forces')):
                ti.atomic_add(self.link_coupling_forces[geoms_info.link_idx[geom_idx], i_b], force)

        return vel

    @ti.func
    def _func_mpm_tool(self, f, pos_world, vel, i_b):
        for entity in ti.static(self.tool_solver.entities):
            if ti.static(entity.material.collision):
                vel = entity.collide(f, pos_world, vel, i_b)
        return vel

    @ti.kernel
    def mpm_grid_op(
        self,
        f: ti.i32,
        t: ti.f32,
        geoms_state: array_class.GeomsState,
        geoms_info: array_class.GeomsInfo,
        links_state: array_class.LinksState,
        rigid_global_info: array_class.RigidGlobalInfo,
        sdf_info: array_class.SDFInfo,
        collider_static_config: ti.template(),
    ):
        for ii, jj, kk, i_b in ti.ndrange(*self.mpm_solver.grid_res, self.mpm_solver._B):
            I = (ii, jj, kk)
            if self.mpm_solver.grid[f, I, i_b].mass > gs.EPS:
                #################### MPM grid op ####################
                # Momentum to velocity
                vel_mpm = (1 / self.mpm_solver.grid[f, I, i_b].mass) * self.mpm_solver.grid[f, I, i_b].vel_in

                # Thermal: normalize mass-weighted temperature
                if ti.static(self.mpm_solver._enable_thermal):
                    if self.mpm_solver.grid[f, I, i_b].mass_thermal > 0:
                        self.mpm_solver.grid[f, I, i_b].temp = (
                            self.mpm_solver.grid[f, I, i_b].temp
                            / self.mpm_solver.grid[f, I, i_b].mass_thermal
                        )
                        
                        # --- Air Cooling (Convection) ---
                        is_surface = 0
                        I_left = I + ti.Vector([-1, 0, 0])
                        I_right = I + ti.Vector([1, 0, 0])
                        I_down = I + ti.Vector([0, -1, 0])
                        I_up = I + ti.Vector([0, 1, 0])
                        I_back = I + ti.Vector([0, 0, -1])
                        I_front = I + ti.Vector([0, 0, 1])
                        
                        if I_left[0] < 0:
                            is_surface = 1
                        elif self.mpm_solver.grid[f, I_left, i_b].mass_thermal < gs.EPS:
                            is_surface = 1

                        if I_right[0] >= self.mpm_solver.grid_res[0]:
                            is_surface = 1
                        elif self.mpm_solver.grid[f, I_right, i_b].mass_thermal < gs.EPS:
                            is_surface = 1

                        if I_down[1] < 0:
                            is_surface = 1
                        elif self.mpm_solver.grid[f, I_down, i_b].mass_thermal < gs.EPS:
                            is_surface = 1

                        if I_up[1] >= self.mpm_solver.grid_res[1]:
                            is_surface = 1
                        elif self.mpm_solver.grid[f, I_up, i_b].mass_thermal < gs.EPS:
                            is_surface = 1

                        if I_back[2] < 0:
                            is_surface = 1
                        elif self.mpm_solver.grid[f, I_back, i_b].mass_thermal < gs.EPS:
                            is_surface = 1

                        if I_front[2] >= self.mpm_solver.grid_res[2]:
                            is_surface = 1
                        elif self.mpm_solver.grid[f, I_front, i_b].mass_thermal < gs.EPS:
                            is_surface = 1
                        
                        if is_surface == 1:
                            h_air = self.mpm_solver._h_air
                            # k = h * A / (M * Cp) -> approx A = dx^2 for a cell face
                            k_air = (h_air * (self.mpm_solver.dx ** 2)) / (self.mpm_solver.grid[f, I, i_b].mass_thermal * self.mpm_solver._default_heat_capacity)
                            decay_air = ti.exp(-k_air * self.mpm_solver.substep_dt)
                            T_air = 293.15 # Room temp
                            self.mpm_solver.grid[f, I, i_b].temp = T_air + (self.mpm_solver.grid[f, I, i_b].temp - T_air) * decay_air
                        
                        # --- Contact Cooling (Conduction with Rigid Bodies) ---
                        if ti.static(self.rigid_solver.is_active):
                            pos_world = (I + self.mpm_solver.grid_offset) * self.mpm_solver.dx
                            for i_g in range(self.rigid_solver.n_geoms):
                                if geoms_info.needs_coup[i_g]:
                                    signed_dist = sdf.sdf_func_world(
                                        geoms_state=geoms_state,
                                        geoms_info=geoms_info,
                                        sdf_info=sdf_info,
                                        pos_world=pos_world,
                                        geom_idx=i_g,
                                        batch_idx=i_b,
                                    )
                                    # If within 1 grid cell of the surface, apply contact cooling (overrides air)
                                    if signed_dist < self.mpm_solver.dx:
                                        h_contact = self.mpm_solver._h_contact
                                        k_contact = (h_contact * (self.mpm_solver.dx ** 2)) / (self.mpm_solver.grid[f, I, i_b].mass_thermal * self.mpm_solver._default_heat_capacity)
                                        decay_contact = ti.exp(-k_contact * self.mpm_solver.substep_dt)
                                        T_rigid = 293.15 # Rigid bodies are assumed infinite heat sinks
                                        self.mpm_solver.grid[f, I, i_b].temp = T_rigid + (self.mpm_solver.grid[f, I, i_b].temp - T_rigid) * decay_contact

                # gravity
                vel_mpm += self.mpm_solver.substep_dt * self.mpm_solver._gravity[i_b]

                pos = (I + self.mpm_solver.grid_offset) * self.mpm_solver.dx
                mass_mpm = self.mpm_solver.grid[f, I, i_b].mass / self.mpm_solver._particle_volume_scale

                # external force fields
                for i_ff in ti.static(range(len(self.mpm_solver._ffs))):
                    vel_mpm += self.mpm_solver._ffs[i_ff].get_acc(pos, vel_mpm, t, -1) * self.mpm_solver.substep_dt

                #################### MPM <-> Tool ####################
                if ti.static(self.tool_solver.is_active):
                    vel_mpm = self._func_mpm_tool(f, pos, vel_mpm, i_b)

                #################### MPM <-> Rigid ####################
                vel_mpm = self._func_collide_with_rigid(
                    f,
                    pos,
                    vel_mpm,
                    mass_mpm,
                    i_b,
                    geoms_state=geoms_state,
                    geoms_info=geoms_info,
                    links_state=links_state,
                    rigid_global_info=rigid_global_info,
                    sdf_info=sdf_info,
                    collider_static_config=collider_static_config,
                )

                #################### MPM <-> SPH ####################
                if ti.static(self._mpm_sph):
                    # using the lower corner of MPM cell to find the corresponding SPH base cell
                    base = self.sph_solver.sh.pos_to_grid(pos - 0.5 * self.mpm_solver.dx)

                    # ---------- SPH -> MPM ----------
                    sph_vel = ti.Vector([0.0, 0.0, 0.0])
                    colliding_particles = 0
                    for offset in ti.grouped(
                        ti.ndrange(self.mpm_sph_stencil_size, self.mpm_sph_stencil_size, self.mpm_sph_stencil_size)
                    ):
                        slot_idx = self.sph_solver.sh.grid_to_slot(base + offset)
                        for i in range(
                            self.sph_solver.sh.slot_start[slot_idx, i_b],
                            self.sph_solver.sh.slot_start[slot_idx, i_b] + self.sph_solver.sh.slot_size[slot_idx, i_b],
                        ):
                            if (
                                ti.abs(pos - self.sph_solver.particles_reordered.pos[i, i_b]).max()
                                < self.mpm_solver.dx * 0.5
                            ):
                                sph_vel += self.sph_solver.particles_reordered.vel[i, i_b]
                                colliding_particles += 1
                    if colliding_particles > 0:
                        vel_old = vel_mpm
                        vel_mpm = sph_vel / colliding_particles

                        # ---------- MPM -> SPH ----------
                        delta_mv = mass_mpm * (vel_mpm - vel_old)

                        for offset in ti.grouped(
                            ti.ndrange(self.mpm_sph_stencil_size, self.mpm_sph_stencil_size, self.mpm_sph_stencil_size)
                        ):
                            slot_idx = self.sph_solver.sh.grid_to_slot(base + offset)
                            for i in range(
                                self.sph_solver.sh.slot_start[slot_idx, i_b],
                                self.sph_solver.sh.slot_start[slot_idx, i_b]
                                + self.sph_solver.sh.slot_size[slot_idx, i_b],
                            ):
                                if (
                                    ti.abs(pos - self.sph_solver.particles_reordered.pos[i, i_b]).max()
                                    < self.mpm_solver.dx * 0.5
                                ):
                                    self.sph_solver.particles_reordered[i, i_b].vel = (
                                        self.sph_solver.particles_reordered[i, i_b].vel
                                        - delta_mv / self.sph_solver.particles_info_reordered[i, i_b].mass
                                    )

                #################### MPM <-> PBD ####################
                if ti.static(self._mpm_pbd):
                    # using the lower corner of MPM cell to find the corresponding PBD base cell
                    base = self.pbd_solver.sh.pos_to_grid(pos - 0.5 * self.mpm_solver.dx)

                    # ---------- PBD -> MPM ----------
                    pbd_vel = ti.Vector([0.0, 0.0, 0.0])
                    colliding_particles = 0
                    for offset in ti.grouped(
                        ti.ndrange(self.mpm_pbd_stencil_size, self.mpm_pbd_stencil_size, self.mpm_pbd_stencil_size)
                    ):
                        slot_idx = self.pbd_solver.sh.grid_to_slot(base + offset)
                        for i in range(
                            self.pbd_solver.sh.slot_start[slot_idx, i_b],
                            self.pbd_solver.sh.slot_start[slot_idx, i_b] + self.pbd_solver.sh.slot_size[slot_idx, i_b],
                        ):
                            if (
                                ti.abs(pos - self.pbd_solver.particles_reordered.pos[i, i_b]).max()
                                < self.mpm_solver.dx * 0.5
                            ):
                                pbd_vel += self.pbd_solver.particles_reordered.vel[i, i_b]
                                colliding_particles += 1
                    if colliding_particles > 0:
                        vel_old = vel_mpm
                        vel_mpm = pbd_vel / colliding_particles

                        # ---------- MPM -> PBD ----------
                        delta_mv = mass_mpm * (vel_mpm - vel_old)

                        for offset in ti.grouped(
                            ti.ndrange(self.mpm_pbd_stencil_size, self.mpm_pbd_stencil_size, self.mpm_pbd_stencil_size)
                        ):
                            slot_idx = self.pbd_solver.sh.grid_to_slot(base + offset)
                            for i in range(
                                self.pbd_solver.sh.slot_start[slot_idx, i_b],
                                self.pbd_solver.sh.slot_start[slot_idx, i_b]
                                + self.pbd_solver.sh.slot_size[slot_idx, i_b],
                            ):
                                if (
                                    ti.abs(pos - self.pbd_solver.particles_reordered.pos[i, i_b]).max()
                                    < self.mpm_solver.dx * 0.5
                                ):
                                    if self.pbd_solver.particles_reordered[i, i_b].free:
                                        self.pbd_solver.particles_reordered[i, i_b].vel = (
                                            self.pbd_solver.particles_reordered[i, i_b].vel
                                            - delta_mv / self.pbd_solver.particles_info_reordered[i, i_b].mass
                                        )

                #################### MPM boundary ####################
                _, self.mpm_solver.grid[f, I, i_b].vel_out = self.mpm_solver.boundary.impose_pos_vel(pos, vel_mpm)

    @ti.kernel
    def mpm_surface_to_particle(
        self,
        f: ti.i32,
        geoms_state: array_class.GeomsState,
        geoms_info: array_class.GeomsInfo,
        sdf_info: array_class.SDFInfo,
        rigid_global_info: array_class.RigidGlobalInfo,
        collider_static_config: ti.template(),
    ):
        for i_p, i_b in ti.ndrange(self.mpm_solver.n_particles, self.mpm_solver._B):
            if self.mpm_solver.particles_ng[f, i_p, i_b].active:
                for i_g in range(self.rigid_solver.n_geoms):
                    if geoms_info.needs_coup[i_g]:
                        sdf_normal = sdf.sdf_func_normal_world(
                            geoms_state=geoms_state,
                            geoms_info=geoms_info,
                            rigid_global_info=rigid_global_info,
                            collider_static_config=collider_static_config,
                            sdf_info=sdf_info,
                            pos_world=self.mpm_solver.particles[f, i_p, i_b].pos,
                            geom_idx=i_g,
                            batch_idx=i_b,
                        )
                        # we only update the normal if the particle does not the object
                        if sdf_normal.dot(self.mpm_rigid_normal[i_p, i_g, i_b]) >= 0:
                            self.mpm_rigid_normal[i_p, i_g, i_b] = sdf_normal

    def fem_rigid_link_constraints(self):
        if self.fem_solver._constraints_initialized and self.rigid_solver.is_active:
            self.fem_solver._kernel_update_linked_vertex_constraints(self.rigid_solver.links_state)

    @ti.kernel
    def fem_surface_force(
        self,
        f: ti.i32,
        geoms_state: array_class.GeomsState,
        geoms_info: array_class.GeomsInfo,
        links_state: array_class.LinksState,
        rigid_global_info: array_class.RigidGlobalInfo,
        sdf_info: array_class.SDFInfo,
        collider_static_config: ti.template(),
    ):
        # TODO: all collisions are on vertices instead of surface and edge
        for i_s, i_b in ti.ndrange(self.fem_solver.n_surfaces, self.fem_solver._B):
            if self.fem_solver.surface[i_s].active:
                dt = self.fem_solver.substep_dt
                iel = self.fem_solver.surface[i_s].tri2el
                mass = self.fem_solver.elements_i[iel].mass_scaled / self.fem_solver.vol_scale

                p1 = self.fem_solver.elements_v[f, self.fem_solver.surface[i_s].tri2v[0], i_b].pos
                p2 = self.fem_solver.elements_v[f, self.fem_solver.surface[i_s].tri2v[1], i_b].pos
                p3 = self.fem_solver.elements_v[f, self.fem_solver.surface[i_s].tri2v[2], i_b].pos
                u = p2 - p1
                v = p3 - p1
                surface_normal = ti.math.cross(u, v)
                surface_normal = surface_normal / surface_normal.norm(gs.EPS)

                # FEM <-> Rigid
                if ti.static(self._rigid_fem):
                    # NOTE: collision only on surface vertices
                    for j in ti.static(range(3)):
                        iv = self.fem_solver.surface[i_s].tri2v[j]
                        vel_fem_sv = self._func_collide_with_rigid(
                            f,
                            self.fem_solver.elements_v[f, iv, i_b].pos,
                            self.fem_solver.elements_v[f + 1, iv, i_b].vel,
                            mass / 3.0,  # assume element mass uniformly distributed to vertices
                            i_b,
                            geoms_state,
                            geoms_info,
                            links_state,
                            rigid_global_info,
                            sdf_info,
                            collider_static_config,
                        )
                        self.fem_solver.elements_v[f + 1, iv, i_b].vel = vel_fem_sv

                # FEM <-> MPM (interact with MPM grid instead of particles)
                # NOTE: not doing this in mpm_grid_op otherwise we need to search for fem surface for each particles
                #       however, this function is called after mpm boundary conditions.
                if ti.static(self._fem_mpm):
                    for j in ti.static(range(3)):
                        iv = self.fem_solver.surface[i_s].tri2v[j]
                        pos = self.fem_solver.elements_v[f, iv, i_b].pos
                        vel_fem_sv = self.fem_solver.elements_v[f + 1, iv, i_b].vel
                        mass_fem_sv = mass / 4.0  # assume element mass uniformly distributed

                        # follow MPM p2g scheme
                        vel_mpm = ti.Vector([0.0, 0.0, 0.0])
                        mass_mpm = 0.0
                        mpm_base = ti.floor(pos * self.mpm_solver.inv_dx - 0.5).cast(gs.ti_int)
                        mpm_fx = pos * self.mpm_solver.inv_dx - mpm_base.cast(gs.ti_float)
                        mpm_w = [0.5 * (1.5 - mpm_fx) ** 2, 0.75 - (mpm_fx - 1.0) ** 2, 0.5 * (mpm_fx - 0.5) ** 2]
                        new_vel_fem_sv = vel_fem_sv
                        for mpm_offset in ti.static(ti.grouped(self.mpm_solver.stencil_range())):
                            mpm_grid_I = mpm_base - self.mpm_solver.grid_offset + mpm_offset
                            mpm_grid_mass = (
                                self.mpm_solver.grid[f, mpm_grid_I, i_b].mass / self.mpm_solver.particle_volume_scale
                            )

                            mpm_weight = gs.ti_float(1.0)
                            for d in ti.static(range(3)):
                                mpm_weight *= mpm_w[mpm_offset[d]][d]

                            # FEM -> MPM
                            mpm_grid_pos = (mpm_grid_I + self.mpm_solver.grid_offset) * self.mpm_solver.dx
                            signed_dist = (mpm_grid_pos - pos).dot(surface_normal)
                            if signed_dist <= self.mpm_solver.dx:  # NOTE: use dx as minimal unit for collision
                                vel_mpm_at_cell = mpm_weight * self.mpm_solver.grid[f, mpm_grid_I, i_b].vel_out
                                mass_mpm_at_cell = mpm_weight * mpm_grid_mass

                                vel_mpm += vel_mpm_at_cell
                                mass_mpm += mass_mpm_at_cell

                                if mass_mpm_at_cell > gs.EPS:
                                    delta_mpm_vel_at_cell_unmul = (
                                        vel_fem_sv * mpm_weight - self.mpm_solver.grid[f, mpm_grid_I, i_b].vel_out
                                    )
                                    mass_mul_at_cell = (
                                        mpm_grid_mass / mass_fem_sv
                                    )  # NOTE: use un-reweighted mass instead of mass_mpm_at_cell
                                    delta_mpm_vel_at_cell = delta_mpm_vel_at_cell_unmul * mass_mul_at_cell
                                    self.mpm_solver.grid[f, mpm_grid_I, i_b].vel_out += delta_mpm_vel_at_cell

                                    new_vel_fem_sv -= delta_mpm_vel_at_cell * mass_mpm_at_cell / mass_fem_sv

                        # MPM -> FEM
                        if mass_mpm > gs.EPS:
                            # delta_mv = (vel_mpm - vel_fem_sv) * mass_mpm
                            # delta_vel_fem_sv = delta_mv / mass_fem_sv
                            # self.fem_solver.elements_v[f + 1, iv].vel += delta_vel_fem_sv
                            self.fem_solver.elements_v[f + 1, iv, i_b].vel = new_vel_fem_sv

                # FEM <-> SPH TODO: this doesn't work well
                if ti.static(self._fem_sph):
                    for j in ti.static(range(3)):
                        iv = self.fem_solver.surface[i_s].tri2v[j]
                        pos = self.fem_solver.elements_v[f, iv, i_b].pos
                        vel_fem_sv = self.fem_solver.elements_v[f + 1, iv, i_b].vel
                        mass_fem_sv = mass / 4.0

                        dx = self.sph_solver.hash_grid_cell_size  # self._dx
                        stencil_size = 2  # self._stencil_size

                        base = self.sph_solver.sh.pos_to_grid(pos - 0.5 * dx)

                        # ---------- SPH -> FEM ----------
                        sph_vel = ti.Vector([0.0, 0.0, 0.0])
                        colliding_particles = 0
                        for offset in ti.grouped(ti.ndrange(stencil_size, stencil_size, stencil_size)):
                            slot_idx = self.sph_solver.sh.grid_to_slot(base + offset)
                            for k in range(
                                self.sph_solver.sh.slot_start[slot_idx, i_b],
                                self.sph_solver.sh.slot_start[slot_idx, i_b]
                                + self.sph_solver.sh.slot_size[slot_idx, i_b],
                            ):
                                if ti.abs(pos - self.sph_solver.particles_reordered.pos[k, i_b]).max() < dx * 0.5:
                                    sph_vel += self.sph_solver.particles_reordered.vel[k, i_b]
                                    colliding_particles += 1

                        if colliding_particles > 0:
                            vel_old = vel_fem_sv
                            vel_fem_sv_unprojected = sph_vel / colliding_particles
                            vel_fem_sv = (
                                vel_fem_sv_unprojected.dot(surface_normal) * surface_normal
                            )  # exclude tangential velocity

                            # ---------- FEM -> SPH ----------
                            delta_mv = mass_fem_sv * (vel_fem_sv - vel_old)

                            for offset in ti.grouped(ti.ndrange(stencil_size, stencil_size, stencil_size)):
                                slot_idx = self.sph_solver.sh.grid_to_slot(base + offset)
                                for k in range(
                                    self.sph_solver.sh.slot_start[slot_idx, i_b],
                                    self.sph_solver.sh.slot_start[slot_idx, i_b]
                                    + self.sph_solver.sh.slot_size[slot_idx, i_b],
                                ):
                                    if ti.abs(pos - self.sph_solver.particles_reordered.pos[k, i_b]).max() < dx * 0.5:
                                        self.sph_solver.particles_reordered[k, i_b].vel = (
                                            self.sph_solver.particles_reordered[k, i_b].vel
                                            - delta_mv / self.sph_solver.particles_info_reordered[k, i_b].mass
                                        )

                            self.fem_solver.elements_v[f + 1, iv, i_b].vel = vel_fem_sv

                # boundary condition
                for j in ti.static(range(3)):
                    iv = self.fem_solver.surface[i_s].tri2v[j]
                    _, self.fem_solver.elements_v[f + 1, iv, i_b].vel = self.fem_solver.boundary.impose_pos_vel(
                        self.fem_solver.elements_v[f, iv, i_b].pos, self.fem_solver.elements_v[f + 1, iv, i_b].vel
                    )

    def fem_hydroelastic(self, f: ti.i32):
        # Floor contact

        # collision detection
        self.fem_solver.floor_hydroelastic_detection(f)

    @ti.kernel
    def sph_rigid(
        self,
        f: ti.i32,
        geoms_state: array_class.GeomsState,
        geoms_info: array_class.GeomsInfo,
        links_state: array_class.LinksState,
        rigid_global_info: array_class.RigidGlobalInfo,
        sdf_info: array_class.SDFInfo,
        collider_static_config: ti.template(),
    ):
        for i_p, i_b in ti.ndrange(self.sph_solver._n_particles, self.sph_solver._B):
            if self.sph_solver.particles_ng_reordered[i_p, i_b].active:
                for i_g in range(self.rigid_solver.n_geoms):
                    if geoms_info.needs_coup[i_g]:
                        (
                            self.sph_solver.particles_reordered[i_p, i_b].vel,
                            self.sph_rigid_normal_reordered[i_p, i_g, i_b],
                        ) = self._func_collide_with_rigid_geom_robust(
                            self.sph_solver.particles_reordered[i_p, i_b].pos,
                            self.sph_solver.particles_reordered[i_p, i_b].vel,
                            self.sph_solver.particles_info_reordered[i_p, i_b].mass,
                            self.sph_rigid_normal_reordered[i_p, i_g, i_b],
                            i_g,
                            i_b,
                            geoms_state,
                            geoms_info,
                            links_state,
                            rigid_global_info,
                            sdf_info,
                            collider_static_config,
                        )

    @ti.kernel
    def kernel_pbd_rigid_collide(
        self,
        geoms_state: array_class.GeomsState,
        geoms_info: array_class.GeomsInfo,
        links_state: array_class.LinksState,
        sdf_info: array_class.SDFInfo,
        rigid_global_info: array_class.RigidGlobalInfo,
        collider_static_config: ti.template(),
    ):
        for i_p, i_b in ti.ndrange(self.pbd_solver._n_particles, self.sph_solver._B):
            if self.pbd_solver.particles_ng_reordered[i_p, i_b].active:
                # NOTE: Couldn't figure out a good way to handle collision with non-free particle. Such collision is not phsically plausible anyway.
                for i_g in range(self.rigid_solver.n_geoms):
                    if geoms_info.needs_coup[i_g]:
                        (
                            self.pbd_solver.particles_reordered[i_p, i_b].pos,
                            self.pbd_solver.particles_reordered[i_p, i_b].vel,
                            self.pbd_rigid_normal_reordered[i_p, i_b, i_g],
                        ) = self._func_pbd_collide_with_rigid_geom(
                            i_p,
                            self.pbd_solver.particles_reordered[i_p, i_b].pos,
                            self.pbd_solver.particles_reordered[i_p, i_b].vel,
                            self.pbd_solver.particles_info_reordered[i_p, i_b].mass,
                            self.pbd_rigid_normal_reordered[i_p, i_b, i_g],
                            i_g,
                            i_b,
                            geoms_state,
                            geoms_info,
                            links_state,
                            sdf_info,
                            rigid_global_info,
                            collider_static_config,
                        )

    @ti.kernel
    def kernel_attach_pbd_to_rigid_link(
        self,
        particles_idx: ti.types.ndarray(),
        envs_idx: ti.types.ndarray(),
        link_idx: ti.i32,
        links_state: LinksState,
    ) -> None:
        """
        Sets listed particles in listed environments to be animated by the link.

        Current position of the particle, relatively to the link, is stored and preserved.
        """
        pdb = self.pbd_solver

        for i_p_, i_b_ in ti.ndrange(particles_idx.shape[1], envs_idx.shape[0]):
            i_p = particles_idx[i_b_, i_p_]
            i_b = envs_idx[i_b_]
            link_pos = links_state.pos[link_idx, i_b]
            link_quat = links_state.quat[link_idx, i_b]

            # compute local offset from link to the particle
            world_pos = pdb.particles[i_p, i_b].pos
            local_pos = ti_inv_transform_by_trans_quat(world_pos, link_pos, link_quat)

            # set particle to be animated (not free) and store animation info
            pdb.particles[i_p, i_b].free = False
            self.particle_attach_info[i_p, i_b].link_idx = link_idx
            self.particle_attach_info[i_p, i_b].local_pos = local_pos

    @ti.kernel
    def kernel_pbd_rigid_clear_animate_particles_by_link(
        self,
        particles_idx: ti.types.ndarray(),
        envs_idx: ti.types.ndarray(),
    ) -> None:
        """Detach listed particles from links, and simulate them freely."""
        pdb = self.pbd_solver
        for i_p_, i_b_ in ti.ndrange(particles_idx.shape[1], envs_idx.shape[0]):
            i_p = particles_idx[i_b_, i_p_]
            i_b = envs_idx[i_b_]
            pdb.particles[i_p, i_b].free = True
            self.particle_attach_info[i_p, i_b].link_idx = -1
            self.particle_attach_info[i_p, i_b].local_pos = ti.math.vec3([0.0, 0.0, 0.0])

    @ti.kernel
    def kernel_pbd_rigid_solve_animate_particles_by_link(self, clamped_inv_dt: ti.f32, links_state: LinksState):
        """
        Itearates all particles and environments, and sets corrective velocity for all animated particle.

        Computes target position and velocity from the attachment/reference link and local offset position.

        Note, that this step shoudl be done after rigid solver update, and before PDB solver update.
        Currently, this is done after both rigid and PBD solver updates, hence the corrective velocity
        is off by a frame.

        Note, it's adviced to clamp inv_dt to avoid large jerks and instability. 1/0.02 might be a good max value.
        """
        pdb = self.pbd_solver
        for i_p, i_env in ti.ndrange(pdb._n_particles, pdb._B):
            if self.particle_attach_info[i_p, i_env].link_idx >= 0:
                # read link state
                link_idx = self.particle_attach_info[i_p, i_env].link_idx
                link_pos = links_state.pos[link_idx, i_env]
                link_quat = links_state.quat[link_idx, i_env]

                link_lin_vel = links_state.cd_vel[link_idx, i_env]
                link_ang_vel = links_state.cd_ang[link_idx, i_env]
                link_com_in_world = links_state.root_COM[link_idx, i_env] + links_state.i_pos[link_idx, i_env]

                # calculate target pos and vel of the particle
                local_pos = self.particle_attach_info[i_p, i_env].local_pos
                target_world_pos = ti_transform_by_trans_quat(local_pos, link_pos, link_quat)

                world_arm = target_world_pos - link_com_in_world
                target_world_vel = link_lin_vel + link_ang_vel.cross(world_arm)

                # compute and apply corrective velocity
                i_rp = pdb.particles_ng[i_p, i_env].reordered_idx
                particle_pos = pdb.particles_reordered[i_rp, i_env].pos
                pos_correction = target_world_pos - particle_pos
                corrective_vel = pos_correction * clamped_inv_dt
                pdb.particles_reordered[i_rp, i_env].vel = corrective_vel + target_world_vel

    @ti.func
    def _func_pbd_collide_with_rigid_geom(
        self,
        i,
        pos_world,
        vel,
        mass,
        normal_prev,
        geom_idx,
        batch_idx,
        geoms_state: array_class.GeomsState,
        geoms_info: array_class.GeomsInfo,
        links_state: array_class.LinksState,
        sdf_info: array_class.SDFInfo,
        rigid_global_info: array_class.RigidGlobalInfo,
        collider_static_config: ti.template(),
    ):
        """
        Resolves collision when a particle is already in collision with a rigid object.
        This function assumes known normal_rigid and influence.
        """
        signed_dist = sdf.sdf_func_world(
            geoms_state=geoms_state,
            geoms_info=geoms_info,
            sdf_info=sdf_info,
            pos_world=pos_world,
            geom_idx=geom_idx,
            batch_idx=batch_idx,
        )
        contact_normal = sdf.sdf_func_normal_world(
            geoms_state=geoms_state,
            geoms_info=geoms_info,
            rigid_global_info=rigid_global_info,
            collider_static_config=collider_static_config,
            sdf_info=sdf_info,
            pos_world=pos_world,
            geom_idx=geom_idx,
            batch_idx=batch_idx,
        )
        new_pos = pos_world
        new_vel = vel
        if signed_dist < self.pbd_solver.particle_size / 2:  # skip non-penetration particles
            stiffness = 1.0  # value in [0, 1]

            # we don't consider friction for now
            # friction = 0.15
            # vel_rigid = self.rigid_solver._func_vel_at_point(
            #     pos_world=pos_world,
            #     link_idx=geoms_info.link_idx[geom_idx],
            #     i_b=batch_idx,
            #     links_state=links_state,
            # )
            # rvel = vel - vel_rigid
            # rvel_normal_magnitude = rvel.dot(contact_normal)  # negative if inward
            # rvel_tan = rvel - rvel_normal_magnitude * contact_normal
            # rvel_tan_norm = rvel_tan.norm(gs.EPS)

            #################### rigid -> particle ####################

            energy_loss = 0.0  # value in [0, 1]
            new_pos = pos_world + stiffness * contact_normal * (self.pbd_solver.particle_size / 2 - signed_dist)
            prev_pos = self.pbd_solver.particles_reordered[i, batch_idx].ipos
            new_vel = (new_pos - prev_pos) / self.pbd_solver._substep_dt

            #################### particle -> rigid ####################
            delta_mv = mass * (new_vel - vel)
            force = (-delta_mv / self.rigid_solver._substep_dt) * (1 - energy_loss)

            self.rigid_solver._func_apply_coupling_force(
                pos_world,
                force,
                geoms_info.link_idx[geom_idx],
                batch_idx,
                links_state,
            )

        return new_pos, new_vel, contact_normal

    def preprocess(self, f):
        # preprocess for MPM CPIC
        if self._rigid_mpm and self.mpm_solver.enable_CPIC:
            self.mpm_surface_to_particle(
                f,
                self.rigid_solver.geoms_state,
                self.rigid_solver.geoms_info,
                self.rigid_solver.sdf._sdf_info,
                self.rigid_solver._rigid_global_info,
                self.rigid_solver.collider._collider_static_config,
            )

    @ti.kernel
    def mpm_grid_thermal_diffusion(self, f: ti.i32):
        for I in ti.grouped(ti.ndrange(*self.mpm_solver.grid_res)):
            for i_b in range(self.mpm_solver._B):
                m_C = self.mpm_solver.grid[f, I, i_b].mass_thermal
                if m_C > gs.EPS:
                    T_C = self.mpm_solver.grid[f, I, i_b].temp
                    laplacian = 0.0
                    
                    I_left = I + ti.Vector([-1, 0, 0])
                    I_right = I + ti.Vector([1, 0, 0])
                    I_down = I + ti.Vector([0, -1, 0])
                    I_up = I + ti.Vector([0, 1, 0])
                    I_back = I + ti.Vector([0, 0, -1])
                    I_front = I + ti.Vector([0, 0, 1])
                    
                    if I_left[0] >= 0:
                        m_N = self.mpm_solver.grid[f, I_left, i_b].mass_thermal
                        if m_N > gs.EPS:
                            laplacian += (ti.min(m_C, m_N) / m_C) * (self.mpm_solver.grid[f, I_left, i_b].temp - T_C)
                    
                    if I_right[0] < self.mpm_solver.grid_res[0]:
                        m_N = self.mpm_solver.grid[f, I_right, i_b].mass_thermal
                        if m_N > gs.EPS:
                            laplacian += (ti.min(m_C, m_N) / m_C) * (self.mpm_solver.grid[f, I_right, i_b].temp - T_C)
                            
                    if I_down[1] >= 0:
                        m_N = self.mpm_solver.grid[f, I_down, i_b].mass_thermal
                        if m_N > gs.EPS:
                            laplacian += (ti.min(m_C, m_N) / m_C) * (self.mpm_solver.grid[f, I_down, i_b].temp - T_C)
                            
                    if I_up[1] < self.mpm_solver.grid_res[1]:
                        m_N = self.mpm_solver.grid[f, I_up, i_b].mass_thermal
                        if m_N > gs.EPS:
                            laplacian += (ti.min(m_C, m_N) / m_C) * (self.mpm_solver.grid[f, I_up, i_b].temp - T_C)
                            
                    if I_back[2] >= 0:
                        m_N = self.mpm_solver.grid[f, I_back, i_b].mass_thermal
                        if m_N > gs.EPS:
                            laplacian += (ti.min(m_C, m_N) / m_C) * (self.mpm_solver.grid[f, I_back, i_b].temp - T_C)
                            
                    if I_front[2] < self.mpm_solver.grid_res[2]:
                        m_N = self.mpm_solver.grid[f, I_front, i_b].mass_thermal
                        if m_N > gs.EPS:
                            laplacian += (ti.min(m_C, m_N) / m_C) * (self.mpm_solver.grid[f, I_front, i_b].temp - T_C)

                    alpha = self.mpm_solver._alpha_thermal
                    dx = self.mpm_solver.dx
                    dt = self.mpm_solver.substep_dt
                    
                    T_new = T_C + alpha * dt / (dx * dx) * laplacian
                    self.mpm_solver.grid[f, I, i_b].temp_diffused = T_new

    def couple(self, f):
        # MPM <-> all others
        if self.mpm_solver.is_active:
            self.mpm_grid_op(
                f,
                self.sim.cur_t,
                geoms_state=self.rigid_solver.geoms_state,
                geoms_info=self.rigid_solver.geoms_info,
                links_state=self.rigid_solver.links_state,
                rigid_global_info=self.rigid_solver._rigid_global_info,
                sdf_info=self.rigid_solver.sdf._sdf_info,
                collider_static_config=self.rigid_solver.collider._collider_static_config,
            )
            if self.mpm_solver._enable_thermal:
                self.mpm_grid_thermal_diffusion(f)

        # SPH <-> Rigid
        if self._rigid_sph:
            self.sph_rigid(
                f,
                self.rigid_solver.geoms_state,
                self.rigid_solver.geoms_info,
                self.rigid_solver.links_state,
                self.rigid_solver._rigid_global_info,
                self.rigid_solver.sdf._sdf_info,
                self.rigid_solver.collider._collider_static_config,
            )

        # PBD <-> Rigid
        if self._rigid_pbd:
            self.kernel_pbd_rigid_collide(
                geoms_state=self.rigid_solver.geoms_state,
                geoms_info=self.rigid_solver.geoms_info,
                links_state=self.rigid_solver.links_state,
                sdf_info=self.rigid_solver.sdf._sdf_info,
                rigid_global_info=self.rigid_solver._rigid_global_info,
                collider_static_config=self.rigid_solver.collider._collider_static_config,
            )

            # 1-way: animate particles by links
            full_step_inv_dt = 1.0 / self.pbd_solver._dt
            clamped_inv_dt = min(full_step_inv_dt, CLAMPED_INV_DT)
            self.kernel_pbd_rigid_solve_animate_particles_by_link(clamped_inv_dt, self.rigid_solver.links_state)

        if self.fem_solver.is_active:
            self.fem_surface_force(
                f,
                self.rigid_solver.geoms_state,
                self.rigid_solver.geoms_info,
                self.rigid_solver.links_state,
                self.rigid_solver._rigid_global_info,
                self.rigid_solver.sdf._sdf_info,
                self.rigid_solver.collider._collider_static_config,
            )
            self.fem_rigid_link_constraints()

    def couple_grad(self, f):
        if self.fem_solver.is_active:
            self.fem_surface_force.grad(
                f,
                self.rigid_solver.geoms_state,
                self.rigid_solver.geoms_info,
                self.rigid_solver.links_state,
                self.rigid_solver._rigid_global_info,
                self.rigid_solver.sdf._sdf_info,
                self.rigid_solver.collider._collider_static_config,
            )
        if self.mpm_solver.is_active:
            self.mpm_grid_op.grad(
                f,
                self.sim.cur_t,
                geoms_state=self.rigid_solver.geoms_state,
                geoms_info=self.rigid_solver.geoms_info,
                links_state=self.rigid_solver.links_state,
                rigid_global_info=self.rigid_solver._rigid_global_info,
                sdf_info=self.rigid_solver.sdf._sdf_info,
                collider_static_config=self.rigid_solver.collider._collider_static_config,
            )

    @property
    def active_solvers(self):
        """All the active solvers managed by the scene's simulator."""
        return self.sim.active_solvers
import gstaichi as ti

import genesis as gs

from .base import Base


@ti.data_oriented
class ElastoPlastic(Base):
    """
    The elasto-plastic material class for MPM.

    Note
    ----
    Default yield ratio comes from the SNOW material in taichi's MPM implementation:
    https://github.com/taichi-dev/taichi_elements/blob/d19678869a28b09a32ef415b162e35dc929b792d/engine/mpm_solver.py#L434

    Parameters
    ----------
    E: float, optional
        Young's modulus. Default is 1e6.
    nu: float, optional
        Poisson ratio. Default is 0.2.
    rho: float, optional
        Density (kg/m^3). Default is 1000.
    lam: float, optional
        The first Lame's parameter. Default is None, computed by E and nu.
    mu: float, optional
        The second Lame's parameter. Default is None, computed by E and nu.
    sampler: str, optional
        Particle sampler ('pbs', 'regular', 'random'). Note that 'pbs' is only supported on Linux x86 for now. Defaults
        to 'pbs' on supported platforms, 'random' otherwise.
    yield_lower: float, optional
        Lower bound for the yield clamp (ignored if using von Mises). Default is 2.5e-2.
    yield_higher: float, optional
        Upper bound for the yield clamp (ignored if using von Mises). Default is 4.5e-2.
    use_von_mises: bool, optional
        Whether to use von Mises yield criterion. Default is True.
    von_mises_yield_stress: float, optional
        Yield stress for von Mises criterion. Default is 10000.
    """

    def __init__(
        self,
        E=1e6,  # Young's modulus
        nu=0.2,  # Poisson's ratio
        rho=1000.0,  # density (kg/m^3)
        lam=None,
        mu=None,
        sampler=None,
        yield_lower=2.5e-2,
        yield_higher=4.5e-3,
        use_von_mises=True,  # von Mises yield criterion
        von_mises_yield_stress=10000.0,
        T_ref=293.15,  # Reference temperature for Johnson-Cook
        T_melt=1793.0,  # Melting point for 4340 steel
        jc_m=1.03,  # Johnson-Cook thermal softening exponent
    ):
        super().__init__(E, nu, rho, lam, mu, sampler)

        self._yield_lower = yield_lower
        self._yield_higher = yield_higher
        self._use_von_mises = use_von_mises
        self._von_mises_yield_stress = von_mises_yield_stress
        self._T_ref = T_ref
        self._T_melt = T_melt
        self._jc_m = jc_m

    @ti.func
    def update_F_S_Jp(self, J, F_tmp, U, S, V, Jp, temp):
        F_new = ti.Matrix.zero(gs.ti_float, 3, 3)
        S_new = ti.Matrix.zero(gs.ti_float, 3, 3)
        delta_gamma_out = gs.ti_float(0.0)
        effective_yield_out = gs.ti_float(0.0)

        if ti.static(self.use_von_mises):
            S_new = ti.max(S, 0.05)  # to prevent NaN
            epsilon = ti.Vector([ti.log(S_new[0, 0]), ti.log(S_new[1, 1]), ti.log(S_new[2, 2])])
            epsilon_hat = epsilon - (epsilon.sum() / 3)
            epsilon_hat_norm = epsilon_hat.norm(gs.EPS)

            # Temperature-dependent yield (Johnson-Cook thermal term)
            T_star = (temp - self._T_ref) / (self._T_melt - self._T_ref)
            T_star = ti.max(0.0, ti.min(1.0, T_star))
            thermal_softening = 1.0 - ti.pow(T_star, self._jc_m)
            effective_yield = self._von_mises_yield_stress * thermal_softening

            delta_gamma = epsilon_hat_norm - effective_yield / (2 * self._mu)

            if delta_gamma > 0:  # Yields
                epsilon -= (delta_gamma / epsilon_hat_norm) * epsilon_hat
                S_new = ti.Matrix.zero(gs.ti_float, 3, 3)
                for d in ti.static(range(3)):
                    S_new[d, d] = ti.exp(epsilon[d])
                F_new = U @ S_new @ V.transpose()
                delta_gamma_out = delta_gamma
                effective_yield_out = effective_yield
            else:
                F_new = F_tmp

        else:
            S_new = ti.Matrix.zero(gs.ti_float, 3, 3)
            for d in ti.static(range(3)):
                S_new[d, d] = min(max(S[d, d], 1 - self._yield_lower), 1 + self._yield_higher)
            F_new = U @ S_new @ V.transpose()

        Jp_new = Jp
        return F_new, S_new, Jp_new, delta_gamma_out, effective_yield_out

    @property
    def yield_lower(self):
        """Lower bound for the yield clamp (ignored if using von Mises)."""
        return self._yield_lower

    @property
    def yield_higher(self):
        """Upper bound for the yield clamp (ignored if using von Mises)."""
        return self._yield_higher

    @property
    def use_von_mises(self):
        """Whether to use von Mises yield criterion."""
        return self._use_von_mises

    @property
    def von_mises_yield_stress(self):
        """Yield stress for von Mises criterion."""
        return self._von_mises_yield_stress

```
