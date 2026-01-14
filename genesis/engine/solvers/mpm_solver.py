import inspect
import gstaichi as ti
import numpy as np
import torch
import genesis as gs
import genesis.utils.array_class as array_class
import genesis.utils.sdf_decomp as sdf_decomp
from .base_mpm_solver import BaseMPMSolver

_BASE_INIT_SIG = inspect.signature(BaseMPMSolver.__init__)

@ti.data_oriented
class MPMSolver(BaseMPMSolver):
    # ------------------------------------------------------------------------------------
    # --------------------------------- Initialization -----------------------------------
    # ------------------------------------------------------------------------------------

    def __new__(cls, *args, **kwargs):
        options = _BASE_INIT_SIG.bind(None, *args, **kwargs).arguments.get('options')
        use_legacy_solver = options.use_legacy_solver; del options.use_legacy_solver
        if use_legacy_solver:
            return BaseMPMSolver(*args, **kwargs)
        return super().__new__(cls)
    
    def get_particle_state_template(self):
        template = super()._make_particle_state_template()
        template.update({
            "temp": gs.ti_float,
            "plastic_strain": gs.ti_float,
            "plastic_work": gs.ti_float,
        })
        return template
    
    def get_grid_cell_state_template(self):
        template = super()._make_grid_cell_state_template()
        template.update({
            "temp": gs.ti_float,
            "mass_thermal": gs.ti_float,
        })
        return template

    def _make_particle_state_template(self):
        return self.get_particle_state_template()

    def _make_grid_cell_state_template(self):
        return self.get_grid_cell_state_template()

    def init_particle_fields(self):
        super().init_particle_fields()
        # Initialize default values
        self.get_particle_temp_field().fill(self._options.default_initial_temperature)
        # plastic_strain and plastic_work default to 0.0 which is fine

    def get_particle_temp_field(self):
        # Helper to get the temp field from the struct
        # We need to access it via self.particles[...].temp, but since it's an SOA field,
        # we can't easily get the 'whole field' reference for .fill() unless we use the field object directly.
        # However, BaseMPMSolver.init_particle_fields defines:
        # self.particles = struct_particle_state.field(...)
        # So self.particles.temp IS the field.
        return self.particles.temp

    def init_ckpt(self):
        super().init_ckpt()
        self._ckpt_thermal_keys = ["temp", "plastic_strain", "plastic_work"]

    def _alloc_ckpt_buffers(self, ckpt_name):
        super()._alloc_ckpt_buffers(ckpt_name)
        self._ckpt[ckpt_name]["temp"] = torch.zeros((self._B, self._n_particles), dtype=gs.tc_float)
        self._ckpt[ckpt_name]["plastic_strain"] = torch.zeros((self._B, self._n_particles), dtype=gs.tc_float)
        self._ckpt[ckpt_name]["plastic_work"] = torch.zeros((self._B, self._n_particles), dtype=gs.tc_float)

    def _save_ckpt_data(self, ckpt_name):
        super()._save_ckpt_data(ckpt_name)
        self._kernel_save_thermal_state(
            0,
            self._ckpt[ckpt_name]["temp"],
            self._ckpt[ckpt_name]["plastic_strain"],
            self._ckpt[ckpt_name]["plastic_work"],
        )

    def _load_ckpt_data(self, ckpt_name):
        super()._load_ckpt_data(ckpt_name)
        self._kernel_load_thermal_state(
            0,
            self._ckpt[ckpt_name]["temp"],
            self._ckpt[ckpt_name]["plastic_strain"],
            self._ckpt[ckpt_name]["plastic_work"],
        )

    @ti.kernel
    def _kernel_save_thermal_state(
        self,
        f: ti.i32,
        temp: ti.types.ndarray(),
        plastic_strain: ti.types.ndarray(),
        plastic_work: ti.types.ndarray(),
    ):
        for i_p, i_b in ti.ndrange(self._n_particles, self._B):
            temp[i_b, i_p] = self.particles[f, i_p, i_b].temp
            plastic_strain[i_b, i_p] = self.particles[f, i_p, i_b].plastic_strain
            plastic_work[i_b, i_p] = self.particles[f, i_p, i_b].plastic_work

    @ti.kernel
    def _kernel_load_thermal_state(
        self,
        f: ti.i32,
        temp: ti.types.ndarray(),
        plastic_strain: ti.types.ndarray(),
        plastic_work: ti.types.ndarray(),
    ):
        for i_p, i_b in ti.ndrange(self._n_particles, self._B):
            self.particles[f, i_p, i_b].temp = temp[i_b, i_p]
            self.particles[f, i_p, i_b].plastic_strain = plastic_strain[i_b, i_p]
            self.particles[f, i_p, i_b].plastic_work = plastic_work[i_b, i_p]

    # ------------------------------------------------------------------------------------
    # --------------------------------- Thermal Kernels ----------------------------------
    # ------------------------------------------------------------------------------------

    @ti.func
    def p2g_transfer_extra_fields(self, f: ti.i32, i_p: ti.i32, idx: ti.template(), i_b: ti.i32, weight: ti.f32):
        mass = self.particles_info[i_p].mass
        temp = self.particles[f, i_p, i_b].temp
        self.grid[f, idx, i_b].temp += weight * mass * temp
        self.grid[f, idx, i_b].mass_thermal += weight * mass

    @ti.kernel
    def grid_op_thermal(
        self,
        f: ti.i32,
        geoms_state: array_class.GeomsState,
        geoms_info: array_class.GeomsInfo,
        rigid_global_info: array_class.RigidGlobalInfo,
        sdf_info: array_class.SDFInfo,
        collider_static_config: ti.template(),
    ):
        for i, j, k, i_b in ti.ndrange(*self._grid_res, self._B):
            m = self.grid[f, i, j, k, i_b].mass_thermal
            if m > 0:
                # Normalize temperature
                self.grid[f, i, j, k, i_b].temp /= m
                
                # --- Environmental Cooling (Air) ---
                T_curr = self.grid[f, i, j, k, i_b].temp
                T_air = 293.15
                
                # Simple Newton cooling if 'surface'
                rho_cell = m / (self._dx ** 3)
                # Assuming approximate density of steel ~7800
                if rho_cell < 7000.0: # Surface-ish
                     h_conv = 50.0 
                     # Use exponential decay for stability: T_new = T_air + (T_curr - T_air) * exp(-h*dt)
                     # Delta = T_new - T_curr = (T_air - T_curr) * (1 - exp(-h*dt))
                     decay_air = 1.0 - ti.exp(-self.substep_dt * h_conv)
                     dT_air = (T_air - T_curr) * decay_air
                     self.grid[f, i, j, k, i_b].temp += dT_air

                # --- Contact Cooling (Rigid Body) ---
                if ti.static(self.sim.rigid_solver.is_active):
                    pos_world = (ti.Vector([i, j, k]) + self._grid_offset) * self._dx
                    
                    # Check distance to all rigid bodies
                    # Note: Ideally we use efficient broadphase, but iterating geoms is fine for small number of tools
                    for i_g in range(self.sim.rigid_solver.n_geoms):
                        if geoms_info.needs_coup[i_g]:
                            signed_dist = sdf_decomp.sdf_func_world(
                                geoms_state=geoms_state,
                                geoms_info=geoms_info,
                                sdf_info=sdf_info,
                                pos_world=pos_world,
                                geom_idx=i_g,
                                batch_idx=i_b,
                            )
                            
                            # If close enough (e.g. within 1.5 dx), apply contact cooling
                            # We use a slightly generous threshold to capture the interface
                            if signed_dist < self._dx * 1.5:
                                T_rigid = 293.15 # TODO: Fetch actual temp if rigid body has thermal state
                                h_contact = 5000.0 # High conductivity for contact
                                
                                # Current temp might have changed due to air cooling
                                T_curr_updated = self.grid[f, i, j, k, i_b].temp
                                
                                # Use exponential decay for stability
                                decay_contact = 1.0 - ti.exp(-self.substep_dt * h_contact)
                                dT_contact = (T_rigid - T_curr_updated) * decay_contact
                                self.grid[f, i, j, k, i_b].temp += dT_contact

            else:
                 self.grid[f, i, j, k, i_b].temp = 293.15 # Air temp

    @ti.func
    def g2p_prologue(self, f: ti.i32, i_p: ti.i32, i_b: ti.i32):
        self.particles[f + 1, i_p, i_b].temp = 0.0
        self.particles[f + 1, i_p, i_b].plastic_strain = self.particles[f, i_p, i_b].plastic_strain
        self.particles[f + 1, i_p, i_b].plastic_work = self.particles[f, i_p, i_b].plastic_work

    @ti.func
    def g2p_transfer_extra_fields(self, f: ti.i32, i_p: ti.i32, i_b: ti.i32, weight: ti.f32, grid_index: ti.template()):
        self.particles[f + 1, i_p, i_b].temp += weight * self.grid[f, grid_index, i_b].temp

    # ------------------------------------------------------------------------------------
    # --------------------------------- Coupling Logic -----------------------------------
    # ------------------------------------------------------------------------------------

    @ti.func
    def get_particle_stress_scale(self, f: ti.i32, i_p: ti.i32, i_b: ti.i32):
        scale = 1.0
        # Thermal softening
        T = self.particles[f, i_p, i_b].temp
        T_ref = self._options.T_ref
        T_melt = self._options.T_melt
        m = 1.03 # Johnson-Cook thermal softening exponent for 4340 steel
        
        T_star = (T - T_ref) / (T_melt - T_ref)
        T_star = ti.max(0.0, ti.min(1.0, T_star))
        scale = 1.0 - ti.pow(T_star, m)
        return scale

    @ti.func
    def copy_frame_helper(self, source: ti.i32, target: ti.i32, i_p: ti.i32, i_b: ti.i32):
        BaseMPMSolver.copy_frame_helper(self, source, target, i_p, i_b)
        self.particles[target, i_p, i_b].temp = self.particles[source, i_p, i_b].temp
        self.particles[target, i_p, i_b].plastic_strain = self.particles[source, i_p, i_b].plastic_strain
        self.particles[target, i_p, i_b].plastic_work = self.particles[source, i_p, i_b].plastic_work

    @ti.func
    def reset_grid_helper(self, f: ti.i32, i: ti.i32, j: ti.i32, k: ti.i32, i_b: ti.i32):
        BaseMPMSolver.reset_grid_helper(self, f, i, j, k, i_b)
        self.grid[f, i, j, k, i_b].temp = 0.0
        self.grid[f, i, j, k, i_b].mass_thermal = 0.0

    def substep_pre_coupling(self, f):
        super().substep_pre_coupling(f)
        self.grid_op_thermal(
            f,
            self.sim.coupler.rigid_solver.geoms_state,
            self.sim.coupler.rigid_solver.geoms_info,
            self.sim.coupler.rigid_solver._rigid_global_info,
            self.sim.coupler.rigid_solver.sdf._sdf_info,
            self.sim.coupler.rigid_solver.collider._collider_static_config,
        )

