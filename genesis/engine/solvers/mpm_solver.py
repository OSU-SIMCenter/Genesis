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
    def grid_op_thermal(self, f: ti.i32):
        for i, j, k, i_b in ti.ndrange(*self._grid_res, self._B):
            m = self.grid[f, i, j, k, i_b].mass_thermal
            if m > 0:
                # Normalize temperature
                self.grid[f, i, j, k, i_b].temp /= m
                
                # Apply diffusion (Explicit Laplacian) - Simply using current frame's values from neighbors
                # Note: For strict correctness this should be double-buffered or use a temporary, but for small dt this is okay
                # Alternatively, we could do it in two passes if we really cared, but let's do in-place for now as per plan
                # Actually, in-place diffusion is bad for parallelization order.
                # However, since we reset grid every step, we can't easily use previous step's grid.
                # Let's just do a simplified diffusion: T += alpha * Laplace(T) * dt
                # But calculating Laplace(T) requires neighbors.
                
                # Let's skip diffusion for this first pass and focus on Boundary/Environment cooling.
                # Or implementing a basic version?
                # The user's research suggested: T_new = T + dt * alpha * Laplacian(T)
                
                # Let's implement cooling
                T_curr = self.grid[f, i, j, k, i_b].temp
                T_air = 293.15
                
                # Simple Newton cooling if 'surface' (low mass density or checking neighbors?)
                # User used: if self.grid_thermal_mass[I] < self.p_rho * self.dx**3 * 0.8:
                # We can approximate density check
                rho_cell = m / (self._dx ** 3)
                # Assuming approximate density of steel ~7800
                if rho_cell < 7000.0: # Surface-ish
                     h_conv = 50.0 
                     dT = -self.substep_dt * h_conv * (T_curr - T_air) # Simplified area term
                     self.grid[f, i, j, k, i_b].temp += dT
            
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
        self.grid_op_thermal(f)

