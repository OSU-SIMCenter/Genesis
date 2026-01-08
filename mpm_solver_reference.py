import inspect
import taichi as ti
import torch

import genesis as gs

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
        if use_legacy_solver: return BaseMPMSolver(*args, **kwargs)
        return super().__new__(cls)
    
    def get_particle_state_template(self):
        return super().get_particle_state_template() | {"temp": gs.ti_float}
    
    def get_particle_state_render_template(self):
        return super().get_particle_state_render_template() | {"temp": gs.ti_float}

    def get_grid_cell_state_template(self):
        return super().get_grid_cell_state_template() | {"temp_in": gs.ti_float, "temp_out": gs.ti_float}
    
    # ------------------------------------------------------------------------------------
    # ------------------------------------ stepping --------------------------------------
    # ------------------------------------------------------------------------------------

    @ti.func
    def copy_frame_helper(self, source: ti.i32, target: ti.i32, i_p: ti.i32, i_b: ti.i32):
        self.particles[target, i_p, i_b].temp = self.particles[source, i_p, i_b].temp
        super().copy_frame_helper(source, target, i_p, i_b)
    
    @ti.func
    def copy_grad_helper(self, source: ti.i32, target: ti.i32, i_p: ti.i32, i_b: ti.i32):
        self.particles.grad[target, i_p, i_b].temp = self.particles.grad[source, i_p, i_b].temp
        super().copy_grad_helper(source, target, i_p, i_b)
    
    @ti.func
    def reset_grid_helper(self, f: ti.i32, i: ti.i32, j: ti.i32, k: ti.i32, i_b: ti.i32):
        self.grid[f, i, j, k, i_b].temp_in = gs.ti_float(0.0)
        self.grid[f, i, j, k, i_b].temp_out = gs.ti_float(0.0)
        super().reset_grid_helper(f, i, j, k, i_b)
    
    @ti.func
    def reset_grid_grad_helper(self, f: ti.i32, i: ti.i32, j: ti.i32, k: ti.i32, i_b: ti.i32):
        self.grid.grad[f, i, j, k, i_b].temp_in = gs.ti_float(0.0)
        self.grid.grad[f, i, j, k, i_b].temp_out = gs.ti_float(0.0)
        super().reset_grid_grad_helper(f, i, j, k, i_b)

    @ti.func
    def reset_grad_till_frame_helper(self, i_f: ti.i32, i_p: ti.i32, i_b: ti.i32):
        self.particles.grad[i_f, i_p, i_b].temp = gs.ti_float(0.0)
        super().reset_grad_till_frame_helper(i_f, i_p, i_b)
    
    # ------------------------------------------------------------------------------------
    # ------------------------------------ gradient --------------------------------------
    # ------------------------------------------------------------------------------------

    def add_grad_from_state_helper(self, state):
        if state.temp.grad is not None:
            state.temp.assert_contiguous()
            self.add_grad_from_temp(self._sim.cur_substep_local, state.temp.grad)
        super().add_grad_from_state_helper(state)
    
    @ti.kernel
    def add_grad_from_temp(self, f: ti.i32, temp_grad: ti.types.ndarray()):
        # temp_grad shape: [B, n_particles]
        for i_p, i_b in ti.ndrange(self._n_particles, self._B):
            self.particles.grad[f, i_p, i_b].temp += temp_grad[i_b, i_p]
    
    def init_ckpt(self, ckpt_name):
        super().init_ckpt(ckpt_name)
        self._ckpt[ckpt_name]["temp"] = torch.zeros((self._B, self._n_particles), dtype=gs.tc_float)

    def get_ckpt_state(self, ckpt_name):
        return super().get_ckpt_state(ckpt_name) + (self._ckpt[ckpt_name]["temp"],)

    # ------------------------------------------------------------------------------------
    # --------------------------------------- io -----------------------------------------
    # ------------------------------------------------------------------------------------


