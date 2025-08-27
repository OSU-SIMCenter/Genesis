import inspect
import gstaichi as ti
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
