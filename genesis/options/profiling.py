from pydantic import StrictBool

from .options import Options
from genesis.profiling.profiler import Profiler


class ProfilingOptions(Options):
    """
    Profiling options

    Parameters
    ----------
    enabled : bool
        Whether profiling is enabled. Default False
    show_FPS : bool
        Whether to show the frame rate each step. Default True
    FPS_tracker_alpha : float
        Exponential decay momentum for FPS moving average. Default 0.95
    """

    enabled: StrictBool = False
    show_FPS: StrictBool = True
    FPS_tracker_alpha: float = 0.95

    class Configs(Options):
        """Container for all profiling configurations."""
        class Scene(Options):
            """Scene profiling configurations."""
            class Step(Options):
                """Settings for the scene step."""
                sim: bool = False
                visualizer: bool = True
                fps_tracker: bool = True
                recorder_manager: bool = True
            step: Step = Step()
        
        class Simulator(Options):
            """Simulator profiling configurations."""
            preprocess: bool = True
            substep_pre_couple: bool = True
            substep: bool = True
            couple: bool = True
            substep_post_couple: bool = True
            process_input: bool = True
            save_ckpt: bool = True
            clear_external_force: bool = True
            sensor_manager_step: bool = True
            rigid_solver_substep: bool = True

        class Rigid(Options):
            """Rigid solver profiling configurations."""
            step_1: bool = True
            constraints: bool = True
            constraints_detect: bool = True
            constraints_add: bool = True
            constraints_solve: bool = True
            step_2: bool = True
            post_couple: bool = True

        class Teleop(Options):
            """Teleop socket profiling configurations."""
            logic: bool = True
            logic_step: bool = True
            logic_idle: bool = True
            logic_get_resistance: bool = True
            logic_get_pos: bool = True
            logic_check_stop: bool = True
            logic_calc_cmd: bool = True
            logic_apply_vel: bool = True
            logic_update_state: bool = True
            logic_prep: bool = True

            action: bool = True
            clear_force: bool = True
            recon: bool = True
            recon_mesh: bool = True
            recon_update_skinning: bool = True
            recon_check_rebind: bool = True
            io: bool = True
            render_update: bool = True

        scene: Scene = Scene()
        simulator: Simulator = Simulator()
        rigid: Rigid = Rigid()
        teleop: Teleop = Teleop()
    configs: Configs = Configs()

    def __init__(self, **data):
        super().__init__(**data)
        object.__setattr__(self, 'profiler', Profiler(enabled=self.enabled))