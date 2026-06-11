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
                sim: bool = True
                visualizer: bool = True
                fps_tracker: bool = True
                recorder: bool = True
            step: Step = Step()
        
        class Simulator(Options):
            """Simulator profiling configurations."""
            process_input: bool = True
            check_errno: bool = True
            clear_force: bool = True
            sensor_step: bool = True
            substep_pre_couple: bool = True
            substep: bool = True
            couple: bool = True
            couple_mpm_grid_op: bool = True
            couple_thermal_diffusion: bool = True
            substep_post_couple: bool = True
            mpm_g2p: bool = True
            save_ckpt: bool = True
            rigid_solver_substep: bool = True

        class Rigid(Options):
            """Rigid solver profiling configurations."""
            step_1: bool = True
            constraints: bool = True
            constraints_force: bool = True
            constraints_clear: bool = True
            constraints_detect: bool = True
            constraints_add: bool = True
            constraints_solve: bool = True
            step_2: bool = True

        class Teleop(Options):
            """Teleop socket profiling configurations."""
            step: bool = True
            physics: bool = True
            logic: bool = True
            logic_step: bool = True
            logic_apply_vel: bool = True
            logic_get_resistance: bool = True
            logic_update_state: bool = True
            io: bool = True
            recon: bool = True
            recon_mesh: bool = True
            recon_update_skinning: bool = True
            recon_update: bool = True
            recon_get_particles: bool = True
            recon_check_rebind: bool = True
            recon_rebind: bool = True
            recon_transform: bool = True
            render: bool = True
            apply_action: bool = True
            clear_forces: bool = True
            heating: bool = True
            thermal_bcs: bool = True
            thermal_telemetry: bool = True
            stability: bool = True
            stability_check: bool = True
            record: bool = True
            record_gpu_pull: bool = True
            record_append: bool = True
            sim_step: bool = True
            unified_recon: bool = True
            unified_recon_mesh_copy: bool = True
            diagnostics: bool = True
            physics_rebuild: bool = True
            physics_mesh_rebuild: bool = True
            induction_igl_sdf: bool = True
            induction_depth_upload: bool = True
            induction_pos_pull: bool = True
            io_recon_data: bool = True
            io_vertex_temp_prep: bool = True
            io_kdtree_vertex_temps: bool = True
            io_gpu_to_cpu: bool = True
            io_websocket_pack_send: bool = True
            render_mesh_overlay_sync: bool = True
            render_mesh_overlay_refresh: bool = True
            render_particle_sync: bool = True
            render_particle_gpu_pull: bool = True
            render_particle_bucket_write: bool = True
            render_visualizer_update: bool = True
            telemetry_gpu_pull: bool = True
            telemetry_aggregate: bool = True
            viewer_idle_tick: bool = True

        scene: Scene = Scene()
        simulator: Simulator = Simulator()
        rigid: Rigid = Rigid()
        teleop: Teleop = Teleop()
    configs: Configs = Configs()

    def __init__(self, **data):
        super().__init__(**data)
        object.__setattr__(self, 'profiler', Profiler(enabled=self.enabled))