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
        scene: Scene = Scene()
    configs: Configs = Configs()

    def __init__(self, **data):
        super().__init__(**data)
        object.__setattr__(self, 'profiler', Profiler(enabled=self.enabled))