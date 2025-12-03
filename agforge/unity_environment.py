from environment_gym import AgilityForgeGymEnv
from options import AgilityForgeOptions, TeleopOptions
import numpy as np

import torch
from typing import Tuple

def transform_points(
    points: torch.Tensor,
    translation: Tuple[float, float, float],
    scale: Tuple[float, float, float],
    flip_axis: int
) -> torch.Tensor:
    points = points + torch.tensor(translation, device=points.device).view(1, 1, 3)
    points = points * torch.tensor(scale, device=points.device).view(1, 1, 3)
    points[:, :, flip_axis] *= -1
    return points

class UnityEnvironment(AgilityForgeGymEnv):
    def __init__(self, cfg: AgilityForgeOptions=TeleopOptions()):
        super().__init__(cfg)
        self.billet = self.env.mpm_entity

    def do_press(self):
        # currently identical to the update function, will change once presses are implemented
        vertices = self.get_particles()
        print("Got genesis result!")
        return {
            "Vertices": vertices.flatten().tolist(),
            "Steps": [0],
            "Temperatures": np.full(len(vertices), 293.0, dtype=float).tolist(),
            "Pressure": 0, # my stand in pressure value goes from 0 to 100
                            # feel free to change this, but message Jonah if you do
            "StressField": -1,
        }
    
    def temperature_result(self, count):
        return{
            "Temperatures": np.full(count, 293.0, dtype=float).tolist(),
            "Times": [10]
        }
    
    def update(self, translation_x=0.0, euler_x=0.0):
        if not action: action = np.zeros(self.action_space.shape)
        self.step(action=action)
        vertices = self.get_particles()
        print("Got genesis result!")
        return {
            "Vertices": vertices.flatten().tolist(),
            "Steps": [0],
            "Temperatures": np.full(len(vertices), 293.0, dtype=float).tolist(),
            "Pressure": 0, # my stand in pressure value goes from 0 to 100
                            # feel free to change this, but message Jonah if you do
            "StressField": -1,
        }
    
    def get_particles(self):
        return transform_points(
            points=self.billet.get_particles_pos(envs_idx=0),
            translation=-(self.cfg.robot.cylinder_pos + np.array([-0.375 * self.cfg.robot.cylinder_height, 0, 0])),
            scale=(50.0,)*3,
            flip_axis=0
        )