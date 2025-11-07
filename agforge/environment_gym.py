import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch

from options import AgilityForgeOptions, TrainingOptions
from agforge_builder import build_env
from environment import AgilityForgeEnv


class AgilityForgeGymEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, cfg: AgilityForgeOptions):
        super().__init__()
        
        cfg.env.num_envs = 1
        cfg.general.show_viewer = False
        
        print("Building single-instance AgilityForge environment for Gym...")
        self.env: AgilityForgeEnv = build_env(cfg)
        self.cfg = self.env.cfg
        self.device = self.env.device
        print("Gym wrapper initialized.")
        
        low = self.cfg.env.action_lower_bounds.cpu().numpy()
        high = self.cfg.env.action_upper_bounds.cpu().numpy()
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)

        obs_dim = self.env.num_obs
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(obs_dim,), 
            dtype=np.float32
        )

    def step(self, action: np.ndarray):
        # 1. Translate (CPU/Numpy) -> (GPU/Torch/Batched)
        actions_tensor = torch.tensor(action, device=self.device, dtype=torch.float32).unsqueeze(0)

        # 2. Step the underlying environment
        obs_batch, rewards_batch, dones_batch, extras = self.env.step(actions_tensor)

        # 3. Translate (GPU/Torch/Batched) -> (CPU/Numpy/Single)
        obs_np = obs_batch[0].cpu().numpy()
        reward = float(rewards_batch[0].item())
        done = bool(dones_batch[0].item())
        
        terminated = False
        truncated = done
        info = {} 

        return obs_np, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        if seed is not None:
            super().reset(seed=seed)
            
        obs_batch, _ = self.env.reset()
        obs_np = obs_batch[0].cpu().numpy()
        info = {}

        return obs_np, info

    def close(self):
        self.env.scene.close()
        print("AgilityForgeGymEnv closed.")