import torch
import genesis as gs
import abc
from rsl_rl.runners import OnPolicyRunner

from environment import AgilityForgeEnv
from options import TrainingOptions

class BaseTrainer(abc.ABC):
    """Abstract base class for all training algorithms."""
    def __init__(self, env: AgilityForgeEnv, cfg: TrainingOptions):
        self.env = env
        self.cfg = cfg
        self.device = gs.device

    @abc.abstractmethod
    def train(self):
        """Runs the training process."""
        raise NotImplementedError

class SACTrainer(BaseTrainer):
    """Trains a policy using the SAC (PPO) algorithm provided by rsl_rl."""
    def __init__(self, env: AgilityForgeEnv, cfg: TrainingOptions):
        super().__init__(env, cfg)
        self.sac_cfg = cfg.sac

    def _create_train_config(self) -> dict:
        """Builds the configuration dictionary required by OnPolicyRunner."""
        return {
            "algorithm": {
                "class_name": self.sac_cfg.class_name,
                "gamma": self.sac_cfg.gamma,
                "lam": self.sac_cfg.lam,
                "learning_rate": self.sac_cfg.learning_rate,
                "entropy_coef": self.sac_cfg.entropy_coef,
            },
            "policy": {
                "class_name": "ActorCritic",
                "actor_hidden_dims": self.sac_cfg.actor_hidden_dims,
                "critic_hidden_dims": self.sac_cfg.critic_hidden_dims,
            },
            "runner": {
                "max_iterations": self.sac_cfg.max_iterations,
                "run_name": self.sac_cfg.run_name,
            },
            "save_interval": self.sac_cfg.save_interval,
            "runner_class_name": self.sac_cfg.runner_class_name,
            "num_steps_per_env": self.sac_cfg.num_steps_per_env,
            "empirical_normalization": self.sac_cfg.empirical_normalization,
        }

    def train(self):
        """Initializes and runs the OnPolicyRunner learning process."""
        print(f"Starting SAC (PPO) training for {self.sac_cfg.max_iterations} iterations...")
        train_cfg = self._create_train_config()
        runner = OnPolicyRunner(self.env, train_cfg, self.cfg.general.log_dir, device=self.device)
        runner.learn(num_learning_iterations=self.sac_cfg.max_iterations, init_at_random_ep_len=True)
        print("SAC (PPO) training complete.")

class AdamTrainer(BaseTrainer):
    """Performs gradient-based optimization of actions using the Adam optimizer."""
    def __init__(self, env: AgilityForgeEnv, cfg: TrainingOptions):
        super().__init__(env, cfg)
        self.adam_cfg = cfg.adam
        
        self.actions = torch.zeros(
            (self.env.max_episode_length, self.env.num_envs, self.env.num_actions),
            requires_grad=True,
            device=self.device
        )
        self.optimizer = torch.optim.Adam([self.actions], lr=self.adam_cfg.learning_rate)

    def train(self):
        """Runs the gradient-based optimization loop."""
        print(f"Starting Adam optimization for {self.adam_cfg.max_iterations} iterations...")
        for i in range(self.adam_cfg.max_iterations):
            self.optimizer.zero_grad()
            self.env.reset()
            
            total_reward = 0
            for t in range(self.env.max_episode_length):
                _, rewards, _, _ = self.env.step(self.actions[t])
                total_reward += rewards.mean()
            
            loss = -total_reward
            loss.backward()
            self.optimizer.step()
            
            if (i + 1) % 10 == 0:
                print(f"Iteration {i+1}/{self.adam_cfg.max_iterations}, Loss: {loss.item():.4f}")
        
        print("Adam optimization complete.")