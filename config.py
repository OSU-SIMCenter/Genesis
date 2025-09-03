import torch

class SimConfig:
    """Simulation configuration parameters."""
    dt = 0.005
    substeps = 256
    gravity = (0, 0, 0)
    grid_density = 73
    lower_bound = (-0.4, -0.15, 0.15)
    upper_bound = (0.25, 0.15, 0.45)

class EnvConfig:
    """Environment configuration parameters."""
    num_envs = 6
    max_episode_length = 100
    action_duration_steps = 40
    reset_duration_steps = 23
    fixed_region_bounds = torch.tensor([[0.15, 0.25], [-0.1, 0.1], [0.2, 0.4]])
    target_shape_bounds = torch.tensor([[-0.01, 0.1, -0.01], [0.01, 0.5, 0.01]])
    action_lower_bounds = torch.tensor([-0.18, -40.0, 0.95])
    action_upper_bounds = torch.tensor([-0.12, 40.0, 0.1])

class SacConfig:
    """SAC optimizer configuration parameters."""
    class_name = "PPO"
    gamma = 0.99
    lam = 0.95
    learning_rate = 5e-4
    entropy_coef = 0.1
    actor_hidden_dims = [256, 128]
    critic_hidden_dims = [256, 128]
    max_iterations = 1
    run_name = "agforge_demo"
    runner_class_name = "OnPolicyRunner"
    num_steps_per_env = 1
    empirical_normalization = False
    save_interval = 50

class AdamConfig:
    """Adam optimizer configuration parameters."""
    learning_rate = 1e-3
    max_iterations = 1000

class GeneralConfig:
    """General configuration parameters."""
    show_viewer = True
    record = False
    log_dir = "logs/agforge_demo"
