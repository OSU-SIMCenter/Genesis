import argparse
import genesis as gs

from config import SimConfig, EnvConfig, GeneralConfig, SacConfig, AdamConfig
from environment import AgilityForgeEnv
from trainers import SACTrainer, AdamTrainer, BaseTrainer

def main():
    """
    Main entry point for the AgilityForge training pipeline.

    Initializes the simulation environment and runs the selected training
    algorithm (SAC or Adam) based on command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Train a robotic forging task with a selectable optimizer.")
    parser.add_argument(
        "--optimizer", 
        type=str, 
        default="sac", 
        choices=["sac", "adam"],
        help="The optimizer to use for training ('sac' for RL or 'adam' for gradient descent)."
    )
    args = parser.parse_args()

    gs.init(backend=gs.gpu, logging_level="info")

    sim_cfg = SimConfig()
    env_cfg = EnvConfig()
    general_cfg = GeneralConfig()

    env = AgilityForgeEnv(sim_cfg, env_cfg, general_cfg)

    trainer: BaseTrainer
    if args.optimizer == "sac":
        sac_cfg = SacConfig()
        trainer = SACTrainer(env, sac_cfg, general_cfg)
    elif args.optimizer == "adam":
        adam_cfg = AdamConfig()
        trainer = AdamTrainer(env, adam_cfg, general_cfg)
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")

    print(f"Preparing to train with the '{args.optimizer}' optimizer.")
    env.start_recording()
    trainer.train()
    env.stop_recording()
    print("Pipeline execution finished.")

if __name__ == "__main__":
    main()