import argparse
import genesis as gs

# Import config dataclasses and parameters needed for the builder
from options import TrainingOptions
from agforge_builder import build_env
from trainers import SACTrainer, AdamTrainer, BaseTrainer

def main():
    """
    Main entry point for the AgilityForge training pipeline.
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

    # --- Step 1: Load configurations ---
    cfg = TrainingOptions()

    # --- Step 2: Create environment ---
    env = build_env(cfg)

    # --- Step 3: Select trainer ---
    trainer: BaseTrainer
    if args.optimizer == "sac":
        trainer = SACTrainer(env, cfg)
    elif args.optimizer == "adam":
        trainer = AdamTrainer(env, cfg)
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")

    # --- Step 4: Run training ---
    print(f"Preparing to train with the '{args.optimizer}' optimizer.")
    env.start_recording()
    trainer.train()
    env.stop_recording()
    print("Pipeline execution finished.")

if __name__ == "__main__":
    main()