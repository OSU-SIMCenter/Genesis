import argparse
import genesis as gs

# Import config dataclasses and parameters needed for the builder
from config import TrainingConfig
from config import CYLINDER_RADIUS, CYLINDER_HEIGHT, CYLINDER_POS, GENERATED_ROBOT_XML_PATH

# Import the builder class
from agforge_builder import RobotXMLGenerator

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
    cfg = TrainingConfig()

    # --- Step 2: Dynamically generate the robot XML ---
    print(f"Generating robot XML ('{GENERATED_ROBOT_XML_PATH}') from config parameters...")
    generator = RobotXMLGenerator(
        cylinder_radius=CYLINDER_RADIUS,
        cylinder_height=CYLINDER_HEIGHT,
        cylinder_pos=CYLINDER_POS,
        robot_cfg=cfg.robot
    )
    generator.write_to_file()

    # --- Step 3: Initialize Genesis and create environment ---
    gs.init(backend=gs.gpu, logging_level="info")
    
    from environment import AgilityForgeEnv
    from trainers import SACTrainer, AdamTrainer, BaseTrainer
    
    env = AgilityForgeEnv(cfg.sim, cfg.env, cfg.general, cfg.robot)

    # --- Step 4: Select trainer ---
    trainer: BaseTrainer
    if args.optimizer == "sac":
        trainer = SACTrainer(env, cfg.sac, cfg.general)
    elif args.optimizer == "adam":
        trainer = AdamTrainer(env, cfg.adam, cfg.general)
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")

    # --- Step 5: Run training ---
    print(f"Preparing to train with the '{args.optimizer}' optimizer.")
    env.start_recording()
    trainer.train()
    env.stop_recording()
    print("Pipeline execution finished.")

if __name__ == "__main__":
    main()