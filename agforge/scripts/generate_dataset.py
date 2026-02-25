import asyncio
import time
import torch
import numpy as np
import genesis as gs
import argparse
from agforge.options import TeleopOptions
from agforge.agforge_builder import build_env
from agforge.strike_controller import StrikeController, StrikeState

async def simulate_strike(controller, target_strain=0.1, max_steps=1000):
    await controller.trigger_strike(target_strain)
    steps = 0
    while controller.strike_state != StrikeState.IDLE and steps < max_steps:
        await controller.step_simulation()
        steps += 1
    return steps

async def main():
    parser = argparse.ArgumentParser(description="Comprehensive V2.1 Data Recording Test")
    parser.add_argument("--visualize", action="store_true", help="Launch with Genesis renderer active")
    args = parser.parse_args()

    print("--- Starting Comprehensive V2.1 Data Recording Test ---")
    
    gs.init(backend=gs.gpu)
    cfg = TeleopOptions()
    cfg.general.show_viewer = args.visualize
    
    if not args.visualize:
        cfg.vis.visualize_mpm_boundary = False
        cfg.vis.visualize_mpm_grid = False
        cfg.performance_mode = True

    print("1. Initializing Simulation & Controller...")
    env = build_env(cfg)
    controller = StrikeController(env)
    
    # We will force the recorder to shard very early to test rolling files
    controller.recorder.shard_capacity = 2

    # === EPISODE 1: Basic Strikes & Undo ===
    print("\n[Episode 1] Starting basic strike test...")
    # reset_simulation() automatically starts a new episode in the recorder
    await controller.reset_simulation() 
    
    print("[Episode 1] Strike 1 (Target Strain: 0.1)...")
    await simulate_strike(controller, target_strain=0.1)

    print("[Episode 1] Strike 2 (Target Strain: 0.15) - THIS WILL BE UNDONE...")
    await simulate_strike(controller, target_strain=0.15)
    
    print("[Episode 1] Reverting Strike 2 (Testing memory branch rollback)...")
    await controller.load_checkpoint() # This should trigger recorder.handle_undo()

    print("[Episode 1] Strike 2 (Retry: Target Strain: 0.2)...")
    await simulate_strike(controller, target_strain=0.2)
    
    # Flush
    controller.recorder.flush_episode(success_flag=True, language_instruction="Strike the material twice successfully.")

    # === EPISODE 2: Ragged Data (Changing Particle Counts) ===
    print("\n[Episode 2] Testing Ragged Arrays via Reconstruction...")
    await controller.reset_simulation()
    
    print("[Episode 2] Strike 1 (Initial Particle Count)...")
    await simulate_strike(controller, target_strain=0.15)
    
    print("[Episode 2] Simulating brief delay...")
    # Let the simulation settle briefly
    # Let the simulation settle briefly
    for _ in range(50):
        await controller.step_simulation()

    print("[Episode 2] Strike 2 (New Particle Count)...")
    await simulate_strike(controller, target_strain=0.15)

    controller.recorder.flush_episode(success_flag=True, language_instruction="Strike, reconstruct, and strike again.")

    # === EPISODE 3: Shard Rollover Test ===
    print("\n[Episode 3] Testing Shard capacity rollover...")
    await controller.reset_simulation()
    print("[Episode 3] Strike 1...")
    await simulate_strike(controller, target_strain=0.1)
    
    # Since shard_capacity is 2, flushing Episode 3 should trigger a new file (shard_0001.h5)
    controller.recorder.flush_episode(success_flag=False, language_instruction="Trigger shard rollover.")

    print("\n--- Simulation Complete. Checking generated files... ---")
    env.scene.destroy()

if __name__ == "__main__":
    asyncio.run(main())
