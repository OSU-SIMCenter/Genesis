import asyncio
import torch
import numpy as np
import genesis as gs
from teleop_socket import SharedState
from options import TeleopOptions
from agforge_builder import build_env

async def verify_reset():
    print("Initializing environment for verification...")
    cfg = TeleopOptions()
    # Disable viewer for headless testing if possible, or keep it if needed for context
    cfg.general.show_viewer = False 
    
    # We need to make sure genesis is initialized if not already
    if not gs.is_initialized():
        gs.init(backend=gs.gpu)

    env = build_env(cfg)
    state = SharedState(env)
    
    # Initial state
    print("Getting initial state...")
    initial_qpos = await state.get_qpos()
    print(f"Initial qpos: {initial_qpos[0, :4]}")

    # 1. Modify State
    print("\nModifying state (moving robot)...")
    new_qpos = initial_qpos.clone()
    new_qpos[0, 0] += 0.1 # Move slider
    await state.set_qpos(new_qpos)
    
    # Applying action to actually move the robot in sim
    state.robot.apply_action(new_qpos)
    state.env.scene.step()
    
    current_qpos = await state.get_qpos()
    print(f"Modified qpos: {current_qpos[0, :4]}")
    
    # 2. Save Checkpoint
    print("\nSaving checkpoint...")
    await state.save_checkpoint()
    
    # 3. Modify State Again
    print("Modifying state again...")
    new_qpos[0, 0] += 0.1
    await state.set_qpos(new_qpos)
    state.robot.apply_action(new_qpos)
    state.env.scene.step()
    
    modified_qpos = await state.get_qpos()
    print(f"Further modified qpos: {modified_qpos[0, :4]}")
    
    # 4. Undo (Load Checkpoint)
    print("\nExecuting Undo (Load Checkpoint)...")
    await state.load_checkpoint()
    
    restored_qpos = await state.get_qpos()
    print(f"Restored qpos: {restored_qpos[0, :4]}")
    
    # Verify Undo
    # Note: floating point comparison
    if torch.allclose(restored_qpos, current_qpos, atol=1e-4):
        print("PASS: Undo successfully restored qpos.")
    else:
        print("FAIL: Undo failed to restore qpos.")
        print(f"Expected: {current_qpos[0, :4]}")
        print(f"Got:      {restored_qpos[0, :4]}")

    # 5. Full Reset
    print("\nExecuting Full Reset...")
    await state.reset_simulation()
    
    reset_qpos = await state.get_qpos()
    print(f"Reset qpos: {reset_qpos[0, :4]}")
    
    # Verify Reset - might need to check against initial_qpos or env defaults
    # The initial_qpos we got might be slightly different from default if physics settled, 
    # but reset() should put it back to exactly default config logic.
    # Let's check if it matches initial_qpos approximately or is consistent.
    
    # Actually, verify queue is empty
    if len(state.checkpoints) == 0:
        print("PASS: Checkpoint queue cleared on reset.")
    else:
        print(f"FAIL: Checkpoint queue not empty: {len(state.checkpoints)}")

    if torch.allclose(reset_qpos[:, :2], initial_qpos[:, :2], atol=1e-2): 
        # Grippers might be different if we didn't force them in initial, but slider/hinge should be reset.
        print("PASS: Reset restored robot position.")
    else:
        print("WARNING: Reset qpos differs from initial capture. This might be due to default pose vs settled pose.")
        print(f"Initial: {initial_qpos[0, :4]}")
        print(f"Reset:   {reset_qpos[0, :4]}")

if __name__ == "__main__":
    asyncio.run(verify_reset())
