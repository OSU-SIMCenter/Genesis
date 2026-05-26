import asyncio
import argparse
import sys
import yaml
import time
import math
import torch
import genesis as gs

from agforge.options import TeleopOptions
from agforge.agforge_builder import build_env
from agforge.strike_controller import StrikeController, StrikeState

def parse_args():
    parser = argparse.ArgumentParser(description="Run automated forging benchmarks from a recipe.")
    parser.add_argument("recipe", type=str, help="Path to the YAML recipe file")
    parser.add_argument("--viewer", action="store_true", help="Show the Genesis viewer")
    parser.add_argument("--softness", type=float, default=None, help="Override the gripper coup_softness (e.g. 1e-4)")
    return parser.parse_args()

def _find_next_strike(all_steps: list, current_idx: int) -> dict | None:
    """Look ahead in the recipe to find the next strike step after current_idx."""
    for i in range(current_idx + 1, len(all_steps)):
        if all_steps[i].get("type") == "strike":
            return all_steps[i]
    return None

async def execute_step(controller: StrikeController, step_cfg: dict, all_steps: list = None, step_index: int = 0):
    step_type = step_cfg.get("type")
    
    if step_type == "strike":
        target_strain = step_cfg.get("target_strain", 0.5)
        slider_pos = step_cfg.get("slider_position", None)
        hinge_rot = step_cfg.get("hinge_rotation", None)
        
        gs.logger.info(f"--- Benchmark Step: STRIKE (strain={target_strain}) ---")
        
        # Move to requested position if specified
        if slider_pos is not None or hinge_rot is not None:
            qpos = await controller.get_qpos()
            if slider_pos is not None:
                qpos[:, 0] = slider_pos
            if hinge_rot is not None:
                qpos[:, 1] = math.radians(hinge_rot)
            
            # Use TELEPORT to move instantly
            controller.robot.set_control_mode("TELEPORT")
            await controller.set_qpos(qpos)
            controller.robot.apply_action(qpos)
            await controller.step_simulation()
            
        await controller.trigger_strike(target_strain)
        
        # Run until strike completes and controller returns to IDLE
        while controller.strike_state != StrikeState.IDLE or controller.stabilization_steps > 0:
            await controller.step_simulation()
            if controller.stabilization_steps > 0:
                controller.stabilization_steps -= 1
                if controller.stabilization_steps == 0:
                    gs.logger.info("Stabilization complete")
            
        gs.logger.info(f"--- Benchmark Step: STRIKE complete ---")
        
    elif step_type == "heat":
        duration = step_cfg.get("duration_steps", 100)
        slider_pos = step_cfg.get("slider_position", None)
        
        # Auto-calculate: position the coil over the next strike's target area
        # coil_world_x = slider_x + coil_offset_x
        # To heat at strike_slider_x: slider_x = strike_slider_x - coil_offset_x
        if slider_pos is None and all_steps is not None:
            next_strike = _find_next_strike(all_steps, step_index)
            if next_strike is not None:
                strike_slider = next_strike.get("slider_position", 0.0)
                coil_offset = controller.env.cfg.robot.coil_offset_x
                slider_pos = strike_slider - coil_offset
                gs.logger.info(f"  Auto-calculated heating position: slider={slider_pos:.4f} (strike_pos={strike_slider}, coil_offset={coil_offset:.4f})")
            else:
                # No upcoming strike — center coil on billet origin
                coil_offset = controller.env.cfg.robot.coil_offset_x
                slider_pos = -coil_offset
        elif slider_pos is None:
            slider_pos = 0.0
        
        gs.logger.info(f"--- Benchmark Step: HEAT (duration={duration}, slider_pos={slider_pos:.4f}) ---")
        
        qpos = await controller.get_qpos()
        qpos[:, 0] = slider_pos
        controller.robot.set_control_mode("TELEPORT")
        await controller.set_qpos(qpos)
        controller.robot.apply_action(qpos)
        
        await controller.set_thermal_state(True)
        for _ in range(duration):
            await controller.step_simulation()
        await controller.set_thermal_state(False)
        
        gs.logger.info(f"--- Benchmark Step: HEAT complete ---")
        
    elif step_type == "cool":
        duration = step_cfg.get("duration_steps", 100)
        slider_pos = step_cfg.get("slider_position", None)
        
        # Auto-calculate: move billet to the next strike position to cool
        # Since the coil is offset, this guarantees the billet is OUT of the coil.
        if slider_pos is None and all_steps is not None:
            next_strike = _find_next_strike(all_steps, step_index)
            if next_strike is not None:
                slider_pos = next_strike.get("slider_position", 0.0)
                gs.logger.info(f"  Auto-calculated cooling position: slider={slider_pos:.4f} (next strike_pos)")
            else:
                slider_pos = 0.0
        elif slider_pos is None:
            slider_pos = 0.0
            
        gs.logger.info(f"--- Benchmark Step: COOL (duration={duration}, slider_pos={slider_pos:.4f}) ---")
        
        qpos = await controller.get_qpos()
        qpos[:, 0] = slider_pos
        controller.robot.set_control_mode("TELEPORT")
        await controller.set_qpos(qpos)
        controller.robot.apply_action(qpos)
        
        await controller.set_thermal_state(True)
        for _ in range(duration):
            await controller.step_simulation()
        await controller.set_thermal_state(False)
        
        gs.logger.info(f"--- Benchmark Step: COOL complete ---")
        
    elif step_type == "rotate":
        angle_deg = step_cfg.get("angle", 0.0)
        
        gs.logger.info(f"--- Benchmark Step: ROTATE (angle={angle_deg} deg) ---")
        
        qpos = await controller.get_qpos()
        qpos[:, 1] = math.radians(angle_deg)
        controller.robot.set_control_mode("TELEPORT")
        await controller.set_qpos(qpos)
        controller.robot.apply_action(qpos)
        await controller.step_simulation()
        
    else:
        gs.logger.warning(f"Unknown step type: {step_type}")

async def main():
    args = parse_args()
    
    with open(args.recipe, "r") as f:
        recipe = yaml.safe_load(f)
        
    print(f"Loaded recipe: {recipe.get('name', 'Unknown')}")
    
    cfg = TeleopOptions()
    cfg.general.show_viewer = args.viewer
    
    if args.softness is not None:
        cfg.robot.coup_softness = args.softness
        print(f"Overriding coup_softness to {args.softness:.2e}")
    
    # Enable MPM grid/boundary visualization
    cfg.vis.visualize_mpm_boundary = True
    cfg.vis.visualize_mpm_grid = True
    
    print("Building simulation environment...")
    env = build_env(cfg)
    
    shared_state = StrikeController(env)
    shared_state.robot.set_control_mode("TELEPORT")
    
    gs.logger.info("Warming up simulation...")
    await shared_state.reset_simulation()
    
    gs.logger.info("Warming up strike kernels...")
    shared_state.robot.set_control_mode("VELOCITY_CONTROL")
    vel_cmd = torch.zeros(4, device=env.device)
    vel_cmd[2] = 1.0; vel_cmd[3] = 1.0
    shared_state.robot.apply_velocity(vel_cmd, dofs_idx_local=torch.tensor([0, 1, 2, 3], device=env.device))
    env.scene.step()
    shared_state.robot.get_resistance_forces()
    shared_state.robot.set_control_mode("TELEPORT")
    await shared_state.reset_simulation()
    
    env.scene.profiling_options.profiler.reset()
    
    gs.logger.info("=== STARTING BENCHMARK RECIPE ===")
    start_time = time.time()
    
    steps = recipe.get("steps", [])
    for idx, step_cfg in enumerate(steps):
        gs.logger.info(f"Executing step {idx+1}/{len(steps)}")
        await execute_step(shared_state, step_cfg, all_steps=steps, step_index=idx)
        
    gs.logger.info("=== BENCHMARK COMPLETE ===")
    gs.logger.info(f"Total physical steps: {shared_state._physics_step_counter}")
    gs.logger.info(f"Wall time: {time.time() - start_time:.2f}s")
    
    if getattr(shared_state, 'recorder', None) and shared_state.recorder.is_recording:
        soft_tag = f" (softness: {args.softness:.2e})" if args.softness is not None else ""
        shared_state.recorder.flush_episode(
            success_flag=True, 
            language_instruction=f"Automated benchmark: {recipe.get('name')}{soft_tag}"
        )
        
    # Print profiling stats
    if cfg.print_profiling_on_exit:
        profiler = shared_state.env.scene.profiling_options.profiler
        print("\n--- Detailed Profiling Stats ---")
        profiler.rich_table(min_pct=0.0)

if __name__ == "__main__":
    asyncio.run(main())
