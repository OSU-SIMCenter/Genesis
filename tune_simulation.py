
import asyncio
import time
import torch
import numpy as np
import genesis as gs
import pandas as pd
import argparse
from agforge.options import TeleopOptions, RobotOptions, ureg
from agforge.agforge_builder import build_env
from agforge.strike_controller import StrikeController, StrikeState


async def run_episode(env, controller, cfg, visualize):
    """
    Runs a single simulation episode (Reset -> Strike -> Completion) on an existing environment.
    """
    # Warmup
    await controller.reset_simulation()
    
    # Trigger Strike
    print("Triggering Strike...")
    # Use default target strain from options
    await controller.trigger_strike(cfg.strike.target_strain)
    
    total_work = 1e-6
    energy_ratios = []
    
    # Run until IDLE
    start_time = time.time()
    # Use timeout from options
    max_time = cfg.strike.approaching_timeout + cfg.strike.pressing_timeout + 10.0
    
    while controller.strike_state != StrikeState.IDLE:
        if time.time() - start_time > max_time:
            print("Timeout!")
            break
            
        # Step Logic & Physics
        await controller.step_simulation()
        
        # Render
        if visualize and env.scene.visualizer:
                if hasattr(env.scene.visualizer, 'render'):
                    env.scene.visualizer.render()
                elif hasattr(env.scene.visualizer, 'viewer') and hasattr(env.scene.visualizer.viewer, 'render'):
                    env.scene.visualizer.viewer.render()
        
        # Analyze
        vel = env.mpm_entity.get_state().vel
        v_sq = torch.sum(vel ** 2).item()
        mpm_mass = cfg.mat.rho * 3.8e-5 # Approx total mass
        N = vel.shape[1]
        ke = 0.5 * (mpm_mass / N) * v_sq
        
        force_L, force_R = controller.robot.get_resistance_forces()
        f_mag = (torch.norm(force_L) + torch.norm(force_R)).item()
        # Work = Force * distance. Approximate instantaneous work rate = Force * velocity.
        # But we need accumulated work.
        # Pressing speed is constant? No.
        speed = cfg.strike.pressing_speed if controller.strike_state == StrikeState.PRESSING else cfg.strike.approach_speed
    
        work_inc = f_mag * speed * cfg.mpm.dt
        total_work += work_inc
        
        # Always track KE/Work ratio if we have work
        if total_work > 1e-4:
            ratio = ke / total_work
            energy_ratios.append(ratio)
        else:
            ratio = 0.0

        # Print status every ~10 sim steps
        if not hasattr(run_episode, "step_counter"): run_episode.step_counter = 0
        run_episode.step_counter += 1
        
        if run_episode.step_counter % 10 == 0:
             print(f"State: {controller.strike_state.name} | Force: {f_mag:.1f} | KE: {ke:.4f} | Ratio: {ratio:.2%}")

    avg_ratio = np.mean(energy_ratios) if energy_ratios else 1.0
    print(f"Episode Ratio: {avg_ratio:.2%}")
    return avg_ratio

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--visualize", action="store_true", help="Enable viewer")
    parser.add_argument("--best-only", action="store_true", help="Run only best config")
    parser.add_argument("--once", action="store_true", help="Run single episode (alias for --loops 1)")
    parser.add_argument("--loops", type=int, default=0, help="Number of loops (0=infinite)")
    args = parser.parse_args()

    gs.init(backend=gs.gpu)
    
    # Load defaults (Reverted to baseline)
    cfg = TeleopOptions()
    cfg.general.show_viewer = args.visualize
    if args.visualize:
        cfg.vis.visualize_mpm_boundary = True
        cfg.vis.visualize_mpm_grid = True
    
    print("--- Running with Default Parameters (Manual Tuning Mode) ---")
    print(f"Rho: {cfg.mat.rho}")
    print(f"J-C Enabled: {getattr(cfg.mat, 'use_johnson_cook', False)}")
    print(f"DT: {cfg.sim.dt}")
    print(f"Speeds: Approach={cfg.strike.approach_speed}, Press={cfg.strike.pressing_speed}")
    print(f"Timeouts: Approach={cfg.strike.approaching_timeout}, Press={cfg.strike.pressing_timeout}")
    
    env = None
    try:
        env = build_env(cfg)
        controller = StrikeController(env)
        
        loop_count = 0
        while True:
            await run_episode(env, controller, cfg, args.visualize)
            loop_count += 1
            if args.loops > 0 and loop_count >= args.loops:
                break
            if args.once: # Backward compatibility or alias
                break
            print("Looping... (Ctrl+C to stop)")
            
    except Exception as e:
        print(f"Simulation Failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if env:
            env.scene.destroy()

if __name__ == "__main__":
    asyncio.run(main())
