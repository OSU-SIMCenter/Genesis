
import asyncio
import time
import torch
import numpy as np
import genesis as gs
import pandas as pd
import argparse
from agforge.options import TeleopOptions
from agforge.agforge_builder import build_env
from agforge.strike_controller import StrikeController, StrikeState

async def run_episode(speed, visualize=False):
    """
    Runs a single simulation episode with a specific approach speed.
    """
    print(f"--- Running Approach Speed: {speed:.1f} m/s ---")
    
    # Configure options
    cfg = TeleopOptions()
    cfg.general.show_viewer = visualize
    cfg.strike.approach_speed = float(speed)
    # User Request: Hold Simulation after contact (for observation)
    cfg.strike.pressing_speed = 0.0 
    cfg.strike.target_strain = 0.5 
    cfg.strike.pressing_timeout = 10.0 # 10s wall time is enough for ~200-500 steps (instability shows up fast)
    cfg.strike.force_balance_gain = 5e-5
    
    # Performance/Vis flags
    if not visualize:
        cfg.vis.visualize_mpm_boundary = False
        cfg.vis.visualize_mpm_grid = False
        cfg.performance_mode = True # Use performance mode if not visualizing
    else:
        cfg.vis.visualize_mpm_boundary = True
        cfg.vis.visualize_mpm_grid = True
        cfg.performance_mode = False

    env = None
    results = {}
    
    try:
        env = build_env(cfg)
        controller = StrikeController(env)
        
        # Reset
        await controller.reset_simulation()
        
        # Trigger Strike (Straight Hit only for speed test)
        await controller.trigger_strike(cfg.strike.target_strain)
        
        start_time = time.time()
        max_time_steps = 2000 # 2000 steps is plenty for 10s wall time check
        steps = 0
        max_force = 0.0
        max_particle_vel = 0.0
        
        while controller.strike_state != StrikeState.IDLE:
            # Step
            await controller.step_simulation()
            steps += 1
            
            # Monitor Max Force
            f_L, f_R = controller.robot.get_resistance_forces()
            f_max_curr = max(torch.norm(f_L).item(), torch.norm(f_R).item())
            if f_max_curr > max_force:
                max_force = f_max_curr
                
            # Monitor Particle Velocity (CFL Check)
            # Access MPM entity state
            mpm_state = env.mpm_entity.get_state() 
            # Note: get_state() might be slow? It returns a named tuple with tensors.
            # Only do this if we suspect issues or for benchmark?
            # It should be fast enough (just pointer access usually).
            vels = mpm_state.vel # (N, 3)
            # Compute max velocity magnitude
            # Using torch.max(torch.norm(vels, dim=1))
            # Optimization: norm is expensive, check Squared Norm first?
            # but overhead is small compared to sim step
            v_max_curr = torch.max(torch.norm(vels, dim=1)).item()
            if v_max_curr > max_particle_vel:
                max_particle_vel = v_max_curr
                
            # Render
            if visualize and env.scene.visualizer:
                 if hasattr(env.scene.visualizer, 'render'):
                     env.scene.visualizer.render()
                 elif hasattr(env.scene.visualizer, 'viewer') and hasattr(env.scene.visualizer.viewer, 'render'):
                     env.scene.visualizer.viewer.render()
            
            if steps > max_time_steps:
                print("Timeout!")
                break
        
        end_time = time.time()
        wall_time = end_time - start_time
        
        # Calc Stats
        dt = cfg.sim.dt
        dx_grid = 1.0 / cfg.robot.base_grid_density
        
        # CFL based on APPROACH speed
        cfl_approach = (speed * dt) / dx_grid
        
        # CFL based on MAX PARTICLE speed (True CFL)
        cfl_particle = (max_particle_vel * dt) / dx_grid
        
        return {
            "speed": speed,
            "wall_time": wall_time,
            "steps": steps,
            "max_force": max_force,
            "max_p_vel": max_particle_vel,
            "cfl_app": cfl_approach,
            "cfl_part": cfl_particle,
            "status": "Success" if steps <= max_time_steps else "Timeout"
        }

    except Exception as e:
        print(f"Failed: {e}")
        return {
            "speed": speed, 
            "wall_time": 0, 
            "status": f"Crash: {e}"
        }
    finally:
        if env:
            env.scene.destroy()

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--speeds", type=str, default="100,110,120,130,140,150,160,170,180,190,200", help="Comma separated speeds")
    parser.add_argument("--visualize", action="store_true")
    args = parser.parse_args()

    gs.init(backend=gs.gpu)
    
    speeds = [float(s) for s in args.speeds.split(",")]
    all_results = []
    
    for speed in speeds:
        res = await run_episode(speed, args.visualize)
        all_results.append(res)
    
    # Print Table
    df = pd.DataFrame(all_results)
    print("\n=== Speed Benchmark Results ===")
    print(df.to_string(index=False))

if __name__ == "__main__":
    asyncio.run(main())
