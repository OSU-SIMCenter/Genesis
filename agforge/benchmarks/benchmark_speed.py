
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

async def run_episode(env, controller, cfg, speed, mode="approach", visualize=False):
    """
    Runs a single simulation episode with a specific speed (approach or pressing).
    Reuses existing environment and controller.
    """
    print(f"--- Running {mode.capitalize()} Speed: {speed:.1f} m/s ---")
    
    # Update Configuration In-Place
    if mode == "approach":
        cfg.strike.approach_speed = float(speed)
        cfg.strike.pressing_speed = 0.0 
    elif mode == "pressing":
        # Ensure approach speed is set to default (parametric) if not already
        # Ensure approach speed is set to default (parametric)
        temp_opts = TeleopOptions() # This computes the default parametric speed
        cfg.strike.approach_speed = temp_opts.strike.approach_speed
        cfg.strike.pressing_speed = float(speed)
    
    cfg.strike.target_strain = 0.5 
    cfg.strike.pressing_timeout = 10.0
    
    # CRITICAL FIX: Ensure the environment/controller actually sees the updated config
    # cfg is passed by reference, so env.cfg should be the same object.
    # But let's verify and force update if needed (though Python objs are ref).
    # Printing to confirm what the controller sees.

    
    results = {}
    
    try:
        # Reset Simulation (Fast)
        await controller.reset_simulation()
        
        # Trigger Strike
        await controller.trigger_strike(cfg.strike.target_strain)
        
        start_time = time.time()
        max_time_steps = 2000
        steps = 0
        max_force = 0.0
        max_particle_vel = 0.0
        
        # Physics Metrics
        total_work = 1e-6
        energy_ratios = [] # Overall
        
        # Segmented Metrics
        max_p_vel_overall = 0.0
        max_p_vel_pressing = 0.0
        max_p_vel_steady = 0.0
        
        max_ratio_overall = 0.0
        max_ratio_pressing = 0.0
        max_ratio_steady = 0.0
        
        mpm_mass_approx = cfg.mat.rho * 3.8e-5 
        
        pressing_step_count = 0
        STEADY_THRESHOLD = 30 # Steps to ignore after contact for "Steady" (Shock dissipation)

        while controller.strike_state != StrikeState.IDLE:
            # Step
            await controller.step_simulation()
            steps += 1
            
            if controller.strike_state == StrikeState.PRESSING:
                pressing_step_count += 1
            
            # Monitor Max Force
            f_L, f_R = controller.robot.get_resistance_forces()
            f_mag_pair = max(torch.norm(f_L).item(), torch.norm(f_R).item())
            f_total_mag = (torch.norm(f_L) + torch.norm(f_R)).item()

            if f_mag_pair > max_force:
                max_force = f_mag_pair
                
            # Monitor Particle Velocity (CFL Check & KE)
            mpm_state = env.mpm_entity.get_state() 
            vels = mpm_state.vel # Might be (N, 3) or (1, N, 3)
            
            # Handle Batch Dim
            if vels.dim() == 3:
                vels = vels.squeeze(0)
            
            # Max Vel (CFL)
            v_max_curr = torch.max(torch.norm(vels, dim=1)).item()
            
            # Update Max PV Metrics
            if v_max_curr > max_p_vel_overall:
                max_p_vel_overall = v_max_curr
                
            if controller.strike_state == StrikeState.PRESSING:
                if v_max_curr > max_p_vel_pressing:
                    max_p_vel_pressing = v_max_curr
                if pressing_step_count > STEADY_THRESHOLD:
                    if v_max_curr > max_p_vel_steady:
                        max_p_vel_steady = v_max_curr
            
            # KE Ratio
            v_sq = torch.sum(vels ** 2).item()
            N = vels.shape[0]
            ke = 0.5 * (mpm_mass_approx / N) * v_sq
            
            current_speed = 0.0
            if controller.strike_state == StrikeState.APPROACHING:
                current_speed = cfg.strike.approach_speed
            elif controller.strike_state == StrikeState.PRESSING:
                current_speed = cfg.strike.pressing_speed
            
            work_inc = f_total_mag * current_speed * cfg.sim.dt
            total_work += work_inc
            
            ratio = 0.0
            if total_work > 1e-4:
                ratio = ke / total_work
                energy_ratios.append(ratio)
                
                # Update Max Ratio Metrics
                if ratio > max_ratio_overall:
                    max_ratio_overall = ratio
                
                if controller.strike_state == StrikeState.PRESSING:
                    if ratio > max_ratio_pressing:
                        max_ratio_pressing = ratio
                    if pressing_step_count > STEADY_THRESHOLD:
                         if ratio > max_ratio_steady:
                             max_ratio_steady = ratio

            # Logging every 3 steps (matches StrikeController)
            if steps % 3 == 0:
                 print(f"BM Step {steps}: State={controller.strike_state.name}, Force={f_total_mag:.1f}, KE={ke:.4f}, Ratio={ratio:.2%}, MaxPVel={v_max_curr:.1f}")

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
        
        relevant_speed = speed
        cfl_cmd = (relevant_speed * dt) / dx_grid
        cfl_particle = (max_p_vel_overall * dt) / dx_grid
        
        mean_ratio = np.mean(energy_ratios) if energy_ratios else 0.0
        
        return {
            "mode": mode,
            "speed": speed,
            "wall_time": wall_time,
            "steps": steps,
            "max_force": f"{max_force:.0f}",
            "pv_max": f"{max_p_vel_overall:.1f}",
            "pv_press": f"{max_p_vel_pressing:.1f}",
            "pv_stable": f"{max_p_vel_steady:.1f}",
            "cfl_cmd": f"{cfl_cmd:.2f}",
            "cfl_part": f"{cfl_particle:.2f}",
            "mean_ratio": f"{mean_ratio:.2%}",
            "r_max": f"{max_ratio_overall:.2%}",
            "r_press": f"{max_ratio_pressing:.2%}",
            "r_stable": f"{max_ratio_steady:.2%}",
            "status": "Success" if steps <= max_time_steps else "Timeout"
        }

    except Exception as e:
        print(f"Failed: {e}")
        return {
            "mode": mode,
            "speed": speed, 
            "wall_time": 0, 
            "status": f"Crash: {e}"
        }

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--speeds", type=str, default="100,110,120,130,140,150,160,170,180,190,200", help="Comma separated speeds")
    parser.add_argument("--mode", type=str, default="approach", choices=["approach", "pressing"], help="Benchmark mode")
    parser.add_argument("--loops", type=int, default=1, help="Number of loops per speed")
    parser.add_argument("--visualize", action="store_true")
    args = parser.parse_args()

    gs.init(backend=gs.gpu)
    
    # Initialize Environment ONCE
    print("Initializing Environment...")
    cfg = TeleopOptions()
    cfg.general.show_viewer = args.visualize
    
    if not args.visualize:
        cfg.vis.visualize_mpm_boundary = False
        cfg.vis.visualize_mpm_grid = False
        cfg.performance_mode = True
    else:
        cfg.vis.visualize_mpm_boundary = True
        cfg.vis.visualize_mpm_grid = True
        cfg.performance_mode = False
        
    env = build_env(cfg)
    controller = StrikeController(env)
    
    speeds = [float(s) for s in args.speeds.split(",")]
    all_results = []
    
    try:
        for speed in speeds:
            for i in range(args.loops):
                print(f"Loop {i+1}/{args.loops}")
                res = await run_episode(env, controller, cfg, speed, args.mode, args.visualize)
                res['loop'] = i + 1
                all_results.append(res)
    finally:
        if env:
            env.scene.destroy()
    
    # Print Table
    df = pd.DataFrame(all_results)
    # Reorder columns to put loop next to speed
    cols = list(df.columns)
    if 'loop' in cols:
        cols.insert(2, cols.pop(cols.index('loop')))
        df = df[cols]
        
    print("\n=== Speed Benchmark Results ===")
    print(df.to_string(index=False))

if __name__ == "__main__":
    asyncio.run(main())
