
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

async def run_episode_with_gain(gain, visualize=False):
    """
    Runs a single simulation episode with a specific force balance gain.
    Returns statistics on force difference (dF).
    """
    # Configure options
    cfg = TeleopOptions()
    cfg.general.show_viewer = visualize
    cfg.strike.force_balance_gain = gain
    
    # Disable heavy visualization features for speed if not visualizing
    if not visualize:
        cfg.vis.visualize_mpm_boundary = False
        cfg.vis.visualize_mpm_grid = False
    else:
        cfg.vis.visualize_mpm_boundary = True
        cfg.vis.visualize_mpm_grid = True
        
    print(f"--- Running Gain: {gain:.1e} ---")
    
    env = None
    dF_history = []
    
    try:
        env = build_env(cfg)
        controller = StrikeController(env)
        
        # Warmup
        await controller.reset_simulation()
        
        # Sequence: 1. Straight Hit, 2. Angled Hit (15 degrees)
        hits = [
            {"angle": 0.0, "desc": "Straight"},
            {"angle": 30.0, "desc": "Angled"}
        ]
        
        for hit_idx, hit in enumerate(hits):
            print(f"  > Hit {hit_idx+1}: {hit['desc']} (Angle {hit['angle']} deg)")
            
            # Rotate Hinge if needed (Only during IDLE)
            qpos = await controller.get_qpos()
            qpos[0, 1] = np.deg2rad(hit['angle'])
            await controller.set_qpos(qpos)
            # Use entity to set qpos instead of set_dofs_position (which crashed comparison)
            controller.robot.entity.set_qpos(qpos)
            
            # Trigger Strike
            await controller.trigger_strike(cfg.strike.target_strain)
            
            start_time = time.time()
            max_time = cfg.strike.approaching_timeout + cfg.strike.pressing_timeout + 10.0
            
            while controller.strike_state != StrikeState.IDLE:
                if time.time() - start_time > max_time:
                    print("Timeout!")
                    break
                    
                # Step
                await controller.step_simulation()
                
                # Record Data only during PRESSING
                if controller.strike_state == StrikeState.PRESSING:
                    force_L, force_R = controller.robot.get_resistance_forces()
                    f_L_mag = torch.norm(force_L).item()
                    f_R_mag = torch.norm(force_R).item()
                    dF = abs(f_L_mag - f_R_mag)
                    dF_history.append(dF)
                
                if visualize and env.scene.visualizer:
                     if hasattr(env.scene.visualizer, 'render'):
                         env.scene.visualizer.render()
                     elif hasattr(env.scene.visualizer, 'viewer') and hasattr(env.scene.visualizer.viewer, 'render'):
                         env.scene.visualizer.viewer.render()
            
            # Wait a bit between hits? 
            # Controller goes to IDLE automatically.
            # We skip explicit wait, just proceed to next hit.
            
        # Compile Stats
        if dF_history:
            dF_mean = np.mean(dF_history)
            dF_max = np.max(dF_history)
            dF_std = np.std(dF_history)
            dF_final = dF_history[-1]
            return {
                "gain": gain,
                "mean_dF": dF_mean,
                "max_dF": dF_max,
                "std_dF": dF_std,
                "final_dF": dF_final,
                "steps": len(dF_history),
                "status": "Success"
            }
        else:
            return {"gain": gain, "status": "NoData"}

    except Exception as e:
        print(f"Failed: {e}")
        return {"gain": gain, "status": "Failed"}
    finally:
        if env:
            env.scene.destroy()

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gains", type=str, default="0.1e-4,1.0e-4,2.0e-4,5.0e-4", help="Comma separated gains")
    parser.add_argument("--visualize", action="store_true")
    args = parser.parse_args()

    gs.init(backend=gs.gpu)
    
    gains = [float(g) for g in args.gains.split(",")]
    results = []
    
    for gain in gains:
        res = await run_episode_with_gain(gain, args.visualize)
        results.append(res)
        # Re-init Genesis? No, just destroy scene is enough usually.
        # But Genesis might need full restart for clean options? 
        # SimOptions are passed to Scene. Scene usage is isolated? 
        # Yes, as long as we destroy scene.

    # Print Table
    df = pd.DataFrame(results)
    print("\n=== Force Balance Benchmark Results ===")
    print(df.to_string(index=False))
    
    # Analyze Best
    if not df.empty and 'mean_dF' in df.columns:
        valid_runs = df.dropna(subset=['mean_dF'])
        if not valid_runs.empty:
            best = valid_runs.loc[valid_runs['mean_dF'].idxmin()]
            print(f"\nBest Gain based on Mean dF: {best['gain']:.1e} (Mean dF={best['mean_dF']:.1f})")
        else:
            print("\nNo valid runs completed.")
    else:
        print("\nNo data collected.")

if __name__ == "__main__":
    asyncio.run(main())
