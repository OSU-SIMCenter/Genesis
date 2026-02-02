import asyncio
import time
import json
import os
import copy
import argparse
import itertools
import numpy as np
import genesis as gs
import torch

from agforge.options import TeleopOptions
from agforge.agforge_builder import build_env
from agforge.strike_controller import StrikeController, StrikeState

def get_current_time_ms():
    return time.time() * 1000.0

async def run_benchmark():
    parser = argparse.ArgumentParser()
    parser.add_argument("--visualize", action="store_true", help="Enable viewer")
    parser.add_argument("--output_dir", default="benchmark_data", help="Directory to save results")
    args = parser.parse_args()

    results_dir = os.path.join(os.path.dirname(__file__), args.output_dir)
    os.makedirs(results_dir, exist_ok=True)
    
    # --- Benchmark Configuration ---

    # 1. Fetch Defaults from TeleopOptions for "Production" Config
    default_opts = TeleopOptions()
    default_grid_density = default_opts.robot.base_grid_density
    default_substeps = default_opts.sim.substeps
    
    # "Hard" params require rebuilding the environment
    HARD_CONFIGS = [
        # Exact Teleop Settings
        {"name": "Production", "grid_density": default_grid_density, "particle_multiplier": 1.0, "substeps": default_substeps},
        
        # Variations
        {"name": "LowRes", "grid_density": int(default_grid_density * 0.5), "particle_multiplier": 1.0, "substeps": default_substeps},
    ]
    
    # "Soft" params can be applied at runtime via reset
    SOFT_CONFIGS = [
        {"name": "Default_Press", "speed": 1.0, "force_param": 0.5}, # 50% Strain
        # {"name": "High_Force_Press", "speed": 1.0, "force_param": 0.8}, 
    ]

    results = []
    
    # 1. Outer Loop: Hard Configs (Env Rebuild)
    for hard_cfg in HARD_CONFIGS:
        print(f"\n[Hard Config] {hard_cfg['name']} -> Grid: {hard_cfg['grid_density']}, Substeps: {hard_cfg['substeps']}")
        
        # Build Env
        cfg = TeleopOptions()
        
        # Force CPU for bounds to prevent "can't convert cuda:0 to numpy" error in gs.morphs
        if isinstance(cfg.robot.target_shape_bounds, torch.Tensor):
             cfg.robot.target_shape_bounds = cfg.robot.target_shape_bounds.cpu()
        if isinstance(cfg.robot.fixed_region_bounds, torch.Tensor):
             cfg.robot.fixed_region_bounds = cfg.robot.fixed_region_bounds.cpu()
        if isinstance(cfg.robot.action_lower_bounds, torch.Tensor):
             cfg.robot.action_lower_bounds = cfg.robot.action_lower_bounds.cpu()
        if isinstance(cfg.robot.action_upper_bounds, torch.Tensor):
             cfg.robot.action_upper_bounds = cfg.robot.action_upper_bounds.cpu()

        cfg.general.show_viewer = args.visualize
        cfg.profiling.enabled = True
        
        # Enable MPM grid/boundary visualization when viewer is active
        if args.visualize:
            cfg.vis.visualize_mpm_boundary = True
            cfg.vis.visualize_mpm_grid = True
        
        # Force enable granular profiling
        cfg.profiling.configs.scene.step.sim = True
        cfg.profiling.configs.scene.step.visualizer = True
        cfg.profiling.configs.scene.step.fps_tracker = True
        
        # Simulator options
        for field in cfg.profiling.configs.simulator.__dict__:
             if not field.startswith('_'): setattr(cfg.profiling.configs.simulator, field, True)

        # Rigid options
        for field in cfg.profiling.configs.rigid.__dict__:
             if not field.startswith('_'): setattr(cfg.profiling.configs.rigid, field, True)

        # Teleop options
        for field in cfg.profiling.configs.teleop.__dict__:
             if not field.startswith('_'): setattr(cfg.profiling.configs.teleop, field, True)

        cfg.robot.base_grid_density = hard_cfg["grid_density"]
        cfg.mpm.grid_density = hard_cfg["grid_density"]
        cfg.sim.substeps = hard_cfg["substeps"]
        cfg.sim.dt = 1.4e-6 * hard_cfg["substeps"] 
        
        # Adjust particle size
        base_ref_size = 0.8 * 0.01 * 64.0 / hard_cfg["grid_density"]
        cfg.mpm.particle_size = base_ref_size * hard_cfg["particle_multiplier"]

        # Recalculate MPM bounds
        dx = 1.0 / hard_cfg["grid_density"]
        mpm_solver_padding = 3 * dx
        
        cyl_h = cfg.robot.cylinder_height
        cyl_r = cfg.robot.cylinder_radius
        cyl_pos = cfg.robot.cylinder_pos
        
        mpm_x_padding_lower = cyl_h * 0.85
        mpm_x_padding_upper = cyl_h * 0.52
        mpm_yz_padding = cyl_r * 1.6
        
        mpm_lower_offset = np.array([mpm_x_padding_lower, mpm_yz_padding, mpm_yz_padding]) + mpm_solver_padding
        mpm_upper_offset = np.array([mpm_x_padding_upper, mpm_yz_padding, mpm_yz_padding]) + mpm_solver_padding
        
        cfg.mpm.lower_bound = tuple(cyl_pos - mpm_lower_offset)
        cfg.mpm.upper_bound = tuple(cyl_pos + mpm_upper_offset)

        try:
            env = build_env(cfg)
            controller = StrikeController(env)
            
            # 2. Inner Loop: Soft Configs (Controller Reset)
            for soft_cfg in SOFT_CONFIGS:
                print(f"  > [Soft Config] {soft_cfg['name']} (Strain: {soft_cfg['force_param']:.2f})")
                
                # Apply Soft Params
                # Use scalar from config to modulate the default options
                # (Fetch fresh defaults in case they were modified)
                _temp_defaults = TeleopOptions()
                base_approach = _temp_defaults.strike.approach_speed
                base_pressing = _temp_defaults.strike.pressing_speed
                
                env.cfg.strike.approach_speed = base_approach * soft_cfg["speed"]
                env.cfg.strike.pressing_speed = base_pressing * soft_cfg["speed"]
                
                # Reset
                await controller.reset_simulation()
                
                # Warmup
                controller.env.scene.step()
                
                # Trigger Strike
                await controller.trigger_strike(soft_cfg["force_param"])
                
                # Run Loop until IDLE
                start_time = time.time()
                step_count = 0
                profiler = env.scene.profiling_options.profiler
                profiler.reset()
                
                max_steps = 5000 
                
                while controller.strike_state != StrikeState.IDLE and step_count < max_steps:
                    # Root Frame
                    with profiler.time("teleop_step"):
                        # Logic + Physics + Render
                        await controller.step_simulation()
                        
                        # Reconstruction
                        await controller.update_and_get_recon_data()
                        
                        # Update Viewer Context explicitly if visualizing
                        # Update Viewer Context explicitly if visualizing
                        if args.visualize and env.scene.visualizer:
                             if hasattr(env.scene.visualizer, 'render'):
                                 env.scene.visualizer.render()
                             elif hasattr(env.scene.visualizer, 'viewer') and hasattr(env.scene.visualizer.viewer, 'render'):
                                 env.scene.visualizer.viewer.render()
                    
                    step_count += 1
                    
                    if step_count % 100 == 0:
                         pass

                duration = time.time() - start_time
                fps = step_count / duration if duration > 0 else 0
                
                print(f"    Done: {step_count} steps in {duration:.2f}s ({fps:.1f} FPS)")
                
                # Detailed Profiler Output (Rich Table)
                print("\n    --- Detailed Profiling Stats (Rich Table - Full) ---")
                profiler.rich_table(min_pct=0.0)
                
                # Detailed Profiler Output (ASCII Tree)
                print("\n    --- Detailed Profiling Hierarchy (ASCII Tree - >2%) ---")
                profiler.print_tree(min_pct=2.0)
                
                # Detailed Profiler Output (Flat Hot Spots)
                print("\n    --- Profiling Hot-Spots (Flat - >1.5%) ---")
                profiler.print_flat(sort_by="self", min_pct=1.5)
                print("    --------------------------------\n")

                # Collect Metrics
                run_data = {
                    "timestamp": int(time.time()),
                    "config_name": f"{hard_cfg['name']}_{soft_cfg['name']}",
                    "hard_config": hard_cfg,
                    "soft_config": soft_cfg,
                    "metrics": {
                        "steps": step_count,
                        "duration_sec": duration,
                        "fps": fps,
                        "final_state": controller.strike_state.name
                    },
                    "profiler": {}
                }
                
                # Extract profiler stats (backward compatibility map)
                for name, stat in profiler.stats.items():
                    run_data["profiler"][name] = {
                        "count": stat.count,
                        "total": stat.total,
                        "mean": stat.mean,  
                        "std": stat.std,    
                        "min": stat.min,    
                        "max": stat.max,    
                        "avg": stat.mean
                    }
                    
                results.append(run_data)
                
                # Incremental Save - Standard JSON
                fname = f"result_{hard_cfg['name']}_{soft_cfg['name']}_{run_data['timestamp']}.json"
                with open(os.path.join(results_dir, fname), "w") as f:
                    json.dump(run_data, f, indent=2)

                # Save Speedscope Profile
                profile_fname = f"profile_{hard_cfg['name']}_{soft_cfg['name']}_{run_data['timestamp']}.speedscope.json"
                profiler.save_speedscope(os.path.join(results_dir, profile_fname))

        except Exception as e:
            print(f"Error running config {hard_cfg}: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            if 'env' in locals() and env:
                env.scene.destroy()

    print(f"\nBenchmark Complete. Saved {len(results)} runs to {results_dir}")

if __name__ == "__main__":
    asyncio.run(run_benchmark())
