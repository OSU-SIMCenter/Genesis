import time
import json
import torch
import numpy as np
import genesis as gs
import sys
import os
import trimesh
import argparse
import copy

# Optional Rerun dependency
try:
    import rerun as rr
except ImportError:
    rr = None

# Ensure we can import from local directory
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from options import TeleopOptions
from agforge_builder import build_env
from reconstruction import SurfaceReconstructor

def run_reconstruction_benchmark():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=25, help="Simulation frames to run")
    parser.add_argument("--visualize", action="store_true", help="Show Genesis viewer")
    parser.add_argument("--rerun", action="store_true", help="Log visualization to Rerun")
    parser.add_argument("--save-meshes", action="store_true", help="Save meshes to disk (legacy method)")
    args = parser.parse_args()

    print("--- Starting Surface Reconstruction Benchmark ---")
    
    if args.rerun:
        if rr is None:
            print("ERROR: Rerun SDK not found.")
            print("Please run with 'pixi run -e dev python ...' or install rerun-sdk.")
            sys.exit(1)
        rr.init("surface_reconstruction_benchmark", spawn=True)

    # Define test configurations
    # We add an offset to visualize them side-by-side in Rerun
    configs = [
        {"name": "Full_Reconstruction", "skinning": False, "fraction": 1.0, "offset": [0.0, 0.0, 0.0]},
        {"name": "Incremental_Skinning", "skinning": True, "fraction": 1.0, "offset": [0.0, 0.0, 0.06]},
        {"name": "Downsampled_Skinning", "skinning": True, "fraction": 0.5, "offset": [0.0, 0.0, 0.12]},
    ]
    
    results = {}

    # Common Teleop Options
    cfg = TeleopOptions()
    cfg.general.show_viewer = args.visualize 
    cfg.profiling.enabled = False
    
    # Run loop for each config
    for conf in configs:
        name = conf["name"]
        offset = np.array(conf["offset"], dtype=np.float32)
        print(f"\n>>> Benchmarking: {name}")
        
        try:
            # Re-build environment to ensure clean state
            env_cfg = copy.deepcopy(cfg)
            if args.visualize:
                env_cfg.vis.visualize_mpm_grid = True 
            
            env = build_env(env_cfg)
            
            # Setup Reconstructor
            reconstructor = SurfaceReconstructor(env)
            reconstructor.recon_enabled = True
            reconstructor.recon_frame_interval = 1 
            reconstructor.recon_particle_fraction = conf["fraction"]
            reconstructor.skinning_enabled = False 
            
            # Warmup
            print("  > Warmup (Teleport)...")
            env.robot.set_control_mode("TELEPORT")
            warmup_pos = torch.zeros(4, device=env.device)
            warmup_pos[2] = 0.015 
            warmup_pos[3] = 0.015
            env.robot.apply_action(warmup_pos, dofs_idx_local=torch.tensor([0, 1, 2, 3], device=env.device))
            env.scene.step()
            
            # Switch to PD Control
            env.robot.set_control_mode("PD_CONTROL")
            pos_cmd = torch.zeros(4, device=env.device)
            
            # Sim Params
            press_start = 0.015
            press_end = 0.022   
            release_end = 0.018 
            
            loops_per_frame = 10 # Increased from 2 to see more deformation per frame
            total_frames = args.steps
            
            print(f"  > Running {total_frames} frames...")
            if args.visualize:
                print("  > [Genesis] Viewer enabled. Check the window!")
            
            # Metrics
            update_times = []
            
            # Initial Mesh Gen
            t_start_init = time.time()
            reconstructor.create_reconstructed_mesh()
            if conf["skinning"]:
                reconstructor.init_skinning()
            t_init = time.time() - t_start_init
            print(f"  > Init Time: {t_init*1000:.2f} ms")
            vertex_count = len(reconstructor.reconstructed_mesh.vertices)
            
            for i in range(total_frames):
                if args.rerun:
                    rr.set_time("step", sequence=i)
                
                # Sim Logic
                # Aggressive press to ensure visible deformation
                progress = i / max(1, total_frames - 1)
                
                if progress <= 0.7:
                     p_local = progress / 0.7
                     curr_target = press_start + (press_end - press_start) * p_local
                else:
                     p_local = (progress - 0.7) / 0.3
                     curr_target = press_end + (release_end - press_end) * p_local
                
                pos_cmd[2] = curr_target 
                pos_cmd[3] = curr_target 
                env.robot.apply_action(pos_cmd, dofs_idx_local=torch.tensor([0, 1, 2, 3], device=env.device))
                
                for _ in range(loops_per_frame):
                    env.scene.step()
                
                if args.visualize:
                    if hasattr(env.scene.sim.mpm_solver, 'update_render_fields'):
                         env.scene.sim.mpm_solver.update_render_fields()
                    else:
                         env.scene.visualizer.update_visual_states()
                
                # Measure
                t0 = time.time()
                reconstructor.update(should_reconstruct=True)
                t1 = time.time()
                dt_ms = (t1 - t0) * 1000.0
                update_times.append(dt_ms)
                
                # RERUN LOGGING
                if args.rerun:
                    # Log performance metric
                    # In 0.29.0, use rr.Scalars (plural) or just pass the value if supported
                    try:
                        rr.log(f"performance/{name}_ms", rr.Scalars([dt_ms]))
                    except AttributeError:
                         # Fallback for older/newer APIs
                        rr.log(f"performance/{name}_ms", dt_ms)
                    
                    # Log Mesh
                    # We apply the visual offset to vertices so they appear side-by-side
                    mesh = reconstructor.reconstructed_mesh
                    if len(mesh.vertices) > 0:
                        shifted_verts = mesh.vertices + offset
                        rr.log(
                            f"visuals/{name}/mesh",
                            rr.Mesh3D(
                                vertex_positions=shifted_verts,
                                vertex_normals=mesh.vertex_normals,
                                triangle_indices=mesh.faces, 
                                albedo_factor=[0.39, 0.78, 1.0] # Light Blue (0-1 range)
                            )
                        )
                        
                    # Log Particles (Optional, minimal overhead compared to mesh)
                    parts = reconstructor._get_active_particles()
                    if parts is not None:
                         shifted_parts = parts + offset
                         rr.log(
                             f"visuals/{name}/particles",
                             rr.Points3D(
                                 shifted_parts,
                                 radii=0.0005, # Smaller particles
                                 colors=[255, 100, 100] # Red 
                             )
                         )
                
                # Legacy Disk Save
                if args.save_meshes:
                    vis_dir = os.path.join(current_dir, "benchmark_data", "vis_output", name)
                    os.makedirs(vis_dir, exist_ok=True)
                    mesh_path = os.path.join(vis_dir, f"mesh_{i:04d}.obj")
                    reconstructor.reconstructed_mesh.export(mesh_path)

            # Stats
            avg_time = np.mean(update_times)
            max_time = np.max(update_times)
            min_time = np.min(update_times)
            
            results[name] = {
                "init_ms": t_init * 1000.0,
                "avg_ms": avg_time,
                "max_ms": max_time,
                "vertices": vertex_count
            }
            
            print(f"  > Result: Avg {avg_time:.2f} ms")
            env.scene.destroy()
            
        except Exception as e:
            print(f"  ERROR in {name}: {e}")
            import traceback
            traceback.print_exc()

    # --- Final Report ---
    print("\n" + "="*60)
    print(f"{'Method/Config':<30} | {'Init (ms)':<10} | {'Avg Update (ms)':<15} | {'Speedup':<10}")
    print("-" * 60)
    
    base_time = results.get("Full_Reconstruction", {}).get("avg_ms", 0)
    
    for name, metrics in results.items():
        if base_time > 0:
            speedup = base_time / metrics["avg_ms"] if metrics["avg_ms"] > 0 else 0.0
        else:
            speedup = 0.0
        print(f"{name:<30} | {metrics['init_ms']:<10.2f} | {metrics['avg_ms']:<15.2f} | {speedup:<10.1f}x")
    print("="*60)

if __name__ == "__main__":
    run_reconstruction_benchmark()
