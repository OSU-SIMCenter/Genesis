import time
import json
import sys
import torch
import numpy as np
import genesis as gs
import os
import argparse
import copy

try:
    import rerun as rr
except ImportError:
    rr = None

from agforge.options import TeleopOptions
from agforge.agforge_builder import build_env
from agforge.reconstruction import SurfaceReconstructor, SamplingMethod

# Current directory for file path operations
current_dir = os.path.dirname(os.path.abspath(__file__))


def run_reconstruction_benchmark():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=25, help="Simulation frames to run")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup frames (not timed)")
    parser.add_argument("--visualize", action="store_true", help="Show Genesis viewer")
    parser.add_argument("--rerun", action="store_true", help="Log visualization to Rerun")
    parser.add_argument("--save-meshes", action="store_true", help="Save meshes to disk")
    args = parser.parse_args()

    print("--- Starting Surface Reconstruction Benchmark ---")
    
    if args.rerun:
        if rr is None:
            print("ERROR: Rerun SDK not found.")
            sys.exit(1)
        rr.init("surface_reconstruction_benchmark", spawn=True)

    configs = [
        {"name": "Hybrid_128", "grid_res": 128, "backend": "hybrid", "fraction": 1.0, "offset": [0.0, 0.0, 0.0]},
        {"name": "SplashSurf", "grid_res": 128, "backend": "splashsurf", "fraction": 1.0, "offset": [0.0, 0.0, 0.04]},
        # {"name": "Hybrid_64", "grid_res": 64, "backend": "hybrid", "fraction": 1.0, "offset": [0.0, 0.0, 0.0]},
    ]
    
    results = {}

    cfg = TeleopOptions()
    cfg.general.show_viewer = args.visualize 
    cfg.profiling.enabled = False
    
    for conf in configs:
        name = conf["name"]
        offset = np.array(conf["offset"], dtype=np.float32)
        print(f"\n>>> Benchmarking: {name}")
        
        try:
            env_cfg = copy.deepcopy(cfg)
            env = build_env(env_cfg)
            
            reconstructor = SurfaceReconstructor(
                env, 
                grid_res=conf.get("grid_res", 128),
                backend=conf.get("backend", 'hybrid')
            )
            reconstructor.recon_enabled = True
            reconstructor.recon_frame_interval = 1 
            reconstructor.recon_particle_fraction = conf["fraction"]
            reconstructor.sampling_method = SamplingMethod.VOXEL_STRATIFIED
            
            print(f"  > Sampling: {reconstructor.sampling_method.value}, Fraction: {conf['fraction']}, Skinning: {conf.get('skinning', False)}")
            
            env.robot.set_control_mode("TELEPORT")
            warmup_pos = torch.zeros(4, device=env.device)
            warmup_pos[2] = 0.015 
            warmup_pos[3] = 0.015
            env.robot.apply_action(warmup_pos, dofs_idx_local=torch.tensor([0, 1, 2, 3], device=env.device))
            env.scene.step()
            
            env.robot.set_control_mode("PD_CONTROL")
            pos_cmd = torch.zeros(4, device=env.device)
            
            press_start = 0.015
            press_end = 0.022   
            release_end = 0.018 
            loops_per_frame = 10
            
            t_start_init = time.time()
            reconstructor.create_reconstructed_mesh()
            if conf.get("skinning", False):
                reconstructor.init_skinning()
            t_init = time.time() - t_start_init
            vertex_count = len(reconstructor.reconstructed_mesh.vertices)
            print(f"  > Init: {t_init*1000:.2f} ms, Vertices: {vertex_count}")
            
            print(f"  > Warmup ({args.warmup} frames)...")
            for _ in range(args.warmup):
                pos_cmd[2] = press_start
                pos_cmd[3] = press_start
                env.robot.apply_action(pos_cmd, dofs_idx_local=torch.tensor([0, 1, 2, 3], device=env.device))
                for _ in range(loops_per_frame):
                    env.scene.step()
                reconstructor.update(should_reconstruct=True)
            
            update_times = []
            total_frames = args.steps
            print(f"  > Running {total_frames} timed frames...")
            
            for i in range(total_frames):
                if args.rerun:
                    rr.set_time("step", sequence=i)
                
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
                
                t0 = time.time()
                reconstructor.update(should_reconstruct=True)
                dt_ms = (time.time() - t0) * 1000.0
                update_times.append(dt_ms)
                
                if args.rerun:
                    try:
                        rr.log(f"performance/{name}_ms", rr.Scalars([dt_ms]))
                    except AttributeError:
                        rr.log(f"performance/{name}_ms", dt_ms)
                    
                    mesh = reconstructor.reconstructed_mesh
                    if len(mesh.vertices) > 0:
                        rr.log(
                            f"visuals/{name}/mesh",
                            rr.Mesh3D(
                                vertex_positions=mesh.vertices + offset,
                                vertex_normals=mesh.vertex_normals,
                                triangle_indices=mesh.faces, 
                                albedo_factor=[0.39, 0.78, 1.0, 0.5]
                            )
                        )
                    
                    # Log particles (visualize only, use cache if available)
                    # Use the cache from reconstruction to avoid re-fetching
                    particles = reconstructor.get_active_particle_cache()
                    if particles is None:
                        # Fallback if cache is empty (e.g. skinning skipped)
                        particles = reconstructor._get_active_particles(use_cache=False)

                    if particles is not None and len(particles) > 0:
                        # Convert to numpy if tensor
                        if isinstance(particles, torch.Tensor):
                            p_np = particles.cpu().numpy()
                        else:
                            p_np = particles
                            
                        rr.log(
                            f"visuals/{name}/particles",
                            rr.Points3D(
                                p_np + offset,
                                radii=0.0002,
                                colors=[0.8, 0.2, 0.2]
                            )
                        )
                
                if args.save_meshes:
                    vis_dir = os.path.join(current_dir, "benchmark_data", "vis_output", name)
                    os.makedirs(vis_dir, exist_ok=True)
                    reconstructor.reconstructed_mesh.export(os.path.join(vis_dir, f"mesh_{i:04d}.obj"))

            results[name] = {
                "init_ms": t_init * 1000.0,
                "avg_ms": np.mean(update_times),
                "std_ms": np.std(update_times),
                "min_ms": np.min(update_times),
                "max_ms": np.max(update_times),
                "p50_ms": np.percentile(update_times, 50),
                "p95_ms": np.percentile(update_times, 95),
                "vertices": vertex_count,
                "fraction": conf["fraction"],
                "skinning": conf.get("skinning", False),
            }
            
            print(f"  > Result: {results[name]['avg_ms']:.2f} ± {results[name]['std_ms']:.2f} ms")
            env.scene.destroy()
            
        except Exception as e:
            print(f"  ERROR in {name}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*85)
    print(f"{'Config':<25} | {'Init':<8} | {'Avg±Std':<15} | {'P95':<8} | {'Verts':<8} | {'Speedup':<8}")
    print("-" * 85)
    
    base_time = results.get("Full_Reconstruction", {}).get("avg_ms", 1)
    
    for name, m in results.items():
        speedup = base_time / m["avg_ms"] if m["avg_ms"] > 0 else 0
        print(f"{name:<25} | {m['init_ms']:<8.1f} | {m['avg_ms']:.2f}±{m['std_ms']:<6.2f} ms | {m['p95_ms']:<8.2f} | {m['vertices']:<8} | {speedup:<8.1f}x")
    
    print("="*85)
    
    results_path = os.path.join(current_dir, "benchmark_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    run_reconstruction_benchmark()