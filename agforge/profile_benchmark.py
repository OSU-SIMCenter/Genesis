import time
import json
import torch
import numpy as np
import genesis as gs
import sys
import os
import argparse

# Ensure we can import from local directory
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from options import TeleopOptions
from agforge_builder import build_env

def run_benchmark():
    # --- Configuration ---
    
    # "Atomic" unit of time.
    BASE_DT = 1.4e-6 
    
    # Total "Atomic Steps" for the entire test.
    # Scaled to 2048 to allow S=1024 to run at least 2 frames (1 Press, 1 Release)
    TOTAL_ATOMIC_STEPS = 2048 
    
    # Target frames for granular configs
    TARGET_FRAMES = 8 
    
    CONFIG = {
        "grid_densities": [64, 128], 
        "particle_multipliers": [1.0, 0.25], # 1.0=Sparse, 0.25=Dense (1/4 size ~ 64x? No, 8x? particles)
        
        # Wide Sweep including typical (32) and extreme (1024)
        "substeps_sweep": [32, 64, 128, 256, 512, 1024] 
    }
    
    results = []
    
    print(f"Starting Benchmark Sweep...")
    print(json.dumps(CONFIG, indent=2))
    print(f"Total Atomic Steps: {TOTAL_ATOMIC_STEPS}")
    print(f"Total Physical Time: {TOTAL_ATOMIC_STEPS * BASE_DT * 1000:.4f} ms")
    print("-" * 60)
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--visualize", action="store_true", help="Enable viewer to watch the benchmark running")
    args = parser.parse_args()

    # Import dependencies
    import struct
    import contextlib
    from genesis.utils import particle as pu
    import gstaichi as ti
    import itertools
    
    # Helper to mimic SharedState logic
    def mock_reconstruction(env, profiler):
        with profiler.time("teleop_recon"):
            solver = env.scene.sim.mpm_solver
            ti.sync()
            
            if hasattr(solver.particles_render.pos, 'to_numpy'):
                 particles = solver.particles_render.pos.to_numpy()
            else:
                 from genesis.utils.misc import ti_to_numpy
                 particles = ti_to_numpy(solver.particles_render.pos)

            if hasattr(particles, 'cpu'): particles = particles.cpu()
            if hasattr(particles, 'numpy'): particles = particles.numpy()

            particles = particles[:, 0]
            offset = np.zeros(3, dtype=np.float32)
            particles = particles + offset
            radius = solver.particle_radius
            
            mesh = pu.particles_to_mesh(
                positions=particles,
                radius=radius,
                backend='splashsurf'
            )
            return mesh, particles

    def mock_io(mesh, particles, profiler):
        with profiler.time("teleop_io"):
            v_flat = np.array(mesh.vertices).flatten().astype(np.float32)
            t_flat = np.array(mesh.faces).flatten().astype(np.int32)
            p_flat = particles.flatten().astype(np.float32)
            
            header = {
                "steps": [0],
                "Pressure": 0,
                "counts": {
                    "vertices": len(v_flat),
                    "faces": len(t_flat),
                    "particles": len(p_flat)
                }
            }
            header_json = json.dumps(header).encode('utf-8')
            binary_body = v_flat.tobytes() + t_flat.tobytes() + p_flat.tobytes()
            message = struct.pack('<I', len(header_json)) + header_json + binary_body
            return len(message)

    import glob

    from itertools import product
    
    scene_params = list(product(
        CONFIG["grid_densities"], 
        CONFIG["particle_multipliers"],
        CONFIG["substeps_sweep"]
    ))
    
    for grid_res, p_mult, substeps in scene_params:
        
        # Check if already run
        existing_pattern = os.path.join(current_dir, "benchmark_data", f"bench_G{grid_res}_P{p_mult}_{substeps}S_*.json")
        if glob.glob(existing_pattern):
            print(f"Skipping config (already exists): G={grid_res}, P={p_mult}, S={substeps}")
            continue

        # Calculate params for this run
        total_sim_loops = TOTAL_ATOMIC_STEPS // substeps
        
        # Dynamic Frame Logic
        # If we have very few loops (e.g. 2), we reduce frame count to match.
        measure_frames = min(TARGET_FRAMES, total_sim_loops)
        loops_per_frame = total_sim_loops // measure_frames
        
        # Calculated dt logic
        sim_dt = BASE_DT * substeps
        
        print(f"\n>>> Config: Grid={grid_res}, PMult={p_mult}, Substeps={substeps}")
        print(f"    SimDT={sim_dt:.2e}, Loops/Frame={loops_per_frame}, Frames={measure_frames}")

        cfg = TeleopOptions()
        cfg.general.show_viewer = args.visualize 
        cfg.profiling.enabled = True
        
        if args.visualize:
            cfg.vis.visualize_mpm_grid = True
            cfg.vis.visualize_mpm_boundary = True
            cfg.vis.show_world_frame = False
        
        cfg.robot.base_grid_density = grid_res
        cfg.mpm.grid_density = grid_res
        
        base_ref_size = 0.8 * 0.01 * 64.0 / grid_res
        cfg.mpm.particle_size = base_ref_size * p_mult
        
        # Configure Simulation
        cfg.sim.dt = sim_dt
        cfg.sim.substeps = substeps
        
        dx = 1.0 / grid_res
        mpm_solver_padding = 5 * dx 
        
        try:
            b0 = cfg.robot.target_shape_bounds[0]
            b1 = cfg.robot.target_shape_bounds[1]
            if hasattr(b0, 'cpu'): b0 = b0.cpu().numpy()
            elif not isinstance(b0, np.ndarray): b0 = np.array(b0)
            if hasattr(b1, 'cpu'): b1 = b1.cpu().numpy()
            elif not isinstance(b1, np.ndarray): b1 = np.array(b1)
            cfg.robot.target_shape_bounds = (b0, b1)
        except Exception:
             pass

        cylinder_height = cfg.robot.cylinder_height
        cylinder_radius = cfg.robot.cylinder_radius
        cylinder_pos = cfg.robot.cylinder_pos
        mpm_x_padding_lower = cylinder_height * 0.85
        mpm_x_padding_upper = cylinder_height * 0.52
        mpm_yz_padding = cylinder_radius * 1.6
        mpm_lower_offset = np.array([mpm_x_padding_lower, mpm_yz_padding, mpm_yz_padding]) + mpm_solver_padding
        mpm_upper_offset = np.array([mpm_x_padding_upper, mpm_yz_padding, mpm_yz_padding]) + mpm_solver_padding
        cfg.mpm.lower_bound = tuple(cylinder_pos - mpm_lower_offset)
        cfg.mpm.upper_bound = tuple(cylinder_pos + mpm_upper_offset)

        try:
            env = build_env(cfg)
            profiler = env.scene.profiling_options.profiler
            
            # --- Warmup ---
            print("  > Warmup...", end="", flush=True)
            env.robot.set_control_mode("TELEPORT")
            warmup_pos = torch.zeros(4, device=env.device)
            warmup_pos[2] = 0.015
            warmup_pos[3] = 0.015
            env.robot.apply_action(warmup_pos, dofs_idx_local=torch.tensor([0, 1, 2, 3], device=env.device))
            
            # Warmup: Run 1 loop iteration
            env.scene.step()
            if cfg.general.show_viewer:
                  if hasattr(env.scene.sim.mpm_solver, 'update_render_fields'): env.scene.sim.mpm_solver.update_render_fields()
                  else: env.scene.visualizer.update_visual_states()
            print(" Done.")
            
            # --- Benchmark ---
            profiler.reset()
            
            start_time = time.time()
            
            press_start = 0.015
            press_end = 0.022
            release_end = 0.018
            
            env.robot.set_control_mode("PD_CONTROL")
            pos_cmd = torch.zeros(4, device=env.device)

            print("  > Measuring: ", end="", flush=True)

            for i in range(measure_frames):
                print(f"{i+1}..", end="", flush=True)
                
                # Fractional Progress (0.0 to 1.0) through the sequence
                # We split 70% Press, 30% Release
                progress = i / max(1, measure_frames - 1) if measure_frames > 1 else 0.0
                
                with profiler.time("total_frame"):
                    
                    with profiler.time("teleop_logic"):
                            fL, fR = env.robot.get_resistance_forces()
                            
                            if progress <= 0.7:
                                # Press Phase (0.0 to 0.7 remapped to 0.0-1.0)
                                p_local = progress / 0.7
                                curr_target = press_start + (press_end - press_start) * p_local
                            else:
                                # Release Phase (0.7 to 1.0 remapped to 0.0-1.0)
                                p_local = (progress - 0.7) / 0.3
                                curr_target = press_end + (release_end - press_end) * p_local
                            
                            pos_cmd[2] = curr_target 
                            pos_cmd[3] = curr_target 
                            env.robot.apply_action(pos_cmd, dofs_idx_local=torch.tensor([0, 1, 2, 3], device=env.device))
                        
                    with profiler.time("teleop_physics_loop"):
                        for _ in range(loops_per_frame):
                            env.scene.step()
                    
                    if cfg.general.show_viewer:
                            if hasattr(env.scene.sim.mpm_solver, 'update_render_fields'):
                                env.scene.sim.mpm_solver.update_render_fields()
                            else:
                                env.scene.visualizer.update_visual_states()
                        
                    mesh, particles = mock_reconstruction(env, profiler)
                    bytes_sent = mock_io(mesh, particles, profiler)
                    
            print(" Done.")
            end_time = time.time()
            
            # Total Sequence Duration
            total_duration = end_time - start_time
            avg_fps = measure_frames / total_duration if total_duration > 0 else 0
            
            stats_dict = {}
            for name, stat in profiler.stats.items():
                stats_dict[name] = {
                    "count": stat.count,
                    "total": stat.total
                }
            
            n_particles = env.scene.sim.mpm_solver.n_particles
            
            data_point = {
                "config": {
                    "grid_res": grid_res, 
                    "particle_mult": p_mult,
                    "substeps": substeps,
                    "dt": sim_dt,
                    "loops_per_frame": loops_per_frame,
                    "measure_frames": measure_frames
                },
                "metrics": {
                    "n_particles": int(n_particles),
                    "total_sequence_time_ms": total_duration * 1000.0,
                    "avg_fps": avg_fps,
                },
                "cProfile_stats": stats_dict
            }
            results.append(data_point)
            
            print(f"  > Particles: {n_particles}")
            print(f"  > Total Simulation Time: {data_point['metrics']['total_sequence_time_ms']:.2f} ms")
            
            # Incremental Save
            timestamp = int(time.time())
            filename = f"bench_G{grid_res}_P{p_mult}_{substeps}S_{timestamp}.json"
            file_path = os.path.join(current_dir, "benchmark_data", filename)
            
            # Ensure directory exists (just in case)
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            with open(file_path, 'w') as f:
                json.dump(data_point, f, indent=2)
                
            print(f"  -> Saved result to: benchmark_data/{filename}")
            
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if 'env' in locals() and env is not None:
                if hasattr(env, 'scene') and env.scene is not None:
                    env.scene.destroy()

    # Legacy: we don't necessarily need the monolithic file anymore, 
    # but we can print a summary of how many were saved.
    print(f"\nBenchmark Sweep Complete. {len(results)} results generated.")
    print(f"Individual results saved to: {os.path.join(current_dir, 'benchmark_data')}")

if __name__ == "__main__":
    run_benchmark()
