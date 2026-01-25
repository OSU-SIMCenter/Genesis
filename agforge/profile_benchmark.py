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
    CONFIG = {
        # Grid resolutions (cells per unit).
        "grid_densities": [64, 80], # Faster default
        
        # Particle size multipliers
        "particle_multipliers": [0.5, 1.0], 
        
        # Total physics substeps per 'Frame'.
        # Visualizer uses 32 loops of 32 substeps = 1024 total.
        # We will match this exactly.
        "substeps_per_call": 32, 
        "manual_loops": 32,
        
        "warmup_frames": 3,
        "measure_frames": 10
    }
    
    results = []
    
    print(f"Starting Benchmark Sweep...")
    print(json.dumps(CONFIG, indent=2))
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

    from itertools import product
    combinations = list(product(
        CONFIG["grid_densities"], 
        CONFIG["particle_multipliers"]
    ))
    
    # We removed Stepping Mode "Internal/External" because we rely on the Hybrid (Visualizer) approach for correctness.
    # It is effectively "Internal" with manual loops.
    
    for grid_res, p_mult in combinations:
        print(f"\n>>> Running Config: Grid={grid_res}, PMult={p_mult}")
        
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
        
        # Match Visualizer exactly
        cfg.sim.substeps = CONFIG["substeps_per_call"] # 32
        loop_iterations = CONFIG["manual_loops"] # 32
        
        # Total Substeps = 32 * 32 = 1024
            
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
            
            for _ in range(CONFIG["warmup_frames"]):
                for _ in range(loop_iterations):
                    env.scene.step()
                    if cfg.general.show_viewer:
                         if hasattr(env.scene.sim.mpm_solver, 'update_render_fields'): env.scene.sim.mpm_solver.update_render_fields()
                         else: env.scene.visualizer.update_visual_states()
            print(" Done.")
            
            # --- Benchmark ---
            profiler.reset()
            
            measure_frames = CONFIG["measure_frames"]
            
            start_time = time.time()
            sim_t = 0.0
            dt = cfg.sim.dt * (cfg.sim.substeps * loop_iterations)
            
            press_start = 0.015
            press_end = 0.022
            press_frames = 7
            
            release_end = 0.018
            release_frames = 3
            
            env.robot.set_control_mode("PD_CONTROL")
            pos_cmd = torch.zeros(4, device=env.device)

            print("  > Measuring: ", end="", flush=True)

            for i in range(measure_frames):
                print(f"{i+1}..", end="", flush=True)
                
                with profiler.time("total_frame"):
                    
                    with profiler.time("teleop_logic"):
                            fL, fR = env.robot.get_resistance_forces()
                            
                            if i < press_frames:
                                p = i / (press_frames - 1) if press_frames > 1 else 1.0
                                curr_target = press_start + (press_end - press_start) * p
                            else:
                                rel_i = i - press_frames
                                p = rel_i / (release_frames)
                                curr_target = press_end + (release_end - press_end) * p
                            
                            pos_cmd[2] = curr_target 
                            pos_cmd[3] = curr_target 
                            env.robot.apply_action(pos_cmd, dofs_idx_local=torch.tensor([0, 1, 2, 3], device=env.device))
                        
                    with profiler.time("teleop_physics_loop"):
                        for _ in range(loop_iterations):
                            env.scene.step()
                    
                    sim_t += dt

                    if cfg.general.show_viewer:
                            if hasattr(env.scene.sim.mpm_solver, 'update_render_fields'):
                                env.scene.sim.mpm_solver.update_render_fields()
                            else:
                                env.scene.visualizer.update_visual_states()
                        
                    mesh, particles = mock_reconstruction(env, profiler)
                    bytes_sent = mock_io(mesh, particles, profiler)
                    
            print(" Done.")
            end_time = time.time()
            total_duration = end_time - start_time
            avg_fps = measure_frames / total_duration
            
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
                    "stepping_mode": "hybrid",
                    "substeps_per_call": cfg.sim.substeps,
                    "loop_iterations": loop_iterations
                },
                "metrics": {
                    "n_particles": int(n_particles),
                    "avg_frame_time_ms": (total_duration / measure_frames) * 1000.0,
                    "avg_fps": avg_fps,
                },
                "cProfile_stats": stats_dict
            }
            results.append(data_point)
            
            print(f"  > Particles: {n_particles}")
            print(f"  > FPS: {avg_fps:.2f}")
            print(f"  > Avg Frame Time: {data_point['metrics']['avg_frame_time_ms']:.2f} ms")
            
            env.scene.destroy()
            
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    output_path = os.path.join(current_dir, "profiling_results.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_path}")

if __name__ == "__main__":
    run_benchmark()
