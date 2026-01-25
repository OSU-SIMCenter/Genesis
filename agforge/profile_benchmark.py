import time
import json
import torch
import numpy as np
import genesis as gs
import sys
import os

# Ensure we can import from local directory
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from options import TeleopOptions
from agforge_builder import build_env

def run_benchmark():
    # --- Configuration ---
    # Customize these ranges as needed
    CONFIG = {
        # Grid resolutions (cells per unit, roughly). Higher = finer grid.
        "grid_densities": [100, 200], 
        
        # Particle size multipliers relative to grid cell size.
        # size = (1.0 / grid_density) * multiplier
        # Lower multiplier = smaller particles = MORE particles.
        "particle_multipliers": [0.25, 0.5, 1.0], 
        
        # Total physics substeps per 'Frame'.
        # We will compare doing these all-at-once (Internal) vs one-by-one (External).
        "total_substeps_target": 32,
        
        # Stepping Modes:
        # "internal": sim.substeps = total_target. env.step() called once.
        # "external": sim.substeps = 1. env.step() called total_target times.
        "stepping_modes": ["internal", "external"],
        
        "warmup_frames": 3,
        "measure_frames": 10
    }
    
    results = []
    
    print(f"Starting Benchmark Sweep...")
    print(json.dumps(CONFIG, indent=2))
    print("-" * 60)

    # Import dependencies for Logic/Recon/IO
    import struct
    import contextlib
    from genesis.utils import particle as pu
    import gstaichi as ti
    import itertools
    
    # Helper to mimic SharedState logic
    def mock_reconstruction(env, profiler):
        # Based on TeleopSocket logic
        with profiler.time("teleop_recon"):
            solver = env.scene.sim.mpm_solver
            # Sync needed for accurate timing of GPU ops
            # gs.tools.run_in_thread(ti.sync)
            ti.sync()
            
            # Access particles (expensive copy to CPU usually)
            if hasattr(solver.particles_render.pos, 'to_numpy'):
                 particles = solver.particles_render.pos.to_numpy()
            else:
                 from genesis.utils.misc import ti_to_numpy
                 particles = ti_to_numpy(solver.particles_render.pos)

            # Ensure particles is numpy CPU (handle Tensor case)
            if hasattr(particles, 'cpu'): particles = particles.cpu()
            if hasattr(particles, 'numpy'): particles = particles.numpy()

            # Slice batch dim
            particles = particles[:, 0]
            
            # offset = env.scene.envs_offset[0]
            # Hardcoding offset to unblock profiling - performance impact is negligible
            offset = np.zeros(3, dtype=np.float32)
            
            particles = particles + offset
            
            # Subsample (simulate production setting)
            # In teleop this is configurable, let's assume 1.0 for stress testing
            
            radius = solver.particle_radius
            
            # Mesh Gen
            mesh = pu.particles_to_mesh(
                positions=particles,
                radius=radius,
                backend='splashsurf'
            )
            return mesh, particles

    def mock_io(mesh, particles, profiler):
        with profiler.time("teleop_io"):
            # Prepare arrays
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

    # Generate combinations
    from itertools import product
    combinations = list(product(
        CONFIG["grid_densities"], 
        CONFIG["particle_multipliers"], 
        CONFIG["stepping_modes"]
    ))
    
    total_substeps = CONFIG["total_substeps_target"]
    
    for grid_res, p_mult, step_mode in combinations:
        print(f"\n>>> Running Config: Grid={grid_res}, PMult={p_mult}, Mode={step_mode}")
        
        cfg = TeleopOptions()
        cfg.general.show_viewer = False 
        cfg.profiling.enabled = True
        
        # --- Parameter Setup ---
        # 1. Grid Resolution
        cfg.robot.base_grid_density = grid_res
        cfg.mpm.grid_density = grid_res
        
        # 2. Particle Size (Independent of Grid somewhat, but relative to it)
        # Standard: 1.0/res. scaled by multiplier.
        # cfg.mpm.particle_size = (1.0 / grid_res) * p_mult 
        # Note: 'particle_size' in Genesis usually means DIAMETER or RADIUS? 
        # Reference options.py: particle_size = 0.8 * 0.01 * 64.0 / grid_density
        # Let's just scale that reference formula.
        base_ref_size = 0.8 * 0.01 * 64.0 / grid_res
        cfg.mpm.particle_size = base_ref_size * p_mult
        
        # 3. Stepping Mode
        if step_mode == "internal":
            cfg.sim.substeps = total_substeps
            loop_iterations = 1 # One python call, N internal substeps
        else:
            cfg.sim.substeps = 1
            loop_iterations = total_substeps # N python calls, 1 internal substep each
            
        # 4. Bounds (Robust)
        dx = 1.0 / grid_res
        mpm_solver_padding = 5 * dx 
        
        # Force cleanup of bounds (prevent GPU Tensor contamination)
        try:
            b0 = cfg.robot.target_shape_bounds[0]
            b1 = cfg.robot.target_shape_bounds[1]
            if hasattr(b0, 'cpu'): b0 = b0.cpu().numpy()
            elif not isinstance(b0, np.ndarray): b0 = np.array(b0)
            if hasattr(b1, 'cpu'): b1 = b1.cpu().numpy()
            elif not isinstance(b1, np.ndarray): b1 = np.array(b1)
            cfg.robot.target_shape_bounds = (b0, b1)
        except Exception:
             # Fallback if accessing fails
             pass

        # Recalculate full bounds for safety (same code as before)
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
            # Teleport grippers to near-contact position so the benchmark measures actual collision physics
            # Calculated Contact Point is approx 0.0165 joint val.
            # We teleport to 0.015 (Pre-Contact)
            env.robot.set_control_mode("TELEPORT")
            warmup_pos = torch.zeros(4, device=env.device)
            warmup_pos[2] = 0.015
            warmup_pos[3] = 0.015
            env.robot.apply_action(warmup_pos, dofs_idx_local=torch.tensor([0, 1, 2, 3], device=env.device))
            
            for _ in range(CONFIG["warmup_frames"]):
                # Run complete "frame" (N substeps)
                for _ in range(loop_iterations):
                    env.scene.step()
            
            # --- Benchmark ---
            profiler.reset()
            
            measure_frames = CONFIG["measure_frames"]
            
            start_time = time.time()
            sim_t = 0.0
            dt = cfg.sim.dt * total_substeps # approx frame dt
            
            # Squeeze Parameters (Matches Visualization)
            press_start = 0.015
            press_end = 0.022
            press_frames = 7
            
            release_end = 0.018
            release_frames = 3
            
            env.robot.set_control_mode("PD_CONTROL")
            pos_cmd = torch.zeros(4, device=env.device)

            for i in range(measure_frames):
                with profiler.time("total_frame"):
                    
                    # Logic (Once per frame)
                    with profiler.time("teleop_logic"):
                            fL, fR = env.robot.get_resistance_forces()
                            
                            # Simulate Split Press/Release Motion
                            if i < press_frames:
                                # PRESS
                                p = i / (press_frames - 1) if press_frames > 1 else 1.0
                                curr_target = press_start + (press_end - press_start) * p
                            else:
                                # RELEASE
                                rel_i = i - press_frames
                                p = rel_i / (release_frames)
                                curr_target = press_end + (release_end - press_end) * p
                            
                            pos_cmd[2] = curr_target 
                            pos_cmd[3] = curr_target 
                            env.robot.apply_action(pos_cmd, dofs_idx_local=torch.tensor([0, 1, 2, 3], device=env.device))
                        
                        # Physics (Loop based on mode)
                        # Note: We group the entire physics loop under 'teleop_physics_loop' for clarity
                        # The internal genesis profiler will capture 'substep' calls inside.
                        with profiler.time("teleop_physics_loop"):
                            for _ in range(loop_iterations):
                                env.scene.step()
                        
                        sim_t += dt

                        # Updates
                        if hasattr(env.scene.sim.mpm_solver, 'update_render_fields'):
                            env.scene.sim.mpm_solver.update_render_fields()
                        else:
                            env.scene.visualizer.update_visual_states()
                        
                    # Recon (Once per frame)
                    mesh, particles = mock_reconstruction(env, profiler)
                    
                    # IO (Once per frame)
                    bytes_sent = mock_io(mesh, particles, profiler)
                    
            end_time = time.time()
            total_duration = end_time - start_time
            avg_fps = measure_frames / total_duration
            
            # Extract Stats
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
                    "stepping_mode": step_mode,
                    "total_substeps": total_substeps
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
