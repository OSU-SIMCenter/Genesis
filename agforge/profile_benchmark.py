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
    # Define parameter ranges for the sweep
    resolutions = [100, 200] # Reduced for testing speed, user can expand
    substep_counts = [4, 8]

    # Benchmark config
    warmup_steps = 5
    measure_steps = 20
    
    results = []

    print(f"Starting Benchmark Sweep...")
    print(f"Resolutions: {resolutions}")
    print(f"Substeps: {substep_counts}")
    print(f"Measure Steps: {measure_steps}")
    print("-" * 60)

    # Import dependencies for Logic/Recon/IO
    import struct
    import contextlib
    from genesis.utils import particle as pu
    import gstaichi as ti
    
    # Helper to mimic SharedState logic
    def mock_reconstruction(env, profiler):
        # Based on TeleopSocket logic
        with profiler.time("teleop_recon"):
            solver = env.scene.sim.mpm_solver
            # Sync needed for accurate timing of GPU ops
            # gs.tools.run_in_thread(ti.sync)
            ti.sync()
            
            # Access particles (expensive copy to CPU usually)
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

    for res in resolutions:
        for substeps in substep_counts:
            print(f"\n>>> Running Config: Resolution={res}, Substeps={substeps}")
            
            cfg = TeleopOptions()
            cfg.general.show_viewer = False 
            cfg.profiling.enabled = True
            
            # Force cleanup of bounds (prevent GPU Tensor contamination)
            b0 = cfg.robot.target_shape_bounds[0]
            b1 = cfg.robot.target_shape_bounds[1]
            if hasattr(b0, 'cpu'): b0 = b0.cpu().numpy()
            elif not isinstance(b0, np.ndarray): b0 = np.array(b0)
            
            if hasattr(b1, 'cpu'): b1 = b1.cpu().numpy()
            elif not isinstance(b1, np.ndarray): b1 = np.array(b1)
            cfg.robot.target_shape_bounds = (b0, b1)
            
            # Override params
            cfg.robot.base_grid_density = res
            cfg.mpm.grid_density = res
            cfg.mpm.particle_size = 0.8 * 0.01 * 64.0 / res
            cfg.sim.substeps = substeps
            
            # Recalculate bounds with new density (logic from options.py)
            # Use safer padding (5*dx instead of 3*dx) or just add constant padding
            dx = 1.0 / res
            mpm_solver_padding = 5 * dx # Increased from 3*dx
            
            # We need to access the derived values that were calculated in model_post_init
            # But we can just grab the existing bounds and expand them slightly?
            # Or better, recalculate fully if we have the cylinder params.
            # cfg.robot.cylinder_pos is available.
            
            # Re-implementing logic for safety:
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
                for _ in range(warmup_steps):
                    env.scene.step()
                
                # --- Benchmark ---
                profiler.reset()
                
                start_time = time.time()
                for _ in range(measure_steps):
                    with profiler.time("total_frame"):
                        # 1. Logic / Forces
                        with profiler.time("teleop_logic"):
                            # Mimic getting forces
                            fL, fR = env.robot.get_resistance_forces()
                            # Mimic simple logic
                            force_param = 0.5
                            target_strain = force_param * 10
                        
                        # 2. Physics
                        # env.scene.step() handles its own internal profiling
                        env.scene.step()
                        
                        # 3. Render Fields Update
                        if hasattr(env.scene.sim.mpm_solver, 'update_render_fields'):
                            env.scene.sim.mpm_solver.update_render_fields()
                        else:
                            env.scene.visualizer.update_visual_states()
                            
                        # 4. Reconstruction
                        mesh, particles = mock_reconstruction(env, profiler)
                        
                        # 5. IO
                        bytes_sent = mock_io(mesh, particles, profiler)
                        
                end_time = time.time()
                total_duration = end_time - start_time
                avg_fps = measure_steps / total_duration
                
                # Extract Stats
                # stats is Dict[str, ProfileStats]
                # We want to serialize this. ProfileStats object is not JSON serializable.
                # We need to convert it.
                stats_dict = {}
                for name, stat in profiler.stats.items():
                    stats_dict[name] = {
                        "count": stat.count,
                        "total": stat.total,
                        "mean": stat.mean,
                        "min": stat.min,
                        "max": stat.max,
                        "std": stat.std
                    }
                
                n_particles = env.scene.sim.mpm_solver.n_particles
                
                data_point = {
                    "config": {"resolution": res, "substeps": substeps},
                    "metrics": {
                        "n_particles": int(n_particles),
                        "avg_step_time_ms": (total_duration / measure_steps) * 1000.0,
                        "avg_fps": avg_fps,
                    },
                    "cProfile_stats": stats_dict # Detailed breakdown
                }
                results.append(data_point)
                
                print(f"  > Particles: {n_particles}")
                print(f"  > FPS: {avg_fps:.2f}")
                print(f"  > Avg Step Time: {data_point['metrics']['avg_step_time_ms']:.2f} ms")
                
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
