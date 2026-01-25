import time
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

def run_visualization():
    print("Initializing Benchmark Visualization...")
    print("This runs the 'Standard' config (Grid=100, PMult=1.0) with the viewer enabled.")
    print("Use this to verify what is being simulated.")
    print("-" * 60)

    # Standard Config
    grid_res = 100
    p_mult = 1.0
    
    cfg = TeleopOptions()
    cfg.general.show_viewer = True  # <--- VISUALIZATION ON
    cfg.profiling.enabled = False   # Profiling OFF (we just want to see it)
    
    # 1. Grid Resolution
    cfg.robot.base_grid_density = grid_res
    cfg.mpm.grid_density = grid_res
    
    # 2. Particle Size
    base_ref_size = 0.8 * 0.01 * 64.0 / grid_res
    cfg.mpm.particle_size = base_ref_size * p_mult
    
    # 3. Mode (Use Internal for smooth viewing)
    cfg.sim.substeps = 32

    # 4. Bounds (Robust)
    dx = 1.0 / grid_res
    mpm_solver_padding = 5 * dx 

    # 5. Visualization Settings (Grid & Boundary)
    cfg.vis.visualize_mpm_grid = True
    cfg.vis.visualize_mpm_boundary = True
    cfg.vis.show_world_frame = False
    
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

    # Recalculate full bounds
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
        print("Building Environment (this may take a moment)...")
        env = build_env(cfg)
        
        print("\n\n>>> Simulation Running. Press Ctrl+C to stop.\n")
        
        print("\n\n>>> Simulation Running. Press Ctrl+C to stop.\n")
        
        # Benchmark-Mirrored Loop
        FRAME_STEPS = 32 # Total substeps per frame in internal mode
        WARMUP_FRAMES = 3 
        MEASURE_FRAMES = 10
        
        while True:
            # --- PHASE 1: RESET (Open Grippers + Reset MPM) ---
            print("\n>> RESET: Opening Grippers & Resetting MPM")
            
            # Reset MPM state
            # AgilityForgeEnv.reset_idx() handles scene reset if we call it correctly, 
            # or we can call scene.reset() directly.
            # env.scene.reset() resets particles to initial state.
            env.scene.reset() 
            
            env.robot.set_control_mode("TELEPORT")
            pos_cmd = torch.zeros(4, device=env.device)
            pos_cmd[2] = 0.0 
            pos_cmd[3] = 0.0 
            env.robot.apply_action(pos_cmd, dofs_idx_local=torch.tensor([0, 1, 2, 3], device=env.device))
            
            # Step a bit to let it settle
            for _ in range(10):
                env.scene.step()
                if hasattr(env.scene.sim.mpm_solver, 'update_render_fields'): env.scene.sim.mpm_solver.update_render_fields()
                else: env.scene.visualizer.update_visual_states()
                
            # --- PHASE 2: WARMUP (Teleport to Contact) ---
            print(">> WARMUP: Teleport to Contact (0.015)")
            pos_cmd[2] = 0.015
            pos_cmd[3] = 0.015
            env.robot.apply_action(pos_cmd, dofs_idx_local=torch.tensor([0, 1, 2, 3], device=env.device))
            
            for i in range(WARMUP_FRAMES):
                for _ in range(FRAME_STEPS): env.scene.step() # Simulate frame loop
                
                if hasattr(env.scene.sim.mpm_solver, 'update_render_fields'): env.scene.sim.mpm_solver.update_render_fields()
                else: env.scene.visualizer.update_visual_states()
                time.sleep(0.05) 
                
            # --- PHASE 3: MEASURE (Press 7 -> Release 3) ---
            print(">> BENCHMARK: Press (7 frames) -> Release (3 frames)")
            env.robot.set_control_mode("PD_CONTROL")
            
            # Press Config
            press_start = 0.015
            press_end = 0.022
            press_frames = 7
            
            # Release Config
            release_end = 0.018 # Partial release
            release_frames = 3 # Remaining frames (Total 10)
            
            for i in range(MEASURE_FRAMES):
                
                # Logic: Determine Target based on Frame Index
                if i < press_frames:
                    # PRESS PAASE
                    p = i / (press_frames - 1) if press_frames > 1 else 1.0
                    curr_target = press_start + (press_end - press_start) * p
                    stage = "Press"
                else:
                    # RELEASE PHASE
                    rel_i = i - press_frames
                    p = rel_i / (release_frames) # Start release immediately after max press
                    curr_target = press_end + (release_end - press_end) * p
                    stage = "Rel  "

                # Apply Action
                pos_cmd[2] = curr_target
                pos_cmd[3] = curr_target
                env.robot.apply_action(pos_cmd, dofs_idx_local=torch.tensor([0, 1, 2, 3], device=env.device))
                
                # Physics
                env.robot.get_resistance_forces() 
                for _ in range(FRAME_STEPS): env.scene.step() 
                
                # Visuals
                if hasattr(env.scene.sim.mpm_solver, 'update_render_fields'): env.scene.sim.mpm_solver.update_render_fields()
                else: env.scene.visualizer.update_visual_states()
                
                # Debug Info
                pos_L = env.robot.left_gripper.get_pos().cpu().numpy()
                pos_R = env.robot.right_gripper.get_pos().cpu().numpy()
                dist = np.linalg.norm(pos_L - pos_R)
                print(f"\r  [{stage}] {i+1}/{MEASURE_FRAMES} | Width: {dist:.4f} | Tgt: {curr_target:.4f}", end="")
                time.sleep(0.05) 
                
            print(f" -> Done. Looping...")
            time.sleep(1.0)
            
            # 3. Update Visuals
            if hasattr(env.scene.sim.mpm_solver, 'update_render_fields'):
                env.scene.sim.mpm_solver.update_render_fields()
            else:
                env.scene.visualizer.update_visual_states()
            
    except KeyboardInterrupt:
        print("\nStopping...")
        env.scene.destroy()
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_visualization()
