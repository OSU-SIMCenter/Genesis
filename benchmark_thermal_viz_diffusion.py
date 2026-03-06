"""
Thermal Diffusion Visualization Test (Rerun)
Visualizes subsurface heat diffusion in a perfectly insulated block.
The right half starts at 1000K, the left half at 293.15K.
Heat diffuses to a uniform ~653K.
"""
import genesis as gs
import numpy as np
import rerun as rr
from agforge.reconstruction import SurfaceReconstructor, SamplingMethod
from agforge.agforge_builder import build_env
from agforge.options import TeleopOptions

def get_coolwarm_color(t_norm):
    colors = np.zeros((len(t_norm), 3), dtype=np.uint8)
    cold_mask = t_norm < 0.5
    t_cold = t_norm[cold_mask] * 2.0  
    colors[cold_mask, 0] = (t_cold * 220).astype(np.uint8)
    colors[cold_mask, 1] = (t_cold * 220).astype(np.uint8)
    colors[cold_mask, 2] = (255 - t_cold * 35).astype(np.uint8)
    
    hot_mask = ~cold_mask
    t_hot = (t_norm[hot_mask] - 0.5) * 2.0 
    colors[hot_mask, 0] = (220 + t_hot * 35).astype(np.uint8)
    colors[hot_mask, 1] = (220 - t_hot * 220).astype(np.uint8)
    colors[hot_mask, 2] = (220 - t_hot * 220).astype(np.uint8)
    return colors

def main():
    rr.init("genesis_thermal_diffusion", spawn=True)
    gs.init(backend=gs.gpu, logging_level="warning")

    # Use agforge environment builder for accurate forging setup
    cfg = TeleopOptions()
    cfg.general.show_viewer = False  # Disable Genesis Viewer to improve FPS (Rerun only)
    
    # Overwrite thermal params for diffusion test
    cfg.mpm.enable_thermal = True
    cfg.mpm.default_initial_temperature = 293.15
    cfg.mpm.thermal_time_scale = 10000.0  # Phase 5 SOTA Time-Scaling
    cfg.mpm.thermal_air_conductivity = 0.0      # Insulated
    cfg.mpm.thermal_contact_conductivity = 0.0  # Insulated
    
    env = build_env(cfg)
    scene = env.scene
    solver = scene.sim.mpm_solver

    # Set up Surface Reconstructor
    reconstructor = SurfaceReconstructor(env)
    reconstructor.recon_enabled = True
    reconstructor.recon_frame_interval = 1 
    reconstructor.recon_particle_fraction = 1.0
    reconstructor.sampling_method = SamplingMethod.VOXEL_STRATIFIED
    reconstructor.create_reconstructed_mesh()
    reconstructor.init_skinning()

    # --- Set up the temperature gradient ---
    pos_np = solver.particles.pos.to_numpy()
    temp_np = solver.particles.temp.to_numpy()
    active_np = solver.particles_ng.active.to_numpy()

    positions = pos_np[0, :, 0, :]
    active = active_np[0, :, 0]

    right_mask = (positions[:, 0] > 0.0) & (active > 0)
    left_mask = (positions[:, 0] <= 0.0) & (active > 0)

    for f_idx in range(temp_np.shape[0]):
        temp_np[f_idx, right_mask, 0] = 1000.0
        temp_np[f_idx, left_mask, 0] = 293.15
    solver.particles.temp.from_numpy(temp_np)

    print("Starting simulation to generate Rerun visualization...")
    
    MIN_TEMP = 293.15
    MAX_TEMP = 1000.0

    STEPS_PER_RENDER = 50
    TOTAL_RENDER_FRAMES = 250

    for i in range(TOTAL_RENDER_FRAMES):
        # Step Physics
        for _ in range(STEPS_PER_RENDER):
            scene.step()
        
        # Update Genesis render fields
        if hasattr(solver, 'update_render_fields'):
            solver.update_render_fields()
        # scene.visualizer is skipped since we run headless
            
        # Update Rerun Recon Mesh
        reconstructor.update(should_reconstruct=True)

        # Log to Rerun every few steps
        if i % 2 == 0:
            rr.set_time("step", sequence=i)
            
            # Fetch current state
            pos = solver.particles.pos.to_numpy()[0, :, 0, :]
            temps = solver.particles.temp.to_numpy()[0, :, 0]
            curr_active = solver.particles_ng.active.to_numpy()[0, :, 0] > 0
            
            p_active = pos[curr_active]
            t_active = temps[curr_active]
            
            # Normalize temperatures for colormap
            t_norm = np.clip((t_active - MIN_TEMP) / (MAX_TEMP - MIN_TEMP), 0.0, 1.0)
            colors_rgb = get_coolwarm_color(t_norm)
            
            # 1. Log exact particles (with temperature colors)
            rr.log("mpm/particles", rr.Points3D(p_active, colors=colors_rgb, radii=0.002))
            
            # 2. Log reconstructed surface mesh (with temperature coloring via vertex colors!)
            mesh = reconstructor.reconstructed_mesh
            if len(mesh.vertices) > 0:
                # We need to compute vertex colors correctly. 
                # For now, just log the mesh with standard coloring or white, since vertex coloring 
                # from particles requires spatial interpolation. We'll stick to a translucent gray mesh to see the glowing particles inside.
                rr.log(
                    "mpm/surface",
                    rr.Mesh3D(
                        vertex_positions=mesh.vertices,
                        vertex_normals=mesh.vertex_normals,
                        triangle_indices=mesh.faces, 
                        albedo_factor=[0.6, 0.6, 0.7, 0.6]  # Translucent glass/gray
                    )
                )

    print("\n✅ Visualization complete. Viewer should be open.")

if __name__ == "__main__":
    main()
