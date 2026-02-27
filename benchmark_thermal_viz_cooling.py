"""
Thermal Cooling Visualization Test (Rerun)
Visualizes air convection vs. floor contact conduction.
Two cylinders start at 1000K.
One is suspended in the air. One drops and touches the 293K floor.
"""
import genesis as gs
import numpy as np
import rerun as rr
from agforge.reconstruction import SurfaceReconstructor, SamplingMethod
from agforge.materials import JohnsonCookPlasticity

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
    rr.init("genesis_thermal_cooling", spawn=True)
    gs.init(backend=gs.gpu, logging_level="warning")

    # ----- Genesis Scene Setup using Agility Forge Options -----
    from agforge.options import TeleopOptions
    from agforge.materials import JohnsonCookPlasticity

    cfg = TeleopOptions()
    cfg.general.show_viewer = True
    
    # Overwrite thermal params for cooling test
    cfg.mpm.enable_thermal = True
    cfg.mpm.default_initial_temperature = 293.15
    cfg.mpm.default_thermal_diffusivity = 0.01
    cfg.mpm.thermal_air_conductivity = 0.01
    cfg.mpm.thermal_contact_conductivity = 0.5
    
    # Expand MPM bounds to fit two standing cylinders
    cfg.mpm.lower_bound = (-0.2, -0.1, 0.0)
    cfg.mpm.upper_bound = (0.2, 0.1, 0.6)

    # Enable Genesis Grid/Boundary Visualizations
    cfg.vis.visualize_mpm_boundary = True
    cfg.vis.visualize_mpm_grid = True

    scene = gs.Scene(
        sim_options=cfg.sim,
        viewer_options=cfg.viewer,
        rigid_options=gs.options.RigidOptions(dt=cfg.sim.dt),
        mpm_options=cfg.mpm,
        vis_options=cfg.vis,
        profiling_options=cfg.profiling,
        show_viewer=cfg.general.show_viewer,
    )

    # Add Floor
    scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", pos=(0, 0, 0.01), fixed=True))

    # Materials
    mat = JohnsonCookPlasticity(
        E=cfg.mat.E, nu=cfg.mat.nu, rho=cfg.mat.rho,
        A=cfg.mat.jc_A, B=cfg.mat.jc_B, n=cfg.mat.jc_n, C=cfg.mat.jc_C, eps0=cfg.mat.jc_eps0,
        sampler="pbs"  # Poisson disk sampling for better particle distribution
    )

    # Properties matching original AgilityForge cylinder but standing up
    cyl_radius = cfg.robot.cylinder_radius
    cyl_height = cfg.robot.cylinder_height

    # 1. Floating Cylinder (Air Cooling)
    floating_entity = scene.add_entity(
        material=mat,
        morph=gs.morphs.Cylinder(
            radius=cyl_radius, height=cyl_height, 
            pos=(-0.05, 0, 0.3), # High up on the left
            euler=(0, 0, 0)
        ),
        surface=gs.surfaces.Metal(color=(0.8, 0.4, 0.0), vis_mode="particle"),
    )

    # 2. Dropped Cylinder (Contact Cooling)
    dropped_entity = scene.add_entity(
        material=mat,
        morph=gs.morphs.Cylinder(
            radius=cyl_radius, height=cyl_height, 
            pos=(0.05, 0, 0.25), # Lower on the right, drops to floor
            euler=(0, 0, 0)
        ),
        surface=gs.surfaces.Metal(color=(0.8, 0.4, 0.0), vis_mode="particle"),
    )

    scene.build()

    # Freeze the floating cylinder so it doesn't fall
    floating_entity.set_free(np.zeros(floating_entity.n_particles, dtype=bool))
    # Ensure dropped cylinder falls
    dropped_entity.set_free(np.ones(dropped_entity.n_particles, dtype=bool))
    
    solver = scene.sim.mpm_solver
    
    # Initialize cylinder temperatures to 1000K
    pos_np = solver.particles.pos.to_numpy()
    temp_np = solver.particles.temp.to_numpy()
    active_np = solver.particles_ng.active.to_numpy()

    active = active_np[0, :, 0]
    hot_mask = active > 0

    for f_idx in range(temp_np.shape[0]):
        temp_np[f_idx, hot_mask, 0] = 1000.0
    solver.particles.temp.from_numpy(temp_np)

    print("Starting simulation to generate Rerun visualization...")
    
    MIN_TEMP = 293.15
    MAX_TEMP = 1000.0

    for i in range(400):
        # Step Physics
        scene.step()
        
        # Update Genesis render fields
        if hasattr(solver, 'update_render_fields'):
            solver.update_render_fields()
        else:
            scene.visualizer.update_visual_states()

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
            
            # Log exact particles (with temperature colors)
            rr.log("mpm/particles", rr.Points3D(p_active, colors=colors_rgb, radii=0.003))

    print("\n✅ Visualization complete. Viewer should be open.")

if __name__ == "__main__":
    main()
