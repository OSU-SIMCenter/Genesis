import genesis as gs
import numpy as np
from agforge.reconstruction import SurfaceReconstructor
from agforge.thermal import InductionHeater

def main():
    gs.init(backend=gs.gpu)

    scene = gs.Scene(
        show_viewer=True,
        sim_options=gs.options.SimOptions(
            dt=2e-3, 
            substeps=10
        ),
        viewer_options=gs.options.ViewerOptions(
            res=(1280, 960),
            camera_pos=(1, 1, 1),
            camera_lookat=(0, 0.2, 0),
            camera_fov=30,
        ),
        mpm_options=gs.options.MPMOptions(
            lower_bound=(-0.5, 0.0, -0.5),
            upper_bound=(0.5, 1.0, 0.5),
            grid_density=64,
            enable_thermal=True,
            default_initial_temperature=293.15,
            default_thermal_diffusivity=0.01,
        ),
    )

    # Floor
    scene.add_entity(gs.morphs.Plane())

    # Load billet
    billet = scene.add_entity(
        material=gs.materials.MPM.Elastic(
            E=2e8,
            nu=0.3,
            rho=8000.0,
            sampler="regular",
        ),
        morph=gs.morphs.Box(
            pos=(0, 0.4, 0),
            size=(0.2, 0.4, 0.2),
        ),
    )

    scene.build()

    class MockEnv:
        def __init__(self, scene, mpm_entity):
            self.scene = scene
            self.mpm_entity = mpm_entity
            if not hasattr(self.scene, 'profiling_options'):
                self.scene.profiling_options = None
                
    mock_env = MockEnv(scene, billet)

    print("Setting up Induction Heater...")
    # 1. We create the surface reconstructor explicitly
    reconstructor = SurfaceReconstructor(
        env=mock_env,
        grid_res=64, # matches default
        backend="hybrid",
    )

    # 2. Setup the Heater
    heater = InductionHeater(
        solver=scene.sim.mpm_solver,
        entity=billet,
        reconstructor=reconstructor
    )

    # Run the induction heating loop
    print("Beginning 1.0 second of induction heating...")
    dt = scene.sim.dt
    heating_duration = 1.0
    power = 2000.0 # Kelvin/sec at surface
    skin_depth = 0.02 # 2cm skin depth

    # Simulation loop
    for i in range(int(heating_duration / dt)):
        # Calculate new surface mesh (only costs 10-15ms!)
        reconstructor.create_reconstructed_mesh()
        
        # Apply heat
        heater.step_heat(dt, power, skin_depth)
        
        # Step physics
        scene.step()
        
        if i % 50 == 0:
            temps = billet.get_particles_temp().cpu().numpy().squeeze()
            print(f"Frame {i:3d}: Max Temp = {np.max(temps):.1f} K, Min Temp = {np.min(temps):.1f} K")

    print("\nHeating complete. Allowing thermal diffusion to settle for 2.0 seconds...")
    for i in range(int(2.0 / dt)):
        scene.step()
        
        if i % 50 == 0:
            temps = billet.get_particles_temp().cpu().numpy().squeeze()
            print(f"Diffusion {i:3d}: Max Temp = {np.max(temps):.1f} K, Min Temp = {np.min(temps):.1f} K")

if __name__ == "__main__":
    main()
