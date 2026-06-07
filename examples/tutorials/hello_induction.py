import genesis as gs
import numpy as np
from agforge.reconstruction import SurfaceReconstructor
from agforge.thermal import InductionHeater

def main():
    gs.init(backend=gs.gpu)

    # CFL note: For E=2e8, rho=8000 → wave speed c = √(E/ρ) ≈ 158 m/s.
    # With grid_density=64, dx ≈ 0.0156 m → CFL limit: sub_dt < dx/c ≈ 9.9e-5 s.
    # Using dt=2e-3, substeps=32 → sub_dt = 6.25e-5 s (safety factor ~0.63). ✓
    dt = 2e-3
    substeps = 32

    scene = gs.Scene(
        show_viewer=True,
        sim_options=gs.options.SimOptions(
            dt=dt,
            substeps=substeps,
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
        grid_res=64,
        backend="hybrid",
    )

    # 2. Setup the Heater
    heater = InductionHeater(
        solver=scene.sim.mpm_solver,
        entity=billet,
        reconstructor=reconstructor,
    )

    # Run the induction heating loop
    print("Beginning 1.0 second of induction heating...")
    heating_duration = 1.0

    # Induction config. heating_power is a PEAK VOLUMETRIC POWER DENSITY [W/m^3] (coil field
    # intensity), not a total wattage — geometry-independent. Heat is deposited on the GPU
    # inside the MPM P2G kernel; we only publish the coil uniforms each frame.
    q_peak = 5.0e7        # Peak volumetric power density [W/m^3]
    skin_depth = 0.05     # EM skin depth [m]

    # Precompute the per-particle skin depth ONCE from the surface mesh (recompute only after
    # large deformations). This is the expensive CPU/SDF step; per-frame heating stays on GPU.
    reconstructor.create_reconstructed_mesh()
    heater.recompute_and_upload()

    # Coil coaxial with the billet along the X axis, centered on the billet centroid.
    centroid = billet.get_particles_pos().cpu().numpy().reshape(-1, 3).mean(axis=0)
    coil_center = [float(centroid[0]), float(centroid[1]), float(centroid[2])]
    coil_half_length = 0.2   # covers the billet's X extent
    coil_radius = 0.15

    # Simulation loop
    for i in range(int(heating_duration / dt)):
        # Publish coil uniforms; the engine deposits heat in P2G during scene.step().
        scene.sim.mpm_solver.set_induction_params(
            center=coil_center,
            half_length=coil_half_length,
            radius=coil_radius,
            q_peak=q_peak,
            skin_depth=skin_depth,
            active=True,
        )

        # Step physics (induction + diffusion + cooling all happen here)
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
