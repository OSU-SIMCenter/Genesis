"""
Thermal Diffusion Verification Test (Internal Conduction)
Verifies that heat diffuses internally through the material via the
mass-conservative volume-fraction Laplacian, with perfect energy conservation.
"""
import genesis as gs
import numpy as np

def main():
    gs.init(backend=gs.gpu, logging_level="warning")

    # Use a CFL-safe configuration:
    # dx = 1/32 = 0.03125, alpha = 0.01
    # CFL limit: dx^2 / (6*alpha) = 0.03125^2 / 0.06 = 0.01627
    # substep_dt = 2e-3 / 10 = 2e-4, well under CFL limit
    scene = gs.Scene(
        show_viewer=False,
        sim_options=gs.options.SimOptions(
            dt=2e-3,
            substeps=10,
        ),
        mpm_options=gs.options.MPMOptions(
            lower_bound=(0.0, 0.0, 0.0),
            upper_bound=(1.0, 1.0, 1.0),
            grid_density=32,
            enable_thermal=True,
            default_initial_temperature=293.15,  # Start cold, we'll heat half manually
            default_thermal_diffusivity=0.01,    # High but CFL-safe diffusivity
            thermal_air_conductivity=0.0,        # PERFECTLY INSULATED (0 air cooling)
            thermal_contact_conductivity=0.0,
        ),
    )

    rho = 7850.0  # Steel density
    cp = 450.0    # J/(kg K)

    # Create a 20cm perfectly insulated cube in zero gravity
    scene.add_entity(
        morph=gs.morphs.Box(
            pos=(0.5, 0.5, 0.5),
            size=(0.2, 0.2, 0.2),
        ),
        material=gs.materials.MPM.Elastic(
            E=2e8,
            nu=0.3,
            rho=rho,
        ),
    )

    scene.build()

    solver = scene.sim.mpm_solver

    # --- Set up the temperature gradient ---
    # Read current particle positions and temps
    pos_np = solver.particles.pos.to_numpy()   # shape: (substeps+1, n_particles, B, 3)
    temp_np = solver.particles.temp.to_numpy()  # shape: (substeps+1, n_particles, B)
    active_np = solver.particles_ng.active.to_numpy()  # shape: (substeps+1, n_particles, B)

    # Work with frame 0, batch 0
    positions = pos_np[0, :, 0, :]  # (n_particles, 3)
    active = active_np[0, :, 0]     # (n_particles,)

    # Heat the right half (x > 0.5) to 1000K, keep left half at 293.15K
    right_mask = (positions[:, 0] > 0.5) & (active > 0)
    left_mask = (positions[:, 0] <= 0.5) & (active > 0)

    n_right = np.sum(right_mask)
    n_left = np.sum(left_mask)
    print(f"Particles: {n_right} right (hot), {n_left} left (cold), {np.sum(active > 0)} total active")

    # Write directly to ALL frames and batches in the Taichi field
    for f_idx in range(temp_np.shape[0]):
        temp_np[f_idx, right_mask, 0] = 1000.0
        temp_np[f_idx, left_mask, 0] = 293.15
    solver.particles.temp.from_numpy(temp_np)

    # --- Energy Conservation: Step 0 ---
    mass_np = solver.particles_info.mass.to_numpy()  # shape: (n_particles,)
    T_0 = solver.particles.temp.to_numpy()[0, :, 0]
    active_mask = active > 0
    Energy_0 = np.sum(mass_np[active_mask] * T_0[active_mask] * cp)
    
    print(f"\n[Step 0] Total Thermal Energy: {Energy_0:.2f} J")
    print(f"[Step 0] Max Temp: {T_0[active_mask].max():.2f} K")
    print(f"[Step 0] Min Temp: {T_0[active_mask].min():.2f} K")

    # Step the simulation
    for i in range(201):
        scene.step()
        if i % 40 == 0:
            T = solver.particles.temp.to_numpy()[0, :, 0]
            t_active = T[active_mask]
            E = np.sum(mass_np[active_mask] * t_active * cp)
            pct = abs(E - Energy_0) / Energy_0 * 100
            print(f"Step {i:3d} | Mean: {np.mean(t_active):.2f}K | "
                  f"Min: {np.min(t_active):.2f}K | Max: {np.max(t_active):.2f}K | "
                  f"Energy drift: {pct:.4f}%")

    # --- Final Assertions ---
    T_final = solver.particles.temp.to_numpy()[0, :, 0]
    t_final_active = T_final[active_mask]
    Energy_final = np.sum(mass_np[active_mask] * t_final_active * cp)

    energy_diff_pct = abs(Energy_final - Energy_0) / Energy_0 * 100
    print(f"\n--- Final Results ---")
    print(f"Energy conservation: {energy_diff_pct:.4f}% drift")
    print(f"Max Temp: {t_final_active.max():.2f} K")
    print(f"Min Temp: {t_final_active.min():.2f} K")

    # 1. Energy conservation within 0.5% (P2G/G2P interpolation adds some noise)
    assert energy_diff_pct < 0.5, f"Energy conservation FAILED! {energy_diff_pct:.4f}% drift"
    
    # 2. Diffusion occurred (heat flowed from hot to cold)
    assert t_final_active.max() < 999.0, f"Max temp did not decrease! {t_final_active.max()} K"
    assert t_final_active.min() > 294.0, f"Min temp did not increase! {t_final_active.min()} K"
    
    # 3. No NaN explosions
    assert not np.isnan(t_final_active).any(), "NaN values detected!"
    
    print("\n✅ Heat Diffusion benchmark PASSED! Energy conserved, heat diffused correctly.")

if __name__ == "__main__":
    main()
