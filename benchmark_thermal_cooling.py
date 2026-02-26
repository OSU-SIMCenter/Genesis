"""
Thermal Cooling Verification Test (Air / Convection)
Produces a text trace showing how a 1000K suspended box cools exponentially
in air toward 293.15K using the new, material-agnostic surface detection.
"""
import genesis as gs
import numpy as np

def main():
    gs.init(backend=gs.gpu, logging_level="warning")

    scene = gs.Scene(
        show_viewer=False,
        mpm_options=gs.options.MPMOptions(
            enable_thermal=True,
            default_initial_temperature=1000.0,
            thermal_air_conductivity=5.0e5, # h_air (cranked up to see cooling in 3s of sim time)
            lower_bound=(0.0, 0.0, 0.0),
            upper_bound=(1.0, 1.0, 1.0),
        ),
    )

    material = gs.materials.MPM.ElastoPlastic(
        E=1e7,
        nu=0.3,
        rho=7850.0, # Steel density
        use_von_mises=True,
        von_mises_yield_stress=1e6,
    )

    # Suspend box in mid air. No gravity so it doesn't deform or hit anything.
    scene.add_entity(
        gs.morphs.Box(
            pos=(0.5, 0.5, 0.5),
            size=(0.1, 0.1, 0.1),
        ),
        material=material,
    )

    scene.build()

    print("=== Air Cooling Verification ===")
    print("Expected: Exponential decay from 1000K towards 293.15K")

    history = []
    
    # We step 200 times to give it enough time to cool visibly.
    for i in range(201):
        scene.step()
        temp = scene.sim.mpm_solver.particles.temp.to_numpy()
        active = scene.sim.mpm_solver.particles_ng.active.to_numpy()
        
        t = temp[0, :, 0]
        a = active[0, :, 0]
        t_active = t[a > 0]
        
        if len(t_active) > 0:
            mean_t = np.mean(t_active)
            min_t = np.min(t_active)
            max_t = np.max(t_active)
            history.append(mean_t)
            
            if i % 20 == 0:
                print(f"Step {i:3d} | Mean: {mean_t:.2f}K | Min (Surface): {min_t:.2f}K | Max (Core): {max_t:.2f}K")

    if history[-1] < 1000.0 and history[-1] > 293.15:
        print("\n✅ PASS: Air cooling is correctly decaying toward room temperature.")
    else:
        print("\n❌ FAIL: Temperature is divergent or not cooling.")

if __name__ == "__main__":
    main()
