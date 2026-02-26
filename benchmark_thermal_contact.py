"""
Thermal Contact Verification Test (Conduction to rigid floor)
Produces a text trace showing how a 1000K box cools extremely fast when in contact
with a rigid floor (infinite heatsink) compared to air.
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
            thermal_air_conductivity=5.0e3, # very low air cooling
            thermal_contact_conductivity=5.0e7, # very high floor cooling
            lower_bound=(0.0, 0.0, 0.0),
            upper_bound=(1.0, 1.0, 1.0),
        ),
        rigid_options=gs.options.RigidOptions(
            enable_collision=False, # Disable physical explosion, allow thermal SDF conduction only
            enable_joint_limit=True,
        )
    )

    material = gs.materials.MPM.ElastoPlastic(
        E=1e7,
        nu=0.3,
        rho=7850.0,
        use_von_mises=True,
    )

    # Box dropping onto the floor
    scene.add_entity(
        gs.morphs.Box(
            pos=(0.5, 0.5, 0.1), # start at 0.1
            size=(0.1, 0.1, 0.1),
        ),
        material=material,
    )
    
    # A Rigid floor
    scene.add_entity(
        gs.morphs.Plane(
            pos=(0.0, 0.0, 0.046875), # exactly match MPM lower padding
        ),
    )

    scene.build()

    print("=== Contact Cooling Verification ===")
    print("Expected: Min temperature on bottom surface plunges rapidly toward 293.15K")

    history = []
    
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
            
            p = scene.sim.mpm_solver.particles.pos.to_numpy()[0, :, 0]
            p_active = p[a > 0]
            min_z = np.min(p_active[:, 2])
            
            history.append((mean_t, min_t))
            
            if i % 20 == 0:
                print(f"Step {i:3d} | Mean: {mean_t:.2f}K | Min (Floor contact): {min_t:.2f}K | Box Z: {min_z:.4f}")

    min_recorded_temp = min([h[1] for h in history])
    if min_recorded_temp < 960.0:
        print("\n✅ PASS: Contact cooling to rigid floor is functioning robustly.")
    else:
        print(f"\n❌ FAIL: Contact cooling might not be triggering. Min recorded temp was {min_recorded_temp:.2f}K")

if __name__ == "__main__":
    main()
