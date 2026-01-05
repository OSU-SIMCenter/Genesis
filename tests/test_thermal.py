
import genesis as gs
import numpy as np

def main():
    gs.init(backend=gs.cpu)
    
    # Define options with thermal enabled
    mpm_options = gs.options.solvers.MPMOptions(
        use_legacy_solver=False,
        default_initial_temperature=1000.0, # Hot!
        grid_density=32,
        lower_bound=(-0.5, 0.0, -0.5),
        upper_bound=(0.5, 1.0, 0.5),
    )
    
    scene = gs.Scene(
        show_viewer=False,
        mpm_options=mpm_options,
        sim_options=gs.options.SimOptions(dt=1e-3)
    )
    
    plane = scene.add_entity(gs.morphs.Plane())
    
    # Add a sphere
    sphere = scene.add_entity(
        gs.morphs.Sphere(
            pos=(0, 0.2, 0),
            radius=0.1,
        ),
        material=gs.materials.MPM.Elastic(),
    )
    
    scene.build()
    
    # Get solver
    # Get solver
    for s in scene.sim.solvers:
        if isinstance(s, gs.engine.solvers.MPMSolver):
            solver = s
            break
    else:
        raise RuntimeError("MPMSolver not found")
    
    print("Simulation initialized.")
    print(f"Initial Avg Temp: {solver.particles.temp.to_numpy().mean():.2f}")
    
    # Run steps to see cooling
    for i in range(50):
        scene.step()
        
    avg_temp = solver.particles.temp.to_numpy().mean()
    print(f"Final Avg Temp after 50 steps: {avg_temp:.2f}")
    
    if avg_temp < 1000.0:
        print("[PASS] Cooling is working (Temp decreased).")
    else:
        print("[FAIL] Cooling is NOT working.")

    # Check if plastic_strain field exists (even if 0)
    if hasattr(solver.particles, 'plastic_strain'):
        print("[PASS] Thermal fields (plastic_strain) exist.")
    else:
        print("[FAIL] Thermal fields missing.")

if __name__ == "__main__":
    main()
