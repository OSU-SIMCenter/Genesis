
import genesis as gs
import numpy as np

def main():
    gs.init(backend=gs.cpu)
    
    mpm_options = gs.options.solvers.MPMOptions(
        use_legacy_solver=False,
        default_initial_temperature=1000.0, # Hot
        grid_density=32,
        lower_bound=(-0.5, -0.5, -0.5),
        upper_bound=(0.5, 1.0, 0.5),
        thermal_contact_conductivity=1000.0, # High conductivity to make effect obvious
    )
    
    scene = gs.Scene(
        show_viewer=False,
        mpm_options=mpm_options,
        sim_options=gs.options.SimOptions(dt=1e-3)
    )
    
    # Cold floor
    floor = scene.add_entity(gs.morphs.Plane())
    
    # Hot cube sitting on the floor
    cube = scene.add_entity(
        gs.morphs.Box(
            pos=(0, 0.05, 0),
            size=(0.1, 0.1, 0.1),
        ),
        material=gs.materials.MPM.Elastic(),
        surface=gs.surfaces.Metal(color=(0.8, 0.4, 0.0), vis_mode="particle"),
    )
    
    scene.build()
    
    # Get solver
    for s in scene.sim.solvers:
        if isinstance(s, gs.engine.solvers.MPMSolver):
            solver = s
            break
    
    print("Simulation initialized.")
    initial_temp = solver.particles.temp.to_numpy().mean()
    print(f"Initial Avg Temp: {initial_temp:.2f}")
    
    # Run steps
    for i in range(50):
        scene.step()
        
    final_temp = solver.particles.temp.to_numpy().mean()
    print(f"Final Avg Temp: {final_temp:.2f}")
    
    diff = initial_temp - final_temp
    print(f"Temp Drop: {diff:.2f}")

    # We expect SIGNIFICANT cooling if contact heat transfer is working.
    # Air cooling alone (from previous test) caused ~3 degree drop.
    # Contact with infinitely cold floor with high conductivity should be much faster.
    if diff > 10.0:
        print("[PASS] Contact cooling detected (Significant drop).")
    else:
        print("[FAIL] Contact cooling NOT effective/detected.")

if __name__ == "__main__":
    main()
