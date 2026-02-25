import genesis as gs
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy", action="store_true", help="Use legacy monolithic solver")
    args = parser.parse_args()

    gs.init(backend=gs.gpu, logging_level="info")
    
    scene = gs.Scene(
        show_viewer=False,
        mpm_options=gs.options.MPMOptions(
            use_legacy_solver=args.legacy,
            default_initial_temperature=1000.0,
            lower_bound=(0.0, 0.0, 0.0),
            upper_bound=(1.0, 1.0, 1.0),
        ),
    )
    
    material = gs.materials.MPM.ElastoPlastic(
        E=1e7,
        nu=0.3,
        rho=7850.0,  # Steel density
        use_von_mises=True,
        von_mises_yield_stress=1e6,
    )
    
    scene.add_entity(
        gs.morphs.Box(
            pos=(0.5, 0.5, 0.5),
            size=(0.2, 0.2, 0.2),
        ),
        material=material,
    )
    
    scene.build()
    
    for i in range(50):
        scene.step()
        if not args.legacy:
            temp = scene.sim.mpm_solver.particles.temp.to_numpy()
            print(f"Step {i}: max temp f0={temp[0].max():.2f}, f1={temp[1].max():.2f}")
        
    print(f"Successfully ran 50 steps.")
    
    if not args.legacy:
        # Check temperature
        temp = scene.sim.mpm_solver.particles.temp.to_numpy()
        mean_temp = temp.mean()
        print(f"Mean temperature at step 50: {mean_temp:.2f} K")

if __name__ == "__main__":
    main()
