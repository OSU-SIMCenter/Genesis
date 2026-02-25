"""
Thermal integration visualization test.
Produces a PNG plot showing temperature evolution over time for two scenarios:
  1. Static hold: A 1000K steel box at rest — temperature should stay ~constant.
  2. Impact heating: A 1000K steel box dropped from height — temperature should
     rise slightly from adiabatic plastic work, then stabilize.

Each scenario runs in a separate subprocess because Genesis only allows one init per process.

Usage:
  python test_thermal_viz.py [--output OUTPUT_PATH]
"""
import numpy as np
import argparse
import subprocess
import sys
import json
import os

SCENARIO_SCRIPT_TEMPLATE = '''
import genesis as gs
import numpy as np
import json

gs.init(backend=gs.gpu, logging_level="warning")

scene = gs.Scene(
    show_viewer=False,
    mpm_options=gs.options.MPMOptions(
        use_legacy_solver=False,
        default_initial_temperature=1000.0,
        lower_bound=(0.0, 0.0, 0.0),
        upper_bound=(1.0, 1.0, 1.0),
    ),
)

material = gs.materials.MPM.ElastoPlastic(
    E=1e7,
    nu=0.3,
    rho=7850.0,
    use_von_mises=True,
    von_mises_yield_stress=1e6,
)

scene.add_entity(
    gs.morphs.Box(
        pos={box_pos},
        size=(0.15, 0.15, 0.15),
    ),
    material=material,
)

scene.build()

results = {{"steps": [], "mean": [], "max": [], "min": []}}

for i in range({n_steps}):
    scene.step()
    temp = scene.sim.mpm_solver.particles.temp.to_numpy()
    t = temp[0, :, 0]
    active = scene.sim.mpm_solver.particles_ng.active.to_numpy()[0, :, 0]
    t_active = t[active > 0]

    if len(t_active) > 0:
        results["steps"].append(i)
        results["mean"].append(float(np.mean(t_active)))
        results["max"].append(float(np.max(t_active)))
        results["min"].append(float(np.min(t_active)))

with open("{output_json}", "w") as f:
    json.dump(results, f)
print("Done")
'''


def run_single_scenario(name, box_pos, n_steps, output_json):
    """Run a single scenario in its own subprocess."""
    script = SCENARIO_SCRIPT_TEMPLATE.format(
        box_pos=box_pos, n_steps=n_steps, output_json=output_json
    )
    print(f"  Running: {name}...")
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, cwd=os.getcwd()
    )
    if result.returncode != 0:
        # Show last 300 chars of stderr for debugging
        print(f"  FAILED (exit {result.returncode})")
        stderr_tail = result.stderr.strip()[-300:]
        print(f"  stderr: ...{stderr_tail}")
        return None

    print(f"  OK — {result.stdout.strip()}")
    with open(output_json) as f:
        data = json.load(f)
    data["name"] = name
    for k in ["steps", "mean", "max", "min"]:
        data[k] = np.array(data[k])
    os.remove(output_json)
    return data


def main():
    parser = argparse.ArgumentParser(description="Thermal integration visualization")
    parser.add_argument("--output", default="thermal_test_results.png",
                        help="Output PNG path")
    args = parser.parse_args()

    print("=== Thermal Integration Visualization Tests ===\n")

    # Scenario 1: Box centered at z=0.5, should rest on floor after drop
    # With gravity, it falls and yields slightly — expect small temp rise
    static = run_single_scenario(
        name="Static Hold (centered)",
        box_pos=(0.5, 0.5, 0.5),
        n_steps=60,
        output_json="/tmp/_thermal_static.json",
    )

    # Scenario 2: Box starts higher at z=0.8, falls further, more deformation
    impact = run_single_scenario(
        name="Drop Impact (z=0.8)",
        box_pos=(0.5, 0.5, 0.8),
        n_steps=80,
        output_json="/tmp/_thermal_impact.json",
    )

    scenarios = [s for s in [static, impact] if s is not None]
    if not scenarios:
        print("\nAll scenarios failed!")
        sys.exit(1)

    # --- Plot ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — printing text summary instead.")
        for data in scenarios:
            drift = data["mean"][-1] - 1000.0
            print(f"\n  {data['name']}:")
            print(f"    Final mean: {data['mean'][-1]:.4f} K (drift: {drift:+.4f} K)")
            print(f"    Final max:  {data['max'][-1]:.4f} K")
            print(f"    Final min:  {data['min'][-1]:.4f} K")
        return

    fig, axes = plt.subplots(1, len(scenarios), figsize=(7 * len(scenarios), 5))
    if len(scenarios) == 1:
        axes = [axes]
    fig.suptitle("Thermal Integration Verification", fontsize=14, fontweight="bold")

    for ax, data in zip(axes, scenarios):
        ax.fill_between(data["steps"], data["min"], data["max"],
                         alpha=0.2, color="tab:red", label="min–max range")
        ax.plot(data["steps"], data["mean"], "tab:blue", linewidth=2, label="mean temp")
        ax.axhline(y=1000.0, color="gray", linestyle="--", alpha=0.5, label="initial (1000 K)")
        ax.set_xlabel("Simulation Step")
        ax.set_ylabel("Temperature (K)")
        ax.set_title(data["name"])
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)

        # Annotate final values
        final_mean = data["mean"][-1]
        final_max = data["max"][-1]
        ax.annotate(f"mean={final_mean:.2f}K\nmax={final_max:.2f}K",
                     xy=(data["steps"][-1], final_mean),
                     xytext=(-80, 20), textcoords="offset points",
                     fontsize=8, arrowprops=dict(arrowstyle="->", color="gray"),
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))

    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"\nPlot saved to: {args.output}")

    # Print summary
    print("\n=== Summary ===")
    for data in scenarios:
        drift = data["mean"][-1] - 1000.0
        print(f"  {data['name']}:")
        print(f"    Final mean: {data['mean'][-1]:.4f} K (drift: {drift:+.4f} K)")
        print(f"    Final max:  {data['max'][-1]:.4f} K")
        print(f"    Final min:  {data['min'][-1]:.4f} K")


if __name__ == "__main__":
    main()
