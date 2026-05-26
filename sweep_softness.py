import argparse
import subprocess
import sys

def main():
    parser = argparse.ArgumentParser(description="Run a softness sweep over a benchmark recipe.")
    parser.add_argument("recipe", type=str, help="Path to the recipe to sweep over.")
    parser.add_argument("--viewer", action="store_true", help="Launch viewer for each sweep (not recommended for large sweeps).")
    args = parser.parse_args()

    # Values to sweep (1e-2, 5e-3, 1e-3, 5e-4, 1e-4)
    softness_values = [1e-2, 5e-3, 1e-3, 5e-4, 1e-4]

    print(f"Starting Softness Sweep for recipe: {args.recipe}")
    print(f"Sweeping over values: {softness_values}")
    print("-" * 50)

    for i, softness in enumerate(softness_values):
        print(f"\n[{i+1}/{len(softness_values)}] Running benchmark with softness = {softness:.2e}")
        
        # Build command array
        cmd = [
            "pixi", "run", "python", "-m", "agforge.benchmark_runner",
            args.recipe,
            "--softness", str(softness)
        ]
        if args.viewer:
            cmd.append("--viewer")
            
        try:
            # We use subprocess to isolate the genesis environment completely
            result = subprocess.run(cmd, check=True)
            print(f"✅ Completed run for softness = {softness:.2e}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Run failed for softness = {softness:.2e} with exit code {e.returncode}")
            sys.exit(e.returncode)
            
    print("\n" + "=" * 50)
    print("🎉 Sweep Complete!")
    print("You can plot the results using `pixi run python plot_all_recorded_data.py -e <episode_index>`")

if __name__ == "__main__":
    main()
