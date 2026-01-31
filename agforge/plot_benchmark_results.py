import json
import os
import glob
import math

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

def load_results(data_dir="benchmark_data"):
    results = []
    pattern = os.path.join(os.path.dirname(__file__), data_dir, "result_*.json")
    files = glob.glob(pattern)
    
    if not files:
        print(f"No results found in {pattern}")
        return []

    print(f"Loading {len(files)} result files...")
    for fpath in files:
        try:
            with open(fpath, "r") as f:
                entry = json.load(f)
                results.append(entry)
        except Exception as e:
            print(f"Skipping {fpath}: {e}")
            
    return results

def print_summary_table(results):
    print("\n" + "="*80)
    print(f"{'Config Name':<30} | {'FPS':<8} | {'Steps':<6} | {'Phys(ms)':<8} | {'Recon(ms)':<8}")
    print("-" * 80)
    
    parsed_results = []
    
    for entry in results:
        try:
            name = entry["config_name"]
            metrics = entry["metrics"]
            fps = metrics["fps"]
            steps = metrics["steps"]
            
            res = {
                "name": name,
                "fps": fps,
                "steps": steps
            }
            
            # Profiling Stats (Phys vs Recon)
            # Prioritize 'mean' (new format) over 'avg' (old format)
            profiler_data = entry.get("profiler", {})
            
            # Physics: Sum of rigid solver steps + simulator steps (approximate top-level items)
            # Or just check for specific known expensive kernels
            phys_time = 0.0
            
            # Try to grab common physics items if they exist
            phys_items = ["rigid_step_1", "rigid_constraints", "rigid_step_2", "rigid_post_couple", "sim_substep", "sim_substep_pre_couple"]
            for item in phys_items:
                if item in profiler_data:
                    stat = profiler_data[item]
                    phys_time += stat.get("mean", stat.get("avg", 0))

            # Reconstruction: teleop_recon
            recon_time = 0.0
            if "teleop_recon" in profiler_data:
                stat = profiler_data["teleop_recon"]
                recon_time += stat.get("mean", stat.get("avg", 0))
            
            res["phys_time_avg"] = phys_time * 1000 # ms
            res["recon_time_avg"] = recon_time * 1000 # ms
            
            parsed_results.append(res)
            
            print(f"{name:<30} | {fps:<8.1f} | {steps:<6} | {res['phys_time_avg']:<8.2f} | {res['recon_time_avg']:<8.2f}")
            
        except KeyError as e:
            print(f"Skipping malformed entry: {e}")
            continue
            
    print("="*80 + "\n")
    return parsed_results

def plot_ascii_chart(parsed_results):
    if not parsed_results:
        return

    print("FPS Comparison (Higher is better):")
    max_fps = max(r["fps"] for r in parsed_results) if parsed_results else 1.0
    
    for r in parsed_results:
        bar_len = int((r["fps"] / max_fps) * 40)
        bar = "=" * bar_len
        print(f"{r['name']:<30} |[{bar:<40}]| {r['fps']:.2f}")
    print("\n")

def plot_matplotlib(parsed_results, output_dir="benchmark_data"):
    if not HAS_MATPLOTLIB or not parsed_results:
        return

    names = [r["name"] for r in parsed_results]
    fps_vals = [r["fps"] for r in parsed_results]
    phys_vals = [r["phys_time_avg"] for r in parsed_results]
    recon_vals = [r["recon_time_avg"] for r in parsed_results]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 1. FPS
    y_pos = range(len(names))
    ax1.barh(y_pos, fps_vals, align='center', color='skyblue')
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(names)
    ax1.invert_yaxis() 
    ax1.set_xlabel('FPS')
    ax1.set_title('Benchmark FPS Comparison')

    # 2. Component Timing
    width = 0.35
    y_pos = range(len(names))
    ax2.barh([y - width/2 for y in y_pos], phys_vals, width, label='Physics (Select Kernels)', color='salmon')
    ax2.barh([y + width/2 for y in y_pos], recon_vals, width, label='Reconstruction', color='lightgreen')
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(names)
    ax2.invert_yaxis()
    ax2.set_xlabel('Time (ms)')
    ax2.set_title('Average Component Time per Step')
    ax2.legend()
    
    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(__file__), output_dir, "benchmark_visualization.png")
    plt.savefig(out_path)
    print(f"[INFO] Generated graphical plot: {out_path}")

def main():
    results = load_results()
    if not results:
        print("No benchmark results found.")
        return
        
    parsed = print_summary_table(results)
    plot_ascii_chart(parsed)
    plot_matplotlib(parsed)

if __name__ == "__main__":
    main()
