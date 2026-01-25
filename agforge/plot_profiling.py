import glob
import os
import json
import numpy as np
import matplotlib.pyplot as plt

def parse_stats(stats_dict):
    """Extract key metrics from nested stats."""
    logic = stats_dict.get("teleop_logic", {}).get("total", 0)
    recon = stats_dict.get("teleop_recon", {}).get("total", 0)
    io = stats_dict.get("teleop_io", {}).get("total", 0)
    
    physics_substep = stats_dict.get("substep", {}).get("total", 0)
    
    if physics_substep == 0:
        granular_keys = ["mpm_reset_grid_grad", "mpm_compute_F_tmp", "mpm_svd", "mpm_p2g", "mpm_g2p", "substep_post_couple", "couple"]
        for k in granular_keys:
             physics_substep += stats_dict.get(k, {}).get("total", 0)

    physics_overhead = stats_dict.get("process_input", {}).get("total", 0) + stats_dict.get("rigid_solver_substep", {}).get("total", 0)
    
    total_frame = stats_dict.get("total_frame", {}).get("total", 0)
    
    return {
        "Logic": logic,
        "Physics": physics_substep + physics_overhead,
        "Recon": recon,
        "IO": io,
        "Other": max(0, total_frame - (logic + recon + io + physics_substep + physics_overhead))
    }

def generate_plots():
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark_data")
    data_files = glob.glob(os.path.join(data_dir, "*.json"))
    
    if not data_files:
        print(f"No benchmark data found in {data_dir}")
        return

    data = []
    print(f"Loading {len(data_files)} result files from {data_dir}...")
    for fpath in data_files:
        try:
            with open(fpath, "r") as f:
                d = json.load(f)
                # Verify it's a list (legacy) or dict (new single point)
                if isinstance(d, list):
                    data.extend(d)
                else:
                    data.append(d)
        except Exception as e:
            print(f"Skipping {fpath}: {e}")

    # Deduplicate or group? 
    # For now, we plot everything found. If duplicates exist (same config), they will just appear as multiple dots or bars.
    # Maybe good to sort by substeps for cleaner plotting.

    # Extract configs keys for labels
    configs = []
    
    # Sort data by Substeps (Low to High)
    data.sort(key=lambda x: x['config']['substeps'])
    
    for d in data:
        c = d["config"]
        # Label: "S=32 (L=4)"
        configs.append(f"S={c['substeps']}\n(L={c['loops_per_frame']})")
    
    breakdown_data = [parse_stats(d["cProfile_stats"]) for d in data]
    
    categories = ["Logic", "Physics", "Recon", "IO", "Other"]
    
    # --- Plot 1: Overall Breakdown ---
    bar_data = {cat: [] for cat in categories}
    for item in breakdown_data:
        for cat in categories:
            bar_data[cat].append(item[cat] * 1000) # Convert to ms
            
    x = np.arange(len(configs))
    width = 0.6
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bottom = np.zeros(len(configs))
    
    colors = {
        "Logic": "#FF9999",
        "Physics": "#66B2FF",
        "Recon": "#99FF99",
        "IO": "#FFCC99",
        "Other": "#CCCCCC"
    }
    
    for cat in categories:
        values = bar_data[cat]
        ax.bar(x, values, width, bottom=bottom, label=cat, color=colors.get(cat, None))
        bottom += values
        
    ax.set_ylabel('Frame Time (ms)')
    ax.set_title('Profiling Breakdown: Substeps Efficiency Scan')
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=0)
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig("agforge/profiling_breakdown.png")
    print("Generated agforge/profiling_breakdown.png")
    
    # --- Plot 2: Cost Curve ---
    substeps = [d['config']['substeps'] for d in data]
    times = [d['metrics']['total_sequence_time_ms'] for d in data]
    
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    ax2.plot(substeps, times, marker='o', linestyle='-', linewidth=2, color='green')
    
    ax2.set_xlabel('Substeps (Power of 2)')
    ax2.set_ylabel('Total Sequence Time (ms)')
    ax2.set_title('Performance vs Substep Granularity (Total Work)')
    ax2.grid(True, which="both", ls="-")
    ax2.set_xscale('log', base=2)
    ax2.set_xticks(substeps)
    ax2.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    
    plt.tight_layout()
    plt.savefig("agforge/substeps_curve.png")
    print("Generated agforge/substeps_curve.png")

    # --- Text Report ---
    print("\n" + "="*60)
    print(f"{'PERFORMANCE REPORT: SUBSTEP SWEEP':^60}")
    print("="*60)
    
    for item, d in zip(breakdown_data, data):
        c = d['config']
        cfg_str = f"Substeps: {c['substeps']}, Loops: {c['loops_per_frame']}, Grid: {c['grid_res']}, PMult: {c['particle_mult']}"
        metrics = d['metrics']
        
        print(f"\nConfiguration: {cfg_str}")
        print(f"  Sequence Time: {metrics['total_sequence_time_ms']:.2f} ms")
        print(f"  FPS (Eff): {metrics['avg_fps']:.2f}")
        print("-" * 40)

    print("\n" + "="*60)

if __name__ == "__main__":
    generate_plots()
