import json
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# Organize data for plotting
# We want to show:
# 1. Total Step Time vs Resolution (for each substep count)
# 2. Total Step Time vs Substeps (for each resolution)
# 3. Stacked Bar Chart of Breakdown (Logic, Physics, Recon, IO) for each config

def parse_stats(stats_dict):
    """Extract key metrics from nested stats."""
    # Mapping of high level keys to specific profiler sections
    breakdown = {
        "Logic": ["teleop_logic"],
        "Recon": ["teleop_recon"],
        "IO": ["teleop_io"],
        "Physics (Compute)": ["substep_pre_couple", "couple", "substep_post_couple", "mpm_compute_F_tmp", "mpm_svd", "mpm_p2g", "mpm_g2p"],
        "Physics (Overhead)": ["process_input", "save_ckpt", "sensor_manager_step", "preprocess"],
    }
    
    # "substep" wraps the physics loop, but inside it calls "substep_pre/post_couple" etc.
    # We should be careful not to double count.
    # The 'total_frame' wraps everything.
    
    # Genesis profiler is flat (not hierarchical in storage, but usage is hierarchical).
    # If we sum up leaf nodes we get total.
    
    # Leaf nodes for Physics in MPM:
    # mpm_reset_grid_grad, mpm_compute_F_tmp, mpm_svd, mpm_p2g, g2p, couple, etc.
    # Note: 'substep' calls 'preprocess', 'substep_pre_couple', 'couple', 'substep_post_couple'.
    # 'substep_pre_couple' calls 'mpm_compute_F_tmp' etc.
    
    # So we should pick the HIGHEST level mutually exclusive blocks for the main chart.
    
    # Teleop Structure:
    # 1. teleop_logic
    # 2. env.scene.step() -> Calls 'process_input', Loop('substep'), 'save_ckpt', 'sensor_manager_step'
    #    -> 'substep' calls 'preprocess', 'couple', 'substep_pre', 'substep_post'
    # 3. teleop_recon
    # 4. teleop_io
    
    # So Breakdown:
    # Logic: teleop_logic
    # Recon: teleop_recon
    # IO: teleop_io
    # Physics: (Total Frame) - (Logic + Recon + IO) ? 
    # Or just use 'substep' (simulation loop) + 'process_input' ?
    
    # Let's try to grab 'substep' total.
    # Note: 'substep' is called N times. The 'total' in stats is sum of all calls.
    
    total_frame = stats_dict.get("total_frame", {}).get("total", 0)
    
    logic = stats_dict.get("teleop_logic", {}).get("total", 0)
    recon = stats_dict.get("teleop_recon", {}).get("total", 0)
    io = stats_dict.get("teleop_io", {}).get("total", 0)
    
    # Physics (Simulation Step)
    # This might be missing overheads outside 'substep' inside scene.step()
    # But 'substep' captures the heavy lifting.
    physics_substep = stats_dict.get("substep", {}).get("total", 0)
    physics_overhead = stats_dict.get("process_input", {}).get("total", 0) + stats_dict.get("rigid_solver_substep", {}).get("total", 0)
    
    # Recalculate physics to be precise:
    # If we have total_frame, physics = total_frame - logic - recon - io (approx)
    # But let's trust the 'substep' timer.
    
    return {
        "Logic": logic,
        "Physics": physics_substep + physics_overhead,
        "Recon": recon,
        "IO": io,
        "Other": max(0, total_frame - (logic + recon + io + physics_substep + physics_overhead))
    }

def generate_plots():
    try:
        with open("agforge/profiling_results.json", "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print("Error: agforge/profiling_results.json not found.")
        return

    # Extract configs
    configs = [f"R{d['config']['resolution']}_S{d['config']['substeps']}" for d in data]
    
    breakdown_data = [parse_stats(d["cProfile_stats"]) for d in data]
    
    categories = ["Logic", "Physics", "Recon", "IO", "Other"]
    
    # Prepare data for stacked bar
    bar_data = {cat: [] for cat in categories}
    for item in breakdown_data:
        for cat in categories:
            bar_data[cat].append(item[cat] * 1000) # Convert to ms
            
    x = np.arange(len(configs))
    width = 0.5
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bottom = np.zeros(len(configs))
    
    for cat in categories:
        values = bar_data[cat]
        ax.bar(x, values, width, bottom=bottom, label=cat)
        bottom += values
        
    ax.set_ylabel('Time (ms)')
    ax.set_title('Profiling Breakdown by Config')
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=45, ha='right')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig("agforge/profiling_breakdown.png")
    print("Generated agforge/profiling_breakdown.png")
    
    # --- Line Plots for Scaling ---
    # Filter by substeps to see scaling with Resolution (Particles)
    substeps_set = sorted(list(set(d['config']['substeps'] for d in data)))
    
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    
    for ss in substeps_set:
        subset = [d for d in data if d['config']['substeps'] == ss]
        subset.sort(key=lambda x: x['config']['resolution'])
        
        n_particles = [d['metrics']['n_particles'] for d in subset]
        times = [d['metrics']['avg_step_time_ms'] for d in subset]
        
        ax2.plot(n_particles, times, marker='o', label=f"Substeps={ss}")
        
    ax2.set_xlabel('Number of Particles')
    ax2.set_ylabel('Total Step Time (ms)')
    ax2.set_title('Performance Scaling with Particle Count')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig("agforge/scaling_particles.png")
    print("Generated agforge/scaling_particles.png")
    
    # --- Text Report ---
    print("\n" + "="*60)
    print(f"{'PERFORMANCE REPORT':^60}")
    print("="*60)
    
    for item, d in zip(breakdown_data, data):
        cfg_str = f"Resolution: {d['config']['resolution']}, Substeps: {d['config']['substeps']}"
        metrics = d['metrics']
        raw_stats = d['cProfile_stats']
        
        print(f"\nConfiguration: {cfg_str}")
        print(f"  Particles: {metrics['n_particles']}")
        print(f"  Total Step Time: {metrics['avg_step_time_ms']:.2f} ms")
        print(f"  FPS: {metrics['avg_fps']:.2f}")
        print("-" * 40)
        
        # Calculate percentages
        total_ms = metrics['avg_step_time_ms']
        
        def print_row(label, val_ms, indent=0):
            pct = (val_ms / total_ms) * 100
            print(f"{'  '*indent}├── {label:<20} {val_ms:>8.2f} ms ({pct:>5.1f}%)")

        print("  Breakdown:")
        print_row("Logic", item['Logic'], indent=1)
        print_row("Recon", item['Recon'], indent=1)
        print_row("IO", item['IO'], indent=1)
        print_row("Physics (Total)", item['Physics'], indent=1)
        
        # Physics Details (if available in raw stats)
        # These are internal genesis timers
        physics_keys = [
            ("mpm_reset_grid_grad", "Reset Grid"),
            ("mpm_compute_F_tmp", "F_tmp / SVD"), # SVD often inside here or separate? BaseMPM separates them
            ("mpm_svd", "SVD"),
            ("mpm_p2g", "P2G"),
            ("mpm_g2p", "G2P", "substep_post_couple"), # G2P is inside post_couple usually
            ("couple", "Coupling")
        ]
        
        for key, label in physics_keys:
            if key in raw_stats:
                val = raw_stats[key]['total'] * 1000 / d['cProfile_stats']['total_frame']['count'] # Per step avg
                # Wait, raw stats total is sum of all steps. 
                # breakdown_data values were NOT normalized by count?
                # parse_stats used total. But breakdown_data was for stacked BAR which compares totals?
                # Ah, parse_stats uses 'total'. 
                # But 'avg_step_time_ms' is (Duration / Steps).
                # raw_stats['total'] is sum over 'measure_steps'.
                # So we need to divide by measuring steps to get per-step ms.
                
                # Correction: The 'item' dict from parse_stats contains TOTAL accumulated time?
                # Let's check parse_stats implementation ... 
                # Yes: `logic = stats_dict.get("teleop_logic", {}).get("total", 0)`
                # So 'item' values are TOTALS for the whole run.
                pass

        # Re-calc per-step for report
        n_steps = raw_stats['total_frame']['count']
        def get_ms(key):
            return (raw_stats.get(key, {}).get('total', 0) / n_steps) * 1000.0

        print_row("  ↳ Reset Grid", get_ms("mpm_reset_grid_grad"), indent=2)
        print_row("  ↳ P2G", get_ms("mpm_p2g"), indent=2)
        print_row("  ↳ SVD", get_ms("mpm_svd"), indent=2)
        print_row("  ↳ Coupling", get_ms("couple"), indent=2)
        # G2P is usually in post_couple
        print_row("  ↳ Post-Couple/G2P", get_ms("substep_post_couple"), indent=2)
        
        if item['Other'] > 0:
             # 'Other' in item is total.
             other_ms = (item['Other'] / n_steps) * 1000.0
             print_row("Overhead/Other", other_ms, indent=1)
             
    print("\n" + "="*60)
