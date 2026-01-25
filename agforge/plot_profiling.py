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
    
    # Fallback: Sum granular physics keys if top-level 'substep' is missing
    # Note: If running in 'External' mode, 'substep' might be called N times per frame.
    # The 'total' value in stats is cumulative.
    if physics_substep == 0:
        granular_keys = ["mpm_reset_grid_grad", "mpm_compute_F_tmp", "mpm_svd", "mpm_p2g", "mpm_g2p", "substep_post_couple", "couple"]
        for k in granular_keys:
             physics_substep += stats_dict.get(k, {}).get("total", 0)

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

    # Extract configs keys for labels
    # Format: G{grid}_P{mult}_{mode}
    configs = []
    for d in data:
        c = d["config"]
        configs.append(f"G{c['grid_res']}_P{c['particle_mult']}_{c['stepping_mode']}")
    
    breakdown_data = [parse_stats(d["cProfile_stats"]) for d in data]
    
    categories = ["Logic", "Physics", "Recon", "IO", "Other"]
    
    # --- Plot 1: Overall Breakdown ---
    # Prepare data for stacked bar
    bar_data = {cat: [] for cat in categories}
    for item in breakdown_data:
        for cat in categories:
            bar_data[cat].append(item[cat] * 1000) # Convert to ms
            
    x = np.arange(len(configs))
    width = 0.5
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bottom = np.zeros(len(configs))
    
    for cat in categories:
        values = bar_data[cat]
        ax.bar(x, values, width, bottom=bottom, label=cat)
        bottom += values
        
    ax.set_ylabel('Frame Time (ms)')
    ax.set_title('Profiling Breakdown by Config')
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=45, ha='right')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig("agforge/profiling_breakdown.png")
    print("Generated agforge/profiling_breakdown.png")
    
    # --- Plot 2: Stepping Mode Comparison (Internal vs External) ---
    # We want to see the overhead of python calls.
    # Group by (Grid, ParticleMult)
    # Compare Frame Time for Internal vs External
    
    unique_params = set((d['config']['grid_res'], d['config']['particle_mult']) for d in data)
    unique_params = sorted(list(unique_params))
    
    internal_times = []
    external_times = []
    labels = []
    
    for g, p in unique_params:
        internal = next((d for d in data if d['config']['grid_res'] == g and d['config']['particle_mult'] == p and d['config']['stepping_mode'] == 'internal'), None)
        external = next((d for d in data if d['config']['grid_res'] == g and d['config']['particle_mult'] == p and d['config']['stepping_mode'] == 'external'), None)
        
        if internal and external:
            internal_times.append(internal['metrics']['avg_frame_time_ms'])
            external_times.append(external['metrics']['avg_frame_time_ms'])
            labels.append(f"G{g}_P{p}")
            
    if internal_times:
        x2 = np.arange(len(labels))
        width2 = 0.35
        
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        rects1 = ax2.bar(x2 - width2/2, internal_times, width2, label='Internal (Fused)')
        rects2 = ax2.bar(x2 + width2/2, external_times, width2, label='External (Split)')
        
        ax2.set_ylabel('Frame Time (ms)')
        ax2.set_title('Stepping Overhead: Internal vs External Loop')
        ax2.set_xticks(x2)
        ax2.set_xticklabels(labels)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig("agforge/stepping_overhead.png")
        print("Generated agforge/stepping_overhead.png")

    # --- Plot 3: Scaling (Particles vs Time) for Internal Mode ---
    # Only use 'internal' as baseline
    internal_data = [d for d in data if d['config']['stepping_mode'] == 'internal']
    internal_data.sort(key=lambda x: x['metrics']['n_particles'])
    
    if internal_data:
        n_parts = [d['metrics']['n_particles'] for d in internal_data]
        times = [d['metrics']['avg_frame_time_ms'] for d in internal_data]
        labels_pts = [f"G{d['config']['grid_res']}" for d in internal_data]
        
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        ax3.scatter(n_parts, times, color='red')
        for i, txt in enumerate(labels_pts):
            ax3.annotate(txt, (n_parts[i], times[i]))
            
        ax3.plot(n_parts, times, linestyle='--', color='gray', alpha=0.5)
        
        ax3.set_xlabel('Number of Particles')
        ax3.set_ylabel('Frame Time (ms)')
        ax3.set_title('Scaling: Frame Time vs Particle Count (Internal Mode)')
        ax3.grid(True)
        
        plt.tight_layout()
        plt.savefig("agforge/scaling_particles.png")
        print("Generated agforge/scaling_particles.png")

    # --- Text Report ---
    print("\n" + "="*60)
    print(f"{'PERFORMANCE REPORT':^60}")
    print("="*60)
    
    for item, d in zip(breakdown_data, data):
        c = d['config']
        cfg_str = f"Grid: {c['grid_res']}, P-Mult: {c['particle_mult']}, Mode: {c['stepping_mode']}"
        metrics = d['metrics']
        raw_stats = d['cProfile_stats']
        
        print(f"\nConfiguration: {cfg_str}")
        print(f"  Particles: {metrics['n_particles']}")
        print(f"  Frame Time: {metrics['avg_frame_time_ms']:.2f} ms")
        print(f"  FPS: {metrics['avg_fps']:.2f}")
        print("-" * 40)
        
        # Calculate percentages
        total_ms = metrics['avg_frame_time_ms']
        
        # NOTE: raw_stats count is Total Calls. 
        # In External mode, step() is called N times per frame.
        # But we aggregated 'total' time.
        # item[] contains TOTAL time for the RUN.
        # We need to divide item[] by 'measure_frames' (which we don't strictly have here, but we calculated avg_frame_time_ms).
        # Ah, item[] values are raw totals from profiler.
        # avg_frame_time_ms = total_duration / measure_frames.
        # So we can just use the item ratios.
        
        # But for absolute ms printing?
        # item['Logic'] is Sum of logic over M frames.
        # We need (item['Logic'] / item['TotalFrame']) * avg_frame_time_ms?
        # Or just (item['Logic'] / num_measured_frames).
        
        # We can extract num_measured_frames from 'total_frame' count in Internal mode.
        # In External mode, 'total_frame' count is M.
        # Wait, in benchmark script:
        # for _ in range(measure_frames): with profiler.time("total_frame"): ...
        # So 'total_frame' count is ALWAYS equal to measure_frames.
        
        measure_count = raw_stats.get('total_frame', {}).get('count', 1)
        
        def print_row(label, total_val_sec, indent=0):
            val_ms = (total_val_sec / measure_count) * 1000.0
            pct = (val_ms / total_ms) * 100
            print(f"{'  '*indent}├── {label:<20} {val_ms:>8.2f} ms ({pct:>5.1f}%)")

        print("  Breakdown:")
        print_row("Logic", item['Logic'], indent=1)
        print_row("Recon", item['Recon'], indent=1)
        print_row("IO", item['IO'], indent=1)
        print_row("Physics (Dispatch)", item['Physics'], indent=1)
        
        # Physics Details
        physics_keys = [
            ("mpm_reset_grid_grad", "Reset Grid"),
            ("mpm_compute_F_tmp", "F_tmp / SVD"),
            ("mpm_svd", "SVD"),
            ("mpm_p2g", "P2G"),
            ("mpm_g2p", "G2P"),
            ("couple", "Coupling"),
            ("substep_post_couple", "Post-Couple")
        ]
        
        for key, label in physics_keys:
            if key in raw_stats:
                val = raw_stats[key]['total']
                print_row(f"↳ {label}", val, indent=2)

        if item['Other'] > 0:
             print_row("Overhead/Other", item['Other'], indent=1)

    print("\n" + "="*60)

if __name__ == "__main__":
    generate_plots()
