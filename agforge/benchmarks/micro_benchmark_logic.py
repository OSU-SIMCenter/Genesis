
import time
import torch
import genesis as gs
import numpy as np
import argparse

# Mocking or importing modules
from agforge.options import AgilityForgeOptions, TeleopOptions, StrikeOptions
from agforge.environment import AgilityForgeEnv
from agforge.strike_controller import StrikeController, StrikeState

def run_micro_benchmark(n_steps=1000, device="cuda"):
    print(f"Initializing Genesis on {device}...")
    gs.init(backend=gs.gpu, precision="32")
    
    # Minimal config using TeleopOptions to include 'strike'
    cfg = TeleopOptions(
        general=dict(record=False, show_viewer=False),
        sim=dict(dt=0.01), # Dummy DT
        strike=StrikeOptions(approach_speed=1.0) # Ensure strike options exist
    )
    # FIX: Options class unconditionally sets show_viewer=True in post_init, so we must force it here.
    cfg.general.show_viewer = False
    
    print("Creating Environment...")
    env = AgilityForgeEnv(cfg)
    # env.scene.build(n_envs=1) # Already built in __init__
    
    controller = StrikeController(env)
    
    # Set to APPROACHING to trigger logic
    # We cheat and set state directly to avoid async complexity if possible
    # But trigger_strike is async. We can just set the enum manually for the benchmark.
    controller.strike_state = StrikeState.APPROACHING
    controller.stage_start_time = time.time()
    
    print(f"Starting Logic Micro-benchmark ({n_steps} steps)...")
    
    # Warmup
    for _ in range(10):
        # We need to run the async method. In a sync script, we can just call it if it doesn't await much.
        # update_logic IS async. We need an event loop.
        pass
        
    import asyncio
    
    async def benchmark_loop():
        # Warmup
        for _ in range(50):
            await controller.update_logic()
        
        torch.cuda.synchronize()
        start_time = time.time()
        
        for i in range(n_steps):
            await controller.update_logic()
            
            # Artificially keep it in APPROACHING state to stress test that logic
            if controller.strike_state != StrikeState.APPROACHING:
                 controller.strike_state = StrikeState.APPROACHING
                 controller.stage_start_time = time.time()
        
        torch.cuda.synchronize()
        total_time = time.time() - start_time
        return total_time

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    total_time = loop.run_until_complete(benchmark_loop())
    
    avg_time_ms = (total_time / n_steps) * 1000
    print(f"Total Time: {total_time:.4f}s")
    print(f"Avg Time per Step: {avg_time_ms:.4f}ms")
    print(f"Est. Logic FPS: {1.0 / (total_time / n_steps):.1f}")
    
    # Print Profiler Stats
    print("\n--- Micro-benchmark Profile ---")
    env.scene.profiling_options.profiler.print_flat(sort_by="total")
    env.scene.profiling_options.profiler.print_tree()
    
    # Cleanup
    # env.scene.sim.close() # Removing this as it caused AttributeError

if __name__ == "__main__":
    run_micro_benchmark()
