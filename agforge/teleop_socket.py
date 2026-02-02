import asyncio
import json
import traceback
import sys
import signal
import functools
import struct
import math
import time

import websockets
import logging
import torch
import numpy as np

import genesis as gs

from agforge.options import TeleopOptions
from agforge.agforge_builder import build_env
from agforge.strike_controller import StrikeController, StrikeState

# Suppress websockets connection errors
logging.getLogger("websockets").setLevel(logging.CRITICAL)

# Configuration constants
TARGET_FPS = 60  # Target frame rate for the sim loop
FORCE_SCALE = 10.0  # Scale factor for client force/strain values (client sends 0-0.1, we want 0-1)


class InputMapper:
    """
    Maps client (Unity) coordinates to Genesis robot qpos.
    """
    def __init__(self, cylinder_height: float,
                 genesis_billet_end: float = -0.0383,
                 unity_base_offset: float = -0.59):
        self.genesis_end = genesis_billet_end
        self.genesis_start = genesis_billet_end + cylinder_height
        self.unity_end = -0.14 
        self.unity_start = 1.04 
    
    def map_client_to_qpos(self, translation: float, rotation: float):
        x = translation
        x1, y1 = self.unity_end, self.genesis_end
        x2, y2 = self.unity_start, self.genesis_start
        slider_qpos = y1 + (x - x1) * (y2 - y1) / (x2 - x1)
        
        hinge_qpos = math.radians(rotation)
        return slider_qpos, hinge_qpos


def _prepare_array(arr, dtype):
    """Convert array to flattened numpy array of specified dtype."""
    if arr is None or len(arr) == 0:
         return np.array([], dtype=dtype), 0

    if isinstance(arr, torch.Tensor):
        arr = arr.detach().cpu().numpy()
    elif hasattr(arr, 'to_numpy'): # Taichi fields
        arr = arr.to_numpy()
    elif hasattr(arr, 'numpy'):
        try:
            arr = arr.numpy()
        except Exception:
            pass
             
    if isinstance(arr, list):
        arr = np.array(arr)
        
    flat = arr.flatten().astype(dtype)
    return flat, len(flat)


async def simulation_loop(websocket, state: StrikeController):
    """Runs the simulation and sends state updates to the client."""
    gs.logger.debug("Simulation loop started")
    
    try:
        while True:
            # Determine if we need to step physics
            is_active = (state.strike_state != StrikeState.IDLE)
            needs_stabilization = (state.stabilization_steps > 0)
            
            # Helper dynamic attribute for manual inputs (socket only)
            has_input = getattr(state, 'new_input_received', False)
            
            # Check if we need to send mesh data (e.g., after undo/reset)
            pending_send = getattr(state, 'pending_mesh_send', False)
            
            should_step = is_active or needs_stabilization or has_input
            should_send = should_step or pending_send
            
            if not should_send:
                await asyncio.sleep(0.001)
                continue
            
            # Root of the hierarchy for this frame/step
            with state._profile("teleop_step"):
                # 1. atomic step (Logic + Physics + Render) - only if needed
                if should_step:
                    await state.step_simulation()
                
                # 2. Reconstruction & IO (always send when should_send is True)
                # Note: state.env.scene.profiling_options is accessible
                with state._profile("teleop_io"):
                    vertices, triangles, particles = await state.update_and_get_recon_data()
                    
                    v_flat, v_count = _prepare_array(vertices, np.float32)
                    t_flat, t_count = _prepare_array(triangles, np.int32)
                    p_flat, p_count = _prepare_array(particles, np.float32)
                    
                    header = {
                        "steps": [0],
                        "Pressure": 0,
                        "StressField": -1,
                        "is_pressing": state.strike_state != StrikeState.IDLE,
                        "counts": {
                            "vertices": v_count,
                            "faces": t_count,
                            "particles": p_count
                        }
                    }
                    header_json = json.dumps(header).encode('utf-8')
                    binary_body = v_flat.tobytes() + t_flat.tobytes() + p_flat.tobytes()
                    message = struct.pack('<I', len(header_json)) + header_json + binary_body
                    
                    await websocket.send(message)
            
            await asyncio.sleep(1/TARGET_FPS)
            
            # Cleanup flags
            if hasattr(state, 'new_input_received'):
                state.new_input_received = False
            if hasattr(state, 'pending_mesh_send'):
                state.pending_mesh_send = False
            
            if state.stabilization_steps > 0:
                state.stabilization_steps -= 1
                if state.stabilization_steps == 0:
                    gs.logger.info("Stabilization complete")

    except websockets.ConnectionClosed:
        gs.logger.debug("Simulation loop: client disconnected")
    except Exception as e:
        gs.logger.error(f"Simulation loop error: {e}")
        traceback.print_exc()
    except asyncio.CancelledError:
        gs.logger.debug("Simulation loop cancelled")
    finally:
        gs.logger.debug("Simulation loop finished")

async def viewer_idle_loop(state: StrikeController):
    """Keeps the viewer responsive when no client is connected."""
    try:
        while True:
            # Helper dynamic attribute
            is_connected = getattr(state, 'is_client_connected', False)
            
            if not is_connected:
                if hasattr(state.env.scene, 'visualizer') and state.env.scene.visualizer:
                     vis = state.env.scene.visualizer
                     if hasattr(vis, 'render'):
                         vis.render()
                     elif hasattr(vis, 'viewer'):
                         try:
                             if hasattr(vis.viewer, 'dispatch_events'):
                                 vis.viewer.dispatch_events()
                             if hasattr(vis.viewer, 'flip'):
                                 vis.viewer.flip()
                         except Exception:
                             pass
                     else:
                         vis.update_visual_states()
            
            await asyncio.sleep(0.02)
    except asyncio.CancelledError:
        pass
    except Exception as e:
        gs.logger.error(f"Viewer idle loop error: {e}")
        traceback.print_exc()


async def handle_client(websocket, state: StrikeController, path=None):
    """Listens for client messages and updates the shared state."""
    gs.logger.info("Client connected")
    state.is_client_connected = True
    
    # Input mapper is specific to this socket interface
    if not hasattr(state, 'input_mapper'):
        state.input_mapper = InputMapper(cylinder_height=state.env.cfg.robot.cylinder_height)

    producer_task = asyncio.create_task(simulation_loop(websocket, state))

    try:
        async for msg in websocket:
            try:
                packet = json.loads(msg)
                qpos = await state.get_qpos()

                if packet.get("request") == "reset":
                    await state.reset_simulation()

                elif packet.get("request") == "undo":
                    await state.load_checkpoint()

                elif packet.get("request") == "update":
                    if state.strike_state == StrikeState.IDLE:
                        translation = packet.get("translation", 0.0)
                        rotation = packet.get("rotation", 0.0)
                        slider_qpos, hinge_qpos = state.input_mapper.map_client_to_qpos(translation, rotation)
                        qpos[0, 0] = slider_qpos
                        qpos[0, 1] = hinge_qpos
                        await state.set_qpos(qpos)
                        state.new_input_received = True

                elif packet.get("request") == "strike":
                    if state.strike_state == StrikeState.IDLE:
                         raw_force = packet.get("force", 0.05)  # Default 0.05 -> 0.5 strain after scaling
                         
                         # Scale client value (0-0.1) to strain range (0-1)
                         scaled_strain = min(1.0, max(0.0, raw_force * FORCE_SCALE))
                         
                         gs.logger.info(f"Strike request: raw={raw_force}, scaled_strain={scaled_strain:.2f}")
                         await state.trigger_strike(scaled_strain)

            except json.JSONDecodeError:
                gs.logger.warning("Invalid JSON from client")
            except Exception as e:
                gs.logger.error(f"Error processing message: {e}")
    
    finally:
        gs.logger.debug("Cancelling simulation loop")
        producer_task.cancel()
        try:
            await producer_task
        except asyncio.CancelledError:
            pass
        except Exception as e:
            gs.logger.error(f"Task cancellation error: {e}")
        gs.logger.info("Client disconnected")
        state.is_client_connected = False


async def main():
    sys.stdout.reconfigure(line_buffering=True)
    
    print("Building simulation environment...")
    cfg = TeleopOptions()
    cfg.general.show_viewer = True
    
    # Enable MPM grid/boundary visualization
    cfg.vis.visualize_mpm_boundary = True
    cfg.vis.visualize_mpm_grid = True
    
    env = build_env(cfg)
    
    # Instantiate the new controller
    shared_state = StrikeController(env)
    shared_state.robot.set_control_mode("TELEPORT")
    
    # Add dynamic attributes needed for socket loop
    shared_state.new_input_received = False
    shared_state.is_client_connected = False

    gs.logger.info("Warming up simulation...")
    await shared_state.reset_simulation()

    gs.logger.info("Warming up strike kernels...")
    shared_state.robot.set_control_mode("VELOCITY_CONTROL")
    vel_cmd = torch.zeros(4, device=env.device)
    vel_cmd[2] = 1.0; vel_cmd[3] = 1.0
    shared_state.robot.apply_velocity(vel_cmd, dofs_idx_local=torch.tensor([0, 1, 2, 3], device=env.device))
    env.scene.step()
    shared_state.robot.get_resistance_forces()
    shared_state.robot.set_control_mode("TELEPORT")
    await shared_state.reset_simulation()
    
    # Reset profiler after warmup so only actual operation is profiled
    env.scene.profiling_options.profiler.reset()

    gs.logger.info("Server ready on port 8765")
    handler = functools.partial(handle_client, state=shared_state)
    
    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    def _handle_sigint():
        gs.logger.info("SIGINT received, shutting down...")
        stop_event.set()
        
    try:
        loop.add_signal_handler(signal.SIGINT, _handle_sigint)
    except NotImplementedError:
        pass

    async with websockets.serve(handler, "localhost", 8765):
        idle_task = asyncio.create_task(viewer_idle_loop(shared_state))
        try:
             await stop_event.wait()
        finally:
             gs.logger.debug("Cleaning up tasks")
             idle_task.cancel()
             try:
                 await idle_task
             except asyncio.CancelledError:
                 pass
             gs.logger.info("Shutdown complete")
             
             # Print all profiler visualizations if enabled
             if cfg.print_profiling_on_exit:
                 profiler = shared_state.env.scene.profiling_options.profiler
                 print("\n--- Detailed Profiling Stats (Rich Table - Full) ---")
                 profiler.rich_table(min_pct=0.0)
                 print("\n--- Detailed Profiling Hierarchy (ASCII Tree - >2%) ---")
                 profiler.print_tree(min_pct=2.0)
                 print("\n--- Profiling Hot-Spots (Flat - >1.5%) ---")
                 profiler.print_flat(sort_by="self", min_pct=1.5)

if __name__ == "__main__":
    asyncio.run(main())
