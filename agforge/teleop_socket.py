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
    """Translates billet-local offsets (in physics meters) to Genesis robot qpos."""
    def __init__(self, billet_base_x: float = None):
        self.billet_base_x = billet_base_x
    
    def map_client_to_qpos(self, physics_offset: float, rotation: float):
        if self.billet_base_x is None:
            gs.logger.warning("Attempted to map inputs before dynamic mesh bounds were captured!")
            return 0.0, 0.0
            
        # Moving in a positive translation from Unity directly translates down the billet
        slider_qpos = self.billet_base_x + physics_offset
        hinge_qpos = math.radians(-rotation)  # hinge rotation = -billet rotation
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
            is_heating = getattr(state, 'thermal_enabled', False)
            
            # Check if we need to send mesh data (e.g., after undo/reset)
            pending_send = getattr(state, 'pending_mesh_send', False)
            
            should_step = is_active or needs_stabilization or has_input or is_heating
            should_send = should_step or pending_send
            
            if not should_send:
                await asyncio.sleep(0.001)
                continue
            
            # Root of the hierarchy for this frame/step
            with state._profile("teleop_step"):
                # 1. atomic step (Logic + Physics + Render) - only if needed
                if should_step:
                    await state.step_simulation()
                # IDLE SMOOTHING REMOVED:
                # We reverted to "Periodic Full Recon" strategy which is handled inside step().
                # No need to burn CPU smoothing a static mesh.
                
                # 2. Reconstruction & IO (always send when should_send is True)
                # Note: state.env.scene.profiling_options is accessible
                with state._profile("teleop_io"):
                    vertices, triangles, particles, vertices_temp = await state.update_and_get_recon_data()
                    
                    v_flat, v_count = _prepare_array(vertices, np.float32)
                    
                    # Safety net: If billet_base_x wasn't set from config (shouldn't happen),
                    # fall back to deriving it from the first mesh vertices.
                    if getattr(state, "input_mapper", None) and state.input_mapper.billet_base_x is None:
                        if v_count > 0:
                            state.input_mapper.billet_base_x = float(np.max(v_flat[0::3]))
                            gs.logger.warning(f"billet_base_x fallback from mesh (maxX): {state.input_mapper.billet_base_x}")
                            
                    t_flat, t_count = _prepare_array(triangles, np.int32)
                    p_flat, p_count = _prepare_array(particles, np.float32)
                    temp_flat, temp_count = _prepare_array(vertices_temp, np.float32)
                    
                    header = {
                        "stage": state.strike_state.name,
                        "is_pressing": state.strike_state != StrikeState.IDLE,
                        "thermal_enabled": getattr(state, 'thermal_enabled', False),
                        "checkpoint_count": len(state.checkpoints),
                        "counts": {
                            "vertices": v_count,
                            "faces": t_count,
                            "particles": p_count,
                            "temperatures": temp_count
                        }
                    }
                    header_json = json.dumps(header).encode('utf-8')
                    binary_body = v_flat.tobytes() + t_flat.tobytes() + p_flat.tobytes() + temp_flat.tobytes()
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
    
    # Initialize InputMapper with billet_base_x from known cylinder geometry.
    # This eliminates the race condition where a strike could arrive before the
    # first mesh reconstruction has finished.
    robot_cfg = state.env.cfg.robot
    billet_base_x = float(robot_cfg.cylinder_pos[0] + robot_cfg.cylinder_height / 2.0)
    state.input_mapper = InputMapper(billet_base_x=billet_base_x)
    gs.logger.info(f"InputMapper initialized from config (billet_base_x={billet_base_x:.6f})")

    producer_task = asyncio.create_task(simulation_loop(websocket, state))

    try:
        async for msg in websocket:
            try:
                packet = json.loads(msg)
                qpos = await state.get_qpos()

                if packet.get("request") == "reset":
                    if state.strike_state != StrikeState.IDLE:
                        gs.logger.warning(f"Ignoring reset during active strike ({state.strike_state.name})")
                    else:
                        await state.reset_simulation()

                elif packet.get("request") == "undo":
                    if state.strike_state != StrikeState.IDLE:
                        gs.logger.warning(f"Ignoring undo during active strike ({state.strike_state.name})")
                    else:
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

                elif packet.get("request") == "thermal_toggle":
                    enabled = packet.get("enabled", False)
                    await state.set_thermal_state(enabled)

                elif packet.get("request") == "strike":
                    if state.strike_state == StrikeState.IDLE:
                         # GUARANTEE PHYSICAL ALIGNMENT BEFORE STRIKE LOCK
                         translation = packet.get("translation", 0.0)
                         rotation = packet.get("rotation", 0.0)
                         slider_qpos, hinge_qpos = state.input_mapper.map_client_to_qpos(translation, rotation)

                         qpos[0, 0] = slider_qpos
                         qpos[0, 1] = hinge_qpos
                         await state.set_qpos(qpos)
                         
                         # Force the robot to this exact position NOW before the state
                         # transitions to APPROACHING (which skips the normal apply_action path)
                         state.robot.apply_action(qpos)

                         # PROCEED WITH STRIKE
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
    cfg.general.show_viewer = False
    
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
             
             # Safety flush: ensure any accumulated recording data is saved if we Ctrl+C
             if getattr(shared_state, 'recorder', None) and shared_state.recorder.is_recording:
                 gs.logger.info("Flushing final episode data before shutdown...")
                 shared_state.recorder.flush_episode(success_flag=True, language_instruction="Episode finished cleanly before exit.")
                 
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
