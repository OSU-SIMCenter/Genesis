import asyncio
import json
import traceback
import sys
import signal
import websockets
import logging

# Suppress websockets connection errors (caused by port checks like 'nc')
logging.getLogger("websockets").setLevel(logging.CRITICAL)

import torch
import numpy as np
import functools
import trimesh
import struct
import math
import time

import genesis.utils.particle as pu
from genesis.utils.misc import ti_to_numpy
import gstaichi as ti

from options import TeleopOptions
from agforge_builder import build_env, RobotXMLGenerator
from environment import AgilityForgeEnv

# Transformation constants (used by both InputMapper and SharedState)
TRANSFORM_SCALE = 31.275
TRANSFORM_HEIGHT_FACTOR = 0.375


class InputMapper:
    """
    Maps client (Unity) coordinates to Genesis robot qpos.
    
    Unity positions are calculated automatically from the transformation parameters
    (scale, cylinder_height) that are also used in _apply_transformation.
    """
    def __init__(self, cylinder_height: float,
                 genesis_billet_end: float = -0.0383,
                 unity_base_offset: float = -0.59): # 0.57, 1.16
        """
        Args:
            cylinder_height: Height of the cylinder in Genesis (meters)
            genesis_billet_end: Genesis x-position of the fixed end of the billet
            unity_base_offset: Base offset for Unity billet end position
        """
        # Genesis positions
        self.genesis_end = genesis_billet_end
        self.genesis_start = genesis_billet_end + cylinder_height
        
        # Unity positions (calculated from transformation parameters)
        self.unity_end = -0.14 #unity_base_offset + (TRANSFORM_HEIGHT_FACTOR * cylinder_height * TRANSFORM_SCALE)
        # unity_billet_length = cylinder_height * TRANSFORM_SCALE
        self.unity_start = 1.04 #self.unity_end + unity_billet_length
    
    def map_client_to_qpos(self, translation: float, rotation: float):
        """
        Map client inputs to robot qpos values.
        
        Args:
            translation: Client translation input (Unity x position)
            rotation: Client rotation input (degrees)
            
        Returns:
            tuple: (slider_qpos, hinge_qpos)
        """
        x = translation
        x1, y1 = self.unity_end, self.genesis_end
        x2, y2 = self.unity_start, self.genesis_start
        slider_qpos = y1 + (x - x1) * (y2 - y1) / (x2 - x1)
        
        hinge_qpos = math.radians(rotation)
        return slider_qpos, hinge_qpos


class SharedState:
    def __init__(self, env: AgilityForgeEnv):
        self.env = env
        self.robot = env.robot
        self.qpos = self.robot.entity.get_dofs_position()
        self.dof_limits = self.robot.entity.get_dofs_limit()
        self.lock = asyncio.Lock()
        
        # Input mapper configured with cylinder height from the environment
        self.input_mapper = InputMapper(cylinder_height=env.cfg.robot.cylinder_height)

        # Dynamically get gripper limits
        xml_generator = RobotXMLGenerator(robot_cfg=env.cfg.robot)
        self.gripper_closed_pos = xml_generator.gripper_slide_range[1]
        self.gripper_open_pos = xml_generator.gripper_slide_range[0]

        self.is_pressing = False
        self.press_start_time = 0.0
        self.press_duration = 4.0  # seconds

        # Surface reconstruction
        self.reconstructed_mesh = trimesh.Trimesh()
        self.recon_enabled = True
        self.recon_frame_interval = 2
        self.recon_particle_fraction = 1.0
        self.frame_counter = 0

        self.create_reconstructed_mesh()

        # Checkpoint storage
        self.checkpoints = []
        
        # Connection state for idle loop coordination
        self.is_client_connected = False

    async def set_qpos(self, new_qpos):
        async with self.lock:
            new_qpos[:, :2] = torch.clamp(new_qpos[:, :2], self.dof_limits[0][:2], self.dof_limits[1][:2])
            self.qpos = new_qpos

    async def get_qpos(self):
        async with self.lock:
            return self.qpos.clone()

    async def reset_simulation(self):
        async with self.lock:
            # Full reset
            self.env.reset()

            # --- Synchronization for Visualization ---
            ti.sync()
            if hasattr(self.env.scene.sim.mpm_solver, 'update_render_fields'):
                self.env.scene.sim.mpm_solver.update_render_fields()
            else:
                self.env.scene.visualizer.update_visual_states()

            self.qpos = self.robot.entity.get_dofs_position()
            self.is_pressing = False
            self.checkpoints = []
            self.create_reconstructed_mesh()
            print("Simulation reset.")

    async def save_checkpoint(self):
        async with self.lock:
            # Genesis SimState
            sim_state = self.env.scene.sim.get_state()
            sim_state.serializable()
            
            # Clear queried states in simulator to prevent memory leak
            # Simulator appends to _queried_states on every get_state()
            if hasattr(self.env.scene.sim, '_queried_states'):
                self.env.scene.sim._queried_states.clear()

            ckpt = {
                'sim_state': sim_state,
                'is_pressing': self.is_pressing,
                'press_start_time': self.press_start_time,
                'qpos': self.qpos.clone()
            }
            self.checkpoints.append(ckpt)
            print(f"Checkpoint saved. Total checkpoints: {len(self.checkpoints)}")

    async def load_checkpoint(self):
        async with self.lock:
            if not self.checkpoints:
                print("No checkpoints to undo.")
                return

            ckpt = self.checkpoints.pop()
            
            # Restore SimState
            self.env.scene.sim.reset(ckpt['sim_state'])

            # --- Synchronization for Visualization ---
            ti.sync()
            if hasattr(self.env.scene.sim.mpm_solver, 'update_render_fields'):
                self.env.scene.sim.mpm_solver.update_render_fields()
            else:
                self.env.scene.visualizer.update_visual_states()
            
            # Restore Aux State
            self.is_pressing = ckpt['is_pressing']
            self.press_start_time = ckpt['press_start_time']
            self.qpos = ckpt['qpos']
            
            # Force update robot physics state to match restored qpos if needed, 
            # though sim.reset() should handle it. 
            # We also ensure our local qpos tracks what we just restored.

            self.create_reconstructed_mesh()
            print("Checkpoint loaded (Undo).")

    def _apply_transformation(self, points):
        """Transform points from Genesis space to Unity space (supports Tensor and Numpy)."""
        cfg = self.env.cfg.robot
        translation = -(cfg.cylinder_pos + np.array([-TRANSFORM_HEIGHT_FACTOR * cfg.cylinder_height, 0, 0]))
        scale = (TRANSFORM_SCALE,) * 3
        
        if isinstance(points, torch.Tensor):
            points = points + torch.tensor(translation, dtype=torch.float32, device=points.device).view(1, 3)
            points = points * torch.tensor(scale, dtype=torch.float32, device=points.device).view(1, 3)
            points[:, 0] *= -1  # Flip x-axis
        else: # Numpy path for CPU mesh vertices
            points = points + translation.reshape(1, 3)
            points = points * np.array(scale).reshape(1, 3)
            points[:, 0] *= -1
            
        return points

    async def get_reconstructed_mesh_and_particles(self):
        """Return transformed mesh vertices, triangles, and particle positions."""
        points = self._apply_transformation(
            self.env.mpm_entity.get_particles_pos(envs_idx=0).squeeze(0)
        )
        vertices = self._apply_transformation(self.reconstructed_mesh.vertices)
        triangles = self.reconstructed_mesh.faces
        return vertices, triangles, points

    async def update_reconstructed_mesh(self):
        if not self.recon_enabled or not self.is_pressing:
            return

        self.frame_counter += 1
        if self.frame_counter % self.recon_frame_interval != 0:
            return

        self.create_reconstructed_mesh()
    
    async def update_press_state(self):
        """Updates the press/strike state based on elapsed time."""
        if not self.is_pressing:
            return

        elapsed = time.time() - self.press_start_time
        should_update = False
        
        # We need to fetch current qpos to modify it
        qpos = await self.get_qpos()

        if elapsed > self.press_duration:
            qpos[0, 2] = self.gripper_open_pos
            qpos[0, 3] = self.gripper_open_pos
            self.robot.set_control_mode("TELEPORT")
            self.is_pressing = False
            should_update = True
        elif elapsed > self.press_duration / 1.15:
            qpos[0, 2] = self.gripper_open_pos
            qpos[0, 3] = self.gripper_open_pos
            should_update = True
            
        if should_update:
            await self.set_qpos(qpos)

    def create_reconstructed_mesh(self):
        """Reconstruct surface mesh from active particles using splashsurf."""
        solver = self.env.scene.sim.mpm_solver
        
        # Get environment offset
        # Get particle positions and active mask
        # Explicit synchronization for Metal backend
        ti.sync()
        
        try:
            t0 = time.time()
            if hasattr(solver.particles_render.pos, 'to_numpy'):
                 particles = solver.particles_render.pos.to_numpy()[:, 0]
            else:
                 particles = ti_to_numpy(solver.particles_render.pos)[:, 0]
            
            # Get environment offset
            offset = self.env.scene.envs_offset[0]
            
            # Ensure it is on CPU and float32
            if hasattr(offset, 'cpu'):
                offset = offset.cpu().numpy()
            elif hasattr(offset, 'numpy'):
                offset = offset.numpy()
                
            particles = particles + offset
            
            if hasattr(solver.particles_render.active, 'to_numpy'):
                active = solver.particles_render.active.to_numpy()[:, 0].astype(bool)
            else:
                active = ti_to_numpy(solver.particles_render.active)[:, 0].astype(bool)
                
            particles = particles[active]
            
            # Subsample if needed
            if self.recon_particle_fraction < 1.0:
                num_particles = int(len(particles) * self.recon_particle_fraction)
                indices = np.random.choice(len(particles), num_particles, replace=False)
                particles = particles[indices]

            radius = solver.particle_radius
            
            # Check if particles is valid
            if len(particles) == 0:
                 print("Warning: No active particles for reconstruction.")
                 return

            # print(f"DEBUG: Reconstructing {len(particles)} particles. Backend: splashsurf (in-process)")
            self.reconstructed_mesh = pu.particles_to_mesh(
                positions=particles,
                radius=radius,
                backend='splashsurf'
            )
            # t1 = time.time()
            # print(f"DEBUG: Reconstruction took {t1-t0:.4f}s")
            
        except Exception as e:
            print(f"Error during surface reconstruction: {e}")
            self.reconstructed_mesh = trimesh.Trimesh()


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
        except:
             pass
             
    if isinstance(arr, list):
        arr = np.array(arr)
        
    flat = arr.flatten().astype(dtype)
    return flat, len(flat)


async def simulation_loop(websocket, state: SharedState):
    """Runs the simulation and sends state updates to the client."""
    print("Starting simulation loop...")
    
    try:
        while True:
            qpos = await state.get_qpos()
            state.robot.apply_action(qpos)
            state.env.scene.step()
            
            # Update render fields for particle access
            if hasattr(state.env.scene.sim.mpm_solver, 'update_render_fields'):
                state.env.scene.sim.mpm_solver.update_render_fields()
            else:
                state.env.scene.visualizer.update_visual_states()

            await state.update_press_state()
            await state.update_reconstructed_mesh()
            vertices, triangles, particles = await state.get_reconstructed_mesh_and_particles()
            
            # Prepare binary message
            v_flat, v_count = _prepare_array(vertices, np.float32)
            t_flat, t_count = _prepare_array(triangles, np.int32)
            p_flat, p_count = _prepare_array(particles, np.float32)
            
            header = {
                "steps": [0],
                "Pressure": 0,
                "StressField": -1,
                "is_pressing": state.is_pressing,
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
            await asyncio.sleep(1/60)

    except websockets.ConnectionClosed:
        print("Simulation loop stopped: Client disconnected.")
    except Exception as e:
        print(f"Error in simulation loop: {e}")
        traceback.print_exc()
    except asyncio.CancelledError:
        print("Simulation loop cancelled.")
    finally:
        print("Simulation loop task finished.")

async def viewer_idle_loop(state: SharedState):
    """Keeps the viewer responsive when no client is connected."""
    try:
        while True:
            if not state.is_client_connected:
                # If no client, we must step the viewer manually to keep window responsive
                if hasattr(state.env.scene, 'visualizer') and state.env.scene.visualizer:
                     vis = state.env.scene.visualizer
                     
                     # Try standard render which should poll events
                     if hasattr(vis, 'render'):
                         vis.render()
                     # Fallback/Additional: Pump Pyglet events if accessible
                     elif hasattr(vis, 'viewer'):
                         try:
                             if hasattr(vis.viewer, 'dispatch_events'):
                                 vis.viewer.dispatch_events()
                             if hasattr(vis.viewer, 'flip'):
                                 vis.viewer.flip()
                         except Exception:
                             pass
                     else:
                         # Last resort: just update states (unlikely to poll events)
                         vis.update_visual_states()
            
            # Sleep a bit to yield to other tasks
            await asyncio.sleep(0.02) # Slightly faster poll (50Hz)
    except asyncio.CancelledError:
        pass
    except Exception as e:
        print(f"Error in viewer idle loop: {e}")
        traceback.print_exc()


async def handle_client(websocket, state: SharedState, path=None):
    """Listens for client messages and updates the shared state."""
    print("Client connected. Ready to receive commands.")
    state.is_client_connected = True
    
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
                    if state.is_pressing:
                        continue
                    translation = packet.get("translation", 0.0)
                    rotation = packet.get("rotation", 0.0)
                    slider_qpos, hinge_qpos = state.input_mapper.map_client_to_qpos(translation, rotation)
                    qpos[0, 0] = slider_qpos
                    qpos[0, 1] = hinge_qpos
                    await state.set_qpos(qpos)

                elif packet.get("request") == "strike":
                    await state.save_checkpoint() # Save checkpoint before strike
                    force = (packet.get("force", 0.1) * 0.35) + 0.05
                    qpos[0, 2] = state.gripper_closed_pos * force * 10.0 * 1.3
                    qpos[0, 3] = state.gripper_closed_pos * force * 10.0 * 1.3
                    state.robot.set_control_mode("PD_CONTROL")
                    state.is_pressing = True
                    state.press_start_time = time.time()
                    await state.set_qpos(qpos)

                elif packet.get("request") == "temperature":
                    pass  # Placeholder for future implementation

            except json.JSONDecodeError:
                print("Invalid JSON received from client.")
            except Exception as e:
                print(f"Error processing client message: {e}")
    
    finally:
        print("Cancelling simulation loop...")
        producer_task.cancel()
        try:
            await producer_task
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"Error during producer task cancellation: {e}")
        print("Client disconnected cleanly.")
        state.is_client_connected = False


async def main():
    # Force line buffering for stdout so logs appear immediately (fixes buffering issue)
    sys.stdout.reconfigure(line_buffering=True)
    
    print("Building simulation environment...")
    cfg = TeleopOptions()
    cfg.general.show_viewer = True
    env = build_env(cfg)
    shared_state = SharedState(env)
    shared_state.robot.set_control_mode("TELEPORT")

    # Perform warm-up reset to ensure JIT/Allocations for reset are done
    print("Warming up simulation reset...")
    await shared_state.reset_simulation()

    print("Environment ready. Server listening on port 8765.")
    handler = functools.partial(handle_client, state=shared_state)

    
    # Create an event to signal shutdown
    stop_event = asyncio.Event()

    # Register signal handler for SIGINT (Ctrl+C)
    loop = asyncio.get_running_loop()
    def _handle_sigint():
        print("\nReceived SIGINT (Ctrl+C). Shutting down...")
        stop_event.set()
        
    try:
        loop.add_signal_handler(signal.SIGINT, _handle_sigint)
    except NotImplementedError:
        pass

    async with websockets.serve(handler, "localhost", 8765):
        # Start idle loop to keep window fresh when no client
        idle_task = asyncio.create_task(viewer_idle_loop(shared_state))
        try:
             # Wait until stop signal is received
             await stop_event.wait()
        finally:
             print("Cleaning up tasks...")
             idle_task.cancel()
             try:
                 await idle_task
             except asyncio.CancelledError:
                 pass
             print("Shutdown complete.")


if __name__ == "__main__":
    # Required for 'spawn' method to work correctly on Windows and compiled Linux,
    # preventing infinite recursive spawning of the application.
    import multiprocessing
    multiprocessing.freeze_support()
    # Increase recursion depth for Pyglet on macOS
    sys.setrecursionlimit(3000)
    asyncio.run(main())