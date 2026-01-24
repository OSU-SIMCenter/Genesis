import asyncio
import json
import traceback
import sys
import signal
import functools
import struct
import math
import time
from enum import Enum

import websockets
import logging
import torch
import numpy as np
import trimesh
import gstaichi as ti
import genesis as gs

import genesis.utils.particle as pu
from genesis.utils.misc import ti_to_numpy
from options import TeleopOptions
from agforge_builder import build_env, RobotXMLGenerator
from environment import AgilityForgeEnv

# Suppress websockets connection errors (caused by port checks like 'nc')
logging.getLogger("websockets").setLevel(logging.CRITICAL)

# Transformation constants
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




class StrikeState(Enum):
    IDLE = 0
    APPROACHING = 1
    HOLDING = 2
    PRESSING = 3
    RELEASE = 4

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

        # Strike Logic State
        self.strike_state = StrikeState.IDLE
        self.contact_L = False
        self.contact_R = False
        self.target_strain = 0.5  # Default, updated by client force param

        self.press_start_time = 0.0
        self.contact_width = 0.0

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
            # Only clamp slider/hinge, grippers managed by logic if striking
            new_qpos[:, :2] = torch.clamp(new_qpos[:, :2], self.dof_limits[0][:2], self.dof_limits[1][:2])
            self.qpos = new_qpos

    async def get_qpos(self):
        async with self.lock:
            return self.qpos.clone()

    async def trigger_strike(self, force_param):
        async with self.lock:
            if self.strike_state != StrikeState.IDLE:
                return
            
            gs.logger.info(f"Strike → APPROACHING (target_strain={force_param * 10:.2f})")
            # Reset contact flags
            self.contact_L = False
            self.contact_R = False
            self.strike_state = StrikeState.APPROACHING
            
            # Map force parameter (0-1) to target strain
            self.target_strain = force_param * 10.0
            
            self.robot.set_control_mode("VELOCITY_CONTROL")

    async def update_strike_logic(self):
        """Called every simulation step to handle strike state machine."""
        if self.strike_state == StrikeState.IDLE:
            return


        
        # --- APPROACHING STAGE ---
        if self.strike_state == StrikeState.APPROACHING:
            approach_speed = self.env.cfg.strike.approach_speed
            contact_threshold = self.env.cfg.strike.contact_force_threshold
            
            # Get resistance forces (projected along closing axis)
            force_L, force_R = self.robot.get_resistance_forces()
            gs.logger.info(f"Forces: L={force_L:.1f} R={force_R:.1f}")
            
            # Check for contact
            if not self.contact_L and force_L > contact_threshold:
                self.contact_L = True
                gs.logger.info(f"Contact L (force={force_L:.1f})")
                
            if not self.contact_R and force_R > contact_threshold:
                self.contact_R = True
                gs.logger.info(f"Contact R (force={force_R:.1f})")

            # Build velocity command: positive = closing
            vel_cmd = torch.zeros(4, device=self.env.device)
            vel_cmd[2] = 0.0 if self.contact_L else approach_speed
            vel_cmd[3] = 0.0 if self.contact_R else approach_speed
            self.robot.apply_velocity(vel_cmd, dofs_idx_local=torch.tensor([0, 1, 2, 3], device=self.env.device))
            
            # Transition Condition
            if self.contact_L and self.contact_R:
                self.strike_state = StrikeState.PRESSING
                self.press_start_time = time.time()
                
                # Record initial separation
                pos_L = self.robot.left_gripper.get_pos()
                pos_R = self.robot.right_gripper.get_pos()
                self.contact_width = torch.norm(pos_L - pos_R).item()
                gs.logger.info(f"Strike → PRESSING (width={self.contact_width:.4f})")
                
        # --- PRESSING STAGE ---
        elif self.strike_state == StrikeState.PRESSING:
            pressing_speed = self.env.cfg.strike.pressing_speed
            target_strain = self.target_strain  # Use client-provided target strain
            max_force = self.env.cfg.strike.max_force
            pressing_timeout = self.env.cfg.strike.pressing_timeout
            force_balance_gain = self.env.cfg.strike.force_balance_gain

            # Get current forces
            force_L, force_R = self.robot.get_resistance_forces()
            gs.logger.info(f"Forces: L={force_L:.1f} R={force_R:.1f}")
            
            # Calculate separation and strain
            pos_L = self.robot.left_gripper.get_pos()
            pos_R = self.robot.right_gripper.get_pos()
            current_width = torch.norm(pos_L - pos_R).item()
            
            if self.contact_width > 1e-6:
                current_strain = (self.contact_width - current_width) / self.contact_width
            else:
                current_strain = 0.0
                
            elapsed_time = time.time() - self.press_start_time

            # Termination Conditions
            stop_reason = None
            if current_strain >= target_strain:
                stop_reason = "Target Strain"
            elif force_L > max_force or force_R > max_force:
                stop_reason = "Max Force"
            elif elapsed_time > pressing_timeout:
                stop_reason = "Timeout"
                
            if stop_reason:
                gs.logger.info(f"Strike → RELEASE ({stop_reason}, strain={current_strain:.4f})")
                self.strike_state = StrikeState.RELEASE
                # Stop immediately
                vel_cmd = torch.zeros(4, device=self.env.device)
                self.robot.apply_velocity(vel_cmd, dofs_idx_local=torch.tensor([0, 1, 2, 3], device=self.env.device))
                return

            # Force balancing: reduce speed on the side with higher force
            imbalance = force_L - force_R
            correction = imbalance * force_balance_gain
            
            v_L = max(0.0, pressing_speed - correction)
            v_R = max(0.0, pressing_speed + correction)
            
            vel_cmd = torch.zeros(4, device=self.env.device)
            vel_cmd[2] = v_L
            vel_cmd[3] = v_R
            
            self.robot.apply_velocity(vel_cmd, dofs_idx_local=torch.tensor([0, 1, 2, 3], device=self.env.device))

        # --- RELEASE STAGE ---
        elif self.strike_state == StrikeState.RELEASE:
            release_speed = self.env.cfg.strike.pressing_speed
            contact_threshold = 10.0
            
            # ALWAYS apply opening velocity first - grippers need to physically separate
            v_open = -release_speed
            vel_cmd = torch.zeros(4, device=self.env.device)
            vel_cmd[2] = v_open
            vel_cmd[3] = v_open
            self.robot.apply_velocity(vel_cmd, dofs_idx_local=torch.tensor([0, 1, 2, 3], device=self.env.device))
            
            # Check if grippers have separated enough
            pos_L = self.robot.left_gripper.get_pos()
            pos_R = self.robot.right_gripper.get_pos()
            current_width = torch.norm(pos_L - pos_R).item()
            
            force_L, force_R = self.robot.get_resistance_forces()
            gs.logger.info(f"Forces: L={force_L:.1f} R={force_R:.1f}")

            # Exit only if forces low AND grippers have separated
            min_release_width = self.contact_width * 1.1
            if force_L < contact_threshold and force_R < contact_threshold and current_width > min_release_width:
                 # Stop velocities
                 vel_cmd = torch.zeros(4, device=self.env.device)
                 self.robot.apply_velocity(vel_cmd, dofs_idx_local=torch.tensor([0, 1, 2, 3], device=self.env.device))
                 
                 # Teleport to open (use direct qpos access to avoid lock issues)
                 current_qpos = self.qpos.clone()
                 current_qpos[:, 2] = self.gripper_open_pos
                 current_qpos[:, 3] = self.gripper_open_pos
                 
                 self.robot.set_control_mode("TELEPORT")
                 self.qpos = current_qpos
                 self.robot.apply_action(current_qpos)
                 
                 # Reset state
                 self.strike_state = StrikeState.IDLE
                 self.contact_L = False
                 self.contact_R = False
                 self.contact_width = 0.0
                 gs.logger.info("Strike → IDLE")
                 
                 # Save checkpoint after strike completes (not before)
                 # Note: This is outside the normal async flow, but update_strike_logic
                 # is already awaited from simulation_loop
                 await self.save_checkpoint()
                 return

        # --- HOLDING STAGE ---
        elif self.strike_state == StrikeState.HOLDING:
            # Just maintain position (handled by standard loop using self.qpos)
            pass 

    def _save_checkpoint_impl(self):
        """Internal helper to save checkpoint without lock (caller must hold lock)."""
        # Genesis SimState
        sim_state = self.env.scene.sim.get_state()
        sim_state.serializable()
        
        # Clear queried states in simulator to prevent memory leak
        # Simulator appends to _queried_states on every get_state()
        if hasattr(self.env.scene.sim, '_queried_states'):
            self.env.scene.sim._queried_states.clear()

        ckpt = {
            'sim_state': sim_state,
            'strike_state': self.strike_state,
            'qpos': self.qpos.clone()
        }
        self.checkpoints.append(ckpt)
        
        # Enforce max checkpoints limit
        if len(self.checkpoints) > 50:
            self.checkpoints.pop(0)

        gs.logger.info(f"Checkpoint saved ({len(self.checkpoints)} total)")

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
            self.strike_state = StrikeState.IDLE
            self.contact_L = False
            self.contact_R = False
            self.checkpoints = []
            self.contact_width = 0.0
            self.create_reconstructed_mesh()
            
            # Save initial state as first checkpoint
            self._save_checkpoint_impl()
            
            gs.logger.info("Simulation reset")

    async def save_checkpoint(self):
        async with self.lock:
            self._save_checkpoint_impl()

    async def load_checkpoint(self):
        """Undo to previous state. Since checkpoints are saved AFTER strikes,
        we need to pop the current state (discard) and load the previous one."""
        async with self.lock:
            if len(self.checkpoints) < 2:
                gs.logger.warning("No previous checkpoint to undo to")
                return

            # Pop current state (discard - it's the current state)
            self.checkpoints.pop()
            
            # Peek at previous state (don't pop - we want to keep it as current)
            ckpt = self.checkpoints[-1]
            
            # Restore SimState (MPM particles, etc.)
            self.env.scene.sim.reset(ckpt['sim_state'])

            # --- Synchronization for Visualization ---
            ti.sync()
            if hasattr(self.env.scene.sim.mpm_solver, 'update_render_fields'):
                self.env.scene.sim.mpm_solver.update_render_fields()
            else:
                self.env.scene.visualizer.update_visual_states()
            
            # Reset to IDLE state (don't restore strike_state - that would repeat the strike)
            self.strike_state = StrikeState.IDLE
            self.contact_L = False
            self.contact_R = False
            self.contact_width = 0.0
            
            # Restore base position but reset grippers to open
            self.qpos = ckpt['qpos'].clone()
            self.qpos[:, 2] = self.gripper_open_pos
            self.qpos[:, 3] = self.gripper_open_pos
            
            # Apply the restored position
            self.robot.set_control_mode("TELEPORT")
            self.robot.apply_action(self.qpos)
            
            self.create_reconstructed_mesh()
            gs.logger.info(f"Undo complete ({len(self.checkpoints)} checkpoints remaining)")

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
        """Reconstruct mesh during active strike stages."""
        if not self.recon_enabled or self.strike_state == StrikeState.IDLE:
            return

        self.frame_counter += 1
        if self.frame_counter % self.recon_frame_interval != 0:
            return

        self.create_reconstructed_mesh()
    


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
                 gs.logger.warning("No active particles for reconstruction")
                 return

            self.reconstructed_mesh = pu.particles_to_mesh(
                positions=particles,
                radius=radius,
                backend='splashsurf'
            )
            
        except Exception as e:
            gs.logger.error(f"Surface reconstruction failed: {e}")
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
        except Exception:
            pass
             
    if isinstance(arr, list):
        arr = np.array(arr)
        
    flat = arr.flatten().astype(dtype)
    return flat, len(flat)


async def simulation_loop(websocket, state: SharedState):
    """Runs the simulation and sends state updates to the client."""
    gs.logger.debug("Simulation loop started")
    
    try:
        while True:
            # 1. Clear accumulators


            # 2. Logic Update (State Machine)
            await state.update_strike_logic()
            
            # 3. Apply Actions based on State
            if state.strike_state == StrikeState.IDLE or state.strike_state == StrikeState.HOLDING:
                # Standard Teleoperation / Holding
                qpos = await state.get_qpos()
                state.robot.apply_action(qpos)
            
            # Clear accumulators (Before stepping, but after logic/reading!)
            # Logic (step N) reads forces from Step N-1.
            # Then we clear accumulator.
            # Then Step N calculates new forces.
            if hasattr(state.env.scene.sim.coupler, 'clear_link_coupling_forces'):
                state.env.scene.sim.coupler.clear_link_coupling_forces()
            
            # 4. Physics Step
            state.env.scene.step()
            
            # Update render fields for particle access
            if hasattr(state.env.scene.sim.mpm_solver, 'update_render_fields'):
                state.env.scene.sim.mpm_solver.update_render_fields()
            else:
                state.env.scene.visualizer.update_visual_states()

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
            await asyncio.sleep(1/60)

    except websockets.ConnectionClosed:
        gs.logger.debug("Simulation loop: client disconnected")
    except Exception as e:
        gs.logger.error(f"Simulation loop error: {e}")
        traceback.print_exc()
    except asyncio.CancelledError:
        gs.logger.debug("Simulation loop cancelled")
    finally:
        gs.logger.debug("Simulation loop finished")

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
        gs.logger.error(f"Viewer idle loop error: {e}")
        traceback.print_exc()


async def handle_client(websocket, state: SharedState, path=None):
    """Listens for client messages and updates the shared state."""
    gs.logger.info("Client connected")
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
                    if state.strike_state == StrikeState.IDLE:
                        translation = packet.get("translation", 0.0)
                        rotation = packet.get("rotation", 0.0)
                        slider_qpos, hinge_qpos = state.input_mapper.map_client_to_qpos(translation, rotation)
                        qpos[0, 0] = slider_qpos
                        qpos[0, 1] = hinge_qpos
                        await state.set_qpos(qpos)

                elif packet.get("request") == "strike":
                    if state.strike_state == StrikeState.IDLE:
                         force = packet.get("force", 0.5)
                         await state.trigger_strike(force)

                elif packet.get("request") == "temperature":
                    pass  # Placeholder for future implementation

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
    # Force line buffering for stdout so logs appear immediately (fixes buffering issue)
    sys.stdout.reconfigure(line_buffering=True)
    
    # Use print before gs.init() is called by build_env()
    print("Building simulation environment...")
    cfg = TeleopOptions()
    cfg.general.show_viewer = True
    env = build_env(cfg)  # gs.init() is called inside here
    shared_state = SharedState(env)
    shared_state.robot.set_control_mode("TELEPORT")

    # gs.logger is now available after build_env()
    gs.logger.info("Warming up simulation...")
    await shared_state.reset_simulation()

    # Warm-up strike to pre-compile velocity control and force reading kernels
    # This eliminates the JIT compilation pause on the first real strike
    gs.logger.info("Warming up strike kernels...")
    shared_state.robot.set_control_mode("VELOCITY_CONTROL")
    vel_cmd = torch.zeros(4, device=env.device)
    vel_cmd[2] = 1.0  # Small gripper velocity
    vel_cmd[3] = 1.0
    shared_state.robot.apply_velocity(vel_cmd, dofs_idx_local=torch.tensor([0, 1, 2, 3], device=env.device))
    env.scene.step()  # Step once to compile the kernels
    shared_state.robot.get_resistance_forces()  # Compile force reading
    shared_state.robot.set_control_mode("TELEPORT")
    await shared_state.reset_simulation()  # Reset back to initial state

    gs.logger.info("Server ready on port 8765")
    handler = functools.partial(handle_client, state=shared_state)

    
    # Create an event to signal shutdown
    stop_event = asyncio.Event()

    # Register signal handler for SIGINT (Ctrl+C)
    loop = asyncio.get_running_loop()
    def _handle_sigint():
        gs.logger.info("SIGINT received, shutting down...")
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
             gs.logger.debug("Cleaning up tasks")
             idle_task.cancel()
             try:
                 await idle_task
             except asyncio.CancelledError:
                 pass
             gs.logger.info("Shutdown complete")


if __name__ == "__main__":
    # Required for 'spawn' method to work correctly on Windows and compiled Linux,
    # preventing infinite recursive spawning of the application.
    import multiprocessing
    multiprocessing.freeze_support()
    # Increase recursion depth for Pyglet on macOS
    sys.setrecursionlimit(3000)
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        # Clean exit on Ctrl+C (especially for Windows where signal handlers don't work)
        pass

