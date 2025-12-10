import asyncio
import json
import websockets
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
        self.press_duration = 7.0  # seconds

        # Surface reconstruction
        self.reconstructed_mesh = trimesh.Trimesh()
        self.recon_enabled = True
        self.recon_frame_interval = 2
        self.recon_particle_fraction = 1.0
        self.frame_counter = 0

        self.create_reconstructed_mesh()

    async def set_qpos(self, new_qpos):
        async with self.lock:
            new_qpos[:, :2] = torch.clamp(new_qpos[:, :2], self.dof_limits[0][:2], self.dof_limits[1][:2])
            self.qpos = new_qpos

    async def get_qpos(self):
        async with self.lock:
            return self.qpos.clone()

    def _apply_transformation(self, points):
        """Transform points from Genesis space to Unity space."""
        cfg = self.env.cfg.robot
        translation = -(cfg.cylinder_pos + np.array([-TRANSFORM_HEIGHT_FACTOR * cfg.cylinder_height, 0, 0]))
        scale = (TRANSFORM_SCALE,) * 3
        
        points = points + torch.tensor(translation, device=points.device).view(1, 3)
        points = points * torch.tensor(scale, device=points.device).view(1, 3)
        points[:, 0] *= -1  # Flip x-axis
        return points

    async def get_reconstructed_mesh_and_particles(self):
        """Return transformed mesh vertices, triangles, and particle positions."""
        points = self._apply_transformation(
            self.env.mpm_entity.get_particles_pos(envs_idx=0).squeeze(0)
        )
        vertices = self._apply_transformation(
            torch.tensor(self.reconstructed_mesh.vertices, device=self.env.device)
        )
        triangles = self.reconstructed_mesh.faces
        return vertices, triangles, points

    async def update_reconstructed_mesh(self):
        if not self.recon_enabled or not self.is_pressing:
            return

        self.frame_counter += 1
        if self.frame_counter % self.recon_frame_interval != 0:
            return

        self.create_reconstructed_mesh()
    
    def create_reconstructed_mesh(self):
        """Reconstruct surface mesh from active particles using splashsurf."""
        solver = self.env.scene.sim.mpm_solver
        
        # Get environment offset
        offset = self.env.scene.envs_offset[0]
        if hasattr(offset, 'cpu'):
            offset = offset.cpu().numpy()
        elif hasattr(offset, 'numpy'):
            offset = offset.numpy()
        
        # Get particle positions and active mask
        particles = ti_to_numpy(solver.particles_render.pos)[:, 0] + offset
        active = ti_to_numpy(solver.particles_render.active)[:, 0].astype(bool)
        particles = particles[active]

        # Subsample if needed
        if self.recon_particle_fraction < 1.0:
            num_particles = int(len(particles) * self.recon_particle_fraction)
            indices = np.random.choice(len(particles), num_particles, replace=False)
            particles = particles[indices]

        radius = solver.particle_radius
        self.reconstructed_mesh = pu.particles_to_mesh(
            positions=particles,
            radius=radius,
            backend='splashsurf'
        )


def _prepare_array(arr, dtype):
    """Convert array to flattened numpy array of specified dtype."""
    if isinstance(arr, torch.Tensor):
        arr = arr.cpu().numpy()
    if isinstance(arr, list):
        arr = np.array(arr)
    flat = arr.flatten().astype(dtype)
    return flat, len(flat)


async def simulation_loop(websocket, state: SharedState):
    """Runs the simulation and sends state updates to the client."""
    print("Starting simulation loop...")
    
    while True:
        try:
            qpos = await state.get_qpos()
            state.robot.apply_action(qpos)
            state.env.scene.step()
            ti.sync()
            
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
            break
        except Exception as e:
            print(f"Error in simulation loop: {e}")
            break


async def handle_client(websocket, state: SharedState, path=None):
    """Listens for client messages and updates the shared state."""
    print("Client connected. Ready to receive commands.")
    
    producer_task = asyncio.create_task(simulation_loop(websocket, state))

    try:
        async for msg in websocket:
            try:
                packet = json.loads(msg)
                qpos = await state.get_qpos()

                if state.is_pressing:
                    elapsed = time.time() - state.press_start_time
                    if elapsed > state.press_duration:
                        qpos[0, 2] = state.gripper_open_pos
                        qpos[0, 3] = state.gripper_open_pos
                        state.robot.set_control_mode("TELEPORT")
                        state.is_pressing = False
                    elif elapsed > state.press_duration / 1.15:
                        qpos[0, 2] = state.gripper_open_pos
                        qpos[0, 3] = state.gripper_open_pos
                    await state.set_qpos(qpos)

                elif packet.get("request") == "update":
                    translation = packet.get("translation", 0.0)
                    rotation = packet.get("rotation", 0.0)
                    slider_qpos, hinge_qpos = state.input_mapper.map_client_to_qpos(translation, rotation)
                    qpos[0, 0] = slider_qpos
                    qpos[0, 1] = hinge_qpos
                    await state.set_qpos(qpos)

                elif packet.get("request") == "strike":
                    force = packet.get("force", 0.1)
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
        producer_task.cancel()
        await asyncio.gather(producer_task, return_exceptions=True)
        print("Client disconnected.")


async def main():
    print("Building simulation environment...")
    cfg = TeleopOptions()
    cfg.general.show_viewer = True
    env = build_env(cfg)
    shared_state = SharedState(env)
    shared_state.robot.set_control_mode("TELEPORT")

    print("Environment ready. Server listening on port 8765.")
    handler = functools.partial(handle_client, state=shared_state)

    async with websockets.serve(handler, "localhost", 8765):
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())