import asyncio
import json
import websockets
import torch
import numpy as np
import functools

from options import TeleopOptions
from agforge_builder import build_env, RobotXMLGenerator
from environment import AgilityForgeEnv

import math
import time

# --- 1. Shared State for Asynchronous Tasks ---
class InputMapper:
    """
    Configurable mapping from client inputs to robot qpos.
    Parameters can be tuned to match client coordinate system to robot joints.
    """
    def __init__(self):
        # Mapping parameters - these can be tuned
        # Unity: -0.329346, 0.83443
        # Genesis: 0.0296, -0.0384
        pass
    
    def lerp_two_points(self, x, x1=-0.329346, y1=-0.0384, x2=0.83443, y2=0.0296):
        return y1 + (x - x1) * (y2 - y1) / (x2 - x1)
        
    def map_client_to_qpos(self, translation, rotation):
        """
        Map client inputs to robot qpos values.
        
        Args:
            translation: Client translation input (x translation)
            rotation: Client rotation input (x euler angle)
            
        Returns:
            tuple: (slider_qpos, hinge_qpos)
        """
        slider_qpos = self.lerp_two_points(translation)
        hinge_qpos = math.radians(rotation)
        return slider_qpos, hinge_qpos

class SharedState:
    def __init__(self, env: AgilityForgeEnv):
        self.env = env
        self.robot = env.robot
        self.qpos = self.robot.entity.get_dofs_position()
        self.dof_limits = self.robot.entity.get_dofs_limit()
        print(f"Robot DOF limits: {self.dof_limits}")
        self.lock = asyncio.Lock()
        
        # Input mapper for client -> robot coordinate transformation
        self.input_mapper = InputMapper()

        # Dynamically get gripper limits
        xml_generator = RobotXMLGenerator(robot_cfg=env.cfg.robot)
        self.gripper_closed_pos = xml_generator.gripper_slide_range[1]
        self.gripper_open_pos = xml_generator.gripper_slide_range[0]

        self.is_pressing = False
        self.press_start_time = 0.0
        self.press_duration = 5.5  # seconds

    async def set_qpos(self, new_qpos):
        async with self.lock:
            # self.qpos = torch.clamp(new_qpos, self.dof_limits[0], self.dof_limits[1])
            new_qpos[:,:2] = torch.clamp(new_qpos[:,:2], self.dof_limits[0][:2], self.dof_limits[1][:2])
            self.qpos = new_qpos

    async def get_qpos(self):
        async with self.lock:
            return self.qpos.clone()

    async def get_particles(self):
        # This transformation can be defined as in unity_environment.py
        # to match client-side expectations.
        points = self.env.mpm_entity.get_particles_pos(envs_idx=0)
        cfg = self.env.cfg.robot
        translation=-(cfg.cylinder_pos + np.array([-0.2 * cfg.cylinder_height, 0, 0]))
        scale=(33.0, 33.0, 33.0)
        flip_axis=0
        points = points + torch.tensor(translation, device=points.device).view(1, 1, 3)
        points = points * torch.tensor(scale, device=points.device).view(1, 1, 3)
        points[:, :, flip_axis] *= -1
        return points.flatten().tolist()


# --- 2. Simulation and Producer Loop ---
async def simulation_loop(websocket, state: SharedState):
    """
    Runs the simulation continuously and sends state updates to the client.
    """
    print("Starting simulation loop...")
    while True:
        try:
            # Get the latest target positions
            qpos = await state.get_qpos()

            # Apply action and step simulation
            state.robot.apply_action(qpos)
            state.env.scene.step()

            # Send particle data back to the client in the requested format
            vertices = await state.get_particles()
            response = {
                "Vertices": vertices,
                "Steps": [0],
                "Temperatures": np.full(len(vertices) // 3, 293.0, dtype=float).tolist(),
                "Pressure": 0,
                "StressField": -1,
                "is_pressing": state.is_pressing,
            }
            await websocket.send(json.dumps(response))
            
            # Run at ~60Hz
            await asyncio.sleep(1/60)

        except websockets.ConnectionClosed:
            print("Simulation loop stopped: Client disconnected.")
            break
        except Exception as e:
            print(f"Error in simulation loop: {e}")
            break

# --- 3. Client Message Consumer ---
async def handle_client(websocket, state: SharedState, path=None):
    """
    Listens for client messages and updates the shared state.
    """
    print("Client connected. Ready to receive commands.")
    
    # Create and run the simulation loop as a concurrent task
    producer_task = asyncio.create_task(simulation_loop(websocket, state))

    try:
        async for msg in websocket:
            try:
                packet = json.loads(msg)
                # print(f"Received command: {packet}")

                qpos = await state.get_qpos()

                if state.is_pressing:
                    if time.time() - state.press_start_time > state.press_duration:
                        qpos[0, 2] = state.gripper_open_pos
                        qpos[0, 3] = state.gripper_open_pos
                        state.robot.set_control_mode("TELEPORT")
                        state.is_pressing = False
                    elif time.time() - state.press_start_time > state.press_duration/1.1:
                        qpos[0, 2] = state.gripper_open_pos
                        qpos[0, 3] = state.gripper_open_pos
                    await state.set_qpos(qpos)

                elif packet.get("request") == "update":
                    # Extract translation and rotation from the packet
                    translation = packet.get("translation", 0.0)
                    rotation = packet.get("rotation", 0.0)
                    
                    # Map client inputs to robot qpos using configurable mapping
                    slider_qpos, hinge_qpos = state.input_mapper.map_client_to_qpos(translation, rotation)
                    
                    # Update qpos with mapped values
                    qpos[0, 0] = slider_qpos  # x_slider
                    qpos[0, 1] = hinge_qpos   # x_hinge
                    await state.set_qpos(qpos)

                elif packet.get("request") == "strike":
                    qpos[0, 2] = state.gripper_closed_pos * packet.get("force", 0.1) * 10. * 1.3
                    qpos[0, 3] = state.gripper_closed_pos * packet.get("force", 0.1) * 10. * 1.3
                    
                    state.robot.set_control_mode("PD_CONTROL")
                    state.is_pressing = True
                    state.press_start_time = time.time()
                    await state.set_qpos(qpos)

                elif packet.get("request") == "temperature":
                    # Added for compatibility, but does nothing in this sim
                    pass


            except json.JSONDecodeError:
                print("Invalid JSON received from client.")
            except Exception as e:
                print(f"Error processing client message: {e}")
    
    finally:
        # Clean up the simulation task when the client disconnects
        producer_task.cancel()
        await asyncio.gather(producer_task, return_exceptions=True)
        print("Client disconnected and simulation task cancelled.")


# --- 4. Main Server Execution ---
async def main():
    print("Building the simulation environment... (This may take a moment)")
    cfg = TeleopOptions()
    # Ensure the viewer is off for server mode
    cfg.general.show_viewer = True
    env = build_env(cfg)
    shared_state = SharedState(env)
    shared_state.robot.set_control_mode("TELEPORT")

    print("Environment ready. Server listening on port 8765.")

    # The handler for websockets.serve must accept both websocket and path.
    handler = functools.partial(handle_client, state=shared_state)

    async with websockets.serve(handler, "localhost", 8765):
        await asyncio.Future()  # run forever

if __name__ == "__main__":
    asyncio.run(main())