import asyncio
import json
import websockets
import torch
import numpy as np
import functools

from options import TeleopOptions
from agforge_builder import build_env, RobotXMLGenerator
from environment import AgilityForgeEnv

# --- 1. Shared State for Asynchronous Tasks ---
class SharedState:
    def __init__(self, env: AgilityForgeEnv):
        self.env = env
        self.robot = env.robot
        self.qpos = self.robot.entity.get_dofs_position()
        self.dof_limits = self.robot.entity.get_dofs_limit()
        self.lock = asyncio.Lock()

        # Dynamically get gripper limits
        xml_generator = RobotXMLGenerator(robot_cfg=env.cfg.robot)
        self.gripper_closed_pos = xml_generator.gripper_slide_range[1]
        self.gripper_open_pos = xml_generator.gripper_slide_range[0]


    async def set_qpos(self, new_qpos):
        async with self.lock:
            self.qpos = torch.clamp(new_qpos, self.dof_limits[0], self.dof_limits[1])

    async def get_qpos(self):
        async with self.lock:
            return self.qpos.clone()

    async def get_particles(self):
        # This transformation can be defined as in unity_environment.py
        # to match client-side expectations.
        points = self.env.mpm_entity.get_particles_pos(envs_idx=0)
        cfg = self.env.cfg.robot
        translation=-(cfg.cylinder_pos + np.array([-0.375 * cfg.cylinder_height, 0, 0]))
        scale=(50.0,)*3
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
                print(f"Received command: {packet}")

                qpos = await state.get_qpos()

                if packet.get("request") == "update":
                    # This now matches genesis_socket.py, which takes no arguments.
                    # The simulation loop will continue to send back state data.
                    pass

                elif packet.get("request") == "press":
                    # For compatibility with genesis_socket.py, this performs a quick clamp and release.
                    qpos[0, 2] = state.gripper_closed_pos
                    qpos[0, 3] = state.gripper_closed_pos
                    await state.set_qpos(qpos)
                    
                    await asyncio.sleep(0.5)
                    
                    qpos = await state.get_qpos()
                    qpos[0, 2] = state.gripper_open_pos
                    qpos[0, 3] = state.gripper_open_pos
                    await state.set_qpos(qpos)

                elif packet.get("request") == "release":
                    # A new command, not in genesis_socket.py, to open the grippers.
                    qpos[0, 2] = state.gripper_open_pos
                    qpos[0, 3] = state.gripper_open_pos
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
    cfg.general.show_viewer = False
    env = build_env(cfg)
    shared_state = SharedState(env)
    print("Environment ready. Server listening on port 8765.")

    # The handler for websockets.serve must accept both websocket and path.
    handler = functools.partial(handle_client, state=shared_state)

    async with websockets.serve(handler, "localhost", 8765):
        await asyncio.Future()  # run forever

if __name__ == "__main__":
    asyncio.run(main())