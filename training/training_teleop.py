import threading
import genesis as gs
import numpy as np
import torch
from pynput import keyboard

# Import configs and environment from the training folder
from config import TeleopConfig, GENERATED_ROBOT_XML_PATH
from config import CYLINDER_RADIUS, CYLINDER_HEIGHT, CYLINDER_POS
from environment import AgilityForgeEnv
from agforge_builder import RobotXMLGenerator

class KeyboardDevice:
    def __init__(self):
        self.pressed = set()
        self.lock = threading.Lock()
        self.listener = keyboard.Listener(on_press=self._down, on_release=self._up)
        self.action = np.zeros(3)

    def start(self):
        self.listener.start()

    def stop(self):
        self.listener.stop()
        self.listener.join()

    def _down(self, key):
        with self.lock:
            self.pressed.add(key)

    def _up(self, key):
        with self.lock:
            self.pressed.discard(key)

def build_scene_from_training_env():
    """
    Builds the simulation scene using the configuration from the training environment.
    """
    # --- Step 1: Load configurations ---
    cfg = TeleopConfig()

    # --- Step 2: Dynamically generate the robot XML ---
    print(f"Generating robot XML ('{GENERATED_ROBOT_XML_PATH}') from config parameters...")
    generator = RobotXMLGenerator(
        cylinder_radius=CYLINDER_RADIUS,
        cylinder_height=CYLINDER_HEIGHT,
        cylinder_pos=CYLINDER_POS,
        robot_cfg=cfg.robot
    )
    generator.write_to_file()

    # --- Step 3: Initialize Genesis and create environment ---
    gs.init(backend=gs.gpu, logging_level="info", performance_mode=cfg.performance_mode)
    
    env = AgilityForgeEnv(cfg.sim, cfg.env, cfg.general, cfg.robot)
    
    return env

def run():
    kb = KeyboardDevice()
    kb.start()

    env = build_scene_from_training_env()
    robot = env.robot

    # Key → (action_index, direction)
    # Note: Action space is [slider, hinge, gripper]
    controls = {
        keyboard.Key.up:       (0, 1.0),  # Slider forward
        keyboard.Key.down:     (0, -1.0), # Slider backward
        keyboard.Key.right:    (1, 1.0),  # Hinge right
        keyboard.Key.left:     (1, -1.0), # Hinge left
        keyboard.KeyCode.from_char("j"): (2, 1.0),  # Close gripper
        keyboard.KeyCode.from_char("k"): (2, -1.0), # Open gripper
    }

    print(
        "Teleop Controls:\n"
        "  ←/→: Hinge\n"
        "  ↑/↓: Slider\n"
        "  j/k: Grippers\n"
        "  b: Slow mode\n"
        "  u: Reset environment\n"
        "  esc: Quit"
    )
    
    obs, _ = env.reset()
    qpos = robot.entity.get_dofs_position()
    lower, upper = robot.entity.get_dofs_limit()

    while True:
        keys = kb.pressed.copy()

        if keyboard.Key.esc in keys:
            break

        if keyboard.KeyCode.from_char("u") in keys:
            obs, _ = env.reset()
            qpos = robot.entity.get_dofs_position()
            continue
        
        # Set speed modifier
        speed_modifiers = [0.25, 0.35, 0.5] if keyboard.KeyCode.from_char("b") in keys else [1., 1., 1.]

        # Update joint positions based on key presses
        for key, (index, direction) in controls.items():
            if key in keys:
                if index == 0: # Slider
                    qpos[0, index] += direction * env.robot_cfg.slider_speed * speed_modifiers[0]
                elif index == 1: # Hinge
                    qpos[0, index] += direction * env.robot_cfg.hinge_speed * speed_modifiers[1]
                elif index == 2: # Gripper
                    # Asymmetrical speed for opening/closing
                    gripper_speed = env.robot_cfg.gripper_speed * 2 if direction < 0 else env.robot_cfg.gripper_speed
                    # Both grippers move together
                    qpos[0, index] += direction * gripper_speed * speed_modifiers[2]
                    qpos[0, index + 1] += direction * gripper_speed * speed_modifiers[2]
        
        qpos = torch.clamp(qpos, lower, upper)

        # Apply the target joint positions
        robot.apply_action(qpos)
        
        # Manually step the simulation
        env.scene.step()

    kb.stop()

if __name__ == "__main__":
    run()