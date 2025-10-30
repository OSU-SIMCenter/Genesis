import threading
import genesis as gs
import numpy as np
import torch
from pynput import keyboard

# Import configs and environment from the training folder
from options import TeleopOptions
from agforge_builder import build_env

# Profiling
import contextlib
from genesis.profiling.profiler import Profiler

class KeyboardDevice:
    def __init__(self):
        self.pressed = set()
        self.prev_pressed = set()
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
    
    def get_newly_pressed(self):
        """Returns keys that are pressed now but weren't in the previous frame"""
        with self.lock:
            newly_pressed = self.pressed - self.prev_pressed
            self.prev_pressed = self.pressed.copy()
            return newly_pressed

def run():
    kb = KeyboardDevice()
    kb.start()

    cfg = TeleopOptions()
    env = build_env(cfg)
    
    robot = env.robot
    profiling_cfg = env.cfg.profiling

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

    # Control mode: "continuous" or "incremental"
    control_mode = "incremental"
    incremental_multiplier = 15.0

    print(
        "Teleop Controls:\n"
        "  ←/→: Hinge\n"
        "  ↑/↓: Slider\n"
        "  j/k: Grippers\n"
        "  b: Slow mode\n"
        "  m: Toggle control mode (Continuous/Incremental)\n"
        "  u: Reset environment\n"
        "  esc: Quit\n"
        f"\nCurrent mode: {control_mode.upper()}"
    )
    
    obs, _ = env.reset()
    qpos = robot.entity.get_dofs_position()
    lower, upper = robot.entity.get_dofs_limit()

    profiler = Profiler(enabled=profiling_cfg.enabled)
    while True:
        with profiler.time("input_handling") if profiling_cfg.enabled else contextlib.suppress():
            keys = kb.pressed.copy()
            newly_pressed = kb.get_newly_pressed()

            if keyboard.Key.esc in keys:
                break

            # Toggle control mode
            if keyboard.KeyCode.from_char("m") in newly_pressed:
                control_mode = "incremental" if control_mode == "continuous" else "continuous"
                print(f"Control mode switched to: {control_mode.upper()}")

            if keyboard.KeyCode.from_char("u") in keys:
                obs, _ = env.reset()
                qpos = robot.entity.get_dofs_position()
                continue
            
            # Set speed modifier
            speed_modifiers = [0.25, 0.35, 0.5] if keyboard.KeyCode.from_char("b") in keys else [1., 1., 1.]

            # Determine which keys to process based on control mode
            active_keys = newly_pressed if control_mode == "incremental" else keys

            # Update joint positions based on key presses
            for key, (index, direction) in controls.items():
                if key in active_keys:
                    # Calculate movement multiplier
                    move_mult = incremental_multiplier if control_mode == "incremental" else 1.0
                    
                    if index == 0: # Slider
                        qpos[0, index] += direction * env.cfg.slider_speed * speed_modifiers[0] * move_mult
                    elif index == 1: # Hinge
                        qpos[0, index] += direction * env.cfg.hinge_speed * speed_modifiers[1] * move_mult
                    elif index == 2: # Gripper
                        # Asymmetrical speed for opening/closing
                        gripper_speed = env.cfg.gripper_speed * 2 if direction < 0 else env.cfg.gripper_speed
                        # Both grippers move together
                        qpos[0, index] += direction * gripper_speed * speed_modifiers[2] * move_mult
                        qpos[0, index + 1] += direction * gripper_speed * speed_modifiers[2] * move_mult
            
            qpos = torch.clamp(qpos, lower, upper)

        with profiler.time("action_application") if profiling_cfg.enabled else contextlib.suppress():
            # Apply the target joint positions
            robot.apply_action(qpos)
        
        with profiler.time("simulation_step") if profiling_cfg.enabled else contextlib.suppress():
            # Manually step the simulation
            env.scene.step()

    profiler.print()
    env.scene.profiling_options.profiler.print()
    kb.stop()

if __name__ == "__main__":
    run()