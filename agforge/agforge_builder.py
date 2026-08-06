import numpy as np
import torch
import genesis as gs

import platform
from agforge.options import AgilityForgeOptions, RobotOptions, GENERATED_ROBOT_XML_PATH
from agforge.environment import AgilityForgeEnv


class RobotXMLGenerator:
    """
    Generates a MuJoCo XML file for a robot designed to fit the simulation.
    
    The robot's dimensions and joint ranges are parametrically derived from 
    the cylinder's properties in the provided configuration.
    """
    def __init__(self, robot_cfg: RobotOptions):
        self.kp = robot_cfg.kp
        self.kv = robot_cfg.kv
        
        # Die half-extents. X (along the bar) and Z (across it) are the CONTACT
        # FOOTPRINT and now come from the real Tool2 geometry rather than being
        # scaled off the billet -- see RobotOptions.die_axial_width_m.
        #
        # Y stays on the legacy billet-radius rule on purpose: it is the die
        # thickness along the approach axis, and gripper_closed_y below derives
        # the closed gap from it, so changing Y would move the KINEMATICS
        # (how far the press can close) instead of the contact area.
        self.gripper_size = np.array([
            robot_cfg.die_axial_width_m * 0.5,   # X: along-bar contact width
            robot_cfg.cylinder_radius * 0.4,     # Y: thickness / approach axis
            robot_cfg.die_lateral_span_m * 0.5,  # Z: across-bar span
        ])

        # Induction coil visualizer metrics
        self.coil_offset_x = robot_cfg.coil_offset_x
        self.coil_length = robot_cfg.coil_length
        self.coil_radius = robot_cfg.coil_radius
        
        # Gripper movement is defined to open and close around the cylinder
        gripper_closed_y = robot_cfg.cylinder_radius - (self.gripper_size[1] * 0.5)
        gripper_open_y = robot_cfg.cylinder_radius * 2.5
        slide_distance = gripper_open_y - gripper_closed_y
        
        self.gripper_start_pos_y = gripper_open_y
        self.gripper_slide_range = np.array([0, slide_distance])
        
        # The main slider's range covers the length of the cylinder with massive padding
        # to ensure the offset induction coil can travel fully over the metal billet
        self.slider_range = np.array([
            robot_cfg.cylinder_pos[0] - robot_cfg.cylinder_height * 2.0 - abs(self.coil_offset_x),
            robot_cfg.cylinder_pos[0] + robot_cfg.cylinder_height * 2.0 + abs(self.coil_offset_x)
        ])
        
        # The hinge is positioned vertically centered with the cylinder
        self.hinge_pos_z = robot_cfg.cylinder_pos[2]

    def _to_str(self, arr, precision=4):
        """Formats a numpy array into an XML-compatible string."""
        return ' '.join(f'{x:.{precision}f}' for x in arr)

    def generate_content(self):
        """Populates the XML template with the derived parameters."""
        return f"""
<mujoco model="agforge_demo">
  <compiler angle="degree" inertiafromgeom="false"/>

  <asset>
    <material name="gripper" rgba="0.8 0.2 0.2 1"/>
  </asset>

  <worldbody>
    <light cutoff="100" diffuse="1 1 1" dir="0 0 -1" directional="true" pos="0 0 3"/>
    <body name="base_plate" pos="0 0 0">
      <joint name="x_slider" type="slide" axis="1 0 0" range="{self._to_str(self.slider_range)}"/>
      <inertial pos="0 0 0" mass="1e-7" diaginertia="1e-7 1e-7 1e-7"/>

      <!-- Translucent Induction Coil visualizer linked safely to the slider arm without physics collision -->
      <geom name="induction_coil_visual" type="cylinder" size="{self.coil_radius:.4f} {self.coil_length/2.0:.4f}" pos="{self.coil_offset_x:.4f} 0 {self.hinge_pos_z:.4f}" euler="0 90 0" rgba="1 0.5 0.0 0.4" contype="0" conaffinity="0"/>

      <body name="hinge_cylinder" pos="0 0 {self.hinge_pos_z:.4f}">
        <joint name="x_hinge" type="hinge" axis="1 0 0"/>
        <inertial pos="0 0 0" mass="1e-7" diaginertia="1e-7 1e-7 1e-7"/>

        <body name="clamp_bar" pos="0 0 0">
          <inertial pos="0 0 0" mass="1.0" diaginertia="0.01 0.01 0.01"/>
          <body name="left_gripper" pos="0 {-self.gripper_start_pos_y:.4f} 0">
            <inertial pos="0 0 0" mass="0.5" diaginertia="0.001 0.001 0.001"/>
            <joint name="left_gripper_slide" type="slide" axis="0 1 0" range="{self._to_str(self.gripper_slide_range)}"/>
            <geom type="box" size="{self._to_str(self.gripper_size)}" material="gripper"/>
          </body>
          <body name="right_gripper" pos="0 {self.gripper_start_pos_y:.4f} 0">
            <inertial pos="0 0 0" mass="0.5" diaginertia="0.001 0.001 0.001"/>
            <joint name="right_gripper_slide" type="slide" axis="0 -1 0" range="{self._to_str(self.gripper_slide_range)}"/>
            <geom type="box" size="{self._to_str(self.gripper_size)}" material="gripper"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>

  <actuator>
    <position joint="x_slider"           kp="{self.kp}" kv="{self.kv}"/>
    <position joint="x_hinge"            kp="{self.kp}" kv="{self.kv}"/>
    <position joint="left_gripper_slide"  kp="{self.kp}" kv="{self.kv}"/>
    <position joint="right_gripper_slide" kp="{self.kp}" kv="{self.kv}"/>
  </actuator>
</mujoco>
""".strip()

    def write_to_file(self):
        """Generates the XML content and writes it to the path specified in config.py."""
        content = self.generate_content()
        with open(GENERATED_ROBOT_XML_PATH, "w") as f:
            f.write(content)
        print(f"✅ Dynamically generated robot XML file: {GENERATED_ROBOT_XML_PATH}")


def build_env(cfg: AgilityForgeOptions) -> AgilityForgeEnv:
    """
    Builds the simulation scene using the provided configuration.
    """
    # --- Step 1: Dynamically generate the robot XML ---
    print(f"Generating robot XML ('{GENERATED_ROBOT_XML_PATH}') from config parameters...")
    generator = RobotXMLGenerator(robot_cfg=cfg.robot)
    generator.write_to_file()

    # --- Step 2: Initialize Genesis and create environment ---
    backend = gs.cpu
    
    # Check for GPU availability via Torch first
    if torch.cuda.is_available():
        try:
            # Test tensor allocation to ensure CUDA is actually usable
            _ = torch.tensor([1.0], device="cuda")
            print("✅ Torch reports CUDA is available. Attempting Genesis GPU init...")
            backend = gs.gpu
        except Exception as e:
            print(f"⚠️ Torch CUDA check failed ({e}). Defaulting to CPU.")
    elif platform.system() == "Darwin" and platform.machine() == "arm64":
         print("✅ Detected Apple Silicon (M-series). Attempting Genesis Metal init...")
         backend = gs.metal
    else:
        print("⚠️ Torch reports CUDA not available. Defaulting to CPU.")

    # Only init if not already initialized
    if not gs._initialized:
        try:
            gs.init(backend=backend, logging_level="info", performance_mode=cfg.performance_mode)
            print(f"Genesis initialized with backend: {backend}")
        except Exception as e:
            # If GPU/Metal init specifically fails (and we were trying it), we might want to fallback.
            if backend in [gs.gpu, gs.metal]:
                 print(f"⚠️ Genesis {backend} init failed ({e}). Attempting fallback to CPU...")
                 gs.init(backend=gs.cpu, logging_level="info", performance_mode=cfg.performance_mode)
            else:
                 raise e
    
    env = AgilityForgeEnv(cfg)
    
    return env