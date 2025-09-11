import numpy as np
from config import GENERATED_ROBOT_XML_PATH

class RobotXMLGenerator:
    """
    Generates a MuJoCo XML file for a robot designed to fit the simulation.
    
    The robot's dimensions and joint ranges are parametrically derived from 
    the cylinder's properties in the provided configuration.
    """
    def __init__(self, cylinder_radius, cylinder_height, cylinder_pos, robot_cfg):
        self.kp = robot_cfg.kp
        self.kv = robot_cfg.kv
        
        # Gripper geometry is sized to handle the cylinder
        self.gripper_size = np.array([
            cylinder_radius * 0.5,  # X-dimension (height)
            cylinder_radius * 0.4,  # Y-dimension (width/thickness)
            cylinder_radius * 1.2,  # Z-dimension (depth)
        ])
        
        # Gripper movement is defined to open and close around the cylinder
        gripper_closed_y = cylinder_radius - (self.gripper_size[1] * 0.5)
        gripper_open_y = cylinder_radius * 2.5
        slide_distance = gripper_open_y - gripper_closed_y
        
        self.gripper_start_pos_y = gripper_open_y
        self.gripper_slide_range = np.array([0, slide_distance])
        
        # The main slider's range covers the length of the cylinder with padding
        self.slider_range = np.array([
            cylinder_pos[0] - cylinder_height * 0.75,
            cylinder_pos[0] + cylinder_height * 0.75
        ])
        
        # The hinge is positioned vertically centered with the cylinder
        self.hinge_pos_z = cylinder_pos[2]

    def _to_str(self, arr, precision=4):
        """Formats a numpy array into an XML-compatible string."""
        return ' '.join(f'{x:.{precision}f}' for x in arr)

    def generate_content(self):
        """Populates the XML template with the derived parameters."""
        return f"""
<mujoco model="agforge_demo">
  <compiler angle="degree" coordinate="local" inertiafromgeom="true"/>
  <option timestep="0.01" iterations="4" gravity="0 0 0"/>

  <default>
    <joint armature="0.1" damping="1" limited="true"/>
    <geom contype="1" conaffinity="1" condim="3" density="1000" friction="1 0.5 0.5"/>
  </default>

  <asset>
    <material name="gripper" rgba="0.8 0.2 0.2 1"/>
  </asset>

  <worldbody>
    <light cutoff="100" diffuse="1 1 1" dir="0 0 -1" directional="true" pos="0 0 3"/>
    <body name="base_plate" pos="0 0 0">
      <joint name="x_slider" type="slide" axis="1 0 0" range="{self._to_str(self.slider_range)}"/>
      <inertial pos="0 0 0" mass="1e-7" diaginertia="1e-7 1e-7 1e-7"/>

      <body name="hinge_cylinder" pos="0 0 {self.hinge_pos_z:.4f}">
        <joint name="x_hinge" type="hinge" axis="1 0 0" limited="false"/>
        <inertial pos="0 0 0" mass="1e-7" diaginertia="1e-7 1e-7 1e-7"/>

        <body name="clamp_bar" pos="0 0 0">
          <body name="left_gripper" pos="0 {-self.gripper_start_pos_y:.4f} 0">
            <joint name="left_gripper_slide" type="slide" axis="0 1 0" range="{self._to_str(self.gripper_slide_range)}"/>
            <geom type="box" size="{self._to_str(self.gripper_size)}" material="gripper"/>
          </body>
          <body name="right_gripper" pos="0 {self.gripper_start_pos_y:.4f} 0">
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