import torch
import math
import genesis as gs

class AgilityForgeEnv:
    # Constants
    CTRL_DT = 0.005
    NUM_ACTIONS = 3
    ACTION_LOWER_BOUNDS = [-0.2, -1.0, 0.09]
    ACTION_UPPER_BOUNDS = [-0.05, 1.0, 0.095]
    ACTION_DURATION_STEPS = 20
    RESET_DURATION_STEPS = 10
    
    # Target shape constants
    TARGET_X_MIN, TARGET_X_MAX = -0.01, 0.01
    TARGET_Y_MIN, TARGET_Y_MAX = 0.1, 0.5
    TARGET_Z_MIN, TARGET_Z_MAX = -0.01, 0.01
    
    # Fixed region bounds
    X_MIN_FIXED, X_MAX_FIXED = 0.15, 0.25
    Y_MIN_FIXED, Y_MAX_FIXED = -0.1, 0.1
    Z_MIN_FIXED, Z_MAX_FIXED = 0.2, 0.4
    
    # PD gains constants
    KP_GAINS = [1000, 1000, 1000, 1000]
    KV_GAINS = [100, 100, 100, 100]
    DEFAULT_JOINT_ANGLES = [0.0, 0.0, 0.0, 0.0]
    
    def __init__(self, num_envs, show_viewer=False):
        self.num_envs = num_envs
        self.num_actions = self.NUM_ACTIONS
        self.device = gs.device
        self.ctrl_dt = self.CTRL_DT

        # Define action bounds
        self.action_lower_bounds = torch.tensor(self.ACTION_LOWER_BOUNDS, device=self.device)
        self.action_upper_bounds = torch.tensor(self.ACTION_UPPER_BOUNDS, device=self.device)

        # Action scales
        self.action_duration_steps = self.ACTION_DURATION_STEPS
        self.reset_duration_steps = self.RESET_DURATION_STEPS
        self.initial_robot_qpos = None

        # Scene setup
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.ctrl_dt, substeps=64, requires_grad=True),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(2.0, 0.0, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                ) if show_viewer else gs.options.ViewerOptions(headless=True),
            rigid_options=gs.options.RigidOptions(dt=self.ctrl_dt),
            mpm_options=gs.options.MPMOptions(
                lower_bound=(-0.4, -0.15, 0.15),
                upper_bound=(0.25, 0.15, 0.45),
                gravity=(0.0, 0.0, 0.0),
                grid_density=100,
                particle_size=0.008
            ),
            vis_options=gs.options.VisOptions(
                show_world_frame=True,
                visualize_mpm_boundary=True
            ),
            show_viewer=show_viewer,
        )

        # Add ground
        self.scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))

        # Add robot
        self.robot = AgilityForgeManipulator(
            num_envs=self.num_envs,
            scene=self.scene,
        )
        # Initialize initial_robot_qpos here after robot is created
        self.initial_robot_qpos = torch.tensor(
            self.robot.default_joint_angles, dtype=torch.float32, device=self.device
        ).repeat(self.num_envs, 1)

        # Add MPM entity
        self.mpm_entity = self.scene.add_entity(
            material=gs.materials.MPM.ElastoPlastic(
                E=5.e5,
                nu=0.3,
                rho=100.,
                von_mises_yield_stress=280.
            ),
            morph=gs.morphs.Cylinder(
                radius=0.05,
                height=0.4,
                pos=(0.0, 0.0, 0.3),
                euler=(0.0, 90.0, 0.0)
            ),
            surface=gs.surfaces.Metal(
                color=(1.0, 0.5, 0.0),
                vis_mode="particle",
            ),
        )

        self.target_mpm_particles = None # This will store the target shape's particle positions

        # Add a visual representation of the target box (non-physical)
        target_box_pos = (-0.15, 0.0, 0.3) # Centered at the same position as the initial cylinder
        target_box_size = (0.3, 0.06, 0.06) # Double the half_extents to get full size
        self.target_box_entity = self.scene.add_entity(
            morph=gs.morphs.Box(size=target_box_size, pos=target_box_pos, fixed=True, collision=False),
            surface=gs.surfaces.Default(color=(1.0, 0.0, 0.0, 0.3)), # Red, semi-transparent
        )

        # Add a visual representation of the fixed region (non-physical)
        fixed_box_pos = ((self.X_MAX_FIXED+self.X_MIN_FIXED)/2, (self.Y_MAX_FIXED+self.Y_MIN_FIXED)/2, (self.Z_MAX_FIXED+self.Z_MIN_FIXED)/2)
        fixed_box_size = (self.X_MAX_FIXED-self.X_MIN_FIXED, self.Y_MAX_FIXED-self.Y_MIN_FIXED, self.Z_MAX_FIXED-self.Z_MIN_FIXED)
        self.fixed_box_entity = self.scene.add_entity(
            morph=gs.morphs.Box(size=fixed_box_size, pos=fixed_box_pos, fixed=True, collision=False),
            surface=gs.surfaces.Default(color=(0.0, 0.0, 1.0, 0.3)), # Blue, semi-transparent
        )

        self.scene.build(n_envs=num_envs)
        self.robot.set_pd_gains()
        self.initial_robot_qpos = self.robot.robot_entity.get_qpos().clone().detach()

        # Store initial MPM particle positions
        self.initial_mpm_particles_pos = self.mpm_entity.get_state().pos.clone().detach()

        # Create a mask for particles within the fixed region
        # Assuming single environment for initial setup, so taking the first element of the batch dimension
        mpm_pos = self.initial_mpm_particles_pos[0] 
        free_mask = torch.ones(mpm_pos.shape[0], dtype=torch.bool, device=self.device)

        # Check if particles are within the bounding box
        x_check = (mpm_pos[:, 0] >= self.X_MIN_FIXED) & (mpm_pos[:, 0] <= self.X_MAX_FIXED)
        y_check = (mpm_pos[:, 1] >= self.Y_MIN_FIXED) & (mpm_pos[:, 1] <= self.Y_MAX_FIXED)
        z_check = (mpm_pos[:, 2] >= self.Z_MIN_FIXED) & (mpm_pos[:, 2] <= self.Z_MAX_FIXED)

        fixed_particles_mask = x_check & y_check & z_check
        free_mask[fixed_particles_mask] = False

        # Apply the free mask to the MPM entity
        self.mpm_entity.set_free(free_mask)

        self.target_mpm_particles = self.initial_mpm_particles_pos.clone() # Initialize target_mpm_particles

    def reset(self):
        self.scene.reset()

        # Get current min/max of initial MPM particles using more efficient operations
        current_x_min = torch.amin(self.initial_mpm_particles_pos[:, :, 0])
        current_x_max = torch.amax(self.initial_mpm_particles_pos[:, :, 0])
        current_y_min = torch.amin(self.initial_mpm_particles_pos[:, :, 1])
        current_y_max = torch.amax(self.initial_mpm_particles_pos[:, :, 1])
        current_z_min = torch.amin(self.initial_mpm_particles_pos[:, :, 2])
        current_z_max = torch.amax(self.initial_mpm_particles_pos[:, :, 2])

        scale_x = (self.TARGET_X_MAX - self.TARGET_X_MIN) / (current_x_max - current_x_min)
        offset_x = self.TARGET_X_MIN - current_x_min * scale_x

        scale_y = (self.TARGET_Y_MAX - self.TARGET_Y_MIN) / (current_y_max - current_y_min)
        offset_y = self.TARGET_Y_MIN - current_y_min * scale_y

        scale_z = (self.TARGET_Z_MAX - self.TARGET_Z_MIN) / (current_z_max - current_z_min)
        offset_z = self.TARGET_Z_MIN - current_z_min * scale_z

        self.target_mpm_particles = self.initial_mpm_particles_pos.clone()
        self.target_mpm_particles[:, :, 0] = self.target_mpm_particles[:, :, 0] * scale_x + offset_x
        self.target_mpm_particles[:, :, 1] = self.target_mpm_particles[:, :, 1] * scale_y + offset_y
        self.target_mpm_particles[:, :, 2] = self.target_mpm_particles[:, :, 2] * scale_z + offset_z

    def apply_action_sequence(self, actions):
        for action in actions:
            clipped_action = torch.max(torch.min(action, self.action_upper_bounds), self.action_lower_bounds)
            
            current_qpos = self.robot.robot_entity.get_qpos()

            # Phase 1: Apply slider and hinge action
            target_qpos_slider_hinge = current_qpos.clone()
            target_qpos_slider_hinge[:, 0] = clipped_action[0]
            target_qpos_slider_hinge[:, 1] = clipped_action[1]
            self.robot.apply_action(target_qpos_slider_hinge)
            for _ in range(self.action_duration_steps):
                self.scene.step()
            
            current_qpos = self.robot.robot_entity.get_qpos()

            # Phase 2: Apply gripper action
            target_qpos_gripper = current_qpos.clone()
            target_qpos_gripper[:, 2] = clipped_action[2]
            target_qpos_gripper[:, 3] = clipped_action[2]
            self.robot.apply_action(target_qpos_gripper)
            for _ in range(self.action_duration_steps):
                self.scene.step()

            # Reset only hinge and gripper positions, keeping slider position
            current_qpos_before_reset = self.robot.robot_entity.get_qpos()
            reset_qpos = self.initial_robot_qpos.clone()
            reset_qpos[:, 0] = current_qpos_before_reset[:, 0]
            for _ in range(self.reset_duration_steps):
                self.robot.apply_action(reset_qpos)
                self.scene.step()

    def compute_loss(self):
        current_mpm_pos = self.mpm_entity.get_state().pos
        dist = torch.norm(current_mpm_pos - self.target_mpm_particles, p=2, dim=-1).mean(dim=-1)
        return dist

class AgilityForgeManipulator:
    def __init__(self, num_envs: int, scene: gs.Scene):
        self.device = gs.device
        self.scene = scene
        self.num_envs = num_envs

        morph = gs.morphs.MJCF(file="xml/agforge.xml")
        self.robot_entity = scene.add_entity(morph=morph)
        self.ee_link = self.robot_entity.get_link("clamp_bar")
        self.default_joint_angles = torch.tensor(AgilityForgeEnv.DEFAULT_JOINT_ANGLES, device=self.device)

    def set_pd_gains(self):
        self.robot_entity.set_dofs_kp(torch.tensor(AgilityForgeEnv.KP_GAINS, device=self.device))
        self.robot_entity.set_dofs_kv(torch.tensor(AgilityForgeEnv.KV_GAINS, device=self.device))

    def reset(self, envs_idx: torch.IntTensor):
        if not envs_idx:
            return
        default_joint_angles = self.default_joint_angles.repeat(len(envs_idx), 1)
        self.robot_entity.set_qpos(default_joint_angles, envs_idx=envs_idx)

    def apply_action(self, position: torch.Tensor):
        self.robot_entity.control_dofs_position(position=position)

    @property
    def ee_pose(self) -> torch.Tensor:
        pos, quat = self.ee_link.get_pos(), self.ee_link.get_quat()
        return torch.cat([pos, quat], dim=-1)

def main():
    # --- Hyperparameters ---
    NUM_ENVS = 1
    MAX_ITERATIONS = 1000
    SHOW_VIEWER = True 
    LEARNING_RATE = 1e-2
    ACTION_SEQUENCE_LENGTH = 4
    # ---------------------

    try:
        gs.init(logging_level="warning", precision="32", backend=gs.gpu)

        env = AgilityForgeEnv(num_envs=NUM_ENVS, show_viewer=SHOW_VIEWER)

        # Define the policy as a sequence of actions
        action_sequence = [torch.zeros(env.num_actions, requires_grad=True, device=gs.device) for _ in range(ACTION_SEQUENCE_LENGTH)]
        
        optimizer = torch.optim.Adam(action_sequence, lr=LEARNING_RATE)

        for i in range(MAX_ITERATIONS):
            env.reset()
            
            # Forward pass
            env.apply_action_sequence(action_sequence)
            loss = env.compute_loss()

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"Iteration {i}, Loss: {loss.item()}")
    except Exception as e:
        print(f"Error during training: {e}")
        raise

if __name__ == "__main__":
    main()
