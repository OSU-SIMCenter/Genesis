import torch
import numpy as np
import genesis as gs
import sys
import os

from agforge.options import (
    AgilityForgeOptions,
    RobotOptions,
    GENERATED_ROBOT_XML_PATH,
)
from agforge.materials import JohnsonCookPlasticity
from agforge.profiling_util import teleop_profile
from agforge.env_knobs import env_bool, env_float

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except AttributeError:
        base_path = os.path.dirname(__file__)

    return os.path.join(base_path, relative_path)


def _billet_morph(robot_cfg):
    """The billet's initial condition: parametric cylinder (default) or the real scanned bar.

    The cylinder is the BOUNDING BOX of the hit-1-before scan -- which is how
    forge_common's REAL_STOCK_RADIUS_MM = 20.0 was picked, a bare constant with no
    derivation in that file. Measured, it starts the sim with +10.9% too much material
    (74,128 mm^3 sampled against the scan's 66,825), and volume error puts a hard ceiling on
    achievable IoU. Seeding from the scan itself measures +0.1%, and additionally reproduces
    the tapered ends, which no choice of radius can.

    AGF_BILLET_MESH=<path.obj> switches it on. Build the file with
    agforge/analysis/make_billet_mesh.py -- the mesh must be CENTRED on its bounding-box
    centroid (the morph's `pos` then places it exactly where the cylinder sat) and decimated:
    Genesis binds every visual vertex to every particle in one dense (V, P, 3) float32 array,
    so the raw 119k-vertex scan asks for 11.2 GiB. 8k faces costs -0.14% of volume, and at
    2 mm particle spacing the discarded detail was never resolvable anyway.

    Grid sizing is deliberately NOT touched. cylinder_diameter still drives
    base_grid_density, so dx, substep_dt and particle_size are identical across the switch
    and runs stay comparable; only the material being seeded changes.
    """
    mesh_path = os.environ.get("AGF_BILLET_MESH", "").strip()
    if not mesh_path:
        return gs.morphs.Cylinder(
            radius=robot_cfg.cylinder_radius,
            height=robot_cfg.cylinder_height,
            pos=robot_cfg.cylinder_pos,
            euler=robot_cfg.cylinder_euler,
        )
    if not os.path.isfile(mesh_path):
        raise FileNotFoundError("AGF_BILLET_MESH=%s does not exist" % mesh_path)
    # The scan is in millimetres; Genesis works in metres.
    scale = env_float("AGF_BILLET_MESH_SCALE", 0.001)
    # The prepared mesh already lies along +x, which is where euler=(0,90,0) puts the
    # cylinder's axis, so no rotation is applied by default.
    euler = tuple(float(v) for v in
                  os.environ.get("AGF_BILLET_MESH_EULER", "0,0,0").split(","))
    print("[agforge] billet seeded from MESH %s (scale %g, euler %s) -- NOT the nominal "
          "cylinder" % (mesh_path, scale, euler))
    return gs.morphs.Mesh(
        file=mesh_path,
        scale=scale,
        pos=tuple(float(v) for v in robot_cfg.cylinder_pos),
        euler=euler,
    )

class AgilityForgeManipulator:
    """Encapsulates the robot's properties and provides a clean action interface."""
    def __init__(self, scene: gs.Scene, robot_cfg: RobotOptions):
        self.scene = scene
        self.device = gs.device
        self.robot_cfg = robot_cfg
        morph = gs.morphs.MJCF(file=GENERATED_ROBOT_XML_PATH)
        material = gs.materials.Rigid(
            coup_softness=self.robot_cfg.coup_softness,
            gravity_compensation=1.0,
        )
        self.entity = scene.add_entity(morph=morph, material=material)
        self.ee_link = self.entity.get_link("clamp_bar")
        self.left_gripper = self.entity.get_link("left_gripper")
        self.right_gripper = self.entity.get_link("right_gripper")
        
        # Cache indices for fast access
        self._left_gripper_link_idx = self.entity.get_link("left_gripper").idx
        self._right_gripper_link_idx = self.entity.get_link("right_gripper").idx
        self._coupler = None # Will cache on first access or setup
        
        # Calculate indices relative to the entity start for force lookup
        self.left_gripper_idx = self.left_gripper.idx - self.entity.link_start
        self.right_gripper_idx = self.right_gripper.idx - self.entity.link_start
        
        self.default_joint_angles = torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=self.device)
        self.control_mode = "PD_CONTROL"  # Initial control mode
        
        # Pre-allocated tensors for resistance force calculation (Issue #16 optimization)
        self._squeeze_dir_L = torch.tensor([0.0, 1.0, 0.0], device=self.device)
        self._squeeze_dir_R = torch.tensor([0.0, -1.0, 0.0], device=self.device)
        self._resistance_quat_cache: torch.Tensor | None = None
        self._global_squeeze_L_cache: torch.Tensor | None = None
        self._global_squeeze_R_cache: torch.Tensor | None = None

    def set_pd_gains(self):
        """Sets the Proportional-Derivative controller gains for the robot's joints."""
        self.entity.set_dofs_kp(torch.full((4,), self.robot_cfg.kp, dtype=torch.float32, device=self.device))
        self.entity.set_dofs_kv(torch.full((4,), self.robot_cfg.kv, dtype=torch.float32, device=self.device))

    def set_clamp_force_range(self):
        """Sets the force range for the gripper joints."""
        max_force = self.robot_cfg.clamp_force
        # The gripper DOFs are the last two
        dofs_idx = torch.tensor([2, 3], dtype=torch.int32, device=self.device)
        lower = torch.full((2,), -max_force, dtype=torch.float32, device=self.device)
        upper = torch.full((2,), max_force, dtype=torch.float32, device=self.device)
        self.entity.set_dofs_force_range(lower, upper, dofs_idx_local=dofs_idx)

    def set_control_mode(self, mode: str):
        """Sets the control mode for the manipulator."""
        if mode not in ["PD_CONTROL", "TELEPORT", "VELOCITY_CONTROL"]:
            raise ValueError("Invalid control mode. Must be 'PD_CONTROL', 'TELEPORT', or 'VELOCITY_CONTROL'.")
        self.control_mode = mode

    def reset(self, envs_idx: torch.Tensor):
        """Resets the robot's joint positions to default for specified environments."""
        if len(envs_idx) > 0:
            qpos_to_set = self.default_joint_angles.expand(len(envs_idx), -1)
            self.entity.set_qpos(qpos_to_set, envs_idx=envs_idx)

    def apply_action(self, action: torch.Tensor, dofs_idx_local=None):
        """Applies an action based on the current control mode."""
        if self.control_mode == "PD_CONTROL":
            with teleop_profile(self.scene, "teleop_apply_action_control"):
                self.entity.control_dofs_position(position=action, dofs_idx_local=dofs_idx_local)
        elif self.control_mode == "TELEPORT":
            with teleop_profile(self.scene, "teleop_apply_action_control"):
                self.entity.control_dofs_position(position=action, dofs_idx_local=dofs_idx_local)
            with teleop_profile(self.scene, "teleop_apply_action_set_pos"):
                self.entity.set_dofs_position(action, dofs_idx_local=dofs_idx_local)
        elif self.control_mode == "VELOCITY_CONTROL":
            with teleop_profile(self.scene, "teleop_apply_action_control"):
                self.apply_velocity(velocity=action, dofs_idx_local=dofs_idx_local)

    def apply_velocity(self, velocity: torch.Tensor, dofs_idx_local=None):
        """
        Applies velocity control by setting both the physical velocity state and the PD controller target.
        This provides 'stiff' velocity tracking suitable for the approaching stage.
        """
        self.entity.set_dofs_velocity(velocity, dofs_idx_local=dofs_idx_local)
        self.entity.control_dofs_velocity(velocity, dofs_idx_local=dofs_idx_local)

    def get_gripper_net_contact_force(self):
        """
        Returns the net external contact force on the left and right grippers.
        Returns:
            torch.Tensor: Shape (n_envs, 2, 3) where dim 1 is [Left, Right]
        """
        # (n_envs, n_links, 3)
        with teleop_profile(self.scene, "teleop_logic_resistance_rigid_pull"):
            net_forces = self.entity.get_links_net_contact_force()
        # Select gripper links
        # Assuming single env for now or preserving batch dim
        if net_forces.dim() == 2: # (n_links, 3) - happens if n_envs=0 or something weird, but likely (n_envs, n_links, 3)
             return torch.stack([net_forces[self.left_gripper_idx], net_forces[self.right_gripper_idx]], dim=0).unsqueeze(0)
             
        # (n_envs, n_links, 3)
        # (n_envs, n_links, 3)
        rigid_forces = torch.stack([
            net_forces[:, self.left_gripper_idx, :],
            net_forces[:, self.right_gripper_idx, :]
        ], dim=1)
        
        # Add MPM forces if available (now from RigidEntity native API)
        try:
            with teleop_profile(self.scene, "teleop_logic_resistance_mpm_pull"):
                mpm_forces = self.entity.get_links_mpm_force()
            # mpm_forces shape: (n_envs, n_links, 3) or (n_links, 3)
            if mpm_forces.dim() == 2:
                 mpm_stack = torch.stack([mpm_forces[self.left_gripper_idx], mpm_forces[self.right_gripper_idx]], dim=0).unsqueeze(0)
            else:
                 mpm_stack = torch.stack([
                     mpm_forces[:, self.left_gripper_idx, :],
                     mpm_forces[:, self.right_gripper_idx, :]
                 ], dim=1)
            
            return rigid_forces + mpm_stack
        except AttributeError:
            pass
             
        return rigid_forces

    def _rotate_vector_by_quat(self, vector, quat):
        """
        Rotates a vector by a quaternion.
        vector: (3,) or (N, 3)
        quat: (4,) [w, x, y, z] or (N, 4)  # Genesis uses wxyz format
        """
        # Standard implementation: v + 2*cross(q_xyz, cross(q_xyz, v) + q_w*v)
        # Ensure dimensions match
        if vector.dim() == 1: vector = vector.unsqueeze(0)
        if quat.dim() == 1: quat = quat.unsqueeze(0)
        
        # Genesis quaternion format is [w, x, y, z]
        q_w = quat[:, 0].unsqueeze(-1)
        q_xyz = quat[:, 1:4]
        
        t = 2.0 * torch.cross(q_xyz, vector, dim=-1)
        return vector + q_w * t + torch.cross(q_xyz, t, dim=-1)

    def get_resistance_forces(self):
        """
        Returns the scalar resistance force opposing the squeeze motion.
        Projects the net contact force onto the INVERSE squeeze direction.
        Positive value means resistance (pushing back against squeeze).
        
        Returns:
            tuple[torch.Tensor, torch.Tensor]: (Force_L, Force_R) resistance forces for each gripper
        """
        with teleop_profile(self.scene, "teleop_logic_resistance_ee_quat"):
            quat = self.ee_link.get_quat()
            if quat.dim() == 1:
                quat = quat.unsqueeze(0)

            if (
                self._resistance_quat_cache is not None
                and self._global_squeeze_L_cache is not None
                and self._global_squeeze_R_cache is not None
                and torch.equal(quat, self._resistance_quat_cache)
            ):
                global_squeeze_L = self._global_squeeze_L_cache
                global_squeeze_R = self._global_squeeze_R_cache
            else:
                local_squeeze_L = self._squeeze_dir_L.expand(quat.shape[0], 3)
                local_squeeze_R = self._squeeze_dir_R.expand(quat.shape[0], 3)
                global_squeeze_L = self._rotate_vector_by_quat(local_squeeze_L, quat)
                global_squeeze_R = self._rotate_vector_by_quat(local_squeeze_R, quat)
                self._resistance_quat_cache = quat
                self._global_squeeze_L_cache = global_squeeze_L
                self._global_squeeze_R_cache = global_squeeze_R

        with teleop_profile(self.scene, "teleop_logic_resistance_contact_pull"):
            contact_forces = self.get_gripper_net_contact_force()

        with teleop_profile(self.scene, "teleop_logic_resistance_project"):
            force_L = contact_forces[:, 0, :]
            force_R = contact_forces[:, 1, :]
            resist_L = -torch.sum(force_L * global_squeeze_L, dim=-1)
            resist_R = -torch.sum(force_R * global_squeeze_R, dim=-1)
        
        # Return tuple of tensors for single environment (compatibility with teleop_socket)
        # Assuming batch size 1 for teleop
        return resist_L[0], resist_R[0]

    @property
    def ee_pose(self) -> torch.Tensor:
        """Returns the end-effector pose as (position, quaternion)."""
        pos, quat = self.ee_link.get_pos(), self.ee_link.get_quat()
        return torch.cat([pos, quat], dim=-1)

class AgilityForgeEnv:
    """An RL environment for the robotic forging task, configured parametrically."""
    def __init__(self, cfg: AgilityForgeOptions):
        self.cfg = cfg
        self.device = gs.device

        # Cache constant tensors on device for performance (must be done before usage)
        self.action_lower_bounds = self.cfg.env.action_lower_bounds.to(dtype=torch.float32, device=self.device)
        self.action_upper_bounds = self.cfg.env.action_upper_bounds.to(dtype=torch.float32, device=self.device)
        self.fixed_region_bounds = self.cfg.env.fixed_region_bounds.to(dtype=torch.float32, device=self.device)
        self.target_shape_bounds = self.cfg.env.target_shape_bounds.to(dtype=torch.float32, device=self.device)

        self._setup_scene()
        self.robot = AgilityForgeManipulator(scene=self.scene, robot_cfg=self.cfg.robot)
        self._setup_entities()
        self._add_visual_guides()
        
        if self.cfg.general.record:
            self._setup_recording_camera()
        
        self.scene.build(n_envs=self.cfg.env.num_envs, env_spacing=(0.75, 0.5))

        self.robot.set_pd_gains()
        self.robot.set_clamp_force_range()
        self.initial_robot_qpos = self.robot.entity.get_qpos().clone()
        self.initial_mpm_pos = self.mpm_entity.get_state().pos.clone()
        self._set_fixed_particles()
        
        num_particles = self.initial_mpm_pos.shape[1]
        self.num_obs = 3 + num_particles * 9
        self.num_actions = self.cfg.env.num_actions
        self.num_envs = self.cfg.env.num_envs
        self.max_episode_length = self.cfg.env.max_episode_length
        self.target_mpm_pos = torch.zeros_like(self.initial_mpm_pos)
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.int32)
        self.reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.bool)
        self.extras = {}

    def _setup_scene(self):
        """Configures and initializes the simulation scene from config objects."""
        viewer_options = self.cfg.viewer if self.cfg.general.show_viewer else gs.options.ViewerOptions()
        
        # Rigid-MPM contact method. Previously unreachable: no coupler_options were passed, so
        # the scene silently used the default LegacyCouplerOptions(). Defaults below reproduce
        # that exactly.
        #   AGF_CONTACT_MODE=grid|particle|fluidlab|penalty|none
        #   Non-grid modes compose WITH grid contact -- grid is the baseline and supplies
        #   the non-penetration floor; the selected mode is a correction on top of it.
        #   AGF_CONTACT_PER_NODE=1        (particle mode only)
        #   AGF_CONTACT_C_INJECTION=1     (particle mode only)
        #   AGF_PENALTY_K=5e7             (penalty mode only, N/m)
        # NB particle/fluidlab require AGF_ENABLE_CPIC=0 -- both resolve contact in g2p.
        coupler_options = gs.options.LegacyCouplerOptions(
            rigid_mpm_contact_mode=os.environ.get("AGF_CONTACT_MODE", "grid"),
            rigid_mpm_contact_per_node=env_bool("AGF_CONTACT_PER_NODE", False),
            rigid_mpm_contact_c_injection=env_bool("AGF_CONTACT_C_INJECTION", False),
            #   AGF_CONTACT_FTMP_PROJ=1 <cmd>
            rigid_mpm_contact_ftmp_projection=env_bool("AGF_CONTACT_FTMP_PROJ", False),
            # Compile all contact paths and gate them on runtime fields so an arm sweep can
            # share one scene build. Must match base_mpm_solver._pc_switchable, which reads the
            # same variable; the coupler raises at build time if the two disagree.
            #   AGF_CONTACT_RUNTIME_SWITCH=1 <cmd>
            rigid_mpm_contact_runtime_switchable=env_bool("AGF_CONTACT_RUNTIME_SWITCH", False),
            #   AGF_DIAG_PENETRATION=1 <cmd>
            rigid_mpm_penetration_probe=env_bool("AGF_DIAG_PENETRATION", False),
            rigid_mpm_penalty_stiffness=env_float("AGF_PENALTY_K", 5e7),
            rigid_mpm_penalty_damping=env_float("AGF_PENALTY_DAMPING", 1.0),
        )

        self.scene = gs.Scene(
            sim_options=self.cfg.sim,
            viewer_options=viewer_options,
            rigid_options=gs.options.RigidOptions(dt=self.cfg.sim.dt),
            mpm_options=self.cfg.mpm,
            coupler_options=coupler_options,
            vis_options=self.cfg.vis,
            profiling_options=self.cfg.profiling,
            show_viewer=self.cfg.general.show_viewer,
        )
        self.scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", pos=(0, 0, 0.01), fixed=True))

    def _setup_entities(self):
        """Creates the MPM object and the target shape visual guide."""
        self._target_bounds_entity = None
        self._fixed_region_guide_entity = None
        custom_sampler = None
        if self.cfg.env.particle_sampler != "default":
            custom_sampler = self.cfg.env.particle_sampler

        material_kwargs = dict(
            E=self.cfg.mat.E, nu=self.cfg.mat.nu, rho=self.cfg.mat.rho,
        )
        if custom_sampler is not None:
            material_kwargs["sampler"] = custom_sampler

        self.mpm_entity = self.scene.add_entity(
            material=gs.materials.MPM.ElastoPlastic(
                von_mises_yield_stress=self.cfg.mat.von_mises_yield_stress,
                **material_kwargs,
            ) if not getattr(self.cfg.mat, 'use_johnson_cook', False) else JohnsonCookPlasticity(
                A=self.cfg.mat.jc_A, B=self.cfg.mat.jc_B, n=self.cfg.mat.jc_n,
                C=self.cfg.mat.jc_C, eps0=self.cfg.mat.jc_eps0,
                T_ref=self.cfg.mat.jc_T_ref, T_melt=self.cfg.mat.jc_T_melt,
                jc_m=self.cfg.mat.jc_m,
                **material_kwargs,
            ),
            morph=_billet_morph(self.cfg.robot),
            surface=gs.surfaces.Metal(color=(0.8, 0.4, 0.0), vis_mode="particle"),
        )
        if self.cfg.env.show_target_bounds:
            self._target_bounds_entity = self.scene.add_entity(
                morph=gs.morphs.Box(size=(self.cfg.robot.target_shape_bounds[1] - self.cfg.robot.target_shape_bounds[0]).cpu(), pos=((self.cfg.robot.target_shape_bounds[0] + self.cfg.robot.target_shape_bounds[1]) / 2).cpu(), fixed=True, collision=False),
                surface=gs.surfaces.Default(color=(1.0, 0.0, 0.0, 0.4)),
            )
        else:
            self._target_bounds_entity = None

    def _add_visual_guides(self):
        """Adds a non-physical box to visualize the fixed particle region."""
        bounds = self.cfg.env.fixed_region_bounds
        box_size = (bounds[:, 1] - bounds[:, 0]).cpu()
        box_pos = bounds.mean(dim=1).cpu()
        self._fixed_region_guide_entity = self.scene.add_entity(
            morph=gs.morphs.Box(size=box_size, pos=box_pos, fixed=True, collision=False),
            surface=gs.surfaces.Default(color=(0.1, 0.1, 0.6, 0.4)),
        )

    def _set_fixed_particles(self):
        """Identifies and sets the fixed MPM particles based on the defined region."""
        bounds = self.fixed_region_bounds
        mpm_pos_env0 = self.initial_mpm_pos[0]
        is_in_box = ((mpm_pos_env0 >= bounds[:, 0]) & (mpm_pos_env0 <= bounds[:, 1])).all(dim=1)
        self.mpm_entity.set_free(~is_in_box)

    def _setup_recording_camera(self):
        """Initializes the camera used for video recording."""
        self.recording_camera = self.scene.add_camera(
            res=self.cfg.viewer.res,
            pos=self.cfg.viewer.camera_pos,
            lookat=self.cfg.viewer.camera_lookat,
            fov=45,
            GUI=False
        )
        self.scene.add_camera(GUI=True)

    def start_recording(self):
        if self.cfg.general.record:
            self.recording_camera.start_recording()

    def stop_recording(self, filename="agforge_demo.mp4", fps=30):
        if self.cfg.general.record:
            self.recording_camera.stop_recording(save_to_filename=filename, fps=fps)

    def reset_idx(self, envs_idx: torch.Tensor):
        """Resets specified environments to their initial state."""
        if len(envs_idx) == 0: return
        
        self.episode_length_buf[envs_idx] = 0
        self.robot.reset(envs_idx)
        self.scene.reset(envs_idx=envs_idx)

        init_bounds = torch.stack([self.initial_mpm_pos[0].min(dim=0).values, self.initial_mpm_pos[0].max(dim=0).values])
        target_bounds = self.target_shape_bounds
        
        eps = 1e-8
        scale = (target_bounds[1] - target_bounds[0]) / (init_bounds[1] - init_bounds[0] + eps)
        offset = target_bounds[0] - init_bounds[0] * scale
        
        self.target_mpm_pos = self.initial_mpm_pos * scale + offset

    def reset(self) -> tuple[torch.Tensor, None]:
        """Resets all environments and returns the initial observation."""
        all_envs = torch.arange(self.num_envs, device=self.device)
        self.reset_idx(all_envs)
        obs, _ = self.get_observations()
        return obs, None

    def _step_sim(self, num_steps: int):
        """Steps the simulation for a given number of steps, with optional recording."""
        if self.cfg.general.record:
            for _ in range(num_steps):
                self.scene.step()
                self.recording_camera.render(rgb=True)
        else:
            for _ in range(num_steps):
                self.scene.step()

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Applies an action and steps the simulation forward."""
        self.episode_length_buf += 1
        actions = torch.clamp(
            actions, self.action_lower_bounds, self.action_upper_bounds
        )

        qpos = self.robot.entity.get_qpos()
        
        # Teleport to the starting position for the clamp
        self.robot.set_control_mode("TELEPORT")
        qpos[:, 0:2] = actions[:, 0:2]
        self.robot.apply_action(qpos[:, 0:2], dofs_idx_local=[0, 1])
        self._step_sim(1)  # One step to ensure the state is updated

        # Switch to PD control for the clamping action
        self.robot.set_control_mode("PD_CONTROL")
        qpos[:, 2:4] = actions[:, 2].unsqueeze(-1)
        self.robot.apply_action(qpos[:, 2:4], dofs_idx_local=[2, 3])
        self._step_sim(self.cfg.env.action_duration_steps)

        # Reset for the next action
        qpos_prev = self.robot.entity.get_qpos()
        reset_qpos = self.initial_robot_qpos.clone()
        reset_qpos[:, 0:2] = qpos_prev[:, 0:2]
        self.robot.apply_action(reset_qpos)
        self._step_sim(self.cfg.env.reset_duration_steps)

        self.reset_buf = (self.episode_length_buf >= self.max_episode_length)
        if self.reset_buf.any():
            self.reset_idx(self.reset_buf.nonzero(as_tuple=False).squeeze(-1))

        rewards = self._reward_mpm_deformation()
        obs, self.extras = self.get_observations()
        return obs, rewards, self.reset_buf, self.extras

    def get_observations(self) -> tuple[torch.Tensor, dict]:
        """Constructs the observation tensor from the current simulation state."""
        mpm_pos = self.mpm_entity.get_state().pos
        mpm_obs = torch.cat([mpm_pos, self.target_mpm_pos, mpm_pos - self.target_mpm_pos], dim=-1).flatten(start_dim=1)
        obs = torch.cat([self.robot.ee_pose[:, :3], mpm_obs], dim=-1)
        
        self.extras['observations'] = {'critic': obs}
        return obs, self.extras
    
    def get_privileged_observations(self) -> None:
        return None

    def _reward_mpm_deformation(self) -> torch.Tensor:
        """Calculates reward based on the mean distance to the target shape."""
        dist = torch.norm(self.mpm_entity.get_state().pos - self.target_mpm_pos, p=2, dim=-1).mean(dim=-1)
        return torch.exp(-dist)