import threading
import genesis as gs
import numpy as np
from pynput import keyboard

class KeyboardDevice:
    def __init__(self):
        self.pressed = set()
        self.lock = threading.Lock()
        self.listener = keyboard.Listener(on_press=self._down, on_release=self._up)

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

def build_scene():
    gs.init(seed=0, precision="32", backend=gs.gpu, logging_level="warning")
    dt = 0.005

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=dt, substeps=64),
        rigid_options=gs.options.RigidOptions(
            dt=dt, enable_joint_limit=True, enable_collision=True, gravity=(0, 0, 0)
        ),
        mpm_options=gs.options.MPMOptions(
            lower_bound=(-0.4, -0.15, 0.15),
            upper_bound=(0.25, 0.15, 0.45),
            gravity=(0, 0, 0),
            grid_density=100,
            particle_size=0.008,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(2, 0, 2.5), camera_lookat=(0, 0, 0.5), camera_fov=50, max_FPS=60
        ),
        vis_options=gs.options.VisOptions(
            show_world_frame=False, visualize_mpm_boundary=False
        ),
        show_viewer=True,
    )

    add = scene.add_entity
    add(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))
    robot = add(
        material=gs.materials.Rigid(gravity_compensation=1.0),
        morph=gs.morphs.MJCF(file="xml/agforge_demo.xml"),
    )
    mpm = add(
        material=gs.materials.MPM.ElastoPlastic(
            E=5e5, nu=0.3, rho=100.0, von_mises_yield_stress=200.0
        ),
        morph=gs.morphs.Cylinder(
            radius=0.05, height=0.4, pos=(0, 0, 0.3), euler=(0, 90, 0)
        ),
        surface=gs.surfaces.Metal(color=(1, 0.5, 0), vis_mode="particle"),
    )

    # helper to draw non‐physical boxes
    def draw_box(size, pos, color):
        add(
            morph=gs.morphs.Box(size=size, pos=pos, fixed=True, collision=False),
            surface=gs.surfaces.Default(color=color),
        )

    draw_box((0.3, 0.06, 0.06), (-0.15, 0, 0.3), (1, 0, 0, 0.3))

    # fixed‐region mask
    x0, x1, y0, y1, z0, z1 = 0.15, 0.25, -0.1, 0.1, 0.2, 0.4
    center = ((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2)
    extent = (x1 - x0, y1 - y0, z1 - z0)
    draw_box(extent, center, (0, 0, 1, 0.3))

    scene.build()

    robot.set_dofs_kp([1000] * 4)
    robot.set_dofs_kv([100] * 4)

    # freeze particles in fixed region
    positions = mpm.get_state().pos.cpu().numpy().squeeze(0)
    mask = ~(
        (positions[:, 0] >= x0)
        & (positions[:, 0] <= x1)
        & (positions[:, 1] >= y0)
        & (positions[:, 1] <= y1)
        & (positions[:, 2] >= z0)
        & (positions[:, 2] <= z1)
    )
    mpm.set_free(mask)

    return scene, robot

def run():
    kb = KeyboardDevice()
    kb.start()

    scene, robot = build_scene()
    n = robot.n_dofs
    vel = np.zeros(n)

    # key → (joint_indices,list_velocity)
    controls = {
        keyboard.Key.up:   ([0],  +1.6),
        keyboard.Key.down: ([0],  -1.6),
        keyboard.Key.right:([1], +14),
        keyboard.Key.left: ([1], -14),
        keyboard.KeyCode.from_char("j"): ([2, 3], +0.39),
        keyboard.KeyCode.from_char("k"): ([2, 3], -0.78),
    }

    print(
        "←/→ slide | ↑/↓ hinge | j/k grippers | space stop | u reset | esc quit"
    )
    scene.reset()

    while True:
        keys = kb.pressed.copy()

        if keyboard.Key.esc in keys:
            break

        if keyboard.KeyCode.from_char("u") in keys:
            scene.reset()
            vel.fill(0)

        if keyboard.Key.space in keys:
            vel.fill(0)
        else:
            vel.fill(0)  # zero out all
            for key, (j_idxs, v) in controls.items():
                if key in keys:
                    for j in j_idxs:
                        vel[j] = v

        # apply velocity control
        for j in range(n):
            robot.control_dofs_velocity([vel[j]], [j])

        scene.step()

    kb.stop()

if __name__ == "__main__":
    run()
