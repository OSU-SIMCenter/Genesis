"""
Replay a recorded HDF5 episode using the Genesis viewer.

Usage:
    pixi run python -m agforge.replay_episode [--data PATH] [-e EPISODE]

This script:
  1. Loads the recorded episode from HDF5 (particle positions + robot qpos per frame).
  2. Builds the same Genesis scene used for live simulation.
  3. For each frame, teleports the MPM particles and robot joints to their recorded state
     and refreshes the viewer, producing a visual replay without running physics.
"""

import argparse
import os
import sys
import time

import h5py
import numpy as np
import torch
import genesis as gs

from agforge.options import TeleopOptions
from agforge.agforge_builder import build_env


def load_episode(data_path: str, episode_arg):
    """Load particle positions and robot qpos from an HDF5 episode."""
    with h5py.File(data_path, "r") as f:
        ep_names = sorted(f.keys())
        if not ep_names:
            print(f"No episodes found in {data_path}")
            sys.exit(1)

        # Resolve episode argument
        if isinstance(episode_arg, int):
            ep_name = ep_names[episode_arg]
        elif isinstance(episode_arg, str) and episode_arg in f:
            ep_name = episode_arg
        else:
            print(f"Episode '{episode_arg}' not found. Available: {ep_names}")
            sys.exit(1)

        print(f"Loading episode: {ep_name}")
        ep = f[ep_name]

        # Read metadata
        num_timesteps = int(ep.attrs["num_timesteps"])
        lang = ep.attrs.get("language_instruction", "")
        print(f"  Timesteps: {num_timesteps}")
        print(f"  Description: {lang}")

        # Read particle data (ragged via offsets)
        particles_grp = ep["observations/state/particles"]
        pos_values = particles_grp["pos_values"][:]      # (total_particles, 3)
        offsets = particles_grp["offsets"][:]             # (T+1,)
        
        temp_values = None
        if "temp_values" in particles_grp:
            temp_values = particles_grp["temp_values"][:]

        # Read robot qpos
        qpos_arr = ep["observations/state/scene/qpos"][:]  # (T, 4)

    return {
        "ep_name": ep_name,
        "num_timesteps": num_timesteps,
        "pos_values": pos_values,
        "temp_values": temp_values,
        "offsets": offsets,
        "qpos": qpos_arr,
        "description": lang,
    }


def main():
    parser = argparse.ArgumentParser(description="Replay a recorded episode in the Genesis viewer.")
    parser.add_argument("--data", type=str, default="data/train/shard_0000.h5",
                        help="Path to the HDF5 shard file.")
    parser.add_argument("-e", "--episode", type=str, default="-1",
                        help="Episode index (e.g. 0, -1) or name (e.g. ep_000000). Defaults to -1 (most recent).")
    parser.add_argument("--fps", type=float, default=10.0,
                        help="Playback frames per second (default: 10).")
    parser.add_argument("--loop", action="store_true",
                        help="Loop the replay continuously.")
    args = parser.parse_args()

    # Resolve data path
    data_path = args.data
    if not os.path.isabs(data_path):
        data_path = os.path.join(os.path.dirname(__file__), "..", data_path)

    # Parse episode argument
    try:
        ep_arg = int(args.episode)
    except ValueError:
        ep_arg = args.episode

    # Load episode data
    episode = load_episode(data_path, ep_arg)

    # Build the Genesis scene (with viewer)
    print("Building Genesis scene for replay...")
    cfg = TeleopOptions()
    cfg.general.show_viewer = True
    cfg.vis.visualize_mpm_grid = True

    env = build_env(cfg)

    # Get references to entities
    mpm_entity = env.mpm_entity
    robot = env.robot

    print("Scene built. Starting replay...")

    frame_delay = 1.0 / args.fps
    n_frames = episode["num_timesteps"]
    offsets = episode["offsets"]
    pos_values = episode["pos_values"]
    temp_values = episode.get("temp_values")
    qpos_data = episode["qpos"]

    # ---- Temperature colormap setup ----
    import matplotlib as mpl
    from genesis.ext import pyrender
    import genesis.utils.mesh as mu

    N_BUCKETS = 8
    TEMP_MIN = 293.0
    TEMP_MAX = 1450.0
    cmap = mpl.colormaps.get_cmap("inferno")

    bucket_colors = []
    for i in range(N_BUCKETS):
        t = 0.15 + (i / (N_BUCKETS - 1)) * 0.85
        bucket_colors.append(tuple(cmap(t)))

    n_particles = mpm_entity.n_particles
    particle_radius = env.scene.sim.mpm_solver.particle_radius
    ctx = env.scene.visualizer._context

    # ---- Pre-compute bucket sizes across all frames ----
    print("  Pre-computing temperature buckets...")
    frame_bucket_indices = []
    max_bucket_sizes = np.zeros(N_BUCKETS, dtype=int)

    for fi in range(n_frames):
        s, e = offsets[fi], offsets[fi + 1]
        if temp_values is not None:
            ft = temp_values[s:e]
            tn = np.clip((ft - TEMP_MIN) / (TEMP_MAX - TEMP_MIN), 0.0, 1.0)
            bi = np.minimum((tn * N_BUCKETS).astype(int), N_BUCKETS - 1)
        else:
            bi = np.full(e - s, N_BUCKETS // 2, dtype=int)
        frame_bucket_indices.append(bi)
        for b in range(N_BUCKETS):
            count = (bi == b).sum()
            if count > max_bucket_sizes[b]:
                max_bucket_sizes[b] = count

    total_instances = int(max_bucket_sizes.sum())
    print(f"  Max bucket sizes: {max_bucket_sizes.tolist()} (total: {total_instances})")

    # ---- Remove default orange node ----
    for idx in ctx.rendered_envs_idx:
        key = (idx, mpm_entity.uid)
        if key in ctx.static_nodes:
            old_node = ctx.static_nodes.pop(key)
            ctx.remove_node(old_node)

    # ---- Pre-create right-sized bucket nodes ----
    OFF_SCREEN = np.array([0.0, 0.0, -1000.0])
    bucket_nodes = []
    bucket_buf_ids = []
    bucket_max = []

    for b in range(N_BUCKETS):
        n_slots = int(max_bucket_sizes[b])
        if n_slots == 0:
            bucket_nodes.append(None)
            bucket_max.append(0)
            continue

        mesh = mu.create_sphere(particle_radius, subdivisions=1, color=bucket_colors[b])
        OFF_SCREEN = np.array([0.0, 0.0, -0.05])
        tfs = np.tile(np.eye(4), (n_slots, 1, 1))
        tfs[:, :3, 3] = OFF_SCREEN
        # Distribute the first few to ensure bounding box covers the scene
        n_dist = min(n_slots, len(pos_values))
        tfs[:n_dist, :3, 3] = pos_values[:n_dist]

        pr_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=True, poses=tfs)
        scene_node = ctx.add_node(pr_mesh)
        bucket_nodes.append(scene_node)
        bucket_max.append(n_slots)

    active_buckets = sum(1 for n in bucket_nodes if n is not None)
    print(f"  Created {active_buckets} bucket nodes ({total_instances} total slots)")

    # ---- Monkeypatch update_mpm to update bucket nodes inside the render pipeline ----
    # jit.update_buffer only works when called from within context.update().
    # By replacing update_mpm, we take full control of particle visualization.
    frame_pos = None
    bucket_idx = None

    def _replay_update_mpm():
        # print("HOOK CALLED")
        if frame_pos is None:
            return

        for b in range(N_BUCKETS):
            if bucket_nodes[b] is None:
                continue
            n_slots = bucket_max[b]

            OFF_SCREEN = np.array([0.0, 0.0, -0.05])
            tfs = np.tile(np.eye(4), (n_slots, 1, 1))
            tfs[:, :3, 3] = OFF_SCREEN

            mask = bucket_idx == b
            count = mask.sum()
            if count > 0:
                tfs[:count, :3, 3] = frame_pos[mask]

            node = bucket_nodes[b]
            for prim in node.mesh.primitives:
                prim.poses = tfs
            
            buf_id = ctx._scene.get_buffer_id(node, "model")
            if buf_id != -1:
                ctx.jit.update_buffer(buf_id, tfs.transpose((0, 2, 1)))

    ctx.update_mpm = _replay_update_mpm

    # ---- Stop physics from advancing time ----
    env.scene.sim._dt = 0.0
    env.scene.sim._substep_dt = 0.0

    playing = True
    try:
        while playing:
            for frame_idx in range(n_frames):
                t_start = time.time()

                # --- 1. Extract particle positions for this frame ---
                start = offsets[frame_idx]
                end = offsets[frame_idx + 1]
                frame_pos = pos_values[start:end]

                # --- 2. Teleport MPM particles ---
                pos_tensor = torch.tensor(frame_pos, dtype=gs.tc_float, device=gs.device).unsqueeze(0)
                mpm_entity.set_position(pos_tensor)

                # --- 3. Teleport robot joints ---
                qpos = torch.tensor(qpos_data[frame_idx], dtype=gs.tc_float, device=gs.device).unsqueeze(0)
                robot.entity.set_qpos(qpos)

                # --- 4. Set frame data for the monkeypatched update_mpm ---
                # frame_pos is already set above
                bucket_idx = frame_bucket_indices[frame_idx]

                # --- 5. Step + render (update_mpm hook updates bucket nodes) ---
                env.scene._t += 1
                env.scene.step()

                # --- 6. Frame pacing ---
                elapsed = time.time() - t_start
                sleep_time = frame_delay - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

                if frame_idx % max(1, n_frames // 10) == 0:
                    pct = (frame_idx / n_frames) * 100
                    print(f"  Frame {frame_idx}/{n_frames} ({pct:.0f}%)")

            print(f"  Replay complete ({n_frames} frames).")
            if not args.loop:
                playing = False

        print("Done. Close the viewer window to exit.")
        while True:
            env.scene.visualizer.update(force=False, auto=True)
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Replay stopped: {e}")


if __name__ == "__main__":
    main()
