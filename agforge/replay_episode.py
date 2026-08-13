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

from agforge.wsl_graphics import apply_early_wsl_graphics_defaults

apply_early_wsl_graphics_defaults()

import h5py
import numpy as np
import torch
import genesis as gs

from agforge.options import TeleopOptions
from agforge.agforge_builder import build_env
from agforge.vis.temperature_particles import N_BUCKETS, TemperatureParticleRenderer

# Defaults match forge_common.real_scale (real Agility Forge stock / Tool2 bite).
# replay_episode runs inside the Genesis pixi env and must not require forge_common.
_DEFAULT_STOCK_DIAMETER_M = 0.040  # 2 * 20mm
_DEFAULT_STOCK_LENGTH_M = 0.059
_DEFAULT_GRIPPER_AXIAL_WIDTH_M = 0.0158187


def _attr_float(attrs, key):
    if key not in attrs:
        return None
    return float(attrs[key])


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

        wall_time_s = None
        scene_grp = ep["observations/state/scene"]
        if "wall_time_s" in scene_grp:
            wall_time_s = np.asarray(scene_grp["wall_time_s"][:], dtype=np.float64)

        stock_diameter_m = _attr_float(ep.attrs, "stock_diameter_m")
        stock_length_m = _attr_float(ep.attrs, "stock_length_m")
        gripper_axial_width_m = _attr_float(ep.attrs, "gripper_axial_width_m")
        n_particles_frame0 = int(offsets[1] - offsets[0]) if len(offsets) > 1 else 0
        has_wall = bool(ep.attrs.get("has_wall_timestamps", wall_time_s is not None))
        wall_duration_s = _attr_float(ep.attrs, "wall_duration_s")
        if wall_duration_s is None and wall_time_s is not None and wall_time_s.size:
            wall_duration_s = float(wall_time_s[-1] - wall_time_s[0])

    return {
        "ep_name": ep_name,
        "num_timesteps": num_timesteps,
        "pos_values": pos_values,
        "temp_values": temp_values,
        "offsets": offsets,
        "qpos": qpos_arr,
        "wall_time_s": wall_time_s,
        "has_wall_timestamps": has_wall,
        "wall_duration_s": wall_duration_s,
        "description": lang,
        "stock_diameter_m": stock_diameter_m,
        "stock_length_m": stock_length_m,
        "gripper_axial_width_m": gripper_axial_width_m,
        "n_particles_frame0": n_particles_frame0,
    }


def _wall_playback_schedule(wall_time_s: np.ndarray, collapse_gaps_over: float) -> np.ndarray:
    """Return playback target offsets (seconds from loop start) from recorded wall times.

    Gaps larger than ``collapse_gaps_over`` (e.g. JIT compile stalls) are collapsed to
    zero so watching is not stuck for ~100s on hit 1, while normal headless pacing
    is preserved.
    """
    t = np.asarray(wall_time_s, dtype=np.float64).reshape(-1)
    if t.size == 0:
        return t
    t = t - t[0]
    if collapse_gaps_over is None or collapse_gaps_over <= 0 or t.size < 2:
        return t
    dt = np.diff(t, prepend=t[0])
    dt[0] = 0.0
    dt = np.minimum(dt, float(collapse_gaps_over))
    return np.cumsum(dt)


def _resolve_stock_m(cli_value, episode_value, default, name):
    if cli_value is not None:
        return float(cli_value)
    if episode_value is not None:
        return float(episode_value)
    print(
        f"  No {name} in episode; using real-scale default {default} m "
        f"(pass --{name.replace('_', '-')} to override)"
    )
    return default


def main():
    parser = argparse.ArgumentParser(description="Replay a recorded episode in the Genesis viewer.")
    parser.add_argument("--data", type=str, default="data/train/shard_0000.h5",
                        help="Path to the HDF5 shard file.")
    parser.add_argument("-e", "--episode", type=str, default="-1",
                        help="Episode index (e.g. 0, -1) or name (e.g. ep_000000). Defaults to -1 (most recent).")
    parser.add_argument("--fps", type=float, default=10.0,
                        help="Playback frames per second when --playback fps (default: 10).")
    parser.add_argument(
        "--playback",
        choices=("fps", "wall"),
        default="fps",
        help="fps: fixed rate via --fps. wall: pace using recorded wall_time_s (headless real-time).",
    )
    parser.add_argument(
        "--wall-collapse-gaps-over",
        type=float,
        default=5.0,
        help="With --playback wall, collapse recorded inter-frame gaps larger than this "
             "many seconds (default 5; use 0 to keep JIT/long stalls).",
    )
    parser.add_argument("--loop", action="store_true",
                        help="Loop the replay continuously.")
    parser.add_argument(
        "--no-particle-color",
        action="store_true",
        help="Skip temperature coloring; keep default orange Metal MPM particles.",
    )
    parser.add_argument(
        "--no-mpm-boundary",
        action="store_true",
        help="Hide the MPM domain boundary box (grid overlay kept).",
    )
    parser.add_argument("--stock-diameter-m", type=float, default=None,
                        help="Stock diameter [m] for the replay scene (default: episode attr or real-scale 0.04).")
    parser.add_argument("--stock-length-m", type=float, default=None,
                        help="Stock length [m] for the replay scene (default: episode attr or real-scale 0.059).")
    parser.add_argument("--gripper-axial-width-m", type=float, default=None,
                        help="Die axial width [m] (default: episode attr or real-scale Tool2 bite).")
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

    playback = args.playback
    wall_schedule = None
    if playback == "wall":
        if episode.get("wall_time_s") is None:
            print(
                "ERROR: --playback wall requires wall_time_s in the episode "
                "(re-record with the updated AgForgeRecorder)."
            )
            sys.exit(1)
        wall_schedule = _wall_playback_schedule(
            episode["wall_time_s"], args.wall_collapse_gaps_over
        )
        wall_span = float(wall_schedule[-1]) if wall_schedule.size else 0.0
        raw_span = float(episode["wall_duration_s"] or 0.0)
        print(
            f"  Wall playback: {episode['num_timesteps']} frames over {wall_span:.2f}s "
            f"(recorded span {raw_span:.2f}s; collapse gaps > "
            f"{args.wall_collapse_gaps_over:g}s)"
        )

    stock_diameter_m = _resolve_stock_m(
        args.stock_diameter_m,
        episode["stock_diameter_m"],
        _DEFAULT_STOCK_DIAMETER_M,
        "stock_diameter_m",
    )
    stock_length_m = _resolve_stock_m(
        args.stock_length_m,
        episode["stock_length_m"],
        _DEFAULT_STOCK_LENGTH_M,
        "stock_length_m",
    )
    gripper_axial_width_m = _resolve_stock_m(
        args.gripper_axial_width_m,
        episode["gripper_axial_width_m"],
        _DEFAULT_GRIPPER_AXIAL_WIDTH_M,
        "gripper_axial_width_m",
    )

    # Build the Genesis scene (with viewer) matching the recorded stock.
    # Bare TeleopOptions() uses ~1in stock (~8458 particles) and will not
    # accept real-scale episode frames (~3160 particles).
    print("Building Genesis scene for replay...")
    print(
        f"  stock_diameter={stock_diameter_m*1000:.2f}mm, "
        f"stock_length={stock_length_m*1000:.2f}mm, "
        f"gripper_axial_width={gripper_axial_width_m*1000:.3f}mm, "
        f"recorded_particles/frame0={episode['n_particles_frame0']}"
    )
    cfg = TeleopOptions(
        stock_diameter=stock_diameter_m,
        stock_length=stock_length_m,
        gripper_axial_width=gripper_axial_width_m,
    )
    cfg.general.show_viewer = True
    cfg.vis.visualize_mpm_grid = True
    cfg.vis.visualize_mpm_boundary = not args.no_mpm_boundary
    # Match viewer cap to playback rate (TeleopOptions defaults to 10).
    if playback == "wall":
        viewer_fps = 60
    else:
        viewer_fps = max(1, int(round(args.fps)))
    cfg.performance.target_viewer_fps = viewer_fps
    cfg.viewer.max_FPS = viewer_fps
    cfg.viewer.refresh_rate = viewer_fps

    env = build_env(cfg)

    # Get references to entities
    mpm_entity = env.mpm_entity
    robot = env.robot

    if mpm_entity.n_particles != episode["n_particles_frame0"]:
        print(
            f"ERROR: scene has {mpm_entity.n_particles} particles but episode "
            f"frame 0 has {episode['n_particles_frame0']}. Stock dims still "
            f"don't match the recording -- re-record or pass explicit "
            f"--stock-diameter-m / --stock-length-m."
        )
        sys.exit(1)

    print("Scene built. Starting replay...")

    frame_delay = 1.0 / args.fps
    n_frames = episode["num_timesteps"]
    offsets = episode["offsets"]
    pos_values = episode["pos_values"]
    temp_values = episode.get("temp_values")
    qpos_data = episode["qpos"]

    renderer = None
    if args.no_particle_color:
        print("  Particle coloring disabled (default Genesis MPM spheres).")
    else:
        temp_min = cfg.general.particle_temp_min
        temp_max = cfg.general.particle_temp_max
        ctx = env.scene.visualizer._context
        particle_radius = env.scene.sim.mpm_solver.particle_radius

        # Pre-compute bucket capacities across all frames for efficient instancing
        print("  Pre-computing temperature buckets...")
        if temp_values is not None:
            max_bucket_sizes = TemperatureParticleRenderer.max_bucket_sizes_from_frames(
                temp_values, offsets, n_frames, temp_min, temp_max,
            )
        else:
            per_bucket = (mpm_entity.n_particles + N_BUCKETS - 1) // N_BUCKETS
            max_bucket_sizes = np.full(N_BUCKETS, per_bucket, dtype=np.int32)

        total_instances = int(max_bucket_sizes.sum())
        print(f"  Max bucket sizes: {max_bucket_sizes.tolist()} (total: {total_instances})")

        renderer = TemperatureParticleRenderer(
            ctx,
            mpm_entity,
            particle_radius,
            mpm_entity.n_particles,
            temp_min=temp_min,
            temp_max=temp_max,
            bucket_max_sizes=max_bucket_sizes,
        )
        active_buckets = sum(1 for n in renderer._bucket_nodes if n is not None)
        print(f"  Created {active_buckets} bucket nodes ({total_instances} total slots)")

    # Stop physics from advancing time
    env.scene.sim._dt = 0.0
    env.scene.sim._substep_dt = 0.0

    playing = True
    try:
        while playing:
            loop_t0 = time.monotonic()
            for frame_idx in range(n_frames):
                start = offsets[frame_idx]
                end = offsets[frame_idx + 1]
                frame_pos = pos_values[start:end]

                pos_tensor = torch.tensor(frame_pos, dtype=gs.tc_float, device=gs.device).unsqueeze(0)
                mpm_entity.set_position(pos_tensor)

                qpos = torch.tensor(qpos_data[frame_idx], dtype=gs.tc_float, device=gs.device).unsqueeze(0)
                robot.entity.set_qpos(qpos)

                if renderer is not None:
                    if temp_values is not None:
                        frame_temp = temp_values[start:end]
                        renderer.set_frame(frame_pos, temps=frame_temp)
                    else:
                        renderer.set_frame(frame_pos)

                env.scene._t += 1
                env.scene.step()

                if playback == "wall":
                    target = float(wall_schedule[frame_idx])
                    sleep_time = target - (time.monotonic() - loop_t0)
                else:
                    # Fixed-fps pacing measured per frame render cost.
                    # (Approximate: sleep residual of 1/fps after this frame's work.)
                    # Use schedule relative to loop start for less drift than per-frame delay.
                    target = frame_idx * frame_delay
                    sleep_time = target - (time.monotonic() - loop_t0)
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
