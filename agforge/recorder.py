import os
import time
import h5py
import torch
import numpy as np
import pandas as pd

def to_cpu(tensor):
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    if isinstance(tensor, np.ndarray):
        return tensor
    return np.array(tensor)

class AgForgeRecorder:
    """
    Records dense physics simulation data into Sharded HDF5 files 
    using Open-X-Embodiment (RLDS) schemas and 'Values + Offsets' 
    encoding for ragged (variable length) arrays.
    """
    def __init__(self, data_dir="data", shard_capacity=100):
        self.data_dir = data_dir
        self.shard_capacity = shard_capacity
        
        self.train_dir = os.path.join(data_dir, "train")
        os.makedirs(self.train_dir, exist_ok=True)
        
        self.catalog_path = os.path.join(data_dir, "episodes_catalog.parquet")
        
        self.global_episode_id = self._get_next_global_episode_id()
        self.current_shard_id = self.global_episode_id // self.shard_capacity
        
        self._init_buffer()
        self.current_target_id = None
        self.is_recording = False
        
    def _get_next_global_episode_id(self):
        if os.path.exists(self.catalog_path):
            try:
                df = pd.read_parquet(self.catalog_path)
                if not df.empty:
                    return int(df["episode_global_id"].max()) + 1
            except Exception as e:
                print(f"[Recorder] Error reading catalog: {e}")
        return 0

    def _init_buffer(self):
        # We store flattened particle values and tracking offsets array.
        self.buffer = {
            "particles_pos": [], # list of (N, 3) arrays to be concatenated later
            "particles_vel": [], # list of (N, 3) arrays
            "particles_temp": [], # list of (N,) arrays
            "particles_detF": [], # list of (N,) arrays
            "particles_offsets": [0], # Starts at 0, length will be T+1
            "particle_vol": 0.0,
            "qpos": [],
            "force_torque": [],
            "dof_cmd": [],
        }
        self.strike_boundaries = [] # For undo logic
        
    def start_new_episode(self, target_id):
        self._init_buffer()
        self.current_target_id = target_id
        self.is_recording = True
        print(f"[Recorder] Started new episode: ID {self.global_episode_id} (Target: {target_id})")

    def mark_strike_start(self):
        """Marks the boundary before a strike begins, allowing for undo."""
        current_len = len(self.buffer["qpos"])
        current_particle_offset = self.buffer["particles_offsets"][-1]
        
        self.strike_boundaries.append({
            "frame_idx": current_len,
            "offsets_idx": len(self.buffer["particles_offsets"]) - 1,
            "particle_idx": current_particle_offset,
            "particles_values_len": len(self.buffer["particles_pos"])
        })

    def handle_undo(self):
        """Removes the last recorded strike sequence from the buffer."""
        if not self.strike_boundaries:
            return
            
        boundary = self.strike_boundaries.pop()
        f_idx = boundary["frame_idx"]
        o_idx = boundary["offsets_idx"]
        
        # Truncate lists back to pre-strike lengths
        self.buffer["qpos"] = self.buffer["qpos"][:f_idx]
        self.buffer["force_torque"] = self.buffer["force_torque"][:f_idx]
        self.buffer["dof_cmd"] = self.buffer["dof_cmd"][:f_idx]
        
        self.buffer["particles_offsets"] = self.buffer["particles_offsets"][:o_idx + 1]
        self.buffer["particles_pos"] = self.buffer["particles_pos"][:f_idx]
        self.buffer["particles_vel"] = self.buffer["particles_vel"][:f_idx]
        self.buffer["particles_temp"] = self.buffer["particles_temp"][:f_idx]
        self.buffer["particles_detF"] = self.buffer["particles_detF"][:f_idx]
        
        print(f"[Recorder] Undid strike. Reverted buffer from frame {len(self.buffer['qpos'])} to {f_idx}.")

    def record_frame(self, particles_pos, particles_vel, particles_temp, particles_detF, particle_vol, qpos, force_L, force_R, dof_cmd):
        """Buffer a single timestep of data from the simulation."""
        if not self.is_recording:
            return
            
        p_pos = to_cpu(particles_pos)
        if p_pos.ndim == 3: p_pos = p_pos[0]
        
        p_vel = to_cpu(particles_vel)
        if p_vel.ndim == 3: p_vel = p_vel[0]
        
        p_temp = to_cpu(particles_temp)
        if p_temp.ndim == 3: p_temp = p_temp[0]
        
        p_detF = to_cpu(particles_detF)
        if p_detF.ndim == 3: p_detF = p_detF[0]
        
        self.buffer["particles_pos"].append(p_pos.astype(np.float32))
        self.buffer["particles_vel"].append(p_vel.astype(np.float32))
        self.buffer["particles_temp"].append(p_temp.astype(np.float32))
        self.buffer["particles_detF"].append(p_detF.astype(np.float32))
        self.buffer["particle_vol"] = float(particle_vol)
        
        # Update offsets math
        n_particles = p_pos.shape[0]
        last_offset = self.buffer["particles_offsets"][-1]
        self.buffer["particles_offsets"].append(last_offset + n_particles)
        
        # Robot
        qpos_arr = to_cpu(qpos)
        if qpos_arr.ndim == 2: qpos_arr = qpos_arr[0]
        self.buffer["qpos"].append(qpos_arr.astype(np.float32))
        
        # Forces
        f_L = float(to_cpu(force_L).item()) if to_cpu(force_L).size == 1 else to_cpu(force_L)[0]
        f_R = float(to_cpu(force_R).item()) if to_cpu(force_R).size == 1 else to_cpu(force_R)[0]
        force_arr = np.array([f_L, f_R], dtype=np.float32)
        self.buffer["force_torque"].append(force_arr)
        
        # Actions
        cmd_arr = to_cpu(dof_cmd)
        if cmd_arr.ndim == 2: cmd_arr = cmd_arr[0]
        self.buffer["dof_cmd"].append(cmd_arr.astype(np.float32))

    def flush_episode(self, success_flag=True, language_instruction="Strike the hot steel billet."):
        """Flushes the buffered episode into the current HDF5 shard and updates Parquet."""
        if not self.is_recording or len(self.buffer["qpos"]) == 0:
            print("[Recorder] Buffer empty or not recording, skipping flush.")
            return
            
        num_timesteps = len(self.buffer["qpos"])
        
        if num_timesteps > 0:
            flattened_pos = np.concatenate(self.buffer["particles_pos"], axis=0)
            flattened_vel = np.concatenate(self.buffer["particles_vel"], axis=0)
            flattened_temp = np.concatenate(self.buffer["particles_temp"], axis=0)
            flattened_detF = np.concatenate(self.buffer["particles_detF"], axis=0)
        else:
            flattened_pos = np.array([], dtype=np.float32)
            flattened_vel = np.array([], dtype=np.float32)
            flattened_temp = np.array([], dtype=np.float32)
            flattened_detF = np.array([], dtype=np.float32)
            
        offsets_arr = np.array(self.buffer["particles_offsets"], dtype=np.int64)
        
        qpos_arr = np.array(self.buffer["qpos"], dtype=np.float32)
        force_arr = np.array(self.buffer["force_torque"], dtype=np.float32)
        cmd_arr = np.array(self.buffer["dof_cmd"], dtype=np.float32)
        
        # 2. Sharding Logic
        self.current_shard_id = self.global_episode_id // self.shard_capacity
        shard_filename = f"shard_{self.current_shard_id:04d}.h5"
        shard_path = os.path.join(self.train_dir, shard_filename)
        
        ep_group_name = f"ep_{self.global_episode_id:06d}"
        
        # 3. Write internal RLDS Schema to Sharded HDF5
        try:
            with h5py.File(shard_path, "a") as f:
                ep_grp = f.create_group(ep_group_name)
                
                # Global Track Metadata
                ep_grp.attrs["target_id"] = self.current_target_id
                ep_grp.attrs["num_timesteps"] = num_timesteps
                ep_grp.attrs["language_instruction"] = language_instruction
                ep_grp.attrs["is_terminal"] = True
                ep_grp.attrs["is_first"] = True
                ep_grp.attrs["success_flag"] = success_flag
                
                # --- OBSERVATIONS ---
                obs_grp = ep_grp.create_group("observations")
                state_grp = obs_grp.create_group("state")
                
                # Particles (Ragged via Offsets)
                part_grp = state_grp.create_group("particles")
                part_grp.create_dataset("pos_values", data=flattened_pos, compression="lzf")
                part_grp.create_dataset("vel_values", data=flattened_vel, compression="lzf")
                part_grp.create_dataset("temp_values", data=flattened_temp, compression="lzf")
                part_grp.create_dataset("detF_values", data=flattened_detF, compression="lzf")
                part_grp.create_dataset("offsets", data=offsets_arr)
                if "particle_vol" not in part_grp.attrs:
                    part_grp.attrs["particle_vol"] = self.buffer["particle_vol"]
                
                # Scene & Robot
                scene_grp = state_grp.create_group("scene")
                scene_grp.create_dataset("qpos", data=qpos_arr)
                scene_grp.create_dataset("force_torque", data=force_arr)
                
                # --- ACTIONS ---
                act_grp = ep_grp.create_group("actions")
                act_grp.create_dataset("dof_velocity_cmd", data=cmd_arr)
                
            print(f"[Recorder] Flushed Episodic Group {ep_group_name} into {shard_path} (T={num_timesteps})")
            
            # 4. Parquet Catalog Row append
            # For simplicity & stability, we load, append, save with pandas/pyarrow.
            new_row = pd.DataFrame([{
                "episode_global_id": self.global_episode_id,
                "shard_file": shard_filename,
                "internal_group": ep_group_name,
                "language_instruction": language_instruction,
                "length": num_timesteps,
                "success": success_flag,
                "target_id": self.current_target_id
            }])
            
            if os.path.exists(self.catalog_path):
                df = pd.read_parquet(self.catalog_path)
                df = pd.concat([df, new_row], ignore_index=True)
                df.to_parquet(self.catalog_path, engine="pyarrow")
            else:
                new_row.to_parquet(self.catalog_path, engine="pyarrow")
                
            # Bump global tracker
            self.global_episode_id += 1
            
        except Exception as e:
            print(f"[Recorder] CRITICAL Error writing episode {self.global_episode_id}: {e}")
            import traceback
            traceback.print_exc()
            
        # Clean state for next recording block
        self.is_recording = False
        self._init_buffer()
