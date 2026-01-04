import h5py
import duckdb
import numpy as np
import os
import time
import torch

class AgForgeRecorder:
    def __init__(self, data_dir="data", db_name="metadata.duckdb"):
        self.data_dir = data_dir
        self.train_dir = os.path.join(data_dir, "train")
        self.db_path = os.path.join(data_dir, db_name)
        
        os.makedirs(self.train_dir, exist_ok=True)
        
        self.buffer = {
            "mpm_pos": [],
            "mpm_stress": [], # Storing Von Mises stress or plastic deformation
            "force_torque": [],
            "qpos": [],
            "target_id": None,
            "strike_indices": [] # Tuples of (start_index, end_index) for each strike
        }
        
        self.is_recording = False
        self.current_episode_id = self._get_next_episode_id()
        self.current_target_id = "default"
        
        self._init_db()

    def _init_db(self):
        """Initialize DuckDB table if it doesn't exist."""
        con = duckdb.connect(self.db_path)
        con.execute("""
            CREATE TABLE IF NOT EXISTS episodes (
                episode_id INTEGER PRIMARY KEY,
                target_id VARCHAR,
                file_path VARCHAR,
                num_strikes INTEGER,
                max_stress FLOAT,
                sim_duration_steps INTEGER,
                timestamp TIMESTAMP
            )
        """)
        con.close()

    def _get_next_episode_id(self):
        """Query DB to find the next available episode ID."""
        try:
            con = duckdb.connect(self.db_path)
            res = con.execute("SELECT MAX(episode_id) FROM episodes").fetchone()
            con.close()
            return (res[0] + 1) if res[0] is not None else 0
        except:
            return 0

    def start_new_episode(self, target_id):
        """
        Starts a new recording episode.
        If data exists in buffer for a *different* target or from a previous run, 
        flush_to_disk() should have been called before this.
        """
        self.buffer = {
            "mpm_pos": [],
            "mpm_stress": [],
            "force_torque": [],
            "qpos": [],
            "target_id": target_id,
            "strike_indices": []
        }
        self.current_target_id = target_id
        self.is_recording = True
        # Episode ID increments only on successful save, but we track 'current' for logging
        print(f"[Recorder] Started new episode for target: {target_id}")

    def record_frame(self, sim_state, robot_state):
        """
        Appends a single frame of data to the buffer.
        
        Args:
            sim_state: The MPM entity state object (providing pos, Jp, etc.)
            robot_state: The robot state object (providing ee_force, qpos)
        """
        if not self.is_recording:
            return

        # MPM Data
        # Assuming sim_state is the MPM entity state
        # For deformation, 'Jp' (plastic deformation) or 'von_mises' is useful.
        # We'll calculate Von Mises from F if not directly available, or just store Jp.
        # Let's stick to the plan: Particles + Stress (if cheap) or Jp.
        # genesis.utils.particle doesn't give stress directly usually, but check solver.
        
        # Helper to convert to numpy safely
        def to_cpu(tensor):
            if isinstance(tensor, torch.Tensor):
                return tensor.detach().cpu().numpy()
            return tensor

        # Positions
        pos = to_cpu(sim_state.pos) # (B, N, 3) -> (N, 3) if B=1
        if pos.ndim == 3: pos = pos[0]
        
        # Plastic Deformation (Jp) - good proxy for "how much it deformed"
        # If Jp is not available, we might skip it or use F.
        jp = to_cpu(sim_state.Jp)
        if jp.ndim == 2: jp = jp[0]  # (B, N) -> (N,)

        # Robot Data
        # Force/Torque from wrist sensor
        # Genesis robot entities might store this differently. 
        # Using a placeholder if specific API isn't known, usually get_dofs_force() or similar.
        # For now, let's assume we can get feedback or just record qpos/effort.
        # The user's prompt mentioned "force/strain", implying simulation feedback.
        
        # qpos
        qpos = to_cpu(robot_state.get_qpos())
        if qpos.ndim == 2: qpos = qpos[0]

        # Append
        self.buffer["mpm_pos"].append(pos)
        self.buffer["mpm_stress"].append(jp) # Using Jp as the scalar field for now
        self.buffer["qpos"].append(qpos)
        
        # Force - Ideally we want EE force. 
        # For now, using dof forces as proxy if EE force not available directly.
        forces = to_cpu(robot_state.get_dofs_force())
        if forces.ndim == 2: forces = forces[0]
        self.buffer["force_torque"].append(forces)

    def mark_strike_start(self):
        """Call this when a strike begins (to track segments for Undo)."""
        current_idx = len(self.buffer["mpm_pos"])
        self.buffer["strike_indices"].append({"start": current_idx, "end": -1})

    def mark_strike_end(self):
        """Call this when a strike ends."""
        if self.buffer["strike_indices"]:
            self.buffer["strike_indices"][-1]["end"] = len(self.buffer["mpm_pos"])

    def handle_undo(self):
        """
        Removes the data corresponding to the last recorded strike.
        """
        if not self.buffer["strike_indices"]:
            print("[Recorder] Nothing to undo.")
            return

        last_strike = self.buffer["strike_indices"].pop()
        start_idx = last_strike["start"]
        
        # Truncate lists to the start of the undone strike
        print(f"[Recorder] Undoing strike. Reverting buffer from {len(self.buffer['mpm_pos'])} to {start_idx} frames.")
        
        self.buffer["mpm_pos"] = self.buffer["mpm_pos"][:start_idx]
        self.buffer["mpm_stress"] = self.buffer["mpm_stress"][:start_idx]
        self.buffer["force_torque"] = self.buffer["force_torque"][:start_idx]
        self.buffer["qpos"] = self.buffer["qpos"][:start_idx]

    def flush_to_disk(self):
        """
        Writes the buffered data to an HDF5 file and updates DuckDB.
        Handles variable particle counts by padding to N_max and saving a mask.
        """
        if not self.buffer["mpm_pos"]:
            print("[Recorder] Buffer empty, skipping save.")
            return

        ep_id = self._get_next_episode_id()
        
        # 1. Sharding
        shard_id = f"{ep_id // 1000:03d}"
        shard_path = os.path.join(self.train_dir, shard_id)
        os.makedirs(shard_path, exist_ok=True)
        filename = f"ep_{ep_id:06d}.h5"
        file_path = os.path.join(shard_path, filename)
        
        # 2. Key Metrics & Ragged Array Handling
        # Determine max particles in this episode
        max_particles = max(len(frame) for frame in self.buffer["mpm_pos"]) if self.buffer["mpm_pos"] else 0
        num_frames = len(self.buffer["mpm_pos"])
        
        # Pre-allocate padded arrays
        start_time = time.time()
        mpm_pos_arr = np.zeros((num_frames, max_particles, 3), dtype=np.float32)
        mpm_stress_arr = np.zeros((num_frames, max_particles), dtype=np.float32)
        mpm_mask_arr = np.zeros((num_frames, max_particles), dtype=bool)
        
        # Fill padded arrays
        for i in range(num_frames):
            n_p = len(self.buffer["mpm_pos"][i])
            mpm_pos_arr[i, :n_p, :] = self.buffer["mpm_pos"][i]
            mpm_stress_arr[i, :n_p] = self.buffer["mpm_stress"][i]
            mpm_mask_arr[i, :n_p] = True # Mark active particles
            
        force_arr = np.array(self.buffer["force_torque"], dtype=np.float32)
        qpos_arr = np.array(self.buffer["qpos"], dtype=np.float32)
        
        max_stress = float(np.max(mpm_stress_arr)) if mpm_stress_arr.size > 0 else 0.0
        num_strikes = len(self.buffer["strike_indices"])
        duration = num_frames
        
        # 3. Write HDF5
        try:
            with h5py.File(file_path, "w") as f:
                # Metadata
                f.attrs["sim_version"] = "1.0"
                f.attrs["target_id"] = self.buffer["target_id"]
                
                # Datasets with LZF Compression for large arrays
                # LZF is extremely efficient for zero-padded regions
                f.create_dataset("state/mpm_particles", data=mpm_pos_arr, compression="lzf")
                f.create_dataset("state/mpm_stress", data=mpm_stress_arr, compression="lzf")
                f.create_dataset("state/mpm_mask", data=mpm_mask_arr, compression="lzf") # New Mask
                
                # Smaller arrays don't strict need compression but good practice
                f.create_dataset("obs/force_torque", data=force_arr)
                f.create_dataset("obs/qpos", data=qpos_arr)
                
            print(f"[Recorder] Saved Episode {ep_id} to {file_path} (Max Particles: {max_particles})")
            
            # 4. Update DuckDB
            con = duckdb.connect(self.db_path)
            con.execute(f"""
                INSERT INTO episodes VALUES 
                ({ep_id}, '{self.buffer['target_id']}', '{file_path}', {num_strikes}, {max_stress}, {duration}, current_timestamp)
            """)
            con.close()
            
            # Success - update ID for next time (strictly the DB query handles this but good to have local sync)
            self.current_episode_id = ep_id + 1
            
        except Exception as e:
            print(f"[Recorder] Failed to save episode: {e}")
            # If failed, we might want to delete the partial file?
            if os.path.exists(file_path):
                os.remove(file_path)

        # Clear buffer after flush
        self.buffer = {
            "mpm_pos": [],
            "mpm_stress": [],
            "force_torque": [],
            "qpos": [],
            "target_id": self.buffer["target_id"], # Keep target ID for next sequence
            "strike_indices": []
        }
