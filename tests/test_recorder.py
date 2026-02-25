import os
import sys
import shutil
import torch
import numpy as np
import h5py
import duckdb
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from agforge.recorder import AgForgeRecorder

class MockEntity:
    def __init__(self, name):
        self.name = name
        self.n_particles = 100
        
    def get_qpos(self):
        return torch.rand((1, 4))
    
    @property
    def pos(self):
        return torch.rand((1, self.n_particles, 3))
    
    @property
    def vel(self):
        return torch.rand((1, self.n_particles, 3))

    def get_dofs_force(self):
        return torch.rand((1, 4))

class MockRobot:
    def __init__(self):
        self.entity = MockEntity("robot")
    
    def get_dofs_force(self):
        return self.entity.get_dofs_force()
    
    def get_qpos(self):
        return self.entity.get_qpos()

def test_recording_flow():
    test_dir = "test_data"
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    os.makedirs(test_dir, exist_ok=True)
    
    print("--- Starting Recorder Test (V2.1 Values+Offsets) ---")
    
    recorder = AgForgeRecorder(data_dir=test_dir, shard_capacity=2)
    mock_sim_state = MockEntity("mpm")
    mock_robot = MockRobot()
    
    # === EPISODE 0 ===
    print("\n[Test] === EPISODE 0 ===")
    recorder.start_new_episode("target_0")
    
    print("[Test] Simulating Strike 1 (N=100)...")
    mock_sim_state.n_particles = 100
    recorder.mark_strike_start()
    for _ in range(5):
        recorder.record_frame(mock_sim_state.pos, mock_sim_state.vel, mock_robot.get_qpos(), torch.tensor(1.0), torch.tensor(2.0), torch.rand(4))
        
    print("[Test] Simulating Cut/Loss (N=80)...")
    mock_sim_state.n_particles = 80
    recorder.mark_strike_start()
    for _ in range(5):
        recorder.record_frame(mock_sim_state.pos, mock_sim_state.vel, mock_robot.get_qpos(), torch.tensor(1.5), torch.tensor(2.5), torch.rand(4))
        
    print("[Test] Flushing Episode 0...")
    recorder.flush_episode()
    
    # === EPISODE 1 ===
    print("\n[Test] === EPISODE 1 ===")
    recorder.start_new_episode("target_1")
    print("[Test] Simulating Strike (N=80)...")
    mock_sim_state.n_particles = 80
    recorder.mark_strike_start()
    for _ in range(10):
        recorder.record_frame(mock_sim_state.pos, mock_sim_state.vel, mock_robot.get_qpos(), torch.tensor(1.0), torch.tensor(2.0), torch.rand(4))
    
    print("[Test] Undo Strike!")
    recorder.handle_undo()
    
    print("[Test] Simulating Real Strike (N=90)...")
    mock_sim_state.n_particles = 90
    recorder.mark_strike_start()
    for _ in range(10):
        recorder.record_frame(mock_sim_state.pos, mock_sim_state.vel, mock_robot.get_qpos(), torch.tensor(1.0), torch.tensor(2.0), torch.rand(4))
        
    print("[Test] Flushing Episode 1...")
    recorder.flush_episode()
    
    # --- VERIFICATION ---
    print("\n--- Verifying Output ---")
    
    # Look for file in parquet
    df = pd.read_parquet(os.path.join(test_dir, "episodes_catalog.parquet"))
    print("\n=== Parquet Catalog ===")
    print(df)
    
    assert len(df) == 2, "Should have 2 episodes in catalog"
    assert df.iloc[0]["shard_file"] == "shard_0000.h5"
    assert df.iloc[1]["shard_file"] == "shard_0000.h5"
    
    file_path = os.path.join(test_dir, "train", "shard_0000.h5")
    assert os.path.exists(file_path), "HDF5 shard file not found"
    
    # Check HDF5 Content
    with h5py.File(file_path, "r") as f:
        print("\n=== HDF5 File (shard_0000.h5) ===")
        print("Groups:", list(f.keys()))
        assert "ep_000000" in f
        assert "ep_000001" in f
        
        ep0 = f["ep_000000"]
        
        pos = ep0["observations/state/particles/pos_values"][:]
        offsets = ep0["observations/state/particles/offsets"][:]
        qpos = ep0["observations/state/scene/qpos"][:]
        forces = ep0["observations/state/scene/force_torque"][:]
        
        # Math: 5 frames at N=100 + 5 frames at N=80 = 500 + 400 = 900
        assert pos.shape == (900, 3), f"Expected 900 particles, got {pos.shape[0]}"
        assert offsets.shape == (11,), f"Expected 11 offsets (10 frames + 1), got {offsets.shape[0]}"
        assert offsets[0] == 0
        assert offsets[5] == 500
        assert offsets[10] == 900
        
        assert qpos.shape == (10, 4)
        assert forces.shape == (10, 2)
        
        ep1 = f["ep_000001"]
        pos1 = ep1["observations/state/particles/pos_values"][:]
        offsets1 = ep1["observations/state/particles/offsets"][:]
        
        # Math: The first 10 frames (N=80) were undone.
        # Only 10 frames (N=90) should remain. = 900
        assert pos1.shape == (900, 3)
        assert offsets1.shape == (11,)
        
    print("\n--- TEST PASSED ---")

if __name__ == "__main__":
    test_recording_flow()
