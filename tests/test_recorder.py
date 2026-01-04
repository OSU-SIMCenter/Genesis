import sys
import os
import shutil
import numpy as np
import h5py
import duckdb
import torch
import time

# Add agforge to path
sys.path.append(os.path.join(os.getcwd(), 'agforge'))

from recorder import AgForgeRecorder

# Mock Classes
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
    def Jp(self):
        return torch.rand((1, self.n_particles))

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
    
    print("--- Starting Recorder Test (Ragged Support) ---")
    
    recorder = AgForgeRecorder(data_dir=test_dir)
    mock_sim_state = MockEntity("mpm")
    mock_robot = MockRobot()
    
    target_id = "cylinder_test_ragged"
    
    # 1. Start Episode
    print(f"1. Starting Episode for {target_id}")
    recorder.start_new_episode(target_id)
    
    # 2. Simulate Strike 1 (5 frames at N=100)
    print("2. Simulating Strike 1 (N=100)...")
    mock_sim_state.n_particles = 100
    recorder.mark_strike_start()
    for _ in range(5):
        recorder.record_frame(mock_sim_state, mock_robot.entity)
    recorder.mark_strike_end()
    
    # 3. Simulate Cut/Loss (5 frames at N=80)
    print("3. Simulating Strike 2 after cutting (N=80)...")
    mock_sim_state.n_particles = 80
    recorder.mark_strike_start()
    for _ in range(5):
        recorder.record_frame(mock_sim_state, mock_robot.entity)
    recorder.mark_strike_end()
    
    # 6. Flush (End of Episode)
    print("4. Flushing to disk...")
    recorder.flush_to_disk()
    
    # --- VERIFICATION ---
    print("\n--- Verifying Output ---")
    
    # Look for file in db
    db_con = duckdb.connect(os.path.join(test_dir, "metadata.duckdb"))
    res = db_con.execute("SELECT file_path FROM episodes").fetchone()
    db_con.close()
    file_path = res[0]
    
    # Check HDF5 Content
    with h5py.File(file_path, "r") as f:
        print("HDF5 Keys:", list(f.keys()))
        particles = f["state/mpm_particles"][:]
        mask = f["state/mpm_mask"][:]
        
        print(f"Particles Shape: {particles.shape}")
        print(f"Mask Shape: {mask.shape}")
        
        # Check Total Frames
        assert particles.shape[0] == 10, "Total frames should be 10"
        
        # Check Max Dimension
        assert particles.shape[1] == 100, "2nd dim should be max particles (100)"
        
        # Check First 5 Frames (N=100)
        assert np.all(mask[0:5, :]), "First 5 frames should have all mask=True"
        
        # Check Last 5 Frames (N=80)
        # First 80 should be True, last 20 should be False
        assert np.all(mask[5:, :80]), "Last 5 frames: First 80 particles should be True"
        assert not np.any(mask[5:, 80:]), "Last 5 frames: Last 20 particles should be False (Padding)"
        
        # Check Padding Correctness (should be zero in pos)
        assert np.all(particles[5:, 80:] == 0), "Padded particle values should be 0"

    print("\n--- TEST PASSED ---")

if __name__ == "__main__":
    test_recording_flow()
