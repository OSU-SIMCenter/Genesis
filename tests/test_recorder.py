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
        self.pos = torch.rand((1, 100, 3)) # 100 particles
        self.Jp = torch.rand((1, 100))     # 100 stress values

    def get_qpos(self):
        return torch.rand((1, 4))
    
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
    
    print("--- Starting Recorder Test ---")
    
    recorder = AgForgeRecorder(data_dir=test_dir)
    mock_sim_state = MockEntity("mpm")
    mock_robot = MockRobot()
    
    target_id = "cylinder_test_01"
    
    # 1. Start Episode
    print(f"1. Starting Episode for {target_id}")
    recorder.start_new_episode(target_id)
    
    # 2. Simulate Strike 1 (5 frames)
    print("2. Simulating Strike 1...")
    recorder.mark_strike_start()
    for _ in range(5):
        recorder.record_frame(mock_sim_state, mock_robot.entity)
    recorder.mark_strike_end()
    
    # 3. Simulate Strike 2 (5 frames) - TO BE UNDONE
    print("3. Simulating Strike 2 (Mistake)...")
    recorder.mark_strike_start()
    for _ in range(5):
        recorder.record_frame(mock_sim_state, mock_robot.entity)
    recorder.mark_strike_end()
    
    # 4. Undo Strike 2
    print("4. Undoing Strike 2...")
    recorder.handle_undo()
    
    # Verify buffer len (should be 5, not 10)
    print(f"   Buffer length after undo: {len(recorder.buffer['mpm_pos'])}")
    assert len(recorder.buffer['mpm_pos']) == 5, "Undo failed to remove frames!"
    
    # 5. Simulate Strike 3 (5 frames) - Valid
    print("5. Simulating Strike 3...")
    recorder.mark_strike_start()
    for _ in range(5):
        recorder.record_frame(mock_sim_state, mock_robot.entity)
    recorder.mark_strike_end()
    
    # 6. Flush (End of Episode)
    print("6. Flushing to disk...")
    recorder.flush_to_disk()
    
    # --- VERIFICATION ---
    print("\n--- Verifying Output ---")
    
    # Check File Existence
    db_con = duckdb.connect(os.path.join(test_dir, "metadata.duckdb"))
    res = db_con.execute("SELECT * FROM episodes").fetchall()
    db_con.close()
    
    print(f"DB Content: {res}")
    assert len(res) == 1, "Database should have 1 entry"
    
    row = res[0]
    ep_id = row[0]
    file_path = row[2]
    num_strikes = row[3]
    
    assert num_strikes == 2, f"Should have 2 valid strikes, got {num_strikes}"
    assert os.path.exists(file_path), "HDF5 file not found"
    
    # Check HDF5 Content
    with h5py.File(file_path, "r") as f:
        print("HDF5 Keys:", list(f.keys()))
        particles = f["state/mpm_particles"][:]
        stress = f["state/mpm_stress"][:]
        
        print(f"Particles Shape: {particles.shape}")
        
        assert particles.shape[0] == 10, "Should have 10 frames total (5 from Strike 1, 5 from Strike 3)"
        assert f.attrs["target_id"] == target_id, "Target ID mismatch"
        
    print("\n--- TEST PASSED ---")

if __name__ == "__main__":
    test_recording_flow()
