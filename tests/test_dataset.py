import os
import sys
import torch
import numpy as np
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from agforge.dataset import AgForgeIterableDataset, ragged_collate_fn

def test_dataset_single_worker(catalog_path, data_dir):
    """Test standard single-process loading."""
    print("--- Testing Single Worker Loading ---")
    dataset = AgForgeIterableDataset(catalog_path, data_dir, shuffle_buffer_size=50)
    
    dataloader = DataLoader(
        dataset, 
        batch_size=8, 
        num_workers=0, 
        collate_fn=ragged_collate_fn
    )
    
    for i, batch in enumerate(dataloader):
        print(f"Batch {i}: Particles Shape={batch['particles'].shape}, Mask Shape={batch['mask'].shape}")
        
        # Verify padding correctness
        assert batch['particles'].dtype == torch.float32
        assert batch['particles_temp'].dtype == torch.float32
        assert batch['mask'].dtype == torch.bool
        
        # Check that mask length matches point cloud length
        assert batch['particles'].shape[1] == batch['mask'].shape[1]
        assert batch['particles_temp'].shape[1] == batch['mask'].shape[1]
        assert batch['particles'].shape[0] == batch['mask'].shape[0] == 8
        assert batch['qpos'].shape[0] == 8
        assert batch['force'].shape[0] == 8
        assert batch['action'].shape[0] == 8
        
        # force_torque is just [f_L, f_R] scalars
        assert batch['force'].shape[1] == 2
        
        if i >= 2:
            break
            
    print("Single worker test passed!\n")

def test_dataset_multi_worker(catalog_path, data_dir):
    """Test distributed loading with HDF5 SWMR locking."""
    print("--- Testing Multi Worker Loading (4 Workers) ---")
    dataset = AgForgeIterableDataset(catalog_path, data_dir, shuffle_buffer_size=100)
    
    try:
        dataloader = DataLoader(
            dataset, 
            batch_size=32, 
            num_workers=4, 
            prefetch_factor=2,
            collate_fn=ragged_collate_fn
        )
        
        for i, batch in enumerate(dataloader):
            print(f"Distributed Batch {i}: Particles={batch['particles'].shape}")
            if i >= 5:
                break
                
        print("Multi worker test passed! No HDF5 deadlocks detected.\n")
    except Exception as e:
        print(f"Multi worker test FAILED: {e}")
        assert False

if __name__ == "__main__":
    if not os.path.exists("data/episodes_catalog.parquet"):
        print("Catalog not found. Run simulation first: pixi run sim")
        sys.exit(1)
        
    test_dataset_single_worker("data/episodes_catalog.parquet", "data")
    test_dataset_multi_worker("data/episodes_catalog.parquet", "data")
