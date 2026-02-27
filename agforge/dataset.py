import os
import random
import torch
import h5py
import pandas as pd
import numpy as np
from torch.utils.data import IterableDataset, get_worker_info

class AgForgeIterableDataset(IterableDataset):
    """
    An IterableDataset specifically optimized for Sharded HDF5 + Parquet.
    Uses '3-Tier Weak Shuffling' to maximize sequential disk read throughput while
    providing randomized batches for RL training.
    """
    def __init__(self, catalog_path: str, data_dir: str, shuffle_buffer_size: int = 2000):
        super().__init__()
        self.catalog_path = catalog_path
        self.data_dir = data_dir
        self.shuffle_buffer_size = shuffle_buffer_size
        
        if not os.path.exists(self.catalog_path):
            raise FileNotFoundError(f"Parquet catalog not found: {self.catalog_path}")
            
        # 1. Master reads Parquet ONCE
        df = pd.read_parquet(self.catalog_path)
        
        # 2. Group by shard to guarantee sequential disk access
        # Convert to native Python dicts to prevent PyArrow multiprocessing deadlocks/memory leaks
        self.shards_dict = {}
        for shard_file, group in df.groupby('shard_file'):
            self.shards_dict[shard_file] = group.to_dict('records')
            
        self.shard_names = list(self.shards_dict.keys())
        
        if len(self.shard_names) == 0:
            raise ValueError("Catalog is empty; no episodes to load.")

    def __iter__(self):
        worker_info = get_worker_info()
        
        # 1. Assign specific shards to specific workers
        if worker_info is None:
            # Single-process data loading
            worker_shards = self.shard_names
        else:
            # Distributed data loading (e.g., Worker 0 gets shards 0, 8, 16... Worker 1 gets 1, 9, 17...)
            worker_shards = self.shard_names[worker_info.id :: worker_info.num_workers]

        # 2. Tier 1 Shuffle: Shuffle the order of shards assigned to this worker
        random.shuffle(worker_shards)

        shuffle_buffer = []

        # Process one shard at a time
        for shard_file in worker_shards:
            episodes = self.shards_dict[shard_file]
            
            # 3. Tier 2 Shuffle: Shuffle episodes within the shard
            random.shuffle(episodes)
            
            shard_path = os.path.join(self.data_dir, "train", shard_file)

            # 4. OPEN HDF5 SAFELY: Inside __iter__ (Worker process only)
            # swmr=True prevents OS-level locking issues if multiple readers touch the file
            try:
                with h5py.File(shard_path, 'r', swmr=True) as h5f:
                    
                    for ep_meta in episodes:
                        group_name = ep_meta['internal_group']
                        if group_name not in h5f:
                            continue # Skip corrupted/missing groups
                            
                        ep_group = h5f[group_name]
                        
                        # 5. Read the ragged arrays and continuous state into memory for this episode
                        # Using [:] pulls data from HDF5 disk into an in-memory NumPy array instantly
                        particles_flat = ep_group['observations/state/particles/pos_values'][:]
                        offsets = ep_group['observations/state/particles/offsets'][:]
                        qpos_seq = ep_group['observations/state/scene/qpos'][:]
                        force_seq = ep_group['observations/state/scene/force_torque'][:]
                        action_seq = ep_group['actions/dof_velocity_cmd'][:]
                        
                        # Extract individual frames/point clouds from the ragged format
                        num_frames = len(offsets) - 1
                        for t in range(num_frames):
                            start_idx = offsets[t]
                            end_idx = offsets[t+1]
                            
                            point_cloud_t = particles_flat[start_idx:end_idx]
                            
                            sample = {
                                'metadata': {
                                    'episode_id': ep_meta['episode_global_id'],
                                    'timestep': t,
                                    'language_instruction': ep_meta.get('language_instruction', b"").decode('utf-8') if isinstance(ep_meta.get('language_instruction', ""), bytes) else ep_meta.get('language_instruction', "")
                                },
                                'particles': point_cloud_t,
                                'qpos': qpos_seq[t],
                                'force': force_seq[t],
                                'action': action_seq[t]
                            }
                            
                            shuffle_buffer.append(sample)

                            # 6. Tier 3 Shuffle: Yield randomly from the shuffle buffer using O(1) swap-and-pop
                            if len(shuffle_buffer) >= self.shuffle_buffer_size:
                                idx = random.randint(0, len(shuffle_buffer) - 1)
                                shuffle_buffer[idx], shuffle_buffer[-1] = shuffle_buffer[-1], shuffle_buffer[idx]
                                yield shuffle_buffer.pop()
                                
            except OSError as e:
                print(f"Worker failed to read shard {shard_path}: {e}")
                continue

        # Drain remaining buffer at the end of the epoch
        random.shuffle(shuffle_buffer)
        while shuffle_buffer:
            yield shuffle_buffer.pop()

def ragged_collate_fn(batch):
    """
    Custom collate function for DataLoader.
    Dynamically zero-pads the varying-sized particle structures to the maximum size in the batch,
    and returns a boolean attention mask for downstream ML models.
    """
    # 1. Find the largest particle count in this batch
    max_particles = max(item['particles'].shape[0] for item in batch)
    
    batch_particles = []
    batch_masks = []
    batch_qpos = []
    batch_force = []
    batch_action = []
    batch_metadata = []
    
    for item in batch:
        pts = item['particles']
        num_pts = pts.shape[0]
        
        # Pad particles with zeros to max size
        pad_size = max_particles - num_pts
        if pad_size > 0:
            padded_pts = np.pad(pts, ((0, pad_size), (0, 0)), mode='constant')
        else:
            padded_pts = pts
            
        # Create boolean mask (1 for real data, 0 for padded data)
        mask = np.zeros(max_particles, dtype=bool)
        mask[:num_pts] = True
        
        batch_particles.append(padded_pts)
        batch_masks.append(mask)
        batch_qpos.append(item['qpos'])
        batch_force.append(item['force'])
        batch_action.append(item['action'])
        batch_metadata.append(item['metadata'])

    return {
        'particles': torch.tensor(np.array(batch_particles), dtype=torch.float32),
        'mask': torch.tensor(np.array(batch_masks), dtype=torch.bool),
        'qpos': torch.tensor(np.array(batch_qpos), dtype=torch.float32),
        'force': torch.tensor(np.array(batch_force), dtype=torch.float32),
        'action': torch.tensor(np.array(batch_action), dtype=torch.float32),
        'metadata': batch_metadata
    }
