import os
import argparse
import sys
import contextlib
import h5py
import pandas as pd
import numpy as np
import torch
import rerun as rr
import genesis as gs

# Ensure agforge is reachable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from agforge.reconstruction import SurfaceReconstructor

# Initialize Genesis so utility functions mapped to gs.logger don't crash with NoneType
gs.init(logging_level="warning")

# --- Offline Mocks for SurfaceReconstructor ---
class _MockProfiler:
    @contextlib.contextmanager
    def time(self, name):
        yield

class _MockScene:
    def __init__(self):
        self.profiling_options = type('obj', (object,), {'profiler': _MockProfiler()})
        self.sim = type('obj', (object,), {
            'mpm_solver': type('obj', (object,), {'particle_radius': 0.00186})
        })
        self.visualizer = True  # Must be truthy so update_skinning() copies verts back to numpy

class _MockMPMEntity:
    def __init__(self):
        self.particles = None
        self.active_mask = None
    
    def get_particles_pos(self, envs_idx=0):
        return self.particles.unsqueeze(0)
        
    def get_particles_active(self, envs_idx=0):
        return self.active_mask.unsqueeze(0)

class MockEnv:
    def __init__(self, device):
        self.device = device
        self.scene = _MockScene()
        self.mpm_entity = _MockMPMEntity()
# -----------------------------------------------

def visualize_episode_rerun(data_dir, episode_id, run_splashsurf=True):
    """
    Visualizes an episode's ragged arrays using Rerun.
    Uses the native SurfaceReconstructor with Taubin smoothing for high-quality meshes.
    """
    catalog_path = os.path.join(data_dir, "episodes_catalog.parquet")
    if not os.path.exists(catalog_path):
        print(f"Error: Catalog not found at {catalog_path}")
        return
        
    df = pd.read_parquet(catalog_path)
    ep_rows = df[df["episode_global_id"] == episode_id]
    
    if ep_rows.empty:
        print(f"Error: Episode {episode_id} not found.")
        return
        
    ep_info = ep_rows.iloc[0]
    shard_file = ep_info["shard_file"]
    group_name = ep_info["internal_group"]
    num_frames = ep_info["length"]
    
    shard_path = os.path.join(data_dir, "train", shard_file)
    
    rr.init(f"agforge_episode_{episode_id}")
    out_file = f"episode_{episode_id}.rrd"
    rr.save(out_file)
    print(f"Logging data to {out_file}...")
    
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    # Initialize the Native AgForge Surface Reconstructor
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mock_env = MockEnv(device)
    recon = SurfaceReconstructor(mock_env)
    
    with h5py.File(shard_path, "r") as f:
        ep_group = f[group_name]
        
        particles_flat = ep_group["observations/state/particles/pos_values"][:]
        offsets = ep_group["observations/state/particles/offsets"][:]
        force_seq = ep_group["observations/state/scene/force_torque"][:]
        
        particle_size = 0.00186

        for t in range(num_frames):
            rr.set_time("frame", sequence=t)
            
            start_idx = offsets[t]
            end_idx = offsets[t+1]
            frame_particles = particles_flat[start_idx:end_idx]
            
            if len(frame_particles) == 0:
                continue

            # Log forces
            f_L, f_R = force_seq[t]
            rr.log("metrics/force_left", rr.Scalars(f_L))
            rr.log("metrics/force_right", rr.Scalars(f_R))
            
            # Log raw points
            rr.log(
                "world/material_points",
                rr.Points3D(frame_particles, radii=particle_size * 0.3, colors=[[0, 150, 255, 200]])
            )
            
            if run_splashsurf:
                # Supply GPU particles to MockEnv for Reconstructor
                parts_tensor = torch.from_numpy(frame_particles).float().to(device)
                mock_env.mpm_entity.particles = parts_tensor
                mock_env.mpm_entity.active_mask = torch.ones(len(parts_tensor), dtype=torch.bool, device=device)
                
                # Advance global frame counter manually
                recon._global_frame = t + 1
                
                # We do full recon on first frame, then skinning on subsequent (or rebind if quality drops)
                if not recon.skinning_enabled:
                    recon.create_reconstructed_mesh()
                    recon.init_skinning()
                else:
                    # Update skinning + smoothing
                    recon.update_skinning()
                
                # Fetch output mesh
                mesh = recon.get_mesh_data()
                if mesh is not None and len(mesh.vertices) > 0:
                    rr.log(
                        "world/material_mesh",
                        rr.Mesh3D(
                            vertex_positions=mesh.vertices,
                            triangle_indices=mesh.faces,
                            vertex_normals=mesh.vertex_normals,
                            vertex_colors=[[200, 100, 50, 255]] * len(mesh.vertices)
                        )
                    )

    print("\nOffline playback timeline fully logged to Rerun!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize V2.1 Ragged Data with Rerun")
    parser.add_argument("--dir", type=str, default="data", help="Path to data directory")
    parser.add_argument("--ep", type=int, default=0, help="Global Episode ID to visualize")
    parser.add_argument("--no-mesh", action="store_true", help="Skip surface reconstruction and just show points")
    args = parser.parse_args()
    
    visualize_episode_rerun(args.dir, args.ep, run_splashsurf=not args.no_mesh)
