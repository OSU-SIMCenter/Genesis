import numpy as np
import trimesh
import igl
import time
import sys
from scipy.stats import qmc

def generate_hcp_grid(bounds, p_size):
    """
    Generate points in a Hexagonal Close Packing (HCP) grid within bounds.
    """
    x_min, y_min, z_min = bounds[0]
    x_max, y_max, z_max = bounds[1]
    
    # HCP parameters
    # Distance between stored particles
    d = p_size
    
    # Vertical (Z) spacing
    dz = d * np.sqrt(2/3)
    # Horizontal (Y) spacing
    dy = d * np.sqrt(3) / 2
    # Horizontal (X) spacing
    dx = d

    # Calculate number of points in each dimension
    nz = int(np.ceil((z_max - z_min) / dz)) + 1
    ny = int(np.ceil((y_max - y_min) / dy)) + 1
    nx = int(np.ceil((x_max - x_min) / dx)) + 1

    print(f"Grid dims: {nx}, {ny}, {nz}")

    points = []
    
    # We can iterate or use meshgrid. Use loops for clarity first or meshgrid for speed.
    # Speed is important for volume sampling.
    
    # Create grid indices
    z_idx = np.arange(nz)
    y_idx = np.arange(ny)
    x_idx = np.arange(nx)
    
    # Z coordinate
    z = z_min + z_idx * dz
    
    # For HCP, X and Y shift based on Z layer and Y row
    
    # Vectorized approach:
    # We need to construct all (x,y,z) candidates.
    
    # Let's generate a full block and then shift.
    # XYZ = np.stack(np.meshgrid(x_min + x_idx * dx, y_min + y_idx * dy, z, indexing='ij'), -1).reshape(-1, 3)
    # This is simple cubic.
    
    # HCP logic:
    # shift_x = ( (y_idx % 2) + (z_idx % 2) ) % 2 * (dx / 2)
    # shift_y = ( (z_idx % 2) * (dy / 3) ) ?? No, simpler model:
    # Layer A vs Layer B.
    
    # Standard HCP:
    # Layer 0: Hexagonal grid
    # Layer 1: Shifted centroid
    
    # Let's stick to a simpler "Stacked Triangular Layers" which is basically HCP.
    
    all_points = []
    
    for k in range(nz):
        z_curr = z_min + k * dz
        
        # Shift Y for alternate layers? No, usually Y grid is same, just shifted X?
        # Actually in HCP, every other layer is shifted.
        
        # Shift for Z layers
        offset_y = ((k % 2) * dy) / 3.0 # This is FCC/HCP specific, let's just do simple offsets
        # A common simple dense packing is just offsetting every row and every layer.
        
        # Let's use:
        # Row offset (Y): if row is odd, shift X by 0.5 * dx
        # Layer offset (Z): shift both X and Y?
        
        # Simplest dense packing:
        # If z index is odd: shift x by 0.5 dx, shift y by 0.5 dy (approx)
        
        # Proper HCP:
        # Coordinates:
        # x = i * d + (j % 2) * 0.5 * d + (k % 2) * 0.5 * d
        # y = j * (sqrt(3)/2) * d + (k % 2) * (sqrt(3)/6) * d  <-- this is tricky for exact HCP
        # z = k * sqrt(2/3) * d
        
        ys = y_min + y_idx * dy
        if k % 2 == 1:
             ys += dy / 3.0 # Shift Y
             
        # Create X grid for this Z layer
        # For each Y, we have a line of X
        
        # We can construct X and Y mesh for this layer
        
        # X coordinates depend on Y index (j)
        # x = i * dx + (j%2) * 0.5 * dx
        
        # Let's build 2D hexagonal grid for this Z layer first
        xx, yy = np.meshgrid(x_idx, y_idx, indexing='xy') # Shape (ny, nx)
        
        # Calc actual coords
        x_pos = x_min + xx * dx
        y_pos = y_min + yy * dy
        
        # Shift X based on Y row index
        # shape of yy is (ny, nx), y_idx corresponds to rows 0..ny-1
        # indices of Y:
        j_indices = np.arange(ny)[:, None] # Column vector
        
        x_offset = (j_indices % 2) * 0.5 * dx
        x_pos += x_offset
        
        # Now apply Z shifts to the whole layer
        if k % 2 == 1:
            x_pos += 0.5 * dx
            y_pos += dy / 3.0 # Center of triangle roughly
            
        z_pos = np.full_like(x_pos, z_curr)
        
        layer_points = np.stack([x_pos.flatten(), y_pos.flatten(), z_pos.flatten()], axis=1)
        all_points.append(layer_points)
        
    return np.concatenate(all_points, axis=0)

        
    return np.concatenate(all_points, axis=0)

def generate_scipy_poisson(bounds, p_size):
    """
    Generate points using Scipy's Poisson Disk sampler.
    """
    x_min, y_min, z_min = bounds[0]
    x_max, y_max, z_max = bounds[1]
    
    dims = np.array([x_max - x_min, y_max - y_min, z_max - z_min])
    # PoissonDisk works on unit hypercube [0, 1)^d
    # We need to determine the radius in the unit cube that corresponds to p_size in world space.
    # We normalized by the LARGEST dimension to ensure 'radius' is respected in all directions 
    # (since the space is squashed if we normalize non-uniformly? No, qmc is unit cube).
    
    # Actually, PoissonDisk samples in [0, 1].
    # If we map [0,1] to [0, L], then dist d in [0,1] becomes d*L in world.
    # We want d*L >= p_size  =>  d >= p_size / L.
    # But L is different for x, y, z.
    # We should normalize by the *maximum* dimension to fit the box in the unit cube, 
    # or normalize each independently?
    # If we normalize independently: x' = x/Lx. 
    # Distance in unit cube doesn't map linearly to distance in world if Lx != Ly.
    # So we must scale by max dimension.
    
    scale = np.max(dims)
    if scale == 0: return np.zeros((0, 3))
    
    norm_radius = p_size / scale
    
    # Limit radius to avoid hanging on massive number of points?
    # For testing, we keep it reasonable.
    
    print(f"Scipy Poisson: Scale={scale:.4f}, Norm Radius={norm_radius:.6f}")
    
    engine = qmc.PoissonDisk(d=3, radius=norm_radius, hypersphere='volume', ncandidates=30)
    
    # fill_space might be slow for huge N.
    # It returns samples in [0, 1]
    
    try:
        samples = engine.fill_space()
    except Exception as e:
        print(f"Scipy generation failed: {e}")
        return np.zeros((0, 3))
    
    # Scale back
    # We embedded our box of size `dims` into a cube of size `scale`.
    # So we used [0, 1]^3 covering physical size [scale]^3.
    # Valid physical points are those within `bounds`.
    
    world_samples = samples * scale + bounds[0]
    
    # Crop to actual bounding box (since we generated in a cube of size max_dim)
    # Rejection for box
    mask = (world_samples[:, 0] <= x_max) & \
           (world_samples[:, 1] <= y_max) & \
           (world_samples[:, 2] <= z_max)
           
    return world_samples[mask]

def test_volume_sampling():
    # Create a test mesh (sphere)
    print("Creating test mesh...")
    mesh = trimesh.creation.icosphere(radius=1.0, subdivisions=3)
    # mesh = trimesh.creation.box(extents=(1,1,1))
    
    p_size = 0.05 # 5cm particles -> ~32000 in 2x2x2 box?  (2/.05)^3 = 64000
    
    print(f"Sampling with p_size={p_size}")
    
    # 1. HCP
    print("-" * 30)
    print("Testing HCP Sampler...")
    t0 = time.time()
    pts_hcp = generate_hcp_grid(mesh.bounds, p_size)
    gen_time_hcp = time.time() - t0
    
    t0 = time.time()
    sd, _, _ = igl.signed_distance(pts_hcp, mesh.vertices, mesh.faces)
    final_hcp = pts_hcp[sd < 0]
    filter_time_hcp = time.time() - t0
    
    print(f"HCP: Generated {len(pts_hcp)} in {gen_time_hcp:.4f}s, Final {len(final_hcp)} in {filter_time_hcp:.4f}s")
    
    # 2. Scipy
    print("-" * 30)
    print("Testing Scipy Poisson Sampler...")
    if 'scipy' in sys.modules:
        t0 = time.time()
        pts_scipy = generate_scipy_poisson(mesh.bounds, p_size)
        gen_time_scipy = time.time() - t0
        
        t0 = time.time()
        sd, _, _ = igl.signed_distance(pts_scipy, mesh.vertices, mesh.faces)
        final_scipy = pts_scipy[sd < 0]
        filter_time_scipy = time.time() - t0
        
        print(f"Scipy: Generated {len(pts_scipy)} in {gen_time_scipy:.4f}s, Final {len(final_scipy)} in {filter_time_scipy:.4f}s")
    else:
        print("Scipy not available, skipping.")

if __name__ == "__main__":
    test_volume_sampling()
