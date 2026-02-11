import time
import torch
import numpy as np
import trimesh
import gstaichi as ti
import genesis as gs
from skimage.measure import marching_cubes
from enum import Enum

# Compatibility Enum
class SamplingMethod(Enum):
    RANDOM = "random"
    VOXEL_STRATIFIED = "voxel_stratified"
    FPS = "fps"
    HALTON_LLOYD = "halton_lloyd"

@ti.data_oriented
class SurfaceReconstructor:
    def __init__(self, env):
        self.env = env
        self.reconstructed_mesh = trimesh.Trimesh()
        self.recon_enabled = True
        self.recon_frame_interval = 3  # Reconstruct every 3 frames
        self.frame_counter = 0
        
        # Compatibility / Configuration
        self.sampling_method = SamplingMethod.VOXEL_STRATIFIED
        self.recon_particle_fraction = 1.0
        self.skinning_enabled = False # Dummy flag
        self.bind_indices = None
        self.bind_weights = None
        
        # Caching for Visualization
        self._cached_particles = None
        
        # Grid Configuration
        self.grid_res = 128
        self.density = ti.field(dtype=float, shape=(self.grid_res, self.grid_res, self.grid_res))
        
        # Constants
        self.particle_radius = 0.005 
        try:
            self.particle_radius = self.env.scene.sim.mpm_solver.particle_size / 2.0
            gs.logger.info(f"Reconstruction: Inferred particle radius: {self.particle_radius}")
        except Exception:
            pass
        self.influence_radius = self.particle_radius * 2.5

    def get_state(self):
        return {
            'mesh': self.reconstructed_mesh.copy(),
            'recon_enabled': self.recon_enabled,
            'frame_counter': self.frame_counter,
        }

    def set_state(self, state):
        if state is None: return
        self.reconstructed_mesh = state['mesh'].copy()
        self.recon_enabled = state.get('recon_enabled', True)
        self.frame_counter = state.get('frame_counter', 0)

    def reset(self):
        self.reconstructed_mesh = trimesh.Trimesh()
        self.frame_counter = 0
        self._cached_particles = None
        self.skinning_enabled = False

    # --- Compatibility Interface ---
    def init_skinning(self):
        """No-op compatibility method."""
        pass
        
    def update_skinning(self):
        """No-op compatibility method."""
        pass
        
    def subdivide_long_edges(self) -> bool:
        """No-op compatibility method."""
        return False
        
    def get_active_particle_cache(self):
        """Returns cached particles for visualization."""
        if self._cached_particles is None:
             # Try to fetch if cache is empty
             return self._get_active_particles(use_cache=False)
        return self._cached_particles
        
    def _get_active_particles(self, use_cache: bool = True, apply_subsampling: bool = True):
        """Fetches active particles from MPM entity."""
        if use_cache and self._cached_particles is not None:
            return self._cached_particles
            
        try:
            mpm_entity = self.env.mpm_entity
            parts = mpm_entity.get_particles_pos(envs_idx=0).squeeze(0)
            active = mpm_entity.get_particles_active(envs_idx=0).squeeze(0)
            parts = parts[active]
            
            # Subsampling logic could go here if needed, but skipping for speed/simplicity
            # The new method is fast enough to handle all particles
            
            p_numpy = parts.cpu().numpy()
            self._cached_particles = p_numpy
            return p_numpy
        except Exception:
            return None

    @ti.kernel
    def _compute_density_kernel(
        self, 
        particles_pos: ti.types.ndarray(),
        active_mask: ti.types.ndarray(),
        n_particles: int,
        lower_bound: ti.types.vector(3, float),
        dx: float,
        influence_radius: float
    ):
        for I in ti.grouped(self.density):
            self.density[I] = 0.0
            
        for i in range(n_particles):
            if active_mask[i]:
                pos = ti.Vector([particles_pos[i, 0], particles_pos[i, 1], particles_pos[i, 2]])
                grid_pos = (pos - lower_bound) / dx
                rad_cells = influence_radius / dx
                
                base_idx = ti.cast(ti.floor(grid_pos - rad_cells), ti.int32)
                end_idx = ti.cast(ti.ceil(grid_pos + rad_cells), ti.int32)
                
                for ix in range(base_idx[0], end_idx[0] + 1):
                    for iy in range(base_idx[1], end_idx[1] + 1):
                        for iz in range(base_idx[2], end_idx[2] + 1):
                            if (0 <= ix < self.grid_res and 
                                0 <= iy < self.grid_res and 
                                0 <= iz < self.grid_res):
                                cell_center = lower_bound + ti.Vector([ix, iy, iz]) * dx
                                dist_sq = (pos - cell_center).norm_sqr()
                                if dist_sq < influence_radius**2:
                                    r2 = dist_sq / (influence_radius**2)
                                    val = (1.0 - r2)**3
                                    self.density[ix, iy, iz] += val

    def update(self, should_reconstruct: bool):
        if not self.recon_enabled:
            return
        self.frame_counter += 1
        if not should_reconstruct and (self.frame_counter % self.recon_frame_interval != 0):
            return
        self.create_reconstructed_mesh()

    def create_reconstructed_mesh(self):
        try:
            mpm_entity = self.env.mpm_entity
            particles_pos = mpm_entity.get_particles_pos(envs_idx=0).squeeze(0)
            particles_active = mpm_entity.get_particles_active(envs_idx=0).squeeze(0)
            
            n_particles = particles_pos.shape[0]
            if n_particles == 0:
                return

            # Update cache for visualization
            self._cached_particles = particles_pos[particles_active].cpu().numpy()

            min_bound = particles_pos.min(dim=0).values.cpu().numpy()
            max_bound = particles_pos.max(dim=0).values.cpu().numpy()
            
            padding = self.influence_radius * 2.0
            min_bound -= padding
            max_bound += padding
            
            extent = max_bound - min_bound
            max_extent = np.max(extent)
            if max_extent < 1e-4: max_extent = 1.0
                
            dx = max_extent / (self.grid_res - 1)
            lower_bound_ti = ti.Vector([min_bound[0], min_bound[1], min_bound[2]])
            
            self._compute_density_kernel(
                particles_pos, 
                particles_active, 
                n_particles, 
                lower_bound_ti, 
                float(dx), 
                float(self.influence_radius)
            )
            
            density_cpu = self.density.to_numpy()
            thresh = 0.5 
            
            verts, faces, normals, values = marching_cubes(
                density_cpu, 
                level=thresh,
                spacing=(dx, dx, dx),
                allow_degenerate=False
            )
            verts += min_bound
            
            self.reconstructed_mesh = trimesh.Trimesh(
                vertices=verts, 
                faces=faces, 
                vertex_normals=normals,
                process=False
            )
            
        except Exception as e:
            gs.logger.debug(f"Reconstruction skip: {e}")