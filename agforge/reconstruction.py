import torch
import numpy as np
import trimesh
import trimesh.smoothing
import quadrants as qd  # type: ignore
import genesis as gs
import genesis.utils.particle as pu
import pyvista as pv
import math
from scipy.spatial import cKDTree
from enum import Enum

# Compatibility Enum
class SamplingMethod(Enum):
    RANDOM = "random"
    VOXEL_STRATIFIED = "voxel_stratified"
    FPS = "fps"
    HALTON_LLOYD = "halton_lloyd"

@qd.data_oriented
class SurfaceReconstructor:
    def __init__(self, env, grid_res=128, backend='hybrid'):
        self.env = env
        self.reconstructed_mesh = trimesh.Trimesh()
        self.recon_enabled = True
        self.recon_frame_interval = 3  # Reconstruct every 3 frames
        self.frame_counter = 0
        self.mesh_version = 0
        
        # Compatibility / Configuration
        self.sampling_method = SamplingMethod.VOXEL_STRATIFIED
        self.recon_particle_fraction = 1.0
        self.skinning_enabled = False # Dummy flag
        self.bind_indices = None
        self.bind_weights = None
        
        # Backend Configuration
        self.backend = backend # 'hybrid' or 'splashsurf'
        
        # Caching for Visualization
        self._cached_particles = None
        
        # Grid Configuration
        self.grid_res = grid_res
        self.density = qd.field(dtype=float, shape=(self.grid_res, self.grid_res, self.grid_res))
        self.prev_density = qd.field(dtype=float, shape=(self.grid_res, self.grid_res, self.grid_res))
        self._blur_temp = qd.field(dtype=float, shape=(self.grid_res, self.grid_res, self.grid_res))
        self.temporal_alpha = 0.35 # Blend factor (0.0 = history only, 1.0 = no smoothing)
        self.density_initialized = False
        
        # Constants
        self.particle_radius = 0.005 
        try:
            self.particle_radius = self.env.scene.sim.mpm_solver.particle_size / 2.0
            gs.logger.info(f"Reconstruction: Inferred particle radius: {self.particle_radius}")
        except Exception:
            pass
        self.influence_radius = self.particle_radius * 3.0
        
        # Grid snapping: fixed dx from physical scale, snapped origin
        self._grid_coverage_ratio = 3.0
        self._fixed_dx = None
        self._prev_grid_origin = None
        
        # Density-space smoothing before marching cubes (sigma in grid cells)
        self.density_blur_sigma = 1.25
        
        # Pre-compute 1D Gaussian kernel weights for separable blur  
        self._blur_radius = int(math.ceil(2.0 * self.density_blur_sigma))  # ~3 cells each side
        ksize = 2 * self._blur_radius + 1
        weights = np.array([math.exp(-0.5 * ((i - self._blur_radius) / self.density_blur_sigma) ** 2) 
                           for i in range(ksize)], dtype=np.float32)
        weights /= weights.sum()
        self._blur_weights = qd.field(dtype=float, shape=(ksize,))
        self._blur_weights.from_numpy(weights)
        
        # Cached PyVista grid (reused across frames)
        self._pv_grid = None
        
        # Cached vertices for Post-MC temporal smoothing (Vertex Correspondence Blending)
        self._prev_verts = None
        self.vertex_blend_factor = 0.0  # Blend factor towards previous frame (0.0 = off, was 0.15)

    def get_state(self):
        return {
            'mesh': self.reconstructed_mesh.copy(),
            'recon_enabled': self.recon_enabled,
            'frame_counter': self.frame_counter,
            '_cached_particles': self._cached_particles.copy() if self._cached_particles is not None else None
        }

    def set_state(self, state):
        if state is None: return
        self.reconstructed_mesh = state['mesh'].copy()
        self.recon_enabled = state.get('recon_enabled', True)
        self.frame_counter = state.get('frame_counter', 0)
        
        # Restore the exact particles that were used to generate this mesh
        # This is critical for the visual KD-Tree temperature mapping to remain aligned
        if '_cached_particles' in state and state['_cached_particles'] is not None:
            self._cached_particles = state['_cached_particles'].copy()
        else:
            self._cached_particles = None
            
        self.mesh_version += 1
        self.density_initialized = False
        self._prev_grid_origin = None
        self._prev_verts = None
        if hasattr(self, 'reconstructed_vertices_tensor'):
            self.reconstructed_vertices_tensor = None
        self.prev_density.fill(0)

    def reset(self):
        self.reconstructed_mesh = trimesh.Trimesh()
        self.frame_counter = 0
        self.mesh_version += 1
        self._cached_particles = None
        self.skinning_enabled = False
        self.density_initialized = False
        self._fixed_dx = None
        self._prev_grid_origin = None
        self._prev_verts = None
        if hasattr(self, 'reconstructed_vertices_tensor'):
            self.reconstructed_vertices_tensor = None
        self.prev_density.fill(0)

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
            
            p_numpy = parts.cpu().numpy()
            self._cached_particles = p_numpy
            return p_numpy
        except Exception:
            return None

    @qd.kernel
    def _compute_density_kernel(
        self, 
        particles_pos: qd.types.ndarray(),  # type: ignore
        particles_F: qd.types.ndarray(),  # type: ignore
        active_mask: qd.types.ndarray(),  # type: ignore
        n_particles: int,
        lower_bound: qd.types.vector(3, float),  # type: ignore
        dx: float,
        influence_radius: float
    ):
        for I in qd.grouped(self.density):
            self.density[I] = 0.0
            
        for i in range(n_particles):
            if active_mask[i]:
                pos = qd.Vector([particles_pos[i, 0], particles_pos[i, 1], particles_pos[i, 2]])
                
                # Retrieve the deformation gradient tensor F
                F = qd.Matrix([
                    [particles_F[i, 0, 0], particles_F[i, 0, 1], particles_F[i, 0, 2]],
                    [particles_F[i, 1, 0], particles_F[i, 1, 1], particles_F[i, 1, 2]],
                    [particles_F[i, 2, 0], particles_F[i, 2, 1], particles_F[i, 2, 2]]
                ])
                
                # Phase 6A: Volume-Normalize the F Tensor
                # Prevent anisotropic bounding boxes from changing total kernel volume
                detF = F.determinant()
                vol_scale = (qd.abs(detF) + 1e-6) ** (1.0 / 3.0)
                F_norm = F / vol_scale
                
                # Compute Left Cauchy-Green deformation tensor B = F * F^T
                # We add a small epsilon to the diagonal to prevent singular matrix inversion
                # if the particle is completely flattened or inverted.
                B = F_norm @ F_norm.transpose()
                B_reg = B + qd.Matrix.identity(float, 3) * 1e-4
                B_inv = B_reg.inverse()

                grid_pos = (pos - lower_bound) / dx
                # The bounding box of the ellipsoid might be larger than influence_radius
                # in extreme stretch, but for splatting we usually clamp the search radius
                # to the original influence_radius * a small buffer (e.g. 1.5x) to catch stretches.
                # However, for performance we stick to the original conservative radius cell span.
                search_radius = influence_radius * 1.5
                rad_cells = search_radius / dx
                
                base_idx = qd.cast(qd.floor(grid_pos - rad_cells), qd.int32)
                end_idx = qd.cast(qd.ceil(grid_pos + rad_cells), qd.int32)
                
                for ix in range(base_idx[0], end_idx[0] + 1):
                    for iy in range(base_idx[1], end_idx[1] + 1):
                        for iz in range(base_idx[2], end_idx[2] + 1):
                            if (0 <= ix < self.grid_res and 
                                0 <= iy < self.grid_res and 
                                0 <= iz < self.grid_res):
                                cell_center = lower_bound + qd.Vector([ix, iy, iz]) * dx
                                diff = cell_center - pos
                                
                                # Anisotropic warped distance squared: diff^T * B_inv * diff
                                dist_sq = diff.dot(B_inv @ diff)
                                
                                # Use the original isotropic cutoff mathematically mapped to the ellipsoid
                                if dist_sq < influence_radius**2:
                                    r = qd.sqrt(dist_sq) / influence_radius
                                    val = (1.0 - r)**4 * (4.0 * r + 1.0)
                                    self.density[ix, iy, iz] += val

    @qd.kernel
    def _blend_density_temporal(self, alpha: float):
        """Exponential moving average over the density field: D = alpha*D_new + (1-alpha)*D_prev."""
        for I in qd.grouped(self.density):
            blended_val = alpha * self.density[I] + (1.0 - alpha) * self.prev_density[I]
            self.density[I] = blended_val
            self.prev_density[I] = blended_val

    @qd.kernel
    def _blur_pass_x(self, radius: int):
        """Separable Gaussian blur along X axis: density -> _blur_temp."""
        for i, j, k in self._blur_temp:
            acc = 0.0
            for di in range(-radius, radius + 1):
                ni = i + di
                if 0 <= ni < self.grid_res:
                    acc += self.density[ni, j, k] * self._blur_weights[di + radius]
                else:
                    acc += self.density[i, j, k] * self._blur_weights[di + radius]
            self._blur_temp[i, j, k] = acc

    @qd.kernel
    def _blur_pass_y(self, radius: int):
        """Separable Gaussian blur along Y axis: _blur_temp -> density."""
        for i, j, k in self.density:
            acc = 0.0
            for di in range(-radius, radius + 1):
                nj = j + di
                if 0 <= nj < self.grid_res:
                    acc += self._blur_temp[i, nj, k] * self._blur_weights[di + radius]
                else:
                    acc += self._blur_temp[i, j, k] * self._blur_weights[di + radius]
            self.density[i, j, k] = acc

    @qd.kernel
    def _blur_pass_z(self, radius: int):
        """Separable Gaussian blur along Z axis: density -> _blur_temp, then copy back."""
        for i, j, k in self._blur_temp:
            acc = 0.0
            for di in range(-radius, radius + 1):
                nk = k + di
                if 0 <= nk < self.grid_res:
                    acc += self.density[i, j, nk] * self._blur_weights[di + radius]
                else:
                    acc += self.density[i, j, k] * self._blur_weights[di + radius]
            self._blur_temp[i, j, k] = acc

    def _blur_density_gpu(self):
        """Run 3-pass separable Gaussian blur entirely on GPU."""
        r = self._blur_radius
        self._blur_pass_x(r)       # density -> _blur_temp
        self._blur_pass_y(r)       # _blur_temp -> density  
        self._blur_pass_z(r)       # density -> _blur_temp
        # Copy result back: _blur_temp -> density
        self._copy_blur_to_density()

    @qd.kernel
    def _copy_blur_to_density(self):
        for I in qd.grouped(self.density):
            self.density[I] = self._blur_temp[I]

    def update(self, should_reconstruct: bool, is_deforming: bool = False):
        if not self.recon_enabled:
            return
        self.frame_counter += 1
        if not should_reconstruct and (self.frame_counter % self.recon_frame_interval != 0):
            return
        self.create_reconstructed_mesh(is_deforming=is_deforming)
        self.mesh_version += 1

    def create_reconstructed_mesh(self, is_deforming: bool = False):
        # Legacy SplashSurf Path
        if self.backend == 'splashsurf':
            self._create_splashsurf_mesh()
            return

        # Hybrid Path
        import contextlib
        profiler = getattr(self.env.scene.profiling_options, 'profiler', None)
        def profile_block(name):
            if profiler:
                return profiler.time(name)
            return contextlib.nullcontext()
        try:
          with profile_block("hybrid_get_particles"):
            try:
                mpm_entity = self.env.mpm_entity
                particles_pos = mpm_entity.get_particles_pos(envs_idx=0).squeeze(0)
                particles_F = mpm_entity.get_particles_F(envs_idx=0).squeeze(0)
                particles_active = mpm_entity.get_particles_active(envs_idx=0).squeeze(0)
            except Exception:
                mpm_entity = self.env.mpm_entity
                particles_pos = mpm_entity.get_particles_pos()
                particles_F = mpm_entity.get_particles_F()
                particles_active = mpm_entity.get_particles_active()

            n_particles = particles_pos.shape[0]
            if n_particles == 0:
                self.reconstructed_mesh = trimesh.Trimesh()
                return

            active_particles = particles_pos[particles_active]
            n_active_particles = active_particles.shape[0]
            if n_active_particles == 0:
                gs.logger.warning(f"Reconstruction: 0 active of {n_particles} total particles")
                self._cached_particles = np.empty((0, 3), dtype=np.float32)
                self.reconstructed_mesh = trimesh.Trimesh()
                return

            # Update cache for visualization
            self._cached_particles = active_particles.cpu().numpy()

            # Compute fixed dx from physical scale (once, then stable)
            if self._fixed_dx is None:
                self._fixed_dx = self.influence_radius / self._grid_coverage_ratio
                gs.logger.info(
                    f"Reconstruction: Fixed dx={self._fixed_dx:.6f} "
                    f"(influence_r/dx={self.influence_radius / self._fixed_dx:.2f})"
                )
            dx = self._fixed_dx

            # Compute raw bounds with padding
            raw_min = active_particles.min(dim=0).values.cpu().numpy()
            raw_max = active_particles.max(dim=0).values.cpu().numpy()
            padding = self.influence_radius * 2.0
            raw_min -= padding
            raw_max += padding

            # Snap bounds to multiples of dx for temporal coherence
            min_bound = np.floor(raw_min / dx) * dx
            max_bound = np.ceil(raw_max / dx) * dx

            # Verify grid capacity — fall back to dynamic dx if particles exceed grid
            extent = max_bound - min_bound
            max_extent = float(np.max(extent))
            if max_extent < 1e-4:
                max_extent = 1.0
            required_cells = int(np.ceil(max_extent / dx)) + 1
            if required_cells > self.grid_res:
                dx = max_extent / (self.grid_res - 1)
                min_bound = np.floor(raw_min / dx) * dx
                # Suppress warning:
                # gs.logger.warning(
                #     f"Reconstruction: Particles exceed fixed grid ({required_cells} > {self.grid_res}), "
                #     f"adjusting dx={dx:.6f} (influence_r/dx={self.influence_radius / dx:.2f})"
                # )

            # Detect grid origin shift — reset temporal blending when grid jumps
            grid_origin = min_bound.copy()
            if self._prev_grid_origin is not None:
                if not np.allclose(grid_origin, self._prev_grid_origin, atol=dx * 0.01):
                    self.density_initialized = False
                    gs.logger.debug("Reconstruction: Grid origin shifted, resetting temporal blend")
            self._prev_grid_origin = grid_origin

            lower_bound_qd = qd.Vector([min_bound[0], min_bound[1], min_bound[2]])

          with profile_block("hybrid_density_kernel"):
            self._compute_density_kernel(
                particles_pos,
                particles_F,
                particles_active,
                n_particles,
                lower_bound_qd,
                float(dx),
                float(self.influence_radius)
            )

            # Adaptive Temporal Blending (state-based, no extra GPU transfer)
            if not self.density_initialized:
                alpha = 1.0
            elif is_deforming:
                alpha = 0.8  # Fast response during active deformation
            else:
                alpha = 0.2  # Heavy smoothing at rest

            self._blend_density_temporal(alpha)
            self.density_initialized = True

            # GPU Gaussian blur (separable, 3 passes on Quadrants fields)
            if self.density_blur_sigma > 0:
                self._blur_density_gpu()

          with profile_block("hybrid_density_transfer"):
            density_cpu = self.density.to_numpy()

          with profile_block("hybrid_marching_cubes"):
            # Percentile-Based Dynamic Thresholding
            valid_densities = density_cpu[density_cpu > 1e-4]
            if len(valid_densities) == 0:
                gs.logger.warning("Reconstruction: No valid densities found! Mesh will be empty.")
                return

            thresh = float(np.percentile(valid_densities, 95)) * 0.3
            gs.logger.debug(f"Reconstruction: max_density={density_cpu.max():.4f}, threshold={thresh:.4f}")

            # Create PyVista grid (near zero-copy through VTK data adapters)
            grid = pv.ImageData()
            grid.dimensions = np.array(density_cpu.shape)
            grid.spacing = (dx, dx, dx)
            grid.origin = min_bound
            grid.point_data["density"] = density_cpu.flatten(order="F")

            contour = grid.contour(isosurfaces=[thresh], scalars="density", method='flying_edges', compute_normals=False)

            if contour.n_points == 0:
                self.reconstructed_mesh = trimesh.Trimesh()
                self._prev_verts = None
                return

          with profile_block("hybrid_post_process"):
            verts = np.array(contour.points)

            # Vertex Correspondence Blending (Post-MC Temporal Smoothing)
            if self.vertex_blend_factor > 0 and self._prev_verts is not None and len(self._prev_verts) > 0:
                try:
                    tree = cKDTree(self._prev_verts)
                    dists, indices = tree.query(verts)
                    valid_mask = dists < (dx * 2.0)
                    if valid_mask.any():
                        blend = self.vertex_blend_factor
                        verts[valid_mask] = (1.0 - blend) * verts[valid_mask] + blend * self._prev_verts[indices[valid_mask]]
                except Exception as e:
                    gs.logger.debug(f"Reconstruction: Vertex blending failed: {e}")

            self._prev_verts = verts.copy()

            # Reverse winding order so normals point OUTWARD for Unity
            faces = np.array(contour.faces).reshape(-1, 4)[:, 1:4]
            faces = faces[:, ::-1]

            mesh = trimesh.Trimesh(
                vertices=verts,
                faces=faces,
                vertex_normals=None,
                process=False
            )

            # Taubin smoothing (1 iteration for light cleanup)
            if len(mesh.vertices) > 0:
                try:
                    trimesh.smoothing.filter_taubin(mesh, iterations=1)
                except Exception:
                    try:
                        trimesh.smoothing.filter_laplacian(mesh, lamb=0.5, iterations=1)
                    except Exception as e:
                        gs.logger.debug(f"Smoothing failed: {e}")

                if np.isnan(mesh.vertices).any():
                    gs.logger.warning("Smoothing produced NaN vertices, using unsmoothed mesh")
                    mesh = trimesh.Trimesh(
                        vertices=verts, faces=faces,
                        process=False
                    )

            gs.logger.debug(f"Reconstruction: {len(mesh.vertices)} verts, {len(mesh.faces)} faces")
            self.reconstructed_mesh = mesh

        except Exception as e:
            import traceback
            gs.logger.warning(f"Reconstruction failed: {e}")
            traceback.print_exc()

    def _create_splashsurf_mesh(self):
        """Direct in-process SplashSurf reconstruction.

        Bypasses the Genesis pu.particles_to_mesh() wrapper which forks a subprocess
        per frame to work around a pysplashsurf memory leak. Instead, we call
        pysplashsurf directly in-process and use malloc_trim() to reclaim memory.
        This eliminates ~115ms/frame of multiprocessing overhead.
        """
        try:
            import contextlib
            import ctypes
            import ctypes.util
            import pysplashsurf

            profiler = getattr(self.env.scene.profiling_options, 'profiler', None)

            def profile_block(name):
                if profiler:
                    return profiler.time(name)
                return contextlib.nullcontext()

            with profile_block("splashsurf_get_particles"):
                particles = self._get_active_particles(use_cache=False)

            if particles is None or len(particles) == 0:
                return

            radius = self.particle_radius * 1.5

            with profile_block("splashsurf_core_meshing"):
                mesh_with_data, _ = pysplashsurf.reconstruction_pipeline(
                    particles,
                    particle_radius=radius,
                    smoothing_length=2.0,
                    cube_size=0.8,
                    iso_surface_threshold=0.6,
                    mesh_smoothing_weights=True,
                    mesh_smoothing_iters=25,
                    normals_smoothing_iters=10,
                    mesh_cleanup=True,
                    compute_normals=True,
                    multi_threading=True,
                )
                vertices = mesh_with_data.mesh.vertices
                triangles = mesh_with_data.mesh.triangles
                normals = mesh_with_data.point_attributes["normals"]
                del mesh_with_data  # Release C++ wrapper immediately

            with profile_block("splashsurf_build_trimesh"):
                mesh = trimesh.Trimesh(
                    vertices=vertices, faces=triangles,
                    face_normals=normals, process=False
                )
                # Reverse winding order so normals point OUTWARD for Unity
                mesh.faces = mesh.faces[:, ::-1]
                self.reconstructed_mesh = mesh

            with profile_block("splashsurf_mem_cleanup"):
                # Reclaim leaked pysplashsurf memory without subprocess overhead
                try:
                    libc = ctypes.CDLL(ctypes.util.find_library("c"))
                    libc.malloc_trim(0)
                except Exception:
                    pass

        except Exception as e:
            import traceback
            gs.logger.warning(f"SplashSurf failed: {e}")
            traceback.print_exc()