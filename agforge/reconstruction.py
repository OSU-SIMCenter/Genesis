import time
import torch
import numpy as np
import trimesh
import gstaichi as ti
import genesis as gs
import genesis.utils.particle as pu
from genesis.utils.misc import ti_to_numpy
import contextlib
from enum import Enum
from typing import Optional


class SamplingMethod(Enum):
    RANDOM = "random"
    VOXEL_STRATIFIED = "voxel_stratified"  # RECOMMENDED for uniform volume coverage
    FPS = "fps"  # Farthest point sampling (biased to surface)
    HALTON_LLOYD = "halton_lloyd"  # Original method (bad for non-cubic shapes)


class SurfaceReconstructor:
    def __init__(self, env):
        self.env = env
        self.reconstructed_mesh = trimesh.Trimesh()
        self.recon_enabled = True
        self.recon_frame_interval = 2
        self.recon_particle_fraction = 1.0 # 0.5
        self.frame_counter = 0
        
        # Sampling configuration
        self.sampling_method = SamplingMethod.VOXEL_STRATIFIED
        
        # Cached indices for deterministic sampling
        self.main_particle_indices = None
        self.last_total_particles = 0
        
        # Skinning data
        self.skinning_enabled = False
        self.bind_indices = None
        self.bind_weights = None
        self.bind_offsets = None
        
        # Device setup with MPS support
        if hasattr(env, 'device'):
            self.device = env.device
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        
        # Smoothing matrix state
        self.smoothing_matrix = None
        self._use_dense_smoothing = False
        
        # Cache particle radius for skinning sigma calculation
        self._cached_particle_radius = None

        # Particle Cache (Optimization)
        self._cached_particles = None
        self._cached_frame = -1
        
        # FIX #1: Global frame counter that always increments
        self._global_frame = 0
        
        # Rebind check interval (was 30)
        self._rebind_check_interval = 10
        
        # Level 2: Movement limiting - track previous vertex positions
        self._prev_verts = None

    def get_state(self):
        """Returns a snapshot of the current mesh and skinning state (for checkpoints)."""
        state = {
            'mesh': self.reconstructed_mesh.copy(),
            'skinning_enabled': self.skinning_enabled,
            'recon_enabled': self.recon_enabled,
            'frame_counter': self.frame_counter,
            'global_frame': self._global_frame,
            'recon_particle_fraction': self.recon_particle_fraction,
            'main_particle_indices': self.main_particle_indices.copy() if self.main_particle_indices is not None else None,
            'last_total_particles': self.last_total_particles,
        }
        
        # Save skinning tensors if active
        if self.skinning_enabled:
            state.update({
                'bind_indices': self.bind_indices.clone() if self.bind_indices is not None else None,
                'bind_weights': self.bind_weights.clone() if self.bind_weights is not None else None,
                'bind_offsets': self.bind_offsets.clone() if self.bind_offsets is not None else None,
                'smoothing_matrix': self.smoothing_matrix.clone() if self.smoothing_matrix is not None else None,
                'use_dense_smoothing': self._use_dense_smoothing,
                'cached_particle_radius': self._cached_particle_radius
            })
            
        return state

    def set_state(self, state):
        """Restores the reconstructor state from a checkpoint."""
        if state is None:
            return

        # CRITICAL: Deep copy the mesh to prevent mutations from corrupting the checkpoint
        self.reconstructed_mesh = state['mesh'].copy()
        self.skinning_enabled = state['skinning_enabled']
        self.recon_enabled = state['recon_enabled']
        self.frame_counter = state['frame_counter']
        # Do NOT restore global_frame, as that tracks time passage for the server?
        # Actually, if we undo simulation time, we SHOULD undo global_frame to match the sim state.
        self._global_frame = state['global_frame']
        
        self.recon_particle_fraction = state.get('recon_particle_fraction', 1.0)
        # Deep copy numpy array if present
        self.main_particle_indices = state['main_particle_indices'].copy() if state['main_particle_indices'] is not None else None
        self.last_total_particles = state['last_total_particles']
        
        if self.skinning_enabled:
            # CRITICAL: Clone tensors to prevent mutations from corrupting the checkpoint
            self.bind_indices = state['bind_indices'].clone() if state['bind_indices'] is not None else None
            self.bind_weights = state['bind_weights'].clone() if state['bind_weights'] is not None else None
            self.bind_offsets = state['bind_offsets'].clone() if state['bind_offsets'] is not None else None
            self.smoothing_matrix = state['smoothing_matrix'].clone() if state['smoothing_matrix'] is not None else None
            self._use_dense_smoothing = state['use_dense_smoothing']
            self._cached_particle_radius = state['cached_particle_radius']
        else:
            self._invalidate_skinning()
            
        # Clear particle cache to ensure next fetch gets valid data for this restored time
        self._cached_particles = None
        self._cached_frame = -1


    def reset(self):
        """Resets the reconstructor state and sampling cache."""
        self.reconstructed_mesh = trimesh.Trimesh()
        self.main_particle_indices = None
        self.last_total_particles = 0
        self._invalidate_skinning()
        # Reset cache
        self._cached_particles = None
        self._cached_frame = -1
        self._global_frame = 0

    def _invalidate_skinning(self):
        """Properly invalidate skinning state."""
        self.skinning_enabled = False
        self.bind_indices = None
        self.bind_weights = None
        self.bind_offsets = None
        self.smoothing_matrix = None
        self._use_dense_smoothing = False

    def get_mesh_data(self):
        """Returns the current mesh and its vertices/faces."""
        return self.reconstructed_mesh

    def update(self, should_reconstruct: bool):
        """
        Updates the reconstructed mesh if conditions are met.
        args:
            should_reconstruct: External condition (e.g. is striking)
        """
        profiler = self.env.scene.profiling_options.profiler
        # FIX #1: Always increment global frame
        self._global_frame += 1
        
        if not self.recon_enabled:
            return

        # If skinning active, update positions
        if self.skinning_enabled:
            # FIX #10: Respect frame interval for skinning too (Performance)
            if self._global_frame % self.recon_frame_interval != 0:
                return

            # FIX #7: Check if rebind needed periodically
            if self._global_frame % self._rebind_check_interval == 0:
                should_rebind = False
                with profiler.time("recon_check_rebind"):
                    should_rebind = self._should_rebind()

                if should_rebind:
                    gs.logger.info("Large drift detected, rebinding mesh to particles...")
                    with profiler.time("recon_rebind"):
                        with profiler.time("recon_mesh"):
                            self.create_reconstructed_mesh()
                        with profiler.time("recon_update_skinning"):
                            self.init_skinning()
                    return
            
            with profiler.time("recon_update_skinning"):
                self.update_skinning()
            return

        # Fallback to full reconstruction if not skinned or explicitly requested
        if not should_reconstruct:
            return

        self.frame_counter += 1
        if self.frame_counter % self.recon_frame_interval != 0:
            return

        with profiler.time("recon_mesh"):
            self.create_reconstructed_mesh()

    def _should_rebind(self) -> bool:
        """
        FIX #7: Detect when skinning drift is too large and rebind is needed.
        Returns True if mesh vertices have drifted too far from their bound particles.
        """
        if not self.skinning_enabled:
            return False
        if self._cached_particle_radius is None:
            return False
        if self.reconstructed_mesh is None or len(self.reconstructed_mesh.vertices) == 0:
            return True
        
        try:
            particles = self._get_active_particles(use_cache=False)
            if particles is None or len(particles) == 0:
                return True
            
            if isinstance(particles, np.ndarray):
                parts_t = torch.from_numpy(particles).float().to(self.device)
            else:
                parts_t = particles.to(self.device)
            
            verts_t = torch.tensor(
                self.reconstructed_mesh.vertices, 
                device=self.device, 
                dtype=torch.float32
            )
            
            # Sample subset for speed
            n_check = min(100, len(verts_t))
            idx = torch.linspace(0, len(verts_t) - 1, n_check, device=self.device).long()
            
            sample_verts = verts_t[idx]
            sample_bind = self.bind_indices[idx, 0]  # Primary bound particle
            
            # Check bounds
            if sample_bind.max() >= len(parts_t):
                return True
            
            bound_pos = parts_t[sample_bind]
            
            # Compute mean drift
            drift = torch.norm(sample_verts - bound_pos, dim=1).mean().item()
            threshold = self._cached_particle_radius * 8  # Tighter threshold (was 15)
            
            return drift > threshold
            
        except Exception as e:
            gs.logger.warning(f"Rebind check failed: {e}")
            return True

    def init_skinning(self):
        """
        Binds the current reconstructed mesh to current active particles.
        Must be called ONCE after create_reconstructed_mesh produces a valid mesh.
        """
        if self.reconstructed_mesh is None or len(self.reconstructed_mesh.vertices) == 0:
            gs.logger.warning("Cannot init skinning: No mesh generated yet.")
            return

        particles = self._get_active_particles(use_cache=False)
        if particles is None or len(particles) == 0:
            gs.logger.warning("Cannot init skinning: No active particles.")
            return

        gs.logger.info(f"Visual Binding: {len(self.reconstructed_mesh.vertices)} vertices -> {len(particles)} particles")

        # Convert to Torch Tensors on Device
        verts_tensor = torch.tensor(
            self.reconstructed_mesh.vertices, 
            dtype=torch.float32, 
            device=self.device
        )
        
        if isinstance(particles, np.ndarray):
            parts_tensor = torch.from_numpy(particles).float().to(self.device)
        else:
            parts_tensor = torch.tensor(particles, dtype=torch.float32, device=self.device)

        # k-NN parameters
        k = 4
        
        # Auto-compute sigma relative to particle spacing
        if len(parts_tensor) > 1:
            sample_size = min(1000, len(parts_tensor))
            # FIX #2: Specify device on randperm
            sample_idx = torch.randperm(len(parts_tensor), device=self.device)[:sample_size]
            sample = parts_tensor[sample_idx]
            
            sample_dists = torch.cdist(sample, sample)
            sample_dists.fill_diagonal_(float('inf'))
            nn_dists = sample_dists.min(dim=1).values
            mean_spacing = nn_dists.mean().item()
            
            sigma = mean_spacing * 1.0  # Level 1 tuning: tighter binding (was 1.5)
            gs.logger.info(f"Auto-computed sigma: {sigma:.6f} (mean spacing: {mean_spacing:.6f})")
        else:
            sigma = 0.01

        try:
            V = len(verts_tensor)
            self.bind_indices = torch.zeros((V, k), dtype=torch.long, device=self.device)
            self.bind_weights = torch.zeros((V, k), dtype=torch.float32, device=self.device)
            self.bind_offsets = torch.zeros((V, k, 3), dtype=torch.float32, device=self.device)

            chunk_size = 1000
            for i in range(0, V, chunk_size):
                end_i = min(i + chunk_size, V)
                v_chunk = verts_tensor[i:end_i]
                
                dists = torch.cdist(v_chunk, parts_tensor)
                vals, idxs = torch.topk(dists, k=k, dim=1, largest=False)
                
                weights = torch.exp(-vals.pow(2) / (2 * sigma**2))
                weights_sum = weights.sum(dim=1, keepdim=True)
                weights_sum = torch.clamp(weights_sum, min=1e-6)
                weights = weights / weights_sum
                
                self.bind_indices[i:end_i] = idxs
                self.bind_weights[i:end_i] = weights
                
                neighbor_pos = parts_tensor[idxs]
                offsets = v_chunk.unsqueeze(1) - neighbor_pos
                self.bind_offsets[i:end_i] = offsets

            # Precompute smoothing matrix
            self._compute_laplacian_matrix(lamb=0.5)
            
            self.skinning_enabled = True
            gs.logger.info("Skinning initialized successfully.")

        except Exception as e:
            gs.logger.error(f"Failed to init skinning: {e}")
            self._invalidate_skinning()

    def _compute_laplacian_matrix(self, lamb=0.5):
        """Precomputes sparse smoothing matrix with MPS fallback."""
        if self.reconstructed_mesh is None or len(self.reconstructed_mesh.vertices) == 0:
            return

        try:
            faces = torch.from_numpy(self.reconstructed_mesh.faces).long().to(self.device)
            V = len(self.reconstructed_mesh.vertices)

            edges = torch.cat([
                faces[:, [0, 1]],
                faces[:, [1, 2]],
                faces[:, [2, 0]]
            ], dim=0)
            
            edges, _ = torch.sort(edges, dim=1)
            edges = torch.unique(edges, dim=0)

            row = torch.cat([edges[:, 0], edges[:, 1]])
            col = torch.cat([edges[:, 1], edges[:, 0]])
            
            ones = torch.ones(len(row), device=self.device)
            degree = torch.zeros(V, device=self.device)
            degree.scatter_add_(0, row, ones)
            degree = torch.clamp(degree, min=1.0)
            
            norm_vals = (1.0 / degree[row]) * lamb
            indices = torch.stack([row, col], dim=0)
            S_off = torch.sparse_coo_tensor(indices, norm_vals, (V, V))
            
            diag_idx = torch.arange(V, device=self.device).unsqueeze(0).repeat(2, 1)
            diag_val = torch.full((V,), 1.0 - lamb, device=self.device)
            S_diag = torch.sparse_coo_tensor(diag_idx, diag_val, (V, V))
            
            self.smoothing_matrix = (S_diag + S_off).coalesce()
            self._use_dense_smoothing = False
            
            # FIX #6: MPS fallback - test if sparse mm works
            if self.device.type == "mps":
                try:
                    test_input = torch.zeros(V, 3, device=self.device)
                    _ = torch.sparse.mm(self.smoothing_matrix, test_input)
                except Exception:
                    gs.logger.warning("MPS sparse mm not supported, converting to dense matrix")
                    self.smoothing_matrix = self.smoothing_matrix.to_dense()
                    self._use_dense_smoothing = True
            
        except Exception as e:
            gs.logger.warning(f"Failed to build smoothing matrix: {e}")
            self.smoothing_matrix = None

            
        except Exception as e:
            gs.logger.warning(f"Failed to build smoothing matrix: {e}")
            self.smoothing_matrix = None

    def update_skinning(self):
        """Updates mesh vertices based on current particle positions."""
        if not self.skinning_enabled:
            return

        particles = self._get_active_particles()
        if particles is None or len(particles) == 0:
            return
            
        if isinstance(particles, np.ndarray):
            parts_current = torch.from_numpy(particles).float().to(self.device)
        else:
            parts_current = torch.tensor(particles, dtype=torch.float32, device=self.device)

        # Validate particle count matches binding
        expected_count = self.bind_indices.max().item() + 1
        if len(parts_current) < expected_count:
            gs.logger.error(
                f"Particle count mismatch: have {len(parts_current)}, need {expected_count}. "
                "Disabling skinning."
            )
            self._invalidate_skinning()
            return

        try:
            # 1. Gather neighbor positions
            neighbors = parts_current[self.bind_indices]
            
            # 2. Add offsets
            estimated_pos = neighbors + self.bind_offsets
            
            # 3. Weighted Average (Skinned Position)
            new_verts = (estimated_pos * self.bind_weights.unsqueeze(-1)).sum(dim=1)
            
            # 4. Tangential Laplacian Smoothing
            # Goal: Improve triangle quality without shrinking volume.
            # Strategy: Project smoothing force onto tangent plane defined by particle surface normal.
            if self.smoothing_matrix is not None:
                # Compute center of bound particles (Stable anchor for normal estimation)
                # This represents the "core" volume that the vertex is skinning off of.
                particle_centers = (neighbors * self.bind_weights.unsqueeze(-1)).sum(dim=1)
                
                # Approximate Surface Normal: Direction from particle center to vertex
                # This is a robust estimate for convex/semi-convex fluid surfaces.
                # normal = normalize(new_verts - particle_centers)
                # Note: normalize() needs small epsilon to avoid NaN on locally 0-thickness events
                surf_normals = new_verts - particle_centers
                norm_len = torch.norm(surf_normals, dim=1, keepdim=True).clamp(min=1e-6)
                surf_normals = surf_normals / norm_len
                
                for _ in range(3): # 3 Iterations
                    # Standard Laplacian Centroid
                    if self._use_dense_smoothing:
                        centroids = self.smoothing_matrix @ new_verts
                    else:
                        centroids = torch.sparse.mm(self.smoothing_matrix, new_verts)
                    
                    # Smoothing Force
                    smooth_vec = centroids - new_verts
                    
                    # Project out the Normal component (Preserve Volume)
                    # dot(smooth_vec, normal) * normal
                    normal_comp = (smooth_vec * surf_normals).sum(dim=1, keepdim=True) * surf_normals
                    tangential_force = smooth_vec - normal_comp
                    
                    # Apply only Tangential component
                    new_verts = new_verts + tangential_force
            
            # 5. Update mesh
            self.reconstructed_mesh.vertices = new_verts.cpu().numpy()
            
        except Exception as e:
            gs.logger.error(f"Skinning update failed: {e}")
            self._invalidate_skinning()

    def _get_active_particles(self, use_cache: bool = True, apply_subsampling: bool = True):
        """Helper to fetch particles with consistent logic and caching."""
        # FIX #1: Use global frame counter
        current_frame = self._global_frame
        
        # Cache check
        if use_cache and self._cached_frame == current_frame and self._cached_particles is not None:
             # If cached particles are already subsampled, we return them.
             # If we need RAW particles but cache has SUBSAMPLED, we must re-fetch.
             # Current design: cache stores SUBSAMPLED particles (ready for skinning/rendering).
             # So if apply_subsampling=False, we cannot use the Default Cache safely if it's already subsampled.
             if apply_subsampling:
                return self._cached_particles
             else:
                pass # Force fetch for raw particles

        try:
            # OPTIMIZATION: Use mpm_entity.get_particles_pos() instead of particles_render
            # This reads directly from simulation state (no visualizer dependency) and
            # returns a torch tensor on GPU, avoiding CPU round-trip.
            mpm_entity = self.env.mpm_entity
            
            # get_particles_pos returns torch tensor on GPU: shape [n_particles, 3]
            particles_gpu = mpm_entity.get_particles_pos(envs_idx=0).squeeze(0)
            
            # Get active mask (still need to check which particles are active)
            particles_active = mpm_entity.get_particles_active(envs_idx=0).squeeze(0)
            
            # Filter to active particles only (stay on GPU)
            particles_gpu = particles_gpu[particles_active]
            
            # Apply subsampling if indices are set (stay on GPU)
            if apply_subsampling and self.main_particle_indices is not None and len(self.main_particle_indices) > 0:
                # FIX #5: Properly handle particle count change
                if np.max(self.main_particle_indices) < len(particles_gpu):
                    # Convert indices to torch tensor for GPU indexing
                    if not isinstance(self.main_particle_indices, torch.Tensor):
                        indices_gpu = torch.from_numpy(self.main_particle_indices).to(particles_gpu.device)
                    else:
                        indices_gpu = self.main_particle_indices.to(particles_gpu.device)
                    particles_gpu = particles_gpu[indices_gpu]
                else:
                    gs.logger.warning(
                        "Particle count changed! Invalidating cached indices and skinning."
                    )
                    self.main_particle_indices = None
                    self.last_total_particles = 0
                    self._invalidate_skinning()
                    # Return None to force reinitialization
                    return None
            
            # Convert to numpy for downstream compatibility (skinning still needs numpy for some ops)
            particles = particles_gpu.cpu().numpy()
            
            # Update cache ONLY if we are doing the standard subsampled fetch
            if apply_subsampling:
                self._cached_particles = particles
                self._cached_frame = current_frame
            
            return particles

        except Exception as e:
            gs.logger.error(f"Failed to get particles: {e}")
            return None

    def get_active_particle_cache(self):
        """Returns the most recently cached active particles (for visualization)."""
        return self._cached_particles


    def create_reconstructed_mesh(self):
        """Reconstruct surface mesh from active particles using splashsurf."""
        # FIX: Define solver for radius access
        solver = self.env.scene.sim.mpm_solver
        
        # FIX: Use centralized fetcher but get RAW particles for sampling logic
        # We handle subsampling manually below to define the indices
        particles = self._get_active_particles(use_cache=False, apply_subsampling=False)
        
        try:
            if particles is None:
                gs.logger.warning("No active particles for reconstruction")
                return

            # Subsampling logic
            if self.recon_particle_fraction < 1.0:
                num_keep = int(len(particles) * self.recon_particle_fraction)
                if num_keep > 0:
                    if (self.main_particle_indices is None or 
                        len(self.main_particle_indices) != num_keep or
                        self.last_total_particles != len(particles)):
                        
                        gs.logger.info(f"Regenerating sample with {self.sampling_method.value}...")
                        t_start = time.time()
                        
                        particles_tensor = torch.from_numpy(particles).float()
                        if self.device.type != 'cpu':
                            try:
                                particles_tensor = particles_tensor.to(self.device)
                            except Exception:
                                pass
                        
                        try:
                            indices = self._compute_sample_indices(particles_tensor, num_keep)
                            if isinstance(indices, torch.Tensor):
                                self.main_particle_indices = indices.cpu().numpy()
                            else:
                                self.main_particle_indices = indices
                        except Exception as e:
                            gs.logger.error(f"Sampling failed ({e}), falling back to random")
                            rng = np.random.default_rng(seed=42)
                            self.main_particle_indices = rng.choice(
                                len(particles), num_keep, replace=False
                            )
                            
                        self.last_total_particles = len(particles)
                        gs.logger.info(f"Sampling complete in {time.time() - t_start:.2f}s")
                        
                        # Invalidate skinning since particle subset changed
                        self._invalidate_skinning()
                        
                    particles = particles[self.main_particle_indices]

            # FIX: Manually update cache since we skipped it in _get_active_particles
            # This ensures visualization (e.g. Rerun) receives the subsampled particles used for this frame.
            self._cached_particles = particles
            self._cached_frame = self._global_frame

            # Radius scaling
            base_radius = solver.particle_radius
            self._cached_particle_radius = base_radius
            
            if self.recon_particle_fraction < 1.0 and self.recon_particle_fraction > 0:
                radius_scale = (1.0 / self.recon_particle_fraction) ** (1.0 / 3.0)
                radius_scale *= 1.15  # Buffer for overlap
                radius = base_radius * radius_scale
            else:
                radius = base_radius
                
            if len(particles) == 0:
                gs.logger.warning("No active particles for reconstruction")
                return

            self.reconstructed_mesh = pu.particles_to_mesh(
                positions=particles,
                radius=radius,
                backend='splashsurf'
            )
            
        except Exception as e:
            gs.logger.error(f"Surface reconstruction failed: {e}")
            self.reconstructed_mesh = trimesh.Trimesh()

    def _compute_sample_indices(self, particles: torch.Tensor, k: int) -> torch.Tensor:
        """Dispatch to selected sampling method."""
        if self.sampling_method == SamplingMethod.RANDOM:
            return self._sample_random(particles, k)
        elif self.sampling_method == SamplingMethod.VOXEL_STRATIFIED:
            return self._sample_voxel_stratified(particles, k)
        elif self.sampling_method == SamplingMethod.FPS:
            return self._sample_fps(particles, k)
        elif self.sampling_method == SamplingMethod.HALTON_LLOYD:
            return self._sample_halton_lloyd(particles, k)
        else:
            return self._sample_voxel_stratified(particles, k)

    # ==================== SAMPLING METHODS ====================

    def _sample_random(self, particles: torch.Tensor, k: int) -> torch.Tensor:
        """Simple random sampling (baseline)."""
        N = particles.shape[0]
        perm = torch.randperm(N, device=particles.device)
        return perm[:k]

    def _compute_voxel_size(
        self, 
        particles: torch.Tensor, 
        target_k: int, 
        iterations: int = 12
    ) -> float:
        """
        FIX #8: Binary search for voxel size giving ~k occupied voxels.
        Stays on GPU for speed.
        """
        device = particles.device
        min_bound = particles.min(dim=0).values
        max_bound = particles.max(dim=0).values
        extent = max_bound - min_bound
        
        # Initial estimate (assumes uniform fill of bounding box)
        volume = extent.prod().item()
        initial = (volume / target_k) ** (1/3)
        
        low, high = initial * 0.1, initial * 5.0
        
        for _ in range(iterations):
            mid = (low + high) / 2
            voxel_idx = torch.floor((particles - min_bound) / mid).long()
            keys = voxel_idx[:, 0] * 1000003 + voxel_idx[:, 1] * 1009 + voxel_idx[:, 2]
            num_occupied = len(torch.unique(keys))
            
            if num_occupied < target_k:
                high = mid  # Smaller voxels = more of them
            else:
                low = mid
        
        return mid

    def _sample_voxel_stratified(self, particles: torch.Tensor, k: int) -> torch.Tensor:
        """
        GPU-accelerated voxel stratified sampling.
        
        FIX #8: Uses binary search for accurate voxel count.
        FIX #9: Minimizes CPU<->GPU transfers.
        """
        N, D = particles.shape
        device = particles.device
        
        # FIX #8: Binary search for voxel size (stays on GPU)
        voxel_size = self._compute_voxel_size(particles, k)
        
        min_bound = particles.min(dim=0).values
        
        # Voxel assignment (GPU)
        voxel_idx = torch.floor((particles - min_bound) / voxel_size).long()
        voxel_keys = voxel_idx[:, 0] * 1000003 + voxel_idx[:, 1] * 1009 + voxel_idx[:, 2]
        
        # Get unique voxels and inverse mapping
        unique_keys, inverse, counts = torch.unique(
            voxel_keys, return_inverse=True, return_counts=True
        )
        num_voxels = len(unique_keys)
        
        gs.logger.debug(f"Voxel sampling: size={voxel_size:.6f}, voxels={num_voxels}, target={k}")
        
        # Compute distance to voxel center for each particle
        voxel_centers = min_bound + (voxel_idx.float() + 0.5) * voxel_size
        dist_to_center = torch.sum((particles - voxel_centers) ** 2, dim=1)
        
        # Find minimum distance per voxel
        INF = float('inf')
        min_dist_per_voxel = torch.full((num_voxels,), INF, device=device)
        
        # FIX #4: Remove include_self parameter for compatibility
        min_dist_per_voxel.scatter_reduce_(0, inverse, dist_to_center, reduce='amin')
        
        # Find particles that achieved the minimum in their voxel
        is_closest = dist_to_center == min_dist_per_voxel[inverse]
        
        # FIX #3: Use flatten() instead of squeeze() to handle edge cases
        candidates = torch.nonzero(is_closest).flatten()
        
        # Handle case where we have duplicate minima (ties)
        # Get unique voxels from candidates
        candidate_voxels = inverse[candidates]
        
        # Vectorized unique voxel selection - keep first occurrence of each voxel
        # Sort by voxel index to group same voxels together
        sorted_order = torch.argsort(candidate_voxels)
        sorted_voxels = candidate_voxels[sorted_order]
        
        # Find first occurrence: where value differs from previous
        first_mask = torch.ones(len(sorted_voxels), dtype=torch.bool, device=device)
        if len(sorted_voxels) > 1:
            first_mask[1:] = sorted_voxels[1:] != sorted_voxels[:-1]
        
        # Map back to original candidate order
        selected = candidates[sorted_order[first_mask]]
        
        # Adjust to target count k
        if len(selected) > k:
            # FIX #2: Specify device on randperm
            perm = torch.randperm(len(selected), device=device)[:k]
            selected = selected[perm]
            
        elif len(selected) < k:
            # Fill deficit from remaining particles
            deficit = k - len(selected)
            mask = torch.ones(N, dtype=torch.bool, device=device)
            mask[selected] = False
            remaining = torch.nonzero(mask).flatten()
            
            if remaining.numel() > 0:
                if remaining.numel() > deficit:
                    # FIX #2: Specify device on randperm
                    perm = torch.randperm(remaining.numel(), device=device)[:deficit]
                    extra = remaining[perm]
                else:
                    extra = remaining
                selected = torch.cat([selected, extra])

        return selected

    def _sample_fps(self, particles: torch.Tensor, k: int) -> torch.Tensor:
        """
        Farthest Point Sampling.
        
        Note: Biased toward surface/extremities - not ideal for volume sampling.
        """
        N = particles.shape[0]
        device = particles.device
        
        selected = torch.zeros(k, dtype=torch.long, device=device)
        min_dists = torch.full((N,), float('inf'), device=device)
        
        # Start from particle closest to centroid
        centroid = particles.mean(dim=0)
        dists = torch.sum((particles - centroid) ** 2, dim=1)
        first = torch.argmin(dists)
        selected[0] = first
        
        dists = torch.sum((particles - particles[first]) ** 2, dim=1)
        min_dists = torch.minimum(min_dists, dists)
        
        for i in range(1, k):
            farthest = torch.argmax(min_dists)
            selected[i] = farthest
            
            dists = torch.sum((particles - particles[farthest]) ** 2, dim=1)
            min_dists = torch.minimum(min_dists, dists)
            min_dists[farthest] = -float('inf')  # Exclude from future selection
        
        return selected

    def _sample_halton_lloyd(self, particles: torch.Tensor, k: int) -> torch.Tensor:
        """
        Halton sequence initialization + Lloyd's relaxation.
        
        Note: Halton fills bounding BOX, not actual shape. Bad for non-cubic volumes.
        """
        N, D = particles.shape
        device = particles.device
        
        try:
            from scipy.stats import qmc
            
            min_bound = torch.min(particles, dim=0).values
            max_bound = torch.max(particles, dim=0).values
            range_bound = max_bound - min_bound
            
            sampler = qmc.Halton(d=D, scramble=True)
            halton = sampler.random(n=k)
            halton = torch.tensor(halton, dtype=torch.float32, device=device)
            
            targets = min_bound + halton * range_bound
            dists = torch.cdist(targets, particles)
            centroids_idx = torch.argmin(dists, dim=1)
            
        except ImportError:
            gs.logger.warning("scipy not available, using random initialization")
            centroids_idx = torch.randperm(N, device=device)[:k]
        
        # Lloyd's relaxation
        for _ in range(15):
            centroids = particles[centroids_idx]
            D_mat = torch.cdist(particles, centroids)
            labels = torch.argmin(D_mat, dim=1)
            
            cluster_sums = torch.zeros((k, D), device=device)
            cluster_counts = torch.zeros((k, 1), device=device)
            
            labels_exp = labels.view(-1, 1).expand(-1, D)
            cluster_sums.scatter_add_(0, labels_exp, particles)
            cluster_counts.scatter_add_(0, labels.view(-1, 1), torch.ones((N, 1), device=device))
            cluster_counts = torch.clamp(cluster_counts, min=1.0)
            
            geo_centroids = cluster_sums / cluster_counts
            D_geo = torch.cdist(particles, geo_centroids)
            new_idx = torch.argmin(D_geo, dim=0)
            
            if torch.equal(centroids_idx, new_idx):
                break
            centroids_idx = new_idx
        
        return centroids_idx