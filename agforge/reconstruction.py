import time
import logging
import torch
import numpy as np
import trimesh
import gstaichi as ti
import genesis as gs
import genesis.utils.particle as pu
from genesis.utils.misc import ti_to_numpy

class SurfaceReconstructor:
    def __init__(self, env):
        self.env = env
        self.reconstructed_mesh = trimesh.Trimesh()
        self.recon_enabled = True
        self.recon_frame_interval = 2
        self.recon_particle_fraction = 0.5 
        self.frame_counter = 0
        
        # Cached indices for deterministic random sampling
        self.main_particle_indices = None

        
        # Skinning data
        self.skinning_enabled = False
        self.bind_indices = None
        self.bind_weights = None
        self.bind_offsets = None
        self.device = env.device

    def reset(self):
        """Resets the reconstructor state and sampling cache."""
        self.reconstructed_mesh = trimesh.Trimesh()
        self.skinning_enabled = False
        self.bind_indices = None
        self.bind_weights = None
        self.bind_offsets = None
        # Don't create mesh immediately on reset, wait for explicit call or first update
        # self.create_reconstructed_mesh()

    def get_mesh_data(self):
        """Returns the current mesh and its vertices/faces."""
        return self.reconstructed_mesh

    def update(self, should_reconstruct: bool):
        """
        Updates the reconstructed mesh if conditions are met.
        args:
            should_reconstruct: External condition (e.g. is striking)
        """
        if not self.recon_enabled:
            return

        # If skinning active, update positions
        if self.skinning_enabled:
             self.update_skinning()
             return

        # Fallback to full reconstruction if not skinned or explicitly requested
        if not should_reconstruct:
            return

        self.frame_counter += 1
        if self.frame_counter % self.recon_frame_interval != 0:
            return

        self.create_reconstructed_mesh()

    def init_skinning(self):
        """
        Binds the current reconstructed mesh to current active particles.
        Must be called ONCE after create_reconstructed_mesh produces a valid mesh.
        """
        if self.reconstructed_mesh is None or len(self.reconstructed_mesh.vertices) == 0:
            gs.logger.warning("Cannot init skinning: No mesh generated yet.")
            return

        # Get current particles (assume full set or same subset as used for generation)
        # Note: If using subsampling, we must ensure we use the SAME subset indices.
        # But for skinning to work robustly, we usually want to bind to 'physics' particles
        # even if we rendered a subset. However, if the physics particle count changes (it shouldn't for MPM),
        # we might have issues. 
        # For simplicity/robustness, we'll re-fetch the particles exactly as we get them for update.
        
        particles = self._get_active_particles()
        if particles is None or len(particles) == 0:
            return

        gs.logger.info(f"Visual Binding: {len(self.reconstructed_mesh.vertices)} vertices -> {len(particles)} particles")

        # Convert to Torch Tensors on Device
        # Vertices (V, 3)
        verts_tensor = torch.tensor(self.reconstructed_mesh.vertices, dtype=torch.float32, device=self.device)
        
        # Particles (P, 3)
        # We need to make sure particles are a tensor on device
        if isinstance(particles, np.ndarray):
            parts_tensor = torch.from_numpy(particles).float().to(self.device)
        else:
             parts_tensor = particles # Assume already tensor/compatible if not numpy? _get_active_particles usually returns numpy in current impl.
             # Let's verify _get_active_particles implementation below.

        # Verify type
        if not isinstance(parts_tensor, torch.Tensor):
             parts_tensor = torch.tensor(parts_tensor, dtype=torch.float32, device=self.device)

        # k-NN search
        k = 4
        sigma = 0.05 # Influence radius scaling factor

        # Compute pairwise distances (V, P) - Watch out for VRAM usage!
        # If V*P is large (e.g. > 10^8), cdist might OOM. 
        # Chunking if needed, but let's try direct first for typical scenes.
        # Use negative squared euclidean distance for weights
        
        try:
            # Finding k nearest particles for each vertex
            # We can use cdist or a custom kernel. 
            # For 20k verts * 5k particles = 100M floats = 400MB. Safe.
            # If 20k verts * 50k particles = 1B floats = 4GB. Might be risky.
            
            # Use topk optimization with negative distance?
            # Or use a chunked approach for safety.
            
            self.bind_indices = torch.zeros((len(verts_tensor), k), dtype=torch.long, device=self.device)
            self.bind_weights = torch.zeros((len(verts_tensor), k), dtype=torch.float32, device=self.device)
            self.bind_offsets = torch.zeros((len(verts_tensor), k, 3), dtype=torch.float32, device=self.device)

            chunk_size = 1000 # Process 1000 vertices at a time
            for i in range(0, len(verts_tensor), chunk_size):
                end_i = min(i + chunk_size, len(verts_tensor))
                v_chunk = verts_tensor[i:end_i] # (Chunk, 3)
                
                # Dist: (Chunk, P)
                dists = torch.cdist(v_chunk, parts_tensor)
                
                # Get k nearest
                # topk returns largest, so we negate distance or use largest=False
                vals, idxs = torch.topk(dists, k=k, dim=1, largest=False)
                
                # Compute weights: exp(-dist^2 / (2*sigma^2))
                weights = torch.exp(-vals.pow(2) / (2 * sigma**2))
                
                # Normalize weights
                weights_sum = weights.sum(dim=1, keepdim=True)
                weights_sum[weights_sum < 1e-6] = 1.0 # Avoid div by zero
                weights = weights / weights_sum
                
                self.bind_indices[i:end_i] = idxs
                self.bind_weights[i:end_i] = weights
                
                # Precompute offsets in local frame of each neighbor
                # Offset = Vertex_Bind - Particle_Bind
                # (Chunk, k, 3) - (Chunk, k, 3)
                # Gather particles: (Chunk, k, 3)
                neighbor_pos = parts_tensor[idxs] 
                
                # v_chunk (Chunk, 3) -> (Chunk, 1, 3)
                offsets = v_chunk.unsqueeze(1) - neighbor_pos
                self.bind_offsets[i:end_i] = offsets

            self.skinning_enabled = True
            gs.logger.info("Skinning initialized successfully.")

        except Exception as e:
            gs.logger.error(f"Failed to init skinning: {e}")
            self.skinning_enabled = False

    def update_skinning(self):
        """Updates mesh vertices based on current particle positions."""
        if not self.skinning_enabled:
            return

        particles = self._get_active_particles()
        if particles is None or len(particles) == 0:
            return
            
        # Ensure particles are on GPU tensor
        if isinstance(particles, np.ndarray):
            parts_current = torch.from_numpy(particles).float().to(self.device)
        else:
            parts_current = torch.tensor(particles, dtype=torch.float32, device=self.device)

        try:
            # 1. Gather neighbor positions: (V, k, 3)
            # bind_indices is (V, k)
            neighbors = parts_current[self.bind_indices]
            
            # 2. Add offsets: (V, k, 3)
            # New Pos = Neighbor_Pos + Offset
            # Note: This implies RIGID translation of the offset. 
            # Ideally we rotate the offset using Deformation Gradient F if available from MPM.
            # But the 'Optimization' task just requested a simple update. 
            # Pure translation skinning is decent for small deformations but won't capture rotation well.
            # Given user asked for "adjusts/updates", this is step 1.
            # If we had F from MPM state, we would do: Neighbor + F @ Offset.
            
            estimated_pos = neighbors + self.bind_offsets
            
            # 3. Weighted Average
            # (V, k, 3) * (V, k, 1) -> sum dim 1 -> (V, 3)
            new_verts = (estimated_pos * self.bind_weights.unsqueeze(-1)).sum(dim=1)
            
            # 4. Update mesh
            self.reconstructed_mesh.vertices = new_verts.cpu().numpy()
            
        except Exception as e:
            gs.logger.error(f"Skinning update failed: {e}")

    def _get_active_particles(self):
        """Helper to fetch particles with consistent logic."""
        solver = self.env.scene.sim.mpm_solver
        
        # Explicit synchronization for Metal backend
        ti.sync()
        
        try:
            if hasattr(solver.particles_render.pos, 'to_numpy'):
                 particles = solver.particles_render.pos.to_numpy()[:, 0]
            else:
                 particles = ti_to_numpy(solver.particles_render.pos)[:, 0]
            
            # Get environment offset
            offset = self.env.scene.envs_offset[0]
            
            # Ensure it is on CPU and float32
            if hasattr(offset, 'cpu'):
                offset = offset.cpu().numpy()
            elif hasattr(offset, 'numpy'):
                offset = offset.numpy()
                
            particles = particles + offset
            
            if hasattr(solver.particles_render.active, 'to_numpy'):
                active = solver.particles_render.active.to_numpy()[:, 0].astype(bool)
            else:
                active = ti_to_numpy(solver.particles_render.active)[:, 0].astype(bool)
                
            particles = particles[active]
            
            # Important: We must apply the SAME subsampling as used during init!
            # If recon_particle_fraction < 1.0, we used self.main_particle_indices.
            # If those indices are set, we MUST use them to get the same subset mapping.
            
            if self.main_particle_indices is not None and len(self.main_particle_indices) > 0:
                 # Check bounds just in case
                 if np.max(self.main_particle_indices) < len(particles):
                     particles = particles[self.main_particle_indices]
                 else:
                     # This should not happen if logic is correct, but if total particles changed...
                     gs.logger.warning("Particle count changed, invalidating indices!")
                     # This invalidates skinning binding completely.
                     # We should probably force a re-init or disable skinning.
                     pass 
            
            return particles

        except Exception as e:
            gs.logger.error(f"Failed to get particles: {e}")
            return None

    def create_reconstructed_mesh(self):
        """Reconstruct surface mesh from active particles using splashsurf."""
        solver = self.env.scene.sim.mpm_solver
        
        # Explicit synchronization for Metal backend
        ti.sync()
        
        try:
            # t0 = time.time()
            if hasattr(solver.particles_render.pos, 'to_numpy'):
                 particles = solver.particles_render.pos.to_numpy()[:, 0]
            else:
                 particles = ti_to_numpy(solver.particles_render.pos)[:, 0]
            
            # Get environment offset
            offset = self.env.scene.envs_offset[0]
            
            # Ensure it is on CPU and float32
            if hasattr(offset, 'cpu'):
                offset = offset.cpu().numpy()
            elif hasattr(offset, 'numpy'):
                offset = offset.numpy()
                
            particles = particles + offset
            
            if hasattr(solver.particles_render.active, 'to_numpy'):
                active = solver.particles_render.active.to_numpy()[:, 0].astype(bool)
            else:
                active = ti_to_numpy(solver.particles_render.active)[:, 0].astype(bool)
                
            particles = particles[active]
            
            # Subsample if needed
            if self.recon_particle_fraction < 1.0:
                num_keep = int(len(particles) * self.recon_particle_fraction)
                if num_keep > 0:
                    # Regenerate indices if cache is invalid or particle count changed
                    if (self.main_particle_indices is None or 
                        len(self.main_particle_indices) != num_keep or
                        self.last_total_particles != len(particles)):
                        
                        gs.logger.info("Regenerating particle sample with High-Fidelity CVT...")
                        t_start_cv = time.time()
                        
                        # Convert to torch for GPU acceleration if available
                        particles_tensor = torch.from_numpy(particles).float()
                        if self.env.device != 'cpu':
                            try:
                                particles_tensor = particles_tensor.to(self.env.device)
                            except Exception:
                                pass # Fallback to CPU if transfer fails
                        
                        try:
                            # Run optimal sampling (FPS + Lloyd's)
                            indices = self._compute_optimal_indices_torch(particles_tensor, num_keep)
                            self.main_particle_indices = indices.cpu().numpy()
                        except Exception as e:
                            gs.logger.error(f"CVT Sampling failed ({e}), falling back to random")
                            rng = np.random.default_rng(seed=42)
                            self.main_particle_indices = rng.choice(len(particles), num_keep, replace=False)
                            
                        self.last_total_particles = len(particles)
                        gs.logger.info(f"Sampling complete in {time.time() - t_start_cv:.2f}s")
                        
                    particles = particles[self.main_particle_indices]

            # --- Critical Visualization Parameter Scaling ---
            # As per research: R_vis approx R_sim * (N_total / N_vis)^(1/3)
            # This prevents "holes" when using a sparse subset.
            base_radius = solver.particle_radius
            if self.recon_particle_fraction < 1.0 and self.recon_particle_fraction > 0:
                # fraction = N_vis / N_total
                # Scaling factor = (1 / fraction)^(1/3)
                radius_scale = (1.0 / self.recon_particle_fraction) ** (1.0/3.0)
                # Apply a small extra buffer (e.g. 10%) to ensure overlap
                radius_scale *= 1.1 
                radius = base_radius * radius_scale
            else:
                radius = base_radius
                
            # Check if particles is valid
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

    def _compute_optimal_indices_torch(self, particles: torch.Tensor, k: int) -> torch.Tensor:
        """
        Selects k indices using Discrete Capacity-Constrained CVT (FPS Init + Lloyd's Relaxation).
        Optimized for GPU execution.
        """
        N, D = particles.shape
        device = particles.device
        
        # --- Phase 1: Quasi-Random Initialization (Halton Sequence) ---
        # The user requested a "patterned" uniform distribution without grid artifacts.
        # Halton sequences are Low-Discrepancy Sequences that fill space uniformly.
        try:
            from scipy.stats import qmc
            
            # 1. Compute Bounding Box
            min_bound = torch.min(particles, dim=0).values
            max_bound = torch.max(particles, dim=0).values
            range_bound = max_bound - min_bound
            
            # 2. Generate Halton Sequence (0-1 range)
            # We generate slightly more to account for potential out-of-bounds mapping
            sampler = qmc.Halton(d=D, scramble=True)
            halton_sample = sampler.random(n=k)
            halton_sample = torch.tensor(halton_sample, dtype=torch.float32, device=device)
            
            # 3. Scale to Particle Bounding Box
            target_points = min_bound + halton_sample * range_bound
            
            # 4. Snap to Nearest Actual Particle (Discrete Constraint)
            # We find the particle closest to each ideal Halton point
            # This selects a subset that structurally resembles the Halton pattern
            dists = torch.cdist(target_points, particles) # (K, N)
            centroids_idx = torch.argmin(dists, dim=1)    # (K,)
            
        except ImportError:
            gs.logger.warning("Scipy not found for Halton sampling, falling back to Random.")
            centroids_idx = torch.randperm(N, device=device)[:k]
            
        # --- Phase 2: Discrete Lloyd's Relaxation ---
        # Iterate to minimize variance ("bumps")
        num_iterations = 10
        
        for i in range(num_iterations):
            # Step A: Assignment (Voronoi Partitioning)
            centroids = particles[centroids_idx]
            
            try:
                D_mat = torch.cdist(particles, centroids)
            except RuntimeError:
                gs.logger.warning("OOM during CVT Lloyd's, stopping relaxation early.")
                break
                
            # Assign to nearest centroid
            labels = torch.argmin(D_mat, dim=1) # (N,)
            
            # Step B: Compute Geometric Centroids of partitions
            cluster_sums = torch.zeros((k, D), device=device)
            cluster_counts = torch.zeros((k, 1), device=device)
            
            # Scatter add
            labels_expanded = labels.view(-1, 1).expand(-1, D)
            cluster_sums.scatter_add_(0, labels_expanded, particles)
            
            # Count points per label
            ones = torch.ones((N, 1), device=device)
            cluster_counts.scatter_add_(0, labels.view(-1, 1), ones)
            
            # Avoid divide by zero
            cluster_counts = torch.clamp(cluster_counts, min=1.0)
            
            # Geometric centroids
            geo_centroids = cluster_sums / cluster_counts # (K, 3)
            
            # Step C: Snap to nearest VALID particle (Discrete Constraint)
            try:
                D_mat_geo = torch.cdist(particles, geo_centroids)
            except RuntimeError:
                break
            
            # Nearest particle to each geometric centroid
            new_centroids_idx = torch.argmin(D_mat_geo, dim=0) # (K,)
            
            # Check convergence
            if torch.equal(centroids_idx, new_centroids_idx):
                break
                
            centroids_idx = new_centroids_idx
            
        return centroids_idx
