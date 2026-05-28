import numpy as np
import torch
import igl

import genesis as gs

def get_steel_cp_numpy(temp: np.ndarray) -> np.ndarray:
    """Numpy vectorized computation of temperature-dependent specific heat for low-alloy steel."""
    cp = np.full_like(temp, 450.0)
    
    mask_high = temp >= 1000.0
    cp[mask_high] = 750.0
    
    mask_mid = (temp >= 700.0) & (temp < 1000.0)
    if np.any(mask_mid):
        u = (temp[mask_mid] - 700.0) / 300.0
        cp[mask_mid] = 580.0 + u * 70.0
        
    mask_low = (temp > 293.15) & (temp < 700.0)
    if np.any(mask_low):
        u = (temp[mask_low] - 293.15) / 406.85
        cp[mask_low] = 450.0 + u * 130.0
        
    return cp


def get_steel_cp_torch(temp: torch.Tensor) -> torch.Tensor:
    """PyTorch GPU-compatible computation of temperature-dependent specific heat for low-alloy steel.
    
    Same piecewise linear model as get_steel_cp_numpy, but operates entirely on GPU tensors.
    """
    cp = torch.full_like(temp, 450.0)
    
    # T >= 1000K: Cp = 750
    mask_high = temp >= 1000.0
    cp[mask_high] = 750.0
    
    # 700K <= T < 1000K: Cp = 580 + (T-700)/300 * 70
    mask_mid = (temp >= 700.0) & (temp < 1000.0)
    if mask_mid.any():
        u = (temp[mask_mid] - 700.0) / 300.0
        cp[mask_mid] = 580.0 + u * 70.0
    
    # 293K < T < 700K: Cp = 450 + (T-293)/407 * 130
    mask_low = (temp > 293.15) & (temp < 700.0)
    if mask_low.any():
        u = (temp[mask_low] - 293.15) / 406.85
        cp[mask_low] = 450.0 + u * 130.0
    
    return cp


class InductionHeater:
    def __init__(self, solver, entity, reconstructor=None, static_verts=None, static_faces=None):
        """
        Initialize the InductionHeater.

        Parameters
        ----------
        solver : BaseMPMSolver
            The Genesis solver handling the MPM physics.
        entity : MPMEntity
            The MPM billet to heat.
        reconstructor : SurfaceReconstructor, optional
            A pipeline that dynamically provides `reconstructed_mesh`.
            If None, heating will rely on the static CAD mesh.
        static_verts : np.ndarray, optional
            Static vertices structure if no reconstructor is used.
        static_faces : np.ndarray, optional
            Static faces structure if no reconstructor is used.
        """
        self.solver = solver
        self.entity = entity
        self.reconstructor = reconstructor
        self.static_verts = static_verts
        self.static_faces = static_faces
        
        self._cached_mesh_version = -1
        self._cached_weights_gpu = None  # GPU tensor (was numpy _cached_weights)
        self._cached_pos_gpu = None      # GPU tensor (was numpy _cached_pos)
        self._particle_mass = None       # Cached real particle mass (scalar)

    def step_heat(self, dt: float, surface_power: float = 2000.0, skin_depth: float = 0.02, coil_center=None, coil_radius: float = 0.3, profile_ctx=None):
        """
        Inject heat energy into the particle domain using an induction profile.
        Heat falls off exponentially proportional to depth inside the reconstructed surface.
        If coil_center is provided (like [x,y,z]), heating is strictly masked to within coil_radius.

        Temperature rise is computed via energy conservation: dT = P * dt / (m * Cp)

        Parameters
        ----------
        dt : float
            Time step for the heating duration.
        surface_power : float
            Heating power in Watts at the absolute surface boundary.
            Distributed per-particle with exponential skin-depth falloff.
        skin_depth : float
            The e-folding depth parameter for exponential falloff.
        """
        import contextlib
        def prof(name):
            if profile_ctx is not None:
                return profile_ctx(name)
            return contextlib.suppress()

        current_version = getattr(self.reconstructor, 'mesh_version', -1) if self.reconstructor else 0
        cache_miss = current_version != self._cached_mesh_version or self._cached_weights_gpu is None

        # 1. Acquire current boundary surface (CPU-side, only on cache miss)
        with prof("heat_prep_mesh"):
            if cache_miss:
                if (
                    self.reconstructor is not None
                    and hasattr(self.reconstructor, "reconstructed_mesh")
                    and self.reconstructor.reconstructed_mesh is not None
                ):
                    mesh = self.reconstructor.reconstructed_mesh
                    if mesh.vertices is None or len(mesh.vertices) == 0:
                        gs.logger.warning("InductionHeater: empty reconstruction mesh, skipping heat step.")
                        return
                    verts = np.asarray(mesh.vertices, dtype=np.float64)
                    faces = np.asarray(mesh.faces, dtype=np.int32)
                else:
                    if self.static_verts is None or self.static_faces is None:
                        raise ValueError(
                            "InductionHeater needs either a valid SurfaceReconstructor or explicit static_verts/faces."
                        )
                    verts = self.static_verts
                    faces = self.static_faces
    
                if len(verts) < 4 or len(faces) < 4:
                    gs.logger.warning(f"InductionHeater: degenerate mesh ({len(verts)} verts, {len(faces)} faces), skipping.")
                    return

        with prof("heat_sdf"):
            if cache_miss:
                # SDF computation must stay CPU-side (libigl is CPU-only).
                # But we only do this on cache miss (mesh reconstruction events).
                pos_tensor = self.entity.get_particles_pos()
                pos_np = pos_tensor.cpu().numpy().squeeze()
                if pos_np.ndim != 2 or pos_np.shape[-1] != 3:
                    raise ValueError(f"Unexpected particle position shape: {pos_np.shape}. Expected (N, 3).")
    
                distances, _, _, _ = igl.signed_distance(pos_np.astype(np.float64), verts, faces)
                depth = np.abs(distances)
    
                # Guard against NaN/inf from degenerate triangles
                bad_mask = ~np.isfinite(depth)
                if bad_mask.any():
                    n_bad = bad_mask.sum()
                    gs.logger.warning(f"InductionHeater: {n_bad} particles got NaN/inf SDF distance, treating as interior.")
                    depth[bad_mask] = skin_depth * 10.0  # effectively zero heating
                    
                # Cache weights and positions as GPU tensors
                weights_np = np.exp(-depth / skin_depth)
                self._cached_weights_gpu = torch.tensor(weights_np, dtype=torch.float32, device=pos_tensor.device)
                self._cached_pos_gpu = pos_tensor.squeeze(0)  # [N, 3] on GPU
                self._cached_mesh_version = current_version
                
                # Cache the real particle mass (scalar, doesn't change)
                particle_mass_scaled = self.solver.particles_info[0].mass
                self._particle_mass = particle_mass_scaled / self.solver._particle_volume_scale

        # --- Per-step heating: 100% GPU when cache hit ---
        with prof("heat_apply"):
            # Read current temperatures (stays on GPU — no .cpu()!)
            current_temp = self.entity.get_particles_temp()  # [1, N] or [B, N]
            
            # NaN guard (single GPU op, no sync)
            if torch.isnan(current_temp).any():
                gs.logger.warning("InductionHeater: NaN detected in current particle temps, skipping heat step.")
                return

            # Flatten to [N] for math
            orig_shape = current_temp.shape
            t = current_temp.float().view(-1)  # [N]
            
            # Temperature-dependent Cp on GPU
            Cp = get_steel_cp_torch(t)
            
            # Start from cached SDF weights
            weights = self._cached_weights_gpu.clone()

            # Apply coil spatial mask on GPU
            if coil_center is not None:
                # coil_center is a Python list [x, y, z]
                coil_x = coil_center[0]
                dx = self._cached_pos_gpu[:, 0] - coil_x
                mask = torch.abs(dx) <= coil_radius
                weights = weights * mask.float()

            weight_sum = weights.sum()
            if weight_sum.item() < 1e-12:
                return 0.0  # no particles to heat

            # dT = P * w_i / Σw * dt / (m * Cp)
            delta_temp = surface_power * weights / weight_sum * dt / (self._particle_mass * Cp)
            
            # Apply and write back (stays on GPU — no CPU round-trip!)
            new_temp = t + delta_temp
            self.entity.set_particles_temp(new_temp.view(orig_shape))
            return delta_temp.max().item()
