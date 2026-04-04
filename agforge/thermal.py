import numpy as np
import torch
import igl

import genesis as gs


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

    def step_heat(self, dt: float, surface_power: float = 2000.0, skin_depth: float = 0.02, coil_center=None, coil_radius: float = 0.3):
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
        # 1. Acquire current boundary surface
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

        # 2. Extract particle positions
        # shape [B, n_particles, 3] usually, squeeze down to [n_particles, 3] for igl
        pos_tensor = self.entity.get_particles_pos()
        pos_np = pos_tensor.cpu().numpy().squeeze()
        if pos_np.ndim != 2 or pos_np.shape[-1] != 3:
            raise ValueError(f"Unexpected particle position shape: {pos_np.shape}. Expected (N, 3).")

        # 3. Calculate Signed Distance (depth from surface)
        distances, _, _, _ = igl.signed_distance(pos_np.astype(np.float64), verts, faces)
        depth = np.abs(distances)

        # Guard against NaN/inf from degenerate triangles
        bad_mask = ~np.isfinite(depth)
        if bad_mask.any():
            n_bad = bad_mask.sum()
            gs.logger.warning(f"InductionHeater: {n_bad} particles got NaN/inf SDF distance, treating as interior.")
            depth[bad_mask] = skin_depth * 10.0  # effectively zero heating

        # 4. Read current particle temperatures
        # shape [B, n_particles] -> squeeze to [n_particles]
        current_temp_tensor = self.entity.get_particles_temp()
        current_temp = current_temp_tensor.cpu().numpy().squeeze()

        # Guard: if temps are already NaN, skip to avoid cascading corruption
        if not np.isfinite(current_temp).all():
            gs.logger.warning("InductionHeater: NaN detected in current particle temps, skipping heat step.")
            return

        # 5. Compute heating delta via energy conservation: dT = P * dt / (m * Cp)
        # NOTE: Genesis inflates particle mass by _particle_volume_scale (1e3) for numerical
        # stability. This cancels internally in the engine (mass/volume = rho), but we need
        # the real physical mass here for correct energy-to-temperature conversion.
        particle_mass_scaled = self.solver.particles_info[0].mass
        particle_mass = particle_mass_scaled / self.solver._particle_volume_scale
        Cp = self.solver._default_heat_capacity
        delta_temp = surface_power * np.exp(-depth / skin_depth) * dt / (particle_mass * Cp)
        
        # 5b. Apply positional mask if coil bounds provided
        if coil_center is not None:
            c = np.array(coil_center, dtype=np.float64)
            # Usually induction coils check radial distance on XZ plane, or full 3D depending on orientation.
            # Assuming a standard 3D spherical/radial mask
            dists = np.linalg.norm(pos_np - c, axis=1)
            mask = dists <= coil_radius
            delta_temp[~mask] = 0.0

        # 6. Push new heat state
        new_temp = current_temp + delta_temp
        new_temp_tensor = torch.tensor(new_temp, dtype=torch.float32, device=current_temp_tensor.device)

        # Ensure it perfectly matches the original buffer dimensions (e.g., [1, N])
        # If the original was [1, N], unsqueeze back
        if len(current_temp_tensor.shape) > 1 and len(new_temp_tensor.shape) == 1:
            new_temp_tensor = new_temp_tensor.unsqueeze(0)

        self.entity.set_particles_temp(new_temp_tensor)
        return float(delta_temp.max())
