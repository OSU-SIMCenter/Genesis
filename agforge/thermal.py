import numpy as np
import torch
import igl

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

    def step_heat(self, dt: float, surface_power: float, skin_depth: float):
        """
        Apply a single step of radial induction heating using the SDF skin effect.
        The formula applied is:
            delta_T = surface_power * exp(-depth / skin_depth) * dt

        Parameters
        ----------
        dt : float
            Time step for the heating duration.
        surface_power : float
            Heating power measured in Kelvin/second at the absolute boundary.
        skin_depth : float
            The e-folding depth parameter for exponential falloff.
        """
        # 1. Acquire current boundary surface
        if self.reconstructor is not None and hasattr(self.reconstructor, "reconstructed_mesh") and self.reconstructor.reconstructed_mesh is not None:
            verts = np.asarray(self.reconstructor.reconstructed_mesh.vertices)
            faces = np.asarray(self.reconstructor.reconstructed_mesh.faces)
        else:
            if self.static_verts is None or self.static_faces is None:
                raise ValueError("InductionHeater needs either a valid SurfaceReconstructor or explicit static_verts/faces.")
            verts = self.static_verts
            faces = self.static_faces

        # 2. Extract particle positions
        # shape [B, n_particles, 3] usually, squeeze down to [n_particles, 3] for igl
        pos_tensor = self.entity.get_particles_pos()
        pos_np = pos_tensor.cpu().numpy().squeeze()
        if pos_np.ndim != 2 or pos_np.shape[-1] != 3:
            raise ValueError(f"Unexpected particle position shape: {pos_np.shape}. Expected (N, 3).")

        # 3. Calculate Signed Distance (depth from surface)
        distances, _, _, _ = igl.signed_distance(pos_np, verts, faces)
        depth = np.abs(distances)

        # 4. Read current particle temperatures
        # shape [B, n_particles] -> squeeze to [n_particles]
        current_temp_tensor = self.entity.get_particles_temp()
        current_temp = current_temp_tensor.cpu().numpy().squeeze()

        # 5. Compute heating delta
        delta_temp = surface_power * np.exp(-depth / skin_depth) * dt

        # 6. Push new heat state
        new_temp = current_temp + delta_temp
        new_temp_tensor = torch.tensor(new_temp, dtype=torch.float32, device=current_temp_tensor.device)
        
        # Ensure it perfectly matches the original buffer dimensions (e.g., [1, N])
        # If the original was [1, N], unsqueeze back
        if len(current_temp_tensor.shape) > 1 and len(new_temp_tensor.shape) == 1:
            new_temp_tensor = new_temp_tensor.unsqueeze(0)

        self.entity.set_particles_temp(new_temp_tensor)
