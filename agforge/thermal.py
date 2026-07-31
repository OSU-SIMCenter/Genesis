import zlib

import numpy as np
import torch
import igl

import genesis as gs

from agforge.profiling_util import teleop_profile
from agforge.material_properties import CP_316L_SEG_PARAMS, cp_316l_seg


def get_steel_cp_numpy(temp: np.ndarray) -> np.ndarray:
    """Numpy vectorized specific heat [J/kg-K] for 316L austenitic stainless.

    CPU mirror of ``base_mpm_solver.get_steel_cp``. Delegates to
    ``material_properties.cp_316l_seg`` so there is exactly one definition of the
    curve the kernel implements — this used to be an independent transcription,
    which is precisely how the four copies of these constants drifted apart before.
    """
    return cp_316l_seg(temp)


def get_steel_cp_torch(temp: torch.Tensor) -> torch.Tensor:
    """Torch vectorized specific heat [J/kg-K] for 316L austenitic stainless.

    Torch-native reimplementation of the same 3-segment form as
    :func:`get_steel_cp_numpy`; kept on-device because this feeds viewer particle
    colouring. Knot values are imported rather than inlined so they cannot drift.
    """
    v0, t1, v1, t2, v2, slope_hi, t0 = CP_316L_SEG_PARAMS
    t = temp.reshape(-1).float()
    cp = torch.full_like(t, v0)
    cp = torch.where(t >= t2, v2 + (t - t2) * slope_hi, cp)

    mask_mid = (t >= t1) & (t < t2)
    if mask_mid.any():
        u = (t[mask_mid] - t1) / (t2 - t1)
        cp[mask_mid] = v1 + u * (v2 - v1)

    mask_low = (t > t0) & (t < t1)
    if mask_low.any():
        u = (t[mask_low] - t0) / (t1 - t0)
        cp[mask_low] = v0 + u * (v1 - v0)

    return cp


class InductionHeater:
    """Precomputes the per-particle induction skin depth from a surface-mesh SDF.

    The actual heat deposition is performed on the GPU inside the MPM P2G kernel
    (`base_mpm_solver.p2g_induction`), which reads the uploaded depth field together
    with the per-frame coil uniforms set via `solver.set_induction_params`. This class
    only owns the expensive, geometry-dependent CPU step: computing `|SDF|` of every
    particle below the coil-facing surface, which is recomputed once per geometry change
    (init, after each strike, on checkpoint restore) — never per frame.
    """

    def __init__(
        self,
        solver,
        entity,
        physics_mesher=None,
        reconstructor=None,
        static_verts=None,
        static_faces=None,
    ):
        """
        Parameters
        ----------
        solver : BaseMPMSolver
            The Genesis solver handling the MPM physics.
        entity : MPMEntity
            The MPM billet to heat.
        physics_mesher : InductionPhysicsMesher, optional
            Preferred source for the induction SDF surface mesh (high-res MC or SplashSurf).
        reconstructor : SurfaceReconstructor, optional
            Legacy fallback when no physics mesher is available.
        static_verts, static_faces : np.ndarray, optional
            Fallback surface mesh used if no reconstructor is available.
        """
        self.solver = solver
        self.entity = entity
        self.physics_mesher = physics_mesher
        self.reconstructor = reconstructor
        self.static_verts = static_verts
        self.static_faces = static_faces
        self._sdf_mesh_key: tuple | None = None
        self._cached_depth: np.ndarray | None = None

    @staticmethod
    def _mesh_fingerprint(verts: np.ndarray, faces: np.ndarray) -> tuple:
        return (
            int(verts.shape[0]),
            int(faces.shape[0]),
            int(zlib.adler32(verts.tobytes())),
            int(zlib.adler32(faces.tobytes())),
        )

    def _acquire_surface_mesh(self):
        """Return (verts, faces) of the current billet surface, or (None, None)."""
        if self.physics_mesher is not None:
            verts, faces = self.physics_mesher.get_verts_faces()
            if verts is not None:
                return verts, faces
            if self.physics_mesher.rebuild():
                verts, faces = self.physics_mesher.get_verts_faces()
                if verts is not None:
                    return verts, faces

        if (
            self.reconstructor is not None
            and getattr(self.reconstructor, "reconstructed_mesh", None) is not None
            and len(self.reconstructor.reconstructed_mesh.vertices) >= 4
            and len(self.reconstructor.reconstructed_mesh.faces) >= 4
        ):
            mesh = self.reconstructor.reconstructed_mesh
            return np.asarray(mesh.vertices, dtype=np.float64), np.asarray(mesh.faces, dtype=np.int32)

        # No usable mesh yet — force one so induction works before the first strike.
        if self.reconstructor is not None:
            try:
                self.reconstructor.create_reconstructed_mesh(is_deforming=False, one_shot_physics=True)
                self.reconstructor.mesh_version += 1
            except Exception as e:
                gs.logger.warning(f"InductionHeater: failed to force reconstruction: {e}")
            mesh = getattr(self.reconstructor, "reconstructed_mesh", None)
            if mesh is not None and len(mesh.vertices) >= 4 and len(mesh.faces) >= 4:
                return np.asarray(mesh.vertices, dtype=np.float64), np.asarray(mesh.faces, dtype=np.int32)

        if self.static_verts is not None and self.static_faces is not None:
            return np.asarray(self.static_verts, dtype=np.float64), np.asarray(self.static_faces, dtype=np.int32)

        return None, None

    def compute_skin_depth(self):
        """Compute |SDF| depth of every particle below the surface mesh.

        Returns a numpy array of shape [n_particles] (depth in metres), or None if no
        usable surface mesh is available yet (caller should leave the depth field unchanged).
        """
        verts, faces = self._acquire_surface_mesh()
        if verts is None:
            gs.logger.warning("InductionHeater: no surface mesh available, skipping skin-depth recompute.")
            return None

        scene = self.solver.sim.scene
        mesh_key = self._mesh_fingerprint(verts, faces)
        if self._sdf_mesh_key == mesh_key and self._cached_depth is not None:
            return self._cached_depth.copy()

        with teleop_profile(scene, "teleop_induction_pos_pull"):
            pos_tensor = self.entity.get_particles_pos()
            pos_np = np.asarray(pos_tensor.detach().cpu().numpy().reshape(-1, 3), dtype=np.float64)

        with teleop_profile(scene, "teleop_induction_igl_sdf"):
            # NB: igl.signed_distance()'s returned distance scalar is exactly 3x too large in
            # this libigl python binding (its closest points are correct, only the distance is
            # inflated). We only need the unsigned distance (depth uses abs anyway), so use the
            # robust point-to-mesh squared-distance instead. See thermal_skin_diagnostic.py.
            sqr_dist = igl.point_mesh_squared_distance(pos_np, verts, faces)[0]
        depth = np.sqrt(np.maximum(sqr_dist, 0.0))

        # Guard against NaN/inf from degenerate triangles: treat as deep interior (no heating).
        bad = ~np.isfinite(depth)
        if bad.any():
            gs.logger.warning(f"InductionHeater: {int(bad.sum())} particles got NaN/inf SDF distance, treating as interior.")
            depth[bad] = 1.0e9

        self._sdf_mesh_key = mesh_key
        self._cached_depth = depth
        return depth

    def recompute_and_upload(self):
        """Compute the skin depth and push it to the solver's GPU induction field.

        Returns True if the depth field was updated, False otherwise.
        """
        depth = self.compute_skin_depth()
        if depth is None:
            return False
        with teleop_profile(self.solver.sim.scene, "teleop_induction_depth_upload"):
            self.solver.set_induction_depth(depth)
        return True
