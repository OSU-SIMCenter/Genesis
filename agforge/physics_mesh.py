"""One-shot surface meshes for induction SDF / skin-depth computation.

When ``reconstruction.unified_mesh`` is True (default), a single build here also
publishes to the visual reconstructor so Unity/viewer overlays avoid a second pass.

When unified_mesh is False, this remains separate from the real-time visual
reconstructor (grid_res=64, temporal blending during strikes).
"""

from __future__ import annotations

import time
from enum import Enum

import numpy as np
import trimesh

import genesis as gs

from agforge.profiling_util import teleop_profile
from agforge.reconstruction import (
    SurfaceReconstructor,
    build_splashsurf_mesh_from_env,
)


class PhysicsMeshBackend(str, Enum):
    HYBRID_LOW = "hybrid_low"
    HYBRID_HIGH = "hybrid_high"
    SPLASHSURF = "splashsurf"


PHYSICS_MESH_BACKENDS: tuple[str, ...] = tuple(b.value for b in PhysicsMeshBackend)

PHYSICS_MESH_BACKEND_LABELS: dict[str, str] = {
    PhysicsMeshBackend.HYBRID_LOW.value: "MC low-res",
    PhysicsMeshBackend.HYBRID_HIGH.value: "MC high-res",
    PhysicsMeshBackend.SPLASHSURF.value: "SplashSurf",
}


class InductionPhysicsMesher:
    """Builds and caches the mesh used for igl SDF / induction_depth uploads."""

    def __init__(self, env, visual_reconstructor: SurfaceReconstructor, cfg):
        self.env = env
        self.visual_reconstructor = visual_reconstructor
        self.cfg = cfg
        backend_str = getattr(cfg.reconstruction, "physics_mesh_backend", PhysicsMeshBackend.SPLASHSURF.value)
        try:
            self.backend = PhysicsMeshBackend(backend_str)
        except ValueError:
            gs.logger.warning(f"Unknown physics_mesh_backend '{backend_str}', defaulting to splashsurf")
            self.backend = PhysicsMeshBackend.SPLASHSURF
        self.physics_mesh = trimesh.Trimesh()
        self.version = 0

    @property
    def unified_mesh(self) -> bool:
        return bool(getattr(self.cfg.reconstruction, "unified_mesh", True))

    @property
    def backend_label(self) -> str:
        return PHYSICS_MESH_BACKEND_LABELS.get(self.backend.value, self.backend.value)

    def cycle_backend(self, step: int = 1) -> PhysicsMeshBackend:
        backends = list(PhysicsMeshBackend)
        idx = backends.index(self.backend)
        self.backend = backends[(idx + step) % len(backends)]
        self.cfg.reconstruction.physics_mesh_backend = self.backend.value
        return self.backend

    def set_backend(self, backend: str | PhysicsMeshBackend) -> PhysicsMeshBackend:
        if isinstance(backend, PhysicsMeshBackend):
            self.backend = backend
        else:
            self.backend = PhysicsMeshBackend(backend)
        self.cfg.reconstruction.physics_mesh_backend = self.backend.value
        return self.backend

    def _build_hybrid_low(self) -> trimesh.Trimesh:
        # Same pipeline/grid as the teleop visual reconstructor, snapped to current particles.
        self.visual_reconstructor.density_initialized = False
        self.visual_reconstructor._prev_verts = None
        self.visual_reconstructor.create_reconstructed_mesh(
            is_deforming=False,
            one_shot_physics=True,
        )
        return self.visual_reconstructor.reconstructed_mesh.copy()

    def _build_hybrid_high(self) -> trimesh.Trimesh:
        return self.visual_reconstructor.create_physics_reconstructed_mesh(
            is_deforming=False,
            one_shot_physics=True,
        )

    def update_live(self, should_reconstruct: bool = True, is_deforming: bool = True) -> bool:
        """Live unified mesh during strikes (same cadence as legacy visual recon)."""
        if not self.unified_mesh:
            return False
        if not self.visual_reconstructor.update_unified(should_reconstruct, is_deforming):
            return False
        mesh = self.visual_reconstructor.reconstructed_mesh
        if mesh is None or len(mesh.vertices) < 4:
            return False
        with teleop_profile(self.env, "teleop_unified_recon_mesh_copy"):
            self.physics_mesh = mesh.copy()
        self.version = self.visual_reconstructor.mesh_version
        return True

    def _build_splashsurf(self) -> trimesh.Trimesh:
        return build_splashsurf_mesh_from_env(self.env)

    def _publish_to_visual(self, mesh: trimesh.Trimesh) -> None:
        """Copy the physics mesh into the visual reconstructor (unified_mesh mode)."""
        recon = self.visual_reconstructor
        recon.reconstructed_mesh = mesh.copy()
        recon.mesh_version += 1
        recon.density_initialized = False
        recon._prev_verts = None
        if hasattr(recon, "reconstructed_vertices_tensor"):
            recon.reconstructed_vertices_tensor = None
        # Refresh particle cache used for Unity vertex temperature mapping.
        recon._get_active_particles(use_cache=False)

    def rebuild(self, backend: str | PhysicsMeshBackend | None = None) -> bool:
        if backend is not None:
            self.set_backend(backend)

        t0 = time.time()
        try:
            with teleop_profile(self.env, "teleop_physics_mesh_rebuild"):
                if self.backend == PhysicsMeshBackend.HYBRID_LOW:
                    mesh = self._build_hybrid_low()
                elif self.backend == PhysicsMeshBackend.HYBRID_HIGH:
                    mesh = self._build_hybrid_high()
                else:
                    mesh = self._build_splashsurf()

                if mesh is None or len(mesh.vertices) < 4 or len(mesh.faces) < 4:
                    gs.logger.warning(f"Physics mesh [{self.backend_label}] produced an empty mesh")
                    return False

                self.physics_mesh = mesh
                self.version += 1
                if self.unified_mesh:
                    self._publish_to_visual(mesh)
            dt_ms = (time.time() - t0) * 1000.0
            label = self.backend_label
            if self.unified_mesh:
                gs.logger.info(
                    f"Surface mesh [{label}] (visual + physics): "
                    f"{len(mesh.vertices)} verts, {len(mesh.faces)} faces ({dt_ms:.0f} ms)"
                )
            else:
                gs.logger.info(
                    f"Physics mesh [{label}]: "
                    f"{len(mesh.vertices)} verts, {len(mesh.faces)} faces ({dt_ms:.0f} ms)"
                )
            return True
        except Exception as e:
            gs.logger.warning(f"Physics mesh rebuild failed ({self.backend.value}): {e}")
            import traceback

            traceback.print_exc()
            return False

    def get_verts_faces(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        mesh = self.physics_mesh
        if mesh is None or len(mesh.vertices) < 4 or len(mesh.faces) < 4:
            return None, None
        return (
            np.asarray(mesh.vertices, dtype=np.float64),
            np.asarray(mesh.faces, dtype=np.int32),
        )
