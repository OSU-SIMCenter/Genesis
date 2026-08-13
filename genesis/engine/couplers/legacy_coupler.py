from typing import TYPE_CHECKING

import numpy as np
import quadrants as qd

import genesis as gs
import genesis.utils.sdf as sdf

from genesis.options.solvers import LegacyCouplerOptions
from genesis.repr_base import RBC
from genesis.utils import array_class
from genesis.utils.array_class import LinksState
from genesis.utils.geom import qd_inv_transform_by_trans_quat, qd_transform_by_trans_quat

if TYPE_CHECKING:
    from genesis.engine.simulator import Simulator

CLAMPED_INV_DT = 50.0


@qd.data_oriented
class LegacyCoupler(RBC):
    """
    This class handles all the coupling between different solvers. LegacyCoupler will be deprecated in the future.
    """

    # ------------------------------------------------------------------------------------
    # --------------------------------- Initialization -----------------------------------
    # ------------------------------------------------------------------------------------

    def __init__(
        self,
        simulator: "Simulator",
        options: "LegacyCouplerOptions",
    ) -> None:
        self.sim = simulator
        self.options = options

        self.tool_solver = self.sim.tool_solver
        self.rigid_solver = self.sim.rigid_solver
        self.mpm_solver = self.sim.mpm_solver
        self.sph_solver = self.sim.sph_solver
        self.pbd_solver = self.sim.pbd_solver
        self.fem_solver = self.sim.fem_solver
        self.sf_solver = self.sim.sf_solver

    # Contact-mode capability flags. Each enables exactly one contact operation at one point
    # in the substep, so every kernel needs a single qd.static guard and the modes are mutually
    # exclusive by construction.
    _CONTACT_MODES = (
        "grid",
        "particle",
        "fluidlab",
        "postg2p_velocity",
        "postg2p_position",
        "penalty",
        # No coupler-level contact at all. Every gate below is an equality test against a
        # named mode, so 'none' switches them all off without touching any other mode.
        # Exists to isolate base_mpm_solver's custom apply_particle_contact teleport, which
        # has otherwise only ever run underneath grid contact (confound #1).
        "none",
    )

    # Integer ids for the contact modes, used by the runtime-switchable path (see the
    # ``rigid_mpm_contact_runtime_switchable`` option). When that path is active the kernels read
    # the active mode from a small runtime field instead of a ``qd.static`` flag, so the mode can
    # be changed between runs without rebuilding the scene. Plain Python ints so they constant-fold
    # where referenced inside kernels.
    CM_GRID = 0
    CM_PARTICLE = 1
    CM_FLUIDLAB = 2
    CM_POSTG2P_VEL = 3
    CM_POSTG2P_POS = 4
    CM_PENALTY = 5
    CM_NONE = 6

    _CONTACT_MODE_TO_ID = {
        "grid": CM_GRID,
        "particle": CM_PARTICLE,
        "fluidlab": CM_FLUIDLAB,
        "postg2p_velocity": CM_POSTG2P_VEL,
        "postg2p_position": CM_POSTG2P_POS,
        "penalty": CM_PENALTY,
        "none": CM_NONE,
    }
    _CONTACT_ID_TO_MODE = {v: k for k, v in _CONTACT_MODE_TO_ID.items()}

    def build(self) -> None:
        self._rigid_mpm = self.rigid_solver.is_active and self.mpm_solver.is_active and self.options.rigid_mpm

        _mode = getattr(self.options, "rigid_mpm_contact_mode", "grid")
        if _mode not in self._CONTACT_MODES:
            gs.raise_exception(
                f"rigid_mpm_contact_mode={_mode!r} is not one of {self._CONTACT_MODES}."
            )
        self._contact_mode = _mode

        # Runtime-switchable contact: compile every contact path into the kernels and gate them on
        # small runtime fields so the mode / refinements / teleport can be changed live. Opt-in and
        # default OFF -- when off every gate collapses back to its original `qd.static` flag at
        # trace time, so kernel specialization (and therefore previously banked results) is
        # unchanged.
        self._runtime_switchable = bool(
            self._rigid_mpm and getattr(self.options, "rigid_mpm_contact_runtime_switchable", False)
        )

        # base_mpm_solver's teleport reads its gates from the same runtime fields, so the two must
        # agree about whether the switchable path is active -- otherwise that kernel would
        # specialize statically while the coupler believed it could still switch, and the mismatch
        # would show up as an arm silently running the wrong configuration. Fail loudly instead.
        _solver_switchable = bool(getattr(self.mpm_solver, "_pc_switchable", False))
        if self._rigid_mpm and _solver_switchable != self._runtime_switchable:
            gs.raise_exception(
                f"runtime-switchable contact configured inconsistently: coupler={self._runtime_switchable}, "
                f"mpm_solver={_solver_switchable}. Both derive from AGF_CONTACT_RUNTIME_SWITCH; "
                "set it once for the whole process."
            )

        self._contact_gridop_vel = self._rigid_mpm and _mode == "grid"
        self._contact_ing2p_vel = self._rigid_mpm and _mode == "particle"
        self._contact_ing2p_fluidlab = self._rigid_mpm and _mode == "fluidlab"
        self._contact_postg2p_vel = self._rigid_mpm and _mode == "postg2p_velocity"
        self._contact_postg2p_pos = self._rigid_mpm and _mode == "postg2p_position"
        self._contact_postg2p = self._contact_postg2p_vel or self._contact_postg2p_pos
        self._contact_penalty = self._rigid_mpm and _mode == "penalty"

        # HYBRID: these refinements were originally gated to mode=='particle'. They are now
        # allowed on top of 'grid' too, so the grid projection (which is what keeps this problem
        # stable) can be combined with a correction that makes F see the contact compression.
        _want_per_node = getattr(self.options, "rigid_mpm_contact_per_node", False)
        _want_c_inj = getattr(self.options, "rigid_mpm_contact_c_injection", False)
        if (_want_per_node or _want_c_inj) and not self._runtime_switchable and _mode not in ("particle", "grid"):
            gs.raise_exception(
                "rigid_mpm_contact_per_node / rigid_mpm_contact_c_injection require "
                "rigid_mpm_contact_mode in ('particle', 'grid')."
            )
        self._contact_per_node = self._rigid_mpm and _want_per_node
        self._contact_c_injection = self._rigid_mpm and _want_c_inj

        # F_tmp projection. genesis-dev gates this to mode=='particle'; here it is allowed on
        # 'grid' too, which is the point -- grid supplies the force balance and two-way
        # coupling, ftmp supplies sub-grid contact compression AS STRESS.
        _want_ftmp = getattr(self.options, "rigid_mpm_contact_ftmp_projection", False)
        if _want_ftmp and not self._runtime_switchable and _mode not in ("particle", "grid"):
            gs.raise_exception(
                "rigid_mpm_contact_ftmp_projection requires "
                "rigid_mpm_contact_mode in ('particle', 'grid')."
            )
        self._contact_ftmp_proj = self._rigid_mpm and _want_ftmp

        # Runtime fields backing the switchable knobs. Always allocated (0-D, negligible) so the
        # kernels' `qd.static(self._runtime_switchable)` else-branches never reference a missing
        # attribute; they are only *read* by kernels when _runtime_switchable is True.
        self._rt_contact_mode = qd.field(gs.qd_int, shape=())
        # Independent grid-contact floor. Modes are mutually exclusive, so without this a particle
        # mode runs with NO grid contact at all and nothing stops deep penetration before the
        # particle correction fires. Set it to run grid + a particle mode together -- which is what
        # FluidLab's own repo offered as its hybrid. Default OFF: flag-off must be bit-identical.
        self._rt_grid_floor = qd.field(gs.qd_int, shape=())
        self._rt_per_node = qd.field(gs.qd_int, shape=())
        self._rt_c_injection = qd.field(gs.qd_int, shape=())
        self._rt_ftmp_proj = qd.field(gs.qd_int, shape=())
        self._rt_penalty_k = qd.field(gs.qd_float, shape=())
        # Teleport gates. Not present upstream -- apply_particle_contact is fork-only -- but
        # grid-vs-grid+teleport is the central comparison axis, so it has to be switchable too.
        self._rt_pc_mech = qd.field(gs.qd_int, shape=())
        self._rt_pc_c_project = qd.field(gs.qd_int, shape=())
        self._rt_pc_f_feedback = qd.field(gs.qd_int, shape=())
        self._rt_pc_c_damp = qd.field(gs.qd_float, shape=())

        self._rt_contact_mode[None] = self._CONTACT_MODE_TO_ID[_mode]
        self._rt_grid_floor[None] = 0
        self._rt_per_node[None] = int(self._contact_per_node)
        self._rt_c_injection[None] = int(self._contact_c_injection)
        self._rt_ftmp_proj[None] = int(self._contact_ftmp_proj)
        self._rt_penalty_k[None] = float(getattr(self.options, "rigid_mpm_penalty_stiffness", 0.0))
        self._rt_pc_mech[None] = int(getattr(self.mpm_solver, "_pc_mech", False))
        self._rt_pc_c_project[None] = int(getattr(self.mpm_solver, "_pc_c_project", False))
        self._rt_pc_f_feedback[None] = int(getattr(self.mpm_solver, "_pc_f_feedback", False))
        self._rt_pc_c_damp[None] = float(getattr(self.mpm_solver, "_pc_c_damp", 1.0))

        # Penetration probe accumulators (see mpm_penetration_probe). Always allocated so the
        # Python-side read/reset never has to test for their existence.
        self._pen_probe = bool(
            self._rigid_mpm and getattr(self.options, "rigid_mpm_penetration_probe", False)
        )
        self._pen_max = qd.field(gs.qd_float, shape=())
        self._pen_sum = qd.field(gs.qd_float, shape=())
        self._pen_count = qd.field(gs.qd_int, shape=())
        self._pen_active = qd.field(gs.qd_int, shape=())
        self.penetration_reset()

        # The in-g2p modes and CPIC both resolve contact inside g2p and would double-count. When
        # switching is on, any in-g2p mode may be selected later, so CPIC is forbidden outright.
        if (
            self._contact_ing2p_vel or self._contact_ing2p_fluidlab or self._runtime_switchable
        ) and self.mpm_solver.enable_CPIC:
            gs.raise_exception(
                f"rigid_mpm_contact_mode={_mode!r} is incompatible with enable_CPIC "
                "(both resolve contact in g2p). Disable CPIC to use this mode."
            )
        self._rigid_sph = self.rigid_solver.is_active and self.sph_solver.is_active and self.options.rigid_sph
        self._rigid_pbd = self.rigid_solver.is_active and self.pbd_solver.is_active and self.options.rigid_pbd
        self._rigid_fem = self.rigid_solver.is_active and self.fem_solver.is_active and self.options.rigid_fem
        self._mpm_sph = self.mpm_solver.is_active and self.sph_solver.is_active and self.options.mpm_sph
        self._mpm_pbd = self.mpm_solver.is_active and self.pbd_solver.is_active and self.options.mpm_pbd
        self._fem_mpm = self.fem_solver.is_active and self.mpm_solver.is_active and self.options.fem_mpm
        self._fem_sph = self.fem_solver.is_active and self.sph_solver.is_active and self.options.fem_sph

        if (self._rigid_mpm or self._rigid_sph or self._rigid_pbd or self._rigid_fem) and any(
            geom.needs_coup for geom in self.rigid_solver.geoms
        ):
            self.rigid_solver.collider._sdf.activate()

        if self._rigid_mpm and self.mpm_solver.enable_CPIC:
            # this field stores the geom index of the thin shell rigid object (if any) that separates particle and its surrounding grid cell
            self.cpic_flag = qd.field(gs.qd_int, shape=(self.mpm_solver.n_particles, 3, 3, 3, self.mpm_solver._B))
            self.mpm_rigid_normal = qd.Vector.field(
                3,
                dtype=gs.qd_float,
                shape=(self.mpm_solver.n_particles, self.rigid_solver.n_geoms_, self.mpm_solver._B),
            )

        if self._rigid_sph:
            self.sph_rigid_normal = qd.Vector.field(
                3,
                dtype=gs.qd_float,
                shape=(self.sph_solver.n_particles, self.rigid_solver.n_geoms_, self.sph_solver._B),
            )
            self.sph_rigid_normal_reordered = qd.Vector.field(
                3,
                dtype=gs.qd_float,
                shape=(self.sph_solver.n_particles, self.rigid_solver.n_geoms_, self.sph_solver._B),
            )

        if self._rigid_pbd:
            self.pbd_rigid_normal_reordered = qd.Vector.field(
                3, dtype=gs.qd_float, shape=(self.pbd_solver.n_particles, self.pbd_solver._B, self.rigid_solver.n_geoms)
            )

            struct_particle_attach_info = qd.types.struct(
                link_idx=gs.qd_int,
                local_pos=gs.qd_vec3,
            )

            self.particle_attach_info = struct_particle_attach_info.field(
                shape=(self.pbd_solver._n_particles, self.pbd_solver._B), layout=qd.Layout.SOA
            )
            self.particle_attach_info.link_idx.fill(-1)
            self.particle_attach_info.local_pos.fill(0.0)

        if self._mpm_sph:
            self.mpm_sph_stencil_size = int(np.floor(self.mpm_solver.dx / self.sph_solver.hash_grid_cell_size) + 2)

        if self._mpm_pbd:
            self.mpm_pbd_stencil_size = int(np.floor(self.mpm_solver.dx / self.pbd_solver.hash_grid_cell_size) + 2)

        ## DEBUG
        self._dx = 1 / 1024
        self._stencil_size = int(np.floor(self._dx / self.sph_solver.hash_grid_cell_size) + 2)

        self.reset(envs_idx=self.sim.scene._envs_idx)

    def reset(self, envs_idx=None) -> None:
        if self._rigid_mpm and self.mpm_solver.enable_CPIC:
            if envs_idx is None:
                self.mpm_rigid_normal.fill(0)
            else:
                self._kernel_reset_mpm(envs_idx)
        
        if self._rigid_sph:
            if envs_idx is None:
                self.sph_rigid_normal.fill(0)
            else:
                self._kernel_reset_sph(envs_idx)

    @qd.kernel
    def _kernel_reset_mpm(self, envs_idx: qd.types.ndarray()):
        for i_p, i_g, i_b_ in qd.ndrange(self.mpm_solver.n_particles, self.rigid_solver.n_geoms, envs_idx.shape[0]):
            self.mpm_rigid_normal[i_p, i_g, envs_idx[i_b_]] = 0.0

    @qd.kernel
    def _kernel_reset_sph(self, envs_idx: qd.types.ndarray()):
        for i_p, i_g, i_b_ in qd.ndrange(self.sph_solver.n_particles, self.rigid_solver.n_geoms, envs_idx.shape[0]):
            self.sph_rigid_normal[i_p, i_g, envs_idx[i_b_]] = 0.0

    @qd.func
    def _func_collide_with_rigid(
        self,
        f,
        pos_world,
        vel,
        mass,
        i_b,
        geoms_state: array_class.GeomsState,
        geoms_info: array_class.GeomsInfo,
        links_state: array_class.LinksState,
        rigid_global_info: array_class.RigidGlobalInfo,
        sdf_info: array_class.SDFInfo,
        collider_static_config: qd.template(),
    ):
        for i_g in range(self.rigid_solver.n_geoms):
            if geoms_info.needs_coup[i_g]:
                vel = self._func_collide_with_rigid_geom(
                    pos_world,
                    vel,
                    mass,
                    i_g,
                    i_b,
                    geoms_state=geoms_state,
                    geoms_info=geoms_info,
                    links_state=links_state,
                    rigid_global_info=rigid_global_info,
                    sdf_info=sdf_info,
                    collider_static_config=collider_static_config,
                )
        return vel

    @qd.func
    def _func_collide_with_rigid_geom(
        self,
        pos_world,
        vel,
        mass,
        geom_idx,
        batch_idx,
        geoms_state: array_class.GeomsState,
        geoms_info: array_class.GeomsInfo,
        links_state: array_class.LinksState,
        rigid_global_info: array_class.RigidGlobalInfo,
        sdf_info: array_class.SDFInfo,
        collider_static_config: qd.template(),
    ):
        signed_dist = sdf.sdf_func_world(
            geoms_state=geoms_state,
            geoms_info=geoms_info,
            sdf_info=sdf_info,
            pos_world=pos_world,
            geom_idx=geom_idx,
            batch_idx=batch_idx,
        )

        # bigger coup_softness implies that the coupling influence extends further away from the object.
        influence = qd.min(qd.exp(-signed_dist / max(1e-10, geoms_info.coup_softness[geom_idx])), 1)

        if influence > 0.1:
            normal_rigid = sdf.sdf_func_normal_world(
                geoms_state=geoms_state,
                geoms_info=geoms_info,
                rigid_global_info=rigid_global_info,
                collider_static_config=collider_static_config,
                sdf_info=sdf_info,
                pos_world=pos_world,
                geom_idx=geom_idx,
                batch_idx=batch_idx,
            )
            vel = self._func_collide_in_rigid_geom(
                pos_world,
                vel,
                mass,
                normal_rigid,
                influence,
                geom_idx,
                batch_idx,
                geoms_info,
                links_state,
                rigid_global_info,
            )

        return vel

    @qd.func
    def _func_collide_with_rigid_geom_robust(
        self,
        pos_world,
        vel,
        mass,
        normal_prev,
        geom_idx,
        batch_idx,
        geoms_state: array_class.GeomsState,
        geoms_info: array_class.GeomsInfo,
        links_state: array_class.LinksState,
        rigid_global_info: array_class.RigidGlobalInfo,
        sdf_info: array_class.SDFInfo,
        collider_static_config: qd.template(),
    ):
        """
        Similar to _func_collide_with_rigid_geom, but additionally handles potential side flip due to penetration.
        """
        signed_dist = sdf.sdf_func_world(
            geoms_state=geoms_state,
            geoms_info=geoms_info,
            sdf_info=sdf_info,
            pos_world=pos_world,
            geom_idx=geom_idx,
            batch_idx=batch_idx,
        )
        normal_rigid = sdf.sdf_func_normal_world(
            geoms_state=geoms_state,
            geoms_info=geoms_info,
            rigid_global_info=rigid_global_info,
            collider_static_config=collider_static_config,
            sdf_info=sdf_info,
            pos_world=pos_world,
            geom_idx=geom_idx,
            batch_idx=batch_idx,
        )

        # bigger coup_softness implies that the coupling influence extends further away from the object.
        influence = qd.min(qd.exp(-signed_dist / max(1e-10, geoms_info.coup_softness[geom_idx])), 1)

        # if normal_rigid.dot(normal_prev) < 0: # side flip due to penetration
        #     influence = 1.0
        #     normal_rigid = normal_prev
        if influence > 0.1:
            vel = self._func_collide_in_rigid_geom(
                pos_world,
                vel,
                mass,
                normal_rigid,
                influence,
                geom_idx,
                batch_idx,
                geoms_info,
                links_state,
                rigid_global_info,
            )

        # attraction force
        # if 0.001 < signed_dist < 0.01:
        #     vel = vel - normal_rigid * 0.1 * signed_dist

        return vel, normal_rigid

    @qd.func
    def _func_collide_in_rigid_geom(
        self,
        pos_world,
        vel,
        mass,
        normal_rigid,
        influence,
        geom_idx,
        i_b,
        geoms_info: array_class.GeomsInfo,
        links_state: array_class.LinksState,
        rigid_global_info: array_class.RigidGlobalInfo,
    ):
        """
        Resolves collision when a particle is already in collision with a rigid object.
        This function assumes known normal_rigid and influence.
        """
        vel_rigid = self.rigid_solver._func_vel_at_point(
            pos_world=pos_world,
            link_idx=geoms_info.link_idx[geom_idx],
            i_b=i_b,
            links_state=links_state,
        )

        # v w.r.t rigid
        rvel = vel - vel_rigid
        rvel_normal_magnitude = rvel.dot(normal_rigid)  # negative if inward

        if rvel_normal_magnitude < 0:  # colliding
            #################### rigid -> particle ####################
            # tangential component
            rvel_tan = rvel - rvel_normal_magnitude * normal_rigid
            rvel_tan_norm = rvel_tan.norm(gs.EPS)

            # tangential component after friction
            rvel_tan = (
                rvel_tan
                / rvel_tan_norm
                * qd.max(0, rvel_tan_norm + rvel_normal_magnitude * geoms_info.coup_friction[geom_idx])
            )

            # normal component after collision
            rvel_normal = -normal_rigid * rvel_normal_magnitude * geoms_info.coup_restitution[geom_idx]

            # normal + tangential component
            rvel_new = rvel_tan + rvel_normal

            # apply influence
            vel_old = vel
            vel = vel_rigid + rvel_new * influence + rvel * (1 - influence)

            #################### particle -> rigid ####################
            # Compute delta momentum and apply to rigid body.
            delta_mv = mass * (vel - vel_old)
            force = -delta_mv / rigid_global_info.substep_dt[None]
            self.rigid_solver._func_apply_coupling_force(
                pos_world,
                force,
                geoms_info.link_idx[geom_idx],
                i_b,
                links_state,
            )

        return vel

    @qd.func
    def _func_mpm_tool(self, f, pos_world, vel, i_b):
        for entity in qd.static(self.tool_solver.entities):
            if qd.static(entity.material.collision):
                vel = entity.collide(f, pos_world, vel, i_b)
        return vel

    @qd.func
    def _func_fluidlab_collide(
        self,
        pos_world,
        vel,
        geom_idx,
        i_b,
        geoms_state: array_class.GeomsState,
        geoms_info: array_class.GeomsInfo,
        links_state: array_class.LinksState,
        rigid_global_info: array_class.RigidGlobalInfo,
        sdf_info: array_class.SDFInfo,
        collider_static_config: qd.template(),
    ):
        """
        FluidLab-style particle collision: one-way, velocity-only, restitution-free, evaluated at
        the (predicted) particle position. Two deliberate differences from
        ``_func_collide_in_rigid_geom``: no reaction force is applied to the rigid body, and when
        called from g2p it runs *after* the affine field is formed, so it does not feed ``C``.
        """
        signed_dist = sdf.sdf_func_world(
            geoms_state=geoms_state,
            geoms_info=geoms_info,
            sdf_info=sdf_info,
            pos_world=pos_world,
            geom_idx=geom_idx,
            batch_idx=i_b,
        )
        influence = qd.min(qd.exp(-signed_dist / max(1e-10, geoms_info.coup_softness[geom_idx])), 1)
        if signed_dist <= 0 or influence > 0.1:
            normal_rigid = sdf.sdf_func_normal_world(
                geoms_state=geoms_state,
                geoms_info=geoms_info,
                rigid_global_info=rigid_global_info,
                collider_static_config=collider_static_config,
                sdf_info=sdf_info,
                pos_world=pos_world,
                geom_idx=geom_idx,
                batch_idx=i_b,
            )
            vel_rigid = self.rigid_solver._func_vel_at_point(
                pos_world=pos_world,
                link_idx=geoms_info.link_idx[geom_idx],
                i_b=i_b,
                links_state=links_state,
            )
            rvel = vel - vel_rigid
            rvel_normal_magnitude = rvel.dot(normal_rigid)

            # tangential component (inward normal velocity removed; restitution implicitly 0)
            rvel_tan = rvel - qd.min(rvel_normal_magnitude, 0.0) * normal_rigid
            rvel_tan_norm = rvel_tan.norm(gs.EPS)
            rvel_tan_friction = (
                rvel_tan
                / rvel_tan_norm
                * qd.max(0.0, rvel_tan_norm + rvel_normal_magnitude * geoms_info.coup_friction[geom_idx])
            )

            # friction only when approaching AND there is tangential motion (FluidLab's flag blend)
            flag = qd.cast(rvel_normal_magnitude < 0 and rvel_tan_norm > gs.EPS, gs.qd_float)
            rvel_tan = rvel_tan_friction * flag + rvel_tan * (1.0 - flag)

            vel = vel_rigid + rvel_tan * influence + rvel * (1 - influence)

        return vel

    @qd.kernel
    def mpm_postg2p_contact(
        self,
        f: qd.i32,
        geoms_state: array_class.GeomsState,
        geoms_info: array_class.GeomsInfo,
        links_state: array_class.LinksState,
        rigid_global_info: array_class.RigidGlobalInfo,
        sdf_info: array_class.SDFInfo,
        collider_static_config: qd.template(),
    ):
        """
        Post-g2p particle-level rigid contact on the advected frame ``f+1``
        (``postg2p_velocity`` / ``postg2p_position``). ``postg2p_position`` additionally
        hard-projects penetrating particles out to a ``particle_size/2`` margin.

        NOTE: because this runs after the affine field is formed, neither mode feeds the
        deformation gradient this substep. That is expected to show up directly as a larger
        det-F-vs-packing gap, and is the reason both modes are in the comparison.
        """
        for i_p, i_b in qd.ndrange(self.mpm_solver.n_particles, self.mpm_solver._B):
            # Which post-g2p sub-mode is active, selected at trace time: the compile-time
            # `qd.static` flag in a normal build (so the unused branch is eliminated), a runtime
            # field read when the contact mode is switchable.
            do_pg_pos = (
                (self._rt_contact_mode[None] == self.CM_POSTG2P_POS)
                if qd.static(self._runtime_switchable)
                else qd.static(self._contact_postg2p_pos)
            )
            do_pg_vel = (
                (self._rt_contact_mode[None] == self.CM_POSTG2P_VEL)
                if qd.static(self._runtime_switchable)
                else qd.static(self._contact_postg2p_vel)
            )
            if self.mpm_solver.particles_ng[f + 1, i_p, i_b].active:
                pos = self.mpm_solver.particles[f + 1, i_p, i_b].pos
                vel = self.mpm_solver.particles[f + 1, i_p, i_b].vel
                mass = self.mpm_solver.particles_info[i_p].mass / self.mpm_solver._particle_volume_scale

                for i_g in range(self.rigid_solver.n_geoms):
                    if geoms_info.needs_coup[i_g]:
                        signed_dist = sdf.sdf_func_world(
                            geoms_state=geoms_state,
                            geoms_info=geoms_info,
                            sdf_info=sdf_info,
                            pos_world=pos,
                            geom_idx=i_g,
                            batch_idx=i_b,
                        )

                        if do_pg_pos:
                            margin = self.mpm_solver._particle_size * 0.5
                            if signed_dist < margin:
                                normal_rigid = sdf.sdf_func_normal_world(
                                    geoms_state=geoms_state,
                                    geoms_info=geoms_info,
                                    rigid_global_info=rigid_global_info,
                                    collider_static_config=collider_static_config,
                                    sdf_info=sdf_info,
                                    pos_world=pos,
                                    geom_idx=i_g,
                                    batch_idx=i_b,
                                )
                                pos = pos - (signed_dist - margin) * normal_rigid
                                self.mpm_solver.particles[f + 1, i_p, i_b].pos = pos
                                vel = self._func_collide_in_rigid_geom(
                                    pos,
                                    vel,
                                    mass,
                                    normal_rigid,
                                    1.0,
                                    i_g,
                                    i_b,
                                    geoms_info,
                                    links_state,
                                    rigid_global_info,
                                )
                                self.mpm_solver.particles[f + 1, i_p, i_b].vel = vel

                        if do_pg_vel:
                            influence = qd.min(
                                qd.exp(-signed_dist / max(1e-10, geoms_info.coup_softness[i_g])), 1
                            )
                            if influence > 0.1:
                                normal_rigid = sdf.sdf_func_normal_world(
                                    geoms_state=geoms_state,
                                    geoms_info=geoms_info,
                                    rigid_global_info=rigid_global_info,
                                    collider_static_config=collider_static_config,
                                    sdf_info=sdf_info,
                                    pos_world=pos,
                                    geom_idx=i_g,
                                    batch_idx=i_b,
                                )
                                vel = self._func_collide_in_rigid_geom(
                                    pos,
                                    vel,
                                    mass,
                                    normal_rigid,
                                    influence,
                                    i_g,
                                    i_b,
                                    geoms_info,
                                    links_state,
                                    rigid_global_info,
                                )
                                self.mpm_solver.particles[f + 1, i_p, i_b].vel = vel

    @qd.kernel
    def mpm_penalty_contact(
        self,
        f: qd.i32,
        geoms_state: array_class.GeomsState,
        geoms_info: array_class.GeomsInfo,
        links_state: array_class.LinksState,
        rigid_global_info: array_class.RigidGlobalInfo,
        sdf_info: array_class.SDFInfo,
        collider_static_config: qd.template(),
    ):
        """
        Penalty-force rigid-MPM contact. Each penetrating particle gets a spring force
        ``f = k * penetration * normal`` scattered onto the background grid momentum, so it
        enters the same force balance as the elastic stress and therefore reaches the affine
        field and the deformation gradient through the subsequent g2p. Runs in ``couple`` before
        ``mpm_grid_op``, which normalizes the grid momentum.

        KNOWN LIMITATIONS (this is a comparison arm, not a production contact model):
          * purely normal -- no Coulomb friction, unlike the grid path. Do not read a
            grid-vs-penalty difference as purely a contact-mechanism effect.
          * undamped -- a pure spring, so it can chatter and inject energy.
          * contact is detected from the particle CENTRE, so a particle can be half inside the
            die before any force appears.
        """
        for i_p, i_b in qd.ndrange(self.mpm_solver.n_particles, self.mpm_solver._B):
            if self.mpm_solver.particles_ng[f, i_p, i_b].active:
                # Spring stiffness, selected at trace time: a compile-time constant from options
                # in a normal build, a runtime field read when the contact mode is switchable.
                k = (
                    self._rt_penalty_k[None]
                    if qd.static(self._runtime_switchable)
                    else self.options.rigid_mpm_penalty_stiffness
                )
                pos = self.mpm_solver.particles[f, i_p, i_b].pos
                for i_g in range(self.rigid_solver.n_geoms):
                    if geoms_info.needs_coup[i_g]:
                        signed_dist = sdf.sdf_func_world(
                            geoms_state=geoms_state,
                            geoms_info=geoms_info,
                            sdf_info=sdf_info,
                            pos_world=pos,
                            geom_idx=i_g,
                            batch_idx=i_b,
                        )
                        if signed_dist < 0:
                            normal_rigid = sdf.sdf_func_normal_world(
                                geoms_state=geoms_state,
                                geoms_info=geoms_info,
                                rigid_global_info=rigid_global_info,
                                collider_static_config=collider_static_config,
                                sdf_info=sdf_info,
                                pos_world=pos,
                                geom_idx=i_g,
                                batch_idx=i_b,
                            )
                            # spring + dashpot. The dashpot resists APPROACH only, so it cannot
                            # pull a separating particle back into the die.
                            m_p = (
                                self.mpm_solver.particles_info[i_p].mass
                                / self.mpm_solver._particle_volume_scale
                            )
                            vel_rigid = self.rigid_solver._func_vel_at_point(
                                pos_world=pos,
                                link_idx=geoms_info.link_idx[i_g],
                                i_b=i_b,
                                links_state=links_state,
                            )
                            v_approach = -(
                                self.mpm_solver.particles[f, i_p, i_b].vel - vel_rigid
                            ).dot(normal_rigid)
                            c_damp = (
                                2.0
                                * self.options.rigid_mpm_penalty_damping
                                * qd.sqrt(qd.max(k * m_p, 0.0))
                            )
                            force = (
                                k * (-signed_dist) + c_damp * qd.max(v_approach, 0.0)
                            ) * normal_rigid
                            self.rigid_solver._func_apply_coupling_force(
                                pos,
                                -force,
                                geoms_info.link_idx[i_g],
                                i_b,
                                links_state,
                            )
                            impulse = (
                                self.mpm_solver._particle_volume_scale * force * self.mpm_solver.substep_dt
                            )
                            base = qd.floor(pos * self.mpm_solver._inv_dx - 0.5).cast(gs.qd_int)
                            fx = pos * self.mpm_solver._inv_dx - base.cast(gs.qd_float)
                            w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
                            for offset in qd.static(qd.grouped(self.mpm_solver.stencil_range())):
                                weight = gs.qd_float(1.0)
                                for d in qd.static(range(3)):
                                    weight *= w[offset[d]][d]
                                self.mpm_solver.grid[
                                    f, base - self.mpm_solver._grid_offset + offset, i_b
                                ].vel_in += (weight * impulse)

    @qd.kernel
    def mpm_ftmp_contact_projection(
        self,
        f: qd.i32,
        geoms_state: array_class.GeomsState,
        geoms_info: array_class.GeomsInfo,
        rigid_global_info: array_class.RigidGlobalInfo,
        sdf_info: array_class.SDFInfo,
        collider_static_config: qd.template(),
    ):
        """
        Inject the contact compression into the *trial* deformation gradient ``F_tmp`` --
        after ``compute_F_tmp`` and before the SVD/stress step -- so the constitutive model,
        including the plastic return mapping, processes the contact-induced deformation and
        the resulting stress reaches the grid through p2g in the same substep.

        A penetrating particle's ``F_tmp`` is compressed along the surface normal by a strain
        proportional to penetration depth over one cell, capped at 0.5 so a deep penetrator
        cannot drive det(F) <= 0 in a single substep.

        This kernel applies NO force to the rigid body itself; it relies on the mode it is
        layered on for two-way coupling (grid mode supplies it via
        ``_func_collide_in_rigid_geom``).
        """
        for i_p, i_b in qd.ndrange(self.mpm_solver.n_particles, self.mpm_solver._B):
            if self.mpm_solver.particles_ng[f, i_p, i_b].active:
                pos = self.mpm_solver.particles[f, i_p, i_b].pos
                for i_g in range(self.rigid_solver.n_geoms):
                    if geoms_info.needs_coup[i_g]:
                        signed_dist = sdf.sdf_func_world(
                            geoms_state=geoms_state,
                            geoms_info=geoms_info,
                            sdf_info=sdf_info,
                            pos_world=pos,
                            geom_idx=i_g,
                            batch_idx=i_b,
                        )
                        if signed_dist < 0:  # penetrating
                            normal_rigid = sdf.sdf_func_normal_world(
                                geoms_state=geoms_state,
                                geoms_info=geoms_info,
                                rigid_global_info=rigid_global_info,
                                collider_static_config=collider_static_config,
                                sdf_info=sdf_info,
                                pos_world=pos,
                                geom_idx=i_g,
                                batch_idx=i_b,
                            )
                            eps = qd.min(-signed_dist * self.mpm_solver._inv_dx, 0.5)
                            proj = qd.Matrix.identity(gs.qd_float, 3) - eps * normal_rigid.outer_product(
                                normal_rigid
                            )
                            self.mpm_solver.particles[f, i_p, i_b].F_tmp = (
                                proj @ self.mpm_solver.particles[f, i_p, i_b].F_tmp
                            )

    @qd.kernel
    def mpm_grid_op(
        self,
        f: qd.i32,
        t: qd.f32,
        geoms_state: array_class.GeomsState,
        geoms_info: array_class.GeomsInfo,
        links_state: array_class.LinksState,
        rigid_global_info: array_class.RigidGlobalInfo,
        sdf_info: array_class.SDFInfo,
        collider_static_config: qd.template(),
    ):
        for ii, jj, kk, i_b in qd.ndrange(*self.mpm_solver.grid_res, self.mpm_solver._B):
            I = (ii, jj, kk)
            if self.mpm_solver.grid[f, I, i_b].mass > gs.EPS:
                #################### MPM grid op ####################
                # Momentum to velocity
                vel_mpm = (1 / self.mpm_solver.grid[f, I, i_b].mass) * self.mpm_solver.grid[f, I, i_b].vel_in

                # Thermal: normalize mass-weighted temperature
                if qd.static(self.mpm_solver._enable_thermal):
                    if self.mpm_solver.grid[f, I, i_b].mass_thermal > 0:
                        self.mpm_solver.grid[f, I, i_b].temp = (
                            self.mpm_solver.grid[f, I, i_b].temp
                            / self.mpm_solver.grid[f, I, i_b].mass_thermal
                        )
                        
                        # --- Air Cooling (Convection) ---
                        is_surface = 0
                        I_left = I + qd.Vector([-1, 0, 0])
                        I_right = I + qd.Vector([1, 0, 0])
                        I_down = I + qd.Vector([0, -1, 0])
                        I_up = I + qd.Vector([0, 1, 0])
                        I_back = I + qd.Vector([0, 0, -1])
                        I_front = I + qd.Vector([0, 0, 1])
                        
                        if I_left[0] < 0:
                            is_surface = 1
                        elif self.mpm_solver.grid[f, I_left, i_b].mass_thermal < gs.EPS:
                            is_surface = 1

                        if I_right[0] >= self.mpm_solver.grid_res[0]:
                            is_surface = 1
                        elif self.mpm_solver.grid[f, I_right, i_b].mass_thermal < gs.EPS:
                            is_surface = 1

                        if I_down[1] < 0:
                            is_surface = 1
                        elif self.mpm_solver.grid[f, I_down, i_b].mass_thermal < gs.EPS:
                            is_surface = 1

                        if I_up[1] >= self.mpm_solver.grid_res[1]:
                            is_surface = 1
                        elif self.mpm_solver.grid[f, I_up, i_b].mass_thermal < gs.EPS:
                            is_surface = 1

                        if I_back[2] < 0:
                            is_surface = 1
                        elif self.mpm_solver.grid[f, I_back, i_b].mass_thermal < gs.EPS:
                            is_surface = 1

                        if I_front[2] >= self.mpm_solver.grid_res[2]:
                            is_surface = 1
                        elif self.mpm_solver.grid[f, I_front, i_b].mass_thermal < gs.EPS:
                            is_surface = 1
                        
                        # --- Fixed-end (cut plane) detection ---
                        # The held end is welded to the unsimulated rod, NOT exposed to air.
                        # A cut-face cell is metal whose +X neighbor is empty and that sits at
                        # the known held-end plane. It drains into the bulk (Robin BC), not air.
                        is_cut_face = 0
                        if qd.static(self.mpm_solver._enable_fixed_end_bc):
                            cell_x = (I + self.mpm_solver.grid_offset)[0] * self.mpm_solver.dx
                            right_empty = 0
                            if I_right[0] >= self.mpm_solver.grid_res[0]:
                                right_empty = 1
                            elif self.mpm_solver.grid[f, I_right, i_b].mass_thermal < gs.EPS:
                                right_empty = 1
                            if right_empty == 1 and cell_x >= (self.mpm_solver._fixed_end_x_cut - self.mpm_solver.dx):
                                is_cut_face = 1

                        if is_cut_face == 1:
                            # --- Fixed-end conduction into bulk rod (Robin BC) ---
                            # Newton-form flux toward the bulk temperature with conductance
                            # h_bulk = k(T) / L_eff. Same stable exponential update as air
                            # convection; replaces (not adds to) air+radiation on this face.
                            T_cell = self.mpm_solver.grid[f, I, i_b].temp
                            T_amb = self.mpm_solver._rt_fixed_end_ambient[None]
                            blend = self.mpm_solver._rt_fixed_end_blend[None]
                            T_end = self.mpm_solver._rt_fixed_end_sink_temp[None]
                            T_bulk = (1.0 - blend) * T_amb + blend * T_end
                            mass_thermal_real = self.mpm_solver.grid[f, I, i_b].mass_thermal / self.mpm_solver._particle_volume_scale
                            Cp = self.mpm_solver.get_steel_cp(T_cell)
                            A_cut = self.mpm_solver.dx ** 2
                            k_local = self.mpm_solver.get_steel_thermal_conductivity(T_cell)
                            h_bulk = k_local / self.mpm_solver._rt_fixed_end_L_eff[None]
                            h_bulk_scaled = h_bulk * self.mpm_solver._rt_thermal_time_scale[None]  # match air pre-scaling
                            k_bulk = (h_bulk_scaled * A_cut) / (mass_thermal_real * Cp)
                            decay_bulk = qd.math.exp(-k_bulk * self.mpm_solver.substep_dt)
                            dT_bulk = (T_cell - T_bulk) * (1.0 - decay_bulk)
                            self.mpm_solver.grid[f, I, i_b].dT_bulk -= dT_bulk
                            self.mpm_solver.grid[f, I, i_b].temp = T_cell - dT_bulk

                        elif is_surface == 1:
                            T_cell = self.mpm_solver.grid[f, I, i_b].temp
                            T_amb = 293.15  # Room temp
                            # N.B. mass_thermal is inflated by _particle_volume_scale; divide it out for real physics
                            mass_thermal_real = self.mpm_solver.grid[f, I, i_b].mass_thermal / self.mpm_solver._particle_volume_scale
                            Cp = self.mpm_solver.get_steel_cp(T_cell)
                            A_cell = self.mpm_solver.dx ** 2  # one cell face area

                            # --- Air Convection: Newton's Law of Cooling ---
                            h_air = self.mpm_solver._rt_h_air[None]  # already scaled by thermal_time_scale
                            k_air = (h_air * A_cell) / (mass_thermal_real * Cp)
                            decay_air = qd.math.exp(-k_air * self.mpm_solver.substep_dt)
                            dT_conv = (T_cell - T_amb) * (1.0 - decay_air)
                            self.mpm_solver.grid[f, I, i_b].dT_conv -= dT_conv

                            T_cell_after_air = T_cell - dT_conv

                            # --- Radiation: Stefan-Boltzmann (linearized for exponential stability) ---
                            # Q_rad = ε·σ·A·(T⁴ - T_amb⁴) ≈ h_rad·A·(T - T_amb)
                            # where h_rad = ε·σ·(T² + T_amb²)·(T + T_amb)
                            emissivity = self.mpm_solver._emissivity
                            sigma = 5.67e-8
                            h_rad = emissivity * sigma * (T_cell_after_air * T_cell_after_air + T_amb * T_amb) * (T_cell_after_air + T_amb)
                            # Scale by thermal_time_scale (h_air is pre-scaled, radiation must be scaled dynamically)
                            h_rad_scaled = h_rad * self.mpm_solver._rt_thermal_time_scale[None]
                            k_rad = (h_rad_scaled * A_cell) / (mass_thermal_real * Cp)
                            decay_rad = qd.math.exp(-k_rad * self.mpm_solver.substep_dt)
                            dT_rad = (T_cell_after_air - T_amb) * (1.0 - decay_rad)
                            self.mpm_solver.grid[f, I, i_b].dT_rad -= dT_rad

                            self.mpm_solver.grid[f, I, i_b].temp = T_cell_after_air - dT_rad
                        


                # gravity
                vel_mpm += self.mpm_solver.substep_dt * self.mpm_solver._gravity[i_b]

                pos = (I + self.mpm_solver.grid_offset) * self.mpm_solver.dx
                mass_mpm = self.mpm_solver.grid[f, I, i_b].mass / self.mpm_solver._particle_volume_scale

                # external force fields
                for i_ff in qd.static(range(len(self.mpm_solver._ffs))):
                    vel_mpm += self.mpm_solver._ffs[i_ff].get_acc(pos, vel_mpm, t, -1) * self.mpm_solver.substep_dt

                #################### MPM <-> Tool ####################
                if qd.static(self.tool_solver.is_active):
                    vel_mpm = self._func_mpm_tool(f, pos, vel_mpm, i_b)

                #################### MPM <-> Rigid ####################
                # Whether grid-level rigid contact is active, selected at trace time: the
                # compile-time `qd.static` flag in a normal build, a runtime field read when the
                # contact mode is switchable.
                do_gridop = (
                    (
                        self._rt_contact_mode[None] == self.CM_GRID
                        or self._rt_grid_floor[None] != 0
                    )
                    if qd.static(self._runtime_switchable)
                    else qd.static(self._contact_gridop_vel)
                )
                if do_gridop:
                    vel_mpm = self._func_collide_with_rigid(
                        f,
                        pos,
                        vel_mpm,
                        mass_mpm,
                        i_b,
                        geoms_state=geoms_state,
                        geoms_info=geoms_info,
                        links_state=links_state,
                        rigid_global_info=rigid_global_info,
                        sdf_info=sdf_info,
                        collider_static_config=collider_static_config,
                    )

                #################### MPM <-> SPH ####################
                if qd.static(self._mpm_sph):
                    # using the lower corner of MPM cell to find the corresponding SPH base cell
                    base = self.sph_solver.sh.pos_to_grid(pos - 0.5 * self.mpm_solver.dx)

                    # ---------- SPH -> MPM ----------
                    sph_vel = qd.Vector([0.0, 0.0, 0.0])
                    colliding_particles = 0
                    for offset in qd.grouped(
                        qd.ndrange(self.mpm_sph_stencil_size, self.mpm_sph_stencil_size, self.mpm_sph_stencil_size)
                    ):
                        slot_idx = self.sph_solver.sh.grid_to_slot(base + offset)
                        for i in range(
                            self.sph_solver.sh.slot_start[slot_idx, i_b],
                            self.sph_solver.sh.slot_start[slot_idx, i_b] + self.sph_solver.sh.slot_size[slot_idx, i_b],
                        ):
                            if (
                                qd.abs(pos - self.sph_solver.particles_reordered.pos[i, i_b]).max()
                                < self.mpm_solver.dx * 0.5
                            ):
                                sph_vel += self.sph_solver.particles_reordered.vel[i, i_b]
                                colliding_particles += 1
                    if colliding_particles > 0:
                        vel_old = vel_mpm
                        vel_mpm = sph_vel / colliding_particles

                        # ---------- MPM -> SPH ----------
                        delta_mv = mass_mpm * (vel_mpm - vel_old)

                        for offset in qd.grouped(
                            qd.ndrange(self.mpm_sph_stencil_size, self.mpm_sph_stencil_size, self.mpm_sph_stencil_size)
                        ):
                            slot_idx = self.sph_solver.sh.grid_to_slot(base + offset)
                            for i in range(
                                self.sph_solver.sh.slot_start[slot_idx, i_b],
                                self.sph_solver.sh.slot_start[slot_idx, i_b]
                                + self.sph_solver.sh.slot_size[slot_idx, i_b],
                            ):
                                if (
                                    qd.abs(pos - self.sph_solver.particles_reordered.pos[i, i_b]).max()
                                    < self.mpm_solver.dx * 0.5
                                ):
                                    self.sph_solver.particles_reordered[i, i_b].vel = (
                                        self.sph_solver.particles_reordered[i, i_b].vel
                                        - delta_mv / self.sph_solver.particles_info_reordered[i, i_b].mass
                                    )

                #################### MPM <-> PBD ####################
                if qd.static(self._mpm_pbd):
                    # using the lower corner of MPM cell to find the corresponding PBD base cell
                    base = self.pbd_solver.sh.pos_to_grid(pos - 0.5 * self.mpm_solver.dx)

                    # ---------- PBD -> MPM ----------
                    pbd_vel = qd.Vector([0.0, 0.0, 0.0])
                    colliding_particles = 0
                    for offset in qd.grouped(
                        qd.ndrange(self.mpm_pbd_stencil_size, self.mpm_pbd_stencil_size, self.mpm_pbd_stencil_size)
                    ):
                        slot_idx = self.pbd_solver.sh.grid_to_slot(base + offset)
                        for i in range(
                            self.pbd_solver.sh.slot_start[slot_idx, i_b],
                            self.pbd_solver.sh.slot_start[slot_idx, i_b] + self.pbd_solver.sh.slot_size[slot_idx, i_b],
                        ):
                            if (
                                qd.abs(pos - self.pbd_solver.particles_reordered.pos[i, i_b]).max()
                                < self.mpm_solver.dx * 0.5
                            ):
                                pbd_vel += self.pbd_solver.particles_reordered.vel[i, i_b]
                                colliding_particles += 1
                    if colliding_particles > 0:
                        vel_old = vel_mpm
                        vel_mpm = pbd_vel / colliding_particles

                        # ---------- MPM -> PBD ----------
                        delta_mv = mass_mpm * (vel_mpm - vel_old)

                        for offset in qd.grouped(
                            qd.ndrange(self.mpm_pbd_stencil_size, self.mpm_pbd_stencil_size, self.mpm_pbd_stencil_size)
                        ):
                            slot_idx = self.pbd_solver.sh.grid_to_slot(base + offset)
                            for i in range(
                                self.pbd_solver.sh.slot_start[slot_idx, i_b],
                                self.pbd_solver.sh.slot_start[slot_idx, i_b]
                                + self.pbd_solver.sh.slot_size[slot_idx, i_b],
                            ):
                                if (
                                    qd.abs(pos - self.pbd_solver.particles_reordered.pos[i, i_b]).max()
                                    < self.mpm_solver.dx * 0.5
                                ):
                                    if self.pbd_solver.particles_reordered[i, i_b].free:
                                        self.pbd_solver.particles_reordered[i, i_b].vel = (
                                            self.pbd_solver.particles_reordered[i, i_b].vel
                                            - delta_mv / self.pbd_solver.particles_info_reordered[i, i_b].mass
                                        )

                #################### MPM boundary ####################
                _, self.mpm_solver.grid[f, I, i_b].vel_out = self.mpm_solver.boundary.impose_pos_vel(pos, vel_mpm)

    @qd.kernel
    def mpm_surface_to_particle(
        self,
        f: qd.i32,
        geoms_state: array_class.GeomsState,
        geoms_info: array_class.GeomsInfo,
        sdf_info: array_class.SDFInfo,
        rigid_global_info: array_class.RigidGlobalInfo,
        collider_static_config: qd.template(),
    ):
        for i_p, i_b in qd.ndrange(self.mpm_solver.n_particles, self.mpm_solver._B):
            if self.mpm_solver.particles_ng[f, i_p, i_b].active:
                for i_g in range(self.rigid_solver.n_geoms):
                    if geoms_info.needs_coup[i_g]:
                        sdf_normal = sdf.sdf_func_normal_world(
                            geoms_state=geoms_state,
                            geoms_info=geoms_info,
                            rigid_global_info=rigid_global_info,
                            collider_static_config=collider_static_config,
                            sdf_info=sdf_info,
                            pos_world=self.mpm_solver.particles[f, i_p, i_b].pos,
                            geom_idx=i_g,
                            batch_idx=i_b,
                        )
                        # we only update the normal if the particle does not the object
                        if sdf_normal.dot(self.mpm_rigid_normal[i_p, i_g, i_b]) >= 0:
                            self.mpm_rigid_normal[i_p, i_g, i_b] = sdf_normal

    def fem_rigid_link_constraints(self):
        if self.fem_solver._constraints_initialized and self.rigid_solver.is_active:
            self.fem_solver._kernel_update_linked_vertex_constraints(self.rigid_solver.links_state)

    @qd.kernel
    def fem_surface_force(
        self,
        f: qd.i32,
        geoms_state: array_class.GeomsState,
        geoms_info: array_class.GeomsInfo,
        links_state: array_class.LinksState,
        rigid_global_info: array_class.RigidGlobalInfo,
        sdf_info: array_class.SDFInfo,
        collider_static_config: qd.template(),
    ):
        # TODO: all collisions are on vertices instead of surface and edge
        for i_s, i_b in qd.ndrange(self.fem_solver.n_surfaces, self.fem_solver._B):
            if self.fem_solver.surface[i_s].active:
                dt = self.fem_solver.substep_dt
                iel = self.fem_solver.surface[i_s].tri2el
                mass = self.fem_solver.elements_i[iel].mass_scaled / self.fem_solver.vol_scale

                p1 = self.fem_solver.elements_v[f, self.fem_solver.surface[i_s].tri2v[0], i_b].pos
                p2 = self.fem_solver.elements_v[f, self.fem_solver.surface[i_s].tri2v[1], i_b].pos
                p3 = self.fem_solver.elements_v[f, self.fem_solver.surface[i_s].tri2v[2], i_b].pos
                u = p2 - p1
                v = p3 - p1
                surface_normal = qd.math.cross(u, v)
                surface_normal = surface_normal / surface_normal.norm(gs.EPS)

                # FEM <-> Rigid
                if qd.static(self._rigid_fem):
                    # NOTE: collision only on surface vertices
                    for j in qd.static(range(3)):
                        iv = self.fem_solver.surface[i_s].tri2v[j]
                        vel_fem_sv = self._func_collide_with_rigid(
                            f,
                            self.fem_solver.elements_v[f, iv, i_b].pos,
                            self.fem_solver.elements_v[f + 1, iv, i_b].vel,
                            mass / 3.0,  # assume element mass uniformly distributed to vertices
                            i_b,
                            geoms_state,
                            geoms_info,
                            links_state,
                            rigid_global_info,
                            sdf_info,
                            collider_static_config,
                        )
                        self.fem_solver.elements_v[f + 1, iv, i_b].vel = vel_fem_sv

                # FEM <-> MPM (interact with MPM grid instead of particles)
                # NOTE: not doing this in mpm_grid_op otherwise we need to search for fem surface for each particles
                #       however, this function is called after mpm boundary conditions.
                if qd.static(self._fem_mpm):
                    for j in qd.static(range(3)):
                        iv = self.fem_solver.surface[i_s].tri2v[j]
                        pos = self.fem_solver.elements_v[f, iv, i_b].pos
                        vel_fem_sv = self.fem_solver.elements_v[f + 1, iv, i_b].vel
                        mass_fem_sv = mass / 4.0  # assume element mass uniformly distributed

                        # follow MPM p2g scheme
                        vel_mpm = qd.Vector([0.0, 0.0, 0.0])
                        mass_mpm = 0.0
                        mpm_base = qd.floor(pos * self.mpm_solver.inv_dx - 0.5).cast(gs.qd_int)
                        mpm_fx = pos * self.mpm_solver.inv_dx - mpm_base.cast(gs.qd_float)
                        mpm_w = [0.5 * (1.5 - mpm_fx) ** 2, 0.75 - (mpm_fx - 1.0) ** 2, 0.5 * (mpm_fx - 0.5) ** 2]
                        new_vel_fem_sv = vel_fem_sv
                        for mpm_offset in qd.static(qd.grouped(self.mpm_solver.stencil_range())):
                            mpm_grid_I = mpm_base - self.mpm_solver.grid_offset + mpm_offset
                            mpm_grid_mass = (
                                self.mpm_solver.grid[f, mpm_grid_I, i_b].mass / self.mpm_solver.particle_volume_scale
                            )

                            mpm_weight = gs.qd_float(1.0)
                            for d in qd.static(range(3)):
                                mpm_weight *= mpm_w[mpm_offset[d]][d]

                            # FEM -> MPM
                            mpm_grid_pos = (mpm_grid_I + self.mpm_solver.grid_offset) * self.mpm_solver.dx
                            signed_dist = (mpm_grid_pos - pos).dot(surface_normal)
                            if signed_dist <= self.mpm_solver.dx:  # NOTE: use dx as minimal unit for collision
                                vel_mpm_at_cell = mpm_weight * self.mpm_solver.grid[f, mpm_grid_I, i_b].vel_out
                                mass_mpm_at_cell = mpm_weight * mpm_grid_mass

                                vel_mpm += vel_mpm_at_cell
                                mass_mpm += mass_mpm_at_cell

                                if mass_mpm_at_cell > gs.EPS:
                                    delta_mpm_vel_at_cell_unmul = (
                                        vel_fem_sv * mpm_weight - self.mpm_solver.grid[f, mpm_grid_I, i_b].vel_out
                                    )
                                    mass_mul_at_cell = (
                                        mpm_grid_mass / mass_fem_sv
                                    )  # NOTE: use un-reweighted mass instead of mass_mpm_at_cell
                                    delta_mpm_vel_at_cell = delta_mpm_vel_at_cell_unmul * mass_mul_at_cell
                                    self.mpm_solver.grid[f, mpm_grid_I, i_b].vel_out += delta_mpm_vel_at_cell

                                    new_vel_fem_sv -= delta_mpm_vel_at_cell * mass_mpm_at_cell / mass_fem_sv

                        # MPM -> FEM
                        if mass_mpm > gs.EPS:
                            # delta_mv = (vel_mpm - vel_fem_sv) * mass_mpm
                            # delta_vel_fem_sv = delta_mv / mass_fem_sv
                            # self.fem_solver.elements_v[f + 1, iv].vel += delta_vel_fem_sv
                            self.fem_solver.elements_v[f + 1, iv, i_b].vel = new_vel_fem_sv

                # FEM <-> SPH TODO: this doesn't work well
                if qd.static(self._fem_sph):
                    for j in qd.static(range(3)):
                        iv = self.fem_solver.surface[i_s].tri2v[j]
                        pos = self.fem_solver.elements_v[f, iv, i_b].pos
                        vel_fem_sv = self.fem_solver.elements_v[f + 1, iv, i_b].vel
                        mass_fem_sv = mass / 4.0

                        dx = self.sph_solver.hash_grid_cell_size  # self._dx
                        stencil_size = 2  # self._stencil_size

                        base = self.sph_solver.sh.pos_to_grid(pos - 0.5 * dx)

                        # ---------- SPH -> FEM ----------
                        sph_vel = qd.Vector([0.0, 0.0, 0.0])
                        colliding_particles = 0
                        for offset in qd.grouped(qd.ndrange(stencil_size, stencil_size, stencil_size)):
                            slot_idx = self.sph_solver.sh.grid_to_slot(base + offset)
                            for k in range(
                                self.sph_solver.sh.slot_start[slot_idx, i_b],
                                self.sph_solver.sh.slot_start[slot_idx, i_b]
                                + self.sph_solver.sh.slot_size[slot_idx, i_b],
                            ):
                                if qd.abs(pos - self.sph_solver.particles_reordered.pos[k, i_b]).max() < dx * 0.5:
                                    sph_vel += self.sph_solver.particles_reordered.vel[k, i_b]
                                    colliding_particles += 1

                        if colliding_particles > 0:
                            vel_old = vel_fem_sv
                            vel_fem_sv_unprojected = sph_vel / colliding_particles
                            vel_fem_sv = (
                                vel_fem_sv_unprojected.dot(surface_normal) * surface_normal
                            )  # exclude tangential velocity

                            # ---------- FEM -> SPH ----------
                            delta_mv = mass_fem_sv * (vel_fem_sv - vel_old)

                            for offset in qd.grouped(qd.ndrange(stencil_size, stencil_size, stencil_size)):
                                slot_idx = self.sph_solver.sh.grid_to_slot(base + offset)
                                for k in range(
                                    self.sph_solver.sh.slot_start[slot_idx, i_b],
                                    self.sph_solver.sh.slot_start[slot_idx, i_b]
                                    + self.sph_solver.sh.slot_size[slot_idx, i_b],
                                ):
                                    if qd.abs(pos - self.sph_solver.particles_reordered.pos[k, i_b]).max() < dx * 0.5:
                                        self.sph_solver.particles_reordered[k, i_b].vel = (
                                            self.sph_solver.particles_reordered[k, i_b].vel
                                            - delta_mv / self.sph_solver.particles_info_reordered[k, i_b].mass
                                        )

                            self.fem_solver.elements_v[f + 1, iv, i_b].vel = vel_fem_sv

                # boundary condition
                for j in qd.static(range(3)):
                    iv = self.fem_solver.surface[i_s].tri2v[j]
                    _, self.fem_solver.elements_v[f + 1, iv, i_b].vel = self.fem_solver.boundary.impose_pos_vel(
                        self.fem_solver.elements_v[f, iv, i_b].pos, self.fem_solver.elements_v[f + 1, iv, i_b].vel
                    )

    def fem_hydroelastic(self, f: qd.i32):
        # Floor contact

        # collision detection
        self.fem_solver.floor_hydroelastic_detection(f)

    @qd.kernel
    def sph_rigid(
        self,
        f: qd.i32,
        geoms_state: array_class.GeomsState,
        geoms_info: array_class.GeomsInfo,
        links_state: array_class.LinksState,
        rigid_global_info: array_class.RigidGlobalInfo,
        sdf_info: array_class.SDFInfo,
        collider_static_config: qd.template(),
    ):
        for i_p, i_b in qd.ndrange(self.sph_solver._n_particles, self.sph_solver._B):
            if self.sph_solver.particles_ng_reordered[i_p, i_b].active:
                for i_g in range(self.rigid_solver.n_geoms):
                    if geoms_info.needs_coup[i_g]:
                        (
                            self.sph_solver.particles_reordered[i_p, i_b].vel,
                            self.sph_rigid_normal_reordered[i_p, i_g, i_b],
                        ) = self._func_collide_with_rigid_geom_robust(
                            self.sph_solver.particles_reordered[i_p, i_b].pos,
                            self.sph_solver.particles_reordered[i_p, i_b].vel,
                            self.sph_solver.particles_info_reordered[i_p, i_b].mass,
                            self.sph_rigid_normal_reordered[i_p, i_g, i_b],
                            i_g,
                            i_b,
                            geoms_state,
                            geoms_info,
                            links_state,
                            rigid_global_info,
                            sdf_info,
                            collider_static_config,
                        )

    @qd.kernel
    def kernel_pbd_rigid_collide(
        self,
        geoms_state: array_class.GeomsState,
        geoms_info: array_class.GeomsInfo,
        links_state: array_class.LinksState,
        sdf_info: array_class.SDFInfo,
        rigid_global_info: array_class.RigidGlobalInfo,
        collider_static_config: qd.template(),
    ):
        for i_p, i_b in qd.ndrange(self.pbd_solver._n_particles, self.sph_solver._B):
            if self.pbd_solver.particles_ng_reordered[i_p, i_b].active:
                # NOTE: Couldn't figure out a good way to handle collision with non-free particle. Such collision is not phsically plausible anyway.
                for i_g in range(self.rigid_solver.n_geoms):
                    if geoms_info.needs_coup[i_g]:
                        (
                            self.pbd_solver.particles_reordered[i_p, i_b].pos,
                            self.pbd_solver.particles_reordered[i_p, i_b].vel,
                            self.pbd_rigid_normal_reordered[i_p, i_b, i_g],
                        ) = self._func_pbd_collide_with_rigid_geom(
                            i_p,
                            self.pbd_solver.particles_reordered[i_p, i_b].pos,
                            self.pbd_solver.particles_reordered[i_p, i_b].vel,
                            self.pbd_solver.particles_info_reordered[i_p, i_b].mass,
                            self.pbd_rigid_normal_reordered[i_p, i_b, i_g],
                            i_g,
                            i_b,
                            geoms_state,
                            geoms_info,
                            links_state,
                            sdf_info,
                            rigid_global_info,
                            collider_static_config,
                        )

    @qd.kernel
    def kernel_attach_pbd_to_rigid_link(
        self,
        particles_idx: qd.types.ndarray(),
        envs_idx: qd.types.ndarray(),
        link_idx: qd.i32,
        links_state: LinksState,
    ) -> None:
        """
        Sets listed particles in listed environments to be animated by the link.

        Current position of the particle, relatively to the link, is stored and preserved.
        """
        pdb = self.pbd_solver

        for i_p_, i_b_ in qd.ndrange(particles_idx.shape[1], envs_idx.shape[0]):
            i_p = particles_idx[i_b_, i_p_]
            i_b = envs_idx[i_b_]
            link_pos = links_state.pos[link_idx, i_b]
            link_quat = links_state.quat[link_idx, i_b]

            # compute local offset from link to the particle
            world_pos = pdb.particles[i_p, i_b].pos
            local_pos = qd_inv_transform_by_trans_quat(world_pos, link_pos, link_quat)

            # set particle to be animated (not free) and store animation info
            pdb.particles[i_p, i_b].free = False
            self.particle_attach_info[i_p, i_b].link_idx = link_idx
            self.particle_attach_info[i_p, i_b].local_pos = local_pos

    @qd.kernel
    def kernel_pbd_rigid_clear_animate_particles_by_link(
        self,
        particles_idx: qd.types.ndarray(),
        envs_idx: qd.types.ndarray(),
    ) -> None:
        """Detach listed particles from links, and simulate them freely."""
        pdb = self.pbd_solver
        for i_p_, i_b_ in qd.ndrange(particles_idx.shape[1], envs_idx.shape[0]):
            i_p = particles_idx[i_b_, i_p_]
            i_b = envs_idx[i_b_]
            pdb.particles[i_p, i_b].free = True
            self.particle_attach_info[i_p, i_b].link_idx = -1
            self.particle_attach_info[i_p, i_b].local_pos = qd.math.vec3([0.0, 0.0, 0.0])

    @qd.kernel
    def kernel_pbd_rigid_solve_animate_particles_by_link(self, clamped_inv_dt: qd.f32, links_state: LinksState):
        """
        Itearates all particles and environments, and sets corrective velocity for all animated particle.

        Computes target position and velocity from the attachment/reference link and local offset position.

        Note, that this step shoudl be done after rigid solver update, and before PDB solver update.
        Currently, this is done after both rigid and PBD solver updates, hence the corrective velocity
        is off by a frame.

        Note, it's adviced to clamp inv_dt to avoid large jerks and instability. 1/0.02 might be a good max value.
        """
        pdb = self.pbd_solver
        for i_p, i_env in qd.ndrange(pdb._n_particles, pdb._B):
            if self.particle_attach_info[i_p, i_env].link_idx >= 0:
                # read link state
                link_idx = self.particle_attach_info[i_p, i_env].link_idx
                link_pos = links_state.pos[link_idx, i_env]
                link_quat = links_state.quat[link_idx, i_env]

                link_lin_vel = links_state.cd_vel[link_idx, i_env]
                link_ang_vel = links_state.cd_ang[link_idx, i_env]
                link_com_in_world = links_state.root_COM[link_idx, i_env] + links_state.i_pos[link_idx, i_env]

                # calculate target pos and vel of the particle
                local_pos = self.particle_attach_info[i_p, i_env].local_pos
                target_world_pos = qd_transform_by_trans_quat(local_pos, link_pos, link_quat)

                world_arm = target_world_pos - link_com_in_world
                target_world_vel = link_lin_vel + link_ang_vel.cross(world_arm)

                # compute and apply corrective velocity
                i_rp = pdb.particles_ng[i_p, i_env].reordered_idx
                particle_pos = pdb.particles_reordered[i_rp, i_env].pos
                pos_correction = target_world_pos - particle_pos
                corrective_vel = pos_correction * clamped_inv_dt
                pdb.particles_reordered[i_rp, i_env].vel = corrective_vel + target_world_vel

    @qd.func
    def _func_pbd_collide_with_rigid_geom(
        self,
        i,
        pos_world,
        vel,
        mass,
        normal_prev,
        geom_idx,
        batch_idx,
        geoms_state: array_class.GeomsState,
        geoms_info: array_class.GeomsInfo,
        links_state: array_class.LinksState,
        sdf_info: array_class.SDFInfo,
        rigid_global_info: array_class.RigidGlobalInfo,
        collider_static_config: qd.template(),
    ):
        """
        Resolves collision when a particle is already in collision with a rigid object.
        This function assumes known normal_rigid and influence.
        """
        signed_dist = sdf.sdf_func_world(
            geoms_state=geoms_state,
            geoms_info=geoms_info,
            sdf_info=sdf_info,
            pos_world=pos_world,
            geom_idx=geom_idx,
            batch_idx=batch_idx,
        )
        contact_normal = sdf.sdf_func_normal_world(
            geoms_state=geoms_state,
            geoms_info=geoms_info,
            rigid_global_info=rigid_global_info,
            collider_static_config=collider_static_config,
            sdf_info=sdf_info,
            pos_world=pos_world,
            geom_idx=geom_idx,
            batch_idx=batch_idx,
        )
        new_pos = pos_world
        new_vel = vel
        if signed_dist < self.pbd_solver.particle_size / 2:  # skip non-penetration particles
            stiffness = 1.0  # value in [0, 1]

            # we don't consider friction for now
            # friction = 0.15
            # vel_rigid = self.rigid_solver._func_vel_at_point(
            #     pos_world=pos_world,
            #     link_idx=geoms_info.link_idx[geom_idx],
            #     i_b=batch_idx,
            #     links_state=links_state,
            # )
            # rvel = vel - vel_rigid
            # rvel_normal_magnitude = rvel.dot(contact_normal)  # negative if inward
            # rvel_tan = rvel - rvel_normal_magnitude * contact_normal
            # rvel_tan_norm = rvel_tan.norm(gs.EPS)

            #################### rigid -> particle ####################

            energy_loss = 0.0  # value in [0, 1]
            new_pos = pos_world + stiffness * contact_normal * (self.pbd_solver.particle_size / 2 - signed_dist)
            prev_pos = self.pbd_solver.particles_reordered[i, batch_idx].ipos
            new_vel = (new_pos - prev_pos) / self.pbd_solver._substep_dt

            #################### particle -> rigid ####################
            delta_mv = mass * (new_vel - vel)
            force = (-delta_mv / self.rigid_solver._substep_dt) * (1 - energy_loss)

            self.rigid_solver._func_apply_coupling_force(
                pos_world,
                force,
                geoms_info.link_idx[geom_idx],
                batch_idx,
                links_state,
            )

        return new_pos, new_vel, contact_normal

    def preprocess(self, f):
        # preprocess for MPM CPIC
        if self._rigid_mpm and self.mpm_solver.enable_CPIC:
            self.mpm_surface_to_particle(
                f,
                self.rigid_solver.geoms_state,
                self.rigid_solver.geoms_info,
                self.rigid_solver.collider._sdf._sdf_info,
                self.rigid_solver._rigid_global_info,
                self.rigid_solver.collider._collider_static_config,
            )

    @qd.kernel
    def mpm_grid_thermal_diffusion(self, f: qd.i32):
        for I in qd.grouped(qd.ndrange(*self.mpm_solver.grid_res)):
            for i_b in range(self.mpm_solver._B):
                m_C = self.mpm_solver.grid[f, I, i_b].mass_thermal
                if m_C > gs.EPS:
                    T_C = self.mpm_solver.grid[f, I, i_b].temp
                    laplacian = gs.qd_float(0.0)
                    
                    I_left = I + qd.Vector([-1, 0, 0])
                    I_right = I + qd.Vector([1, 0, 0])
                    I_down = I + qd.Vector([0, -1, 0])
                    I_up = I + qd.Vector([0, 1, 0])
                    I_back = I + qd.Vector([0, 0, -1])
                    I_front = I + qd.Vector([0, 0, 1])
                    
                    if I_left[0] >= 0:
                        m_N = self.mpm_solver.grid[f, I_left, i_b].mass_thermal
                        if m_N > gs.EPS:
                            laplacian += (qd.math.min(m_C, m_N) / m_C) * (self.mpm_solver.grid[f, I_left, i_b].temp - T_C)
                    
                    if I_right[0] < self.mpm_solver.grid_res[0]:
                        m_N = self.mpm_solver.grid[f, I_right, i_b].mass_thermal
                        if m_N > gs.EPS:
                            laplacian += (qd.math.min(m_C, m_N) / m_C) * (self.mpm_solver.grid[f, I_right, i_b].temp - T_C)
                            
                    if I_down[1] >= 0:
                        m_N = self.mpm_solver.grid[f, I_down, i_b].mass_thermal
                        if m_N > gs.EPS:
                            laplacian += (qd.math.min(m_C, m_N) / m_C) * (self.mpm_solver.grid[f, I_down, i_b].temp - T_C)
                            
                    if I_up[1] < self.mpm_solver.grid_res[1]:
                        m_N = self.mpm_solver.grid[f, I_up, i_b].mass_thermal
                        if m_N > gs.EPS:
                            laplacian += (qd.math.min(m_C, m_N) / m_C) * (self.mpm_solver.grid[f, I_up, i_b].temp - T_C)
                            
                    if I_back[2] >= 0:
                        m_N = self.mpm_solver.grid[f, I_back, i_b].mass_thermal
                        if m_N > gs.EPS:
                            laplacian += (qd.math.min(m_C, m_N) / m_C) * (self.mpm_solver.grid[f, I_back, i_b].temp - T_C)
                            
                    if I_front[2] < self.mpm_solver.grid_res[2]:
                        m_N = self.mpm_solver.grid[f, I_front, i_b].mass_thermal
                        if m_N > gs.EPS:
                            laplacian += (qd.math.min(m_C, m_N) / m_C) * (self.mpm_solver.grid[f, I_front, i_b].temp - T_C)

                    # Dynamic thermal diffusivity: α(T) = k(T) / (ρ · Cp(T))
                    k_local = self.mpm_solver.get_steel_thermal_conductivity(T_C)
                    Cp_local = self.mpm_solver.get_steel_cp(T_C)
                    rho = gs.qd_float(7850.0)  # kg/m^3 (AISI 4340 steel density)
                    alpha = (k_local / (rho * Cp_local)) * self.mpm_solver._rt_alpha_thermal[None]
                    dx = self.mpm_solver.dx
                    dt = self.mpm_solver.substep_dt
                    
                    T_new = T_C + alpha * dt / (dx * dx) * laplacian
                    self.mpm_solver.grid[f, I, i_b].dT_diffusion = T_new - T_C
                    self.mpm_solver.grid[f, I, i_b].temp_diffused = T_new
                else:
                    # Empty cells must have ambient temp so surface particles don't read 0K via G2P.
                    # Without this, the B-spline stencil bleeds zeroed memory into surface particles.
                    self.mpm_solver.grid[f, I, i_b].temp_diffused = gs.qd_float(293.15)

    # ------------------------------------------------------------------------------------
    # ------------------------------ Penetration probe -----------------------------------
    # ------------------------------------------------------------------------------------

    @qd.kernel
    def mpm_penetration_probe(
        self,
        f: qd.i32,
        geoms_state: array_class.GeomsState,
        geoms_info: array_class.GeomsInfo,
        sdf_info: array_class.SDFInfo,
    ):
        """
        How far material has actually gotten inside the die, measured identically for every
        contact mode.

        Reads frame f+1 -- after g2p and after every contact operation of this substep -- so it
        reports the penetration a method LEAVES BEHIND, not the penetration it was presented with.
        Accumulates across substeps into 0-D fields; Python resets and reads them per recorded
        frame (penetration_reset / penetration_read).
        """
        for i_p, i_b in qd.ndrange(self.mpm_solver.n_particles, self.mpm_solver._B):
            if self.mpm_solver.particles_ng[f + 1, i_p, i_b].active:
                pos = self.mpm_solver.particles[f + 1, i_p, i_b].pos
                # Signed distance is negative inside a geom, so the MINIMUM over coupled geoms is
                # this particle's worst penetration. Seeded well outside any plausible SDF value.
                worst = gs.qd_float(1e10)
                for i_g in qd.static(range(self.rigid_solver.n_geoms)):
                    if geoms_info.needs_coup[i_g]:
                        signed_dist = sdf.sdf_func_world(
                            geoms_state=geoms_state,
                            geoms_info=geoms_info,
                            sdf_info=sdf_info,
                            pos_world=pos,
                            geom_idx=i_g,
                            batch_idx=i_b,
                        )
                        if signed_dist < worst:
                            worst = signed_dist
                qd.atomic_add(self._pen_active[None], 1)
                if worst < 0.0:
                    qd.atomic_max(self._pen_max[None], -worst)
                    qd.atomic_add(self._pen_sum[None], -worst)
                    qd.atomic_add(self._pen_count[None], 1)

    def penetration_reset(self) -> None:
        """Zero the accumulators. Called once per recorded frame, after reading."""
        self._pen_max[None] = 0.0
        self._pen_sum[None] = 0.0
        self._pen_count[None] = 0
        self._pen_active[None] = 0

    def penetration_read(self) -> dict:
        """
        Penetration since the last reset, in raw sim length units, alongside the two scales that
        make it interpretable: the grid spacing (the resolution grid contact is inherently limited
        to) and the particle size (the scale the teleport's margin is defined at). Both are
        reported rather than divided in, so the analysis can normalize without this code baking in
        a unit assumption it has not verified.

        ``pen_frac`` is over particle-substep samples, not particles: the denominator counts every
        active particle each substep, so it is a duty-cycle-weighted fraction, not "what fraction
        of the billet is inside the die right now".
        """
        n = int(self._pen_count[None])
        n_active = int(self._pen_active[None])
        return {
            "pen_max": float(self._pen_max[None]),
            "pen_mean": (float(self._pen_sum[None]) / n) if n else 0.0,
            "pen_frac": (n / n_active) if n_active else 0.0,
            "pen_samples": n_active,
            "dx": float(self.mpm_solver.dx),
            "particle_size": float(self.mpm_solver._particle_size),
        }

    # ------------------------------------------------------------------------------------
    # --------------------- Runtime-switchable contact controls --------------------------
    # ------------------------------------------------------------------------------------

    def _assert_runtime_switchable(self) -> None:
        if not getattr(self, "_runtime_switchable", False):
            gs.raise_exception(
                "Live contact controls require building the scene with "
                "LegacyCouplerOptions(rigid_mpm_contact_runtime_switchable=True) "
                "(AGF_CONTACT_RUNTIME_SWITCH=1)."
            )

    @property
    def contact_mode(self) -> str:
        """The currently active rigid-MPM contact mode (string name)."""
        if getattr(self, "_runtime_switchable", False):
            return self._CONTACT_ID_TO_MODE[int(self._rt_contact_mode[None])]
        return self._contact_mode

    def set_contact_mode(self, mode: str) -> None:
        """
        Switch the active rigid-MPM contact mode live -- no scene rebuild, no recompile of the
        already-traced kernels. ``mode`` is one of the entries in ``_CONTACT_MODES``.

        The Python-side dispatch gates are updated in step, because whether the separately-launched
        postg2p / penalty / ftmp kernels run at all is decided in Python, not inside a kernel.
        """
        self._assert_runtime_switchable()
        if mode not in self._CONTACT_MODE_TO_ID:
            gs.raise_exception(f"Unknown contact mode {mode!r}. Valid: {list(self._CONTACT_MODE_TO_ID)}.")
        mode_id = self._CONTACT_MODE_TO_ID[mode]
        self._rt_contact_mode[None] = mode_id
        self._contact_mode = mode

        self._contact_gridop_vel = mode_id == self.CM_GRID
        self._contact_ing2p_vel = mode_id == self.CM_PARTICLE
        self._contact_ing2p_fluidlab = mode_id == self.CM_FLUIDLAB
        self._contact_postg2p_vel = mode_id == self.CM_POSTG2P_VEL
        self._contact_postg2p_pos = mode_id == self.CM_POSTG2P_POS
        self._contact_postg2p = self._contact_postg2p_vel or self._contact_postg2p_pos
        self._contact_penalty = mode_id == self.CM_PENALTY
        # ftmp is dispatched from Python and, unlike upstream, is allowed on 'grid' as well as
        # 'particle' here -- grid supplies the force balance, ftmp supplies contact compression
        # as stress.
        self._contact_ftmp_proj = bool(self._rt_ftmp_proj[None]) and mode_id in (
            self.CM_PARTICLE,
            self.CM_GRID,
        )

    def set_refinement(
        self, *, per_node=None, c_injection=None, ftmp_proj=None, grid_floor=None
    ) -> None:
        """Toggle the contact refinements live (``None`` leaves a toggle unchanged).

        ``grid_floor`` enables grid-level rigid contact ALONGSIDE whatever contact mode is
        selected. Without it, any non-``grid`` mode runs with no non-penetration floor at all.
        """
        self._assert_runtime_switchable()
        if grid_floor is not None:
            self._rt_grid_floor[None] = int(bool(grid_floor))
        if per_node is not None:
            self._rt_per_node[None] = int(bool(per_node))
        if c_injection is not None:
            self._rt_c_injection[None] = int(bool(c_injection))
        if ftmp_proj is not None:
            self._rt_ftmp_proj[None] = int(bool(ftmp_proj))
        self._contact_ftmp_proj = bool(self._rt_ftmp_proj[None]) and int(
            self._rt_contact_mode[None]
        ) in (self.CM_PARTICLE, self.CM_GRID)

    def set_particle_contact(
        self, *, mech=None, c_project=None, f_feedback=None, c_damp=None
    ) -> None:
        """
        Toggle base_mpm_solver's ``apply_particle_contact`` teleport live (``None`` leaves a knob
        unchanged). This is the axis the grid-vs-grid+teleport comparison turns on, and it has no
        upstream equivalent.

        Note ``mech`` gates only the MECHANICAL projection. Die<->billet heat transfer lives in the
        same kernel behind ``_pc_thermal`` and is deliberately left alone, so a contact-method arm
        does not silently also change the thermal boundary condition.
        """
        self._assert_runtime_switchable()
        if mech is not None:
            self._rt_pc_mech[None] = int(bool(mech))
        if c_project is not None:
            self._rt_pc_c_project[None] = int(bool(c_project))
        if f_feedback is not None:
            self._rt_pc_f_feedback[None] = int(bool(f_feedback))
        if c_damp is not None:
            self._rt_pc_c_damp[None] = float(c_damp)

    def get_contact_config(self) -> dict:
        """The full live contact configuration -- what an arm in a batched sweep actually ran."""
        self._assert_runtime_switchable()
        return {
            "mode": self._CONTACT_ID_TO_MODE[int(self._rt_contact_mode[None])],
            "grid_floor": bool(self._rt_grid_floor[None]),
            "per_node": bool(self._rt_per_node[None]),
            "c_injection": bool(self._rt_c_injection[None]),
            "ftmp_proj": bool(self._rt_ftmp_proj[None]),
            "penalty_k": float(self._rt_penalty_k[None]),
            "pc_mech": bool(self._rt_pc_mech[None]),
            "pc_c_project": bool(self._rt_pc_c_project[None]),
            "pc_f_feedback": bool(self._rt_pc_f_feedback[None]),
            "pc_c_damp": float(self._rt_pc_c_damp[None]),
        }

    def set_penalty_stiffness(self, k: float) -> None:
        """Set the spring stiffness used by the ``penalty`` contact mode (live)."""
        self._assert_runtime_switchable()
        self._rt_penalty_k[None] = float(k)

    def couple(self, f):
        import contextlib

        profiler = self.sim.scene.profiling_options.profiler
        sim_cfg = self.sim.scene.profiling_options.configs.simulator
        # MPM <-> all others
        if self.mpm_solver.is_active:
            # Penalty contact scatters its force into the grid momentum BEFORE the grid op
            # normalizes it, so the contact force reaches C and hence F via the later g2p.
            if self._contact_penalty:
                self.mpm_penalty_contact(
                    f,
                    self.rigid_solver.geoms_state,
                    self.rigid_solver.geoms_info,
                    self.rigid_solver.links_state,
                    self.rigid_solver._rigid_global_info,
                    self.rigid_solver.collider._sdf._sdf_info,
                    self.rigid_solver.collider._collider_static_config,
                )
            with profiler.time("couple_mpm_grid_op") if sim_cfg.couple_mpm_grid_op else contextlib.suppress():
                self.mpm_grid_op(
                    f,
                    self.sim.cur_t,
                    geoms_state=self.rigid_solver.geoms_state,
                    geoms_info=self.rigid_solver.geoms_info,
                    links_state=self.rigid_solver.links_state,
                    rigid_global_info=self.rigid_solver._rigid_global_info,
                    sdf_info=self.rigid_solver.collider._sdf._sdf_info,
                    collider_static_config=self.rigid_solver.collider._collider_static_config,
                )
            if self.mpm_solver._enable_thermal:
                with profiler.time("couple_thermal_diffusion") if sim_cfg.couple_thermal_diffusion else contextlib.suppress():
                    self.mpm_grid_thermal_diffusion(f)

        # SPH <-> Rigid
        if self._rigid_sph:
            self.sph_rigid(
                f,
                self.rigid_solver.geoms_state,
                self.rigid_solver.geoms_info,
                self.rigid_solver.links_state,
                self.rigid_solver._rigid_global_info,
                self.rigid_solver.collider._sdf._sdf_info,
                self.rigid_solver.collider._collider_static_config,
            )

        # PBD <-> Rigid
        if self._rigid_pbd:
            self.kernel_pbd_rigid_collide(
                geoms_state=self.rigid_solver.geoms_state,
                geoms_info=self.rigid_solver.geoms_info,
                links_state=self.rigid_solver.links_state,
                sdf_info=self.rigid_solver.collider._sdf._sdf_info,
                rigid_global_info=self.rigid_solver._rigid_global_info,
                collider_static_config=self.rigid_solver.collider._collider_static_config,
            )

            # 1-way: animate particles by links
            full_step_inv_dt = 1.0 / self.pbd_solver._dt
            clamped_inv_dt = min(full_step_inv_dt, CLAMPED_INV_DT)
            self.kernel_pbd_rigid_solve_animate_particles_by_link(clamped_inv_dt, self.rigid_solver.links_state)

        if self.fem_solver.is_active:
            self.fem_surface_force(
                f,
                self.rigid_solver.geoms_state,
                self.rigid_solver.geoms_info,
                self.rigid_solver.links_state,
                self.rigid_solver._rigid_global_info,
                self.rigid_solver.collider._sdf._sdf_info,
                self.rigid_solver.collider._collider_static_config,
            )
            self.fem_rigid_link_constraints()

    def couple_grad(self, f):
        if self.fem_solver.is_active:
            self.fem_surface_force.grad(
                f,
                self.rigid_solver.geoms_state,
                self.rigid_solver.geoms_info,
                self.rigid_solver.links_state,
                self.rigid_solver._rigid_global_info,
                self.rigid_solver.collider._sdf._sdf_info,
                self.rigid_solver.collider._collider_static_config,
            )
        if self.mpm_solver.is_active:
            self.mpm_grid_op.grad(
                f,
                self.sim.cur_t,
                geoms_state=self.rigid_solver.geoms_state,
                geoms_info=self.rigid_solver.geoms_info,
                links_state=self.rigid_solver.links_state,
                rigid_global_info=self.rigid_solver._rigid_global_info,
                sdf_info=self.rigid_solver.collider._sdf._sdf_info,
                collider_static_config=self.rigid_solver.collider._collider_static_config,
            )

    @property
    def active_solvers(self):
        """All the active solvers managed by the scene's simulator."""
        return self.sim.active_solvers
