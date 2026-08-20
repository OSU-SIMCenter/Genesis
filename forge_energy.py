#!/usr/bin/env python3
"""Kinetic and internal energy accounting for the MPM forging sim.

Answers two questions that nothing in agforge could answer before:

  1. Is the press still quasi-static under time scaling?  That is acceptance
     criterion 4 in the coupling doc's section 9.4, KE/IE < 5%, and it has been
     unevaluable because no kinetic energy monitor existed anywhere.

  2. When a result fails to converge in timestep -- as g1_grid_prod's work done
     does, marching 84.6 -> 89.6 -> 94.2 mm across CFL 0.90 / 0.45 / 0.225 --
     is the extra deformation physical, or is energy leaking into or out of the
     integrator?  A budget separates those; geometry alone cannot.

DESIGN

The numeric core takes plain arrays and imports nothing from genesis or
agforge, so it can be unit-checked without a GPU and reused from a post-hoc
analyser.  Only EnergyMonitor touches a live entity.

THE UNIT TRAP THIS MODULE EXISTS TO NOT REPEAT

particles_info[i].mass is scaled by _particle_volume_scale (1e3), because the
solver carries an inflated particle volume for conditioning.  A previous
diagnosis on this project was wrong by exactly 1000x from applying that scale
twice.  So the unscaling happens in ONE function here, real_particle_mass,
every consumer goes through it, and self_check asserts the resulting density is
physically plausible before any energy number is believed.

plastic_work is a SPECIFIC work, J/m^3, not an energy: the kernel accumulates
effective_yield * delta_gamma.  Multiply by the REAL particle volume to get
joules.  Reading it as joules is an error that looks merely small.

WHY INTERNAL ENERGY IS DERIVED RATHER THAN READ

plastic_work has NO Python accessor.  It is not on the entity, and the thermal
telemetry bundle returns only temp plus the seven dT_* fields.  The field
accumulates correctly inside the solver and is invisible from outside it, so a
plan that says "the inputs already exist" is half right: the state exists, the
way to read it does not.

Rather than edit shared source in two files to add a getter, this recovers the
same quantity from dT_adiabatic, which IS exposed, using an identity that falls
out of the kernel:

    dT = 0.9 * vol_work / (rho * Cp)
    m * Cp * dT = 0.9 * vol_work * (m / rho) = 0.9 * vol_work * V_real
    sum(m * Cp * dT) = 0.9 * E_plastic

Cp and mass cancel exactly; the module's smoke test confirms this to 2e-16.
The cost is that internal energy is no longer INDEPENDENT of dT_adiabatic, so
the cross-check that would have caught an error in either is unavailable.  That
is reported in summary() rather than skipped quietly.  Adding a plastic_work
getter later restores the check.

CUMULATIVE VERSUS PER-STEP

plastic_work is cumulative since reset -- nothing clears it except reset.
dT_adiabatic is per-macro-step, because StrikeController calls
clear_thermal_telemetry_buffers() before every scene.step(), unconditionally.
They are therefore NOT directly comparable; the monitor differences the
cumulative one before cross-checking.  Getting this backwards makes the check
fail by roughly the number of steps taken.

Cite symbols, not line numbers: this project's line numbers drift between
branches, and a sibling's comment-only commit has silently rotted citations
before.

--------------------------------------------------------------------------
THREE THINGS A USER OF THIS HARNESS NEEDS TO KNOW BEFORE TRUSTING A NUMBER
--------------------------------------------------------------------------

1. SUPPRESSING THE DIE-BALANCE CONTROLLER IS NOT A FREE OPERATION.

   This module's headline comparison lowered force_balance_gain to 1.5e-5 to
   take the controller out of the kinetic energy, and that was the right
   experiment: it overturned a wrong conclusion (see below).  But workstream B
   then ran the fully-suppressed case and the SIM ITSELF DESTABILISES -- three
   of four cells died with SimulationStabilityError, Supersonic Velocity
   >100 m/s, where all four completed 17/17 in the archive.  The loop's own
   source calls it "ADVANCED PROTECTION" and that is literal, not decorative.

   The distinction that appears to matter is QUIET versus ABSENT.  A-9 held the
   controller quiet (gain 1.5e-5, <=0.96% exposure) and completed 12/12 where
   fully-suppressed cells died.  So: lower the gain to make the controller
   quiet, do not assume that "quieter" extrapolates to "removed".

2. ONE INSTRUMENTED AXIS CANNOT TELL TWO CAUSES APART.

   p4_pg2p_vel reads Lx = 70.00 mm at CFL 0.90, and 70.11 mm at 12.5 m/s.
   Two entirely different causes, 0.11 mm apart.  Anyone measuring only
   geometry would call those the same run.

   Worse, the two axes move in OPPOSITE directions for the same doubling of
   substep count: halving CFL at fixed speed gives Lx +9.49 mm, halving press
   speed gives -9.38 mm.  "More steps" is therefore not a mechanism, and any
   argument resting on step count as a shared error term is unsound.

   That is the reason to record energy alongside geometry rather than instead
   of it: KE, plastic work and elastic energy separate cases that a single
   length cannot.

3. THE OPEN ANOMALY THIS HARNESS FOUND AND HAS NOT EXPLAINED.

   At MATCHED strain (0.2496) with stalling suppressed, 25.0 vs 6.25 m/s:

       total internal energy   1154.6 J   vs   1344.7 J    (+16.5%)
       of which elastic         177.0 J   vs    530.6 J    (3.0x)
       of which plastic         977.6 J   vs    814.1 J    (-17%)

   Below jc_T_ref the Johnson-Cook law is rate-independent (jc_C is dead code),
   so NONE of this should depend on press speed -- and the total, not just the
   split, is the part that should not move at all.  Explain the +16.5% before
   interpreting the 3x.

   Ruled out: explicit hardening.  sigma_y = A + B*eps_p^n is evaluated at the
   START-OF-STEP Jp with no iteration against the updated yield surface, which
   genuinely can manufacture rate dependence from a rate-independent law -- but
   it has the WRONG SIGN here.  After return ||eps_hat|| is pinned at
   sigma_y/(2 mu), so elastic deviatoric energy tracks sigma_y^2; the slow run
   does LESS plastic work, so lower eps_p, so lower sigma_y, so it should store
   LESS elastic energy.  It stores 3x more.

   Still live: deformation LOCALIZATION (yielding concentrated in a band at
   25 m/s and spread at 6.25 would put more particles elastically loaded, and
   would mean the model is behaving CORRECTLY), or a defect in this
   measurement.  Distinguishing them wants the per-particle plastic_strain
   field, which has no Python accessor.
"""

from __future__ import annotations

import math

import numpy as np

# Taylor-Quinney fraction, hardcoded in base_mpm_solver.p2g_post_constitutive.
TAYLOR_QUINNEY = 0.9


# --------------------------------------------------------------------------
# numeric core -- plain arrays in, joules out, no genesis/agforge imports
# --------------------------------------------------------------------------

def _as_np(a):
    """Accept torch tensors, numpy arrays, or scalars; return float64 numpy."""
    if a is None:
        return None
    if hasattr(a, "detach"):           # torch
        a = a.detach().cpu().numpy()
    return np.asarray(a, dtype=np.float64)


def kinetic_energy(vel, mass_real):
    """Sum of 1/2 m v^2 over particles, in joules.

    vel        : (N, 3) velocities [m/s]
    mass_real  : scalar or (N,) REAL particle mass [kg] -- see real_particle_mass
    """
    v = _as_np(vel).reshape(-1, 3)
    m = _as_np(mass_real)
    v2 = np.einsum("ij,ij->i", v, v)
    return float(np.sum(0.5 * m * v2))


def plastic_energy(plastic_work, particle_volume_real):
    """Cumulative dissipated plastic work, in joules.

    plastic_work         : (N,) SPECIFIC work [J/m^3] from the particle field
    particle_volume_real : scalar REAL particle volume [m^3]
    """
    w = _as_np(plastic_work).reshape(-1)
    return float(np.sum(w) * float(particle_volume_real))


def elastic_energy(F, mu, lam, particle_volume_real):
    """Recoverable elastic strain energy, in joules.

    Matched to the solver's actual potential rather than assumed.  The material
    uses HENCKY (logarithmic) elasticity with a von Mises radial return:

        epsilon      = log(singular values of F)      [materials.py, S_clamped]
        epsilon_hat  = epsilon - trace(epsilon)/3
        yields when  ||epsilon_hat|| > sigma_y / (2 mu)

    The potential whose derivative reproduces that yield check is

        W = mu * ||epsilon_hat||^2 + (kappa/2) * trace(epsilon)^2

    since tau = dW/depsilon gives tau_dev = 2 mu epsilon_hat, so
    ||tau_dev|| = 2 mu ||epsilon_hat||, which is exactly the quantity the solver
    compares against sigma_y.  kappa = lam + 2 mu / 3.

    F here is the ELASTIC deformation gradient: on yield the solver stores
    F_new = U @ S_new @ V^T with the plastic stretch already returned out, so
    the stored F carries only the elastic part.  Using it as though it were the
    total deformation gradient would double-count the plastic work.

    W is an energy per unit REFERENCE volume, so multiply by the real particle
    volume.
    """
    f = _as_np(F).reshape(-1, 3, 3)
    if f.shape[0] == 0:
        return 0.0
    sv = np.linalg.svd(f, compute_uv=False)          # (N, 3)
    sv = np.clip(sv, 1e-6, None)                     # solver clamps at 1e-6
    eps = np.log(sv)
    tr = eps.sum(axis=1)
    eps_hat = eps - (tr / 3.0)[:, None]
    kappa = float(lam) + 2.0 * float(mu) / 3.0
    w = float(mu) * np.einsum("ij,ij->i", eps_hat, eps_hat) + 0.5 * kappa * tr * tr
    return float(np.sum(w) * float(particle_volume_real))


def lame_from_E_nu(E, nu):
    """(mu, lambda) from Young's modulus and Poisson ratio."""
    E = float(E); nu = float(nu)
    mu = E / (2.0 * (1.0 + nu))
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    return mu, lam


def thermal_energy(temp, mass_real, cp):
    """Sum of m*Cp*T, in joules.

    This mirrors the one energy readout that already existed in the codebase --
    StrikeController's THERMAL log line, particle_mass * (t * cp_tensor).sum()
    -- which is dimensionally sound and correctly unscales the mass exactly
    once.  That line is display-only and gated behind a flag that is never true
    in the batch path, so this is a promotion of existing correct work rather
    than new physics.

    Absolute, i.e. referenced to 0 K.  For budgets take a difference against a
    baseline sample rather than using the absolute value.
    """
    t = _as_np(temp).reshape(-1)
    m = _as_np(mass_real)
    c = _as_np(cp)
    return float(np.sum(m * c * t))


def adiabatic_heating_energy(dT_adiabatic, mass_real, cp):
    """Energy represented by the solver's own plastic-heating temperature rise.

    Independent path to the same physical quantity as
    TAYLOR_QUINNEY * (change in plastic_energy): the kernel computes
    dT = 0.9 * vol_work / (rho * Cp) in one place and accumulates plastic_work
    in another.  Comparing them is a real check rather than a mirror test,
    because the two numbers travel through different code.
    """
    d = _as_np(dT_adiabatic).reshape(-1)
    m = _as_np(mass_real)
    c = _as_np(cp)
    return float(np.sum(m * c * d))


def steel_cp_numpy(temp, seg_params=None):
    """Specific heat [J/kg-K] for 316L, 3-segment form.

    Mirrors agforge.thermal.get_steel_cp_numpy.  Pass seg_params pulled from
    agforge.thermal.CP_316L_SEG_PARAMS so the knots cannot drift from the
    solver's; the fallback constant is only for array-shape smoke tests and is
    flagged in the returned provenance when used.
    """
    t = _as_np(temp).reshape(-1)
    if seg_params is None:
        return np.full_like(t, 500.0)
    v0, t1, v1, t2, v2, slope_hi, t0 = seg_params
    cp = np.full_like(t, float(v0))
    lo = (t >= t0) & (t < t1)
    denom_lo = (t1 - t0) if (t1 - t0) != 0 else 1.0
    cp = np.where(lo, v0 + (t - t0) * (v1 - v0) / denom_lo, cp)
    mid = (t >= t1) & (t < t2)
    denom = (t2 - t1) if (t2 - t1) != 0 else 1.0
    cp = np.where(mid, v1 + (t - t1) * (v2 - v1) / denom, cp)
    cp = np.where(t >= t2, v2 + (t - t2) * slope_hi, cp)
    return cp


# --------------------------------------------------------------------------
# the one place the volume scale is undone
# --------------------------------------------------------------------------

def real_particle_mass(solver, n_particles=None, particle_start=0):
    """REAL per-particle mass [kg], with _particle_volume_scale divided out once.

    The solver stores particles_info[i].mass = _particle_volume * mat_rho where
    _particle_volume is inflated by _particle_volume_scale.  Every consumer in
    the solver that wants a physical mass divides by that scale exactly once;
    so does this.

    Returns (mass_real, provenance_dict).
    """
    scale = float(solver._particle_volume_scale)
    prov = {"particle_volume_scale": scale, "mass_source": None}

    idx = int(particle_start)
    try:
        m_scaled = float(solver.particles_info[idx].mass)
        prov["mass_source"] = "particles_info[%d].mass" % idx
    except Exception as exc:
        raise RuntimeError(
            "could not read particles_info[%d].mass: %r" % (idx, exc))

    if not math.isfinite(m_scaled) or m_scaled <= 0.0:
        raise ValueError("implausible scaled particle mass %r" % m_scaled)

    mass_real = m_scaled / scale
    prov["mass_real_kg"] = mass_real
    prov["uniform_mass_assumed"] = True
    return mass_real, prov


def self_check(mass_real, n_particles, particle_volume_real, rho_expected,
               tol=0.01):
    """Assert the unscaled mass is physically consistent before trusting energies.

    mass_real / particle_volume_real must recover the material density.  If the
    volume scale were applied twice this lands 1000x off and the check fires,
    which is the entire point: the 1000x error that burned this project once
    already was invisible in every downstream number until someone re-derived
    it by hand.

    rho_expected MUST be the CONFIGURED density, cfg.mat.rho, not a literature
    value.  Those are two different questions -- "is my unscaling right" and "is
    the card's density right" -- and only the first is this module's business.
    Checking against a remembered number instead conflates them: the first run
    of this check used a plausible 7900 and fired at 7.2%, which looked like a
    unit bug and was in fact a wrong expectation.  The card ships 7334, sourced
    as NIST2021 SRM 1155a D(T) = 8052 - 0.564 T evaluated at 1273.15 K.  It is
    deliberate and temperature-corrected; do not "fix" it back to a
    room-temperature 8000.

    The tolerance is tight on purpose.  Against the configured value this is an
    identity -- particles_info.mass is literally _particle_volume * mat_rho --
    so anything above rounding is a real defect, not a modelling difference.

    Returns a dict; raises AssertionError on a real inconsistency.
    """
    rho_implied = mass_real / float(particle_volume_real)
    billet_kg = mass_real * int(n_particles)
    rel = abs(rho_implied - rho_expected) / rho_expected
    out = {
        "rho_implied_kg_m3": rho_implied,
        "rho_expected_kg_m3": rho_expected,
        "rho_rel_error": rel,
        "billet_mass_kg": billet_kg,
        "n_particles": int(n_particles),
        "passed": bool(rel <= tol),
    }
    if not out["passed"]:
        raise AssertionError(
            "particle mass inconsistent with density: implied rho=%.1f vs "
            "expected %.1f (rel %.3f). A factor near 1000 means "
            "_particle_volume_scale was applied twice; near 1e-3 means it was "
            "not applied at all. Details: %r" % (rho_implied, rho_expected, rel, out))
    return out


# --------------------------------------------------------------------------
# live monitor
# --------------------------------------------------------------------------

class EnergyMonitor:
    """Samples KE and IE from a live MPM entity at macro-step boundaries.

    Deliberately read-only: it calls existing getters and never writes solver
    state, so it can attach to any run on either branch without changing what
    that run simulates.  Attach, call sample() after each scene.step(), read
    summary().

    Availability is reported, never faked.  plastic_work and dT_adiabatic only
    exist when the solver was built with enable_thermal=True -- they sit inside
    an if self._enable_thermal guard in the particle-state template.  With
    thermal off, the plastic channel is ABSENT, and this reports it absent
    rather than reporting zero.  A diagnostic that quietly returns zero for a
    channel it cannot see is worse than one that refuses.
    """

    def __init__(self, entity, solver, rho_expected=None, cfg=None,
                 cp_seg_params=None, jsonl_path=None):
        self.entity = entity
        self.solver = solver

        self.n_particles = int(entity.n_particles)
        self.particle_start = int(getattr(entity, "_particle_start", 0))
        self.particle_volume_real = float(solver._particle_volume_real)

        self.mass_real, self.mass_prov = real_particle_mass(
            solver, self.n_particles, self.particle_start)

        # The density to check against comes from the CONFIG, not from memory.
        # See self_check's docstring for why that distinction is load-bearing.
        self.rho_source = "caller-supplied"
        if rho_expected is None:
            if cfg is not None:
                try:
                    rho_expected = float(cfg.mat.rho)
                    self.rho_source = "cfg.mat.rho"
                except Exception:
                    rho_expected = None
            if rho_expected is None:
                try:
                    from agforge.options import TeleopOptions
                    rho_expected = float(
                        TeleopOptions.model_fields["mat"].default.rho)
                    self.rho_source = "TeleopOptions.mat default rho"
                except Exception:
                    raise RuntimeError(
                        "cannot determine the configured density; pass "
                        "rho_expected= or cfg=. Refusing to check against a "
                        "remembered constant, which is how the first run of "
                        "this check produced a false alarm.")
        self.rho_expected = float(rho_expected)

        # Fail loudly at construction, not silently at analysis time.
        self.check = self_check(
            self.mass_real, self.n_particles, self.particle_volume_real,
            rho_expected=self.rho_expected)
        self.check["rho_source"] = self.rho_source

        if cp_seg_params is None:
            try:
                from agforge.thermal import CP_316L_SEG_PARAMS
                cp_seg_params = CP_316L_SEG_PARAMS
                self.cp_source = "agforge.thermal.CP_316L_SEG_PARAMS"
            except Exception:
                self.cp_source = "FALLBACK_CONSTANT_500"
        else:
            self.cp_source = "caller-supplied"
        self.cp_seg_params = cp_seg_params

        self.has_thermal = bool(getattr(solver, "_enable_thermal", False))

        # Elastic constants, for the recoverable half of mechanical IE. Read
        # from the built material where possible so they cannot drift from what
        # the scene actually ran.
        self.mu = self.lam = None
        self.elastic_source = "UNAVAILABLE"
        mat = getattr(entity, "material", None)
        for src, obj in (("entity.material", mat), ("cfg.mat", getattr(cfg, "mat", None))):
            if obj is None:
                continue
            E = getattr(obj, "E", None)
            nu = getattr(obj, "nu", None)
            if E is not None and nu is not None:
                self.mu, self.lam = lame_from_E_nu(E, nu)
                self.elastic_source = "%s (E=%.4g, nu=%.4g)" % (src, float(E), float(nu))
                break
            if getattr(obj, "mu", None) is not None and getattr(obj, "lam", None) is not None:
                self.mu, self.lam = float(obj.mu), float(obj.lam)
                self.elastic_source = "%s (mu, lam direct)" % src
                break

        # The supersonic guard. This defines "unstable" for every stability
        # claim this project has made, and it has never been varied, so an arm
        # that trips it may be failing an arbitrary threshold rather than
        # diverging. Recording the threshold alongside the observed speeds makes
        # the headroom computable after the fact, which is what tells a real
        # divergence apart from a guard trip.
        self.v_guard = None
        if cfg is not None:
            try:
                self.v_guard = float(cfg.safety.max_particle_velocity)
            except Exception:
                self.v_guard = None
        if self.v_guard is None:
            try:
                from agforge.options import TeleopOptions
                self.v_guard = float(
                    TeleopOptions.model_fields["safety"].default.max_particle_velocity)
            except Exception:
                self.v_guard = None

        self.samples = []
        self._prev_E_plastic = None
        self._prev_E_adiabatic = None
        self._adiabatic_running = 0.0
        self._saw_decrease = False
        self._last_step = None
        self._step_gaps = 0

        # Arrhenius fit window, for the clamp-saturation counts. Read from the
        # class default rather than a live instance, since a Johnson-Cook run
        # has no ArrheniusPlasticity object to ask -- the window is a property
        # of the fit, not of the run.
        self.T_fit_min, self.T_fit_max = 1073.15, 1473.15
        try:
            from agforge.materials import ArrheniusPlasticity as _A
            self.T_fit_min = float(_A.model_fields["T_fit_min"].default)
            self.T_fit_max = float(_A.model_fields["T_fit_max"].default)
        except Exception:
            pass

        # Streaming export. Written per sample rather than at the end, so a run
        # that dies at hit 12 of 17 still leaves eleven hits of usable data --
        # which matters here, because arms failing partway through is a normal
        # outcome in this project rather than an exceptional one.
        self.jsonl_path = jsonl_path
        self._jsonl_fh = None
        if jsonl_path:
            self._jsonl_fh = open(jsonl_path, "a", encoding="utf-8")

    def _try(self, name):
        """Call an entity getter if it exists; return None if absent or failing."""
        fn = getattr(self.entity, name, None)
        if fn is None:
            return None
        try:
            return fn()
        except Exception:
            return None

    def sample(self, tag=None, step=None, extra=None):
        """Take one energy sample. Call after EVERY scene.step().

        `extra` is merged into the record BEFORE it is streamed.  Callers must
        pass per-step context (stop reason, strike state) this way rather than
        mutating the returned dict: the returned dict is the in-memory copy, and
        a late mutation lands in the final JSON while the streamed JSONL -- the
        artifact that survives a crash -- keeps the stale version.  That
        happened once already and left every stop_reason null in the durable
        file while the summary looked complete.

        EVERY step, not every Nth.  The plastic channel is accumulated from
        dT_adiabatic, which StrikeController clears before each scene.step().
        A sample taken every Nth step therefore sees only 1/N of the plastic
        heating and silently undercounts internal energy by roughly (1 - 1/N) --
        which inflates KE/IE by a factor of N and would fail the quasi-static
        criterion for a purely bookkeeping reason.  The monitor detects gaps in
        the step sequence and refuses to report a plastic-derived IE if it finds
        any.
        """
        vel = self._try("get_particles_vel")
        if vel is None:
            raise RuntimeError("entity has no get_particles_vel; cannot measure KE")

        rec = {"tag": tag, "step": step, "n_particles": self.n_particles}

        # Gap detection for the accumulation hazard described above.
        if step is not None and self._last_step is not None and step != self._last_step + 1:
            self._step_gaps += 1
        if step is not None:
            self._last_step = step

        # KE OF THE DEFORMING MATERIAL ONLY.
        #
        # This reads the MPM entity's particles. The dies are RIGID bodies in
        # the rigid solver and are not in this array, so they contribute
        # nothing here -- which is the required behaviour. A global kinetic
        # energy would be dominated by two dies travelling at the press speed
        # and would measure the press rather than the billet, making the ratio
        # meaningless. Abaqus flags exactly this: rigid-body KE must be
        # subtracted before the comparison means anything. Here it is never
        # added in the first place.
        rec["KE_J"] = kinetic_energy(vel, self.mass_real)
        rec["KE_scope"] = "MPM particles only; rigid dies excluded by construction"

        # Deformation measure for criterion (iv), taken from particle positions
        # so the monitor can test it without an external geometry series.
        pos = self._try("get_particles_pos")
        if pos is not None:
            p = _as_np(pos).reshape(-1, 3)
            if p.shape[0]:
                rec["span_mm"] = ((p.max(axis=0) - p.min(axis=0)) * 1000.0).tolist()

        v = _as_np(vel).reshape(-1, 3)
        speed = np.sqrt(np.einsum("ij,ij->i", v, v))
        rec["v_max_m_s"] = float(speed.max()) if speed.size else 0.0
        rec["v_mean_m_s"] = float(speed.mean()) if speed.size else 0.0
        # Serves the separate question of whether max_particle_velocity is
        # failing arms by guard rather than by physics: recording headroom lets
        # a run that never approached the guard be told apart from one that
        # repeatedly grazed it.
        rec["v_p99_m_s"] = float(np.percentile(speed, 99.0)) if speed.size else 0.0
        if self.v_guard:
            rec["v_guard_frac"] = rec["v_max_m_s"] / self.v_guard
            # How many particles are within 10% of tripping, not just the one
            # fastest. A single outlier grazing the guard and a broad front
            # approaching it are different failures with different fixes.
            rec["n_near_guard"] = int((speed > 0.9 * self.v_guard).sum())

        temp = self._try("get_particles_temp")
        t = None
        rec["physics_valid"] = True
        if temp is not None:
            t = _as_np(temp).reshape(-1)
            rec["T_mean_K"] = float(t.mean())
            rec["T_min_K"] = float(t.min())
            rec["T_max_K"] = float(t.max())

            # REFUSE RATHER THAN REPORT NONSENSE.
            #
            # A first version of this happily returned E_thermal = -6.07e7 J,
            # which is impossible for m*Cp*T with T in kelvin, because the run
            # had entered the known thermal instability whose documented
            # signature is negative-temperature undershoot. Cp is a piecewise
            # fit in T, so once T goes negative every downstream energy is
            # extrapolated garbage -- including the adiabatic channel that IE is
            # derived from. Marking the sample invalid keeps a failed hit from
            # quietly contaminating a criterion.
            n_bad = int((t <= 0.0).sum())
            if n_bad or not np.all(np.isfinite(t)):
                rec["physics_valid"] = False
                rec["invalid_reason"] = (
                    "%d particles at T <= 0 K (or non-finite): the thermal "
                    "solver has diverged, so Cp(T) and every energy derived "
                    "from it are meaningless for this step" % n_bad)
                rec["E_thermal_J"] = None
            else:
                cp = steel_cp_numpy(t, self.cp_seg_params)
                rec["E_thermal_J"] = thermal_energy(t, self.mass_real, cp)
        else:
            rec["E_thermal_J"] = None
            rec["thermal_channel"] = "ABSENT"

            # In-sim clamp saturation on the temperature axis, recorded even on
            # a Johnson-Cook run. clamp_probe.py measured the PROCESS envelope,
            # which is a different question from how much of the SIM field sits
            # outside the Arrhenius fit window at a given step. Measuring it
            # here answers "if Arrhenius were switched on, how much of the field
            # would be extrapolated or pinned" WITHOUT needing an Arrhenius run
            # -- the temperature field does not depend on which flow rule reads
            # it, because on the JC path it is mechanically inert anyway.
            if rec["physics_valid"]:
                rec["arrhenius_clamp"] = temperature_clamp_saturation(
                    t, self.T_fit_min, self.T_fit_max)

        if not rec["physics_valid"]:
            t = None          # block the adiabatic/IE path below

        # Direct plastic work would be preferable, but plastic_work is NOT
        # exposed to Python anywhere: it has no getter on the entity and the
        # thermal telemetry bundle returns only temp plus the seven dT_* fields.
        # The field accumulates correctly inside the solver and is unreadable
        # from outside it. Adding a getter means editing shared source in two
        # files, so this takes the route that needs no such edit.
        pw = self._try("get_particles_plastic_work")
        if pw is not None:
            rec["E_plastic_direct_J"] = plastic_energy(pw, self.particle_volume_real)
        else:
            rec["E_plastic_direct_J"] = None

        # dT_adiabatic IS exposed, and the kernel identity recovers the plastic
        # energy from it exactly:
        #
        #   dT = 0.9 * vol_work / (rho * Cp)     [p2g_post_constitutive]
        #   so  m*Cp*dT = 0.9 * vol_work * (m/rho) = 0.9 * vol_work * V_real
        #   so  sum(m*Cp*dT) = 0.9 * E_plastic
        #
        # This is an identity, not an approximation -- the Cp and the mass
        # cancel. The only inexactness is that Cp is evaluated here at the
        # end-of-step temperature and in the kernel per substep.
        dta = self._try("get_particles_dT_adiabatic")
        if dta is not None and t is not None:
            cp = steel_cp_numpy(t, self.cp_seg_params)
            e_ad = adiabatic_heating_energy(dta, self.mass_real, cp)
            rec["E_adiabatic_J"] = e_ad
            rec["dT_adiabatic_sum_K"] = float(_as_np(dta).sum())
            self._adiabatic_running += e_ad
        else:
            rec["E_adiabatic_J"] = None
            rec["plastic_channel"] = (
                "ABSENT -- no dT_adiabatic getter and no plastic_work getter, so "
                "internal energy cannot be measured at all. Needs "
                "enable_thermal=True.")

        # THE CLEARING REGIME IS NOT ASSUMED.
        #
        # StrikeController calls clear_thermal_telemetry_buffers() before every
        # scene.step(), which makes dT_adiabatic per-macro-step. A runner that
        # does not clear leaves it cumulative instead. Guessing wrong is not a
        # small error: accumulating an already-cumulative field overstates the
        # internal energy by roughly the number of steps taken, which would make
        # criterion 4 pass spuriously.
        #
        # So both interpretations are carried, and the evidence for which one
        # holds is recorded rather than inferred silently. Plastic heating is
        # non-negative, so a cumulative field can never decrease between
        # samples; any decrease proves the per-step regime.
        if rec.get("E_adiabatic_J") is not None:
            if (self._prev_E_adiabatic is not None
                    and rec["E_adiabatic_J"] < self._prev_E_adiabatic - 1e-12):
                self._saw_decrease = True
            self._prev_E_adiabatic = rec["E_adiabatic_J"]
            # If per-step: cumulative plastic energy is the running sum / 0.9.
            rec["E_plastic_if_per_step_J"] = self._adiabatic_running / TAYLOR_QUINNEY
            # If already cumulative: this sample alone / 0.9.
            rec["E_plastic_if_cumulative_J"] = rec["E_adiabatic_J"] / TAYLOR_QUINNEY
        else:
            rec["E_plastic_if_per_step_J"] = None
            rec["E_plastic_if_cumulative_J"] = None

        # Cross-check, available only if a direct plastic_work getter exists.
        # Without it this monitor DERIVES plastic energy from dT_adiabatic, so
        # the two are no longer independent and there is nothing to check them
        # against. Recorded as unavailable rather than quietly skipped.
        if rec.get("E_plastic_direct_J") is not None and self._prev_E_plastic is not None:
            dEp = rec["E_plastic_direct_J"] - self._prev_E_plastic
            rec["dE_plastic_J"] = dEp
            pred = TAYLOR_QUINNEY * dEp
            if rec.get("E_adiabatic_J") is not None and abs(pred) > 1e-9:
                rec["tq_check_ratio"] = rec["E_adiabatic_J"] / pred
        if rec.get("E_plastic_direct_J") is not None:
            self._prev_E_plastic = rec["E_plastic_direct_J"]

        # Criterion 4. IE is taken as the dissipated plastic work, which is the
        # deformation energy that dominates in forging. Elastic strain energy is
        # NOT included -- computing it exactly needs the solver's specific
        # elastic potential, which has not been verified here. Omitting it makes
        # IE a LOWER bound, so KE/IE is an UPPER bound, so the criterion is
        # tested conservatively. Said out loud because an unstated conservative
        # bias is indistinguishable from an error.
        #
        # The per-step reading is used for the headline ratio because that is
        # the regime under StrikeController; summary() reports the other reading
        # alongside so a wrong regime is visible rather than silent.
        # Elastic strain energy completes MECHANICAL internal energy. Thermal
        # energy is NOT part of this: the quasi-static criterion compares
        # kinetic energy against the energy of DEFORMATION, and m*Cp*T is a
        # different quantity that happens to be far larger (~400 kJ against
        # ~600 J here). Substituting it would make the ratio pass by a factor
        # of a thousand for no physical reason.
        if self.mu is not None:
            Fm = self._try("get_particles_F")
            if Fm is not None:
                rec["E_elastic_J"] = elastic_energy(
                    Fm, self.mu, self.lam, self.particle_volume_real)
            else:
                rec["E_elastic_J"] = None
        else:
            rec["E_elastic_J"] = None

        e_pl = (rec.get("E_plastic_direct_J")
                if rec.get("E_plastic_direct_J") is not None
                else rec.get("E_plastic_if_per_step_J"))
        rec["E_plastic_J"] = e_pl
        ie = None
        if e_pl is not None:
            ie = e_pl + (rec.get("E_elastic_J") or 0.0)
        elif rec.get("E_elastic_J") is not None:
            ie = rec["E_elastic_J"]
        rec["IE_J"] = ie
        rec["IE_complete"] = (e_pl is not None and rec.get("E_elastic_J") is not None)
        rec["KE_over_IE"] = (rec["KE_J"] / ie) if (ie is not None and ie > 0.0) else None

        if extra:
            rec.update(extra)

        self.samples.append(rec)
        if self._jsonl_fh is not None:
            import json as _json
            self._jsonl_fh.write(_json.dumps(rec, default=str) + "\n")
            self._jsonl_fh.flush()
        return rec

    def close(self):
        if self._jsonl_fh is not None:
            self._jsonl_fh.close()
            self._jsonl_fh = None

    def summary(self):
        """Aggregate across samples, with provenance and honest absences."""
        if not self.samples:
            return {"n_samples": 0, "error": "no samples taken"}

        # Every criterion below is computed over VALID samples only. A hit that
        # ends in thermal divergence would otherwise drag its garbage energies
        # into the statistics of the hits that were fine.
        n_invalid = sum(1 for s in self.samples if not s.get("physics_valid", True))
        valid = [s for s in self.samples if s.get("physics_valid", True)]
        if not valid:
            return {"n_samples": len(self.samples), "n_invalid": n_invalid,
                    "error": "every sample invalid; the solver diverged"}
        all_samples, self_samples_backup = self.samples, self.samples
        self.samples = valid
        try:
            out = self._summary_inner()
        finally:
            self.samples = self_samples_backup
        out["n_invalid_samples_excluded"] = n_invalid
        out["step_gaps"] = self._step_gaps
        if self._step_gaps and out.get("IE_source", "").startswith("DERIVED"):
            out["IE_UNRELIABLE"] = (
                "%d gaps in the step sequence. The plastic channel accumulates "
                "dT_adiabatic, which is cleared every step, so skipped steps are "
                "lost energy and IE is UNDERCOUNTED. Every KE/IE number here is "
                "inflated. Re-run sampling every step." % self._step_gaps)
            out["criterion_i_KE_le_5pct_of_IE"]["passes"] = None
            out["criterion_ii_KE_over_IE_lt_0p001_at_steady_state"]["passes"] = None
        if n_invalid:
            out["validity_warning"] = (
                "%d of %d samples were excluded for non-physical temperature. "
                "The run diverged; treat the surviving statistics as covering "
                "only the steps before that." % (n_invalid, len(all_samples)))
        return out

    def _summary_inner(self):
        ke = [s["KE_J"] for s in self.samples]
        ratios = [s["KE_over_IE"] for s in self.samples
                  if s.get("KE_over_IE") is not None]
        vmax = [s["v_max_m_s"] for s in self.samples]

        # CRITERION 4 MUST NOT BE READ OFF THE RAW MAXIMUM.
        #
        # At press onset the dies are already moving while essentially no
        # plastic work has accumulated, so KE/IE is a ratio of two negligible
        # numbers and spikes for arithmetic reasons rather than physical ones.
        # The first run of this monitor reported max = 0.155 at a step where IE
        # was 0.20 J out of a final 600 J, which reads as a decisive failure of
        # quasi-staticness and is nothing of the kind.
        #
        # The defensible statistic is the ratio where the deformation energy is
        # actually established. Reported at peak kinetic energy, which is the
        # most demanding moment that is still physically meaningful.
        ie_final = self.samples[-1].get("IE_J") or 0.0
        established = [s for s in self.samples
                       if s.get("KE_over_IE") is not None
                       and (s.get("IE_J") or 0.0) > 0.25 * ie_final]
        peak = max(self.samples, key=lambda s: s["KE_J"]) if self.samples else None
        est_ratios = [s["KE_over_IE"] for s in established]

        # STEADY STATE = after the transient has decayed. Defined as the samples
        # following the last time KE exceeded 10% of its peak, rather than as a
        # fixed tail fraction, so it tracks the physics instead of the sample
        # count.
        ke_peak = max(ke) if ke else 0.0
        last_hot = 0
        for i, s in enumerate(self.samples):
            if ke_peak > 0 and s["KE_J"] > 0.10 * ke_peak:
                last_hot = i
        steady = self.samples[last_hot + 1:]
        ss_ratios = [s["KE_over_IE"] for s in steady
                     if s.get("KE_over_IE") is not None]

        # (i) KE <= ~5% of IE throughout most of the analysis.
        frac_under_5 = ((sum(1 for r in est_ratios if r <= 0.05) / len(est_ratios))
                        if est_ratios else None)
        # (iii) time rate of change of IE negligible at steady state, expressed
        # as the fractional drift of IE across the steady window.
        ie_drift = None
        if len(steady) >= 2:
            a = steady[0].get("IE_J")
            b = steady[-1].get("IE_J")
            if a and b:
                ie_drift = abs(b - a) / abs(b)
        # (iv) peak deformation constant at steady state.
        span_drift_mm = None
        spans = [s.get("span_mm") for s in steady if s.get("span_mm")]
        if len(spans) >= 2:
            a = np.array(spans[0]); b = np.array(spans[-1])
            span_drift_mm = float(np.max(np.abs(b - a)))

        out = {
            "n_samples": len(self.samples),
            "KE_J_max": max(ke),
            "KE_J_mean": sum(ke) / len(ke),
            "v_max_m_s": max(vmax),
            "KE_scope": "MPM particles only; rigid dies excluded by construction",
            # Headline: the ratio at peak KE, over established deformation.
            "KE_over_IE_at_peak_KE": peak.get("KE_over_IE") if peak else None,
            "KE_over_IE_max_established": max(est_ratios) if est_ratios else None,
            "KE_over_IE_mean_established": (
                sum(est_ratios) / len(est_ratios)) if est_ratios else None,
            "n_established_samples": len(established),

            # ---- the four published quasi-static criteria ----
            "criterion_i_KE_le_5pct_of_IE": {
                "frac_established_samples_under_0.05": frac_under_5,
                "max_established": max(est_ratios) if est_ratios else None,
                "passes": (frac_under_5 is not None and frac_under_5 >= 0.95),
                "source": "Abaqus: KE should not exceed a small fraction "
                          "(typically 5-10%) of IE throughout most of the process",
            },
            "criterion_ii_KE_over_IE_lt_0p001_at_steady_state": {
                "n_steady_samples": len(steady),
                "max_at_steady_state": max(ss_ratios) if ss_ratios else None,
                "passes": (max(ss_ratios) < 0.001) if ss_ratios else None,
            },
            "criterion_iii_dIE_dt_negligible_at_steady_state": {
                "IE_fractional_drift_over_steady_window": ie_drift,
                "passes": (ie_drift is not None and ie_drift < 0.01),
            },
            "criterion_iv_peak_deformation_constant_at_steady_state": {
                "max_span_drift_mm": span_drift_mm,
                "passes": (span_drift_mm is not None and span_drift_mm < 0.1),
            },
            "steady_state_definition": (
                "samples after the last step where KE exceeded 10% of peak KE"),

            # Kept only so the artifact stays visible rather than being silently
            # dropped; do NOT use this for any criterion.
            "KE_over_IE_max_raw_DO_NOT_USE": max(ratios) if ratios else None,
            "IE_basis": ("MECHANICAL internal energy = accumulated plastic work "
                         "+ elastic strain energy. THERMAL energy (m*Cp*T) is "
                         "deliberately excluded: it is a different quantity and "
                         "is ~1000x larger here, so substituting it would make "
                         "every criterion pass for no physical reason."),
            "IE_complete": self.samples[-1].get("IE_complete"),
            "elastic_source": self.elastic_source,
            "has_thermal": self.has_thermal,
            "cp_source": self.cp_source,
            "mass_provenance": self.mass_prov,
            "mass_self_check": self.check,
        }

        # Velocity-guard headroom. max_particle_velocity has never been varied,
        # yet it is what "unstable" means in every stability claim here.
        # Worst-case in-sim clamp saturation across the run.
        cl = [s["arrhenius_clamp"] for s in self.samples if s.get("arrhenius_clamp")]
        if cl:
            out["arrhenius_clamp_worst"] = {
                "max_frac_below_T_fit_min": max(c["frac_below"] for c in cl),
                "max_frac_above_T_fit_max": max(c["frac_above"] for c in cl),
                "max_frac_clamped": max(c["frac_clamped"] for c in cl),
                "coldest_T_K": min(c["T_min_K"] for c in cl),
                "T_fit_min": self.T_fit_min,
                "T_fit_max": self.T_fit_max,
                "note": ("measured on the SIM field. Applies whichever flow "
                         "rule ran: the temperature field is the same, and on "
                         "the Johnson-Cook path it is mechanically inert, so "
                         "this is what Arrhenius WOULD see if switched on."),
            }

        out["v_guard_m_s"] = self.v_guard
        if self.v_guard:
            fracs = [s.get("v_guard_frac", 0.0) for s in self.samples]
            near = [s.get("n_near_guard", 0) for s in self.samples]
            out["v_guard_frac_max"] = max(fracs)
            out["n_steps_within_10pct_of_guard"] = sum(1 for f in fracs if f > 0.9)
            out["max_particles_near_guard"] = max(near) if near else 0
            out["v_guard_note"] = (
                "v_guard_frac_max well below 1.0 means the guard was never in "
                "play and cannot explain a failure. Approaching 1.0 means the "
                "run was decided by this threshold, and since the threshold has "
                "never been swept, that is a guard trip and not established "
                "physics. max_particles_near_guard separates a lone outlier "
                "from a broad front.")
        last = self.samples[-1]
        for k in ("E_plastic_direct_J", "E_thermal_J", "plastic_channel",
                  "thermal_channel"):
            if k in last:
                out[k] = last[k]

        # Both readings of the dT_adiabatic clearing regime, side by side, so a
        # wrong assumption shows up as a large gap instead of a plausible number.
        out["E_plastic_if_per_step_J"] = last.get("E_plastic_if_per_step_J")
        out["E_plastic_if_cumulative_J"] = last.get("E_plastic_if_cumulative_J")
        out["IE_source"] = (
            "direct plastic_work getter"
            if last.get("E_plastic_direct_J") is not None
            else "DERIVED from dT_adiabatic via the 0.9 Taylor-Quinney identity "
                 "(plastic_work has no Python accessor)")
        out["dT_adiabatic_regime"] = {
            "assumed": "per_step",
            "evidence_decrease_seen": self._saw_decrease,
            "note": ("A decrease between samples proves per-step, because "
                     "plastic heating is non-negative and a cumulative field "
                     "cannot fall. No decrease seen is NOT proof of cumulative "
                     "-- a monotonically rising per-step series looks the same. "
                     "If the runner does not clear the buffers, read "
                     "E_plastic_if_cumulative_J instead, and note that "
                     "criterion 4 would then be roughly n_samples times "
                     "easier to pass than it should be."),
        }

        ratios_cum = [
            (s["KE_J"] / s["E_plastic_if_cumulative_J"])
            for s in self.samples
            if s.get("E_plastic_if_cumulative_J")
        ]
        out["KE_over_IE_max_if_cumulative"] = max(ratios_cum) if ratios_cum else None

        tq = [s["tq_check_ratio"] for s in self.samples
              if s.get("tq_check_ratio") is not None]
        if tq:
            out["tq_check_ratio_median"] = float(np.median(tq))
            out["tq_check_n"] = len(tq)
            out["tq_check_note"] = (
                "should be ~1.0; compares the solver's dT_adiabatic against "
                "0.9 * delta plastic_work through independent code paths. Small "
                "deviation is expected because Cp is evaluated at the "
                "end-of-step temperature here and per-substep in the kernel.")
        else:
            out["tq_check_ratio_median"] = None
            out["tq_check_note"] = (
                "UNAVAILABLE, and this is a real gap rather than a skipped "
                "nicety. Internal energy here is DERIVED from dT_adiabatic, so "
                "it cannot be checked against dT_adiabatic. Independence "
                "returns only if plastic_work gains a Python accessor -- see "
                "the note in sample(). Until then the plastic channel is "
                "measured, not verified.")
        return out


# --------------------------------------------------------------------------
# in-sim clamp saturation, temperature axis
# --------------------------------------------------------------------------

def temperature_clamp_saturation(temp, T_fit_min, T_fit_max):
    """Count particles pinned against the Arrhenius temperature clamp.

    clamp_probe.py already measured the PROCESS envelope -- what temperatures
    the real forge visits.  This is the different, still-unmeasured question:
    how many particles IN THE SIM are sitting on the clamp, on which side, at a
    given moment.  A process envelope that mostly clears the floor can still
    have a cold surface layer pinned against it every step.

    The floor matters more than the ceiling.  ArrheniusPlasticity's own
    docstring says that below T_fit_min it returns the T_fit_min flow stress,
    which "UNDERSTATES cold strength badly" -- so a pinned particle is not
    merely inaccurate, it is systematically too soft, and it will over-deform.

    Only meaningful when use_arrhenius is on.  Under the default Johnson-Cook
    path there is no such clamp; the relevant pathology there is different --
    T_star pins at zero below jc_T_ref, killing the temperature derivative
    outright rather than saturating a fit window.
    """
    t = _as_np(temp).reshape(-1)
    n = t.size
    if n == 0:
        return {"n": 0}
    below = int((t < T_fit_min).sum())
    above = int((t > T_fit_max).sum())
    return {
        "n_particles": n,
        "n_below_T_fit_min": below,
        "n_above_T_fit_max": above,
        "frac_below": below / n,
        "frac_above": above / n,
        "frac_clamped": (below + above) / n,
        "T_fit_min": float(T_fit_min),
        "T_fit_max": float(T_fit_max),
        "T_min_K": float(t.min()),
        "T_max_K": float(t.max()),
        "note": ("frac_below is the one that bites: below T_fit_min the flow "
                 "stress is held at its T_fit_min value, which understates cold "
                 "strength, so those particles deform too easily."),
    }


# --------------------------------------------------------------------------
# die work -- the "energy in" side of the budget
# --------------------------------------------------------------------------

def capture_die_state(controller):
    """Read the die forces and the full DOF vector for one step.

    Returns a dict, or None if the handles are not reachable.  Deliberately
    captures the RAW series rather than computing work here: which DOF indices
    are the dies, and which sign means closing, are conventions this module has
    not verified.  Guessing them would produce a confident work number that
    could be wrong in sign or by a factor, which is worse than no number.
    Instrumentation records; analysis interprets.
    """
    robot = getattr(controller, "robot", None)
    if robot is None:
        return None
    out = {}
    try:
        fL, fR = robot.get_resistance_forces()
        out["force_L_N"] = float(_as_np(fL).reshape(-1)[0])
        out["force_R_N"] = float(_as_np(fR).reshape(-1)[0])
    except Exception:
        return None
    try:
        qpos = robot.entity.get_dofs_position()
        out["qpos"] = _as_np(qpos).reshape(-1).tolist()
    except Exception:
        out["qpos"] = None
    return out


def die_work_from_series(records, dof_indices=None):
    """Integrate die work from a captured series, inferring the sign convention.

    records : list of capture_die_state() dicts, in step order
    dof_indices : (i_L, i_R) DOF indices for the two dies. If None, they are
        inferred as the two DOFs whose motion correlates most strongly with
        periods of nonzero resistance force -- a die that is not moving while
        the billet pushes back is not the DOF doing the work.

    Work done BY a die ON the billet is force times displacement INTO the
    billet.  The resistance force is defined positive when opposing the
    squeeze, so the closing direction is whichever sign of DOF motion
    coincides with positive force.  That is measured here and reported, not
    assumed.

    Returns a dict including the inferred convention, so a reader can reject
    the number if the inference looks wrong.
    """
    recs = [r for r in records if r and r.get("qpos")]
    if len(recs) < 2:
        return {"error": "need at least 2 records with qpos", "n": len(recs)}

    q = np.array([r["qpos"] for r in recs], dtype=np.float64)     # (T, ndof)
    fL = np.array([r.get("force_L_N", 0.0) for r in recs], dtype=np.float64)
    fR = np.array([r.get("force_R_N", 0.0) for r in recs], dtype=np.float64)
    dq = np.diff(q, axis=0)                                       # (T-1, ndof)
    f_mid = 0.5 * (np.abs(fL[:-1]) + np.abs(fR[:-1]))

    if dof_indices is None:
        # Which DOFs actually move while the billet is resisting?
        loaded = f_mid > (0.05 * f_mid.max() if f_mid.max() > 0 else np.inf)
        if loaded.sum() < 2:
            return {"error": "no loaded steps; cannot infer die DOFs",
                    "n": len(recs)}
        motion = np.abs(dq[loaded]).sum(axis=0)
        order = np.argsort(motion)[::-1]
        dof_indices = (int(order[0]), int(order[1]))
        inferred = True
        motion_ranking = motion.tolist()
    else:
        inferred = False
        motion_ranking = None

    iL, iR = int(dof_indices[0]), int(dof_indices[1])

    def _work(dq_col, f):
        # Sign convention measured, not assumed: the closing direction is the
        # one that dominates while the force is high.
        loaded = np.abs(f) > (0.05 * np.abs(f).max() if np.abs(f).max() > 0 else np.inf)
        if loaded.sum() == 0:
            return 0.0, 0.0, +1
        net = float(dq_col[loaded].sum())
        sign = -1 if net < 0 else +1
        closing = dq_col * sign          # positive where closing
        w = float(np.sum(np.abs(f) * np.clip(closing, 0.0, None)))
        w_signed = float(np.sum(np.abs(f) * closing))
        return w, w_signed, sign

    wL, wL_signed, sL = _work(dq[:, iL], fL[:-1])
    wR, wR_signed, sR = _work(dq[:, iR], fR[:-1])

    return {
        "W_die_in_J": wL + wR,
        "W_die_signed_J": wL_signed + wR_signed,
        "W_L_J": wL,
        "W_R_J": wR,
        "dof_indices": [iL, iR],
        "dof_indices_inferred": inferred,
        "dof_motion_ranking": motion_ranking,
        "closing_sign_L": sL,
        "closing_sign_R": sR,
        "n_steps": len(recs),
        "note": ("W_die_in_J counts only closing motion; W_die_signed_J lets "
                 "recoil return energy. A large gap between them means the "
                 "dies rebound substantially, which matters for the budget. "
                 "DOF indices and closing signs are MEASURED from the series; "
                 "check dof_motion_ranking looks like two dominant DOFs before "
                 "trusting the work number."),
    }


def budget_residual(W_die_in_J, KE_J, IE_J, E_thermal_delta_J=None):
    """Close the energy budget and report the residual as a fraction of input.

    This is the measurement that separates the two hypotheses for a result that
    does not converge in timestep. If the extra deformation at a finer timestep
    is PHYSICAL, the residual stays near zero at every timestep and the work
    input rises to match the extra plastic dissipation. If it is NUMERICAL, the
    residual grows with the timestep, because the integrator is creating or
    destroying energy that no work input accounts for.

    A single run's residual means little; the SLOPE across timesteps is the
    answer.
    """
    if not W_die_in_J:
        return {"error": "no die work input"}
    accounted = (KE_J or 0.0) + (IE_J or 0.0)
    resid = W_die_in_J - accounted
    return {
        "W_die_in_J": W_die_in_J,
        "KE_J": KE_J,
        "IE_J": IE_J,
        "accounted_J": accounted,
        "residual_J": resid,
        "residual_frac": resid / W_die_in_J,
        "interpretation": ("compare residual_frac ACROSS timesteps: flat and "
                           "near zero => physical; growing with dt => the "
                           "integrator is not conserving energy. Note IE here "
                           "excludes elastic strain energy, so a modest "
                           "positive residual is expected even when conserving."),
    }


if __name__ == "__main__":
    # Numeric-core smoke test. Runs without genesis, a GPU, or a scene.
    import json

    n = 1000
    m = 1.58e-5                      # kg, ~2 mm particle of 7900 kg/m^3 steel
    vol = m / 7900.0
    v = np.zeros((n, 3))
    v[:, 0] = 2.0
    assert abs(kinetic_energy(v, m) - 0.5 * m * 4.0 * n) < 1e-12

    w = np.full(n, 1.0e6)            # 1 MJ/m^3 specific plastic work
    assert abs(plastic_energy(w, vol) - 1.0e6 * vol * n) < 1e-6

    t = np.full(n, 1200.0)
    cp = np.full(n, 500.0)
    assert abs(thermal_energy(t, m, cp) - m * 500.0 * 1200.0 * n) < 1e-9

    # The adiabatic cross-check must be exact when Cp is held fixed:
    # sum(m*Cp*dT) with dT = 0.9*w/(rho*Cp) equals 0.9 * w * V_real * n.
    rho = 7900.0
    dT = 0.9 * w / (rho * cp)
    lhs = adiabatic_heating_energy(dT, m, cp)
    rhs = TAYLOR_QUINNEY * plastic_energy(w, vol)
    assert abs(lhs - rhs) / rhs < 1e-12, (lhs, rhs)

    # Elastic energy, against closed forms.
    E_mod, nu_p = 121.5e9, 0.383
    mu_, lam_ = lame_from_E_nu(E_mod, nu_p)
    kappa_ = lam_ + 2.0 * mu_ / 3.0
    # Undeformed: exactly zero.
    F_I = np.tile(np.eye(3), (n, 1, 1))
    assert abs(elastic_energy(F_I, mu_, lam_, vol)) < 1e-9
    # Pure dilation: only the volumetric term survives, W = kappa/2 * (3 ln s)^2.
    s = 1.001
    F_d = np.tile(np.eye(3) * s, (n, 1, 1))
    want = 0.5 * kappa_ * (3.0 * math.log(s)) ** 2 * vol * n
    got = elastic_energy(F_d, mu_, lam_, vol)
    assert abs(got - want) / want < 1e-9, (got, want)
    # Pure shear-free deviatoric stretch: trace(log) = 0, only the mu term.
    F_s = np.tile(np.diag([s, 1.0 / s, 1.0]), (n, 1, 1))
    eps_ = np.array([math.log(s), -math.log(s), 0.0])
    want_s = mu_ * float(eps_ @ eps_) * vol * n
    got_s = elastic_energy(F_s, mu_, lam_, vol)
    assert abs(got_s - want_s) / want_s < 1e-9, (got_s, want_s)

    # The 1000x trap, both directions. RHO here is the density this synthetic
    # particle was built from, so the check is an identity, exactly as it is
    # against cfg.mat.rho on a real scene.
    RHO = 7900.0
    for bad, name in ((m * 1000.0, "scale applied twice"),
                      (m / 1000.0, "scale never applied")):
        try:
            self_check(bad, n, vol, rho_expected=RHO)
        except AssertionError:
            pass
        else:
            raise SystemExit("self_check FAILED to catch: %s" % name)

    ok = self_check(m, n, vol, rho_expected=RHO)
    print(json.dumps({"smoke": "PASS",
                      "tq_crosscheck_rel_err": abs(lhs - rhs) / rhs,
                      "self_check": ok}, indent=2))
