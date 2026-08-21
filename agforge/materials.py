from typing import Any
import quadrants as qd
import genesis as gs
from genesis.typing import ValidFloat

# --------------------------------------------------------------------------
# [Song2020] Table 1 - strain-compensated Arrhenius constants for 316L.
# Mirrors agforge/material_properties_mechanical.py, which is the CPU reference
# implementation and the thing the tests check this kernel against. Kept as
# module-level tuples so they resolve at kernel-compile time.
# alpha in 1/MPa, Q in J/mol, A in 1/s.
# --------------------------------------------------------------------------
_R_GAS = 8.314
_ARR_STRAIN0 = 0.05      # first tabulated strain
_ARR_DSTRAIN = 0.05      # uniform spacing
_ARR_ALPHA = (0.0091, 0.0081, 0.0076, 0.0073, 0.0071, 0.0070, 0.0071, 0.0073, 0.0081)
_ARR_N = (6.0452, 5.4551, 5.2624, 5.1774, 5.0818, 5.0140, 4.9624, 4.8867, 5.2208)
_ARR_Q = (476670.0, 446370.0, 433830.0, 430110.0, 426470.0, 426650.0,
          429750.0, 432760.0, 458300.0)
_ARR_A = (7.3454e17, 4.737e16, 1.493e16, 1.069e16, 7.41e15, 7.49e15,
          9.96e15, 1.21e16, 9.634e16)

@qd.data_oriented
class JohnsonCookPlasticity(gs.materials.MPM.Base):
    """
    Johnson-Cook elasto-plastic material for MPM.
    
    Flow Stress: sigma_y = (A + B * eps_p^n) * (1 + C * ln(eps_dot_star))
    
    Utilizes 'Jp' particle field to store accumulated equivalent plastic strain (epsilon_p).
    """

    # 316L at the forging operating point (1000 C, 1/s), matching the sourced card
    # in MaterialOptions. These used to be AISI 4340's room-temperature constants
    # (A=792e6, B=510e6, n=0.26, C=0.014, T_ref=293.15, T_melt=1793, m=1.03), which
    # were harmless only because environment.py overrides every one of them. They
    # are defaults for a 316L billet now, so a partial construction cannot silently
    # fall back to the wrong alloy. See docs/316L_MECHANICAL_PROPERTIES.md.
    A: ValidFloat = 100.3e6
    B: ValidFloat = 195.0e6
    n: ValidFloat = 0.417
    #: Currently INERT - the kernel below never reads it; see _update_F_S_Jp_jc.
    C: ValidFloat = 0.120
    eps0: ValidFloat = 1.0
    #: The FORGING temperature, not room temperature: A/B are already the 1000 C
    #: values, so T* must be 0 there or the card gets thermally softened twice.
    T_ref: ValidFloat = 1273.15
    T_melt: ValidFloat = 1675.0
    jc_m: ValidFloat = 1.0

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        self._default_Jp = 0.0
        self.update_F_S_Jp = self._update_F_S_Jp_jc

    @qd.func
    def _update_F_S_Jp_jc(self, J, F_tmp, U, S, V, Jp, temp):
        """
        Updates Deformation Gradient (F), Singular Values (S), and Plastic Strain (Jp).
        Jp here stores Equivalent Plastic Strain (epsilon_p).
        """
        F_new = qd.Matrix.zero(gs.qd_float, 3, 3)
        S_new = qd.Matrix.zero(gs.qd_float, 3, 3)
        delta_gamma = gs.qd_float(0.0)
        
        # 1. Trial Deviatoric Strain (Elastic Predictor)
        S_clamped = qd.max(S, 1e-6)  # Limit to avoid log(0)
        epsilon = qd.Vector([qd.math.log(S_clamped[0, 0]), qd.math.log(S_clamped[1, 1]), qd.math.log(S_clamped[2, 2])])
        
        trace_eps = epsilon.sum()
        epsilon_hat = epsilon - (trace_eps / 3.0)
        epsilon_hat_norm = epsilon_hat.norm(gs.EPS)
        
        # 2. Determine Flow Stress (Sigma_y)
        eps_p = Jp
        sigma_y_static = self.A + self.B * qd.math.pow(eps_p, self.n)
        
        # Thermal softening (Johnson-Cook melting term)
        T_star = qd.math.clamp((temp - self.T_ref) / (self.T_melt - self.T_ref), gs.qd_float(0.0), gs.qd_float(1.0))
        thermal_softening = gs.qd_float(1.0) - qd.math.pow(qd.math.max(T_star, gs.qd_float(1e-8)), self.jc_m)
        sigma_y = sigma_y_static * thermal_softening
        
        # 3. Yield Condition (Von Mises)
        yield_dist = epsilon_hat_norm - sigma_y / (2.0 * self.mu)
        Jp_new = Jp 
        
        if yield_dist > 0:  # Yields
            delta_gamma = yield_dist 
            Jp_new = eps_p + delta_gamma
            
            epsilon -= (delta_gamma / epsilon_hat_norm) * epsilon_hat
            
            # Reconstruct S
            S_new = qd.Matrix.zero(gs.qd_float, 3, 3)
            for d in qd.static(range(3)):
                S_new[d, d] = qd.math.exp(epsilon[d])
            
            F_new = U @ S_new @ V.transpose()
            
        else:  # Elastic
            F_new = F_tmp
            S_new = S
            delta_gamma = gs.qd_float(0.0)
        
        return F_new, S_new, Jp_new, delta_gamma, sigma_y


@qd.data_oriented
class ArrheniusPlasticity(gs.materials.MPM.Base):
    """316L hot-working plasticity: strain-compensated hyperbolic-sine Arrhenius.

    sigma = (1/alpha) * asinh[ (Z/A)^(1/n) ],   Z = eps_dot * exp(Q / (R*T))

    with alpha, n, Q, A tabulated against plastic strain ([Song2020] Table 1).
    This is the model the literature endorses for 316L in the DRX regime, where
    Johnson-Cook reports ~48% AARE against Arrhenius' ~7.7%. Unlike the
    Johnson-Cook path in this module it is genuinely rate- AND
    temperature-coupled, and it reproduces the DRX peak-then-soften flow curve
    that JC structurally cannot.

    Like JohnsonCookPlasticity, 'Jp' stores accumulated equivalent plastic strain.

    IMPORTANT - validity window. The fit is a HOT-WORKING model, calibrated over
    800-1000 C. It is meaningless outside that range: extrapolated to room
    temperature it returns ~3.8 GPa (~18x the forging-temperature value), and the
    exp() argument reaches 176, which overflows float32 above ~88. Temperature is
    therefore CLAMPED to [T_fit_min, T_fit_max]; clamping also caps the exponent
    near 53, keeping the whole evaluation inside float32 range.

    That clamp is a guard, not a physical model: below T_fit_min this returns the
    T_fit_min flow stress, which UNDERSTATES cold strength badly. A billet that
    is not at forging temperature is outside what this material can describe -
    see the note on default_initial_temperature in options.py.
    """

    #: Validity window of the [Song2020] fit. See the class docstring - these are
    #: guards against unphysical extrapolation, not tunables.
    T_fit_min: ValidFloat = 1073.15
    #: Raised 1273.15 -> 1473.15 on 2026-08-07. [Song2020] is fitted to 1000 C,
    #: and clamping there meant a billet at real forging heat (1150-1260 C) was
    #: evaluated with the 1000 C flow stress: 181.1 MPa at eps 0.207, 0.41 /s,
    #: where extrapolating Song's own form gives 77.8 MPa. A 2.3x error coming
    #: entirely from the clamp.
    #:
    #: Extrapolating is corroborated, not assumed. [RyanMcQueen1990] is type 316
    #: torsion measured IN DOMAIN over 900-1200 C, and at 1200 C / 0.41 /s it
    #: brackets 41.6 - 74.3 MPa (eps 0.1 to DRV saturation). Song extrapolated
    #: gives 77.8 -- just above that bracket, the same ~5-10% offset it shows at
    #: 1000 C where both fits apply. The temperature dependence tracks; Song
    #: simply reads slightly stiffer throughout, consistent with a different
    #: material state and test mode.
    #:
    #: 1473.15 K is exactly where Ryan & McQueen's window ends. Do NOT raise it
    #: further without a source that reaches higher - above 1200 C the
    #: extrapolation is unchecked again. See material_properties_mechanical.
    T_fit_max: ValidFloat = 1473.15

    #: Substep timestep [s], used to turn the plastic strain increment into a
    #: strain rate. Must be kept in sync with the solver's substep_dt; the
    #: environment sets it from MaterialOptions at construction time.
    substep_dt: ValidFloat = 1.208297e-06

    #: Strain-rate clamp. The fit domain tops out near 1e-2 /s and forging runs
    #: 1-100 /s, so the upper end is already an extrapolation - see
    #: docs/316L_MECHANICAL_PROPERTIES.md on the +14% check at 100 /s.
    rate_min: ValidFloat = 1e-4
    rate_max: ValidFloat = 1e3

    #: Seed rate for the first pass of the rate fixed-point iteration. The flow
    #: stress is very weakly rate-sensitive (~1.75x per 100x rate), so a single
    #: correction iteration from this seed is more than enough to converge.
    rate_seed: ValidFloat = 1.0

    #: TIME-SCALE DIVISOR N applied to the DERIVED rate. This is the similarity
    #: transform, and it is the difference between a rate-coupled model and a
    #: rate-anchored one.
    #:
    #: The press runs at pressing_speed for numerical affordability, so
    #:     N = pressing_speed / real_die_speed
    #: and the rate the solver timestep implies is N times the physical one:
    #:     eps_dot_derived / N = (v/h) / (v/v_real) = v_real/h
    #: which is INVARIANT IN v. Dividing by N therefore anchors the global
    #: magnitude at the real process rate while leaving the LOCAL, per-particle
    #: rate free to vary with deformation -- so contact method still couples to
    #: flow stress, which a contact benchmark requires and which
    #: ``process_strain_rate`` destroys (it applies one scalar to every particle).
    #:
    #: ⚠️ Invariant in v; NOT obviously invariant in dt. The derived rate is
    #: dg_trial/dt, and dg_trial is a trial excess rather than a clean per-step
    #: increment, so its statistics move with the timestep. Treat any
    #: CFL sweep on the rate-sensitive arm as measuring both until that is
    #: checked -- see ``rate_floor``, which removes the worst of it.
    #:
    #: Backed by a RUNTIME field so a sweep over press speed can run many arms in
    #: ONE process. A plain attribute read inside a @qd.func is baked at trace
    #: time, so changing it would force a kernel recompile per arm and destroy
    #: one-process batching -- and scene build is ~93% of run time.
    rate_time_scale: ValidFloat = 1.0

    #: Lower bound on the DERIVED rate, in physical units (i.e. applied after the
    #: rate_time_scale division). 0.0 means "use rate_seed".
    #:
    #: This exists because of a real defect. When the trial state is elastic at
    #: the seed rate, dg_trial <= 0, so rate_est collapses to 0 and the second
    #: pass evaluates the flow stress at rate_min = 1e-4 -- about 3x softer than
    #: the process rate. That can push yield_dist positive and MAKE A PARTICLE
    #: YIELD THAT WAS ELASTIC, then plastify it by the whole trial excess.
    #:
    #: 🚨 And it is CFL-DEPENDENT: near-elastic trial states are more common when
    #: per-step increments are small, so the artefact grows as the timestep
    #: tightens -- on exactly the axis a convergence study varies. An elastic
    #: trial state means "no plastic increment, so no meaningful rate", NOT
    #: "deforming at 1e-4 /s". Flooring at the process rate says the first.
    rate_floor: ValidFloat = 0.0

    #: PRESCRIBED process strain rate [1/s]. When > 0 the flow stress is
    #: evaluated at this rate instead of the one implied by the solver timestep.
    #:
    #: This exists because the SIMULATED strain rate is not the PHYSICAL one.
    #: The press runs at strike.pressing_speed = 25 m/s for numerical
    #: affordability, giving a nominal rate of v/D = 656 /s. The real Agility
    #: Forge press measures 14.1 mm/s => 0.41 /s (2026-06-15 T4 dataset, first
    #: blow: gap 38.41 -> 31.06 mm in 0.505 s). The sim is therefore ~1770x
    #: fast, and deriving the rate from substep_dt feeds this fit a rate ~3e4
    #: ABOVE its fitted domain of 2e-4..2e-2 /s.
    #:
    #: Note rate_max = 1e3 does NOT catch this: it was sized against the real
    #: process ("forging runs 1-100 /s"), so 656 slips underneath the guard.
    #:
    #: Johnson-Cook is unaffected only because its rate term is dead code
    #: (self.C is never referenced), and running fast is otherwise defensible
    #: for it -- inertial stress rho*v^2 ~ 4.6 MPa is ~2% of flow stress. But a
    #: genuinely rate-coupled law cannot use a fictional rate, so decoupling the
    #: constitutive rate from the numerical one is required, not optional. This
    #: is standard practice for quasi-static problems solved with explicit
    #: dynamics.
    #:
    #: 0.0 keeps the legacy behaviour (derive from substep_dt).
    process_strain_rate: ValidFloat = 0.0

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        self._default_Jp = 0.0
        self.update_F_S_Jp = self._update_F_S_Jp_arrhenius
        # Effective floor: 0.0 means "the seed", which is by construction the
        # best available estimate of the process rate.
        self._rate_floor_eff = (
            float(self.rate_floor) if self.rate_floor > 0.0
            else float(self.rate_seed))
        # Runtime field for N. Allocation needs an initialised backend; unit
        # tests construct this class without one, so fall back to the
        # compile-time constant there. Production always takes the field path.
        self._rt_rate_time_scale = None
        try:
            self._rt_rate_time_scale = qd.field(dtype=gs.qd_float, shape=())
            self._rt_rate_time_scale[None] = float(self.rate_time_scale)
        except Exception:
            self._rt_rate_time_scale = None

    def set_rate_time_scale(self, value: float) -> None:
        """Retune N without recompiling the kernel.

        This is the whole reason N is a runtime field: a press-speed sweep at
        fixed CFL varies N while dt stays put, so every arm can share one
        compiled kernel and one built scene.
        """
        value = float(value)
        if not value > 0.0:
            raise ValueError("rate_time_scale must be > 0, got %r" % (value,))
        if self._rt_rate_time_scale is None:
            raise RuntimeError(
                "no runtime field for rate_time_scale -- the backend was not "
                "initialised when this material was constructed, so N is baked "
                "in at %r and changing it would need a recompile"
                % (float(self.rate_time_scale),))
        self._rt_rate_time_scale[None] = value

    @property
    def effective_rate_time_scale(self) -> float:
        """N as the kernel will actually read it."""
        if self._rt_rate_time_scale is not None:
            return float(self._rt_rate_time_scale[None])
        return float(self.rate_time_scale)

    @qd.func
    def _flow_stress_pa(self, eps_p, rate, temp):
        """Arrhenius flow stress in Pa.

        Piecewise-linear in plastic strain across the tabulated points, evaluated
        as a branch-free hat-function sum so it matches numpy.interp on the
        uniform strain grid exactly while staying SIMT friendly.
        """
        tk = qd.math.clamp(temp, gs.qd_float(self.T_fit_min), gs.qd_float(self.T_fit_max))
        r = qd.math.clamp(rate, gs.qd_float(self.rate_min), gs.qd_float(self.rate_max))

        # Position on the tabulated strain grid, clamped to the ends (matches the
        # CPU reference, which clamps plastic_strain into [0.05, 0.45]).
        t = qd.math.clamp(
            (eps_p - gs.qd_float(_ARR_STRAIN0)) / gs.qd_float(_ARR_DSTRAIN),
            gs.qd_float(0.0),
            gs.qd_float(8.0),
        )

        sigma_mpa = gs.qd_float(0.0)
        for k in qd.static(range(9)):
            # NOTE: quadrants.math has no abs(); the builtin lowers correctly.
            w = qd.math.max(
                gs.qd_float(0.0),
                gs.qd_float(1.0) - abs(t - gs.qd_float(float(k))),
            )
            # Skip the transcendentals when this node carries no weight.
            if w > 0.0:
                z = r * qd.math.exp(gs.qd_float(_ARR_Q[k]) / (gs.qd_float(_R_GAS) * tk))
                x = qd.math.pow(z / gs.qd_float(_ARR_A[k]), gs.qd_float(1.0 / _ARR_N[k]))
                # asinh(x) = log(x + sqrt(x^2 + 1)); the DSL has no asinh.
                asinh_x = qd.math.log(x + qd.math.sqrt(x * x + gs.qd_float(1.0)))
                sigma_mpa += w * asinh_x / gs.qd_float(_ARR_ALPHA[k])

        return sigma_mpa * gs.qd_float(1.0e6)

    @qd.func
    def _update_F_S_Jp_arrhenius(self, J, F_tmp, U, S, V, Jp, temp):
        """Return mapping with a rate- and temperature-coupled yield surface.

        Mirrors the Johnson-Cook return map above; only the flow stress differs.
        The rate dependence is resolved by one fixed-point pass: the plastic
        strain increment implies a rate, which sets the flow stress, which sets
        the increment. No new particle field and no solver change are needed -
        the increment is already local to this kernel.
        """
        F_new = qd.Matrix.zero(gs.qd_float, 3, 3)
        S_new = qd.Matrix.zero(gs.qd_float, 3, 3)
        delta_gamma = gs.qd_float(0.0)

        # 1. Trial deviatoric strain (elastic predictor)
        S_clamped = qd.max(S, 1e-6)
        epsilon = qd.Vector([
            qd.math.log(S_clamped[0, 0]),
            qd.math.log(S_clamped[1, 1]),
            qd.math.log(S_clamped[2, 2]),
        ])
        trace_eps = epsilon.sum()
        epsilon_hat = epsilon - (trace_eps / 3.0)
        epsilon_hat_norm = epsilon_hat.norm(gs.EPS)

        eps_p = Jp
        inv_dt = gs.qd_float(1.0) / gs.qd_float(self.substep_dt)

        # 2. Flow stress.
        #    A PRESCRIBED process rate collapses the fixed point to a single
        #    evaluation - but it also applies ONE scalar to every particle, which
        #    discards the local rate and stops contact method coupling to flow
        #    stress. Prefer rate_time_scale; see its note.
        #
        #    Otherwise derive the rate from the timestep: pass 1 seeds it, pass 2
        #    uses the rate the trial state implies, then N-scale and floor it.
        sigma_y = gs.qd_float(0.0)
        if qd.static(self.process_strain_rate > 0.0):
            sigma_y = self._flow_stress_pa(
                eps_p, gs.qd_float(self.process_strain_rate), temp)
        else:
            sigma_y = self._flow_stress_pa(
                eps_p, gs.qd_float(self.rate_seed), temp)
            dg_trial = epsilon_hat_norm - sigma_y / (2.0 * self.mu)
            rate_est = qd.math.max(dg_trial, gs.qd_float(0.0)) * inv_dt
            # Similarity transform: the derived rate is measured in SIM time.
            if qd.static(self._rt_rate_time_scale is not None):
                rate_est = rate_est / self._rt_rate_time_scale[None]
            else:
                rate_est = rate_est / gs.qd_float(self.rate_time_scale)
            # Floor, in physical units. Without it an elastic trial state is read
            # as "1e-4 /s" and softens ~3x, which can yield a particle that was
            # elastic -- and does so more often as dt falls.
            rate_est = qd.math.max(rate_est, gs.qd_float(self._rate_floor_eff))
            sigma_y = self._flow_stress_pa(eps_p, rate_est, temp)

        # 3. Yield condition (von Mises)
        yield_dist = epsilon_hat_norm - sigma_y / (2.0 * self.mu)
        Jp_new = Jp

        if yield_dist > 0:  # Yields
            delta_gamma = yield_dist
            Jp_new = eps_p + delta_gamma

            epsilon -= (delta_gamma / epsilon_hat_norm) * epsilon_hat

            S_new = qd.Matrix.zero(gs.qd_float, 3, 3)
            for d in qd.static(range(3)):
                S_new[d, d] = qd.math.exp(epsilon[d])

            F_new = U @ S_new @ V.transpose()

        else:  # Elastic
            F_new = F_tmp
            S_new = S
            delta_gamma = gs.qd_float(0.0)

        return F_new, S_new, Jp_new, delta_gamma, sigma_y
