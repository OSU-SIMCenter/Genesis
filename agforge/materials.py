
import quadrants as qd
import genesis as gs


@qd.data_oriented
class JohnsonCookPlasticity(gs.materials.MPM.Base):
    """
    Johnson-Cook elasto-plastic material for MPM.
    
    Flow Stress: sigma_y = (A + B * eps_p^n) * (1 + C * ln(eps_dot_star))
    
    Utilizes 'Jp' particle field to store accumulated equivalent plastic strain (epsilon_p).
    """

    def __init__(
        self,
        E=200e9,      # Young's modulus (Steel)
        nu=0.3,       # Poisson's ratio
        rho=7850.0,   # Density
        A=792e6,      # Initial Yield (Pa)
        B=510e6,      # Hardening Coeff (Pa)
        n=0.26,       # Hardening Exponent
        C=0.014,      # Strain Rate Sensitivity
        eps0=1.0,     # Reference Strain Rate
        T_ref=293.15, # Reference Temperature (K)
        T_melt=1793.0,# Melting Point (K)
        jc_m=1.03,    # Thermal Softening Exponent
        eta_over_dt=0.0, # Pre-computed viscosity per dt
        sampler=None,
    ):
        super().__init__(E, nu, rho, sampler=sampler)

        self._A = A
        self._B = B
        self._n = n
        self._C = C
        self._eps0 = eps0
        self._T_ref = T_ref
        self._T_melt = T_melt
        self._jc_m = jc_m
        self._eta_over_dt = eta_over_dt
        
        # Initial Jp (Plastic Strain) = 0.0
        self._default_Jp = 0.0

    @qd.func
    def update_F_S_Jp(self, J, F_tmp, U, S, V, Jp, temp):
        """
        Updates Deformation Gradient (F), Singular Values (S), and Plastic Strain (Jp).
        Jp here stores Equivalent Plastic Strain (epsilon_p).
        """
        F_new = qd.Matrix.zero(gs.qd_float, 3, 3)
        S_new = qd.Matrix.zero(gs.qd_float, 3, 3)
        delta_gamma = 0.0
        
        # 1. Trial Deviatoric Strain (Elastic Predictor)
        S_clamped = qd.max(S, 0.05)  # Limit to avoid log(0) and extreme artificial stress
        epsilon = qd.Vector([qd.math.log(S_clamped[0, 0]), qd.math.log(S_clamped[1, 1]), qd.math.log(S_clamped[2, 2])])
        
        trace_eps = epsilon.sum()
        epsilon_hat = epsilon - (trace_eps / 3.0)
        epsilon_hat_norm = epsilon_hat.norm(gs.EPS)
        
        # 2. Thermal softening (Johnson-Cook melting term)
        T_star = qd.math.clamp((temp - self._T_ref) / (self._T_melt - self._T_ref), gs.qd_float(0.0), gs.qd_float(1.0))
        thermal_softening = gs.qd_float(1.0) - qd.math.pow(qd.math.max(T_star, gs.qd_float(1e-8)), self._jc_m)
        # 3. Yield Condition
        eps_p = qd.math.max(Jp, gs.qd_float(1e-6))  # Guard for pow() edge case
        sigma_y = (self._A + self._B * qd.math.pow(eps_p, self._n)) * thermal_softening
        yield_dist = epsilon_hat_norm - sigma_y / (2.0 * self._mu)
        Jp_new = Jp
        
        if yield_dist > 0:  # Yields
            delta_gamma = gs.qd_float(0.0)
            two_mu = gs.qd_float(2.0) * self._mu
            
            # K controls the exponential growth of viscosity as the metal cools.
            # A value of K=10.0 means cold steel is ~22,000x more viscous than hot steel.
            K = gs.qd_float(10.0) 
            
            # eta_dt grows exponentially as T_star approaches 0 (cold)
            eta_dt = self._eta_over_dt * qd.math.exp(K * (gs.qd_float(1.0) - T_star))
            
            # --- 1. Bisection (Robust Bracket Search - 10 iterations) ---
            g_low = gs.qd_float(0.0)
            g_high = epsilon_hat_norm
            for _ in qd.static(range(10)):
                g_mid = gs.qd_float(0.5) * (g_low + g_high)
                eps_p_mid = qd.math.max(Jp + g_mid, gs.qd_float(1e-6))
                sy_mid = (self._A + self._B * qd.math.pow(eps_p_mid, self._n)) * thermal_softening
                R_mid = epsilon_hat_norm - g_mid - sy_mid / two_mu - eta_dt * g_mid
                
                # Branchless bracket update
                if R_mid > 0.0:
                    g_low = g_mid
                else:
                    g_high = g_mid

            delta_gamma = gs.qd_float(0.5) * (g_low + g_high)
            
            # --- 2. Newton-Raphson Polish (Quadratic Convergence - 3 iterations) ---
            for _ in qd.static(range(3)):
                eps_p_trial = qd.math.max(Jp + delta_gamma, gs.qd_float(1e-6))
                sy = (self._A + self._B * qd.math.pow(eps_p_trial, self._n)) * thermal_softening
                R = epsilon_hat_norm - delta_gamma - sy / two_mu - eta_dt * delta_gamma
                R_prime = gs.qd_float(-1.0) - (self._n * self._B * qd.math.pow(eps_p_trial, self._n - gs.qd_float(1.0))) * thermal_softening / two_mu - eta_dt
                step = R / qd.math.min(R_prime, gs.qd_float(-1e-10))  # Prevent division by zero
                delta_gamma = qd.math.clamp(delta_gamma - step, gs.qd_float(0.0), epsilon_hat_norm)
            Jp_new = Jp + delta_gamma

            # Apply return mapping to strain
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
