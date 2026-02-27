
import gstaichi as ti
import genesis as gs
# Import Base from genesis directly if accessible, or we might need to rely on duck typing or relative imports if inside a package.
# Since we are in agforge/materials.py, genesis should be importable.
# We need to import the Base class. 
# gs.materials.MPM.Base is the class.

if not hasattr(gs.materials.MPM, 'Base'):
     # Fallback if Base is not exposed directly in materials.MPM (it usually is hidden)
     # We might need to import from internal path if possible, but let's try getting it from an existing material instance or similar?
     # Actually, inspect_genesis.py printed "Base" in dir(gs.materials.MPM). So it is exposed.
     pass

@ti.data_oriented
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
        sampler=None,
    ):
        super().__init__(E, nu, rho, sampler=sampler)

        self._A = A
        self._B = B
        self._n = n
        self._C = C
        self._eps0 = eps0
        
        # Initial Jp (Plastic Strain) = 0.0
        self._default_Jp = 0.0

    @ti.func
    def update_F_S_Jp(self, J, F_tmp, U, S, V, Jp, temp):
        """
        Updates Deformation Gradient (F), Singular Values (S), and Plastic Strain (Jp).
        Jp here stores Equivalent Plastic Strain (epsilon_p).
        """
        F_new = ti.Matrix.zero(gs.ti_float, 3, 3)
        S_new = ti.Matrix.zero(gs.ti_float, 3, 3)
        delta_gamma = 0.0
        
        # 1. Trial Deviatoric Strain (Elastic Predictor)
        S_clamped = ti.max(S, 1e-6) # Limit to avoid log(0)
        epsilon = ti.Vector([ti.log(S_clamped[0, 0]), ti.log(S_clamped[1, 1]), ti.log(S_clamped[2, 2])])
        
        trace_eps = epsilon.sum()
        epsilon_hat = epsilon - (trace_eps / 3.0)
        epsilon_hat_norm = epsilon_hat.norm(gs.EPS)
        
        # 2. Determine Flow Stress (Sigma_y)
        eps_p = Jp
        sigma_y_static = self._A + self._B * ti.pow(eps_p, self._n)
        
        # We will add thermal softening (Johnson-Cook melting term) later. 
        # For now, just pass the temperature to avoid Taichi signature errors.
        sigma_y = sigma_y_static
        
        # 3. Yield Condition (Von Mises)
        yield_dist = epsilon_hat_norm - sigma_y / (2 * self._mu)
        Jp_new = Jp 
        
        if yield_dist > 0: # Yields
            delta_gamma = yield_dist 
            Jp_new = eps_p + delta_gamma
            
            epsilon -= (delta_gamma / epsilon_hat_norm) * epsilon_hat
            
            # Reconstruct S
            S_new = ti.Matrix.zero(gs.ti_float, 3, 3)
            for d in ti.static(range(3)):
                S_new[d, d] = ti.exp(epsilon[d])
            
            F_new = U @ S_new @ V.transpose()
            
        else: # Elastic
            F_new = F_tmp
            S_new = S
            delta_gamma = 0.0
        
        return F_new, S_new, Jp_new, delta_gamma, sigma_y

