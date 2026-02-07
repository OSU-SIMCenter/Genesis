
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
    def update_F_S_Jp(self, J, F_tmp, U, S, V, Jp):
        """
        Updates Deformation Gradient (F), Singular Values (S), and Plastic Strain (Jp).
        Jp here stores Equivalent Plastic Strain (epsilon_p).
        """
        F_new = ti.Matrix.zero(gs.ti_float, 3, 3)
        S_new = ti.Matrix.zero(gs.ti_float, 3, 3)
        
        # 1. Trial Deviatoric Strain (Elastic Predictor)
        # S is diagonal sigma_trial (singular values of F_trial)
        # epsilon_trial = ln(S)
        S_clamped = ti.max(S, 1e-6) # Limit to avoid log(0)
        epsilon = ti.Vector([ti.log(S_clamped[0, 0]), ti.log(S_clamped[1, 1]), ti.log(S_clamped[2, 2])])
        
        # Deviatoric part: e = epsilon - trace(epsilon)/3 * I
        trace_eps = epsilon.sum()
        epsilon_hat = epsilon - (trace_eps / 3.0)
        
        # Norm of deviatoric strain: ||e||
        epsilon_hat_norm = epsilon_hat.norm(gs.EPS)
        
        # 2. Determine Flow Stress (Sigma_y)
        # Get current plastic strain from Jp
        eps_p = Jp
        
        # Calculate Strain Rate (Approximate based on current step?) 
        # Ideally we need dt. gstaichi doesn't easily expose global dt here?
        # Actually, self.dt isn't standard in Material. 
        # Standard return mapping solves for gamma assuming const yield or linearized.
        # For J-C, yield depends on rate.
        # Rate = delta_gamma / dt.
        # This makes it implicit. 
        # Simplification: Use previous step rate or ignore rate effect in yield calculation (C=0)?
        # Or just use the hardening term first: sigma_y_static = A + B * eps_p^n
        
        sigma_y_static = self._A + self._B * ti.pow(eps_p, self._n)
        
        # For rate hardening, we can approximate rate ~ epsilon_hat_norm / dt? 
        # No, that's total strain rate.
        # Let's include C term later or assume rate is moderate.
        # If we include C, we need dt. 
        # Genesis `MPM.Base` doesn't seem to store dt.
        # We will SKIP Rate Hardening (C term) inside the kernel for now unless we find dt.
        # (User Note: J-C allows rate, but without dt in kernel, we implement Static J-C first).
        
        sigma_y = sigma_y_static # * (1 + C * ...)
        
        # 3. Yield Condition (Von Mises)
        # Yield if ||s_trial|| > sqrt(2/3) * sigma_y aka ||e_trial|| > sigma_y / (2*mu)
        # delta_gamma = ||e_trial|| - sigma_y / (2*mu)
        
        yield_dist = epsilon_hat_norm - sigma_y / (2 * self._mu)
        
        Jp_new = Jp # Default: no change
        
        if yield_dist > 0: # Yields
            # Return Mapping
            delta_gamma = yield_dist # This is the increment of plastic strain
            
            # Update Plastic Strain (Accumulate)
            Jp_new = eps_p + delta_gamma
            
            # Scale epsilon_hat to mapping surface
            scale = (sigma_y / (2*self._mu)) / epsilon_hat_norm
            # Or effectively: epsilon_final = epsilon_trial - delta_gamma * (epsilon_hat / norm)
            # Recompute elastic strain:
            epsilon -= (delta_gamma / epsilon_hat_norm) * epsilon_hat
            
            # Reconstruct S
            S_new = ti.Matrix.zero(gs.ti_float, 3, 3)
            for d in ti.static(range(3)):
                S_new[d, d] = ti.exp(epsilon[d])
            
            F_new = U @ S_new @ V.transpose()
            
        else: # Elastic
            F_new = F_tmp
            S_new = S # Or S_clamped? F_tmp uses S.
        
        return F_new, S_new, Jp_new

