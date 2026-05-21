from typing import Any
import quadrants as qd
import genesis as gs
from genesis.typing import ValidFloat

@qd.data_oriented
class JohnsonCookPlasticity(gs.materials.MPM.Base):
    """
    Johnson-Cook elasto-plastic material for MPM.
    
    Flow Stress: sigma_y = (A + B * eps_p^n) * (1 + C * ln(eps_dot_star))
    
    Utilizes 'Jp' particle field to store accumulated equivalent plastic strain (epsilon_p).
    """

    A: ValidFloat = 792e6
    B: ValidFloat = 510e6
    n: ValidFloat = 0.26
    C: ValidFloat = 0.014
    eps0: ValidFloat = 1.0
    T_ref: ValidFloat = 293.15
    T_melt: ValidFloat = 1793.0
    jc_m: ValidFloat = 1.03

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
