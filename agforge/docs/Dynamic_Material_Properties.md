# Dynamic Viscosity Architecture

## 1. Overview
To maximize physical accuracy in the hot forging simulation without sacrificing GPU performance or explicit timestep stability, we will isolate and make **Viscosity ($\eta$)** dynamic and temperature-dependent. 

While elastic properties ($E$, $\nu$) are kept artificially softened to maintain low wave speeds (stiffness scaling), making viscosity dynamic is critical. It ensures that cold regions of the billet act like rigid metal, while hot regions flow naturally under the forging press.

---

## 2. Dynamic Viscosity ($\eta$) Formula

*   **Source:** Thermo-Viscoplastic Return Mapping Literature / Abaqus VUMAT standards for Perzyna overstress models.
*   **Physical Behavior:** Cold steel fractures or yields rigidly (infinite viscosity). Hot steel flows like a highly viscous fluid.
*   **The Perzyna Linkage:** In standard Johnson-Cook models, yield stress softens with temperature, but this does not dictate the *speed* of flow. By integrating a temperature-dependent Perzyna viscosity ($\eta = 1/\gamma$), we ensure the flow rate degrades physically as the metal cools.
*   **Formula (Arrhenius-like Softening):** 
    Instead of calculating a true, computationally expensive Arrhenius exponential ($\eta = A \exp(Q/RT)$), we use the normalized homologous temperature ($T_{star}$) to create an efficient, numerically stable exponential scaling factor.
    
    $\eta(T) = \eta_{hot\_base} \times \exp(K \times (1.0 - T_{star}))$
    
    *   $K$ controls the exponential rigidity of the cold steel.
    *   If $T_{star} = 1.0$ (Hot), $\eta = \eta_{hot\_base}$. 
    *   If $T_{star} = 0.0$ (Cold), $\eta$ grows exponentially by a factor of $e^K$. This naturally suppresses the viscoplastic strain increment ($\Delta \gamma$) in the return mapping algorithm, making cold steel mathematically act as a rigid solid.

---

## 3. Implementation Strategy

To implement this efficiently in the Taichi/Genesis architecture, we will update the constitutive viscoplastic kernel directly. This avoids any need to modify the core Genesis engine's `update_stress` signature.

### Step 1: Update the Constitutive Kernel
Inside `agforge/materials.py` (`JohnsonCookPlasticity`), locate the `update_F_S_Jp` kernel. Replace the static constant `eta_dt = self._eta_over_dt` with the dynamic scaled Arrhenius equation.

```python
# K controls the exponential growth of viscosity as the metal cools.
# A value of K=10.0 means cold steel is ~22,000x more viscous than hot steel.
K = gs.qd_float(10.0) 

# eta_dt grows exponentially as T_star approaches 0 (cold)
eta_dt = self._eta_over_dt * qd.math.exp(K * (gs.qd_float(1.0) - T_star))
```

### Why this is the optimal approach:
1.  **Accuracy:** It perfectly bridges the Johnson-Cook yield envelope with the Perzyna rate-dependent flow rule, matching standard FEA literature for hot forging.
2.  **Performance:** Evaluates a single `exp()` per particle without modifying the complex implicit Jacobian.
3.  **Numerical Stability:** Because the implicit backward-Euler Return Map places `eta_dt` in the denominator of the Newton-Raphson step (`R_prime`), a massive viscosity (cold steel) naturally pushes the step size (`delta_gamma`) to zero. The solver remains perfectly stable, it simply refuses to flow.
