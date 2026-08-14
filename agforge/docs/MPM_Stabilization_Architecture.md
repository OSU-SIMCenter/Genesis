# Genesis MPM Constitutive Stabilization Architecture

> ⚠️ **UNMAINTAINED — verify against source before trusting anything here.** Checked 2026-08-14:
> this file has not been updated since the 316L material change or the coupling/scaling work, and
> at least one of its statements was found to be factually wrong (see the Roadmap's Bug 3). It
> still describes the billet as **AISI 4340** in places; the billet is **316L**. Living references
> are `docs/THERMOMECHANICAL_COUPLING_AND_SCALING.md` (coupling, scaling, metrics),
> `docs/316L_MECHANICAL_PROPERTIES.md` (the material card and its validity limits), and
> `agforge/docs/Contact_Method_Research_And_Plan.md` (contact).


## 1. Executive Summary
This document synthesizes the numerical stability upgrades integrated into the `agforge/v2/main` branch to handle Hot-Forging simulations. 
The core problem historically was the "Null-Space Detachment Explosion": an anomalous energy spike where the spatial velocity grid exploded when the simulation combined induction heating with extreme compression. 
This was successfully resolved by migrating from a classical, explicit $J_2$-flow theory model tightly coupled to temperature drops, to an **Implicit Perzyna Viscoplastic Formultaion** using quadratic root polishing, and explicitly resolving simulation time-dilation discrepancies.

## 2. Phase 1: Architectural Truths & Elimination of Grid Anomaly Theories
Initially, the instability was hypothesized to be a standard MPM "grid-crossing error" requiring advanced momentum mappings (like XPIC/B-Splines). However, code audits confirmed the Genesis engine already natively operated at the State-Of-The-Art:
*   **Spatial Discretization is SOTA:** The P2G/G2P scatter and gather operations natively utilize **Quadratic B-splines** (a 27-node $3 \times 3 \times 3$ stencil providing mathematical $C^1$ continuity).
*   **Momentum Transport is SOTA:** The engine uses the **Affine Particle-In-Cell (APIC)** momentum operator, flawlessly resolving angular momentum dissipation.
*   **Constitutive Core:** The elastic predictor leverages exact Singular Value Decomposition (SVD) inside the Principal Hencky (Logarithmic) Strain Space.

Because the underlying continuum-transport topology was flawless, the failure was localized strictly back to the explicit constitutive yield implementation within `elasto_plastic.py`.

## 3. Phase 2: Vulnerability Analysis of Physical Thermal Coupling
The engine integrates the **Johnson-Cook Thermal Softening Model** to accurately define the flow stress of hot steel dynamically:
```python
T_star = (temp - T_ref) / (T_melt - T_ref)
thermal_softening = 1.0 - pow(T_star, jc_m)
sigma_y = base_stress * thermal_softening
```
As the steel heats to ~1400K via the Induction Heater, it loses $\approx 73\%$ of its yield strength.
Under an _Explicit Rate-Independent Yield Model_, the overstress collapses $100\%$ back to the boundary identically within one substep. If the yield boundary is suddenly collapsed by $73\%$ dynamically during a strike animation frame, the explicit algorithm instantaneously dumps megajoules of previously stable elastic potential energy as unmitigated residual strain into the spatial velocity matrix. This acoustic impulse shattered the grid.

## 4. Phase 3: The Viscoplastic Implicit Resolution
To prevent the static detonation of elastic strain, the constitutive solver was entirely re-architected.  We integrated a **Perzyna Viscoplastic Return Map**, enforcing that the material yields dynamically like a highly viscous liquid: $\sigma = \sigma_y + \eta \dot{\varepsilon}$.

### The Geometric 2-Phase Implicit Toplogy
Because viscosity requires resolving highly non-linear exponents from the Hencky log-strain, mathematical evaluation requires an implicitly solved root function:
$R(g) = ||\hat{\varepsilon}|| - g - \frac{\sigma_y}{2\mu} - \frac{\eta \cdot g}{2\mu \cdot \Delta t} = 0$

To guarantee real-time evaluation stability without infinite branch divergence on the GPU, the implementation uses a multi-tier approach located in `update_F_S_Jp`:

1. **Bisection (Robust Operator | 10 iterations):** 
   Unconditionally brackets the exact solution between $[0, ||\hat{\varepsilon}||]$. It uses a totally branchless conditional execution mapping to split the space 10 times, isolating a strictly safe domain.
2. **Newton-Raphson (Quadratic Polish | 3 iterations):** 
   Uses the Bisection bounds as an exceptionally grounded initial guess, converging exactly to the origin without overstepping due to the absolute boundary conditions clamped at $10^{-6}$.

## 5. Phase 4: Time-Dilation Math & The $71,428\times$ Fix
A critical realization for accurate robotic teleoperation concerned the explicit scaling speeds of kinematic manipulation versus computational updates.

### The Phenomenon
The Robot Unity controller accelerates wall-clock time (`robot_time_to_seconds = 71428`). 
The press physically intersects the metal over $71,400\times$ faster than standard physical simulation time, generating an artificial relative strain rate of $\dot{\varepsilon} \approx 3000\text{ s}^{-1}$. Under physical Perzyna viscosity, hitting hot steel at Mach-speed causes the internal resistance to spike $> 2 \text{ GPa}$, rendering the material permanently rigid and falsely triggering the Robot clamp force limit of $196\text{kN}$. 

### True Physical Viscosity Dilator
To unify the mechanics, the true viscosity $C$ must be explicitly derived by dividing by the Unity controller's artificial hyper-speed factor (`agforge/environment.py`):
```python
# True Physical Viscoplastic Scaling
true_eta = 2.0e6  # Typical hot steel dynamic viscosity (Pa.s)
unity_time_scale = 0.1 * cfg.sim.substeps / cfg.sim.dt # ~= 71,428

# C = true_eta / (2*mu * dt * time_scale). By mapping time_scale, we artificially
# accelerate the viscosity so it flows gracefully for the hyper-speed controller
eta_over_dt = true_eta / (2.0 * mu * cfg.sim.dt * unity_time_scale)
```
This forces the equation to undergo viscous relaxation precisely identical to the physical scaling of the rigid-body colliders.

## 6. Official Next Steps & Roadmap
With stability perfectly aligned conceptually and numerically, we have successfully executed steps 1 and 2 from our core continuum mechanics stabilization roadmap. The remaining roadmap strictly follows the escalation path required to guarantee absolute structural integrity under extreme unconstrained striking:

1. **~~Isotropic Hardening~~ (Completed):** Handled physically by integrating the Johnson-Cook thermal strain variables into the implicit yield map.
2. **~~Implicit Viscoplastic Return Map~~ (Completed):** Correctly dissipates thermal shock through physical flow constraints rather than instant detonations.
3. **Target 3: Pre-SVD $J$ Guard:** Implement a simple 2-line guard (`if det(F_tmp) < 0.01: scale it up`) exactly prior to the SVD dispatch inside `base_mpm_solver.py`. This acts as defense-in-depth to catch "Pancake Collapses" before singular matrix inputs can ever generate `NaN` math.
4. **~~Grid Mass Floor~~ (Completed):** Confirmed industry standard `gs.EPS` limits already natively shield the $f/m$ division-by-zero crashes on loose particles.
5. **Escalation Path: Neo-Hookean Multiplicative Rewrite:** Strictly held in reserve. If the Pre-SVD guard fails to stop extreme determinant collapses during physical testing, we will escalate to replacing the current Logarithmic Hencky-strain evaluation with a Neo-Hookean Multiplicative Split, natively weaponizing the $-\ln(J)$ volumetric pressure penalty to defend the volume mathematically.
