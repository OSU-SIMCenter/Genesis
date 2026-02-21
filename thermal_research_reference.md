# Thermal Physics: Theoretical Foundations & Technical Reference

This document provides the theoretical background, governing equations, material models, and Taichi/Genesis architectural context needed to implement and extend thermal physics in the Genesis MPM solver. It is intended as a durable reference; consult the implementation plan for sequencing.

---

## 1. Governing Equations

A thermomechanical simulation solves three coupled conservation laws simultaneously.

### 1.1 Conservation of Mass
$$\frac{D\rho}{Dt} + \rho \nabla \cdot \mathbf{v} = 0$$

In MPM, mass is carried by particles and automatically conserved as long as no particles are created or destroyed.

### 1.2 Conservation of Momentum
$$\rho \mathbf{a} = \nabla \cdot \boldsymbol{\sigma} + \mathbf{b}$$

where $\boldsymbol{\sigma}$ is the Cauchy stress tensor and $\mathbf{b}$ represents body forces (gravity). This is what the existing MPM solver already handles (P2G momentum transfer → grid velocity update → G2P advection).

### 1.3 Conservation of Energy (The Heat Equation)
$$\rho C_p \frac{DT}{Dt} = \nabla \cdot (k \nabla T) + \dot{Q}_{plastic} + \dot{Q}_{external}$$

where:
- $T$ is the temperature field
- $C_p$ is specific heat capacity at constant pressure [J/(kg·K)]
- $k$ is thermal conductivity [W/(m·K)]
- $\alpha = k / (\rho C_p)$ is thermal diffusivity [m²/s]
- $\dot{Q}_{plastic}$ is the internal heat generation rate from plastic work
- $\dot{Q}_{external}$ represents external sources (boundary cooling, radiation, induction, laser)

### 1.4 Coupling Mechanisms

The three conservation laws couple in two directions:

**Mechanical → Thermal**: Plastic deformation generates heat ($\dot{Q}_{plastic}$). This is the Taylor-Quinney effect: a fraction $\chi$ (typically 0.9 for metals) of the plastic work rate converts to heat.

**Thermal → Mechanical**: Temperature changes affect mechanical behavior through:
1. **Thermal softening**: Yield stress decreases with temperature (J-C model).
2. **Thermal expansion**: Volume changes with temperature ($\mathbf{F}^\theta$).
3. **Elastic moduli reduction**: Young's modulus decreases with temperature (secondary effect).

---

## 2. The Johnson-Cook Constitutive Model

The Johnson-Cook model is the standard for metals under high strain rate and temperature. The dynamic yield stress is:

$$\sigma_y = \underbrace{(A + B \varepsilon_p^n)}_{\text{strain hardening}} \cdot \underbrace{(1 + C \ln(\dot{\varepsilon}/\dot{\varepsilon}_0))}_{\text{strain rate}} \cdot \underbrace{(1 - T^{*m})}_{\text{thermal softening}}$$

where the homologous temperature is:
$$T^* = \frac{T - T_{ref}}{T_{melt} - T_{ref}}, \quad T^* \in [0, 1]$$

### 2.1 Parameter Table: AISI 4340 Steel

| Parameter | Symbol | Value | Unit |
|-----------|--------|-------|------|
| Reference yield stress | $A$ | 792 | MPa |
| Hardening modulus | $B$ | 510 | MPa |
| Hardening exponent | $n$ | 0.26 | — |
| Strain rate constant | $C$ | 0.014 | — |
| Thermal softening exponent | $m$ | 1.03 | — |
| Melting temperature | $T_{melt}$ | 1793 | K |
| Reference temperature | $T_{ref}$ | 293 | K |
| Reference strain rate | $\dot{\varepsilon}_0$ | 1.0 | s⁻¹ |
| Density | $\rho$ | 7800 | kg/m³ |
| Specific heat capacity | $C_p$ | 450 | J/(kg·K) |
| Thermal conductivity | $k$ | 44.5 | W/(m·K) |
| Thermal diffusivity | $\alpha$ | 1.27×10⁻⁵ | m²/s |
| CTE (linear) | $\alpha_{th}$ | 1.23×10⁻⁵ | K⁻¹ |
| Young's modulus | $E$ | 205 | GPa |
| Poisson's ratio | $\nu$ | 0.29 | — |

### 2.2 Return Mapping with Johnson-Cook

The von Mises return mapping in `ElastoPlastic.update_F_S_Jp` currently works in log-strain space:

```python
epsilon = [log(S[0,0]), log(S[1,1]), log(S[2,2])]
epsilon_hat = epsilon - mean(epsilon)          # deviatoric log-strain
delta_gamma = norm(epsilon_hat) - sigma_y / (2 * mu)

if delta_gamma > 0:  # yielding
    epsilon -= (delta_gamma / norm(epsilon_hat)) * epsilon_hat
    S_new = diag(exp(epsilon))
```

To integrate Johnson-Cook, the yield stress `sigma_y` (currently `self._von_mises_yield_stress`, a constant) must become temperature-dependent. The key modifications are:

1. **Pass temperature into the material model**: `update_F_S_Jp` needs access to the particle's current `temp`.
2. **Compute dynamic yield stress**: Replace the static yield with the full J-C expression (or initially just the thermal term if strain rate data isn't readily available).
3. **Capture the plastic strain increment**: `delta_gamma` (when > 0) is the plastic multiplier. This value must be written to the particle's `plastic_strain` field and used to compute adiabatic heating.

### 2.3 Adiabatic Heating (Taylor-Quinney Effect)

When plastic flow occurs, the heat generated per unit volume per step is:

$$\dot{Q}_{plastic} = \chi \cdot \sigma_y \cdot \dot{\varepsilon}_p$$

For a discrete timestep with plastic strain increment $\Delta\gamma$:

$$\Delta T = \frac{\chi \cdot \sigma_y \cdot \Delta\gamma}{\rho \cdot C_p}$$

where $\chi = 0.9$ is the Taylor-Quinney coefficient.

**Important**: This heating must happen at the *particle level* during the constitutive update (inside `update_F_S_Jp` or immediately after), NOT during the P2G transfer. The temperature change from plastic work is a local particle event, not a grid operation.

### 2.4 Strain Rate Calculation

The effective strain rate can be approximated from the velocity gradient (available from the APIC affine matrix $\mathbf{C}$):

$$\dot{\varepsilon} = \sqrt{\frac{2}{3} \mathbf{D}' : \mathbf{D}'}$$

where $\mathbf{D}' = \frac{1}{2}(\mathbf{C} + \mathbf{C}^T) - \frac{1}{3}\text{tr}(\mathbf{C})\mathbf{I}$ is the deviatoric strain rate tensor.

For the initial implementation, setting $\dot{\varepsilon} = \dot{\varepsilon}_0$ (making the strain rate term = 1.0) is a valid simplification.

---

## 3. Thermal Expansion

Temperature changes induce volumetric deformation. The deformation gradient decomposes multiplicatively:

$$\mathbf{F} = \mathbf{F}^e \cdot \mathbf{F}^p \cdot \mathbf{F}^\theta$$

where:
- $\mathbf{F}^e$: elastic (generates stress)
- $\mathbf{F}^p$: plastic (permanent deformation, captured by return mapping)
- $\mathbf{F}^\theta$: thermal (volume change due to temperature)

For isotropic materials:
$$\mathbf{F}^\theta = (1 + \alpha_{th} (T - T_{ref})) \mathbf{I}$$

**Implementation**: Before computing the trial elastic deformation gradient, remove the thermal part:
$$\mathbf{F}^{trial}_e = \mathbf{F} \cdot (\mathbf{F}^\theta)^{-1} \cdot (\mathbf{F}^p)^{-1}$$

For the initial phases, thermal expansion can be deferred (it's a secondary effect compared to softening and heating). But for dimensionally accurate forging results, it must eventually be included.

---

## 4. Heat Transfer Mechanisms

### 4.1 Conduction (Internal Diffusion)

Fourier's law on the MPM grid (after P2G normalization):

$$T_{i,j,k}^{new} = T_{i,j,k} + \alpha \cdot \Delta t \cdot \frac{T_{i+1} + T_{i-1} + T_{j+1} + T_{j-1} + T_{k+1} + T_{k-1} - 6T_{i,j,k}}{\Delta x^2}$$

**CFL stability condition** (explicit scheme):
$$\Delta t < \frac{\Delta x^2}{2 \cdot d \cdot \alpha}$$

where $d = 3$ (spatial dimensions). For steel at $\Delta x = 1/64$: $\Delta t < \frac{(1/64)^2}{6 \times 1.27 \times 10^{-5}} \approx 3.2$ s. Well above typical MPM substep sizes ($10^{-4}$ to $10^{-3}$ s), so explicit diffusion is stable.

**Double-buffering requirement**: Reading and writing the same grid field in a parallel Taichi loop is a race condition. Two approaches:
1. **Extra buffer field**: Add `grid.temp_new`, compute Laplacian into it, then swap.
2. **Two-pass kernel**: First pass computes Laplacian into a temporary, second pass applies it.

Option 1 (extra field) is simpler and recommended.

### 4.2 Convective Cooling (Air)

Newton's law of cooling for a surface cell:

$$\frac{dT}{dt} = -\frac{h_{conv} \cdot (A/V)}{\rho \cdot C_p} \cdot (T - T_{air})$$

The rate constant $k_{air} = h_{conv} \cdot (A/V) / (\rho \cdot C_p)$ has units [1/s]. For a grid cell at the surface, $A/V \approx 1/\Delta x$.

With exponential decay for stability:
$$T_{new} = T_{air} + (T_{curr} - T_{air}) \cdot \exp(-k_{air} \cdot \Delta t)$$

**Typical values** for natural convection on steel: $h_{conv} \approx 10$–$50$ W/(m²·K).

**Surface detection**: The density-threshold heuristic (`rho_cell < 7000`) should be replaced. Better approaches:
- **Empty neighbor check**: A cell is "surface" if at least one of its 6 face-neighbors has zero thermal mass.
- **Relative threshold**: `rho_cell < 0.8 * rho_material` (material-independent).

### 4.3 Contact Heat Transfer (Rigid Bodies)

When the MPM material contacts a rigid tool/die:

$$\frac{dT}{dt} = -\frac{h_{contact} \cdot (A_{contact}/V)}{\rho \cdot C_p} \cdot (T - T_{rigid})$$

where $h_{contact}$ is the interfacial heat transfer coefficient [W/(m²·K)]. For metal-on-metal forging contact: $h_{contact} \approx 1000$–$10000$ W/(m²·K).

Contact is currently detected via the SDF: if a grid node is within $1.5 \Delta x$ of a rigid surface. For the contact area ratio, $A_{contact}/V$ can be approximated as $1/\Delta x$ for cells fully in contact.

**One-way vs. two-way**: Currently the rigid body is an infinite heatsink at 293.15K. For two-way coupling, the rigid solver would need its own thermal state, and heat exchange would be conservative: $Q_{mpm \to rigid} = -Q_{rigid \to mpm}$.

---

## 5. Taichi & Genesis Architectural Constraints

### 5.1 Why Template Method Hooks (Not Separate Kernels)

The MPM cycle (P2G → Grid Op → G2P) loads particle data from GPU VRAM. Each kernel launch requires a full memory traversal. Separate `p2g_thermal` and `p2g_mechanical` kernels would traverse particles twice, wasting bandwidth.

By using `@ti.func` hooks *inside* the existing `p2g_helper`, the thermal scatter happens in the same memory pass as the mechanical scatter. Taichi inlines `@ti.func` calls, so there is zero overhead. This is why the hook approach is superior to separate thermal kernels.

### 5.2 Static Memory Layout

Taichi fields must be declared before any kernel compilation. The template method pattern (`_make_particle_state_template` returning a dict) allows the derived class to extend the struct before `ti.types.struct(...)` is called. This keeps `temp` adjacent to `pos` and `vel` in the SOA layout, preserving cache locality.

### 5.3 Solver Registration in Genesis

Genesis's `Simulator` class imports and instantiates solvers based on options. It does not support dependency injection of custom solver classes. The current approach (naming the file `mpm_solver.py` and using `__new__` to conditionally return the base or derived class) works because Genesis imports `MPMSolver` by name. When porting to a newer branch, this file must be placed at the correct import path.

### 5.4 Differentiability

Genesis supports automatic differentiation for gradient-based optimization (e.g., optimizing robot trajectories). All new kernels should be AutoDiff compatible:

- Avoid hard `if/else` conditionals on continuous variables where possible. Use `ti.max`, `ti.min` for soft clamping.
- The J-C yield condition (`if delta_gamma > 0`) introduces a gradient discontinuity at the yield point. Taichi's AD handles this, but gradient quality near yield may be poor. Consider a smooth regularization if AD is needed.
- The `ti.exp()` decay in cooling is smooth and AD-friendly.

---

## 6. MPM Thermal Integration Loop

The complete thermal-mechanical substep sequence:

```
1. reset_grid_and_grad(f)        -- zero grid fields (incl. temp, mass_thermal)
2. compute_F_tmp(f)              -- trial deformation gradient
3. svd(f)                        -- SVD of F_tmp
4. p2g(f)                        -- per-particle:
   a. update_F_S_Jp(...)         --   constitutive update (return mapping)
                                 --   ** compute delta_gamma, update plastic_strain **
                                 --   ** adiabatic heating: temp += chi*sigma_y*dg/(rho*Cp) **
   b. update_stress(...)         --   compute Cauchy stress from updated F
   c. p2g_modify_stress(...)     --   [FUTURE: remove once softening is in return mapping]
   d. scatter momentum + mass    --   standard MPM
   e. p2g_transfer_extra_fields  --   scatter mass*temp to grid
5. grid_op_thermal(f)            -- normalize temp, diffusion, boundary cooling
6. [coupling step]               -- rigid-MPM velocity coupling
7. g2p(f)                        -- per-particle:
   a. g2p_prologue(...)          --   reset temp=0, copy plastic fields forward
   b. gather velocity, C         --   standard MPM
   c. g2p_transfer_extra_fields  --   gather temp from grid
   d. advect position            --   standard MPM
```

Steps marked with `**` are not yet implemented (Phase 2).

---

## 7. Validation Benchmarks

### 7.1 Hot Bar Test (Intermediate - Phase 1 Validation)

Validates that thermal-mechanical coupling (softening) works:
1. Create two identical bars, one at T=1000K, one at T=300K.
2. Drop an identical weight on each from the same height.
3. **Expected**: The hot bar deforms significantly more because its yield stress is lower.
4. **Metric**: Compare maximum displacement or final height. The ratio should roughly match the J-C softening ratio.

### 7.2 Taylor Impact Test (Full Validation)

The standard thermomechanical benchmark:
- **Setup**: Cylindrical bar of 4340 Steel (L=32.4mm, D=6.4mm), initial velocity 200 m/s, impacts rigid wall.
- **Expected (mechanical)**: Mushrooming at impact end, final length ~21mm.
- **Expected (thermal)**: Impact end heats significantly (~400-800K rise) from plastic work. Deformation exceeds isothermal prediction due to thermal softening feedback.
- **Requires**: Phases 1 + 2 (plastic work heating + correct thermal softening).

### 7.3 Contact Cooling Test (Already Exists)

The existing `test_contact_cooling.py` verifies:
- Hot cube on cold rigid floor cools over time.
- Temperature drop is significantly greater than air cooling alone.
