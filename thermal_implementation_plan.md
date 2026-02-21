# Thermal Physics Implementation Plan

This document defines the sequenced plan for fixing, porting, and extending the thermal physics implementation. Each phase lists concrete steps, files to modify, and validation criteria.

**Context**: This branch is outdated relative to upstream Genesis. The first objective is always to port changes to the current/updated branch. See `thermal_implementation_status.md` for what exists and what's broken. See `thermal_research_reference.md` for equations and theory.

---

## Phase 0: Port to Updated Branch

**Priority**: PREREQUISITE — do this before any feature work.

Because this reference branch is outdated, the first task is to reconstruct the thermal changes on the current upstream branch.

### Steps

1. **Analyze the updated `base_mpm_solver.py`**: On the target branch, read the current `p2g_helper` and `g2p_helper`. Identify the exact lines where hooks should be inserted. Do NOT blindly overwrite the file.

2. **Graft the hooks into the updated base solver**:
   - Add the five `@ti.func` hook methods (see `thermal_implementation_status.md` §2 for exact signatures).
   - Insert `self.p2g_modify_stress(f, i_p, i_b, stress)` in `p2g_helper` after the `update_stress` call.
   - Insert `self.p2g_transfer_extra_fields(f, i_p, idx, i_b, weight)` in `p2g_helper` inside the scatter loop (after mass/momentum scatter, before the free-particle check).
   - Insert `self.g2p_prologue(f, i_p, i_b)` at the top of `g2p_helper`.
   - Insert `self.g2p_transfer_extra_fields(f, i_p, i_b, weight, grid_index)` inside the gather loop (after velocity/C updates).
   - Make `copy_frame_helper` and `reset_grid_helper` callable by the derived class (they already use `self.`, so this should work).
   - Add `_make_particle_state_template` / `_make_grid_cell_state_template` methods if the updated branch doesn't already use a template pattern for struct construction. Wire `init_particle_fields` and `init_grid_fields` to use these templates.

3. **Copy `mpm_solver.py` to the updated branch**: Verify all override methods still match the base class signatures. Check for API changes (e.g., renamed fields, changed arguments to `substep_pre_coupling`).

4. **Update `MPMOptions`**: Add the thermal parameters to the updated branch's `genesis/options/solvers.py`.

5. **Validate**: Run `tests/test_contact_cooling.py`. Verify the simulation runs without crashing and temperature decreases from 1000K.

---

## Phase 1: Fix Existing Implementation Bugs

**Priority**: HIGH — these are physics errors that must be fixed before adding new features.

### Phase 1A: Fix Cooling Rate Constants

**Problem**: `h_conv` and `h_contact` are used as raw rate constants instead of proper $k_{rate} = h \cdot (A/V) / (\rho \cdot C_p)$.

**Files**: `genesis/engine/solvers/mpm_solver.py` (`grid_op_thermal`)

**Steps**:

1. Compute the proper rate constant inside the kernel:
   ```python
   # Air cooling
   inv_dx = self._inv_dx
   rho_material = m / (self._dx ** 3)  # already computed as rho_cell
   Cp = self._options.default_heat_capacity
   surface_to_vol = inv_dx  # A/V ≈ 1/dx for a surface cell
   k_air = h_conv * surface_to_vol / (rho_material * Cp)
   decay_air = 1.0 - ti.exp(-self.substep_dt * k_air)
   ```

2. Do the same for contact cooling:
   ```python
   k_contact = h_contact * inv_dx / (rho_material * Cp)
   decay_contact = 1.0 - ti.exp(-self.substep_dt * k_contact)
   ```

3. Use physically meaningful values for the heat transfer coefficients:
   - `h_conv`: 10–50 W/(m²·K) for natural convection (keep existing value)
   - `h_contact`: 1000–10000 W/(m²·K) for metal-on-metal contact (increase from current effective value)
   - Update defaults in `MPMOptions` if needed.

4. Replace the hardcoded `rho_cell < 7000.0` surface check with a relative threshold:
   ```python
   # Approximate full-cell density from material mass and cell volume
   rho_full = self.particles_info[0].mass / self._particle_volume  # reference density
   if rho_cell < 0.8 * rho_full:
       # This cell is partially filled → likely at a surface
   ```
   Or better: check if any of the 6 face-neighbors has zero `mass_thermal`.

**Validation**: After fixing, verify that:
- Cooling rate is independent of `grid_density` (run same test at density 32 vs 64, cooling rate should be similar).
- The time to cool from 1000K to ~650K (half the excess above 293K) matches the analytical prediction for a steel cube in air.

### Phase 1B: Move Thermal Softening Into the Constitutive Model

**Problem**: Thermal softening scales the entire stress tensor post-hoc instead of modifying the yield stress inside the return mapping.

**Files**:
- `genesis/engine/materials/MPM/elasto_plastic.py` (primary)
- `genesis/engine/materials/MPM/base.py` (interface change)
- `genesis/engine/solvers/base_mpm_solver.py` (`p2g_helper` — pass temp to material)
- `genesis/engine/solvers/mpm_solver.py` (remove stress scaling, add temp passing)

**Steps**:

1. **Extend the material interface**: `update_F_S_Jp` currently takes `(J, F_tmp, U, S, V, Jp)`. Add a `temp` parameter:
   ```python
   # In base.py
   @ti.func
   def update_F_S_Jp(self, J, F_tmp, U, S, V, Jp, temp):
       # Base implementation ignores temp
       raise NotImplementedError
   ```
   Update ALL MPM material classes to accept the new parameter (even if they ignore it).

2. **Pass temperature in `p2g_helper`**: In `BaseMPMSolver.p2g_helper`, read the particle temperature and pass it to `update_F_S_Jp`:
   ```python
   p_temp = self.particles[f, i_p, i_b].temp  # 0.0 for legacy solver (field doesn't exist → need guard)
   F_new, S_new, Jp_new = material.update_F_S_Jp(J, F_tmp, U, S, V, Jp, p_temp)
   ```
   **Note**: For the legacy solver (no `temp` field), the base material ignores `temp`, so passing 0.0 or `T_ref` is safe.

3. **Implement temperature-dependent yield in `ElastoPlastic`**:
   ```python
   @ti.func
   def update_F_S_Jp(self, J, F_tmp, U, S, V, Jp, temp):
       # ... existing SVD / log-strain setup ...
       
       # Temperature-dependent yield (J-C thermal term)
       T_star = (temp - self._T_ref) / (self._T_melt - self._T_ref)
       T_star = ti.max(0.0, ti.min(1.0, T_star))
       thermal_softening = 1.0 - ti.pow(T_star, self._jc_m)
       
       effective_yield = self._von_mises_yield_stress * thermal_softening
       delta_gamma = epsilon_hat_norm - effective_yield / (2 * self._mu)
       
       # ... rest of return mapping ...
   ```

4. **Remove the post-hoc stress scaling**: In `MPMSolver`, change `get_particle_stress_scale` to return `1.0` (or remove the override entirely and delete `p2g_modify_stress` from the base). The hook can be kept for future use but should be a no-op once softening is in the constitutive model.

5. **Add J-C thermal parameters to `ElastoPlastic`**: Add `T_ref`, `T_melt`, `jc_m` as constructor parameters with defaults that preserve backward compatibility:
   ```python
   def __init__(self, ..., T_ref=293.15, T_melt=1793.0, jc_m=1.03):
   ```

**Validation**: Hot Bar Test (see `thermal_research_reference.md` §7.1). Two bars at different temperatures should show different deformation under identical loading.

---

## Phase 2: Plastic Work Heating (Adiabatic Heating)

**Priority**: HIGH — this is the primary missing physics for forging simulation.

**Prerequisite**: Phase 1B (temperature is already being passed into the constitutive model).

### Strategy: Particle Field + Material Model Writes

This is the cleanest approach (Option A from the analysis). The material model computes `delta_gamma` during the return mapping and writes it to a particle field. The solver then converts it to heat.

**Files**:
- `genesis/engine/materials/MPM/elasto_plastic.py`
- `genesis/engine/solvers/mpm_solver.py`
- `genesis/engine/solvers/base_mpm_solver.py` (minor: pass writable particle reference or use a new hook)

### Steps

1. **Expose `delta_gamma` from the return mapping**: Modify `update_F_S_Jp` to also return the plastic strain increment:
   ```python
   @ti.func
   def update_F_S_Jp(self, J, F_tmp, U, S, V, Jp, temp):
       # ... existing logic ...
       delta_gamma_out = 0.0
       if delta_gamma > 0:
           delta_gamma_out = delta_gamma
           # ... existing return mapping ...
       return F_new, S_new, Jp_new, delta_gamma_out
   ```
   Update the base class and ALL material subclasses to return a 4-tuple `(F_new, S_new, Jp_new, delta_gamma)`. Non-plastic materials return `delta_gamma = 0.0`.

2. **Capture `delta_gamma` in `p2g_helper`**: After calling `update_F_S_Jp`:
   ```python
   F_new, S_new, Jp_new, delta_gamma = material.update_F_S_Jp(...)
   self.particles[f + 1, i_p, i_b].F = F_new
   self.particles[f + 1, i_p, i_b].Jp = Jp_new
   ```
   This requires a hook or direct code in `p2g_helper`. Two sub-options:

   **Option 2a (Hook)**: Add a new hook `p2g_post_constitutive(self, f, i_p, i_b, delta_gamma)` called after `update_F_S_Jp`. The base is a no-op. `MPMSolver` overrides it to update plastic strain and compute heating.

   **Option 2b (Direct in base)**: Store `delta_gamma` to the particle field directly in `p2g_helper` (requires the field to exist even in the legacy solver, or use a guard).

   **Recommendation**: Option 2a (hook) is cleaner and consistent with existing architecture.

3. **Implement adiabatic heating in the hook**:
   ```python
   # In MPMSolver
   @ti.func
   def p2g_post_constitutive(self, f, i_p, i_b, delta_gamma, effective_yield):
       if delta_gamma > 0.0:
           chi = 0.9  # Taylor-Quinney coefficient
           rho = self.particles_info[i_p].mass / self._particle_volume
           Cp = self._options.default_heat_capacity
           dT = chi * effective_yield * delta_gamma / (rho * Cp)
           self.particles[f, i_p, i_b].temp += dT
           
           # Update accumulators
           self.particles[f + 1, i_p, i_b].plastic_strain = (
               self.particles[f, i_p, i_b].plastic_strain + delta_gamma
           )
           self.particles[f + 1, i_p, i_b].plastic_work = (
               self.particles[f, i_p, i_b].plastic_work + effective_yield * delta_gamma
           )
   ```

   **Note on timing**: The temperature is updated on the *current* frame's particle (before P2G scatter), so the heat is immediately scattered to the grid in the same substep. This is physically correct — the deformation and heating happen simultaneously.

4. **Pass `effective_yield` out of the material model**: Modify the return to include the yield stress used:
   ```python
   return F_new, S_new, Jp_new, delta_gamma, effective_yield
   ```
   Or compute it in the hook from `delta_gamma` and the stress state. The former is cleaner.

**Validation**:
- Compress a hot steel cube. Verify temperature *increases* during deformation.
- Compare temperature rise against analytical estimate: for 50% strain at yield ~500MPa, expected $\Delta T \approx 0.9 \times 500 \times 10^6 \times 0.5 / (7800 \times 450) \approx 64$K.
- Run the Taylor Impact Test (see `thermal_research_reference.md` §7.2).

---

## Phase 3: Grid-Based Heat Diffusion

**Priority**: MEDIUM — improves physical accuracy of internal heat conduction.

Currently, temperature is only smoothed through the MPM grid interpolation (which provides some implicit diffusion). This phase adds explicit thermal diffusion via the grid Laplacian.

**Files**: `genesis/engine/solvers/mpm_solver.py` (grid_op_thermal, grid template)

### Steps

1. **Add a double-buffer field**: In `get_grid_cell_state_template`, add `temp_new`:
   ```python
   template.update({
       "temp": gs.ti_float,
       "temp_new": gs.ti_float,
       "mass_thermal": gs.ti_float,
   })
   ```

2. **Implement the diffusion kernel** (inside `grid_op_thermal`, after normalization, before cooling):
   ```python
   # First pass: compute Laplacian, write to temp_new
   for i, j, k, i_b in ti.ndrange(*self._grid_res, self._B):
       if self.grid[f, i, j, k, i_b].mass_thermal > 0:
           T_c = self.grid[f, i, j, k, i_b].temp
           laplacian = 0.0
           for d in ti.static(range(3)):
               ip = ti.Vector([i, j, k])
               im = ti.Vector([i, j, k])
               ip[d] += 1; im[d] -= 1
               # Boundary: clamp to grid bounds
               ip[d] = ti.min(ip[d], self._grid_res[d] - 1)
               im[d] = ti.max(im[d], 0)
               T_p = self.grid[f, ip[0], ip[1], ip[2], i_b].temp
               T_m = self.grid[f, im[0], im[1], im[2], i_b].temp
               laplacian += T_p + T_m - 2.0 * T_c
           laplacian *= self._inv_dx * self._inv_dx
           alpha = self._options.default_thermal_diffusivity
           self.grid[f, i, j, k, i_b].temp_new = T_c + self.substep_dt * alpha * laplacian
       else:
           self.grid[f, i, j, k, i_b].temp_new = 293.15

   # Second pass: copy back
   for i, j, k, i_b in ti.ndrange(*self._grid_res, self._B):
       self.grid[f, i, j, k, i_b].temp = self.grid[f, i, j, k, i_b].temp_new
   ```

   **Note**: These two passes must be separate `ti.kernel` calls or separated by `ti.sync()` to avoid race conditions. Alternatively, split `grid_op_thermal` into `grid_op_thermal_diffusion` and `grid_op_thermal_cooling` as separate kernels.

3. **Update `reset_grid_helper`** to also zero `temp_new`.

**Validation**: Initialize a bar with one hot end (1000K) and one cold end (293K). Run without mechanical deformation. Verify temperature profile evolves toward equilibrium following the analytical 1D diffusion solution.

---

## Phase 4: Thermal Expansion

**Priority**: MEDIUM — needed for dimensionally accurate forging.

### Steps

1. **Add CTE parameter** to material models and `MPMOptions`:
   ```python
   thermal_expansion_coefficient: float = 1.23e-5  # 1/K, for 4340 steel
   ```

2. **Modify `compute_F_tmp_helper`** to remove the thermal deformation before computing the trial elastic gradient:
   ```python
   @ti.func
   def compute_F_tmp_helper(self, f, i_p, i_b):
       F_mech = (ti.Matrix.identity(gs.ti_float, 3) + self.substep_dt * self.particles[f, i_p, i_b].C) @ self.particles[f, i_p, i_b].F
       # Remove thermal expansion
       T = self.particles[f, i_p, i_b].temp
       alpha_th = self._options.thermal_expansion_coefficient
       F_theta_inv = 1.0 / (1.0 + alpha_th * (T - self._options.T_ref))
       self.particles[f, i_p, i_b].F_tmp = F_mech * F_theta_inv
   ```

   This is a simplification (isotropic scalar scaling), which is appropriate for most metals.

3. **Add back thermal expansion after return mapping** to get the total deformation gradient:
   ```python
   F_total = F_new * (1.0 + alpha_th * (T - T_ref))
   ```

**Validation**: Heat a free-floating cube from 293K to 1000K. Measure volume change. Expected: $\Delta V / V = 3 \alpha_{th} \Delta T = 3 \times 1.23 \times 10^{-5} \times 707 \approx 2.6\%$.

---

## Phase 5: Explicit Particle Heating API

**Priority**: LOW — useful for general-purpose thermal simulation.

### Steps

1. **`set_temperature(mask, value)`**: Set temperature of specific particles. Useful for initial conditions (e.g., non-uniform temperature distribution).

2. **`apply_heat_flux(mask, flux, dt)`**: Apply a heat source to specific particles over a timestep. Useful for laser heating, induction heating, etc.

3. Both should be Python methods on `MPMSolver` that dispatch to Taichi kernels operating on the particle `temp` field.

---

## Phase 6: Two-Way Thermal Coupling (Rigid Bodies)

**Priority**: LOW — needed for long multi-strike forging simulations.

Currently, rigid bodies are infinite heatsinks at 293.15K. For simulations where dies heat up over repeated strikes, the rigid solver needs thermal state.

### Steps

1. **Add thermal state to rigid solver**: Temperature array per rigid body (or per node for FEM-based rigid bodies).

2. **Conservative heat exchange**: In `grid_op_thermal`, instead of decaying toward a static temperature:
   ```python
   Q = h_contact * A * (T_mpm - T_rigid) * dt
   T_mpm -= Q / (m_mpm * Cp_mpm)
   T_rigid += Q / (m_rigid * Cp_rigid)
   ```

3. **Rigid body cooling**: Apply convective cooling to exposed rigid body surfaces.

---

## Phase Dependency Graph

```
Phase 0 (Port)
    │
    ▼
Phase 1A (Fix cooling units) ──────────┐
    │                                   │
Phase 1B (Fix thermal softening) ◄─────┘
    │
    ▼
Phase 2 (Plastic work heating)
    │
    ├──► Phase 3 (Grid diffusion)     [independent, can be parallel]
    ├──► Phase 4 (Thermal expansion)  [independent, can be parallel]
    │
    ▼
Phase 5 (Particle heating API)        [independent]
    │
    ▼
Phase 6 (Two-way rigid coupling)
    │
    ▼
Taylor Impact Test Validation
```

Phases 3, 4, and 5 are independent of each other and can be done in any order after Phase 2.

---

## Summary of Files Modified Per Phase

| Phase | `base_mpm_solver.py` | `mpm_solver.py` | `elasto_plastic.py` | `base.py` (material) | Other materials | `solvers.py` (options) | Tests |
|-------|---------------------|-----------------|---------------------|-----------------------|-----------------|----------------------|-------|
| 0 | Hook grafting | Copy + verify | — | — | — | Add thermal params | Run existing |
| 1A | — | Fix `grid_op_thermal` | — | — | — | Maybe adjust defaults | New: resolution-independence |
| 1B | Pass `temp` to material | Remove stress scale | Add J-C yield | Add `temp` param | Add `temp` param (ignore) | — | New: Hot Bar Test |
| 2 | Add `p2g_post_constitutive` hook | Implement heating | Return `delta_gamma` | Return 4-tuple | Return 4-tuple | — | New: compression heating, Taylor test |
| 3 | — | Add diffusion kernel + buffer | — | — | — | — | New: 1D diffusion |
| 4 | Modify `compute_F_tmp_helper` | — | — | — | — | Add CTE param | New: expansion test |
| 5 | — | Add API methods | — | — | — | — | New: heat source test |
| 6 | — | Modify `grid_op_thermal` | — | — | — | — | New: multi-strike test |
