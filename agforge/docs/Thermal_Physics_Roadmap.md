# Genesis / AgilityForge Thermal Physics Roadmap

This document outlines the known missing physical phenomena and architectural improvements required to upgrade the hot forging thermal simulation from its current state to a mathematically rigorous metallurgical digital twin.

## 1. Dynamic Thermal Diffusivity ($\alpha$)
**Priority: HIGH** | **Location:** `genesis/options/solvers.py` & `legacy_coupler.py`

Thermal diffusivity dictates how fast internal heat spreads through a material structure.
- **Current Implementation:** A static `1.1e-5 m^2/s` acts globally.
- **Missing Physics:** Diffusivity mathematically drops by >50% at 1000K because thermal conductivity explicitly decays and specific heat drastically spikes. 
- **Impact:** The intensely hot core of the modeled billet bleeds its heat into the colder outer skin almost twice as fast as it physically should, severely blurring our visual thermal gradients.
- **Solution:** Inject $k(T)$ and $C_p(T)$ into the Laplacian diffusion tensor sequentially.

## 2. Conjugate Heat Transfer (Dynamic Die Thermodynamics)
**Priority: HIGH** | **Location:** `genesis/engine/couplers/legacy_coupler.py`

When the hot billet touches the cold die, the simulation currently initiates "Contact Cooling" successfully.
- **Current Implementation:** The rigid die geometries are modeled as **"Infinite Heat Sinks"** locked permanently at $293.15$K. 
- **Missing Physics:** Real dies rapidly absorb extreme heat during high-speed hammer strikes, diminishing their contact cooling effectiveness on subsequent hits.
- **Impact:** The simulation assumes the anvil/press is always perfectly cold, unnaturally maximizing the thermodynamic bleed-off rate from the billet during continuous striking routines.
- **Solution:** Add macroscopic thermal arrays to `RigidSolver` bodies to track cumulative heat deposition from the MPM grid, eventually mapping cooling lines back into the die.

## 3. Electromagnetics: Induction Edge-Effect Crowding
**Priority: HIGH** | **Location:** `agforge/thermal.py`

The current induction heating model applies heat through a basic spatial distance field (SDF), scaling heat deposition symmetrically based on Euclidean distance away from the boundary mesh using $e^{-\text{depth}/\text{skin\_depth}}$.
- **Missing Physics:** True induction heating is governed by magnetic flux lines, which violently crowd around tight geometries (sharp corners, thin flanges, extrusions). 
- **Impact:** The SDF completely ignores geometric curvature, heating a perfectly flat plate and a sharp 90-degree corner at the exact same uniform rate. This ignores the real-world metallurgical hazard where corners melt before the core even begins to warm up.
- **Solution:** Pre-calculate vertex curvatures on the `reconstructed_mesh` and use the curvature scalar to locally amplify the applied `surface_power` per-particle to approximate flux crowding.

## 4. Thermal Strain (Volumetric Expansion)
**Priority: MEDIUM** | **Location:** `genesis/engine/materials/MPM/elasto_plastic.py`

The viscoplastic Johnson-Cook implementation flawlessly handles **Thermal Softening** (exponentially destroying Yield Stress as particles approach $T_{melt}$). However, it lacks **Thermal Strain**.
- **Missing Physics:** Material expands when heated. The Deformation Gradient tensor ($F$) currently lacks a $\alpha \Delta T$ volumetric expansion multiplier. 
- **Impact:** The steel billet does not visually or physically swell when raised from 293K to 1500K. Consequently, when the forged part eventually cools down, the engine cannot mathematically predict volumetric shrinkage tolerances.
- **Solution:** Inject a thermal strain coefficient ($\alpha$) and structurally expand the diagonal components of the particle $U \cdot S \cdot V^T$ tracking matrices proportionally to the current Delta-T relative to $293.15$K.

## 5. Die Contact Pressure vs. Heat Transfer Coupling
**Priority: MEDIUM** | **Location:** `genesis/engine/couplers/legacy_coupler.py`

- **Current Implementation:** Contact heat exchange drains universally at `5000 W/m²K` regardless of physical pressing force.
- **Missing Physics:** Real microscopic metal air gaps severely limit contact conduction ($\approx 2000$) until the hydraulic press physically crushes the structural bounds together tightly ($\approx 10000+$).
- **Impact:** The cold robotic die artificially rapidly drains heat when it is mechanically just resting gently on the billet.
- **Solution:** Couple $h_{contact}$ geometrically to the localized particle stress tensor $\sigma_{contact}$ near the active boundary layer.

## 6. Dynamic Emissivity (Scale Shedding)
**Priority: LOW** | **Location:** `genesis/engine/couplers/legacy_coupler.py`

- **Current Implementation:** Stefan-Boltzmann Emissivity is globally locked to `0.80`.
- **Missing Physics:** While 0.80 is immensely accurate for oxidized heat scale in open air, violently striking the billet mechanically shatters the black oxide scale, exposing the shiny raw internal steel matrix which possesses an $\approx 0.3$ emissivity.
- **Impact:** Struck boundary surfaces should temporarily cool significantly slower via radiation than undisturbed rough surfaces.

---

## Implemented Features

### Thermal Radiation (Stefan-Boltzmann Law)
*Status: Successfully integrated a linearized exponential decay formulation of the Stefan-Boltzmann radiaton curve directly into the MPM grid solver.*

### High-Temperature Specific Heat ($C_p$) Curves
*Status: Replaced scalar constant with a dynamic piecewise tensor evaluated completely natively within Taichi kernels globally, and mapped directly into Python via numpy arrays.*

---

## Known Bugs in Current Implementation

The following are mathematical or architectural flaws in features that *are* implemented but produce incorrect results.

### Bug 1: ~~Induction Heating is Mass-Invariant (Violates Energy Conservation)~~ — FIXED
**Severity: HIGH** | **Location:** `agforge/thermal.py` (`step_heat`, line 102)

- **The Problem:** `delta_temp = surface_power * exp(-depth / skin_depth) * dt` applied a raw Kelvin delta directly to particles. The `surface_power` parameter acted as a "temperature wand" ($K/s$) rather than an energy source ($W$).
- **The Fix:** Now uses energy-based heating: `dT = P * dt / (m * Cp)`. The `surface_power` parameter represents Watts. Particle mass and heat capacity are read from the solver.
- **Future Improvement:** For maximum physical rigor, the heat source term could be injected directly into the grid thermal kernel (`legacy_coupler.py`) as a volumetric source $\dot{q}$ (W/m³) alongside diffusion and convection, eliminating the operator-splitting approximation. At current microsecond timesteps the splitting error is negligible ($\sim dt^2 \approx 10^{-10}$), so this is deferred until a broader thermal system refactor (e.g., implicit solver).

### Bug 2: Air Convection is Topologically Blind (Corner Cooling Rate)
**Severity: LOW** | **Location:** `genesis/engine/couplers/legacy_coupler.py` (`mpm_grid_op`, lines 423-468)

- **The Problem:** The engine checks if any of the 6 neighboring cells are empty, sets a binary `is_surface = 1` flag, and applies a single exponential decay using $h_{air} \cdot dx^2$ (one cell face area). However, corner cells have 3 exposed faces, edges have 2 exposed faces, and flat surfaces have 1.
- **Expected Behavior:** Corner cells should cool ~3x faster than flat surface cells, and edge cells ~2x faster.
- **Fix:** Replace the binary `is_surface` flag with an integer counter `n_exposed_faces` that accumulates how many neighbors are empty, then scale the cooling coefficient by `n_exposed_faces`.
- **Note:** Practical impact is low for cylindrical billets; the exponential decay formulation further dampens the error.

### Bug 3: Johnson-Cook Missing Strain Hardening and Strain-Rate Sensitivity (Incomplete Feature)
**Severity: MEDIUM** | **Location:** `genesis/engine/materials/MPM/elasto_plastic.py` (`update_F_S_Jp`, lines 82-87)

- **The Problem:** The full Johnson-Cook constitutive model is $\sigma_y = (A + B\varepsilon_p^n)(1 + C \ln \dot{\varepsilon}^*)(1 - T^{*m})$. The current implementation only applies the thermal softening term $(1 - T^{*m})$ to a static base yield stress. The strain-hardening term $(B\varepsilon_p^n)$ and the strain-rate sensitivity term $(C \ln \dot{\varepsilon}^*)$ are completely absent.
- **Expected Behavior:** Hitting metal at 10 m/s should require drastically more force than pressing at 0.1 m/s. Repeated strikes should progressively work-harden the material.
- **Fix:** Track cumulative `plastic_strain` (already stored on particles) and compute `strain_rate` from `delta_gamma / dt`. Use these to compute the full three-term Johnson-Cook yield stress.
- **Note:** This is more of an intentionally simplified model than a strict bug. The `plastic_strain` field is already tracked on particles (line 363) — the data is there, it just isn't fed back into the yield calculation.

### Bug 4: Induction Heating Ignores `thermal_time_scale`
**Severity: HIGH** | **Location:** `agforge/strike_controller.py` & `agforge/thermal.py`

- **The Problem:** The engine's native diffusion and convection kernels in `legacy_coupler.py` correctly scale all thermal coefficients by `thermal_time_scale`. However, the Python-side induction heater (`thermal.py`) receives the raw mechanical `dt` without any scaling applied.
- **Expected Behavior:** If `thermal_time_scale = 10` (making cooling 10x faster visually), induction heating should also inject heat 10x faster to maintain energy balance.
- **Fix:** Multiply the `dt` passed to `step_heat()` by `thermal_time_scale`, or equivalently scale `surface_power` by the same factor.

### Bug 5: Contact Cooling Uses Binary SDF Threshold (Stair-Stepping)
**Severity: LOW** | **Location:** `genesis/engine/couplers/legacy_coupler.py` (`mpm_grid_op`, lines 483-489)

- **The Problem:** Contact cooling activates when `signed_dist < dx` (one grid cell width). This is a hard binary switch: a cell barely touching the die surface receives the exact same cooling intensity as a cell deeply embedded in the die.
- **Expected Behavior:** Contact heat transfer should interpolate smoothly based on the actual contact proximity.
- **Fix:** Weight the contact decay coefficient by `(1.0 - signed_dist / dx)` to create a smooth gradient from full contact cooling at the surface to zero at the threshold boundary.

### Bug 6: `mass` vs `mass_thermal` Gate Mismatch in Grid Operations
**Severity: LOW** | **Location:** `genesis/engine/couplers/legacy_coupler.py` (`mpm_grid_op`, line 410 vs 417)

- **The Problem:** The outer mechanical guard checks `mass > EPS` (line 410) while the thermal normalization block checks `mass_thermal > 0` (line 417). These are accumulated from the same particles with identical B-spline weights, so they *should* always agree — but the thresholds differ. If CPIC ever separates a particle mechanically but not thermally, a cell could pass the mass check but have `mass_thermal == 0`, skipping normalization and leaving raw mass-weighted heat energy as the "temperature."
- **Expected Behavior:** Both gates should use the same threshold (`> EPS` or `> 0`).
- **Fix:** Change `mass_thermal > 0` to `mass_thermal > gs.EPS` for consistency, or unify both checks.

### Bug 7: `set_particles_pos()` Silently Wipes Thermal State
**Severity: MEDIUM** | **Location:** `genesis/engine/solvers/base_mpm_solver.py` (`_kernel_set_particles_pos`, lines 999-1007)

- **The Problem:** Any call to `set_particles_pos()` resets each affected particle's `temp` back to `default_initial_temperature`, and zeros out `plastic_strain` and `plastic_work`. This means any code path that repositions particles — resets, constraint enforcement, undo operations — silently destroys the entire accumulated thermal history.
- **Expected Behavior:** Position updates should not implicitly reset thermal state unless explicitly requested.
- **Fix:** Either remove the thermal reset from `_kernel_set_particles_pos` (add a separate `_kernel_reset_particles_thermal` for explicit resets), or add a boolean flag `reset_thermal=True` to the Python-side `set_particles_pos()` API so callers can opt out.

### Bug 8: Wasted Thermal Kernel Work When Heating Is Disabled
**Severity: LOW** | **Location:** `agforge/strike_controller.py` (lines 490-492) & `genesis/engine/couplers/legacy_coupler.py`

- **The Problem:** When `thermal_enabled` is `False` in the strike controller, the code snapshots particle temperatures before the physics step and restores them afterward. However, the engine's Taichi kernels for thermal normalization, air cooling, contact cooling, and grid diffusion still execute every substep — all that GPU work is silently thrown away by the Python-level overwrite.
- **Expected Behavior:** If the user disables heating, the engine should skip thermal kernel execution entirely rather than computing and discarding it.
- **Fix:** Tie the `thermal_enabled` flag in the strike controller to `MPMOptions.enable_thermal` so the engine can skip thermal kernel dispatch at the Taichi level. Alternatively, remove the snapshot/restore hack and just set `enable_thermal=False` in options when heating is not wanted.

---

*Note: The current physics engine excellently captures Adiabatic Self-Heating (Taylor-Quinney Plastic Work), transforming $90\%$ of `delta_gamma` plastic yielding directly into particle heat via `p2g_post_constitutive`. The `particle_volume_scale` (1e3) used for numerical stability cancels cleanly in the `rho = mass / volume` calculation, so the adiabatic heating math is dimensionally correct.*
