# Thermal Physics Porting & Future Plan

This document outlines the sequenced plan to port the existing older-branch implementation to the updated main branch, followed by the newly prioritized thermal capabilities.

## Phase 1: Porting Implementation to Updated Branch
Because this reference branch is outdated, your first major objective is to safely reconstruct the architectural changes on the *current/updated* branch.

**Steps:**
1.  **Read Reference Code**: Analyze the changes in the reference files (`genesis/engine/solvers/base_mpm_solver.py` and `genesis/engine/solvers/mpm_solver.py`) on this branch. Check `thermal_implementation_status.md` to see exactly what hooks were created and how they were overridden. Do not read massive reference txt files fully, just grep for the relevant hook definitions (e.g. `p2g_modify_stress`, `grid_op_thermal`).
2.  **Apply Hooks to Updated Base Solver**: Manually apply the refactoring (the `p2g_*, g2p_*` hooks) into the *updated* version of `base_mpm_solver.py`. Do NOT just copy-paste the whole file, as you may overwrite recent upstream Genesis changes.
3.  **Construct Updated `MPMSolver`**: Copy over the thermal additions to the `MPMSolver` subclass, checking for API compatibility with the new branch. Make sure `grid_op_thermal` and the exponential-decay-based Contact Cooling stay intact.
4.  **Validate Port**: Ensure `tests/test_contact_cooling.py` runs and the simulation behaves correctly on the new branch.

## Phase 2: Implement Plastic Work Heating (Adiabatic Heating)
**Priority**: HIGH
Currently, energy generated from plastic deformation is lost. We need to convert this mechanical work into thermal energy.
*   **Mechanics**: $W_p = \chi \boldsymbol{\sigma} : \mathbf{D}^p$. For metals, the Taylor-Quinney coefficient $\chi$ is typically 0.9.
*   **The Blocker**: The energy from deformation is calculated deep inside the material constitutive models (`genesis/options/materials.py`, inside the `update_stress` hook) and discarded locally in the Taichi kernel.
*   **Strategy**:
    *   You must carefully refactor the material logic to return the `plastic_strain_increment` ($d\gamma$) back up to the `p2g_modify_stress` hook (or a new dedicated hook) so it can be captured by `MPMSolver`.
    *   Once captured, inject the heat: $dT = \frac{0.9 \cdot \sigma_y \cdot d\gamma}{\rho \cdot C_p}$.
    *   *Reference Values* (4340 Steel): $C_p = 450.0$ J/(kg K), $\rho = 7800$ kg/$m^3$.

## Phase 3: Advanced Heat Diffusion
**Priority**: MEDIUM
Currently, `grid_op_thermal` handles boundary/contact cooling but internal conduction between MPM particles is merely approximated by particle interaction (which is poor).
*   **Mechanics**: Proper diffusion requires calculating the Laplacian of the temperature field: $\frac{\partial T}{\partial t} = \alpha \nabla^2 T$.
*   **Strategy**:
    *   Implement an explicit (or implicit if performance requires) grid-based Laplacian.
    *   In `grid_op_thermal`, compute the temperature Laplacian from adjacent grid cells: $T_{i,j,k} += \alpha \cdot \Delta t \cdot \left( T_{i\pm1} + T_{j\pm1} + T_{k\pm1} - 6T_{i,j,k} \right) / dx^2$.
    *   *Constraint*: Must handle double buffering or Gauss-Seidel carefully to avoid parallel read/write race conditions inside the `ti.kernel`.

## Phase 4: Explicit Particle Heating
**Priority**: LOW
*   Provide an API endpoint (e.g., exposed to the user script) to inject heat (`set_temperature_mask`, `apply_laser_flux`) to explicit particle buffers dynamically during the step loop.

## Phase 5: Two-Way Thermal Coupling (Rigid Bodies)
**Priority**: LOW
*   Currently, rigid bodies act as infinite heatsinks locked at `293.15K`.
*   Update the rigid solver to maintain a thermal state. Update `grid_op_thermal` to perform conservative heat exchange (energy lost by MPM = energy gained by rigid tool) rather than simple exponential decay towards a static target.
*   Replace the hacky density check (`rho_cell < 7000.0`) for Air Cooling with a robust surface-detection methodology.

## Final Verification: The Taylor Impact Test
Once the above phases are complete, the implementation must be validated using a fundamental thermomechanical benchmark.
*   **Setup**: A cylindrical bar of 4340 Steel ($L=32.4$mm, $D=6.4$mm) travels at 200 m/s and impacts a rigid wall.
*   **Expected Result (Mechanical)**: The bar mushrooms at the impact end.
*   **Expected Result (Thermal)**: The impact end should heat up significantly (due to $\dot{Q}_{plastic}$). The deformation should be visually larger than an isothermal simulation because the heat softens the material ($T \uparrow \implies \sigma_y \downarrow$).
