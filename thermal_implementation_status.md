# Thermal Physics Implementation Status (Reference Branch)

This document provides a comprehensive overview of the thermal physics implementation completed on this specific (older) branch. Use this as a reference when porting the changes over to the updated branch.

## 1. Architectural Overview
The core philosophy of this implementation is to keep the `BaseMPMSolver` logic-free regarding thermal physics, acting strictly as a mechanical solver but providing "Template Method" hooks. The derived `MPMSolver` overrides these templates/hooks to inject thermal physics (advection, contact cooling, thermal softening).

### `genesis/engine/solvers/base_mpm_solver.py`
This file was refactored to support extensibility:
*   **Virtual Hooks Added**:
    *   `p2g_modify_stress(self, f, i_p, i_b, stress)`: Called inside `p2g_helper` after stress computation. Returns (modified) stress. Base returns `stress`.
    *   `p2g_transfer_extra_fields(self, f, i_p, idx, i_b, weight)`: Called inside the `p2g_helper` scatter loop. Base is `pass`.
    *   `g2p_prologue(self, f, i_p, i_b)`: Called at the start of `g2p_helper`. Base is `pass`.
    *   `g2p_transfer_extra_fields(self, f, i_p, i_b, weight, grid_index)`: Called inside the `g2p_helper` gather loop. Base is `pass`.
    *   `copy_frame_helper` and `reset_grid_helper` were restructured inside the class to be easily overridable.
*   **Kernel Adjustments**:
    *   `p2g_helper` and `g2p_helper` were updated to correctly execute the aforementioned hooks inline.
    *   Removed `enable_thermal` configurations. Thermal activation is handled strictly by Python class instantiation (`use_legacy_solver=False` yields the thermal `MPMSolver`).

### `genesis/engine/solvers/mpm_solver.py`
This is the heavily modified subclass containing the actual thermal logic:
*   **Struct Additions**: Extends particle templates with `temp`, `plastic_strain`, `plastic_work`. Extends grid templates with `temp`, `mass_thermal`.
*   **Advection**:
    *   Overrides `p2g_transfer_extra_fields` to scatter mass-weighted temperature ($m_p T_p$) to grid nodes (`grid.mass_thermal` and `grid.temp`).
    *   Overrides `g2p_prologue` to reset `particle.temp = 0.0` for the next frame.
    *   Overrides `g2p_transfer_extra_fields` to gather grid temp back to particles.
*   **Thermal Softening**: Overrides `get_particle_stress_scale` to implement the Johnson-Cook thermal softening exponent ($1 - T^{*m}$), which is consumed by the overridden `p2g_modify_stress`.
*   **Cooling (`grid_op_thermal`)**: A new kernel invoked from an overridden `substep_pre_coupling`.
    *   **Air Cooling**: Applies Newton cooling to low-density grid cells mimicking "surface exposure."
    *   **Contact Cooling**: Checks Signed Distance Field (SDF) of rigid bodies. If a grid node is within `1.5 * dx` of a rigid collider, it applies heat conduction towards the rigid body.
    *   *Stability*: Both cooling methods use exact exponential decay (`decay = 1.0 - ti.exp(-h*dt)`) to ensure numerical stability even with extremely high conductivity (`h`) coefficients or large `dt`.

### `tests/test_contact_cooling.py`
*   A runtime test was added to verify the simulation doesn't crash and qualitatively verify the cooling logic executes (the scene's particles stabilize to the rigid environment's temperature).

## 2. Critiques & Limitations (For the Next Model)
While the core architecture is robust and mathematically stable, there are temporary assumptions that require future refinement:

*   **The "Air Cooling" Heuristic is Hacky**: Currently, air exposure is determined by checking if the grid cell density is artificially low (`rho_cell < 7000.0`). This hardcodes an assumption that the material is steel (~7800 density). 
    * *Fix Required*: The next model should eventually replace this with a proper surface-detection algorithm (e.g., checking for empty neighboring grid cells, utilizing a color field gradient, or using boundary tracking).
*   **Static Rigid Temperatures**: The contact cooling assumes the rigid tools are infinitely massive heatsinks locked at `293.15K` (room temp).
    * *Fix Required*: If the simulation requires two-way thermal coupling (e.g., the forging dies heat up over time), the rigid solver needs its own thermal state arrays, and the heat transfer equation must become conservative (energy lost by MPM = energy gained by rigid body).
*   **The Plastic Work Heating Blocker**: The biggest challenge for Phase 2 (Plastic Work Heating) is that the energy from deformation ($\sigma : \dot{\epsilon}_p$) is calculated deep inside the material constitutive models (`Material.update_stress` hook) and discarded locally in the Taichi kernel.
    * *Fix Required*: The next model must carefully refactor the material logic to return the `plastic_strain_increment` or `plastic_work` back up to the `p2g_modify_stress` hook so it can be converted into heat.
