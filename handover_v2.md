# Handover V2: Thermal Physics Implementation

## 1. Current Status
- **Phase 1: Refactoring `BaseMPMSolver`** - **COMPLETED**
  - `BaseMPMSolver` now uses a Template Method pattern with hooks (`p2g_modify_stress`, `p2g_transfer_extra_fields`, `g2p_prologue`, `g2p_transfer_extra_fields`).
  - `MPMSolver` has been refactored to override these hooks instead of duplicating the entire `p2g_helper` kernel.
  - `enable_thermal` option has been removed; thermal features are active when `use_legacy_solver=False`.
  - Basic thermal Advection and Cooling (Newton law with air) are working (Verified by `tests/test_thermal.py`).

- **Phase 2: Contact Cooling** - **IN PROGRESS**
  - Goal: Implement heat transfer between MPM particles/grid and Rigid bodies (e.g., hot billet on cold die).
  - Status:
    - Test script created: `tests/test_contact_cooling.py`.
    - **Issue**: The test initially crashed due to particles being outside simulation bounds.
    - **Fix**: I updated `lower_bound` in the test script to `(-0.5, -0.5, -0.5)` to accommodate the floor plane.
  - **Immediate Next Step**: Run `tests/test_contact_cooling.py`. It is expected to pass the "simulation run" but FAIL the content check (detecting significant cooling) because the logic isn't implemented yet.

## 2. Next Steps

### A. Implement Contact Cooling
1. **Run Verification**: Run `pixi run python tests/test_contact_cooling.py`. Confirm `[FAIL] Contact cooling NOT effective/detected`.
2. **Implementation Strategy**:
   - You need to detect when particles or grid cells are in contact with rigid bodies.
   - **Option 1 (Grid-based)**: In `MPMSolver.grid_op_thermal`, utilize `sdf_decomp` to check Signed Distance Field (SDF) of rigid bodies at grid locations. If close, apply Newton cooling with `thermal_contact_conductivity`.
   - **Option 2 (Particle-based)**: In `p2g_transfer_extra_fields` (or a separate kernel), check `self._coupler.mpm_rigid_normal` (if populated) or re-query SDFs.
   - **Recommendation**: Grid-based might be easier for conductive cooling. The `BaseMPMSolver` already has `geoms_info`, `geoms_state`, etc. available in `p2g_helper` arguments, but `grid_op_thermal` (currently in `MPMSolver`) might need to accept these arguments to query SDFs. 
   - Note: `LegacyCoupler` updates `mpm_rigid_normal`. Check if this field is available and up-to-date in `MPMSolver`.

### B. Remaining Thermal Features (Phase 3 & 4)
- **Particle Heating**: Add ability to set temperature of specific particles (e.g. laser heating).
- **Plastic Work Heating**: Implement adiabatic heating.
  - *Challenge*: `p2g_modify_stress` returns stress but doesn't easily allow extracting "Plastic Work".
  - *Hint*: You might need to estimate plastic work from the stress calculation or modify the Material's `update_stress` to returns more data (complex).
  - *Alternative*: Calculate $\sigma : \dot{\epsilon}^p$ approximation inside the hook if possible.

## 3. Key Files
- `genesis/engine/solvers/base_mpm_solver.py`: Contains the hooks (Do not revert this file to pre-hook state!).
- `genesis/engine/solvers/mpm_solver.py`: Contains the thermal logic overrides.
- `genesis/engine/couplers/legacy_coupler.py`: Handles rigid-MPM coupling. Useful for contact detection.
- `tests/test_contact_cooling.py`: The active test case for the current task.
