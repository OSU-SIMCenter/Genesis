# Handover V3: Thermal Physics Implementation

## 1. Project Overview
We are implementing thermal physics into the Genesis MPM solver.
The architecture uses a `BaseMPMSolver` (logic-free base) and an `MPMSolver` (extends base, adds thermal logic).
We are ensuring the new `MPMSolver` remains compatible with the original API but adds thermal features when `use_legacy_solver=False`.

## 2. Current Progress
- **Framework Implemented**: `BaseMPMSolver` refactored with hooks (`p2g_modify_stress`, `p2g_transfer_extra_fields`, etc.). `MPMSolver` overrides hooks and does NOT duplicate kernel logic.
- **Thermal Fields**: `temp`, `plastic_strain`, `plastic_work`, `mass_thermal` added to templates.
- **Advection**: Thermal P2G/G2P transfer implemented via `p2g/g2p_transfer_extra_fields` hooks.
- **Cooling**:
    - **Air Cooling**: Newton cooling with exponential decay (stable) for surface nodes.
    - **Contact Cooling**: Grid-based SDF check against RigidBodies. Implemented in `grid_op_thermal`, called from `substep_pre_coupling`. Stability fixed via exponential decay.
- **Verification**: `tests/test_contact_cooling.py` passes (stable, equilibrates).

## 3. Remaining Tasks (Prioritized)

### Task 1: Plastic Work Heating (High Priority)
**Goal**: Convert plastic deformation energy into heat (Adiabatic Heating).
**Challenge**: Accessing $\dot{\epsilon}_p$ or Plastic Work inside the `p2g` loop.
**Strategy**:
- In `p2g_modify_stress` (or similar hook), approximate work done.
- Or, refactor Material models to return plastic work increment.

### Task 2: Advanced Diffusion (Medium Priority)
**Goal**: Implement proper heat diffusion (Laplacian).
**Current Status**: Placeholder/commented out in `grid_op_thermal`.
**Strategy**: Implement Grid-based explicit diffusion $T += \alpha \nabla^2 T dt$ in `grid_op_thermal`.

### Task 3: Particle Heating (Laser/Boundary) (Low Priority)
**Goal**: Ability to set temperature of specific particles or apply heat flux.
**Strategy**:
- Add `set_temperature(mask, val)` API.
- Add `apply_heat_flux` kernel.

## 4. Key Files
- `genesis/engine/solvers/base_mpm_solver.py`: The base class.
- `genesis/engine/solvers/mpm_solver.py`: The active thermal solver. Contains `grid_op_thermal`.
- `tests/test_contact_cooling.py`: Verification script.
- `genesis/engine/couplers/legacy_coupler.py`: Rigid-MPM coupling logic.
