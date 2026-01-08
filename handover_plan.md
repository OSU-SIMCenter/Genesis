# Thermal Physics Handover Plan

## Current State Analysis

**Completed:**
- **Infrastructure**: `MPMSolver` infrastructure is ready. `enable_thermal` option removed; thermal fields (`temp`, `plastic_strain`, `plastic_work`) are always valid.
- **Coupling**: `get_particle_stress_scale` implements Johnson-Cook thermal softening ($1 - T^{*m}$).
- **Advection**: Thermal P2G and G2P transfers are implemented.
- **Basic Cooling**: `grid_op_thermal` contains a simplified Newton cooling model ($dT = -dt \cdot h \cdot (T - T_{air})$) applied to "surface-like" cells (low density).

**Missing / To Be Implemented:**
1.  **Contact Cooling**: Heat transfer between the MPM material and rigid boundaries/tools (e.g., the die in foraging). Currently, the grid boundary is adiabatic (insulated) unless it hits the "low density" air check.
2.  **Particle Heating**: There is no explicit mechanism to apply external heat sources to specific particles (e.g., induction heating, initial uneven temperature distribution).
3.  **Advanced Diffusion**: The diffusion term (Laplacian) is commented out/placeholder.
4.  **Plastic Work Heating**: The adiabatic heating from plastic work is currently a `pass` in `p2g_helper`. It needs to be approximated from stress/strain if the constitutive model doesn't export plastic strain increment.

## Future Tasks Breakdown

### Task 1: Refactoring Base Solver (Kernel Decomposition)
**Goal**: Break down `BaseMPMSolver` kernels (`p2g`, `g2p`, `grid_op`) into smaller, overridable `ti.func` components.
- **Why**: Currently, `MPMSolver` has to copy-paste the *entire* `p2g_helper` just to insert one line of thermal logic. This is unmaintainable.
- **Approach**: See `refactoring_plan.md`.

### Task 2: Implement Contact Cooling
**Goal**: Allow heat transfer between MPM particles/grid and coupling bodies (RigidSolver properties).
- **Implementation**:
    - In `grid_op_thermal` or a new `grid_op_boundary`, check if a cell is near a rigid collider.
    - Use `self.sim.coupler` info to determine contact.
    - Apply heat flux based on `thermal_contact_conductivity`.

### Task 3: Implement Plastic Work Heating
**Goal**: Convert plastic deformation energy into heat.
- **Implementation**:
    - In `p2g_helper` (derived), calculate/approximate $W_p = \sigma : \dot{\epsilon}_p \Delta t$.
    - Since `d_gamma` (plastic multiplier) is buried in material models, consider:
        - **Option A**: Refactor `Material` classes to return `d_gamma` or `plastic_work_inc`.
        - **Option B (Easier)**: Approximate in `p2g` using $(\sigma : \Delta \epsilon)$ and checking yield condition.
    - Add $W_p / C_p$ to temperature.

### Task 4: Particle Heating / Boundary Conditions
**Goal**: Apply external heat.
- **Implementation**:
    - precise: Add a `HeatSourceEntity` or similar.
    - simple: Add a `set_temperature(mask, temp)` method exposed to Python. 
    - In `substep_pre_coupling`, iterate over particles and apply explicit heat sources if defined.
