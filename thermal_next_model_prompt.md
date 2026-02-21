# PROMPT FOR TRANSFER MODEL

You are an AI assistant placed on an initialized or updated branch of the Genesis repository. Your primary objective is to inherit and extend a Thermal Physics implementation originating from an older branch.

## 1. Mandatory Reading (In Order)

Before writing or modifying any code, you MUST read and understand these documents:

1. **`thermal_implementation_status.md`**: What has been implemented, the exact hook signatures, the known bugs, and the critiques. Pay special attention to §8 (Known Issues) — several existing implementations are physically incorrect and must be fixed.

2. **`thermal_research_reference.md`**: The theoretical foundations: governing equations, Johnson-Cook model with parameter table, thermal expansion, heat transfer physics, Taichi architectural constraints, the full integration loop, and validation benchmarks.

3. **`thermal_implementation_plan.md`**: Your marching orders. Phases 0–6 with concrete steps, file lists, code sketches, and validation criteria. Follow the phase dependency graph.

You may also need to read the actual source files on the reference branch:
- `genesis/engine/solvers/mpm_solver.py` — thermal solver overrides
- `genesis/engine/solvers/base_mpm_solver.py` — hook insertion points in `p2g_helper` and `g2p_helper`
- `genesis/engine/materials/MPM/elasto_plastic.py` — current von Mises return mapping (this is where J-C yield must go)
- `genesis/options/solvers.py` — MPMOptions thermal parameters

## 2. Your Immediate Objectives

### If you are on a NEW/UPDATED branch (Phase 0):
Port the thermal implementation from the reference branch to the current branch. Do NOT copy-paste entire files — the base solver may have received upstream changes. Carefully graft the hooks into the updated `p2g_helper` and `g2p_helper`. See Phase 0 in the implementation plan.

### If the port is already done (Phases 1+):
Your priorities in order:
1. **Fix the cooling rate constants** (Phase 1A) — they use wrong units and produce non-physical results.
2. **Move thermal softening into the constitutive model** (Phase 1B) — it's currently a post-hoc stress scale, which breaks the thermal-mechanical feedback loop.
3. **Implement plastic work heating** (Phase 2) — this is the primary missing physics.
4. **Validate** with the Hot Bar Test and then the Taylor Impact Test.

## 3. Critical Constraints

- **Do NOT duplicate kernels.** Use the hook-based architecture. If you need new behavior, add a new hook to `BaseMPMSolver` (as a no-op) and override it in `MPMSolver`.
- **Do NOT break existing materials.** When adding `temp` as a parameter to `update_F_S_Jp`, ensure all other MPM material classes (`elastic.py`, `sand.py`, `snow.py`, `liquid.py`, `muscle.py`) accept and ignore the new parameter.
- **Do NOT scale the entire stress tensor for thermal softening.** The yield stress must be modified inside the return mapping. See Phase 1B.
- **Use exponential decay** for all cooling/boundary heat transfer. Explicit Euler diverges at high conductivity.
- **Maintain differentiability.** Prefer `ti.max`/`ti.min` over hard `if/else` on continuous variables. Avoid singularities (e.g., `ti.max(denominator, 1e-10)`).
- **Use proper units.** All thermal rate constants must account for material density, heat capacity, and cell geometry. See `thermal_research_reference.md` §4.
- **Keep the `use_legacy_solver` switch working.** When True, the solver must behave identically to the upstream `BaseMPMSolver` with zero overhead.

## 4. Key Physics Reminders

- Johnson-Cook yield: $\sigma_y = (A + B\varepsilon_p^n)(1 + C\ln\dot\varepsilon/\dot\varepsilon_0)(1 - T^{*m})$
- Adiabatic heating: $\Delta T = \chi \sigma_y \Delta\gamma / (\rho C_p)$, where $\chi = 0.9$
- Thermal expansion: $\mathbf{F}^\theta = (1 + \alpha_{th}(T - T_{ref}))\mathbf{I}$
- Grid diffusion CFL: $\Delta t < \Delta x^2 / (6\alpha)$ — always satisfied for typical MPM timesteps with steel

## 5. Validation Checkpoints

| After Phase | Test | Expected Result |
|-------------|------|-----------------|
| 0 | `test_contact_cooling.py` | Runs without crash, temp drops from 1000K |
| 1A | Cooling rate resolution-independence | Similar cooling rate at grid_density 32 vs 64 |
| 1B | Hot Bar Test | Hot bar (1000K) deforms more than cold bar (300K) under same load |
| 2 | Compression heating | Temperature increases during plastic deformation of steel cube |
| 2 | Taylor Impact Test | Mushrooming with thermal concentration at impact end; larger deformation than isothermal case |

Begin by confirming you have read and understood the three reference documents. Then state which phase you are starting from based on the current state of the branch.
