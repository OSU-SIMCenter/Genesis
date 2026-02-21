# Thermal Physics Implementation Status (Reference Branch)

This document provides a comprehensive record of what has been implemented on this (older) branch, how it works, and what is known to be incorrect or incomplete. Read this first to understand the current state before making changes.

## 1. Architectural Overview

The implementation uses a **Template Method** pattern: `BaseMPMSolver` provides hook points (empty `@ti.func` methods) that the derived `MPMSolver` overrides to inject thermal physics. This avoids duplicating the large `p2g_helper` and `g2p_helper` kernels.

**Solver selection** is controlled by `MPMOptions.use_legacy_solver`:
- `True` (default): `MPMSolver.__new__` returns a plain `BaseMPMSolver` (no thermal fields, original behavior).
- `False`: Returns a proper `MPMSolver` instance with all thermal fields and logic active.

### Key Files

| File | Role |
|------|------|
| `genesis/engine/solvers/base_mpm_solver.py` | Base class with hook insertion points |
| `genesis/engine/solvers/mpm_solver.py` | Derived class with all thermal logic |
| `genesis/options/solvers.py` | `MPMOptions` with thermal parameters |
| `genesis/engine/materials/MPM/elasto_plastic.py` | Von Mises return mapping (relevant to plastic work) |
| `genesis/engine/materials/MPM/base.py` | Base material `update_stress` (Neo-Hookean) |
| `tests/test_thermal.py` | Basic cooling verification |
| `tests/test_contact_cooling.py` | Contact cooling verification |

## 2. Hooks in `BaseMPMSolver`

The following `@ti.func` methods were added to `BaseMPMSolver`. They are no-ops in the base class and overridden in `MPMSolver`.

### `p2g_modify_stress(self, f, i_p, i_b, stress) -> stress`
- **Called from**: `p2g_helper`, after `update_stress` computes the Cauchy stress tensor (line ~394).
- **Base behavior**: Returns `stress * self.get_particle_stress_scale(f, i_p, i_b)` where scale = 1.0.
- **Override**: Applies thermal softening scale via `get_particle_stress_scale`.

### `p2g_transfer_extra_fields(self, f, i_p, idx, i_b, weight)`
- **Called from**: `p2g_helper`, inside the grid scattering loop (line ~439), after mass and momentum are scattered.
- **Base behavior**: `pass`.
- **Override**: Scatters mass-weighted temperature to grid (`grid.temp`, `grid.mass_thermal`).

### `g2p_prologue(self, f, i_p, i_b)`
- **Called from**: `g2p_helper`, at the very start before the gather loop (line ~471).
- **Base behavior**: `pass`.
- **Override**: Resets `particles[f+1].temp = 0.0`, copies `plastic_strain` and `plastic_work` forward.

### `g2p_transfer_extra_fields(self, f, i_p, i_b, weight, grid_index)`
- **Called from**: `g2p_helper`, inside the grid gathering loop (line ~502), after velocity/C gather.
- **Base behavior**: `pass`.
- **Override**: Gathers grid temperature back to particles.

### `get_particle_stress_scale(self, f, i_p, i_b) -> float`
- **Called from**: `p2g_modify_stress`.
- **Base behavior**: Returns `1.0`.
- **Override**: Computes Johnson-Cook thermal softening term `(1 - T*^m)`.

### `copy_frame_helper(self, source, target, i_p, i_b)`
- **Called from**: `g2p` kernel for inactive particles.
- **Override**: Copies `temp`, `plastic_strain`, `plastic_work` in addition to base fields.

### `reset_grid_helper(self, f, i, j, k, i_b)`
- **Called from**: `reset_grid_and_grad` kernel.
- **Override**: Zeroes `grid.temp` and `grid.mass_thermal` in addition to base fields.

## 3. Thermal Fields

### Particle Fields (added via `get_particle_state_template`)
| Field | Type | Description |
|-------|------|-------------|
| `temp` | `ti_float` | Temperature in Kelvin. Initialized to `MPMOptions.default_initial_temperature`. |
| `plastic_strain` | `ti_float` | Accumulated equivalent plastic strain. Currently always 0 (placeholder). |
| `plastic_work` | `ti_float` | Accumulated plastic dissipation energy. Currently always 0 (placeholder). |

### Grid Fields (added via `get_grid_cell_state_template`)
| Field | Type | Description |
|-------|------|-------------|
| `temp` | `ti_float` | Grid temperature (mass-weighted sum during P2G, normalized in `grid_op_thermal`). |
| `mass_thermal` | `ti_float` | Thermal mass accumulator for normalization. |

## 4. Thermal Advection (P2G / G2P Cycle)

Temperature is advected via the standard MPM transfer:

**P2G (scatter)**: Each particle scatters `weight * mass * temp` to `grid.temp` and `weight * mass` to `grid.mass_thermal`.

**Grid normalization** (in `grid_op_thermal`): `grid.temp /= grid.mass_thermal` recovers the mass-averaged temperature at each node.

**G2P (gather)**: Each particle gathers `sum(weight * grid.temp)` to reconstruct its temperature from the diffused grid field.

This cycle ensures temperature is carried with the material and naturally smoothed through the grid interpolation.

## 5. Cooling Mechanisms (`grid_op_thermal`)

### Air Cooling (Newton's Law)
Applied to grid cells where `rho_cell < 7000.0` (intended to detect "surface" cells).
Uses exponential decay for numerical stability: `dT = (T_air - T_curr) * (1 - exp(-dt * h_conv))`.

### Contact Cooling (Rigid Body SDF)
For each grid cell, queries the Signed Distance Field of all rigid geometries. If a node is within `1.5 * dx` of a rigid surface, applies conductive cooling toward the rigid body temperature (hardcoded 293.15K).
Also uses exponential decay: `dT = (T_rigid - T_curr) * (1 - exp(-dt * h_contact))`.

### Thermal Softening
`get_particle_stress_scale` computes: `scale = 1.0 - T*^m` where `T* = (T - T_ref) / (T_melt - T_ref)`, clamped to [0, 1]. The Johnson-Cook exponent `m = 1.03` is hardcoded for 4340 steel.

## 6. Checkpoint System

`MPMSolver` extends the checkpoint system to save/restore `temp`, `plastic_strain`, and `plastic_work` via dedicated Taichi kernels (`_kernel_save_thermal_state`, `_kernel_load_thermal_state`).

## 7. MPM Options (Thermal Parameters)

Defined in `genesis/options/solvers.py` on the `MPMOptions` dataclass:

| Parameter | Default | Unit | Description |
|-----------|---------|------|-------------|
| `default_initial_temperature` | 293.15 | K | Starting temperature for all particles |
| `default_thermal_diffusivity` | 1.1e-5 | m²/s | Thermal diffusivity (steel) |
| `default_heat_capacity` | 450.0 | J/(kg·K) | Specific heat capacity (steel) |
| `thermal_contact_conductivity` | 50.0 | W/(m²·K) | Contact conductance coefficient |
| `T_ref` | 293.15 | K | Reference temperature (J-C model) |
| `T_melt` | 1793.0 | K | Melting temperature (J-C model) |
| `use_legacy_solver` | True | bool | False activates thermal MPMSolver |

## 8. Known Issues & Critiques

These are problems that exist in the current implementation and must be addressed.

### CRITICAL: Thermal Softening Applied Incorrectly
The `p2g_modify_stress` hook scales the **entire Cauchy stress tensor** by `(1 - T*^m)`. This is physically wrong. The J-C thermal softening term should reduce the **yield stress** inside the return mapping (constitutive update), not scale all stress post-hoc.

**Consequences**:
- Elastic stress is incorrectly reduced (should affect elastic moduli, not stress directly).
- The return mapping in `ElastoPlastic.update_F_S_Jp` uses the **unmodified** (room-temperature) yield stress, so too little plastic strain is computed at elevated temperatures.
- The thermal-mechanical feedback loop (heat → soften → more deformation → more heat) is broken.

**Required fix**: Temperature must be passed into the constitutive model so thermal softening modifies the yield condition *during* the return mapping. See Phase 1B in the implementation plan.

### CRITICAL: Cooling Rate Constants Have Wrong Units
In `grid_op_thermal`, the exponential decay uses `h_conv = 50.0` and `h_contact = 5000.0` directly as rate constants in `exp(-dt * h)`. Newton's law of cooling gives the rate constant as:

```
k_rate = h * (A/V) / (rho * Cp)    [units: 1/s]
```

The code treats `h` (W/m²·K) as if it were `k_rate` (1/s), making cooling ~100,000x too fast and resolution-dependent (changing `grid_density` changes `dx` which changes the surface-to-volume ratio, but the code doesn't account for this).

**Required fix**: Compute the proper rate constant using material density, heat capacity, and the cell's surface-to-volume ratio. See Phase 1A in the implementation plan.

### MODERATE: Air Cooling Surface Detection Is Material-Specific
The `rho_cell < 7000.0` check assumes the material is steel (~7800 kg/m³). This hardcodes a material assumption into the solver.

**Required fix**: Use a relative threshold (e.g., `rho_cell < 0.9 * rho_material`) or detect empty neighboring cells.

### MODERATE: `__new__` Pattern Is Fragile
`MPMSolver.__new__` uses `inspect.signature` binding and mutates the options object via `del options.use_legacy_solver`. This breaks if `BaseMPMSolver.__init__`'s signature changes upstream.

**Required fix**: Use a simpler dispatch mechanism (e.g., factory function in `simulator.py`, or check the option inside `__init__` without `del`).

### LOW: J-C Exponent m=1.03 Is Hardcoded
Should be an `MPMOptions` parameter or material property, not a magic number in `get_particle_stress_scale`.
