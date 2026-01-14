# BaseMPMSolver Refactoring Summary

This document summarizes the changes made to `BaseMPMSolver` to support thermal physics extensibility via hooks.

## 1. New Hooks in `BaseMPMSolver`

The following `@ti.func` methods were added. They are empty or default in `BaseMPMSolver` but overridden in `MPMSolver`.

### `p2g_modify_stress(self, f, i_p, i_b, stress)`
- **Location**: `p2g_helper`, after stress computation.
- **Purpose**: Modify the stress tensor before it is used to compute affine momentum.
- **Usage**: Used for **Thermal Softening**.
- **Signature**:
  ```python
  @ti.func
  def p2g_modify_stress(self, f: ti.i32, i_p: ti.i32, i_b: ti.i32, stress: ti.template()):
      return stress * self.get_particle_stress_scale(f, i_p, i_b)
  ```

### `get_particle_stress_scale(self, f, i_p, i_b)`
- **Location**: Helper called by `p2g_modify_stress`.
- **Purpose**: Return a scalar [0, 1] to scale stress.
- **Usage**: Used for **Johnson-Cook Thermal Softening**.

### `p2g_transfer_extra_fields(self, f, i_p, idx, i_b, weight)`
- **Location**: `p2g_helper`, inside the grid scattering loop.
- **Purpose**: Transfer additional particle fields (like Temperature) to grid nodes.
- **Usage**: Maps $m_p T_p \rightarrow m_i T_i$.

### `g2p_prologue(self, f, i_p, i_b)`
- **Location**: `g2p_helper`, before the grid gathering loop.
- **Purpose**: Reset or initialize particle states for the new step.
- **Usage**: Set $T_p^{new} = 0$ before gathering from grid.

### `g2p_transfer_extra_fields(self, f, i_p, i_b, weight, grid_index)`
- **Location**: `g2p_helper`, inside the grid gathering loop.
- **Purpose**: Gather grid fields back to particles.
- **Usage**: Maps $T_i \rightarrow T_p$.

## 2. Updated Kernels

- **`p2g_helper`**: Now calls `p2g_modify_stress` and `p2g_transfer_extra_fields`.
- **`g2p_helper`**: Now calls `g2p_prologue` and `g2p_transfer_extra_fields`.
- **`copy_frame_helper`**: Virtual. Overridden to copy thermal fields.
- **`reset_grid_helper`**: Virtual. Overridden to reset thermal grid fields.

## 3. Important Notes for Developers
- **Do NOT** duplicate `p2g_helper` in derived classes unless absolutely necessary. Use the hooks.
- **Do NOT** add thermal-specific fields to `BaseMPMSolver`. Keep them in `MPMSolver`'s template overrides.
