# BaseMPMSolver Refactoring Plan

## Problem
`MPMSolver` overrides `p2g_helper`, `g2p_helper`, and `reset_grid_helper` by copy-pasting the entire original function. This leads to code duplication and drift.

## Solution
Decompose large kernels in `BaseMPMSolver` into "Template Method" style hooks.

## Proposed Refactoring

### 1. `p2g_helper` Decomposition
Break `p2g_helper` into:
1.  `p2g_prologue(f, i_p, i_b)`: Prepare data, read state.
2.  `update_deformation(f, i_p, i_b)` -> `(F_new, Jp_new)`: Material update.
3.  `compute_stress(f, i_p, i_b, ...)` -> `stress`: Material stress.
4.  `post_stress_update(f, i_p, i_b, stress)` -> `stress_modified`: **Hook for Thermal Softening**.
5.  `p2g_loop_prologue(...)`: Setup stencil.
6.  `p2g_loop_body(offset, weight, ...)`: **Hook for transferring other fields (Thermal P2G)**.
7.  `p2g_epilogue(...)`: Finalize.

**Better Approach for Taichi**:
Since passing many variables between `ti.func`s is messy, a "Context Struct" or just simpler inheritance hooks might be better.
Actually, the most effective hook is inside the loops.

**Proposed Hooks in `BaseMPMSolver`**:

```python
@ti.func
def p2g_modify_stress(self, f, i_p, i_b, stress):
    return stress # Base implementation does nothing

@ti.func
def p2g_transfer_extra_fields(self, f, idx, i_b, weight):
    pass # Base implementation does nothing
```

**Refactored `p2g_helper` in Base**:
```python
# ... inside loop ...
if sep_geom_idx == -1:
    self.grid[...].vel_in += ...
    self.grid[...].mass += ...
    
    # NEW HOOK
    self.p2g_transfer_extra_fields(f, base - self._grid_offset + offset, i_b, weight)
```

**Refactored `p2g_helper` in Thermal (Derived)**:
No need to override `p2g_helper`. Just override:
```python
@ti.func
def p2g_modify_stress(self, f, i_p, i_b, stress):
    return stress * self.get_particle_stress_scale(...)

@ti.func
def p2g_transfer_extra_fields(self, f, idx, i_b, weight):
    # Thermal transfer logic here
    mass = self.particles_info[i_p].mass
    temp = self.particles[f, i_p, i_b].temp
    self.grid[f, idx, i_b].temp += weight * mass * temp
    self.grid[f, idx, i_b].mass_thermal += weight * mass
```

### 2. `grid_op` Decomposition
Currently `reset_grid_helper` is okay, but `grid_op` itself (which might be in `substep_pre_coupling`?) handles grid normalization.

The base solver actually *doesn't* have a monolithic `grid_op`. It has `reset_grid_and_grad` and `p2g`/`g2p`.
The thermal solver added `grid_op_thermal` separately. This is actually fine! It doesn't need refactoring.

### 3. `g2p_helper` Decomposition
Similar to P2G.
**Proposed Hook**:
```python
@ti.func
def g2p_transfer_extra_fields(self, f, i_p, i_b, weight, grid_index):
    pass
```

## Execution Steps for Refactoring Agent

1.  **Modify `BaseMPMSolver`**:
    - Add `p2g_modify_stress` (identity).
    - Add `p2g_transfer_extra_fields` (no-op).
    - Add `g2p_transfer_extra_fields` (no-op).
    - Insert calls to these hooks in `p2g_helper` and `g2p_helper`.
2.  **Modify `MPMSolver`**:
    - **Delete** the huge copy-pasted `p2g_helper`.
    - Implement `p2g_modify_stress` with Johnson-Cook logic.
    - Implement `p2g_transfer_extra_fields` with thermal transfer logic.
    - **Delete** `p2g_thermal_transfer` kernel (it becomes integrated into the main loop, which is MORE EFFICIENT).
    - Do the same for G2P.
