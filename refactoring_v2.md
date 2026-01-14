# Refactoring Plan V2 (Verification Only)

## Status Update
Analysis of `mpm_solver.py` confirms that the **Kernel Refactoring Goal has been largely met**.
- `p2g_helper` is NOT duplicated in `MPMSolver`.
- `g2p_helper` is NOT duplicated in `MPMSolver`.
- Logic is correctly placed in `p2g_transfer_extra_fields` and `g2p_transfer_extra_fields` hooks.

## Remaining Verification Steps (For Next Model)

1.  **Verify Hook Coverage**:
    - Ensure `p2g_modify_stress` is correctly scaling stress for thermal softening (It is).
    - Ensure `g2p_prologue` resets particle temp correctly (It does).

2.  **Future Architecting**:
    - If **Plastic Work Heating** requires access to internal material variables (like `d_gamma`), we might need to modify `BaseMPMSolver.p2g_helper` or `Material.update_stress` to expose these values.
    - *Plan*: Attempt implementation of Plastic Work Heating first. Only refactor `p2g_helper` or hooks if strictly necessary to access plastic strain data.

## Conclusion
The architecture is solid. Proceed to feature implementation (Plastic Work).
