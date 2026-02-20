# PROMPT FOR TRANSFER MODEL

You are an AI assistant placed on an initialized or updated branch of the Genesis repository. Your primary objective is to inherit and expand upon a Thermal Physics implementation originating from an older branch.

## 1. Context Acquisition
Before writing or replacing any code, you must thoroughly understand the architecture developed on the older branch. Avoid reading massive files like `base_mpm_solver_refactor_diff.txt` in their entirety; `grep` for what you need.

**Your Mandatory Reading List:**
1.  `thermal_implementation_status.md`: This file details exactly what the previous model achieved, the architecture chosen (Base class hooks vs Derived class overrides), and the logic for Contact Cooling.
2.  `thermal_future_plan.md`: This file outlines your exact marching orders (Phase 1, Phase 2, etc.).

*Note: You may need to read snippets of the older branch versions of `genesis/engine/solvers/mpm_solver.py` and `genesis/engine/solvers/base_mpm_solver.py` to see the actual hook implementations if they were not transported via patches.*

## 2. Your Immediate Objective: Phase 1 (Porting)
Your task is to manually transplant the thermal features into the *current* version of the codebase.
*   **Warning**: The `base_mpm_solver.py` on your current branch may have received upstream updates. **Do not** blindly overwrite the entire file with the old branch's file. Instead, carefully graft the `p2g_*` and `g2p_*` hooks into the relevant kernels.
*   Once grafted, ensure `mpm_solver.py` overrides these hooks identically to the old branch. Ensure the solver configuration doesn't require a redundant `enable_thermal` option (as it was removed to be unconditionally active in the new `MPMSolver`).

## 3. Secondary Objective: Phase 2 (Plastic Work)
If you successfully port the implementation and verify it compiles, proceed to **Phase 2: Plastic Work Heating** (as detailed in `thermal_future_plan.md`). This will require deep analysis of `genesis/options/materials.py`.

## 4. User Constraints & Guidelines
*   **Optimization**: Ensure Exponential Decay for contact cooling remains intact (explicit Euler explodes with high thermal conductivity limits).
*   **Elegance**: "I want to break the kernels into small pieces/functions so we can use inheritance better." Ensure the hook system logic shines.
*   **Aesthetics**: "Visual/Aesthetic excellence is important". Ensure your implementation is logically sound and mathematically correct.
*   **Awareness of Limitations**: You are inheriting a branch with a few temporary hacks (e.g., using a density check `< 7000.0` for detecting air boundary, assuming steel material; assuming rigid bodies are infinite 293K heatsinks). Acknowledge these when porting. 
*   **The Phase 2 Blocker**: Expect to refactor the `genesis/options/materials.py` `update_stress` functions heavily to achieve Phase 2, as plastic work is currently discarded internally.

Begin your task by confirming you've read the reference documents. Good luck.
