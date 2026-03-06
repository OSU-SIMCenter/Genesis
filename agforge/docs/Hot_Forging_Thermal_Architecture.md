# Hot Forging Thermal Architecture

## Overview
Simulating hot forging processes using the Material Point Method (MPM) presents significant computational challenges due to the stark difference between mechanical and thermal timescales. Real-world forging takes seconds, while explicit MPM mechanical stability (CFL condition) dictates microsecond or nanosecond timesteps. Simulating the full physical time is computationally infeasible.

To bridge this gap and achieve observable thermal evolution (cooling, conduction) within a practical simulation runtime without compromising mechanical accuracy, we employ **Thermal Time Scaling**.

## Thermal Time Scaling (SOTA Approach)
Instead of simulating millions of mechanical steps to observe thermal changes, we artificially accelerate the thermal physics. We scale the thermal parameters so that the heat transfer that *would* occur over the real forging time happens within the shortened simulation time.

### The Problem: Timescale Disparity
*   **Mechanical Timestep ($\Delta t_{mech}$):** Governed by the speed of sound in the material (e.g., steel at $1000^\circ\text{C}$). Typically $\approx 1 \times 10^{-6}$ to $1 \times 10^{-7}$ seconds.
*   **Total Sim Steps:** To simulate 10 seconds of forging at $\Delta t = 10^{-6}$, it requires $10,000,000$ steps. Very expensive.
*   **Thermal Evolution:** Noticeable cooling and heat diffusion take seconds to minutes.

### The Solution: Scaling Parameters
We introduce a `thermal_time_scale` multiplier ($K_{scale}$).

Instead of reducing the number of steps by increasing the mass (Mass Scaling) which can alter dynamic effects, we scale the thermal transfer coefficients:
1.  **Thermal Diffusivity ($\alpha$):** $\alpha' = \alpha \times K_{scale}$
2.  **Air Convection Coefficient ($h_{air}$):** $h_{air}' = h_{air} \times K_{scale}$
3.  **Contact Conductance ($h_{contact}$):** $h_{contact}' = h_{contact} \times K_{scale}$

If $K_{scale} = 1000$, the thermal physics evolves 1000 times faster per mechanical step. The mechanical behavior (yield stress, plasticity, deformation) remains physically accurate *for the current temperature at that step*.

### Advantages
*   **Mechanical Integrity:** No artificial mass or damping is added. The dynamic response (forces, stresses) remains true.
*   **Implementation Simplicity:** Directly scales existing constants before solving the thermal step.
*   **Controllable:** The user sets `thermal_time_scale` directly in the solver options based on the desired physical runtime vs. simulated runtime.

## Mathematical Formulation
The standard heat equation in MPM:
$c_p \rho \frac{\partial T}{\partial t} = \nabla \cdot (k \nabla T) + \dot{q}_{mech} - h_{air}(T - T_{env}) - h_{contact}(T - T_{die})$

With Thermal Time Scaling, we modify the effective terms (or intuitively, we multiply the $dt$ purely for the thermal update equation):
$T_{n+1} = T_n + \left( \alpha \nabla^2 T + \dots \right) (dt \times K_{scale})$

*Note: In our implementation, we directly scale the parameters $\alpha$, $h_{air}$, and $h_{contact}$ passed to the Taichi kernels.*

## Thermal CFL Limitation
Accelerating thermal physics means we take effectively larger thermal timesteps. This is bound by the explicit thermal CFL condition:
$\Delta t_{thermal_{effective}} \le \frac{\Delta x^2}{2d \alpha'}$
where $d$ is spatial dimensions, $\Delta x$ is grid spacing.

If $K_{scale}$ is too large, the scaled diffusivity $\alpha'$ will violate this condition, causing the simulation to explode (nan temperatures).

**Detection:** The `base_mpm_solver` checks this condition during initialization.
If $dt_{mech} > \frac{dx^2}{4 \cdot \alpha \cdot K_{scale}}$, a warning is logged:
`"Thermal CFL violation: The timestep size is too large for the current thermal diffusivity and thermal_time_scale..."`

## Implementation Details
*   **Configuration:** `thermal_time_scale` is exposed in `genesis/options/solvers.py` -> `MPMOptions`.
*   **Application:** Applied in `genesis/engine/solvers/base_mpm_solver.py` during `build` and `_step_thermal`.
*   **Benchmarks:** Verified correctly in `benchmark_thermal_viz_cooling.py` and `benchmark_thermal_viz_diffusion.py`.

## Future Alternatives: Selective Mass Scaling (SMS)
If further performance is needed, **Selective Mass Scaling (SMS)** can be explored.
*   **Concept:** Artificially increase the density (mass) of the material by a factor $f^2$. This lowers the speed of sound $c = \sqrt{E/\rho_{scaled}}$, allowing a larger mechanical timestep $\Delta t' = f \Delta t$.
*   **Trade-off:** Increases inertia. This can cause unrealistic dynamic forces (like bouncing or ringing). For quasi-static processes like slow hydraulic forging, SMS is acceptable. For hammer forging, SMS can corrupt the results.
*   Because our simulator handles dynamic strikes, Thermal Time Scaling is currently preferred over Mass Scaling to preserve impact dynamics.

## Future Alternatives: Implicit Thermal Solver
To overcome the Thermal CFL limit when using very high $K_{scale}$, transitioning from an explicit integration (Forward Euler) to an implicit integration (Backward Euler) for the thermal step is recommended. This requires solving a linear system (e.g., using Conjugate Gradient in Taichi) each step but allows unconditionally stable large thermal timesteps.

---

## Appendix A: Thermal Simulation Debug Scenarios (Rerun Benchmark)
During the creation of the `benchmark_thermal_viz_cooling.py` visualization, several profound numerical instabilities were discovered and isolated. They serve as critical reference points for tuning future Hot Forging scenes:

### 1. The CFL / Stiffness Explosion (1000K to 7000K instant jump)
When using the `JohnsonCookPlasticity` material configured for Steel ($E = 50$ GPa), the speed of sound $c = \sqrt{E/\rho}$ restricts the maximum allowable timestep.
*   **The Bug:** The explicit simulator was running at $dt = 1.12 \times 10^{-5}$ s, which violated the $10^{-6}$ s limit for 50 GPa steel. 
*   **The Symptom:** Visual "freezing" on frame 1, and the cylinders instantly turned pure red (1000K to 7000K+). The numerical instability caused particles to explode outward at 10 m/s. The Johnson-Cook thermal model converted this massive false "plastic work" into adiabatic heat.
*   **The Fix:** Artificially soften the elasticity (e.g., $E = 10$ MPa) for the visualization benchmark to satisfy the CFL condition without mass scaling.

### 2. Poisson Disk Sampling Collisions
When material entities are initialized with `sampler="pbs"` (Poisson Disk Sampling) combined with extreme material stiffness, even microscopic initial particle overlaps resolve as near-infinite restorative forces on step 1, triggering compounding explosions.

### 3. Grid Boundary Padding Collisions
Genesis enforces an invisible 3-cell boundary wall padding around `lower_bound` and `upper_bound`.
*   **The Bug:** The dropped cylinder was spawned with its base exactly at $Z = 0.0$, while the global `lower_bound` was also $Z = 0.0$.
*   **The Symptom:** The cylinder spawned *inside* the invisible padding boundary wall, triggering massive penalty forces.
*   **The Fix:** Always expand the explicit MPM bounds. E.g., setting `lower_bound = (..., ..., -0.03)` ensures the $Z=0$ floor is well outside the padding wall.

### 4. Timescale Illusion in Benchmarks
A typical 400-step explicit simulation at $dt = 1.12 \times 10^{-5}$ represents exactly **4.5 milliseconds** of physical time. To visually observe an object cooling from 1000K to 293K in 400 frames, the thermal conductivities ($k_{air}$ and $k_{contact}$) must be artificially scaled by factors of $10^6$ or higher, heavily exaggerating the convection/conduction speed.

### 5. Rerun Data Serialization Bottleneck
The visualizer originally attempted to stream hundreds of thousands of data points through Python to Rerun on every micro-step.
*   **The Bug:** Sending `rr.log` over a local socket every frame at a high frequency caused the rendering pipeline to stall out completely or repeatedly add continuously trailing particles.
*   **The Fix:** We implemented a `STEPS_PER_RENDER` batched IO system. The simulation physics iterates forward silently 50 steps at a time, and we only extract arrays from the GPU and push to Rerun once per 50 steps, massively accelerating total benchmark time. We also ran Genesis headless (`show_viewer=False`) to avoid dual-rendering overhead.

---

## Appendix B: Johnson-Cook Material Customization
Standard Genesis materials (`ElastoPlastic`) are insufficient for the complex rheology of hot forging. The Johnson-Cook Viscoplastic model decoupled hardening, rate, and thermal softening:
$\sigma_y = (A + B \varepsilon_p^n)(1 + C \ln \dot{\varepsilon}^*)(1 - T^{*m})$
The last term ($1 - T^{*m}$) is the critical Thermal Softening factor. Without it, the press force is grossly overestimated because cold steel strength is mistakenly applied to 1000K hot steel. This necessitated coding the `JohnsonCookPlasticity` model natively in Taichi within the MPM solver loop.

---

## Appendix C: Simulation Stability & Logic Performance Optimizations
Beyond thermal simulation, ensuring the overall software architecture runs fast and doesn't crash during edge cases was a major focus.

### 1. Sim Stability Safety Net (Auto-Reset)
Explicit solvers can occasionally experience "rigid solver instability" (NaN forces) during extreme collision states, such as the strike release phase. 
*   **The Problem:** Previously, a single invalid constraint force calculation generated by a bad particle overlap or extremely high velocity would cause a mathematical `NaN` propagation, crashing the server (`GenesisException`), freezing the teleop UI, and requiring a manual restart.
*   **The Fix:** We implemented a `SafetyOptions` configuration to wrap the entire `step_simulation` in a try-except block that looks for `SimulationStabilityError` or `GenesisException`. When detected, it skips the problematic frame, logs a warning, and can trigger an immediate `reset_simulation()`. This builds a robust safety net that lets the application survive physics blow-ups without stalling the user workflow.

### 2. Optimizing Logic Performance (get_resistance_forces)
During profiling, we discovered immense Python overhead stalling the pipeline before physics were even taking place.
*   **The Problem:** `logic_update_state` and `logic_get_resistance` were consuming large chunks of CPU time per frame due to repeated slow dictionary lookups and data transfers from the GPU/C-backend to Python.
*   **The Fix:** We cached the gripper link indices and other frequently accessed geometry data at the `environment.py` initialization stage. Bypassing these repeated string-based lookups drastically reduced the logic step time and brought the loop closer to our <33ms target.

---

## Appendix D: MJCF Asset Color Parsing
A significant hurdle in the UI visualization pipeline was correctly rendering primitive geometry colors from imported `.xml` and `.urdf` files (like the robotic grippers and the underlying press). 
*   **The Problem:** The Genesis physics engine was incorrectly ignoring default colors assigned to raw box primitives configured in MJCF/URDF structures, reverting them to generic gray.
*   **The Fix:** We overrode the asset parsing logic deep within the engine to specifically cascade defined material `<rgba>` color channels onto geometries to ensure high-visibility models during visual demonstration.

---

## Appendix E: MPM Solver Tuning for High Plasticity
In addition to the thermal timescale and material customizations, the MPM solver itself requires tuning to accurately handle the extreme plastic deformations inherent in hot forging.

### 1. Transfer Algorithms (APIC vs. FLIP/PIC)
Standard MPM uses Particle-In-Cell (PIC) or Fluid-Implicit-Particle (FLIP) transfers between the grid and particles. 
*   **The Issue:** PIC causes excessive numerical dissipation (damping), while FLIP produces unphysical "ringing" (noise) in velocity fields during the extreme compression of forging.
*   **The Solution:** The Affine-Particle-In-Cell (APIC) or CPIC formulation is mandatory, as it preserves angular momentum without artificial noise, ensuring the metal billet doesn't artificially stiffen or numerically fracture when flattened.

### 2. Grid Density Management
During a high-ratio forging stroke (e.g., squishing a 10cm billet down to 1cm), existing particles become extremely compacted in the $Z$ axis while tearing outward in the $X/Y$ axes. We rely on the Deformation Gradient Tensor ($F$) and careful tuning of the initial `particle_radius` to ensure particles map to grid cells continuously, preventing "holes" (vacuum gaps) from forming internally inside the metal continuum during the stroke.

### 3. Simulation Energy Balance
When tuning the parameters above, visual results are not enough. Stability is strictly verified via Energy Balance Analysis:
We monitor the total energy ($E_{total} = E_{kinetic} + E_{internal} + E_{thermal} - W_{external}$). If $E_{total}$ grows unnaturally step-over-step, it signifies that solver error (from CFL violations or bad APIC transfers) is numerically injecting artificial energy, demanding a drop in $\Delta t$ or an increase in grid resolution.
