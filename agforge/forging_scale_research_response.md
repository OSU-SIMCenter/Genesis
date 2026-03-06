To address the timing mismatch between high-speed mechanical impacts and slow-moving thermal dynamics in your Genesis Material Point Method (MPM) engine, you must implement techniques that bridge the microscopic mechanical timescale with macroscopic thermal diffusion. State-of-the-art (SOTA) approaches in explicit dynamic thermo-mechanical simulations rely on **Time Scaling (Thermal Fast-Forwarding)**, **Mass Scaling**, and **Staggered Time Integration** to artificially accelerate thermal behavior or increase the allowable computational timestep without violating physical laws. [lsdyna.ansys](https://lsdyna.ansys.com/wp-content/uploads/2025/02/C-II-04-1.pdf)

Here are the optimal, scientifically validated methods to handle these issues in your engine.

## Thermal Fast-Forwarding (Time Scaling)
Time scaling is the standard industry approach for hot forging simulations where mechanical deformation completes before thermal diffusion naturally occurs. The goal is to mathematically compress real-world time into the simulation's millisecond window by inflating thermal fluxes while maintaining thermodynamic equivalence. [lsdyna.ansys](https://lsdyna.ansys.com/wp-content/uploads/2025/02/C-II-04-1.pdf)

To achieve this without breaking spatial temperature gradients, you must keep dimensionless groups like the **Fourier number** (\(Fo\)) and the **Biot number** (\(Bi\)) constant. Assuming you apply a time scaling factor \(\eta\) (e.g., \(\eta = 1000\), making 1 simulated millisecond equal to 1 real-world second): [theronguo](https://theronguo.de/files/preprint1.pdf)
- **Thermal Diffusivity (\(\alpha\)):** Must be scaled up by \(\eta\). Because \(Fo = \frac{\alpha \cdot t}{L^2}\), scaling \(\alpha\) ensures the exact same amount of heat diffuses spatially across the grid during the artificially short simulation time. [theronguo](https://theronguo.de/files/preprint1.pdf)
- **Convection and Contact (\(h_{air}\), \(h_{contact}\)):** Must also be scaled up by \(\eta\) to keep the Biot number (\(Bi = \frac{h \cdot L}{k}\)) constant. This guarantees that surface heat loss accelerates at the exact same rate as internal diffusion. [lsdyna.ansys](https://lsdyna.ansys.com/wp-content/uploads/2025/02/C-II-04-1.pdf)
- **Specific Heat Capacity (\(C_p\)):** Do not scale this value. In your Johnson-Cook implementation, adiabatic heating (\(\Delta T = \frac{W_p}{\rho \cdot C_p}\)) generates instantaneous heat from mechanical plastic work. Because the mechanical energy (\(W_p\)) is not scaled, altering \(C_p\) would artificially distort the temperature spikes caused by the hammer impact. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/86997361/fa0f7d79-8137-4218-a746-ef6316c622f2/forging_scale_research.md)
- **Johnson-Cook Softening (\(jc\_m\)):** Do not scale this value. It is a time-independent material exponent that strictly relies on instantaneous nodal temperature, which is inherently preserved by the scaling of \(\alpha\) and \(h\). [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/86997361/fa0f7d79-8137-4218-a746-ef6316c622f2/forging_scale_research.md)

## Mass Scaling Techniques
If you want to accelerate the simulation by increasing the explicit Courant-Friedrichs-Lewy (CFL) mechanical timestep limit, you must lower the acoustic wave speed (\(c = \sqrt{E/\rho}\)) by altering mass. [arxiv](http://arxiv.org/pdf/2410.23816.pdf)
- **Conventional Mass Scaling:** This involves artificially increasing the material density (\(\rho\)) to allow for a larger \(\Delta t\). However, in violent forging impacts, adding raw mass drastically increases the kinetic energy of the billet and hammer, which can distort dynamic impact forces and ruin your MPM rigid-body collision geometry. [arxiv](http://arxiv.org/pdf/2410.23816.pdf)
- **Selective Mass Scaling (SMS):** This is the SOTA method used in advanced explicit codes and newer MPM frameworks. SMS isolates and adds artificial mass only to the high-frequency vibrational modes of the grid (which dictate the CFL limit) while leaving the low-frequency global translational and rotational inertia unaffected. This safely allows you to increase \(\Delta t\) by orders of magnitude without altering the macro-dynamics of the hammer strike. [linkedin](https://www.linkedin.com/posts/ioannis-koutromanos-28105926_my-paper-multigrid-inspired-selective-mass-activity-7389741418681974784-SjmK)

## Staggered Asynchronous Time Integration
Instead of forcing the mechanical and thermal solvers to share the same micro-timestep, you can decouple their mathematical clocks using a fractional-step or staggered integration scheme. [onlinelibrary.wiley](https://onlinelibrary.wiley.com/doi/10.1002/nag.3794)
- **Sub-cycling:** Run your explicit mechanical Taichi solver for \(N\) micro-steps (e.g., \(dt = 10^{-5}\) seconds). Then, pause the mechanical solver and execute a single thermal step using a macroscopic elapsed time (\(\Delta t_{therm} = N \times 10^{-5}\)). [academiccommons.columbia](https://academiccommons.columbia.edu/doi/10.7916/2c7g-3c72/download)
- **Semi-Implicit Thermal Solver:** While your mechanical MPM remains explicit to handle the violent hammer impacts, SOTA coupled frameworks upgrade the thermal Laplacian diffusion operator to a fully implicit solver. Implicit solvers are unconditionally stable, allowing the thermal diffusion to safely jump forward in massive time increments without mathematically exploding or violating thermal CFL limits (\(\Delta t_{max} = \frac{dx^2}{6\alpha}\)). [linkinghub.elsevier](https://linkinghub.elsevier.com/retrieve/pii/S0266352X24002854)

### Implementation Recommendation
For immediate integration into the Genesis Taichi codebase, **Thermal Parameter Scaling (Time Scaling)** is the easiest and most performant to implement because it requires zero architectural changes to your nested `mpm_grid_op` GPU loops. You simply multiply `_alpha_thermal`, `_h_air`, and `_h_contact` by a global `thermal_time_scale` hyperparameter at initialization. If simulation speed remains a bottleneck, transitioning your thermal diffusion to an implicit solver outside the explicit substep loop is the most scientifically rigorous upgrade. [onlinelibrary.wiley](https://onlinelibrary.wiley.com/doi/10.1002/nag.3794)

# Time-Scaling Thermo-Mechanical Hot Forging Simulations in Explicit MPM

## Executive Summary

Hot forging simulations with explicit time integration suffer from a severe mismatch between the fast mechanical timescale of hammer impacts (microseconds–milliseconds) and the slow thermal timescale of conduction, convection, and radiation (seconds–minutes). This report surveys state-of-the-art (SOTA) methods for reconciling these scales and outlines how they can be adapted to an explicit MPM (Material Point Method) solver like Genesis.

Industrial forming simulations commonly combine three ideas: (1) **dimensionless-similarity-based time scaling** of the thermal problem while preserving Fourier and Biot numbers, (2) **mass and time scaling** of the mechanical problem, ideally via selective mass scaling, and (3) **phase splitting and multi-rate coupling** between forming and cooling stages. In conjugate heat transfer and thermal analysis, related work uses **Bi–Fo time scaling** and **effective heat capacity scaling** to accelerate thermal transients without changing dimensionless groups, providing a principled mathematical basis for accelerated thermal clocks.[1][2][3]

For an explicit MPM forging simulation, the most appropriate SOTA-aligned approach is:

- Treat the mechanical time step as fixed by the CFL limit and **introduce a global thermal time-acceleration factor** \(S_T\) that maps simulation time to real thermal time.
- **Scale thermal diffusivity and all heat transfer coefficients** (air and contact) by \(S_T\) so that Fourier and Biot numbers match those of the real process at the mapped physical time.
- Keep the mechanical response and plastic work generation unchanged, and maintain conservative diffusion operators so that total thermal energy is preserved despite accelerated transport.
- Respect the explicit diffusion CFL limit \(\Delta t \le dx^2 / (6\alpha_{\text{scaled}})\) when choosing \(S_T\), and validate against reference FE or experimental forging data to calibrate \(S_T\) and material parameters (including Johnson–Cook thermal softening).
- Optionally, split the process into **impact (forming) and dwell/cooling phases**, using more aggressive time scaling and/or larger thermal-only time steps during dwell, as is standard in LS-DYNA hot forming workflows.[2]

This combination yields physically interpretable, energy-conserving, and numerically stable accelerated thermal behavior that matches established practice in hot forming simulation while remaining compatible with Genesis’s GPU MPM architecture.

## 1. Problem Context and Constraints

Hot forging processes involve short, violent mechanical events (hammer strokes or press closures) superposed on long thermal histories (heating in the furnace, transfer, tool contact, and inter-stroke cooling). In explicit solvers, the mechanical time step is constrained by a CFL condition based on the smallest spatial scale and elastic wave speed, leading to time steps \(\mathcal{O}(10^{-6}–10^{-5})\) s or smaller for high-speed impact and fine discretizations. At such time steps, advancing simulation for the tens of seconds relevant for thermal evolution is computationally prohibitive.[4][5]

In industrial finite element hot forming codes, e.g., LS-DYNA, the mechanical time step is often the bottleneck, and simulations apply **time and mass scaling** to make explicit analyses tractable while remaining quasi-static. Thermal subproblems are typically much less restrictive; for steel, theoretical limits on thermal time steps derived from diffusivity and mesh size are orders of magnitude larger than explicit mechanical limits.[3][2]

The Genesis MPM setup has similar constraints but adds GPU-accelerated kernels and fully coupled thermo-mechanical MPM–rigid interactions. The research questions are therefore:

1. How to speed up thermal evolution (diffusion, convection, conduction to tools) relative to the mechanical time integration while preserving core physics.
2. How to avoid violating explicit diffusion stability limits when scaling thermal parameters.
3. How to keep the scheme efficient inside tight GPU kernels (no expensive global solves each substep).

## 2. Dimensionless Similarity and Time Scaling in Hot Forming

### 2.1 Fourier and Biot Numbers as Invariants

The key non-dimensional groups for transient heat transfer in forming are the **Fourier number** and **Biot number**:

\[
Fo = \frac{\alpha t}{L^2}, \quad Bi = \frac{h L}{\lambda},
\]

where \(\alpha = \lambda / (\rho C_p)\) is thermal diffusivity, \(t\) is time, \(L\) a characteristic length, \(h\) a heat transfer coefficient (air or contact), and \(\lambda\) the thermal conductivity. If \(Fo\) and \(Bi\) are kept identical between a reference (real) process and an accelerated (simulated) one, the temperature fields as functions of non-dimensional time \(t^*\) should match up to numerical discretization errors.[2][3]

In LS-DYNA hot forming simulations, Lorenz and Haufe explicitly exploit this: when **time scaling** the forming stage by increasing tool velocity, they preserve Fourier and Biot numbers by scaling all thermal conductivities and all heat transfer coefficients by the same factor as the velocity/time scaling. This keeps the thermal evolution in dimensionless form consistent with the real process even though the clock is accelerated.[2]

### 2.2 Bi–Fo Time Scaling in Transient Conjugate Heat Transfer

In transient conjugate heat transfer for gas turbine vanes, a **Bi–Fo time scaling method** is used to reconcile disparate timescales between fluid flow and solid conduction. The approach:[1]

- Applies a similarity transformation to the solid’s heat conduction equation, scaling the product \(\rho C_p\) (thermal capacitance per unit volume) so that the same Fourier number is achieved in less physical time.
- Exploits the fact that, for flows that re-stabilize quickly after perturbations, the **fluid time** can be compressed to match the accelerated solid conduction time without significantly changing the effective thermal boundary condition seen by the solid.

Validation on engine components shows that this Bi–Fo method can greatly reduce CPU cost while maintaining accurate transient temperature histories, as long as the flow re-stabilization assumption holds.[1]

### 2.3 Effective Heat Capacity Scaling for Faster Thermal Steady State

Machining and tool-wear simulations often seek thermal steady state rather than full transient fidelity. A study on accelerating thermal steady state in cutting tools proposes reducing the **effective volumetric heat capacity** (\(\rho C_p\)) of the tool material to reduce its thermal inertia, thereby shortening the time constant of the system while preserving steady-state temperature distributions. This leverages the relationship between characteristic thermal time constant and capacitance/conductance discussed in classical numerical heat transfer texts.[6][3]

In finite difference and finite element formulations, the stable and accurate time step for explicit thermal integration is tied to nodal thermal capacitance and conductance, and the physical transient response time is proportional to \(\rho C_p L^2 / \lambda\). Scaling \(\rho C_p\) thus directly scales the apparent time constant for thermal response.[3]

## 3. Industrial Practice: Hot Forming with LS-DYNA

### 3.1 Splitting Forming and Cooling Stages

Industrial hot forming simulations frequently split the process into two stages: **forming** (including contact with relatively cold tools and rapid local cooling) and **cooling/quenching** (long dwell in tools and subsequent cooling cycles). The forming stage is typically modeled with explicit dynamics for the mechanics and an implicit or explicit thermal solver, while the cooling stage uses a thermal-only or thermo-mechanical implicit analysis.[2]

Lorenz and Haufe describe a workflow where the forming stage uses a detailed shell representation of tool surfaces with thick thermal shells, while the cooling stage introduces a 3D volume mesh of the tools to capture heat dissipation into the dies and cooling channels. Temperatures computed during forming are used as initial conditions for the cooling simulation, which may be repeated over multiple production cycles to capture tool thermal stabilization.[2]

### 3.2 Time and Mass Scaling in Explicit Forming

Because the explicit mechanical time step is often several orders of magnitude smaller than the thermal time step limit, LS-DYNA applies **time scaling** (via increased tool velocities) and **mass scaling** (increased density in selected elements) to speed up forming analyses while keeping the process quasi-static. This is standard in metal forming simulations when strain-rate sensitivity is weak.[5][2]

For hot forming, however, material behavior is **strain-rate and temperature dependent**, so time scaling must be coordinated with the material model, and thermal coupling must preserve Fourier and Biot numbers. Lorenz and Haufe recommend:[2]

- Increasing tool velocity by a factor \(S_t\) relative to the real process.
- Scaling all thermal conductivities \(\lambda\) and heat transfer coefficients \(h\) by \(S_t\) to keep \(Fo\) and \(Bi\) unchanged.
- Carefully limiting mass scaling (or using selective mass scaling) to avoid non-physical inertial effects, particularly in unconstrained regions of the hot blank where deformation is sensitive to inertia.[2]

They show that **selective mass scaling** can achieve an order-of-magnitude increase in time step without noticeable accuracy loss compared to conventional mass scaling, and with lower CPU cost given fewer thermal substeps.[2]

### 3.3 Implications for MPM

Genesis’s explicit MPM solver is conceptually similar to LS-DYNA’s explicit structural solver in its CFL-limited mechanical time step. Adapting LS-DYNA practice suggests:

- When tool kinematics are under user control (e.g., virtual press velocity), impact events can be sped up by increasing tool velocities and, if the material model allows, reinterpreting strain-rate terms in Johnson–Cook to account for the artificial rate.[2]
- Simultaneously, thermal conductivities and heat transfer coefficients should be scaled with the time scaling factor to preserve non-dimensional thermal behavior.
- However, for violent impacts and learning-driven applications, there may be a preference to keep **mechanical trajectories close to real-time kinematics** and instead accelerate only the thermal clock. This motivates a thermal-only scaling strategy described next.

## 4. Multi-Rate and Subcycling Schemes

### 4.1 General Multi-Physics Subcycling

In coupled fluid–thermal–structural problems, partitioned schemes with **different time steps for each physics** are widely used. Miller and McNamara develop a time-marching framework where the fluid solver runs at a small CFD timestep, while structural and thermal solvers take larger subcycled steps, using predictor–corrector coupling to maintain second-order time accuracy. This yields 2–4× speedups while retaining accuracy comparable to strongly coupled reference solutions.[7]

Similarly, explicit–explicit subcycling methods in structural dynamics allow groups of elements with different stability limits to advance with **non-integer time step ratios**, reducing the number of updates for less stiff regions or physics while maintaining stability and acceptable accuracy.[8]

### 4.2 Accelerated Time-Scale Decomposition

Recent work on accelerated time-scale decomposition methods for transient multi-medium problems introduces schemes that explicitly separate fast and slow dynamics, allowing coarse-grained advancement of slow phenomena while resolving fast ones only where needed. In thermo-mechanical contexts, this motivates splitting phases or regions by characteristic time scale and integrating them with different effective clocks.[9]

### 4.3 Applicability to Genesis MPM

For Genesis, which already operates with an extremely small mechanical substep, classical subcycling does **not** directly solve the problem of limited total simulated physical time: subcycling reduces cost per unit simulated time, but the number of mechanical steps still grows linearly with the physical duration of interest. However, subcycling ideas are still useful in two ways:

- **During impact**, the mechanics dictate the time step and number of steps; thermal substepping can be coarse while still capturing coupling, since conduction and convection are slower even in real time.
- **After impact**, when mechanics become quasi-static or frozen, the system can transition to a **thermal-only mode** with much larger time steps (subject to thermal CFL), effectively implementing a multi-rate scheme over the full forging cycle.

This is conceptually aligned with LS-DYNA’s forming vs. cooling stage split and can be emulated in MPM by switching kernels or modes between impact and dwell phases.[2]

## 5. Recommended Thermal Time-Scaling Strategy for Explicit MPM

### 5.1 Define a Thermal Time Acceleration Factor

Introduce a scalar **thermal acceleration factor** \(S_T\) that maps simulation time \(t_{sim}\) to an equivalent **thermal physical time** \(t_{th,phys}\):

\[
t_{th,phys} = S_T\, t_{sim}.
\]

This expresses that for every unit of simulated mechanical time, the thermal field is intended to evolve as if more real time had passed. The target \(S_T\) should be chosen based on the mismatch between mechanical and thermal timescales for the billet size and process: for example, if forming occurs over 10 ms but relevant cooling spans 10 s, \(S_T \approx 1000\) would align the two orders of magnitude.

### 5.2 Scale Diffusivity and Heat Transfer Coefficients

To respect Fourier and Biot similarity, scale the following thermal parameters:

- **Thermal diffusivity**: \(\alpha_{sim} = S_T \alpha_{phys}\).
- **Thermal conductivity** (if explicitly represented): \(\lambda_{sim} = S_T \lambda_{phys}\) while keeping \(\rho C_p\) unchanged, or equivalently using \(\alpha_{sim}\) directly.[2]
- **Convective heat transfer coefficients**: \(h_{air,sim} = S_T h_{air,phys}\).[2]
- **Contact conductance/heat transfer coefficients** to rigid tools: \(h_{contact,sim} = S_T h_{contact,phys}\).[2]

With these scalings, the **simulated Fourier number** over a simulation time interval \(\Delta t_{sim}\) becomes

\[
Fo_{sim} = \frac{\alpha_{sim} \Delta t_{sim}}{L^2} = \frac{S_T \alpha_{phys} \Delta t_{sim}}{L^2} = Fo_{phys}\big|_{t = S_T \Delta t_{sim}},
\]

and similarly \(Bi_{sim} = Bi_{phys}\), ensuring the same non-dimensional thermal response as the real material at a longer physical time.[1][2]

Crucially, the Genesis MPM implementation already uses a **mass-conservative volume-fraction Laplacian** for internal diffusion, so scaling \(\alpha\) only accelerates energy redistribution without creating or destroying energy, provided boundary fluxes are scaled consistently.[10]

### 5.3 Preserve Mechanical–Thermal Energy Coupling

Genesis converts plastic work into temperature via

\[
\Delta T = \frac{f W_p}{\rho C_p},
\]

with \(f\) the fraction of plastic work converted to heat. To keep the **local adiabatic heating relationship physically correct**, \(\rho C_p\) should remain at its physical value in this term. Under the proposed scaling:[10]

- Plastic work per unit volume \(W_p\) and density \(\rho\) are unchanged.
- The effective \(C_p\) used for adiabatic heating remains \(C_{p,phys}\), so the instantaneous temperature rise from a given local plastic strain increment is physical.

Therefore, adiabatic heating is unaffected; only the subsequent spatial transport of that heat is accelerated.

The alternative strategy of scaling \(\rho C_p\) (as in Bi–Fo and effective heat capacity methods) changes thermal inertia and can be justified when accurate transient temperature history is less critical than steady-state or macro-scale behavior, but it introduces more complexity into the interpretation of Johnson–Cook thermal softening and may require re-fitting material parameters.[6][1]

### 5.4 Check Thermal CFL and Stability Limits

Explicit diffusion schemes in finite differences and finite elements are stable if the time step satisfies a bound of the form

\[
\Delta t \le \frac{dx^2}{C_d \alpha},
\]

with \(C_d\) depending on dimensionality (e.g., \(C_d = 2\) in 1D, \(\approx 6\) in 3D). Genesis already issues a warning when \(\Delta t\) exceeds \(dx^2 / (6\alpha)\) for the configured diffusivity.[3][10]

Scaling \(\alpha \rightarrow S_T \alpha\) therefore tightens the thermal CFL bound to

\[
\Delta t \le \frac{dx^2}{6 S_T \alpha_{phys}}.
\]

Given that the mechanical CFL limit usually already enforces very small \(\Delta t\), there is practical headroom to increase \(S_T\) substantially before the thermal CFL becomes active. In practice, one should:[3][2]

1. Compute both mechanical and thermal CFL limits for baseline parameters.
2. Choose \(S_T\) so that the **thermal CFL remains above the actual substep \(\Delta t\)** by a safety factor (e.g., \(\ge 5\)).
3. Add run-time checks that warn or clamp \(S_T\) if the thermal CFL would be violated.

### 5.5 Tuning Genesis Thermal Parameters

Mapping the above strategy to Genesis’s configuration fields suggests:

- `default_thermal_diffusivity` \(\alpha\): set to \(S_T\) times the physical diffusivity of the workpiece material.
- `thermal_air_conductivity` (\(h_{air}\)): set to \(S_T h_{air,phys}\).
- `thermal_contact_conductivity` (\(h_{contact}\)): set to \(S_T h_{contact,phys}\).
- `default_heat_capacity` (\(C_p\)): keep at the physical value initially; adjust only if later calibration shows systematic discrepancies that suggest effective heat capacity scaling is needed.
- `jc_m` (Johnson–Cook thermal softening exponent): treat as a **calibration parameter** to match experimental flow curves under the accelerated thermal evolution; if the accelerated thermal clock changes the perceived temperature–strain path, \(jc_m\) and possibly reference temperatures may need modest adjustment based on comparison to reference FE or experimental data.[11][12]

A practical calibration loop would be:

1. Select a representative forging test (e.g., cylinder compression or a simple preform) with measured temperature and force data over time.[13][14]
2. Run a high-fidelity FE simulation (e.g., with DEFORM or LS-DYNA) as a ground truth for thermo-mechanical response under real-time conditions.[12][11]
3. Choose an initial \(S_T\) based on the ratio of real thermal to mechanical timescales and scale \(\alpha, h_{air}, h_{contact}\) accordingly.
4. Tune \(jc_m\) and, if necessary, \(C_p\) and the plastic-to-heat conversion fraction to match temperature and force histories within acceptable error bounds.

## 6. Phase-Splitting: Forming vs. Dwell/Cooling

### 6.1 Rationale and SOTA Practice

Splitting the process into a **highly dynamic forming stage** and a **quasi-static dwell/cooling stage** is standard practice in hot forming simulations. During forming, accurate representation of contact, inertia, and strain localization is paramount. During dwell and subsequent cooling cycles, mechanics are largely static, and the interest shifts to heat transfer, phase transformations, and residual stresses.[15][2]

LS-DYNA implements this via separate **forming** and **cooling** simulations, with thermal fields handed off between them and different solver settings (explicit vs. implicit, with or without mechanical coupling) used for each stage.[2]

### 6.2 Adapting to Genesis

In Genesis MPM, a similar split can be implemented conceptually:

1. **Forming stage**
   - Run full thermo-mechanical MPM with adiabatic heating and accelerated thermal parameters (\(S_T\) as above) during the period of active contact and significant deformation.
   - Use mechanical CFL-limited \(\Delta t\) and keep \(S_T\) conservative to avoid any thermal CFL issues.

2. **Dwell/cooling stage**
   - Once deformation has essentially ceased (e.g., when hammer velocity is zero and particle velocities drop below a threshold), freeze the mechanical state.
   - Switch to a **thermal-only or weakly-coupled mode** where particle positions and stresses remain fixed, and only the temperature field is updated using the diffusion, convection, and contact conduction operators.
   - In this stage, much larger \(\Delta t_{th}\) can be used, bounded only by the thermal CFL and desired temporal resolution.[3][2]
   - Optionally, further increase \(S_T\) during dwell to simulate seconds of cooling in a small number of steps, especially if only final temperature distribution matters.

3. **Inter-stroke heating/cooling**
   - For multi-stroke forging, inter-stroke furnace heating or air cooling can be modeled with purely thermal steps between mechanical impacts, again using large \(\Delta t_{th}\) and possibly higher \(S_T\).

This phase-splitting approach keeps the MPM mechanics solver focused on the phases where it is essential while still capturing realistic thermal histories over long effective times.

## 7. Alternatives and Trade-Offs

### 7.1 Mass Scaling and Selective Mass Scaling

Mass scaling, especially **selective mass scaling**, is widely used to increase explicit time steps in forming simulations without overly distorting dynamic response. It works by artificially increasing density in selected elements to raise the CFL limit, preferably targeting high-frequency modes while leaving low-frequency (physically relevant) modes relatively unchanged.[4][2]

In MPM, mass is stored at particles, and altering it would affect inertia and momentum balance more directly than in some FE discretizations. While a selective mass scaling analog is conceivable (e.g., increasing mass only in regions far from the region of interest or in less critical materials), this introduces risk of non-physical inertial effects during rapid impact, which are exactly the situations Genesis aims to model accurately.

Given the strong focus on learning-based control and potentially rate-dependent forging behavior, **thermal time scaling without mechanical mass scaling** is generally safer for Genesis, with mass scaling reserved, if at all, for quasi-static parameter studies validated against reference solutions.

### 7.2 Effective Heat Capacity vs. Diffusivity Scaling

As noted earlier, some methods accelerate transients by reducing **effective \(\rho C_p\)** rather than increasing \(\alpha\) and \(h\). The advantages are:[6][1]

- Thermal CFL limits may improve (larger allowed time step) because nodal capacitance decreases.[3]
- Steady-state temperature distributions are unchanged if conductances and boundary conditions are fixed.[6]

However, for a fully coupled thermo-mechanical MPM with adiabatic heating and temperature-dependent yield, this strategy has drawbacks:

- Local temperature rises from plastic work would be artificially amplified (since \(\Delta T \propto 1/C_p\)), requiring re-scaling of the plastic-to-heat conversion fraction or reinterpretation of Johnson–Cook parameters.
- The mapping between simulation time and physical time becomes more opaque because both diffusion and local heating dynamics are altered.

Therefore, effective heat capacity scaling is best viewed as a **secondary calibration lever** after diffusivity and heat transfer scaling, and it should be used only with careful comparison to experiments or high-fidelity simulations.

### 7.3 Subcycling Inside a Single MPM Step

Another theoretical alternative is to perform multiple thermal substeps per mechanical substep (or vice versa). However, given that mechanics already dictates a very small \(\Delta t\), further subdividing it for thermal updates would increase cost without meaningful accuracy gains, since thermal processes are slower than mechanical wave propagation and plastic flow in the regimes of interest.[7][3]

Conversely, under-resolving thermal dynamics within a mechanical step is acceptable as long as Fourier and Biot similarity over the entire step are maintained and the explicit diffusion scheme remains stable. This further supports the strategy of **simple parameter scaling with one thermal update per mechanical step** rather than complex intra-step subcycling.

## 8. Summary of Recommendations for Genesis

Based on the surveyed literature and industrial practice, the following SOTA-consistent approach is recommended for handling thermal–mechanical timescale mismatch in Genesis’s hot forging simulations:

1. **Introduce a global thermal acceleration factor \(S_T\)** that maps simulation time to physical thermal time.
2. **Scale thermal diffusivity and all heat transfer coefficients** (air and contact) by \(S_T\) to preserve Fourier and Biot similarity relative to the real process at the mapped time.[1][2]
3. **Keep \(\rho C_p\) physical** in adiabatic heating and only adjust it as a secondary calibration parameter if needed, aware of its impact on local temperature rise and Johnson–Cook thermal softening.
4. **Respect explicit diffusion CFL limits** when choosing \(S_T\), and integrate checks that warn if the thermal CFL becomes more restrictive than the mechanical CFL.[3][2]
5. **Split simulations into forming and dwell/cooling phases** where appropriate, using full thermo-mechanical MPM with conservative \(S_T\) during forming and thermal-only or weakly coupled steps with larger \(S_T\) and \(\Delta t\) during dwell.[2]
6. **Calibrate against experimental or reference FE data** for representative forging operations to tune \(S_T\), \(\alpha\), \(h_{air}\), \(h_{contact}\), and Johnson–Cook parameters, especially the thermal softening exponent.[13][11][12]

This framework directly leverages established methods in LS-DYNA hot forming, Bi–Fo time scaling, and accelerated thermal analysis, while aligning with Genesis’s existing conservative MPM diffusion implementation and GPU execution model.