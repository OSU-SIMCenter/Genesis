# Architectural Report: Stabilizing Hybrid Geometry/Particle APIC Contact

## Context & Baseline
We are operating a Hot Forging physics simulation reliant on **APIC (Affine Particle-In-Cell) MPM** with an Elasto-Plastic (Von Mises/Neo-Hookean) material model. The simulation is built on Genesis (a Taichi-based physics engine). 

**The Goal:** Eliminate the standard Grid MPM "Sticky/Spongy" collision artifacts by implementing a Hard Particle/Geometry Collision pass. We want to support "Hybrid Contact" (Grid + Particle).

**The Implementation:** We created an `apply_particle_collisions` kernel that runs *after* the `G2P` transfer. It evaluates distance to the SDF, and if penetrating:
1. Translates position out of the object (`pos = pos - (dist - margin) * normal`)
2. Overwrites the particle velocity array incorporating rigid body velocity, elastic restitution, and friction (`vel = vel_rigid + rvel_tan + rvel_normal`).

## The Instability: Autonomous APIC C-Matrix Explosion (Mathematical Severing)
While this stops boundary penetration, it introduces a catastrophic APIC instability. During collisions, the Affine Velocity Gradient (`C`) matrix artificially skyrockets from $~100$ to **`68,000+`** autonomously. 
Because $F_{tmp} = (I + dt \cdot C) \cdot F$, when the steel cools and hardens (losing its plastic yield envelope), this massive false shear instantly evaluates to millions of Pascals of elastic stress, blasting particles away at $100+$ m/s.

### The Mechanics of the Flaw (APIC Severing)
The instability is driven by naively overwriting the `vel` array without maintaining the mathematical coupling to the `C` array.

1. **APIC Coupling:** In APIC, `vel` and `C` are not independent; they represent the local velocity field of the particle: $v_p(x) = v_p + C_p (x - x_p)$.
2. **The Severing:** When a particle hits the anvil, our custom kernel abruptly zeros the `vel` parameter to prevent penetration. **It does not touch $C$.**
3. **The Phantom Force:** In the *next frame's* P2G transfer, the particle dumps momentum into the surrounding grid nodes via `mass * vel + affine_C_force`. Because `vel` is clamped but `C` still holds the steep inherited slope from the previous hammer strike, the particle essentially imparts a "phantom" affine shockwave. It mathematically forces neighboring grid nodes to accelerate to e.g. $10$ m/s, strictly because of the unconstrained `C` slope.
4. **The Spiral:** In the following G2P, the particle reads this $10$ m/s grid acceleration, computes an even steeper $C$-matrix slope ($C=10,000$). The particle drops to the anvil, `vel` is clamped again, and $C$ grows. It spirals to infinity perfectly autonomously.

## Consultation Request for SOTA Fixes
Based on this architectural breakdown, we need standard APIC best-practice recommendations for correctly clamping velocity gradients during local rigid geometric projection.

**Questions for the Expert:**
1. What is the standard literature/SOTA methodology for constraining the $C$ matrix in APIC when forcibly modifying a particle's $vel$ for geometric non-penetration?
2. If restricting $C$ is mathematically complex, is it standard practice to temporarily downgrade a boundary particle to pure PIC (by setting $C = 0$) when undergoing an abrupt inelastic collision constraint?
3. Should the geometric SDF projection occur inside the grid-level `G2P` solver (by directly modulating the interpolation weights or `grid_vel`), rather than as an isolated particle pass?
4. Are there any known implementations of sticky-free APIC contact that cleanly decouple normal forces (particle-level) and friction forces (grid-level)?
