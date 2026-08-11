> ## ⚠️ READ BEFORE USING ANY NUMBER IN THIS DOCUMENT
>
> This document was written **2026-08-04, before the MPM-domain fix of 2026-08-06.**
>
> The simulation domain was too short: the billet hit the `-x` wall and **froze at 77.8 mm from
> hit 13 onward** while the real part grew to 93.4 mm. Every late-hit quantity measured before the
> fix was therefore partly measuring a wall. Fixing it cut late-hit surface error by 46% and
> **reversed the grid-vs-production stability ranking.**
>
> ⇒ **Treat every number here as DOMAIN-LIMITED.** In particular the volume/packing figures (the
> "~12%" loss and the det-F GAP) have **never been recomputed post-fix**.
>
> The current reference for contact-method work — verified code walkthrough, literature findings,
> retracted claims, and the staged implementation plan — is:
> **[`Contact_Method_Research_And_Plan.md`](./Contact_Method_Research_And_Plan.md)**
>
> This document is retained for its mechanism analysis, which remains useful.

# Volume Conservation and Rigid Contact in the Forging MPM

## Overview

The forging simulation conserves volume according to its own bookkeeping and loses roughly
12% of it according to the particles' actual arrangement. Both statements are true, and the
gap between them is the subject of this document.

This is a companion to `MPM_Stabilization_Architecture.md`. That document covers *whether the
simulation runs*; this one covers *whether the material it produces is physical*. They turned
out to be governed by different mechanisms, so they are kept separate.

**Status: characterised, not solved.** Everything below is measured on the 17-hit real-hit
replay. The remedies in "Future Work" are surveyed but unimplemented.

## The Measurement

Three ways of asking "did the bar lose volume", all on the same particle data:

| method | what it measures | res 7 | res 10 | res 14 |
|---|---|---|---|---|
| `mean(det F)` bookkeeping | what the solver believes | −1.64% | −2.95% | −2.90%\* |
| union-of-balls packing `η` | what the particles actually occupy | **−11.66%** | **−12.00%** | **−10.78%** |
| gap | material overlapping itself | 10.0 pp | 9.1 pp | 7.9 pp |

\* res 14 `det F` not measured; assumed.

The packing measure `η = V_union / (N · v_ball)` models each particle as a ball of radius
`psize/2`. At initialisation the lattice spacing is exactly `psize`, so neighbouring balls
precisely touch and `η ≈ 1`; it can fall *only* by particles interpenetrating.

Three properties make this the metric of record for the project:

1. **Parameter-free.** The one radius involved is the particle's own half-size, not a choice.
2. **Resolution-independent.** ±1.2 pp across an 8× particle-count range.
3. **Well conditioned.** Run-to-run scatter is **±0.04 pp at hit 10 (res 7)**, degrading to
   **±0.2-0.6 pp by hit 17 (res 10)** as nondeterminism accumulates. Still small against a
   ~12 pp effect, but quote the late-sequence figure when sizing an experiment -- an early
   measurement of +-0.04 does not license claiming that precision at the final hit.

Validation: summing pairwise sphere-intersection volumes gives 5,892 mm³ against a Monte-Carlo
union deficit of 5,273 mm³ at res 10 hit 17 — the ~12% excess is the expected triple-overlap
double-count of inclusion–exclusion.

Uniform compression does **not** explain it. At `mean(J) = 0.97`, uniform compaction of a
cubic lattice reduces `η` by `0.045%`, against a measured ~12%.

### Why `det F` cannot see it

`base_mpm_solver.py` updates the deformation gradient as

$$\mathbf{F}^{n+1} = \left(\mathbf{I} + \Delta t\, \mathbf{C}\right)\mathbf{F}^{n}$$

where $\mathbf{C}$ is the **APIC affine velocity field**. An affine field represents uniform
stretch, shear and rotation and nothing else. Particles converging *within* one cell is a
sub-affine motion, so it lies in the transfer null space.

`det F ≈ 1` while particles clump is therefore **structurally guaranteed**, not a bug in the
`det F` computation. Any volume diagnostic derived from `F` is blind to this by construction.

This matters beyond bookkeeping: the Johnson–Cook return mapping is driven by `F`. If `F`
understates compression in the deformation zone, plastic flow is understated there too. The
bar reaching only ~56% of real elongation (77.4 mm simulated vs 93.07 mm measured) is
plausibly the same defect observed through a different variable — **untested**.

### Where it happens

Per-particle overlap volume at hit 17, res 10, binned by depth along the die-compression axis:

| \|z\| from centre (mm) | 0–1.8 | 1.8–3.4 | 3.4–5.3 | 5.3–7.1 | 7.1–8.8 | 8.8–10.4 | 10.4–12.2 | 12.2–19.1 |
|---|---|---|---|---|---|---|---|---|
| mean overlap (mm³) | 0.888 | 0.835 | 0.861 | 0.848 | 1.083 | 1.965 | **2.475** | 1.220 |

Overlap peaks at ~3× the core rate in a **subsurface shell**, then falls at the free surface.
Axially it is flat through the struck middle 60% and drops at the ends. Core (inner third)
0.94 vs skin (outer third) 1.29.

That is a forging deformation-zone map, not a contact-interface map — the overlap tracks
plastic shear intensity rather than proximity to the die.

### How much is fixable by timestep

Three `cfl_safety` levels at res 7, compared at hit 10, n=3 each:

| setting | packing drop | scatter |
|---|---|---|
| cfl 0.90 (stock dt) | −11.39% | ±0.04 |
| cfl 0.45 (half dt) | −8.62% | ±0.04 |
| cfl 0.225 (quarter dt) | −8.10% | ±0.04 |

Extrapolating to `dt → 0` gives ≈ **−7.6%**. So roughly **one third of the piling is a
time-integration artifact** that refinement removes, and **two thirds is structural**.

The deliverable configuration already runs `cfl 0.45`, so that third is already banked.

## The Actual Contact Stack (correction)

An earlier draft of this document described contact as grid-node velocity projection. That was
wrong. The production configuration runs THREE mechanisms:

1. **`mpm_grid_op`** -- rigid SDF evaluated at each grid node, node velocity projected. Grid level.
2. **CPIC** (`enable_CPIC=True`) -- in p2g, a particle does not deposit mass/momentum to stencil
   nodes on the far side of a rigid surface; in g2p, those flagged nodes get a particle-level
   collision response before `new_vel`/`new_C` accumulate.
3. **`apply_particle_contact`** (`base_mpm_solver.py`, called unconditionally from the substep
   whenever the rigid solver is active) -- a **custom kernel in this fork**, not upstream and not
   one of the genesis-dev modes. After g2p it applies a HARD non-penetration projection

       pos = pos - (signed_dist - margin) * normal_rigid     # margin = particle_size * 0.5

   plus Coulomb friction and restitution.

So the simulation is already a grid+particle hybrid, and (3) already enforces particle-level
non-penetration.

### Why this is the prime suspect for the piling

`apply_particle_contact` moves particle POSITIONS every substep, after the affine field is
formed, and that displacement never enters `C` -- so `F` cannot see it. That is precisely the
"material moves but det F does not know" mechanism, sitting in the production path. It predicts
the measured 9-10 pp det-F-vs-packing gap directly.

Corollary: `apply_particle_contact` runs in EVERY contact-mode arm, because it is not gated by
contact mode. Any contact-mode comparison therefore measures what a mode adds ON TOP of this
projection, not the mode in isolation. The ported `postg2p_position` mode is very nearly a
duplicate of it (same `psize/2` margin), so that arm double-projects.

## Why the Contact Modes Are Not the Lever

Six switchable rigid-MPM contact modes exist in the `genesis-dev` fork (`grid`, `particle`,
`fluidlab`, `postg2p_velocity`, `postg2p_position`, `penalty`). They differ in *where the
boundary velocity constraint is applied*. Three observations argue they cannot be the main
remedy:

1. **CPIC is already enabled** (`agforge/options.py`). The canonical fix for material smearing
   through a thin rigid boundary is active, and the piling persists. This is not material
   leaking through the die.
2. **The overlap peaks below the surface**, not at the interface.
3. The literature independently reaches the same conclusion — see below.

They remain worth benchmarking (a measured ranking is better than an assumed one), but as
characterisation rather than as the expected fix.

### Contact-mode implementation notes

- `penalty` is the only mode reachable without editing `base_mpm_solver.py`; the others are
  invoked from inside the solver's g2p (`mpm_solver.py:738` in the dev tree).
- The dev `penalty` implementation is a **prototype, not a control**: no damping (pure spring
  ⇒ chatter and energy injection), **no friction** (the `grid` path applies Coulomb friction
  via `coup_friction`, so grid-vs-penalty confounds contact mechanism with friction), contact
  detected from the particle **centre** only, and a default stiffness of `1e5` N/m against a
  physically consistent scale of `k ~ E·psize²/dx ≈ 5e7` N/m for this material.
- At `k = 5e7` the penalty CFL limit is ~2.3e-6 s against our substep of ~7e-7 s — feasible,
  with roughly 3× margin.

## What the Literature Says

Our symptom is a named pathology, described almost verbatim:

> "Excessive compression or tension in the kinematic field tends to entangle the particles from
> the initial uniform configuration, causing **artificial voids or aggregations** that lead to
> loss of the continuity of the stress field and the **singular deformation gradient of the
> particles**"
> — Dong, *Reseeding of particles in the material point method for soil–structure interactions*

And the contact-specific diagnosis:

> "**Significant errors can develop at material interfaces under large compression.** … the new
> methods are the **first MPM contact methods to account for particle deformation, which is
> important in large deformation compaction.**"
> — Nairn et al., [*New MPM Contact Algorithms … and Proper Null-Space Filtering*](https://www.cof.orst.edu/cof/wse/faculty/Nairn/papers/MPMContactRevisited.pdf)

The recurring theme across sources is that **point particles are the wrong primitive for
large-deformation compaction contact** — the particle needs extent. Bird et al.
([arXiv:2412.01565](https://arxiv.org/abs/2412.01565)) reach this independently, detecting
contact via "domains associated with each material point … without introducing boundary
representation in the material point method."

## Future Work

Ordered by implementation cost. **All of these touch `base_mpm_solver.py`**, which is frozen
pending the upstream rebase — sequencing is a coordination decision, not just a technical one.

### Grid modification (cheapest, and the widest unexplored space)

- **Grid shifting.** Randomising the background-grid offset each step stops cell-crossing error
  accumulating coherently. Used in MPM metal-cutting work with a Johnson–Cook law specifically
  "to reduce oscillations and enhance robustness" (Koßler et al., PAMM 2024).
- **Grid reorientation.** The die-compression axis is currently aligned with a lattice axis, so
  cell-crossing error is maximally correlated with the deformation direction. Rotating the grid
  relative to the loading axis is untested and cheap.
- **Non-Cartesian lattices.** BCC/FCC or unstructured backgrounds change the null-space
  structure entirely. Note `Circumventing volumetric locking…` and Wang et al. (IJNME 2021) both
  use simplex/tetrahedral backgrounds. Speculative here, but the search space is unexplored and
  the packing metric is precise enough to evaluate candidates cheaply.

### Transfer-scheme fixes

- **XPIC(m) / FMPM(k)** (Nairn). An explicit null-space filter — "projects current particle
  velocities onto the non-null-space velocities associated with MPM extrapolation matrices."
  Directly targets the diagnosed mechanism. ⚠️ Nairn reports interface artifacts when XPIC is
  applied in contact problems "without regard to contact effects."
- **Two-pass contact resolution.** Nairn finds each step "must resolve contact conditions twice
  — once to correct initial velocity extrapolations to the grid … then a second time to impose
  contact laws after updating grid momenta." Our coupler resolves once. ⚠️ Derived for
  *multimaterial* MPM; ours is MPM-vs-rigid-SDF, so transfer is not guaranteed.

### Particle-representation fixes (most likely to actually work)

- **TLMPM (Total-Lagrangian MPM)**, Vaucorbeil et al., CMAME 2020. Of BSMPM/GIMP/CPDI, "only
  CPDI effectively suppresses numerical fracture", at high complexity; TLMPM claims the same
  benefit while being "more efficient and easier to implement than CPDI". Validated on **impact
  of a steel cylinder bar and necking of cylinder alloy specimens** — very close to this
  problem's geometry, material and regime. **Current best bet.**
- **CPDI / GIMP.** The principled fix: particles carry a deformable domain, so volume is
  tracked geometrically and the F-vs-kinematics split cannot open. Highest cost.

### Ruled out

- **Particle reseeding.** Aimed precisely at our symptom, but non-differentiable and it changes
  particle count. Genesis maintains a working adjoint path (`compute_F_tmp.grad`,
  `mpm_postg2p_contact.grad`); reseeding would break it. Grid-shift, XPIC, CPDI and TLMPM are
  all adjoint-compatible.

### Not a route

- **The upstream rebase.** Across all 236 upstream commits since merge-base `fe76e6b9`, exactly
  two touch the MPM solver (an SVD speedup and a global data-packing refactor) and two touch
  `legacy_coupler.py` (an SPH buoyancy fix and the same refactor). **Zero MPM physics changes.**
  The rebase has value on other grounds; it will contribute nothing here.
- **Genesis's IPC coupler.** `genesis/engine/couplers/ipc_coupler/` contains no MPM references —
  it is FEM/rigid only. Barrier-method contact (cf. BFEMP, arXiv:2108.03349) would have to be
  written from scratch.

## Reproducing the Measurements

Analysis is CPU-only and runs off the recorded run databases — no GPU, no re-simulation. The
DBs store **raw particle positions** when `surface_mesh=False`, which is what makes all of this
possible after the fact.

| script | produces |
|---|---|
| `recon_kernel.py <run> <cpd> <meanJ1> <meanJH>` | kernel sweep + packing series → `review/kernel_<run>.json` |
| `recon_converge.py <run> <cpd> <meanJ1>` | marching-cubes grid convergence → `review/recon_<run>.json` |
| `piling_where2.py <run> <cpd>` | per-particle overlap, spatial breakdown → `review/where2_<run>.json` |
| `piling_vs_dt.py` | the dt ladder → `review/piling_vs_dt.json` |

⚠️ `outputs/` is **gitignored**. The run databases and every review artifact exist on one disk
only and are not version controlled.

## A Retired Metric

Reconstructed "geometric volume" via marching cubes was used earlier and should not be revived
without care. Holding the same particle cloud fixed and varying **only** the reconstruction
smoothing width `h` yields:

| h/psize | 0.70 | 0.85 | 1.00 | 1.20 | 1.50 | 2.00 |
|---|---|---|---|---|---|---|
| volume drop | −7.91% | −7.10% | −6.65% | −6.09% | −5.21% | −4.09% |

A 2× swing from a modelling choice. The marching-cubes *grid* converges cleanly (−6.09%,
Richardson-extrapolated −6.08%), which merely proves the grid was never the free parameter.

Genesis's own mesher reported −11.95% at res 10, within 0.05 pp of the parameter-free packing
measure — it appears to track the particle-body union rather than a smoothed envelope. That
number was right; the reasoning that produced it was not reproducible.
