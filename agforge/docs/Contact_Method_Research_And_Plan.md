# Rigid–MPM Contact Methods: Research, Analysis, and Implementation Plan

**Workstream A — contact fidelity of the Genesis MPM forging simulation.**

Status: **planning complete, implementation not started.**
Last updated: 2026-08-07.
Branch: `agforge/v2/forge-common` @ `cddad882` (1,404 uncommitted insertions across 8 files).

---

## 0. How to read this document

This is written for **whoever implements this next — quite possibly not the agent who wrote it.**
Assume no memory of the conversation that produced it. §22 is a quickstart if you need to be
productive in thirty minutes.

Every claim is tagged:

| tag | meaning |
|---|---|
| **[VERIFIED]** | read directly out of the code, with a line number, in the session that wrote this |
| **[MEASURED]** | a number from an actual simulation output file |
| **[LITERATURE]** | quoted from a named paper; the quote is reproduced so you need not re-fetch it |
| **[INFERRED]** | reasoning from verified facts; plausible, not confirmed |
| **[ASSUMED]** | taken on trust, not checked — treat with suspicion |
| **[RETRACTED]** | previously believed, now known wrong. Both views recorded on purpose. |

**Three hard rules learned the expensive way in this project:**

1. **Tag every number with the configuration it was measured under.** The dominant failure mode
   here has been inheriting numbers whose generating config later turned out broken, then
   reasoning on top of them. Specifically **DOMAIN-LIMITED** vs **DOMAIN-FIXED** (§5.1). An
   untagged number is not usable.
2. **Do not trust inherited claims, including the ones in this document.** Twelve confident
   assertions from earlier sessions turned out wrong (§16). Re-derive anything load-bearing.
3. **One sample is not a measurement.** Several wrong conclusions came from n=1 runs whose
   spread exceeded the effect being claimed (§16). Default to n=3.

**This document is meant to be edited as the work proceeds — see §24 for how, and for an honest
list of its own weak points.** Nothing in the experiment plan has been run yet; every experimental
claim here is a **prediction**.

---

## 1. The project and the objective

### 1.1 Physical setup

An Agility Forge open-die press strikes a 316L stainless steel bar **17 times**. There are **3D
scans of the real bar after each hit**, and instrumented press data. The bar starts at
⌀38.1 mm × 59 mm and ends ~93.4 mm long.

The simulation is an MPM (Material Point Method) model of that process, built on a fork of
Genesis, using a Taichi-derived JIT called `quadrants` (`qd.*` in source).

### 1.2 What we are optimizing — READ THIS CAREFULLY

The goal is **not** to implement the standard-correct contact formulation.

> **Objective:** find the best **sub-grid, particle-level** rigid-contact enforcement that
> (a) costs **less than refining the grid** to equivalent accuracy, and
> (b) damages the mathematical/physical accuracy of the simulation as little as possible.

The grid is the thing that is not accurate enough at the resolution we can afford. **Any proposal
that resolves to "so enforce the constraint on the grid instead" has answered a different
question** and should be rejected on those grounds — this exact mistake was made once during the
research and had to be corrected by the user.

The literature will push toward standard formulations (implicit MPM, augmented Lagrangian, convex
cone programming, barrier methods). Those are correct and largely **out of scope** (§15): they
require solver rewrites explicitly ruled out. **The literature's role here is to enumerate the
damages that sub-grid particle enforcement causes, so we can minimize them** — not to tell us
whether to do it.

We accept that the result may be non-standard and somewhat "hacky". The question is how good a
hack we can build and how honestly we can characterise its costs.

### 1.3 Hard constraints

- **No large solver rewrites.** Non-invasive, local changes only.
- **Must beat grid refinement on cost.** That is the entire justification for the work.
- **Accuracy is multi-dimensional:** geometry, contact/indentation fidelity, stress state, volume
  conservation, forces, particle overlap, convergence. No single metric decides.

### 1.4 Evaluation philosophy (explicit user direction)

- **No single comparison hit. No hard cutoff. No single scalar "winner".**
- Comparison is a **broad, nuanced analysis of full-sequence behaviour** across all relevant
  metrics.
- After the runs, **the analyzing agent selects** which methods are most interesting, important,
  or instructive to present and render — based on behaviour across the whole sequence, explicitly
  including methods that **fail informatively**, not only top scorers.
- Any weighting that collapses axes into one ranking is a **domain judgment belonging to the
  user**, not the agent.

### 1.5 Why testing, not pure theory, is the arbiter

Most of the existing particle methods were designed by earlier AI agents reasoning from theory
(§19). Theory has repeatedly under-determined the outcome here — the domain bug reversed a
stability ranking, and methods cross over mid-sequence (§5.3). The research phase (this document)
exists to make sure the *experiment set* is well-chosen and that we have not missed a factor.
**It does not replace running the experiments.**

---

## 2. Current state

### 2.1 Code [VERIFIED 2026-08-07]

| item | value |
|---|---|
| repo root | `~/GitHub/Genesis/aims-genesis/nsf-demo` ⚠️ **`~/GitHub/Genesis` is NOT a git repo** — it is a plain parent dir that also contains `forge_common/` |
| branch | `agforge/v2/forge-common` |
| HEAD | `cddad882` "Add record_real_hits to capture forge_common real sequences as HDF5." |
| working tree | **1,404 insertions / 53 deletions across 8 files, UNCOMMITTED** |
| untracked | `agforge/analysis/` (18 scripts), `agforge/docs/research/` |

Modified: `agforge/environment.py`, `agforge/options.py`, `agforge/recorder.py`,
`agforge/replay_episode.py`, `agforge/strike_controller.py`,
`genesis/engine/couplers/legacy_coupler.py` (+678),
`genesis/engine/solvers/base_mpm_solver.py` (+320), `genesis/options/solvers.py` (+40).

**Zero changes to `forge_common`** — standing repo rule.

### 2.2 Backups outside git

`~/contact_port_backup/` — `contact-modes-port.patch` (99,273 B, includes the domain fix),
`new-files.tgz` (79,552 B), plus `*.orig` / `*.prePhase0` snapshots.

### 2.3 Data

`~/GitHub/Genesis/forge_common/main/outputs/` — **5.1 GB, gitignored, single disk, no backup.**

| dir | contents | **config** |
|---|---|---|
| `batch/` | 3-hit sweep | DOMAIN-LIMITED |
| `batch17/` | 17-hit sweep | **DOMAIN-LIMITED — do not cite late-hit numbers** |
| `batch17n3/` | n=3 | DOMAIN-LIMITED |
| `batch17fix/` | 17-hit, n=1, 12 arms | **DOMAIN-FIXED** |
| `batch17fix_n3/` | n=3, 5 arms | **DOMAIN-FIXED** |

Per-arm files: `<tag>_hits.npz` (per-hit point clouds), `<tag>_verts.npz` (final surface),
`<tag>.diag.jsonl` (per-strike diagnostics), `_ref_fresh_verts.npz`, `summary.json`.

⚠️ `batch17fix/trajectory.json` covers **6 of 12 arms** — and those 6 are exactly the arms that
survived to hit 17. **Survivorship bias is baked into the only post-fix trajectory data that
exists.** The six casualties have partial data nobody has scored.

⚠️ **Volume, det F, GAP and force have NEVER been recomputed after the domain fix.**
`score_batch.py` was run on `batch17` only. Every packing-η / det-F / force number in any earlier
document or memory is DOMAIN-LIMITED.

### 2.4 Hardware

**NVIDIA GeForce RTX 3060 Laptop, 6144 MiB VRAM.** A real constraint — peak VRAM at grid
resolution 14 is **unmeasured** and may not fit.

### 2.5 Shared-environment hazard

⚠️ **Workstream B (thermal/induction) runs in the same WSL VM, the same pixi environment, and on
the same GPU.** It was observed running `pixi run --frozen python diag_thermal.py` repeatedly.
Two Genesis scenes on a 6 GB card can OOM each other. **Coordinate before launching long runs.**
Do not modify workstream B's branch or thermal defaults.

---

## 3. Glossary and metric definitions

Terms used throughout, and (where applicable) where they are computed.

### 3.1 MPM terms

| term | meaning |
|---|---|
| **P2G / G2P** | particle→grid and grid→particle transfers, once per substep |
| **`F`** | deformation gradient, carried per particle. `F ← (I + dt·C)F`, then SVD/plasticity |
| **`C`** | APIC affine velocity field, per particle. **Re-derived from the grid every G2P** |
| **`dx`** | grid cell size = **4 mm** here |
| **substep_dt** | the internal integration step, CFL-limited |
| **SDF** | signed distance field representing the rigid die; negative inside |
| **`influence`** | the soft-contact blend factor, `min(exp(−d/coup_softness), 1)` |
| **CPIC** | Compatible Particle-In-Cell — refuses P2G/G2P transfer across a thin rigid boundary |
| **GIMP / CPDI** | MPM variants that track the particle *domain*, not just its center |

### 3.2 Accuracy metrics

| metric | definition | where computed |
|---|---|---|
| **packing η** | union-of-balls volume of the particle cloud ÷ nominal volume. Detects particles piling into each other — **the "particle overlap" criterion** | `score_batch.py` |
| **det F** | determinant of the deformation gradient = the sim's *own* volume bookkeeping | `score_batch.py`, `.diag.jsonl` |
| **GAP** | `η − det F`. **The core diagnostic of this workstream**: det F can say volume is conserved while the particles have physically piled up. A large GAP means the bookkeeping and the actual arrangement disagree | `score_batch.py` |
| **surface deviation** (`dev_mean/p95/max`) | distance from sim surface to the real scan surface, mm. **The honest primary shape metric** | `geom_metrics.py`, `trajectory.py` |
| **IoU / Dice** | voxel intersection-over-union vs the real scan | `geom_metrics.py` ⚠️ **misleads here — see §5.5** |
| **elongation / `x_max`** | how far the free end has travelled; the deformation-progress check | `trajectory.py` |
| **penetration** (`pen_max`, `pen_mean_press`, `pen_frac_press`) | worst/mean depth of particles inside the die SDF, per frame, read-and-reset so `pen_max` is the worst over all substeps | coupler `mpm_penetration_probe`; emitted to `.diag.jsonl` |
| **force** (`force_L_peak`, `force_R_peak`, `force_*_press_mean`) | reaction force on each die | `.diag.jsonl`, `force_vs_real.py` |
| **hits reached** | how far the arm got before a `SimulationStabilityError`. **Report always alongside a deformation measure** (§5.4) | `summary.json` |

### 3.3 Arm naming convention

Arms are identified by a short tag used in every output file and plot.

| tag | mode | teleport (`mech`) | extra flags |
|---|---|---|---|
| `g0_grid_alone` | grid | 0 | — |
| `g1_grid_prod` | grid | 1 | — (**production default**) |
| `t1_teleport_only` | none | 1 | — |
| `ctl_none` | none | 0 | — (**do-nothing control**) |
| `h1_grid_cinj` | grid | 1 | `c_injection` |
| `h2_grid_pernode` | grid | 1 | `per_node` |
| `h3_grid_cproj` | grid | 1 | `c_project` |
| `h5_grid_ftmp` | grid | 1 | `ftmp` |
| `h5b_gridonly_ftmp` | grid | 0 | `ftmp` |
| `p1_particle` | particle | 0 | — |
| `p2_fluidlab` | fluidlab | 0 | — **added 2026-08-07, never run** |
| `p3_pg2p_pos` | postg2p_position | 0 | — |
| `p4_pg2p_vel` | postg2p_velocity | 0 | — **added 2026-08-07, never run** |
| `p5_penalty` | penalty | 0 | — |

Convention: `g*` grid family, `h*` hybrid (grid + a refinement flag), `p*` particle family,
`t*`/`ctl*` controls. With `--reps > 1`, tags get `_r1`, `_r2`, `_r3` suffixes.

**`ctl_none` is not decoration.** It has caught three separate traps (including validating the
penetration probe by positive control) and must stay in every sweep.

### 3.4 Run structure — what a "hit" actually is

One **arm** = **17 sequential strikes on the same billet**. The billet is *not* reset between hits;
deformation accumulates, which is why late hits are both the most informative and the most fragile.

Each strike is a variable number of solver frames — [MEASURED, `batch17fix/g0_grid_alone`]:

```
n_frames per strike:  91 105 136 121 101 125 170  91 151 102  83 180 206 130 123 125 147
press frames:         55  58  96  85  64  74  98  48  88  53  45 107 134  93  88  92  92
```

**Strike length varies by ~2.5× (83 → 206 frames).** Any per-hit quantity that is a *sum* or *count*
over frames is therefore not comparable across hits without normalizing. `pen_max` (a max) and the
`*_press_mean` quantities (means over press frames) are already comparable; raw counts are not.

Between **arms** the billet *is* reset, and `batch_arms.py` runs a **fresh-bar guard** (finite /
same particle count / same extent) that **aborts** rather than let a blown-up arm contaminate the
next one.

### 3.5 How to add a new arm

Edit the `ARMS` list in `agforge/analysis/batch_arms.py` (~line 39):

```python
ARMS = [
    dict(tag="g1_grid_prod", mode="grid", mech=1),
    dict(tag="h2_grid_pernode", mode="grid", mech=1, per_node=1),
    ...
]
```

Keys map onto the live setters in `configure()`: `mode` → `set_contact_mode`;
`per_node` / `c_injection` / `ftmp` → `set_refinement`; `mech` / `c_project` / `f_feedback` →
`set_particle_contact`. Then select with `--arms tag1,tag2,…`.

⚠️ **`mode` values are mutually exclusive (§6.3).** New *mechanisms* must be added as flags in the
solver/coupler first (§9.2), then exposed as a new key here — not as a new `mode` string.

---

## 4. Validation targets — what "accurate" is measured against

| target | value | notes |
|---|---|---|
| **Post-hit geometry** | 3D scans after each of the 17 hits | the primary shape reference |
| **Final elongation** | 59 → **93.4 mm** | sim reaches ~83.5 mm post-domain-fix = **89%** |
| **Blow #1 force** | **66.5 kN**, 7.12 mm bite, ε̇ ≈ 0.41 s⁻¹ | the cleanest single force validation point |
| **Press force limit** | **110.2 kN** — the press is force-limited | ⚠️ **no blow can evidence more than this**; do not treat higher sim forces as validated |
| **Die gap** | `live_position_mm` in the mcap; contact at 38.17 mm vs 38.1 mm billet | |

⚠️ **Temperature at the blow is NOT measured** — the thermal camera watches the induction coil, not
the strike. Any thermal comparison at the blow is unvalidated. (Workstream B territory.)

⚠️ **Two standing setup errors bound every arm's absolute accuracy regardless of contact method:**
1. the sim billet is **⌀40 mm vs the real ⌀38.3 mm**;
2. the **real press geometry (`Tool2_clip.obj`) is used for visualization but not for physics**.

These could plausibly reorder close results. They are unfixed and are a decision for the user.

---

## 5. Findings that reframed the work

### 5.1 🚨 The MPM domain was too short — invalidated all late-hit results [MEASURED]

The billet is pinned at +x by `fixed_region_bounds` and elongates in −x. Available room was
`0.85 × 59 − 29.5 = 20.65 mm`. The real sequence needs **34.4 mm**.

The simulated free end grew to `x_max = 77.80 mm` by hit 13 then **froze**: 77.80 / 77.80 / 77.82 /
77.82 for hits 14–17. **Every arm showed the same knee — which is how it was identified as a domain
artifact rather than a contact effect.** The user had independently recalled the sim "behaving well
through about hit 12 then suddenly weirdly," and an earlier reviewer had cut a render at exactly
that point (`outputs/review/review_hits_01_12.gif`) without noticing why.

**Fix** — `agforge/options.py:196`:
```python
mpm_x_padding_lower = self.cylinder_height * float(os.environ.get("AGF_MPM_X_PAD_LOWER", "0.85"))
```
⚠️ **The default is deliberately left at the broken 0.85** so the shared tree is unaffected.
**Any run that forgets `AGF_MPM_X_PAD_LOWER=1.3` silently reproduces the bug.** Minimum that fits
is 1.083. Cost: x cells 28 → ~33.

Effect [MEASURED, DOMAIN-FIXED vs DOMAIN-LIMITED]: surface p95 at hit 17 **11.04 → 5.96 mm
(−46%)**; IoU 0.534 → 0.569; elongation at hits 13–17 `78.3 → 79.9 → 81.1 → 82.6 → 83.5` instead
of frozen at 76.8; the abrupt 1.92× jump at hit 14 becomes 1.32×.

**Lesson worth generalizing:** a defect that affects *every* arm identically is invisible to
between-arm comparison. Only the absolute trajectory against ground truth exposed it.

### 5.2 🚨 The domain fix REVERSED the stability ranking

| config | `g0_grid_alone` | `g1_grid_prod` |
|---|---|---|
| DOMAIN-LIMITED | 7/7 stable | 4/7 stable |
| **DOMAIN-FIXED, n=3** | **[17, 14, 7] = 1/3** | **[17, 17, 17] = 3/3** |

Full n=3 DOMAIN-FIXED: `g1 [17,17,17]`, `h2 [14,17,17]`, `g0 [17,14,7]`, `h5 [7,7,17]`,
`p1 [14,4,14]`.

The wall had been capping deformation and masking an instability. **Any conclusion resting on the
old ordering must be re-derived.** Grid-alone still wins geometry/volume *when it survives*.

### 5.3 Ranking by final state is wrong — methods cross over [MEASURED, DOMAIN-FIXED, n=1]

Surface deviation p95, windowed:

| arm | hits 1–5 | hits 6–11 | hits 12–17 | AUC/hit | reached |
|---|---|---|---|---|---|
| `g0_grid_alone` | **1.081** | **1.532** | **3.943** | 2.251 | 17 |
| `p1_particle` | **1.202** | 7.673 | 18.061 | 9.436 | 17 |
| `g1_grid_prod` | 1.745 | 2.296 | 5.872 | 3.396 | 17 |
| `h2_grid_pernode` | 1.753 | 2.337 | 5.661 | 3.338 | 17 |
| `h3_grid_cproj` | 1.738 | 2.285 | 6.135 | 3.483 | 17 |

`p1_particle` **wins hits 1 and 3 outright** and is second over hits 1–5, then collapses to ~3×
worse. Endpoint scoring is blind to this. `g0` leads **15 of 17** hits.

[INFERRED] §7.1 explains the mechanism: `p1` writes only to the transient velocity channel, so
nothing accumulates to hold the geometry.

### 5.4 "Completed 17/17" is meaningless — three separate occurrences

`p1_particle` completes all 17 hits while reaching only **66.6 mm = 71% of real elongation**. It
survives by under-deforming. **Completion must always be reported alongside a deformation
measure.**

### 5.5 IoU misleads in this problem

`p1_particle` scores **higher IoU** than production while being **27 mm too short**. Voxel overlap
rewards being compactly wrong. **Use surface deviation as the primary shape metric; IoU secondary
at most.**

### 5.6 Discrimination depends on hit count [MEASURED]

At 3 hits: stability, packing η, force and penetration already discriminate. **det F and geometry
do NOT** — the best arm is only 2.6% above the do-nothing floor, and `ctl_none` has the *lowest*
surface deviation. **Geometry comparisons need the long sequence.**

---

## 6. How contact actually works in this codebase [VERIFIED]

Line numbers are against the working tree at `cddad882` + the uncommitted changes.

> 🚨 **LINE-NUMBER DRIFT HAZARD — read before trusting any citation in this section.**
> This document is **committed**; the ~1,404 lines of code it cites **are not**. If those changes
> are committed, rebased, reverted or edited, **every line number below drifts silently** — the
> citation will still look authoritative and point at the wrong code.
> **Before relying on a line number, grep for the symbol instead** (e.g.
> `grep -n '_func_collide_in_rigid_geom' genesis/engine/couplers/legacy_coupler.py`). The symbol
> names are stable; the line numbers are not.
> **The clean fix is to commit the code**, which would make these citations resolvable against a
> real commit. That is a decision for the user (§21).

### 6.1 The single shared primitive

`legacy_coupler.py:431` — `_func_collide_in_rigid_geom(pos, vel, mass, normal, influence, geom, i_b, …)`

```python
rvel = vel - vel_rigid
rvel_normal_magnitude = rvel.dot(normal_rigid)     # negative if approaching
if rvel_normal_magnitude < 0:                       # GATE: only acts on approaching material
    rvel_tan  = rvel - rvel_normal_magnitude * normal_rigid
    rvel_tan  = rvel_tan/|rvel_tan| * max(0, |rvel_tan| + rvel_normal_magnitude * coup_friction)
    rvel_normal = -normal_rigid * rvel_normal_magnitude * coup_restitution
    vel = vel_rigid + (rvel_tan + rvel_normal) * influence + rvel * (1 - influence)
    delta_mv = mass * (vel - vel_old)
    force = -delta_mv / substep_dt
    _func_apply_coupling_force(pos, force, link_idx, i_b, links_state)   # two-way reaction
```

**Key properties:**
- A **velocity-level impulse** with Coulomb friction and restitution.
- The `rvel·n < 0` gate means **a second application at the same sample point is a no-op.**
  Over-application only bites when two sample points **disagree on the normal** — exactly what
  happens when geometry is under-resolved. **[INFERRED — not yet measured. Testable prediction.]**
- The reaction force is the exact momentum change. With `mass = particle_mass × weight` and
  Σweight = 1, the per-particle reaction is **single-counted by construction** — explicit comment
  at `base_mpm_solver.py:862`.

### 6.2 The `influence` function

`legacy_coupler.py:338` (also 405, 531; `base_mpm_solver.py:835`):
```python
influence = min(exp(-signed_dist / coup_softness), 1)
```
active when `influence > 0.1`. Inside the die (`signed_dist < 0`) it clamps to **1 = full
projection**; outside it tapers over `≈ 2.3 × coup_softness`.

### 6.3 Modes are MUTUALLY EXCLUSIVE [VERIFIED — important]

`legacy_coupler.py:154`:
```python
self._rt_contact_mode = qd.field(gs.qd_int, shape=())      # a SINGLE SCALAR
```
Every gate is an equality test (`base_mpm_solver.py:735, 740`; `legacy_coupler.py:989`):
`mode_id == CM_GRID`, `== CM_FLUIDLAB`, …

**You cannot enable two modes at once.** `grid` + `fluidlab` is not expressible.

Modes: `grid`=0, `particle`=1, `fluidlab`=2, `postg2p_velocity`, `postg2p_position`, `penalty`=5,
`none`=6 (`CM_NONE` is fork-only).

By contrast the **refinement and teleport flags are independent booleans** — `_rt_per_node`,
`_rt_c_injection`, `_rt_ftmp_proj`, `_rt_pc_mech`, `_rt_pc_c_project`, `_rt_pc_f_feedback`,
`_rt_pc_c_damp`. **That is why `h1/h2/h3/h5` hybrids work and mode-level hybrids do not.**

⇒ **Design decision (§9.2): add all new mechanisms as independent FLAGS, never as modes.**

### 6.4 CPIC is excluded from every run so far [VERIFIED]

`legacy_coupler.py:187`:
> *"The in-g2p modes and CPIC both resolve contact inside g2p and would double-count. When
> switching is on, any in-g2p mode may be selected later, so CPIC is forbidden outright."*

So `AGF_CONTACT_RUNTIME_SWITCH=1` ⇒ CPIC off. **Every comparison to date is CPIC-off.**

Genesis's CPIC branch calls the primitive with **`influence = 1.0`** — hard contact, no
exponential softening. **Precedent that hard contact already exists in this tree.**

### 6.5 Substep order [VERIFIED]

```
substep_pre_coupling                   (base_mpm_solver.py:959)
    compute_F_tmp + svd
    mpm_ftmp_contact_projection        (:969)   ← ftmp: contact as STRESS, before p2g
    p2g
[coupler]  mpm_grid_op                 (legacy_coupler.py:834)  ← grid contact; penalty near here
          ★ FORECAST KERNEL GOES HERE ★                        ← see §11.1
substep_post_coupling                  (base_mpm_solver.py:1016)
    g2p                                          ← v AND C re-derived from grid (:932)
    mpm_postg2p_contact                (:1033)
    apply_particle_contact             (:1052)   ← the teleport
    mpm_penetration_probe              (:1066)
```

### 6.6 The teleport [VERIFIED — `base_mpm_solver.py:2140`]

```python
margin = self._particle_size * 0.5                      # = 1 mm
if signed_dist < margin:
    pos = pos - (signed_dist - margin) * normal_rigid   # HARD position projection
    self.particles[f+1, i_p, i_b].pos = pos
    ... friction/restitution velocity update as §6.1 ...
    # NO reaction force applied to the die
    if do_pc_c_project:  C = (I - n⊗n) C (I - n⊗n)      # H3
    if pc_c_damp != 1.0: C *= pc_c_damp
    if do_pc_f_feedback:                                 # H4
        d_push = margin - signed_dist
        eps = clamp(pc_f_fb_scale * d_push * inv_dx, 0.0, 0.5)
        F = (I + eps * ...) ...                          # crude linear write-back into F
```

**The teleport is the ONLY mechanism that samples at the particle SURFACE** (center offset by the
particle radius) — the location the literature says is correct (§8). But it enforces this by
**moving position without updating F** (that *is* the particle-piling volume bug) and applies **no
force to the die**.

### 6.7 Mechanism table [VERIFIED]

| mechanism | where sampled | channel written | enters F? | force on die? |
|---|---|---|---|---|
| `grid` | grid node center | grid velocity | ✅ via C | ✅ |
| `particle` | particle center | `grid_vel` in-G2P, before C forms | ✅ via C | ✅ |
| `per_node` | grid node center | `grid_vel` in-G2P | ✅ via C | ✅ |
| `c_injection` | particle center | **C directly** (n⊗n component) | ✅ | ❌ |
| `fluidlab` | **predicted position** `x + dt·v` | `new_vel`, **after C is formed** | ❌ | ❌ |
| `ftmp_projection` | particle | **F_tmp**, pre-p2g ⇒ stress | ✅ | ❌ (indirect) |
| teleport | **particle surface** | **position** | ❌ | ❌ |
| `c_project` | particle surface | C → `(I−n⊗n)C(I−n⊗n)` | ✅ via C | ❌ |
| `f_feedback` | particle surface | **F** (crude, clamped) | ✅ | ❌ |
| `penalty` | particle | spring force | ✅ via C | ✅ |

`c_injection` (`base_mpm_solver.py:~893`):
```python
rvel_n = (new_vel - v_rigid).dot(rigid_normal)
if rvel_n < 0.0:
    d_eff = max(rigid_signed_dist, 0.5 * self._dx)      # ⚠️ GRID-SCALE FLOOR on a sub-grid quantity
    c_nn_target = rvel_n / d_eff
    new_C += (c_nn_target - n·new_C·n) * n⊗n
```

`fluidlab` (`base_mpm_solver.py:~915`):
```python
pred_pos = pos + substep_dt * new_vel                    # ← already a FORECAST
new_vel = _func_fluidlab_collide(pred_pos, new_vel, ...)
```
Our own comment: *"resolved at the predicted position AFTER new_C is formed, so it intentionally
does not feed the affine field, and it applies no reaction force to the rigid body."*

### 6.8 Side-flip protection exists but MPM does not use it [VERIFIED]

`_func_collide_with_rigid_geom_robust` (`legacy_coupler.py:367`) carries a `normal_prev` per
(particle, geom) to detect the SDF normal flipping — a particle crossing the **medial axis** of a
thin body. Docstring: *"additionally handles potential side flip due to penetration."*
**It returns `(vel, normal_rigid)`** (`:428`) — so the caller must persist the normal.

| call site | robust variant? |
|---|---|
| `legacy_coupler.py:994` — **MPM grid contact** | ❌ **no** |
| `legacy_coupler.py:1169` — FEM | ❌ no |
| `legacy_coupler.py:1323` — **SPH** (`sph_rigid`) | ✅ yes |

**The hazard was recognized, the fix was written, and MPM never got it.** [INFERRED] strong
candidate cause for `p5_penalty` dying at hit 2 and `p3_pg2p_pos` at hit 3.

⚠️ **Two implementation gotchas for the port (§11.2):**
1. The persistent-normal field `mpm_rigid_normal` **exists but is allocated only when CPIC is
   enabled** (`legacy_coupler.py:209-212`, guarded by `if self._rigid_mpm and
   self.mpm_solver.enable_CPIC:`). Porting the guard requires allocating it unconditionally or
   under a new flag.
2. `mpm_surface_to_particle` (`:1107`) already maintains that normal, but with a **different
   rule** — it *refuses* to update on a flip (`if sdf_normal.dot(stored) >= 0: stored = sdf_normal`),
   i.e. keep-old-on-flip. That is CPIC's convention, **not** the robust variant's. Do not assume
   the two are interchangeable.

### 6.9 Length scales [VERIFIED / MEASURED]

| quantity | value | source |
|---|---|---|
| grid spacing `dx` | **4 mm** | `pen_dx` in diag output |
| particle spacing | **2 mm** = `dx / AGF_PPC_DIVISOR` (default 2.0) | `agforge/options.py:356` |
| particle count | 9,266 | diag |
| teleport margin | **1 mm** (= particle_size/2) | `base_mpm_solver.py:2168` |
| `coup_softness` | **5e-4 m = 0.5 mm** | `agforge/options.py:133` |
| ⇒ influence band | ≈1.15 mm = **0.29 dx** | at the 0.1 gate |
| die tip width | ~2.4 cells ≈ 9.6 mm | ⚠️ **[ASSUMED — inherited from prior sessions, never re-verified.]** Load-bearing for the whole "under-resolved" premise; worth confirming against the physics die geometry. |

**⭐ `particle_size = dx / AGF_PPC_DIVISOR`** (`agforge/options.py:356`). This is a **coupling nobody
has exploited**: raising `AGF_PPC_DIVISOR` refines the *particles* while leaving the *grid*
untouched. That is a sub-grid accuracy knob orthogonal to everything in §10 — and it is a
**competing hypothesis** to the whole contact-method programme (maybe more particles per cell, not
a better contact operator, is the cheap win). It belongs in the frontier study as its own axis
(§12.4). Cost scales with particle count only, not grid cells.

⚠️ `AGF_CELLS_PER_DIAMETER` **defaults to 7**, but every run in this workstream passes **10**
explicitly. Another case of "the default is not what we run" — same hazard class as
`AGF_MPM_X_PAD_LOWER` (§5.1).

**Two consequences:**
1. The contact band is **narrower than a third of a cell**, so a node feels nothing until it is
   essentially on/inside the die ⇒ strong **aliasing** as the die crosses cells.
2. **`coup_softness` is a fixed absolute length that never scales with `dx`.** Refining the grid
   does not refine the contact regularization ⇒ **the scheme cannot converge to the exact contact
   constraint**; it converges to a 0.5 mm-smeared contact. Confirmed independently three times (§8).

---

## 7. Theory: why sub-grid particle enforcement is hard

### 7.1 The channel-durability hierarchy [INFERRED — the central organizing idea]

Per substep: P2G → grid update → G2P → F update → advect.

| channel | fate next substep | durable? | consistent with constitutive model? |
|---|---|---|---|
| `v`, `C` | **overwritten in G2P from the grid** (`base_mpm_solver.py:932`) | ❌ transient | ✅ |
| `x` (position) | integrated from `v`; never reset | ✅ durable | ❌ **decouples x from F** |
| `F` | updated multiplicatively; never reset from grid | ✅ **durable, accumulates** | ✅ |

**This explains observed behaviour:**
- `p1_particle` writes only to `v` ⇒ nothing accumulates to hold the geometry ⇒ excellent for ~5
  hits then collapses (§5.3).
- The teleport writes to `x` ⇒ durable but F-inconsistent ⇒ holds geometry, wrecks volume.
- `ftmp` / `f_feedback` / `c_injection` write the F channel ⇒ durable *and* consistent.

**Design principle:** sub-grid contact information should reach **F** (via C or stress). Velocity
is the weakest place to put it — and it is where most of our methods put it.

**⚠️ Nuance, do not overstate:** a velocity correction is not *useless* — it still influences the
next P2G momentum. It is *filtered*, not erased. The claim is about relative durability.

### 7.2 The P2G null-space problem [LITERATURE + INFERRED]

P2G is a Galerkin projection onto the span of the grid shape functions. A correction applied at a
particle position generally has a component in the **null space** of that projection, which is
**permanently discarded** at the next P2G.

> *"any portion of that particle velocity field that lies in the null space of the grid's shape
> functions is permanently discarded. This causes artificial numerical damping, strips away part
> of the applied contact force, and causes the material to spuriously **relax back into the rigid
> body** in subsequent steps."*

**Falsifiable prediction**, and the penetration probe is the instrument that would see it.
**Not yet tested.**

### 7.3 Particle-center evaluation under-penetrates by half the particle spacing

> *"Evaluating at particle centers … ignores the particle's volumetric domain. The material will
> unphysically penetrate the rigid body by half the particle spacing before generating a reaction
> force."*

Particle spacing 2 mm ⇒ predicted ~1 mm. [MEASURED] grid-alone penetration plateaus at 0.6–1.7 mm.
And the teleport's `margin` is **exactly** `particle_size/2 = 1 mm` — the code already compensates
for this specific error, apparently without anyone having named it.

### 7.4 Penetration behaviour [MEASURED, DOMAIN-FIXED]

`g0_grid_alone`, 17 hits, `batch17fix`, `pen_max` in mm, with the strike length from the **same
file** so the two are self-consistent:

| hit | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `pen_max` | 0.62 | 0.84 | 1.04 | 1.55 | 1.06 | 1.13 | 1.48 | 1.06 | 1.18 | 0.63 | **0.00** | 1.51 | **1.80** | 0.65 | 1.18 | 1.45 | 1.44 |
| `n_frames` | 91 | 105 | 136 | 121 | 101 | 125 | 170 | 91 | 151 | 102 | **83** | 180 | 206 | 130 | 123 | 125 | 147 |

⇒ **plateaus in the 0.6–1.8 mm band (0.15–0.45 × dx). It does NOT run away.** Max is 1.80 mm at
hit 13.

**Exactly one hit reads 0.00 — hit 11 — and it is the shortest strike in the sequence (83
frames).** So the zero is kinematic (the die barely engages), not instrument dropout.

> ⚠️ **This paragraph previously carried an error worth learning from.** An earlier draft quoted the
> **DOMAIN-LIMITED** series, said *two* hits read zero (11 and 14), and explained them as "the short
> strikes (74/94 frames)". Hit 14 is not short (130 frames) and the 74/94 figures appear in neither
> dataset — the frame counts had been cross-referenced from a **different run configuration** than
> the penetration numbers. This is precisely the failure mode rule 1 in §0 exists to prevent, and it
> got past two drafts. **Never mix quantities from different configs in one sentence.**

⚠️ **Teleport-on arms read exactly 0.0 by construction** (the projection targets
`signed_dist ≥ margin`). **Penetration discriminates only among teleport-off arms.**

**Probe validated by positive control:** `ctl_none` (no contact at all) reads **6.33 mm = 1.58 dx**,
`pen_frac 0.187`, `detF 1.0`, `force 0.0`. Without that control, a zero from a dead probe would be
indistinguishable from a zero from perfect contact.

---

## 8. Literature findings

Quotes below are reproduced from the **primary papers** so this section stands alone. Two AI
research runs also informed §7–§9; they are **not committed** and their substantive content is
already extracted here — see §23 for why, and how to regenerate them.

### 8.1 CPIC — Hu et al. 2018, MLS-MPM (SIGGRAPH) [LITERATURE]

Its **own stated limitation**:
> *"While CPIC tackles infinitely thin boundaries, it only resolves features at a scale of grid
> Δx. Thus we cannot handle sub-grid level boundary configurations such as sharp corners and
> narrow gaps… One possible future direction for increasing the accuracy would be enforcing a
> smoother transition region based on sub-grid features."*

⚠️ **An AI research run recommended CPIC as "the most … accurate explicit choice for your exact
setup." That contradicts the primary source.** CPIC fixes *leakage through* a thin body; our die is
a solid tool contacted on one face, and our problem is *geometric resolution of the contact
surface*, which CPIC explicitly does not solve. **Primary source wins.** CPIC is still worth
testing (never run) but is not the answer. *This is also a worked example of rule 2 in §0.*

### 8.2 Oxford / Warwick / NTUA 2024, arXiv 2403.13534 [LITERATURE]

Penalty-based **material-point-to-segment** contact with an explicit gap function + Extended
B-Splines.
> *"This approach improves the representation of the contact surface, **preventing premature and
> unrealistic body contact, a limitation observed in standard MPM contact algorithms.**"*
> *"The proposed approach converges to exact solutions even with coarse meshes, unlike OBS, which
> necessitates a finer resolution."*
> *"It demonstrates better energy conservation than other state-of-the-art MPM contact variants."*

### 8.3 Durham / Dundee / Arup / BGS 2024, arXiv 2412.01565 [LITERATURE]

Large-deformation engineering contact — the closest regime to forging. Contact detected at **GIMP
domain corners** via closest-point projection:
> *"This is necessary so that contact occurs on the material boundary, **otherwise the contact is
> not consistent and spurious stresses are observed** for the contact GIMPs."*

### 8.4 FluidLab — Xian et al., ICLR 2023 [LITERATURE] — provenance-critical

Flagged by the user as **the one method in our codebase derived from human researchers rather than
an AI agent**, and therefore carrying different (not necessarily higher) evidential weight.

> *"the contact between particle-based fluid bodies and SDF-based rigid objects **occurs in the
> grid operations**… we blend this velocity with the original velocity using a distance-based
> blending factor α, given by **α = min{exp(−d), 1}**… This provides a smoothed contact zone and
> thus **offers smoothed gradient information during contact.**"*

**Two consequences:**
1. **Genesis's `grid` contact IS the FluidLab contact model.** `α = min(exp(−d),1)` is
   character-for-character our `influence`. The mode we call `fluidlab` is named after a paper
   whose published model is the one we call `grid`.
2. **The exponential smoothing exists for DIFFERENTIABILITY, not accuracy.** Its entire stated
   justification is smooth gradients for trajectory optimization. **We do not optimize through
   contact.** ⇒ `coup_softness` is plausibly a pure accuracy tax on us. **[INFERRED — test it, M1.]**

⚠️ FluidLab's coupling is documented (by SoftMAC's comparison table) as **one-way** — no reaction
force on the tool. Its design was never under pressure to get forces right, which is one of our
primary metrics. Weight accordingly.

### 8.5 ⭐ SoftMAC — Liu, Yang, Luo, Shao (CMU / NUS), arXiv 2312.03297 [LITERATURE]

**The closest published match to this project's goal.** Motivating sentence:
> *"MPM struggles with delicate boundaries. **Scaling down grid size and time step alleviates the
> problem, but is not always feasible** … where computational efficiency should be balanced. [Prior
> work applies] **particle forces to reduce penetration, but the method introduces problems such as
> unnatural rebound.**"*

**The forecast-based contact model:**
```
v_init = W · v̂_g                       # G2P forecast — look ahead
v_tgt  = BC_p(v_init)                   # apply the constraint AT THE PARTICLES  (sub-grid)
v_g    = argmin_vg ‖Wᵀ v_g − v_tgt‖²    # solve for grid velocity that best realizes it
       ≈ v̂_g − α · W (Wᵀ v_g − v_tgt)   # ONE gradient step; α = 0.2; recovers 83.1%
```
Cost: *"equivalent to interpolating a particle property to grid and transferring it back."*

Their `BC_p`: drop normal component, Coulomb-decay tangential:
`v_out = v_t · max(0, 1 − μ‖v_n‖/‖v_t‖) + v_c`
plus two options — (1) blend with `s = min{exp(−βd),1}`, *"reduces drastic state changes and
improves gradient quality"* (again a gradient motivation, not accuracy); and (2) *"Advect the
particle with v_out, and if penetration happens, add a component to v_out to ensure that the
particles are moved to the nearest legal position"* — **a velocity-level teleport rather than a
position hack.**

**Their quantitative results (TABLE II)** — 2D fluid in a shaking circular container, 1100 frames,
Tesla T4:

| model | penetrations @ thickness 1/32 | @ 1/64 | unnatural rebound | time |
|---|---|---|---|---|
| Grid | 449 | **5998** | no | 4.90 s |
| Particle (k=400) | 12 | 1512 | no | 4.45 s |
| Particle (k=600) | 0 | 217 | **yes** | 4.33 s |
| **Forecast (theirs)** | **0** | **3** | **no** | 5.24 s |

**Interpretation for us:**
- Grid contact degrades catastrophically as the boundary thins (449 → 5998). **Direct quantitative
  support for the premise that sub-grid particle enforcement is worth doing.**
- Particle-level is 2–3 orders of magnitude better, but the penalty variant trades penetration
  against rebound (raise `k` to kill penetration, get rebound). **This is the same trade our
  `p5_penalty` arm is stuck in.**
- Forecast dominates both with **no rebound at ~+7% runtime**.

⚠️ **Caveats:** validated on **liquid in thin containers**, not 17-blow metal forging with
plasticity. The penetration ranking should transfer; volume/stress behaviour in our regime is
unknown. The +7% is *their* code — ours must be measured.

⚠️ SoftMAC states: *"we maintain a CFL number below the critical threshold of **0.3** to ensure
that the time integration scheme remains stable."* **We run `AGF_CFL_SAFETY=0.45`.** Independent
corroboration of the timestep instability already on record for this workstream, and a confound
under every arm ever compared.

### 8.6 Deep Research bottom line (59 sources) [LITERATURE]

> *"**Avoid combining grid-level multimaterial contact and particle-boundary contact**; instead,
> choose a single, boundary-based contact discretization…"*
> *"If you borrow ideas from IPC/BFEMP, make sure **barrier parameters scale with grid resolution**
> as in convergent IPC to avoid a non-convergent thickened interface."*

Its full recommendation (implicit MPM + augmented Lagrangian / convex cone programming) is **out of
scope** (§15) — recorded for completeness.

### 8.7 Leads not yet pursued

- **CK-MPM** (compact-kernel MPM, arXiv 2412.10399) — the quadratic B-spline's 3Δx support *is* the
  smearing width; a compact kernel narrows it. **Sub-grid contact accuracy at unchanged grid
  resolution — directly our currency.** Probably the highest-value unexplored lead.
- **OS-MPM** (overlapping Schwarz space-time refinement, arXiv 2605.09097) — *local* refinement only
  where deformation/contact concentrates. A legitimate middle path vs global refinement.
- **BFEMP** (barrier contact, arXiv 2108.03349); **convex formulations** (2403.13783, 2503.05046).
- **ManiSkill2** — two-way, penalizes particles; another human-designed particle-level precedent
  worth reading for its penalty tuning.

---

## 9. The answer to the merge question

**Do not "combine" grid and particle contact.** They are two discretizations of *the same*
non-penetration constraint; enabling both applies one constraint twice at two sample points. The
`rvel·n < 0` gate makes a second application at the *same* point a no-op, so the damage occurs
specifically where the two sample points **disagree on the normal** — precisely the under-resolved
regime we are in.

**Instead: the forecast/projection pattern.**
- Detection and constraint formulation stay **at the particles** ⇒ full sub-grid accuracy (the
  thing this project exists to get).
- Only the **result** is projected onto the grid, by explicit least squares.
- Because the corrected velocity goes back through the grid, the real G2P re-derives **both `v` and
  `C`** ⇒ contact reaches `F` and stress by the normal route. No position hacking, no double
  application, and the reaction force is available.

**Note carefully:** this is *not* "move the constraint to the grid." Detection stays sub-grid. Only
the realization is projected. That distinction is the whole point.

| | sub-grid geometry | persists | F consistent | force on die |
|---|---|---|---|---|
| grid | ❌ | ✅ | ✅ | ✅ |
| particle / postg2p_vel | ✅ | ❌ | ✅ | ✅ |
| fluidlab (as built) | ✅ | ❌ | ❌ | ❌ |
| teleport | ✅ | ✅ | ❌ | ❌ |
| c_injection / ftmp | ✅ | ✅ | ✅ | ❌ |
| **forecast** | ✅ | ✅ | ✅ | ✅ |

### 9.1 ⭐ Our `fluidlab` mode is already the forecast, minus two pieces

`fluidlab` **already evaluates contact at the predicted position** `x + dt·v`. That is SoftMAC's
look-ahead, already implemented. It lacks the reaction force and the grid projection.

| step | change | adds |
|---|---|---|
| **0** | `fluidlab` as-is — **already built, never run** | forecast detection at particles |
| **1** | + reaction force | two-way coupling, force fidelity |
| **2** | + grid least-squares writeback | representability, C/F consistency |

Step 2 **is** SoftMAC. Each rung is independently testable and attributable.

### 9.2 Design decision: flags, not modes

Because modes are mutually exclusive (§6.3) and flags are not, **all new mechanisms must be added
as independent boolean flags**, following the `per_node` / `c_injection` / `ftmp_proj` pattern.
This gives combinability **without splitting the mode enum**, which was previously (and wrongly)
assessed as requiring a large refactor. It does not.

---

## 10. Method inventory

### 10.1 Already implemented

| # | mechanism | kind | status |
|---|---|---|---|
| 1 | `grid` | mode | extensively tested |
| 2 | `particle` | mode | tested |
| 3 | **`fluidlab`** | mode | ✅ built — **NEVER RUN** |
| 4 | **`postg2p_velocity`** | mode | ✅ built — **NEVER RUN** |
| 5 | `postg2p_position` | mode | tested; **duplicates the teleport** — only ever run double-projecting |
| 6 | `penalty` | mode | dies at hit 2 |
| 7 | `none` | mode | control; keep in every sweep |
| 8 | `per_node` | flag | tested |
| 9 | `c_injection` | flag | tested |
| 10 | `ftmp_projection` | flag | tested |
| 11 | teleport (`pc_mech`) | flag | production default |
| 12 | `c_project` | flag | tested |
| 13 | **`f_feedback`** | flag | ✅ built — **NEVER IN ANY ARM** |
| 14 | `c_damp` | flag | untested |
| 15 | **CPIC** | build option | ✅ built — **excluded from every run** (§6.4) |

Infrastructure done: runtime switching, penetration probe (validated by positive control),
`batch_arms.py`, `trajectory.py`, the domain fix.

### 10.2 Modifications needed

| # | change | size | rationale |
|---|---|---|---|
| **M1** | `coup_softness ∝ dx`, plus a **hard-contact** option (`influence = 1`) | ~3 lines | fixed length breaks convergence (§6.9); softening serves FluidLab's gradients (§8.4); CPIC already uses hard (§6.4) |
| **M2** | particle-**surface** gap for all particle modes | ~1 line/site | removes the half-particle-spacing error (§7.3) from every particle mode at once |
| **M3** | drop `c_injection`'s `d_eff = max(sd, 0.5·dx)` floor | 1 line | grid-scale floor on a sub-grid quantity, self-defeating |
| **M4** | port the side-flip guard to MPM | small — see §11.2 | §6.8; likely cause of `penalty` / `pg2p_pos` early deaths |
| **M5** | narrow the CPIC-vs-runtime-switching guard | ~5 lines | currently forbids CPIC outright; should forbid only genuine in-G2P conflicts |
| **M6** | add missing arms: `f_feedback`, **teleport + f_feedback**, `c_damp`, CPIC | config only | untested F channel; the F-consistent teleport |

### 10.3 New implementation

| # | item | size | notes |
|---|---|---|---|
| **N1** | reaction force for `fluidlab` | small | ladder rung 1 |
| **N2** | **forecast contact kernel** | **medium — the main build** | §11.1 |
| **N3** | per-arm cost/timing instrumentation | small | mandatory — nothing is judgeable on cost without it |
| **N4** | resolution-sweep harness + **VRAM probe** | small | gated by 6 GB (§2.4) |
| **N5** | trajectory analysis over **all metrics and all arms including failures** | medium | currently 6/12 arms, survivors only (§2.3) |

---

## 11. Implementation specifications

### 11.1 N2 — the forecast contact kernel

**Location:** a new kernel between `mpm_grid_op` and `g2p` (§6.5). It reads finalized grid
velocities and writes grid velocities. **It touches no existing kernel.**

**Structure — two passes, one temp field.**

*Field to allocate:* one `qd.Vector.field(3)` per particle (per batch), e.g. `_fc_residual`.

**Pass 1 — forecast and constrain (per particle):**
```
base, fx, w   = same stencil arithmetic as g2p (base_mpm_solver.py:720-722)
v_init        = Σ_offset  weight * grid[base+offset].vel_out      # this IS g2p's velocity sum
d, n          = SDF distance and normal at the particle
                 ⚠️ use the PARTICLE SURFACE: d_surface = d − particle_size/2   (M2)
if d_surface < threshold:
    v_c   = rigid velocity at the contact point
    rvel  = v_init − v_c
    v_n   = (rvel·n) n ;  v_t = rvel − v_n
    v_out = v_t * max(0, 1 − μ‖v_n‖/‖v_t‖) + v_c        # drop normal, Coulomb-decay tangential
    v_tgt = v_out                                        # optionally blend by `influence` (M1)
else:
    v_tgt = v_init
_fc_residual[p] = v_init − v_tgt
```

**Pass 2 — project onto the grid (scatter):**
```
grid[base+offset].vel_out −= α * weight * _fc_residual[p]        # α ≈ 0.2
```

**Derivation of pass 2** (so the next implementer can check it rather than trust it): minimizing
`‖Wᵀv_g − v_tgt‖²` gives gradient `2W(Wᵀv_g − v_tgt)`, so one descent step is
`v_g ← v_g − α W r` where `r = v_init − v_tgt` is the per-particle residual and `W` scatters with
the same interpolation weights used by G2P. Hence "subtract `α · weight · residual` at each of the
27 nodes."

**⚠️ Open design choice — TEST BOTH:** the plain least-squares form above scatters with the
*interpolation weight only*. Our P2G is **mass-weighted**. Mass weighting changes the norm being
minimized (it becomes a mass-weighted least squares), which is arguably more physical. **Neither is
obviously right; make it a flag and measure.**

**Reaction force:** the impulse applied is `m_p (v_tgt − v_init)`; the reaction on the die is
`−m_p(v_tgt − v_init)/substep_dt`, accumulated per geom via the existing
`_func_apply_coupling_force`. This reuses the `mass × weight` single-count convention (§6.1).

**Cost estimate:** two extra particle passes ≈ one extra G2P + one extra P2G per substep.
SoftMAC measured +7%; **ours must be measured, not assumed** (N3).

**Acceptance test:** with the forecast flag OFF, results must be **bit-identical** to the current
build (the same all-or-nothing discipline used for the runtime-switch port — see §12.6).

### 11.2 M4 — porting the side-flip guard to MPM

1. `_func_collide_with_rigid_geom_robust` returns **`(vel, normal_rigid)`** — the caller must
   persist the normal per (particle, geom, batch).
2. The field `mpm_rigid_normal` **already exists with exactly that shape**
   (`legacy_coupler.py:212`) but is **allocated only when CPIC is enabled**. Allocate it
   unconditionally, or under a new flag, before using it in the MPM path.
3. ⚠️ **Do not reuse `mpm_surface_to_particle`'s update rule** (`:1131`). It uses *keep-old-on-flip*
   (`if sdf_normal.dot(stored) >= 0: stored = sdf_normal`), which is CPIC's convention, not the
   robust variant's. Follow the robust variant's own contract.
4. Switch the MPM grid-contact call site (`legacy_coupler.py:994`) behind a flag so the
   before/after comparison is a controlled experiment.

### 11.3 M1 — contact regularization

Three settings behind one enum-ish flag so they are directly comparable:
- `soft_fixed` — current behaviour, `coup_softness = 5e-4` (the DOMAIN-FIXED baseline)
- `soft_scaled` — `coup_softness = k · dx` (start `k ≈ 0.125` to match today's 0.5 mm at dx = 4 mm,
  so at res 10 it reproduces current behaviour exactly and only *differs under refinement* — that
  makes the convergence study clean)
- `hard` — `influence = 1` inside, no contact outside (matches Genesis's own CPIC path)

### 11.4 M2 — particle-surface gap

Everywhere a particle-sampled `signed_dist` feeds the contact primitive, subtract the particle
radius: `d_surface = signed_dist − particle_size * 0.5`. Behind a flag. Note the teleport already
does this via `margin`; the point is to make the *velocity* methods consistent with it.

---

## 12. Experiment plan

### 12.0 Ordering principle

Harvest free information before building; correct known errors before measuring on top of them;
measure cost before claiming efficiency.

### 12.1 Stage 0 — unblock and harvest (no physics changes)

- Fix the launch: **`pixi run --frozen`** (§18).
- Coordinate the GPU with workstream B (§2.5).
- Run the 9 standalone/control arms at **n=3, 17 hits, DOMAIN-FIXED**, including **`fluidlab` and
  `postg2p_velocity` for the first time ever**.
  ⚠️ **A CPIC arm is NOT part of Stage 0.** CPIC is forbidden whenever runtime switching is on
  (§6.4), so testing it requires **M5**, which is a Stage-1 code change. Stage 0 runs with
  `AGF_ENABLE_CPIC=0` like everything before it. *(An earlier draft listed a CPIC arm here while
  also calling Stage 0 "no physics changes" — a contradiction; this is the correction.)*
  ⚠️ **`p2_fluidlab` and `p4_pg2p_vel` have never executed.** Their mode strings are verified to map
  correctly (`_CONTACT_MODE_TO_ID`), but the in-G2P wiring is unproven — Stage 0's headline "free
  data" may simply throw. **If it does, that is itself a finding**; diagnose before assuming the
  modes are useless (§19: prior "unusable" verdicts are hypotheses, not exclusions).
- Add timing instrumentation (N3).
- **Re-run `score_batch.py` on the DOMAIN-FIXED sweeps** — the volume/force axis has never been
  verified post-fix (§2.3).

**Copy-pasteable** (`p2_fluidlab` and `p4_pg2p_vel` were added to `ARMS` on 2026-08-07 and have
never executed):
```bash
wsl.exe -e bash -lc 'cd ~/GitHub/Genesis && export PATH="$HOME/.pixi/bin:$PATH" \
 && export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH \
 && setsid nohup env AGF_MPM_X_PAD_LOWER=1.3 AGF_CONTACT_RUNTIME_SWITCH=1 \
    AGF_DIAG_PENETRATION=1 AGF_CELLS_PER_DIAMETER=10 AGF_CFL_SAFETY=0.45 AGF_ENABLE_CPIC=0 \
    PYTHONPATH=forge_common/main \
    pixi run --manifest-path aims-genesis/nsf-demo/pixi.toml --frozen \
    python -u aims-genesis/nsf-demo/agforge/analysis/batch_arms.py \
      --n-hits 17 --reps 3 \
      --arms g1_grid_prod,g0_grid_alone,t1_teleport_only,ctl_none,p1_particle,p2_fluidlab,p3_pg2p_pos,p4_pg2p_vel,p5_penalty \
      --out batch_standalone_n3 > ~/standalone_n3.log 2>&1 < /dev/null &'
```
Expected wall time ≈ **25–35 min** (many arms die early and cost little). Verify it actually
started with `ps -eo pid,cmd | grep pixi | grep -v grep` — **not** `pgrep -f` (§18).

**Acceptance / decision:** if `fluidlab` behaves well even without force or writeback, the ladder's
value rises sharply. If it is unstable, diagnose *why* before building rungs 1–2 on top of it.

### 12.2 Stage 1 — the one-liners (M1–M6)

Each independently gated so effects are attributable.

**Questions answered:** does hard contact beat soft? does the surface-gap fix help every particle
mode? does the side-flip guard rescue `penalty` and `pg2p_pos`? does **teleport + f_feedback** fix
the volume bug while keeping the geometry win?

**Acceptance:** each flag OFF must reproduce the Stage-0 baseline to within the measured noise
floor (§12.6).

### 12.3 Stage 2 — the forecast ladder (N1, N2)

`fluidlab` → `+force` → `+grid projection` (= SoftMAC). Measure cost at each rung. Test both
scatter weightings (§11.1).

**Acceptance:** flag-OFF bit-identity; and rung 2 should reduce penetration **without** the
rebound signature that plagues `penalty`.

### 12.4 Stage 3 — the cost–accuracy frontier (N4)

- **VRAM probe first** (cheapest experiment available; gates everything).
- Baseline `grid` at res **8 / 10 / 12 / 14** ⇒ the reference curve.
- Every candidate at res 10.
- **⭐ Also sweep `AGF_PPC_DIVISOR` (2.0 → 3.0 → 4.0) at res 10** — the competing hypothesis that
  more particles per cell, not a better contact operator, is the cheap accuracy win (§6.9). If this
  wins, it is a one-env-var result and the whole contact programme is much less interesting. **Test
  it early enough to matter.**
- **A method earns its place only if it lands above that curve** — reaching res-12/14 accuracy at
  res-10 cost. This is the literal statement of the objective (§1.2).

**Cost model [INFERRED]:**

| option | multiplier | reasoning |
|---|---|---|
| grid res 10 → 14 | **≈3.8×** | cells ×(14/10)³ = 2.74; **particles scale too** because `particle_size = dx/PPC_DIVISOR`; and CFL forces `dt` down ×1.4 |
| `PPC_DIVISOR` 2 → 3 at res 10 | **≈3.4×** particle work, **grid unchanged** | particles ×(3/2)³ = 3.375; grid ops and `dt` unaffected |
| forecast at res 10 | **≈1.1×** | SoftMAC measured +7%; **ours must be measured, not assumed** |

⇒ roughly **3.5× cost headroom** for the forecast against grid refinement. Note the PPC lever is
*not* obviously cheaper than refinement — it trades grid cost for particle cost — which is exactly
why it needs measuring rather than assuming.

⚠️ **You cannot refine the grid without also refining the particles** unless you deliberately raise
`AGF_PPC_DIVISOR`'s inverse effect. Any "grid-only" refinement claim must state what happened to
particle count.

⚠️ **The refinement baseline must itself be run with a *converged* contact regularization**
(M1 `soft_scaled` or `hard`), otherwise the reference curve is contaminated by the same
non-convergence we are trying to measure (§6.9).

### 12.5 Stage 4 — analysis and render (N5)

Full-sequence behavioural profiles over all metrics for **all** arms including failures. The
analyzing agent selects what is most instructive to render (§1.4). Render columns should include
at least one informative failure, not only good arms.

### 12.6 Verification methodology (how to know a change is inert)

This project has a working pattern, worth reusing:
- **Positive control**: `ctl_none` — a mechanism that *should* read badly. Validates instruments.
- **Flag-off equivalence**: the runtime-switch port was verified inert at **2.7e-3 / 9.2e-3 mm** —
  inside the within-group spread at both hits compared.
- **Noise floor from a full pairwise matrix, never a single pair.** `rt4_equiv_check.py` used one
  banked pair and reported a false "5.20×" alarm; `rt5_pairwise.py` showed the run in question was
  an outlier. **One-sample noise floors lie.**
- **Fresh-bar guard**: `batch_arms.py` checks finiteness / particle count / extent after each reset
  and **aborts** rather than let a blown-up arm contaminate later ones.
- **`p0_verify_unchanged.py`** is the standing acceptance test for "did instrumentation change
  production behaviour?"

### 12.7 Confound to settle alongside

**CFL 0.45 vs SoftMAC's stated 0.3 threshold** (§8.5). Sits underneath every comparison ever made.
Related prior finding on record: `substep_dt` is derived from the **bar** wave speed `√(E/ρ)` while
the **P-wave is ~21% faster**, so the sim runs ~1.09× the limit that actually governs stability,
~2% below a measured cliff. A `cfl_use_pwave` option exists (default OFF).

---

## 13. Analysis tooling inventory

All in `agforge/analysis/`, all untracked. Docstring first lines, verified 2026-08-07.

| script | purpose |
|---|---|
| `batch_arms.py` | **the sweep driver** — every arm in ONE process, sharing a single scene build. `--n-hits --reps --arms --out` |
| `trajectory.py` | **per-hit trajectory of every metric, not just final state.** `--selfcheck` verified identical to `geom_metrics.occupancy` (diff 0.00e+00) |
| `score_batch.py` | score a batched sweep: packing, det F, force, penetration |
| `score_arms.py` | score arms at a FIXED hit index |
| `geom_batch.py` | geometry of the batched sweep vs the real scans |
| `geom_metrics.py` | general geometry accuracy vs real scans ⚠️ **sets `RLIMIT_AS = 3 GB` at import — kills VTK** (§18) |
| `force_vs_real.py` | first-ever comparison of simulated force against the real machine's F |
| `render_arms.py` | animated side-by-side comparison vs the real scan ⚠️ **camera-lag bug at :308** |
| `build_artifact.py` | self-contained interactive review page |
| `extract_real_meshes.py` | one-time: real per-hit meshes from the 319 MB `.pt` into compact `.npz` |
| `p0_verify_unchanged.py` | **acceptance test**: did instrumentation change production behaviour? |
| `rt4_equiv_check.py` | did the runtime-switchable port change behaviour? ⚠️ single-pair floor — prefer `rt5` |
| `rt5_pairwise.py` | full pairwise distance matrix — **the correct noise-floor tool** |
| `recon_converge.py` | is the −12/−15% volume drop real or a surface-reconstruction artifact? |
| `recon_kernel.py` | how much of "geometric volume loss" is a modelling choice? |
| `piling_where2.py` | WHERE does the interpenetration happen? |
| `piling_vs_dt.py` | is particle piling a TIME-integration or SPATIAL artifact? |
| `startup_probe.py` | decompose the ~197 s per-run overhead |

---

## 14. Environment variable reference

| variable | used for | note |
|---|---|---|
| `AGF_MPM_X_PAD_LOWER` | domain length multiplier | 🚨 **default 0.85 = the bug. Always set 1.3.** |
| `AGF_CONTACT_RUNTIME_SWITCH` | enable live contact-mode switching | required by `batch_arms.py`; **forces CPIC off** (§6.4) |
| `AGF_DIAG_PENETRATION` | enable the penetration probe | writes `pen_*` into `.diag.jsonl` |
| `AGF_CELLS_PER_DIAMETER` | grid resolution | ⚠️ **defaults to 7**; every run passes **10**. 14 is the refinement target |
| **`AGF_PPC_DIVISOR`** | `particle_size = dx / this` | **default 2.0. Never varied.** Refines particles *without* refining the grid — an unexploited sub-grid axis (§6.9, §12.4) |
| `AGF_CFL_SAFETY` | CFL safety factor | **0.45 currently; SoftMAC says 0.3** (§12.7) |
| `AGF_ENABLE_CPIC` | CPIC on/off | 0 in every run so far |
| `UV_HTTP_TIMEOUT` | pixi/uv download timeout | ⚠️ **does NOT propagate through `pixi run` — use `--frozen`** |
| `PYTHONPATH=forge_common/main` | resolve `forge_common` | required |

---

## 15. Out of scope, and why

Recorded so nobody re-derives these from the literature and proposes them again.

| approach | why not |
|---|---|
| Implicit MPM / backward-Euler | full solver rewrite; violates §1.3 |
| Augmented Lagrangian, convex cone programming (CP-MPM) | needs a global solve per step; rewrite |
| Barrier / IPC / BFEMP | needs implicit integration + a continuous surface representation; rewrite |
| Full CPDI / GIMP domain tracking | changes the basis functions everywhere; large. *(A "CPDI-lite" — sampling the SDF at a few corners of the F-deformed particle domain — is a plausible smaller variant and stays on the table as a stretch item.)* |
| Mortar segment-to-segment contact | requires an explicit meshed contact surface we do not have |
| Global grid refinement as the *solution* | it is the **baseline we must beat**, not a candidate (§1.2) |

**These are correct methods.** They are excluded on engineering grounds, not physical ones. If the
project's constraints ever change, §8.6 is the entry point.

---

## 16. Retracted claims — both views kept deliberately

| claim | status |
|---|---|
| "grid-alone is more stable (7/7 vs 4/7)" | **[RETRACTED]** DOMAIN-LIMITED. Post-fix n=3 reverses it (§5.2). |
| "penetration grows hit over hit" | **[RETRACTED]** it **plateaus** (§7.4). A 3-point monotone prefix was read as a trend. |
| "η noise floor is 0.16 pp" | **[RETRACTED]** same-cloud repeats span 0.005–0.16 pp; 0.16 was the control's real drift. |
| "production failed at hit 8 — the port broke it" | **[RETRACTED]** n=3 gives `17, 8, 17`. Port fine, bad seed. |
| "equivalence off by 5.20×" | **[RETRACTED]** single-pair noise floor; run `N1` is an outlier. **One-sample noise floors lie.** |
| "the teleport arm has pose drift" | **[RETRACTED]** did not reproduce; centroid alignment hurts both arms. |
| "h2 is the best hybrid" | **[RETRACTED]** g1/h2/h3 indistinguishable in every window; n=1 noise on a 0.3 mm difference. |
| "grid + fluidlab needs ZERO code change and is a free arm" | **[RETRACTED]** modes are a single mutually-exclusive scalar (§6.3). Not expressible. |
| "force double-counting is the constraint on hybrids" | **[RETRACTED]** the reaction is already single-counted by `mass × weight` (§6.1). The real issue is **double application of the velocity constraint**. |
| "enforce the constraint at the grid, informed by particle sampling" | **[RETRACTED]** defeats the project's purpose (§1.2). Superseded by the forecast pattern (§9). |
| "splitting the mode enum is required and invasive" | **[RETRACTED]** use independent flags (§9.2). |
| "CPIC is the best choice for this setup" (AI research run) | **[RETRACTED]** contradicted by CPIC's own paper (§8.1). |

---

## 17. Risk register

| risk | likelihood | impact | mitigation |
|---|---|---|---|
| Forecast doesn't transfer from liquid to forging plasticity | medium | high — it's the headline candidate | ladder rungs are independently valuable; Stage 1 has value regardless |
| res 14 doesn't fit in 6 GB VRAM | **medium-high** | high — blocks Stage 3 | measure early (cheapest experiment); cloud GPU is being explored separately |
| GPU contention with workstream B corrupts or OOMs a long run | medium | medium — lost time | coordinate; `batch_arms` has a fresh-bar abort guard |
| The two setup errors (§4) dominate contact differences | **unknown** | high — could invalidate rankings | quantify their effect once, early |
| **`AGF_PPC_DIVISOR` turns out to be the cheap win** | **unknown, untested** | high — would make most of this programme uninteresting | test it in Stage 3, or earlier. **A one-env-var result beating a bespoke contact operator is a good outcome for the project even though it is a bad outcome for the plan.** |
| The die-tip-vs-`dx` ratio (§6.9) is wrong | low-medium | **high — it is the premise** | verify against the physics die geometry |
| CFL 0.45 is above the stability threshold | medium | high — confounds everything | settle in Stage 1 (§12.7) |
| `outputs/` (5.1 GB) lost to disk failure | low | medium — re-run cost ~hours | currently unmitigated; single disk, gitignored |
| Analysis re-derives conclusions from DOMAIN-LIMITED files by accident | **medium** | high | §2.3 table; tag every number |

---

## 18. Operational traps (each cost real time)

| trap | detail |
|---|---|
| **`AGF_MPM_X_PAD_LOWER` defaults to 0.85** | forget it ⇒ the domain bug returns **silently** (§5.1) |
| **pixi "Failed to update PyPI packages"** | a **network timeout**, not code/disk. `export UV_HTTP_TIMEOUT=600` **does NOT propagate** through `pixi run` — verified: uv still reported a 30 s timeout with the var exported. **Use `pixi run --frozen`.** ⚠️ **[INFERRED, NOT VERIFIED]** — `--frozen` is what workstream B uses successfully, but **no sweep in this workstream has ever completed with it.** It is the first thing Stage 0 depends on. If Stage 0 still fails here, that is the known failure point, not a new mystery — and it is the one place to spend diagnosis time rather than retrying. |
| **GPU fallback needs a LOGIN shell** | `wsl.exe -e bash script.sh` silently runs on **CPU** (torch still reports CUDA available). Use `bash -lc`. Tell: ~4 s/press vs ~37 s. |
| **`pgrep -f <pattern>` self-matches** | it matches its own command line ⇒ false "RUNNING". Use `ps -eo pid,cmd \| grep … \| grep -v grep`. Produced a false green light during this work. |
| **importing `geom_metrics` kills VTK** | sets `RLIMIT_AS = 3 GB` at import; VTK's software GL reserves far more *virtual* AS ⇒ exit 1, **no traceback** |
| **pyvista `shape=(3,4)` Plotter dies** | under this VM's software GL. Use one reused single-view Plotter + PIL compositing. |
| **camera lag in `render_arms.py`** | assigning `pl.camera.position/up/focal_point` attribute-by-attribute does not commit before `screenshot` ⇒ each tile renders **one view behind**. **Still present at `render_arms.py:308`.** Fix: `pl.camera_position = [pos, focal, up]` then `pl.render()`. **All shipped renders and the published artifact have mislabelled rows.** Diagnosis is [INFERRED] — never confirmed by a fixed re-render. |
| **framing must be FIXED, not auto-fit** | otherwise a stalled bar looks identical to an elongating one |
| **stale VTK camera clipping range** | renders the real surface as a bare silhouette from some angles; set `cam.clipping_range` explicitly |
| **`df` inside WSL lies** | reports the sparse vhdx's virtual size. Check `Get-PSDrive C` on Windows. |
| **never write into `\\wsl.localhost` from Windows** | zero-pads files to 4096-byte boundaries — silent corruption. Stage on Windows, copy in from `/mnt/c` inside WSL, verify with `diff`. |
| **MCP browser tools write to the PersonalLifeStuff root** | that is the server's CWD. Move outputs into the Genesis tree. |

### Standard run command
```bash
cd ~/GitHub/Genesis && export PATH="$HOME/.pixi/bin:$PATH" \
  && export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH \
  && AGF_MPM_X_PAD_LOWER=1.3 AGF_CONTACT_RUNTIME_SWITCH=1 AGF_DIAG_PENETRATION=1 \
     AGF_CELLS_PER_DIAMETER=10 AGF_CFL_SAFETY=0.45 AGF_ENABLE_CPIC=0 \
     PYTHONPATH=forge_common/main \
     pixi run --manifest-path aims-genesis/nsf-demo/pixi.toml --frozen \
     python -u aims-genesis/nsf-demo/agforge/analysis/batch_arms.py \
       --n-hits 17 --reps 3 --out <dir>
```
Run from a **login shell** (`bash -lc`). Long runs: `setsid nohup … > ~/log 2>&1 < /dev/null &`.

### Measured timings [MEASURED, DOMAIN-FIXED]
- one arm, 17 hits, completing: **~57 s**
- scene build (once per batch): **~5.6 min**
- JIT (once, absorbed into the first arm): **~4 min**
- ⇒ `T ≈ 9.6 min + N_runs × 57 s`. A 21-arm × n=3 matrix ≈ **70 min**.
- Batching all arms into ONE process is the only real lever: 12 arms × 3 hits = **7 min** batched
  vs ~67 min as separate processes. Reps are **rep-major** (all arms at r1, then r2…).

---

## 19. Provenance of the existing methods

**Most of the particle methods here were designed by previous AI agents/models** from their own
reasoning and research across earlier sessions. They carry correspondingly uncertain authority.

**The exception is `fluidlab`**, derived from the FluidLab paper/repo — human researchers
(MIT/CMU/Dartmouth et al.). The FluidLab repo also offered a hybrid mode enabling both grid and
particle collisions, a **human-designed precedent for what this project attempts**. That does not
make it correct — §8.4 shows its smoothing is motivated by differentiability and its coupling is
one-way — but it is a different and useful kind of evidence.

**Practical implication:** treat prior "unusable" verdicts (notably on `fluidlab` and `penalty`) as
**hypotheses to retest, not exclusions.** They were reached under the domain bug, with the teleport
un-gated, and before the CPIC/force confounds were understood.

---

## 20. Open questions and unverified claims

1. **Volume / det F / GAP / force have never been recomputed post-domain-fix.** The workstream's
   headline claim ("removing the teleport halves packing loss") is **DOMAIN-LIMITED and unverified**.
2. **Peak VRAM at res 10 and 14 is unmeasured** on a 6 GB card. Gates Stage 3.
3. **The over-application mechanism (§6.1) is INFERRED, not measured.** Prediction: damage occurs
   where the two sample points disagree on the normal. Testable.
4. **The P2G null-space "relax back into the die" prediction (§7.2) is untested.**
5. **The camera-lag diagnosis is inferred**, never confirmed by a fixed re-render.
6. **SoftMAC's +7% cost does not transfer automatically.** Must be measured on 9,266 particles.
7. **SoftMAC is validated on liquid in thin containers**, not 17-blow forging with plasticity.
8. **CFL 0.45 vs 0.3** — unresolved confound under every comparison.
9. **Adjoint is unwired** for `mpm_penalty_contact` and `mpm_postg2p_contact` ⇒ silently wrong
   gradients on the differentiable path. (Irrelevant to forward accuracy; relevant if anyone ever
   optimizes through this.)
10. **The two standing setup errors** (§4) bound every arm regardless of contact method.
11. **Residual under-elongation post-fix:** 83.5 vs 93.4 mm at hit 17 (89%) — a genuine modelling
    gap, cause unknown. May be material model, may be contact, may be the setup errors.
12. **`postg2p_position` has only ever been tested double-projecting** (it duplicates the teleport).
13. **Unexplained ~8 µm group offset** between banked and post-port runs at hit 1 (~0.02% of bar
    diameter). Below any threshold that matters, but unexplained.
14. **`jc_C` is dead code and JC's `A` cannot be fitted** (material-model side, from the sibling
    workstream) — noted because material and contact effects are entangled in the geometry metrics.
15. **`AGF_PPC_DIVISOR` has never been varied.** Particles-per-cell may be a cheaper accuracy lever
    than any contact operator. **Untested competing hypothesis** (§6.9, §12.4).
16. **The die-tip width of ~2.4 cells is inherited and unverified** (§6.9) — and the entire
    "under-resolved geometry" premise rests on it.
17. **The `influence > 0.1` gate is unexamined.** It is a magic threshold that, combined with a
    fixed `coup_softness`, sets the effective contact band to 0.29 dx. Nobody has asked whether 0.1
    is right or where it came from.

---

## 21. Decisions left to the user

- Whether to **commit** the 1,404 uncommitted lines. *(This document is staged separately.)*
- Whether to flip the **`AGF_MPM_X_PAD_LOWER` default** (affects the shared tree / workstream B).
- Whether to **regenerate and republish** the review artifact after fixing the camera bug.
- Whether to build the **four grid+particle-mode hybrids** originally requested. *Current
  recommendation: no* — the motivation is superseded by §9, and `h1/h2/h3/h5` already sample that
  design space. **This is a recommendation overriding an earlier explicit request; it needs the
  user's agreement.**
- **GPU scheduling vs workstream B.**
- The two standing setup errors (§4).
- Whether to back up `outputs/` (5.1 GB, single disk, no copy).

---

## 22. Quickstart for a new implementer

1. Read §1.2 (the objective — especially what *not* to propose), §9 (the answer), §16 (what's
   already been wrong).
2. `cd ~/GitHub/Genesis/aims-genesis/nsf-demo && git status` — expect ~1,404 uncommitted insertions
   and untracked `agforge/analysis/`. That is the correct state.
3. Check nobody else is on the GPU: `nvidia-smi` and `ps -eo pid,cmd | grep pixi | grep -v grep`.
4. Run Stage 0 (§12.1) using the command in §18. **Do not omit `AGF_MPM_X_PAD_LOWER=1.3`.**
5. Score it: `trajectory.py` for per-hit, `score_batch.py` for volume/force/penetration.
6. Only then start Stage 1.

**If you read nothing else:** the modes are mutually exclusive so add flags not modes (§6.3/§9.2);
the domain env var default is the bug (§5.1); tag every number with its config (§0).

---

## 23. References

| ref | what |
|---|---|
| Hu et al. 2018, *MLS-MPM with Displacement Discontinuity and Two-Way Rigid Body Coupling*, SIGGRAPH — 10.1145/3197517.3201293 | CPIC; the basis of Genesis's MPM |
| Xian et al. 2023, *FluidLab*, ICLR — arXiv 2303.02346 | the `fluidlab` mode's origin; the `min(exp(−d),1)` blend |
| **Liu et al. 2023, *SoftMAC*, arXiv 2312.03297** | **the forecast-based contact model — the key reference** |
| Oxford/Warwick/NTUA 2024, arXiv 2403.13534 | penalty point-to-segment + Extended B-Splines; "premature contact" |
| Durham et al. 2024, arXiv 2412.01565 | large-deformation material-point-to-rigid contact; GIMP corners |
| CK-MPM, arXiv 2412.10399 | compact kernel ⇒ narrower smearing at fixed resolution |
| OS-MPM, arXiv 2605.09097 | local space-time refinement |
| BFEMP, arXiv 2108.03349 | barrier contact |
| Convex MPM-rigid, arXiv 2403.13783 / 2503.05046 | convex frictional formulations |
| ManiSkill2, arXiv 2302.04659 | two-way MPM-rigid, particle penalization |
| Bardenhagen, Guilkey, Nairn | classical multi-velocity-field nodal contact |

### On the AI research runs used while writing this

Two Perplexity runs informed §7–§9: one **Gemini 3.1 Pro Thinking** answer on our exact
configuration, and one **Deep Research** survey citing 59 sources. They live in the user's
Perplexity account history and are reachable through browser access; local copies sit untracked at
`agforge/docs/research/` on the authoring machine.

**They are deliberately NOT committed**, for four reasons:

1. **Fully extracted.** Every substantive point is already in this document — and quoted against
   the **primary paper** wherever one exists, rather than against the AI summary.
2. **No quantitative content.** The Deep Research output contains no measured numbers at all; it is
   a qualitative survey. Every hard figure here comes from a primary paper or our own runs.
3. **Its emphasis points at out-of-scope methods.** Mention counts: barrier 18, IPC 16, mortar 13,
   augmented Lagrangian 8, convex cone 6, BFEMP 6 — all §15 exclusions. Committing it into the repo
   would invite a future reader to re-propose implicit MPM, which §15 exists to prevent.
4. **One of them contains a known error** — the claim that CPIC is the best choice for this setup,
   contradicted by CPIC's own stated limitation (§8.1). Checked-in AI output acquires the
   appearance of a reference it has not earned. This project has already been bitten once by a deep
   research run making a hallucinated recommendation its headline.

**If you need them, regenerate them** — the prompts are reconstructable from §7–§9, and the primary
sources below are the actual evidence.

**Companion docs in this directory:** `Volume_Conservation_And_Contact.md` (pre-domain-fix; carries
a staleness banner), `MPM_Stabilization_Architecture.md`, `Hot_Forging_Thermal_Architecture.md`,
`Thermal_Physics_Roadmap.md`, `Surface_Reconstruction_Architecture.md`.

---

## 24. Maintaining this document

**This document is meant to be edited as the work proceeds, not frozen.** It exists so that
context loss — a new session, a new agent, a new person — costs hours instead of days.

**When you learn something, update it here, not only in a chat or a commit message.** In
particular:

- **A claim turns out wrong** ⇒ move it to §16 with **both** the old and new view. Do not silently
  delete it. The retracted list has repeatedly been more useful than the confirmed one, because it
  shows *how* this project goes wrong (n=1 noise, untagged configs, inherited confidence).
- **A number is measured** ⇒ record it **with its config tag** and the file it came from.
- **A stage completes** ⇒ update §12 with what was actually run and what it showed, and revise §10
  (status column) and §20 (open questions).
- **A trap bites you** ⇒ add it to §18. Every entry there cost someone real time.
- **A new lever is discovered** ⇒ §14 (env vars) and, if it competes with the main programme, §12.4.

**Known weak points in this draft**, flagged honestly for whoever revises it:
- The die-tip width (~2.4 cells) is **inherited and unverified** (§6.9) yet underpins the entire
  "under-resolved" premise.
- The cost model (§12.4) is arithmetic, not measurement. N3 exists to replace it.
- §7.1's channel hierarchy is **[INFERRED]**. It explains the observations well, which is not the
  same as being confirmed.
- Nothing in §12 has been run. **Every experimental claim here is a prediction.**

### Changelog

| date | change |
|---|---|
| 2026-08-07 | First draft (847 lines): state, findings, code walkthrough, theory, literature, the merge answer, inventory, staged plan, retractions, traps. |
| 2026-08-07 | Second pass (1,208 lines): added glossary + metric definitions + arm naming, validation targets, implementation specs (forecast kernel with derivation, side-flip port gotchas), analysis-tooling inventory, env-var reference, out-of-scope rationale, risk register, verification methodology, quickstart. |
| 2026-08-07 | Offboarding pass (post-commit `ef550706`): fixed an internal contradiction in §12.1 (Stage 0 was headed "no physics changes" while listing a CPIC arm that requires the Stage-1 change M5); added the **line-number drift hazard** banner to §6 — this doc is committed while the code it cites is not; downgraded the `--frozen` fix in §18 to **[INFERRED, NOT VERIFIED]** since no sweep has ever completed with it; flagged that `p2_fluidlab`/`p4_pg2p_vel` have never executed at all. |
| 2026-08-07 | Final pass: decided **not** to commit the AI research dumps (§23) — fully extracted, no quantitative content, emphasis skewed toward §15 exclusions, and one contains a known error. Quotes re-anchored to primary papers so §8 stands alone. |
| 2026-08-07 | Third pass: **corrected a cross-config error in §7.4** (penetration series and frame counts had been taken from different runs); replaced it with a self-consistent DOMAIN-FIXED table. Added `AGF_PPC_DIVISOR` as an **unexploited sub-grid axis and competing hypothesis** (§6.9, §12.4); flagged `AGF_CELLS_PER_DIAMETER` default 7 ≠ the 10 we run; added run structure (§3.4) and how-to-add-an-arm (§3.5); added a copy-pasteable Stage 0 command; tagged the die-tip width as unverified; added this section. |
