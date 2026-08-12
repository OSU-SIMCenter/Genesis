# Thermomechanical Coupling and Time Scaling — Research, Findings, and Plan

**Status:** living document. Canonical reference for the coupled thermal-mechanical work on
`agforge/v2/thermal-st-invariance`. Written 2026-08-12 from session `dbc3b95f`
(← `2c4556e5` ← `156eb841`). Repo state at writing: `cb3765a3`, clean.

**Read this before touching the thermal solver, the time-scaling knobs, or the geometry
metric.** Sections 5 and 11 exist specifically to stop work being redone or wrong conclusions
being re-derived — several claims in this project have been confidently wrong, including
claims made by the author of this document.

**Companion documents**
- `docs/316L_MECHANICAL_PROPERTIES.md` — the material card, its sources, and its caveats.
- `aims-genesis/nsf-demo/docs/Contact_Method_Research_And_Plan.md` — the contact workstream's
  equivalent canonical reference. Read it before assuming anything about contact.
- `docs/AS_BUILT_AGILITY_FORGE.md` — the real machine.

---

## 1. Goal and scope

**Goal.** Make the Genesis MPM forging simulation reproduce the real Agility Forge as closely
as possible for 316L, with *fully coupled dynamic thermomechanical physics* — not a prescribed
temperature field. Temperature must evolve, and must feed back into flow stress.

**Constraint.** The mechanical solve is explicit MPM. A forging sequence spans tens of seconds
of real time; the explicit stability limit is under a microsecond. Some form of scaling is
unavoidable. The question this document answers is *which scaling, applied where, and how do we
know it is right*.

**Explicitly in scope:** the thermal solver, the particle↔grid transfer for temperature, the
time-scaling architecture, strain-rate handling in the constitutive law, and the metrics that
validate all of it.

**Explicitly out of scope for this workstream (owned elsewhere):**
- Billet/stock geometry and `forge_common/real_scale.py` → workstream A-7.
- Induction heating calibration → session B-3.
- Contact method selection → contact workstream (nsf-demo).

---

## 2. The system as it stands

### 2.1 Substep sequence

Per substep, in order (`base_mpm_solver.py`, `legacy_coupler.py`):

1. **p2g** (`substep_pre_coupling`, :837) — particle → grid. Includes
   `p2g_post_constitutive` (:492) which applies **adiabatic plastic heating**
   `dT = χ·σ_y·Δγ / (ρ·Cp)` with `χ = 0.9`, and `p2g_induction` (:551).
2. **couple** (`simulator.py:342`) — grid thermal diffusion, then
   `mpm_grid_thermal_surface_flux` (`legacy_coupler.py:~1093`): convection and radiation on
   surface cells, plus the fixed-end bulk BC.
3. **g2p** (`substep_post_coupling`, :864) — grid → particle, including
   `g2p_apply_thermal_from_grid` (:723), the FLIP/PIC thermal blend.
4. **`apply_particle_contact`** (:878, defined :1982) — die-contact heat exchange, written
   directly to `particles[f+1].temp` **after** the g2p blend.

> ⚠️ Step 4 running *after* step 3 is load-bearing and was mis-read once during investigation.
> A pure-PIC gather in step 3 does **not** discard the contact exchange.

### 2.2 Where each scaling knob enters

| Knob | Value | Where set | What it multiplies |
|---|---|---|---|
| `pressing_speed` | 25.0 m/s | `options.py:766` | die velocity during the blow |
| `real_die_speed` | 0.0141 m/s | `options.py:776` | the physical reference |
| **N (mechanical accel.)** | **1,773** | derived, `options.py:652` | = `pressing_speed / real_die_speed` |
| `thermal_time_scale` (S_T) | **237,014** default | `options.py:616` | `dt_th = substep_dt · S_T` — every transport term |
| `thermal_time_scale_mode` | `'cfl'` default | `options.py:516` | `'mechanical'` sets S_T = N = 1,773 |
| `thermal_cfl_fraction` | 0.25 | `options.py:614` | fraction of diffusion CFL used for S_T |
| `arrhenius_process_strain_rate` | 0.41 /s (opt-in) | `MaterialOptions` | overrides the rate fed to the flow law |
| `_thermal_flip_frac` | 0.97 | `base_mpm_solver.py:239` | FLIP/PIC blend for temperature |
| `approach_speed` | **273.2 m/s** | derived, `options.py:940` | `0.35 · (dx / sim.dt)` — see §7 |

### 2.3 Resolved constants (measured, not recalled)

```
base_grid_density   183.0        →  dx           = 5.464 mm
sim.dt              7.000679e-6 s   substeps     = 8
substep_dt          8.750849e-7 s
E = 121.5 GPa, ρ = 7334 kg/m³   →  c = √(E/ρ)   = 4070 m/s
α (runtime)         1.1e-5 m²/s     α_worst (design, 316L) = 6.05e-6 m²/s
h_air 15   h_contact 3000   emissivity 0.40   Cp(1273 K) ≈ 625.5 J/kg·K
grid_cells_across_billet 7 → 3,160 particles;  10 → 9,266 particles
jc_T_ref 1273.15 K   T_melt 1675.0 K   Arrhenius clamp [1073.15, 1473.15] K
```

---

## 3. Validation data — what exists and what each can prove

| Dataset | Contents | Location | Validates |
|---|---|---|---|
| **17-hit per-hit meshes** | billet triangle meshes before/after each of 17 hits, ~100–137k verts, mm | `forge_common/main/outputs/real_meshes/hit_NN.npz`, keys `V_before`/`T_before`/`V_after`/`T_after` | **mechanical deformation** |
| **07-17 mcap** | thermal camera on the **induction coil**, heating curve | `~/GitHub/Genesis/forge-data/20260717_135009.mcap` (1.15 GB, local) | **induction heating** (B-3) |
| **06-15 T4 bulk mcap** | press force/position telemetry **and** thermal camera on the **press** | pCloud `AgilityForge/2026-06-15_T4_bulk/` (8.58 GB) — **not local** | **force**, **billet temperature during forging**, **cooling rate** |

### 3.1 The critical property of the 17-hit set

**The 17-hit sequence was run with the billet reheated to ~900–1000 °C between hits, with
reasonably fast hits.** It is therefore a *controlled-temperature* experiment — deliberately
close to isothermal — and carries **no temperature ground truth**.

Two consequences, both important:

1. It **cannot** validate thermal evolution directly.
2. But it is a **strong test of the thermal model in the other direction**: a correct coupled
   simulation, given the same reheat boundary conditions, should *predict* near-isothermal
   behaviour. If our coupled run predicts large cooling over the sequence, either our boundary
   conditions or our understanding of the experiment is wrong. **Absence of temperature change
   is the signal.**

⇒ Every validation case should be run **twice**: once isothermal at ~900–1000 °C, once fully
coupled. Both scored on the geometry metrics we already have.

> 🚩 **Do not generalise between the two mcap datasets.** The 07-17 camera watches the **coil**;
> the 06-15 camera watches the **press**. This project has already made that error more than
> once, in both directions. The ~960 °C billet measurement and the ~4.8 °C/s cooling rate are
> **06-15 properties** and say nothing about the 17-hit geometry sequence.

### 3.2 Retrieval note

`forge-data/WHERE-IS-THE-DATA.md` documents pCloud retrieval. As of 2026-08-12 the C: drive has
~39.6 GB free, so the 8.58 GB pull is feasible.

> ⚠️ Do **not** let a Windows-side tool write into `\\wsl.localhost\` — the 9P bridge zero-pads
> to 4096-byte boundaries and corrupts silently. Stage on Windows, verify with
> `rclone check --checksum`, copy in from `/mnt/c` *inside* WSL, re-hash.

---

## 4. Established findings

Each item states how it was measured so it can be re-run.

### 4.1 🎯 The thermal instability is the FLIP gather

`g2p_apply_thermal_from_grid` blends
`T_new = flip·(T_particle + Σw·dT_*) + (1−flip)·T_pic`, with `flip = 0.97`. FLIP carries
*increments* with only a 3%-per-step pull toward the grid's absolute field. That is correct for
advection and wrong for a diffusion (parabolic) field.

**Measured.** Johnson-Cook, res 7, thermal fully **ON**, 17 hits requested. `_rt_thermal_flip_frac`
is a **runtime field**, so this needed no source change — poke it after the scene builds.

| Billet | Approach | FLIP | Hits/17 | Failure |
|---|---|---|---|---|
| 1273.15 K | 273 m/s | 0.97 | **1** | Thermal Detonation |
| 1273.15 K | 273 m/s | **0.00** | **12** | Supersonic Velocity |
| 1273.15 K | 35 m/s | 0.97 | **1** | Thermal Detonation |
| 1273.15 K | 35 m/s | **0.00** | **9** | Supersonic Velocity |
| 1273.15 K | 273 m/s | 0.50 | **1** | Thermal Detonation |
| 1200 K | 273 m/s | **0.00** | **13** | Supersonic Velocity |

**Even 0.5 fails — there is no usable FLIP margin.** With the gather fixed, the failure mode
reverts to the ordinary mechanical instability seen everywhere else.

**Why this specifically blocks coupling.** From Nairn, *Coupling Transport Equations to
Mechanics in the Material Point Method*:

> "FLIP methods get excellent grid values, but particle values develop very large oscillations
> near edges, interfaces, or other discontinuities. **This behavior means modeling cannot
> implement features that depend on a particle's transport value (e.g., temperature-dependent
> material properties).**"

Temperature-dependent flow stress **is** the thermomechanical coupling. The published failure
mode of the scheme in use is exactly the capability we want. Nairn also notes null-space effects
are *more severe in transport than in mechanics*, which is why momentum survives and thermal
does not.

**The published fix is neither FLIP nor pure PIC.** PIC (= FMPM(1)) removes oscillation but is
*"unacceptably diffusive and must be rejected"*; **FMPM(k≥2–3)** — an approximate full
heat-capacity-matrix inverse — removes oscillation without significant numerical diffusion.
Note the codebase already uses **APIC** (a filtering transfer with the same motivation) for
momentum, and left thermal on raw 1988-era FLIP.

⇒ **Pure PIC (`flip = 0`) is a diagnostic, not the recommendation.**

### 4.2 🎯 It is not a "hot billet" problem — and it is unconditional

Temperature sweep, JC / res 7 / thermal ON, configs verified by read-back:

| Billet | Hits/17 | Note |
|---|---|---|
| **293 K** | **7** | = ambient 293.15 → all surface-flux driving terms ≈ 0 |
| 600 K | **1** | barely warm; dies like 1273 K |
| 1200 K | **1** | below `jc_T_ref`; JC thermal softening clamped fully off |
| 1273.15 K | **1** | — |

600 K is nowhere near forging heat. The only thing special about 293 K is that it *equals
ambient*, making the thermal solver effectively inert. **The failure occurs whenever the thermal
solver does anything at all.**

**And it is unconditional, not CFL-limited.** `thermal_time_scale_mode='mechanical'` drops S_T
from 237,014 → 1,773 (134×), and survival did not change (`t_1000C_mech` still died at hit 2) —
impossible for a conditionally-stable scheme, since every transport term scales with `dt_th`.
**Do not attempt to fix this by tuning S_T, `thermal_cfl_fraction`, or the timestep.**

### 4.3 CFL audit — all three explicit limits

Computed at 1273 K, res 7 (`~/cfl_audit.py`):

| S_T | `k_diff` | `k_conv` | `k_rad` |
|---|---|---|---|
| 237,014 (shipped default) | **0.4584** | 0.0001 | 0.0005 |
| 1,773 (mechanical) | 0.0034 | ~0 | ~0 |

Surface terms are ~1000× *slacker* than diffusion, not tighter — a cell's thermal mass
`ρ·dx³·Cp` dwarfs single-face flux at `h_air = 15`, `h_rad ≈ 60.6`. Contact `θ ≈ 0.036`.

🚩 **A real adjacent defect, unclaimed:** the solver's runtime `default_thermal_diffusivity`
is **1.1e-5 m²/s** while `options.py` sizes S_T from `alpha_worst = 6.05e-6`. The runtime α is
**1.82× the design α**, so `k_diff` lands at **0.458 against an intended `thermal_cfl_fraction`
of 0.25** — most of the margin gone, against a code comment warning of FTCS checkerboards above
~0.6–0.7.

### 4.4 Disabling thermal improves *accuracy*, not just stability

Same resolution, material and temperature; only `enable_thermal` differs:

| Arm | IoU@2.0 | dev_mean | dev_max |
|---|---|---|---|
| `t_1000C_res10` (thermal on) | 0.7668 | 0.320 | 2.200 |
| `t_1000C_nothermal` (thermal off) | **0.7887** | **0.297** | **1.278** |

+0.0219 IoU (~100× the noise floor) and **dev_max −42%**. The thermal solver was actively
degrading the answer.

### 4.5 The geometry metric — what it can and cannot compare

Measured by filling the **real** mesh with a perfect cubic lattice — a sim with zero shape error
— and running the actual metric on it (`~/null_floor.py`, `~/null_iou.py`).

| Comparison | Trustworthy | Artifact |
|---|---|---|
| **IoU@2.0, same resolution** | ✅ | run-to-run noise **0.0002** |
| IoU@2.0, across resolutions | ❌ | mean artifact 0.0005, but lattice-**phase** spread ±0.03 at res 7 |
| IoU@1.0, across resolutions | ❌ | 0.0338 |
| `dev_mean`, across resolutions | ❌ | floor differs **0.180 mm** (scales ≈0.17·psize) |
| `dev_mean`, same resolution | ✅ | — |

IoU@2.0 is resolution-stable **by construction**: the L∞ (cube) particle representation gives
occupied volume `n·psize³`, invariant with resolution because cubes tile space. That design
choice of the contact workstream's is correct and load-bearing — **do not "fix" it to balls.**

> 🚩 **Absolute IoU is not "percent correct."** A *perfect* sim scores **0.88** at vox 2.0.
> Only differences carry meaning.

### 4.6 The stock-volume ceiling (A-7's to fix; ours to account for)

Across 14 real meshes, verified two independent ways — divergence theorem, and voxel-fill
converging `+9.18% → +4.49% → +2.27%` as pitch halves 1.0 → 0.5 → 0.25 mm:

```
real bar volume (hit 1, after)   67,486 mm³      (median over 14 meshes: 66,652)
sim nominal (π · 20² · 59)       74,142 mm³
                                 → sim carries +9.9% more material

volume ceiling        V_real/V_sim   = 0.910
discretisation ceiling (res 10)      = 0.880
combined structural ceiling         ≈ 0.801
best arm measured                    = 0.7887      ← 98.4% of ceiling
```

`REAL_STOCK_RADIUS_MM = 20.0` is a bare, underived constant in shared
`forge_common/main/forge_common/real_scale.py:47`. The real bar is ordinary **1.5″ stock**
(37.93 mm equivalent diameter), **oval in section with rounded ends** — not faceted/octagonal
(that hypothesis was tested and refuted, §5).

**Consequence for this workstream:** hit-1 geometry is saturated. Material and thermal
differences have almost no room to express themselves there. Until the stock geometry is fixed,
prefer **differential** comparisons (arm vs arm at fixed everything) over absolute scores, and
prefer **accumulated multi-hit** geometry over hit 1.

---

## 5. Refuted claims — kept deliberately

Every one of these was believed, acted on, or written down. Several are the author's.

| Claim | Origin | Refuted by |
|---|---|---|
| Material signal sits below the resolution artifact (24%) | inherited handoff | Material arms were all res 7 — the artifact is common-mode and cancels. Metric resolves 0.0002. |
| Grid resolution is "the single biggest lever" | inherited | Measured on `dev_mean`, which is not cross-resolution safe. res 7→10 is +0.0012…+0.0049 on IoU@2.0, inside ±0.03 phase uncertainty. |
| The confound is "4.3×, the subtracted radius differs 0.431 mm" | prior session | Compares the *subtrahend* difference to the *result* difference, ignoring that the distance term also scales with psize. Measured floor gap is 0.180 mm. |
| Adiabatic heating → thermal-softening feedback is the instability | this session | Predicted survival below `jc_T_ref` (softening clamped off). 1200 K died identically to 1273 K. |
| The real bar is an octagonal cogged section | this session | 0.90 fill ratio was coincidence — it falls monotonically to 0.53 by hit 17 as the bar elongates. `r(θ)` shows a smooth single-lobe oval, not 8 flats. |
| Press speed explains the residual failures | this session | 273 → 35 m/s changed nothing (1 hit either way with default FLIP; PIC runs 12 → 9). |
| Explicit MPM fundamentally cannot do coupled thermomechanics | this session | Thermal stability limit is **517,000× looser** than mechanical on our grid. Fully-explicit coupled thermomechanical forming is standard and validates against implicit quasi-static. |
| B-3's contact fix is root cause; the FLIP finding is a symptom mask | this session | Withdrawn. `apply_particle_contact` runs at :878, **after** g2p (:864), so a PIC gather discards nothing. |
| Surface/contact CFL is the cause, θ = 36 | B-3, self-retracted | 1000× units error — `_particle_volume_scale` applied twice. Real θ = 0.036, matching this session's independent audit. |
| "JC is temperature-blind, therefore not mechanics" | both B sessions, from a stale comment in `material_arms.py` | True only of the *initial* temperature. Plastic work un-clamps `T_star` on the first blow. |
| Temperature at the blow is not measured | inherited, long-standing | Measured ~960 °C from the 06-15 mcap (session `12b6fa7e`). |
| The clamp fix was "the biggest material error, 2.3×" | prior session | Not load-bearing at 960 °C — the clamp only bites above 1000 °C, which this process never reaches. |

> **Two process lessons worth more than any single finding.**
> 1. **A mirror test pins consistency, never correctness.** B-3 had *16 tests passing* against a
>    diagnosis that was wrong by 1000×, because the tests re-implemented the same error.
> 2. **A runner must report what it BUILT, not what it was ASKED for.** An earlier material
>    experiment produced four beautifully consistent arms that had all silently built
>    Johnson-Cook, including the two requesting Arrhenius. `~/material_arms.py` now reads the
>    built scene back and aborts on mismatch. Keep that property in anything that replaces it.

---

## 6. Theory: time scaling as a similarity transform

### 6.1 The derivation

The coupled system in real time, with the mechanical problem quasi-static (real strain rate
≈ 0.35 /s; inertia negligible):

```
ρCp ∂T/∂t  =  ∇·(k∇T)  +  χ σ:ε̇ᵖ        [+ surface BCs]
```

Substituting `t_sim = t_real / N` (so `∂/∂t = (1/N)·∂/∂t_sim`) and noting that the strain rate
the *sim* computes is already `ε̇_sim = N·ε̇_real`:

```
ρCp ∂T/∂t_sim  =  N·∇·(k∇T)  +  χ σ:ε̇ᵖ_sim
                  └── ×N ──┘     └── no scaling ──┘
```

**Transport terms take an explicit factor N. The plastic-heating source does not** — it scales
itself, because per unit of mechanical progress the plastic strain increment is invariant.

This is exactly the split the code implements: `dt_th = substep_dt · S_T` multiplies diffusion,
convection, radiation and contact, while `p2g_post_constitutive` is deliberately unscaled.
**The architecture is correct.** Time scaling is a rigorous similarity transform, not a hack.

### 6.2 One dial, three consumers

The scheme has **one** scaling factor N, not three independent timescales:

| Consumer | Transform | Current state |
|---|---|---|
| Mechanics | runs at **N×** | ✅ N = 1,773 |
| Thermal transport | coefficients **× N** | ❌ S_T = 237,014 — **134× off**; should be N |
| Flow-law strain rate | **÷ N** | ⚠️ prescribed constant 0.41 /s |

They *look* like three clocks only because they are currently set independently. Deriving all
three from one `N` makes inconsistency structurally impossible.

The current default S_T is chosen from a **numerical** criterion (25% of the diffusion CFL
ceiling) that has no connection to how fast the press moves. `options.py` already contains the
correct computation (`mech_accel`, :652) — it is simply not the default.

### 6.3 Strain rate — use the sim's own, divided by N

Time scaling genuinely breaks one thing: **rate sensitivity**. The sim computes
`ε̇_sim = N·ε̇_real ≈ 625 /s` — Hopkinson-bar territory — where the real process is ~0.35 /s.
A rate-dependent flow law (Zener-Hollomon: `Z = ε̇·exp(Q/RT)`) then evaluates at the wrong rate,
which corrupts σ *and* plastic heating, since `dT ∝ σ`.

The code already computes a per-particle rate (`materials.py:283`) and its own comment calls it
*"fictional at the sim's press speed."* It is fictional **only because it is measured in sim
time**:

```python
rate_physical = rate_est / N
```

**This is preferable to a prescribed scalar**, because it preserves what a constant erases:

- **Spatial variation** — strain rate under the die is several times that at the bar ends. A
  single 0.41 /s applies a mid-billet average to material that is barely deforming.
- **Variation through the blow** — for upsetting `ε̇ = v/h`, and `h` falls as the billet
  flattens, so the rate rises ~35% within a single blow at constant die speed.

Since σ ~ ln(Z), a 3× spatial spread in ε̇ shifts flow stress meaningfully.

Keep `arrhenius_process_strain_rate` as an explicit **experimental override** for controlled
studies, documented as such rather than as the production path. Its nominal value is sound:
`ε̇ = v_die/h = 0.0141/0.040 = 0.35 /s`.

### 6.4 Why *not* mass scaling

Mass scaling (inflate ρ so `c = √(E/ρ)` falls and dt rises) is the other standard technique and
is what LS-DYNA uses for quasi-static forming. **It is the wrong choice for this codebase**,
because ρ enters the thermal path twice:

- `base_mpm_solver.py:496` and `:526` — `rho = particles_info[i_p].mass / particle_volume`,
  feeding `dT = χ·σ·Δγ/(ρ·Cp)`
- `mass_thermal_real` in the coupler — the cell thermal mass for every conduction, convection,
  radiation and contact term

Both derive from **particle mass**. Scale mass by *m* and plastic heating divides by *m*, and
all heat transfer divides by *m* — silently. Mass scaling is only safe in codes carrying a
separate *inertial* density. Ours does not.

⇒ **For a thermomechanically coupled code, time scaling is the less invasive choice.**

### 6.5 The real limit on N — inertia

Quasi-static validity requires kinetic energy to stay small against internal energy:

```
KE density ≈ ½ρv²        = ½ · 7334 · 25²         ≈  2.3 MJ/m³
IE density ≈ σ_y·ε_p     ≈ 200e6 · 0.3            ≈ 60   MJ/m³
                                            ratio ≈ 3.8%      (criterion <5%)
```

**N = 1,773 passes, but barely — and KE scales as N².** Doubling N would put this near 15% and
break the quasi-static assumption outright.

⚠️ This is a back-of-envelope figure derived from die speed. It **must** be measured from the
actual particle velocity field and asserted in code. It is also the most likely explanation for
the observed force-vs-press-speed non-convergence, which is currently unexplained.

### 6.6 Why explicit MPM is fine

For the record, since this was questioned and researched:

| Limit | Expression | Value |
|---|---|---|
| Mechanical (hyperbolic) | `dt < dx/c`, c = 4070 m/s | 1.34 × 10⁻⁶ s |
| Thermal (parabolic) | `dt < dx²/(6α)` | 0.452 s |
| **Thermal headroom at the mechanical timestep** | | **~517,000×** |

Coupling thermal to explicit mechanics costs essentially nothing in stability. The two limits
scale differently with `dx` (∝dx vs ∝dx²), but the crossover is at **dx ≈ 16 nm** — we are at
5.5 mm. Fully-explicit coupled thermomechanical bulk forming is standard practice; the
literature reports good agreement between explicit dynamic and implicit quasi-static results
for exactly this class of problem, using the same isothermal-split staggering our code uses.

---

## 7. Known defect inventory

Ranked by impact on the coupled-physics goal.

| # | Defect | Location | Impact | Owner |
|---|---|---|---|---|
| 1 | **FLIP gather on the temperature field** | `base_mpm_solver.py:239, :723` | Blocks coupling entirely — particle temperature unusable for flow stress | B-3's component |
| 2 | **S_T decoupled from N** (237,014 vs 1,773) | `options.py:616` | 134× too much heat transfer per blow | B-3 / shared |
| 3 | **Strain rate not divided by N** | `materials.py:283` | Flow stress and plastic heating evaluated at ~1800× the real rate unless the prescribed override is on | this workstream |
| 4 | **Runtime α 1.82× the design α** | `1.1e-5` vs `6.05e-6` | `k_diff` 0.458 vs intended 0.25 — most of the margin gone | unclaimed |
| 5 | **`approach_speed` = 273.2 m/s** vs a 100 m/s intercept | `options.py:940` — uses macro `dt`, not `substep_dt` | Real defect; **not** the thermal cause | unclaimed |
| 6 | **No KE/IE assertion** | — | The validity of the whole time-scaling scheme is unmonitored | this workstream |
| 7 | Stock volume +9.9% | `real_scale.py:47` | Caps absolute geometry agreement at ~0.80 IoU | A-7 |
| 8 | "Thermal Detonation" is a misleading name | `strike_controller.py:1657` | Checks `T > 4000 K` **or** `T < 0 K` — fires on numerical collapse, not overheating | cosmetic |

---

## 8. Implementation plan

### Phase A — Instrumentation, before touching physics

Nothing here depends on the gather fix, and without it no refactor can be evaluated.

- **A1. KE/IE monitor + assertion.** Measured from the particle velocity field, not estimated.
  This is both the validity check and the ceiling on N.
- **A2. Energy conservation audit.** `Σ χ·σ:ε̇ᵖ·V` in versus `Σ ρCp·ΔT·V` + boundary flux out.
  This is the test class that catches units bugs — precisely what produced the 1000× error and
  its 16 passing mirror tests.
- **A3. Temperature-history telemetry.** Per-hit mean/min/max/std, written to the run summary,
  so thermal behaviour is observable rather than inferred from a crash code.
- **A4. Promote `~/material_arms.py` into a committed sweep harness.** It already carries
  `--flip-frac`, `--billet-k`, `--approach-speed`, `--cells`, `--thermal-mode`, `--no-thermal`
  and hard read-back verification. Add `--N`, `--S_T`, structured output, and **keep the
  abort-on-mismatch property.**

### Phase B — Unify the scaling under one dial

```python
N = pressing_speed / real_die_speed        # the ONE quantity
thermal_time_scale = N                      # 237,014 → 1,773
rate_for_flow_law  = rate_est / N           # per-particle; replaces the 0.41 constant
assert KE/IE < 0.05
```

`arrhenius_process_strain_rate` demoted to a documented experimental override.

⚠️ **Cost:** the induction calibration is tuned against the *current* (wrong) S_T default.
Changing it forces a recalibration. That is B-3's call and must be coordinated, not assumed.

### Phase C — Replace the thermal gather

FMPM(k≥2) per Nairn, or an APIC-style filtering transfer consistent with what momentum already
uses. **B-3's component.** The clean split is *they implement, we measure*.

### Phase D — Validation campaign (see §9)

### Phase E — External validation

Retrieve the 06-15 mcap (8.58 GB; 39.6 GB free as of writing) and validate force and cooling
rate. Blocked only on that retrieval.

---

## 9. Validation and metrics plan

### 9.1 Self-consistency tests — no experimental data required

These are the strongest tools for a scaling refactor, because they test implementation
correctness directly rather than agreement with a noisy measurement.

**D1 — N-invariance. The headline test.**
If S_T = N and `rate = ε̇_sim/N` are correct, **the physical answer must not depend on N.**
Sweep N ∈ {400, 900, 1773, 3500}; geometry and temperature history must collapse onto one
curve. Any systematic drift means the scaling is wrong.

- **D1a — mechanical only (thermal off): runnable immediately.** Validates the mechanical time
  scaling independently of the gather. If geometry drifts with N, every force and material
  result to date is contaminated.
- **D1b — fully coupled:** after Phase C.

**D2 — S_T sweep at fixed N.** Should show a clear optimum at S_T = N. The current default sits
134× away from it.

**D3 — FLIP/gather sweep.** Partially done (0.0 / 0.2 / 0.5 / 0.97); complete against whatever
replaces the gather.

**D4 — Energy conservation.** Closes to within a stated tolerance, per hit and cumulatively.

**D5 — KE/IE.** Tracked per run; asserted.

**D6 — Grid convergence.** Carefully: cross-resolution IoU@2.0 carries ±0.03 lattice-phase
uncertainty, so use phase-averaged or volume/extent-based measures, not the raw score.
**No proper convergence study has ever been done** — res 7→10 is one step, not convergence.
Particles scale as res³ (res 14 ≈ 8× res 7); 6 GB VRAM is unprobed at high res.

### 9.2 Against experimental data

**Every case run twice — isothermal at ~900–1000 °C, and fully coupled.**

| Test | Data | Metric | Notes |
|---|---|---|---|
| Per-hit geometry | 17-hit meshes | IoU@2.0 + dev_* at fixed resolution | Ceiling-capped ~0.80 until stock geometry fixed → prefer differentials |
| **Isothermal prediction** | 17-hit (no temp data) | predicted ΔT over the sequence | Coupled run *should* predict near-isothermal given reheat BCs. Deviation = missing physics or wrong BCs |
| Cooling rate | 06-15 mcap | ~4.8 °C/s through the blow | Needs retrieval |
| Billet temperature | 06-15 mcap | ~960 °C at blow #1, per-blow for all 47 | Needs retrieval |
| Force | 06-15 mcap | peak force per blow | ⚠️ press is **force-limited at 110.2 kN**, and force is nearly **blind to material** (10% flow-stress change → 1.1% force; elastic `nu` change → 10%) |

### 9.3 Sensitivity matrix

Parameters that are weakly constrained and materially affect the answer:

| Parameter | Current | Confidence | Why it matters |
|---|---|---|---|
| `emissivity` | 0.40 | low | Sets radiative loss; camera's own setting is an Optris preset pulled from their servers at runtime, not in the repo |
| `h_air` | 15 W/m²K | low | Convective loss |
| `h_contact` | 3000 W/m²K | low | **Die chill magnitude** — the dominant thermal coupling during a blow |
| Friction | absent | — | Inferred from barrelling; currently unmodelled, stands in as a fitted constraint factor |
| `enable_CPIC` | True | unquestioned | Overrides Genesis's own default of False |
| `coup_softness` | 5e-4 | unquestioned | — |
| `grid_cells_across_billet` | 7 | — | See D6 |

---

## 10. Open questions

1. **Should the 06-15 mcap be retrieved now?** It is the only path to validating thermal
   coupling against reality. 8.58 GB; space is available.
2. **What boundary conditions reproduce the reheat?** The 17-hit bar was reheated to
   ~900–1000 °C between hits. Modelling that explicitly (vs. simply re-initialising temperature
   per hit) determines whether the isothermal-prediction test is meaningful.
3. **Does the induction calibration survive S_T = N?** Phase B forces this question.
4. **What replaces FLIP** — FMPM(k), XPIC, or an APIC-style transfer for temperature? Needs a
   survey against our constraints (GPU kernel, Quadrants DSL, existing APIC momentum path).
5. **Is the force-vs-press-speed non-convergence just the inertial budget?** §6.5 predicts it.
   D1a tests it.

---

## 11. Provenance and confidence — trust at your peril

Statements here carry different weights. These specifically went in **without independent
verification**:

- **The ~960 °C billet measurement** is inherited from session `12b6fa7e`. Good provenance —
  per-frame table, documented isolation method, its own caveats (thresholding not segmentation;
  emissivity unknown ⇒ ±50 K) — but **not re-derived**, and a pushed commit (`cb3765a3`) rests
  on it.
- **KE/IE ≈ 3.8%** is a back-of-envelope from die speed, not a measurement of the velocity
  field. Phase A1 exists to replace it.
- **The FLIP *mechanism* is not pinned.** Contact θ (0.036), `k_conv` (0.0001), `k_rad`
  (0.0005) and `k_diff` (0.458) are all below their limits, yet the gather controls survival
  completely. What is established is the **control**, not the **route**. Do not invent a
  mechanism to fill the gap.
- **`jc_C` is dead code** and **`jc_m` is unsourced** — carried from earlier summaries, not
  re-verified here.
- **The Ryan & McQueen (C, m) pairs remain unverified with no path** — the 1989 Concordia thesis
  is a pure scan (0 `/Font` objects, 287 image XObjects); no OCR tooling installed. Their
  functional form does not match the paper's published equation.
- **`enable_CPIC = True` and `coup_softness = 5e-4`** are live contact settings that **neither**
  workstream has questioned.

---

## 12. Reproduction

### Environment

```bash
# pixi is NOT on PATH
~/.pixi/bin/pixi run --manifest-path aims-genesis/thermal-st-invariance/pixi.toml --frozen python ...

# agforge + forge_common must BOTH resolve, or you silently get the other branch's agforge
export PYTHONPATH=/home/timothy/GitHub/Genesis/forge_common/main
export LD_LIBRARY_PATH=/usr/lib/wsl/lib          # else silent CPU fallback
```

> 🚩 `pixi run --manifest-path aims-genesis/nsf-demo/pixi.toml` resolves **nsf-demo's** agforge,
> which has **no Arrhenius at all**. `material_arms.py`'s read-back guard catches this; keep it.

> 🚩 Heredocs and URLs are mangled through Git Bash → `wsl.exe`, and shell variables in loops get
> eaten. Stage scripts on Windows, `cp` from `/mnt/c` inside WSL, run there. Avoid `$VAR` inside
> `wsl.exe bash -lc '...'`.

### Scoring geometry

```bash
~/.pixi/bin/pixi run --manifest-path aims-genesis/nsf-demo/pixi.toml --frozen \
  python aims-genesis/nsf-demo/agforge/analysis/geom_batch.py \
  --batch material_arms --hit 1 --sim-hit 1 --vox 2.0 1.0
```

Run the contact workstream's scorer **in place** — never copy it into this branch, or the two
sets of numbers become incomparable.

### Diagnostic scripts (WSL home; not yet committed)

| Script | Purpose |
|---|---|
| `~/material_arms.py` | The verified arm runner. Reads the built scene back and **aborts on mismatch.** |
| `~/null_floor.py` | `dev_mean` discretisation floor via perfect-lattice fill |
| `~/null_iou.py` | IoU floor, same method |
| `~/real_stock.py` | Real bar volume by two independent methods |
| `~/cfl_audit.py` | All three thermal CFL numbers + press/approach speeds |
| `~/verify_volume.py` | Watertightness, voxel-fill convergence, hull bound |
| `~/cross_section.py` | Cross-section shape (refuted the octagon hypothesis) |
| `~/viz2.py` | Extent-based comparison plots |

**Phase A4 should commit the useful ones into the repo.**

### Sibling sessions — shared worktree

🚨 This worktree (`aims-genesis/thermal-st-invariance`) is **shared with session B-3**
(`5dbbd727`), which owns the thermal solver. Both sessions independently proposed fixing the
same instability on 2026-08-11. **Check what the sibling is doing before editing
`base_mpm_solver.py` or `legacy_coupler.py`.** Diagnostics that need no source change —
runtime-field pokes, read-only analysis — are always the safer path.
