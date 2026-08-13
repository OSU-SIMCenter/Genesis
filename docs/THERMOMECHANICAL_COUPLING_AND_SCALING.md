# Thermomechanical Coupling and Time Scaling — Research, Findings, and Plan

**Status:** living document. Canonical reference for the coupled thermal-mechanical work on
`agforge/v2/thermal-st-invariance`. Written 2026-08-12 from session `dbc3b95f`
(← `2c4556e5` ← `156eb841`). Repo state at writing: `cb3765a3`, clean.

⚠️ **Source line numbers in this document are valid as of `5f594e5e` (2026-08-13) and drift.**
This worktree is shared: B-3's `d2d7c701` inserted a net +12 lines into `agforge/options.py` and
`bc850e82` a net +10 into `agforge/strike_controller.py`, silently invalidating every citation
below their insertion points — including the one the §3.5.1 force-limit finding rests on. They
were re-pointed at `5f594e5e`. **Grep for the symbol, not the line.**

**Read this before touching the thermal solver, the time-scaling knobs, or the geometry
metric.** Sections 5 and 11 exist specifically to stop work being redone or wrong conclusions
being re-derived — several claims in this project have been confidently wrong, including
claims made by the author of this document.

**Companion documents**
- `docs/316L_MECHANICAL_PROPERTIES.md` — the material card, its sources, and its caveats.
- `aims-genesis/nsf-demo/docs/Contact_Method_Research_And_Plan.md` — the contact workstream's
  equivalent canonical reference. Read it before assuming anything about contact.
- `docs/AS_BUILT_AGILITY_FORGE.md` — the real machine.

**Notation used throughout**

| Symbol | Meaning | Current value |
|---|---|---|
| `N` | mechanical time-scaling factor = `pressing_speed / real_die_speed` | 1,773 |
| `S_T` | `thermal_time_scale` — multiplies every transport term via `dt_th = substep_dt · S_T` | 237,014 (should be `N`) |
| `χ` | Taylor-Quinney coefficient, fraction of plastic work becoming heat | 0.9 |
| `ε̇` | equivalent plastic strain rate | real ≈ 0.35 /s; sim ≈ 625 /s |
| `α` | thermal diffusivity | runtime 1.1e-5 m²/s |
| `k_diff`, `k_conv`, `k_rad` | explicit stability numbers; `k<1` monotone, `k<2` bounded, `k>2` diverges | 0.458 / 0.0001 / 0.0005 |
| `θ` | contact-exchange Euler factor `h·A·dt_th/(m·Cp)` | 0.036 |
| `flip` | FLIP/PIC blend fraction for the temperature gather | 0.97 |
| `Z` | Zener-Hollomon parameter `ε̇·exp(Q/RT)` | — |
| IoU@2.0 | voxel-occupancy intersection-over-union at 2 mm voxels | — |

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
| `pressing_speed` | 25.0 m/s | `options.py:778` | die velocity during the blow |
| `real_die_speed` | 0.0141 m/s | `options.py:788` | the physical reference |
| **N (mechanical accel.)** | **1,773** | derived, `options.py:664` | = `pressing_speed / real_die_speed` |
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
E = 121.5 GPa (CONSTANT)   nu = 0.383 (CONSTANT)   rho = 7334 (CONSTANT)
```

### 2.4 🚩 Two switches named almost the same thing

| Switch | Default | Meaning |
|---|---|---|
| `MPMOptions.enable_thermal` | **True** (`options.py:684`) | The solver *runs* the thermal kernels — diffusion, surface flux, gather, contact |
| `StrikeController.thermal_enabled` | **False** (`strike_controller.py:140`) | "Thermal Freezing" — snapshots particle temperatures before each physics step and **restores them after** (`:958`, `:976`) |

⇒ **In the default forging path the solver computes a full thermal solution every substep and
`StrikeController` then discards it**, restoring the snapshot. A press is therefore exactly
isothermal by construction, verified previously: a 74-step press starting at 1273.15 K ends at
1273.15 K with std 0.0.

### 🚨 There is currently NO code path for coupled forging

This is the single most important structural fact for this workstream, and it is stronger than
"the coupled path is less tested."

- `StrikeController.thermal_enabled` defaults to **False** (`:130`).
- The only setter is `set_thermal_state()` (`:481`), whose docstring reads **"Toggles induction
  heating"** and which logs `Thermal ACTIVATED (q_peak=...)`.
- Its **only caller is `teleop_socket.py:472`** — the interactive teleop path.
- The batch/adapter path never calls it. `genesis_forge_adapter.py:42` says so explicitly:
  **"Cold (no-heating) runs for now: `thermal_enabled` defaults to False"**.
- `options.py:199` independently confirms the state: the JC softening limitation is
  *"inert today (thermal_enabled = False) but it is wrong the moment thermal is switched on."*

⇒ **Every result in this document — §4.9 included — was produced with particle temperatures
frozen and restored around every physics step.** The solver ran the thermal kernels, and the
controller threw the answer away. That is why an instability shows up in runs whose thermal
output is discarded: the divergence happens *within* a step, before the restore, and trips the
diagnostic.

⇒ **Phase C is necessary but not sufficient.** Fixing the gather makes particle temperature
*usable*; it does not make it *used*. Reaching coupled forging also requires plumbing an
enable path into the adapter/batch driver.

⚠️ **And the switch may be overloaded.** Temperature evolution and the induction heat source
appear to share one flag, so `thermal_enabled = True` may enable both — while forging validation
wants evolution (die chill, radiation, plastic heating) *without* induction. The solver has a
separate `_induction_active` gate, so they may in fact be separable. **Unverified — establish
this before building on it** (§10 Q7).

⚠️ The restore at `:965` is guarded by `not physics_failed` — so a diverging run keeps its bad
temperatures and trips the diagnostic rather than being silently rescued.

### 2.5 Coupling terms — what is implemented and what is missing

A fully coupled thermomechanical formulation has four two-way terms. We have two.

| Term | Direction | Status | Where |
|---|---|---|---|
| Plastic work → heat | M→T | ✅ implemented | `p2g_post_constitutive` (:492), `χ = 0.9` |
| Temperature → yield stress | T→M | ✅ implemented | Arrhenius `_flow_stress_pa`; JC `thermal_softening` |
| **Thermal expansion → strain** | **T→M** | ❌ **absent** | no CTE anywhere in the solver |
| **Temperature → elastic moduli** | **T→M** | ❌ **absent** | `E`, `nu` are constants |

Temperature-dependent **Cp** and **k** *are* implemented and are 316L-specific
(`get_steel_cp` :438, `get_steel_thermal_conductivity` :459) — note 316L's conductivity *rises*
with temperature (14.6 → ~29 W/mK), the opposite of the AISI 4340 curve they replaced.

**Why the two absences matter specifically for this workstream.** `E = 121.5 GPa` is already the
*elevated-temperature* value (room-temperature 316L is ~193–200 GPa). So a constant modulus is
**correct for an isothermal run at forging heat, and becomes an error source exactly when
temperature is allowed to vary** — i.e. the moment dynamic coupling is switched on. The same
holds for thermal expansion: 316L's CTE is ~18e-6 /K, so a 700 K excursion is ~1.3% linear /
~3.8% volumetric strain, which is comparable in size to the deformation the geometry metric is
trying to measure.

⇒ **Both absences should be quantified before the coupled results are trusted.** They are
listed as defects in §7 and as open questions in §10.

---

## 3. Validation data — what exists and what each can prove

| Dataset | Contents | Location | Validates |
|---|---|---|---|
| **17-hit per-hit meshes** | billet triangle meshes before/after each of 17 hits, ~100–137k verts, mm | `forge_common/main/outputs/real_meshes/hit_NN.npz`, keys `V_before`/`T_before`/`V_after`/`T_after` | **mechanical deformation** |
| **07-17 mcap** | thermal camera on the **induction coil**, heating curve | `~/GitHub/Genesis/forge-data/20260717_135009.mcap` (1.15 GB, local) | **induction heating** (B-3) |
| **06-15 T4 bulk mcap** | press force/position telemetry **and** thermal camera on the **press** | `/mnt/c/Users/banko/Documents/forge-data-stage/2026-06-15_T4_bulk/20260615_180456_T4_bulk.mcap` (8.58 GB, **local** — staged on Windows, read over `/mnt/c`) | **force**, **billet temperature during forging**, **cooling rate** |

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

### 3.2 Location note — both datasets are local

**Corrected 2026-08-13.** Earlier revisions of this document recorded the 06-15 mcap as
pCloud-only and listed its retrieval as a blocker in §8 (phase E), §9.2, §10 and §12. That was
wrong. The file has been staged on the **Windows** filesystem since 2026-07-23 and verified
there on 2026-08-06 — size `8,580,279,846` B, SHA-1 `c7503c2a074c62fa365dd8b0007a57e3102f8a4a`,
mcap header/footer magic OK.

- Windows: `C:\Users\banko\Documents\forge-data-stage\2026-06-15_T4_bulk\20260615_180456_T4_bulk.mcap`
- From WSL: `/mnt/c/Users/banko/Documents/forge-data-stage/2026-06-15_T4_bulk/…`

It is deliberately **not** copied into the WSL vhdx — analysis reads it over `/mnt/c` so the
distro does not carry a second 8.58 GB. `forge-data/WHERE-IS-THE-DATA.md` records this in its
"Local Windows staging" section; pCloud remains the archive of record.

⇒ **No validation in this document is blocked on a retrieval.** Phase E is runnable now.
⇒ And the first thing it unblocked has already been done — see §3.3.

> 🚩 **Keep the failure, not just the correction.** The earlier check searched only the WSL
> filesystem, concluded "not local", and reported it as blocked *on the user* — who had already
> said both datasets were on disk and was right. A confident negative is only as good as the
> vantage it was searched from.

> ⚠️ Do **not** let a Windows-side tool write into `\\wsl.localhost\` — the 9P bridge zero-pads
> to 4096-byte boundaries and corrupts silently. Stage on Windows, verify with
> `rclone check --checksum`, copy in from `/mnt/c` *inside* WSL, re-hash.

---

### 3.3 The ~960 °C figure — re-derived independently, 2026-08-13

The blow-#1 billet temperature had been carried on trust since session `12b6fa7e`
(§11), with a pushed commit resting on it. It has now been re-derived from the raw
mcap with an extraction written against `agforge/mcap_thermal.py` rather than
against the original analysis.

A re-run of the *same* method would have been a mirror test — same systematic
error, guaranteed agreement, no information. This project has already been caught
by exactly that (16 thermal tests passed because they re-implemented the same
units bug). So the checks were chosen to be independent of the original method.

**(a) Structural.** `20260615_180456_T4_bulk.mcap`: 2,378.0 s, 1,683 chunks, 5
channels — `hmr/sensors/thermalcam`, `hmr/press/state`, `hmr/torm/state`,
`hmr/ard/state`, `forge/deform/u_taken`. Thermal frames are 288×382 as documented.

**(b) Value.** Workpiece isolated as the largest connected region above
max(700 °C, p85), in Celsius:

| t [s] | this session p50 | session `12b6fa7e` p50 | Δ |
|---|---|---|---|
| 297.0 | 901.1 | 904.6 | −3.5 |
| ~298.4 | 964.3 | 966.9 (t=298.0) | −2.6 |
| ~299.8 | 957.1 | 960.9 (t=299.0) | −3.8 |
| ~302.6 | 943.1 | 945.9 (t=302.0) | −2.8 |
| ~309.6 | 911.3 | 909.1 (t=310.0) | +2.2 |

Sample times differ because this pass takes one frame per chunk. **The figure
reproduces within ~4 °C.** Cooling rate over the same span: **4.73 °C/s** here vs
**~4.8 °C/s** originally.

**(c) Method sensitivity — the genuinely new result.** The original flagged
"thresholding, not segmentation" as a caveat. Varying the isolation rule across
four methods (largest connected blob; bare threshold with no connectivity; a fixed
>900 °C cut; the hottest 10% of the frame with no threshold at all) moves the
median by **3.8–16.2 °C** once the bar is in frame — well inside the **±50 K**
emissivity band. ⇒ The caveat is real but **not** the dominant uncertainty; the
original's own risk assessment was right.

**(d) The structural claim, which is the one that matters.** The value is only
meaningful if the camera views the press rather than the induction coil — and that
claim already flipped once, in the opposite direction. Rendered frames settle it:
at t=295.6 the frame shows machine structure and **no billet**; by t=298.4 a large
hot bar fills the frame; it then cools monotonically. A coil-mounted view cannot
produce a workpiece that arrives and departs. **Confirmed.**

⇒ **§11 no longer carries this as unverified.** What it still carries is
emissivity, which no amount of re-extraction can fix — it is a property of the
camera's configuration, not of the analysis.

> ✅ **The blow-#1 timing has since been re-derived — and it moved.** `analyze_press_mcap.py`
> segments the force channel independently and puts blow #1's **force episode at 302.7 s**, peak
> 66.5 kN at 303.18 s. The ~299.25 s figure is the `u_taken` **command** timestamp, not contact.
> Sampling at the command reads the bar ~4 s early and so ~18 °C hot: **942.6 °C at the force
> episode** against the inherited 960.9 °C. Small beside the ±50 K emissivity band, but a real
> systematic, and it ran in the flattering direction. The bar enters frame at ~296.6 s (blob
> growing 5,820 → 15,473 px) and peaks ~298 s, so it is **already cooling** when the press
> engages.

---

### 3.4 🎯 All 47 blows — and blow #1 is the 96th percentile

**Measured 2026-08-13** (`per_blow_temp.py`). Press episodes segmented from the force channel by
`analyze_press_mcap.py`, then the thermal camera sampled at each. Both come from the same file,
so the two clocks are the same clock and no alignment is assumed.

| | °C |
|---|---|
| min | **615.2** |
| p25 | 783.4 |
| **median** | **823.6** |
| p75 | 862.0 |
| max | **967.1** |
| **blow #1** | **942.6 — the 96th percentile, 119 °C above the median** |

🚨 **Every thermal number this project uses is a blow-#1 number, and blow #1 is nearly the
hottest blow in the session.** This is the same error as §4.10's — where every geometry score
had been taken at hit 1 — in a different domain, found five days apart. It is worth treating as
a standing failure mode rather than two coincidences: *the first event of a sequence is the one
that gets analysed, and it is systematically unrepresentative.*

**The bout structure.** Twelve bouts separated by 100–274 s reheats, cooling monotonically
within each:

| bout | blows | T first → last °C | °C/s |
|---|---|---|---|
| 1 | 1–5 | 942.6 → 848.9 | −4.74 |
| 2 | 6–10 | 926.3 → 809.9 | −5.73 |
| 3 | 11–13 | 803.6 → 795.8 | −0.68 |
| 4 | 14–18 | 879.6 → 773.9 | −4.64 |
| 5 | 19–23 | 853.4 → 798.9 | −2.85 |
| 6 | 24–27 | 817.1 → 687.1 | −7.50 |
| 7 | 28–31 | 744.1 → 787.4 | **+2.71** |
| 8 | 32–35 | 830.8 → 615.2 | −11.58 |
| 9 | 36–39 | 967.1 → 839.8 | −7.50 |
| 10 | 40–42 | 862.4 → 776.6 | −6.32 |
| 11 | 43–45 | 829.4 → 722.1 | −6.54 |
| 12 | 46–47 | 834.1 → 765.8 | −10.66 |

**Median within-bout cooling is −6.03 °C/s**, spanning −11.58 to +2.71. ⇒ The inherited
**~4.8 °C/s is also a blow-#1 artifact** — bout 1 is the *slowest-cooling bout in the session*,
and the typical rate is ~25% faster. 🚩 Bout 7 *warms*. Not explained; plausibly a re-grip
exposing a hotter face, or a partial reheat. Recorded, not rationalised.

**Reheats do not return the bar to a fixed temperature.** Bout-opening values run 744–967 °C.
Any coupled model that assumes a constant reheat setpoint will not match this sequence.

**Temperature drop through contact:** median **−16.8 °C**, p10 −30.2, p90 −6.2 — the die-chill
signal a coupled run must reproduce over a ~0.5–1.7 s contact.

> 🚩 **A bug in this analysis, caught and fixed, worth keeping.** The first pass isolated the
> workpiece as the largest region above `max(700 °C, p85)` — the rule inherited from the blow-#1
> analysis. It returned "no bar in frame" for four blows. They were not camera gaps: all four had
> 216 thermal frames available and were simply the **coolest** blows, where too little of the bar
> cleared 700 °C. The floor was silently truncating the low tail, biasing the median **high** —
> the same direction as the error it was embedded in. Removing the floor recovered all 47 blows
> and dropped the minimum from 724 °C to **615 °C**. `p85` alone is the principled rule: the
> workpiece occupies ~16,400 of 110,016 px ≈ 15% of the frame, so the 85th percentile is
> calibrated to its own frame fraction and follows the bar down as it cools.

⚠️ **This is the 06-15 session, not the 17-hit geometry sequence.** Per §3.1 the two must not be
generalised between. The 17-hit set was reheated to ~900–1000 °C; the 06-15 session runs
substantially cooler. They are experiments at *different temperatures*.

### 3.5 🎯 The die-closure shortfall is a FORCE LIMIT, not a closure cap

Workstream A reports that the sim lands on its commanded gap while the real press falls ~2.4 mm
short, and proposes this as a third explanation for the multi-hit decay alongside the initial
condition. They model it with `forge_common/force_correction.py`, which takes
`min(MEAN_ACTUAL_CLOSURE_MM = 7.94, commanded)` on every hit — a cap fitted to the 17-hit
sequence, where *every* hit fell short.

Aligning all 47 06-15 blows to their `u_taken` commands (`align_blows_mcap.py`) says the
mechanism is conditional, not a cap:

| | n | commanded gap, median | peak force, median |
|---|---|---|---|
| **reached command** | **38** | 25.57 mm | 76.2 kN |
| **fell short** | **9** | 17.40 mm | **110.1 kN** |

**38 of 47 blows land within 0.13 mm of their commanded gap.** The 9 that miss are *exactly* the
9 that saturate the press's control stop, and the separation is clean with no overlap:

| | |
|---|---|
| max peak among blows that reached | **105.9 kN** |
| min peak among blows that fell short | **110.1 kN** |

Depth raises the odds but does not decide it — commanded gap ≤ 20 mm is force-limited 39% of the
time, > 20 mm only 7%. **Force is what binds.**

⇒ **A fixed closure cap is the wrong model.** Applied to this session it would under-close 33 of
the 38 blows that the real press completed. It survives on the 17-hit sequence only because that
sequence commands deep closures (8.3–17.7 mm) where the limit binds every time — it is a fit to
the saturated regime, not the mechanism.

### 3.5.1 The sim already implements this mechanism — the problem is calibration

**Corrected 2026-08-13, same day, both errors mine.**

**The sim is not missing a force limit.** `strike_controller.py:671` runs
`cond_force = (force_L > max_force) | (force_R > max_force)` during PRESSING and halts with
`stop_reason = "Max Force"` — structurally the same control stop as the real machine's. The
threshold is `options.py:797`, `max_force = 200000.0` N. So the earlier statement here that the
sim "*cannot* be force-limited where the real press is" was wrong: it can, and today's hit 1
reached **F = [38.1, 192.5] kN against a 200 kN threshold — within 4% of tripping it.** The limit
may already bind occasionally, unnoticed.

🚨 **But the sim's force cannot calibrate that threshold**, so `max_force = 110200` must **not**
be set. Within a single hit-1 run:

| | force | vs real blow #1 (66.5 kN) |
|---|---|---|
| smooth peak | ~89 kN | 1.34× |
| maximum sample | **192 kN** | **2.9×** |
| left vs right die, **same step** | 38.1 vs 192.5 kN | **5× disagreement** |

"Peak force" is not a well-defined quantity here. Add the standing problems: force spans **3.3×
across press speeds and has never converged** even at 111× real die speed (§4.7), it is nearly
**blind to material** (10% flow stress → 1.1%) yet **sensitive to elasticity** (`nu` → 10%), so
it is dominated by contact and elastic terms rather than plasticity; and the one clean 1.34×
calibration was **retracted** as a hybrid-contact artifact (grid-only gave ~1.8×). Three
different sim/real ratios are in circulation and none is converged.

⇒ Setting the threshold to the real 110.2 kN would make the sim trip **early** and under-close
blows the real press completed — the opposite of the observed error, corrupting every downstream
geometry result.

**What would make it calibratable.** The limit does not need the absolute force to be right; it
needs the sim to saturate on **the same blows**. That is a correspondence question, and it is now
answerable for the first time: §3.4 yields **47 real peak forces**, where every previous force
comparison in this project used blow #1 alone — the *third* instance of the first-event failure
mode. Regress sim peak force against real peak force across many blows. If sim ≈ k × real with a
stable k, set the threshold at 110.2 × k **in sim units**. If the relationship is noisy or
non-monotonic, the limit is uncalibratable and this idea parks until the force model converges.

**And the material→geometry route is weaker than first claimed.** The earlier text here said a
force limit makes closure material-sensitive and called it a route nobody had considered. The
route is real, but it scales with the *same* weak coupling that makes force blind to material: if
10% flow stress moves force 1.1%, then near saturation it moves the stopping position by a
correspondingly small amount, further divided by the steep force–closure slope of a stiff
contact. Treat it as a second-order effect worth measuring, **not** as a new lever.

> ⚠️ **Before anyone changes `max_force`.** (1) 110.2 kN is measured on 06-15; it is a
> machine/control property so it *should* carry to the 17-hit sequence, but that is an assumption
> here. (2) It is shared config that changes behaviour for every consumer of this branch, and
> workstream A currently has **uncommitted edits to `options.py`**. **Coordinate; do not edit
> unilaterally.**

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

**Why this specifically blocks coupling.** Nairn's *Coupling Transport Equations to Mechanics in
the Material Point Method* analyses exactly this configuration and reports that FLIP transport
produces good **grid** values while **particle** values oscillate badly near edges and
interfaces — with the consequence, in his words, that modelling
> "cannot implement features that depend on a particle's transport value"

and he gives temperature-dependent material properties as the example. **That is the coupling
itself.** The published failure mode of the scheme in use is precisely the capability we want.
He also notes null-space effects are more severe in transport than in mechanics, which is why
momentum survives here and thermal does not.

**The published fix is neither FLIP nor pure PIC.** Pure PIC (equivalently FMPM(1)) eliminates
the oscillations but introduces numerical diffusion he judges unacceptable; **FMPM(k≥2–3)** —
an approximate full heat-capacity-matrix inverse — removes the oscillations without significant
diffusion. Note the codebase already uses **APIC** (a filtering transfer with the same
motivation) for momentum, and left thermal on raw 1988-era FLIP.

📚 Full reference should be added here once the paper is properly cited — see §13.

### 4.1.1 🎯 What the failure actually is — undershoot, not runaway

**Measured 2026-08-13.** "Thermal Detonation" is a misleading label. `options.py:811-812` fires
it on `temp > 4000 K` **or** `temp < 0 K` — opposite failures with different fixes. This document
previously refused to guess between them, and was right to: the answer was not readable from any
run on disk, for two separate reasons.

1. **The runner could not verify the arm.** `material_arms.py` checked the flip read-back against
   a `1e-9` tolerance. The field is `gs.qd_float` and Genesis runs precision 32, so 0.97 stores as
   `0.9700000286…`, 2.9e-8 out — the guard was satisfiable **only for values exactly representable
   in float32.** 0.0 and 0.5 passed; 0.97 aborted every time. That is why every 0.97 arm on disk
   records `thermal_flip_frac: null`: the flag was dropped rather than the guard relaxed, so the
   headline arms ran on the solver default, unverified. Tolerance widened to `1e-6`; the 0.97 arm
   now reads back `0.9700000286102295`.
2. **The forensic trace is blind to this failure.** `strike_controller.py` locates the offending
   particle through `_per_particle_field_cpu`, which at line 35 does
   `t.reshape(-1, n_particles)[0]` — keeping only the **first** frame buffer — while the trigger
   at `:1657` tests the whole tensor. Particle state is double-buffered, and the FLIP blend
   itself (`base_mpm_solver.py:745`) writes to `particles[f + 1]`. The intercept therefore fires
   on a value the trace cannot find, `bad_indices` comes back empty, and no `GROUND ZERO` block
   is printed. 🚩 **Every "Thermal Detonation" this project had recorded was unattributable for
   this reason.** ✅ **Fixed by B-3 the same day (`bc850e82`)** — the helper now reduces *across*
   frame buffers, keeping the largest-magnitude value per particle, so both a negative undershoot
   and a runaway survive the collapse. Runs from `bc850e82` onward should print a populated
   `GROUND ZERO` block; **anything recorded before it still carries the blind spot.**

Reading `solver.particles.temp` directly, across every buffer, at the moment of failure
(JC, 1273.15 K, res 7, thermal ON, flip 0.97 verified, shape `(2, 3160, 1)`):

| | frame 0 | frame 1 |
|---|---|---|
| min | 687.16 K | **−2.99 × 10⁸ K** |
| max | 1267.64 K | 1290.37 K |
| n < 0 | 0 | **1,284 of 3,160 (40.6%)** |
| n > 4000 K | 0 | **0** |
| NaN | 0 | 0 |

⇒ 🎯 **The failure is one-sided undershoot to negative temperature. It is not runaway.** Not one
particle exceeds 4000 K, and there are no NaNs. Frame 0 is pristine and byte-identical to its
post-hit-1 value, so the collapse happens within the first substeps of hit 2 rather than
accumulating across the hit. Offending values span −16.6 K to −3 × 10⁸ K — an amplified
oscillation, not a single bad transfer.

This is the signature Nairn describes above, and it closes §4.2's loop. Undershoot needs a
**gradient** to amplify: after hit 1 the field spans 669–1268 K from die chill, which is ample.
At a 293 K billet the field sits flat at ambient, there is no gradient, and the run survives —
which is precisely why 293 K was the only temperature that did.

⚠️ **What is established is that the failure is undershoot and that the flip fraction controls
it. The route by which FLIP grows the excursion without bound remains a model, not a
measurement.** Do not upgrade it further without evidence.

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

### ✅ A5 — the noise floor, measured at n = 5 (2026-08-13)

Five replicates of one config (`m1_jc_1273`, res 7, thermal on), each in its **own process**, to
match how the original n = 2 pair arose:

| replicate | IoU@2.0 | dev_mean | dev_p95 | dev_max |
|---|---|---|---|---|
| rep1 | 0.7641 | 0.415 | 0.955 | 1.812 |
| rep2 | 0.7642 | 0.415 | 0.955 | 1.812 |
| rep3 | 0.7639 | 0.414 | 0.955 | 1.812 |
| rep4 | 0.7645 | 0.415 | 0.955 | 1.812 |
| rep5 | 0.7644 | 0.415 | 0.955 | 1.812 |

| | |
|---|---|
| mean | 0.76422 |
| **σ** | **0.00024** |
| **full range** | **0.00060** |
| 2σ / 3σ | 0.00048 / 0.00072 |

**The inherited 0.0002 was right in magnitude — it was a ~1σ difference — but it has been used
as a *detection threshold*, and 1σ is not one.** A 1σ bar calls roughly a third of pure-noise
comparisons significant. The range at n = 5 is **0.0006, three times the quoted figure.**

Re-reading the existing results against σ = 0.00024:

| comparison | Δ IoU | in σ | verdict |
|---|---|---|---|
| JC vs clamped-Arrhenius | 0.0002 | **0.8σ** | indistinguishable — **conclusion strengthened** |
| `m3_arr_raised` | 0.0038 | 15.9σ | real |
| thermal solver OFF | 0.0219 | 91.7σ | real |

⇒ Nothing previously called "real" becomes noise, and the one previously called "exactly noise"
is confirmed. What changes is the **threshold** — see §9.4, criterion 7.

🚩 **`dev_p95` and `dev_max` were identical across all five runs** at printed precision, and
`dev_mean` varied by 0.001 mm. On this evidence the deviation metrics are *more* reproducible
than IoU, which is worth knowing before choosing a discriminator — though 3-decimal output floors
what can be claimed.

⚠️ Still n = 5 at **one** config and **one** hit. Non-determinism could scale with particle count,
hit number, or instability proximity; a late-hit replicate set would be the natural follow-up.

**Superseded — the original n = 2 reasoning, kept because its diagnosis was right:**

It comes from two accidental replicate pairs (`m0` vs `m0_seq` = 0.0003; `m1` vs `m1_seq` =
0.0001). Those pairs differ only in `--n-hits`, which cannot affect the state at hit 1, so in a
fully deterministic solver they should be **bit-identical**. They are not.

The most likely explanation is **GPU non-determinism in the particle-to-grid scatter** —
floating-point atomics accumulate in non-deterministic order, which is a known property of
GPU MPM implementations rather than anything specific to this code. ⚠️ *Plausible but not
verified here.*

Two consequences:

1. If it is atomics, it is a genuine **irreducible floor for a single run** — you cannot average
   it away within one run, only across replicates.
2. ~~**n = 2 is far too thin for a number this load-bearing.**~~ ✅ **Done above at n = 5**; the
   estimate held at ~1σ and the atomics hypothesis is supported (the five runs are close but not
   bit-identical). The original text follows — n ≥ 5 replicates of
   one config, reporting the standard deviation, and separately confirming whether two runs of
   an identical config are bit-identical (which would localise the source).

Until then, treat "above the noise floor" claims as **provisional**, and prefer effects that are
≥10× the quoted floor — which the FLIP result (0.02 IoU, ~100×) and the thermal-off result
(0.0219, ~100×) comfortably are, and which the material-arm differences (0.0002, ~1×) are not.

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
best arm measured                    = 0.7887      ← ~98% of ceiling
```

⚠️ The two ceilings are not strictly independent, so 0.801 is an **estimate**, not a bound. And
because the real bar is slightly *wider* than the sim in y (40.48 vs 40.0 mm), `real ⊄ sim`,
which makes the true ceiling slightly **lower** — the conclusion only strengthens.

`REAL_STOCK_RADIUS_MM = 20.0` is a bare, underived constant in shared
`forge_common/main/forge_common/real_scale.py:47`. The real bar is ordinary **1.5″ stock**
(37.93 mm equivalent diameter).

### 🚨 CROSS-WORKSTREAM CORRECTION (workstream A, session `40379be1`, 2026-08-12)

**The +9.9%/+11.2% volume excess is independently CONFIRMED. What I proposed doing about it is
REFUTED — measured, not argued.**

Four billet initial conditions were run with the grid held identical (dx 4.0 mm, psize 2.0 mm),
so volume was the only variable:

| billet ⌀ | IoU@2.0 |
|---|---|
| **40.0 mm** (current) | **0.7672** |
| 39.3 mm | 0.7536 |
| **38.0 mm** (≈ the 18.96 radius I proposed) | **0.7077** |
| mesh-from-scan | 0.7135 |

**IoU falls monotonically as volume falls.** The ceiling rises while the score drops, because
`IoU_max = V_real/V_sim` assumes an over-filled sim that **nests around** the real bar.
⇒ **IoU@2.0 rewards over-filling and therefore cannot be used to choose the stock volume.**
My "at minimum, correct `REAL_STOCK_RADIUS_MM` 20.0 → ~18.96" is **withdrawn.**

**And a cylinder is the wrong SHAPE at any radius.** The hit-1-*before* scan is already
flattened — a **5.6 mm flat** (Z 34.4 mm across a Y of 40.4 mm) present *before* the recorded
sequence begins. The body radius (~20.4 mm) and the whole-bar volume-equivalent (~19.0 mm)
diverge because the ends taper, so **no single radius satisfies both**; ⌀38.0 is ~5% too thin
in the struck region, which is why its surface deviation is the worst of the four.

⚠️ **This also refines this document's own §5 entry.** My cross-section analysis found a smooth
oval with r ≈ 18.3–21.4 mm and no flats — but it sampled at **mid-length**, and the whole-bar
AABB (Z 39.66 mm) is dominated by unstruck material. A-7's 34.4 mm is the **local struck
region**. Both measurements are probably right and describe different places on the bar; the
flat is **localised, not a uniform section**. Neither has been reconciled against the other
directly. **Do not treat either as the settled cross-section.**

⚠️ Also unverified from A-7: a suspected **die over-closure at hit 1** (real gap 34.5 mm vs sim
30.0 mm). If real, it affects every hit-1 comparison in this document.

⇒ **Do not edit `real_scale.py` on the strength of the ceiling argument.** It is shared by four
adapters, and the one measurement anyone has taken says the change moves the score the wrong
way. See `genesis-contact-method-work` and A-7's canonical doc.

**Consequence for this workstream:** hit-1 geometry is saturated. Material and thermal
differences have almost no room to express themselves there. Until the stock geometry is fixed,
prefer **differential** comparisons (arm vs arm at fixed everything) over absolute scores, and
prefer **accumulated multi-hit** geometry over hit 1.

### 4.7 Peak force is nearly blind to the material — this is why geometry became the metric

| Change | Effect on peak force |
|---|---|
| Flow stress −10% (JC 201.4 → Arrhenius 181.2 MPa) | **1.1%** |
| Purely **elastic** change to `nu` | **10%** |

Force is ~10× more sensitive to an elastic parameter than to a 10% change in the plastic flow
stress. ⇒ **The inherited "3.02× → 1.81× → 1.34×" force-convergence narrative measured contact
and elasticity, not material. Those ratios are retracted.**

Compounding this, the real press is **force-limited at 110.2 kN**, so measured force saturates
and cannot discriminate above that ceiling. Geometry is the only metric that sees the material —
which is why §4.5's characterisation of it is load-bearing.

### 4.7.1 🚨 "Peak force" in this sim is a numerical artifact, not the physical load

**Measured 2026-08-13.** The per-step force trace on hit 1 (JC, 1273.15 K, res 7, grid contact),
logged every third step:

| steps | force L / R | reading |
|---|---|---|
| 12–18 | rise to **89.2 / 88.1 kN** | almost certainly the **approach impact** — `approach_speed` is 273 m/s (§4.4) |
| 21–27 | decay to 50.1 / 49.5 kN | |
| 30–39 | **plateau ≈ 57 kN**, dies agreeing to **0.3%** | the physically meaningful part |
| 42–48 | **76.6/91.4 → 155.2/141.6 → 38.1/192.5 kN** | terminal transient; die asymmetry blows out to **5×** |

**This is deterministic, not noise.** Two independent runs reproduce it to 0.06%
(192,467 vs 192,583 N at the peak), so the shape is a property of the model.

Against the real blow #1 — a **monotonic** rise over 0.505 s to a single peak of **66.5 kN** —
three things follow:

1. **The shape is wrong.** Real is monotonic-to-peak; sim is impact → decay → plateau → terminal
   spike. A scalar "peak force" comparison between the two is comparing a physical maximum
   against a numerical one.
2. **The sustained force is actually reasonable**: the ~57 kN plateau is **0.86×** the real
   66.5 kN peak. The 1.34×–2.9× over-prediction quoted from peak samples is dominated by the
   terminal transient, not by the constitutive response.
3. ⇒ **Every force calibration in this project's history is suspect on this basis alone** —
   2.56×, ~1.8×, the retracted 1.34× — independently of the contact-mode confound already known.
   They were all peak-based, and all on blow #1.

⇒ **A force stop cannot be calibrated against this trace** (§3.5.1). The stop is
`(force_L > max_force) | (force_R > max_force)` — an **OR over individual dies, evaluated per
step**, so it keys on the single noisiest quantity available. With a physical plateau near 57 kN
and a terminal spike near 192 kN, any threshold in between fires on the artifact at the end of
every press, regardless of material. A threshold at the real 110.2 kN sits squarely in that gap.

**What to do instead.** Define the comparison basis before comparing: sustained force at matched
reduction, not peak. Both defects behind the artifact are already known and separately actionable
— the 273 m/s `approach_speed` against a 100 m/s intercept (§4.4), and whatever produces the
PRESSING→HOLDING transient, which has not been diagnosed.

> 🚩 The 5× die asymmetry at step 48 is unexplained and is recorded, not rationalised. It appears
> only in the terminal transient; through the plateau the two dies agree to 0.3%.

#### 4.7.2 🚨 And the press is already stopping on that transient

Three independent runs of hit 1 end identically:

```
Strike -> HOLDING (Max Force, strain=0.2130, steps=49)
```

**Not `Target Strain`. `Max Force`.** Same stop reason, same strain to four decimals, same step,
in every run — the terminal transient of §4.7.1 crosses the 200 kN threshold at step 49 and halts
the press.

This is not the adapter's intent. `genesis_forge_adapter.py:327` triggers the strike with a
placeholder `0.95`, then `:338` sets the real target strain mid-stroke once both jaws touch and
`W_contact` is known, from `hit.rho`. The design is **position** control against the commanded
gap; the force limit is meant as a backstop. The adapter even warns explicitly when `rho` is
below the jaws' mechanical stop — that warning did **not** fire here, so the commanded gap was
geometrically reachable.

⇒ **In this configuration the sim's hit-1 closure is set by where a numerical artifact crosses
200 kN, not by the commanded reduction.** Every geometry number produced in this configuration
inherits that. This is the most consequential item in §4.7, and it was invisible for as long as
nobody read the stop reason.

**The target strain was set correctly — this is not a misconfiguration.** The adapter's
diagnostic fires in every run:

```
[genesis adapter] PRESSING target_strain=0.2484 (W_contact=64.000mm, jaw_pair_thickness=16.000mm)
```

Commanded 0.2484, achieved 0.2130. Working back through
`strain = 1 − (rho + jaw_thickness) / W_contact`:

| | strain | jaw centres | **face gap** |
|---|---|---|---|
| commanded | 0.2484 | 48.102 mm | **32.102 mm** |
| achieved | 0.2130 | 50.368 mm | **34.368 mm** |

The derived command of **32.102 mm** matches workstream A's independently-read `Hit.rho` of
**32.100 mm**, which confirms the derivation. **The press stopped 2.27 mm short of its command.**

🎯 **And that lands 0.13 mm from the real machine.** A-7 measured the *real* press finishing
hit 1 at **34.50 mm** — 2.4 mm short of the same command. This run finishes at **34.37 mm**.
Both replay the same 17-hit sequence, so it is a like-for-like comparison.

⚠️ **Do not read that as "the sim is right."** It is n = 1, one hit, and the sim's gap here is
*derived from the controller's strain* rather than measured from the particle cloud — while A-7
*measures the cloud* and gets ~32.3 mm corrected for the same nominal quantity.

#### 4.7.3 The stop is per-hit and configuration-dependent

**Corrected 2026-08-13 (rev 15). The error was mine.** What stood here said the two sim numbers
"differ by ~2 mm and **cannot both be right**", and called reconciling them the single most
important open item. **The cheap test this section itself prescribed refutes the framing.**
Running `grep 'Strike -> HOLDING'` over a complete 17-hit log already on disk
(`~/profile_g0_17.log`, 2026-08-11) shows the stop reason changing *from hit to hit inside one
run*:

| hit | `W_contact` | commanded | achieved | shortfall | stop |
|---|---|---|---|---|---|
| 1 | 63.600 | 0.2437 | 0.2437 | — | **Target Strain** |
| 2 | 63.600 | 0.2903 | 0.2451 | 0.045 | Max Force |
| 3 | 63.600 | 0.3318 | 0.3325 | — | **Target Strain** |
| 4 | 63.600 | 0.3836 | 0.3719 | 0.012 | Max Force |
| 5 | 58.000 | 0.3100 | 0.3113 | — | **Target Strain** |
| 6 | 58.000 | 0.3672 | 0.3208 | 0.046 | Max Force |
| 7 | 63.600 | 0.3802 | 0.3593 | 0.021 | Max Force |
| 8 | 63.601 | 0.2006 | 0.2035 | — | **Target Strain** |
| 9 | 63.600 | 0.2950 | 0.2950 | — | **Target Strain** |
| 10 | 55.200 | 0.3710 | 0.2935 | 0.078 | Max Force |
| 11 | **49.600** | 0.3061 | 0.2141 | **0.092** | Max Force |
| 12 | 60.800 | 0.3336 | 0.3349 | — | **Target Strain** |
| 13 | 60.800 | 0.4204 | 0.4137 | 0.007 | Max Force |
| 14 | **49.600** | 0.3306 | 0.1951 | **0.136** | Max Force |
| 15 | 63.600 | 0.2619 | 0.2622 | — | **Target Strain** |
| 16 | 60.801 | 0.3217 | 0.3231 | — | **Target Strain** |
| 17 | 58.000 | 0.4310 | 0.3062 | 0.125 | Max Force |

⇒ **The limit binds on 9 of 17 hits and not the other 8, within a single run**, so two workstreams
reporting different answers to "did the press reach its command" can *both* be correct. The
configuration candidate is also evidenced rather than speculative: this run presses hit 1 at
`W_contact = 63.600 mm` and **reaches its command exactly**, while the runs analysed in §4.7.2
press at `64.000 mm` and their hit 1 is force-limited.

🎯 **And the mechanism is contact width, not commanded reduction.** Sort the table by
`W_contact`: **every hit at ≤ 55.2 mm saturates (3 of 3), and the shortfall grows as the contact
narrows** — 0.078 at 55.2, then 0.092 and 0.136 at 49.6. At 63.6 mm, 5 of 8 reach command. This
is not confounded with how hard the hit was: hit 13 commands the *largest* reduction in the run
(0.4204) at a wide 60.8 mm and lands within 0.007, while hit 14 commands a milder 0.3306 at
49.6 mm and misses by 0.136 — twenty times worse for a smaller ask. A narrower contact patch
carries the same die force over less area, so it reaches the threshold earlier in the stroke.
⇒ **the force stop preferentially truncates the narrow-contact hits**, which are exactly the
edging/drawing passes where the geometry change is largest. Across the run, mean `W_contact` is
**58.0 mm on the truncated hits against 62.2 mm on the ones that reached command**.

**Reproduce on any run log:** `python3 stop_reason_table.py ~/profile_g0_17.log` (committed at
repo root). It reads only the two lines the adapter and controller already emit, so it works on
logs already on disk — no re-running. **Any run whose stop reasons are all `Target Strain` was
never force-limited; one with `Max Force` hits is truncated on those hits and its geometry is not
comparable to a run that completed them.**

🚩 **Note which hit is the exception.** Hit 1 is one of the hits where the limit did *not* bind —
and every number in §4.7.2 is hit-1-only. That is §11's first-event pattern again, running the
other way: here hit 1 is unrepresentative by being *better* behaved than its neighbours, not
worse.

⚠️ **The sim saturates far more often than the machine.** 9 of 17 here (53%) against 9 of 47
real blows (19%, §3.5). ⚠️ **Denominators are not matched** — the real 9 are counted across the
whole session while these 17 are one replayed sequence, and the per-hit correspondence has not
been established. The *direction* is the robust part.

**What is still open is narrower than "one of us is wrong":** whether jaw-gap and cloud-span
measure the same quantity, and whether the contact-width mechanism holds outside this one run.

⚠️ **`max_force` is no longer one constant across worktrees.** `aims-genesis/nsf-demo` now reads
it from the environment — `max_force: float = float(os.environ.get("AGF_MAX_FORCE", 200000.0))`
(`nsf-demo/agforge/options.py:452`, uncommitted at time of writing). This worktree still has the
plain literal. **A run's effective threshold now depends on which worktree and which environment
it ran in**, which any cross-workstream force comparison has to control for.

📌 **Second-hand, recorded as a pointer not a finding.** Workstream A-8 reports mid-batch that
arms `p1_particle` and `p3_pg2p_pos` complete 17/17 with the force stop relaxed, having failed at
hits 14 and 10 respectively with it active — same mesh IC, only the stop differs. **Not verified
here**, and their batch was still running. If it holds, the force stop is upstream of stability
failures this project has been attributing to the material and the timestep, which would matter
considerably more than the closure number. **Workstream A owns it; do not open a second front.**

⚠️ Do **not** conclude from this that lowering `max_force` to the real 110.2 kN would help. It
would move an already-active stop from one point on an artifact to an earlier one. §3.5.1's
argument stands: the trace cannot calibrate the threshold. What changes is the urgency — the
threshold is not dormant, it is load-bearing today.

### 4.8 The material card is in-domain at the measured temperature

The billet was **measured at ~960 °C at blow #1** (06-15 mcap; session `12b6fa7e` — see §11 for
its caveats). Song2020 is fitted over **800–1000 °C**; the Arrhenius kernel clamps to
[1073.15, 1473.15] K.

**At blow #1 that is comfortably inside both — nothing extrapolates and nothing clamps.** The
card is calibrated at 1000 °C and the bar runs 40 °C cooler. Flow stress 212.4 MPa at 960 °C vs
181.1 at 1000 °C.

🚨 **But this is a blow-#1 statement, and §3.4 now shows blow #1 is the 96th percentile of its
session.** Across all 47 blows:

| | |
|---|---|
| session median | **823.6 °C** — 24 °C above Song's floor, not 40 °C below its ceiling |
| **below the 800 °C fit floor** | **18 of 47 = 38%** |
| in domain | 29 of 47 |
| above 1000 °C | **0** |
| minimum | **615.2 °C** — 185 °C below the fit |

⇒ **The corrected claim: the card never extrapolates upward, and the clamp demotion stands.
"In domain" holds for the hot end of the session and fails for over a third of it.** Song's fit
is being read below its floor on 38% of blows, and the Arrhenius kernel's lower clamp
(1073.15 K = 800 °C) is therefore *active* on those blows rather than dormant — flow stress is
being evaluated at 800 °C for a bar that is sometimes 185 °C colder, which **under-predicts**
strength exactly where the metal is strongest.

This remains the strongest validation the card has, and it still came from a measurement rather
than a simulation. What changes is its scope: it validates the card **at the top of the
session**, and identifies a live extrapolation at the bottom.

⚠️ The reheat target for the 17-hit sequence (~900–1000 °C, §3.1) *is* inside the fitted window.
The 06-15 session is not the same experiment and runs substantially cooler (§3.4) — **do not
merge the two temperature distributions.**

### 4.9 Primary data

**Geometry, all arms, hit 1** (`geom_batch.py --batch material_arms --hit 1 --sim-hit 1`).
Real scan: 126,898 verts / 252,184 tris / 67,486 mm³. `pose = IoU cen − IoU`; positive means
that arm has pose drift.

| arm | res | thermal | IoU@2.0 | IoU cen | dev_mean | dev_p95 | dev_max | pose |
|---|---|---|---|---|---|---|---|---|
| `m0_jc_293` | 7 | on | 0.7642 | 0.7741 | 0.415 | 0.955 | 1.812 | +0.0099 |
| `m0_jc_293_seq` *(replicate)* | 7 | on | 0.7645 | 0.7746 | 0.415 | 0.955 | 1.812 | +0.0101 |
| `m1_jc_1273` | 7 | on | 0.7642 | 0.7743 | 0.415 | 0.955 | 1.812 | +0.0101 |
| `m1_jc_1273_seq` *(replicate)* | 7 | on | 0.7641 | 0.7740 | 0.415 | 0.955 | 1.812 | +0.0099 |
| `m2_arr_clamped` | 7 | on | 0.7644 | 0.7734 | 0.412 | 0.953 | 1.812 | +0.0090 |
| `m3_arr_raised` | 7 | on | 0.7604 | 0.7504 | 0.432 | 1.001 | 1.993 | −0.0100 |
| `t_900C` | 7 | on | 0.7716 | 0.7514 | 0.409 | 0.956 | 1.812 | −0.0202 |
| `t_1000C` | 7 | on | 0.7656 | 0.7716 | 0.413 | 0.953 | 1.812 | +0.0060 |
| `t_1000C_mech` (S_T=1,773) | 7 | on | 0.7644 | 0.7732 | 0.413 | 0.954 | 1.812 | +0.0088 |
| `t_K_either` | 7 | on | 0.7776 | 0.7429 | 0.421 | 0.993 | 1.812 | −0.0346 |
| `m0_jc_293_res10` | 10 | on | 0.7691 | 0.7655 | 0.315 | 0.743 | 2.104 | −0.0037 |
| `t_1000C_res10` | 10 | on | 0.7668 | 0.7642 | 0.320 | 0.761 | 2.200 | −0.0026 |
| **`t_1000C_nothermal`** | 10 | **off** | **0.7887** | 0.7843 | **0.297** | **0.717** | **1.278** | −0.0043 |

Read this table with §4.5 in hand: **the res-7 and res-10 blocks are not directly comparable.**
The `_seq` pairs are accidental replicates of `m0`/`m1` and give the **run-to-run noise floor of
0.0002 IoU**. Within res 7, `m3_arr_raised` is −0.0038 (≈19× noise, real signal, but it ran at
1473 K — a regime this forge never reaches), while JC and clamped-Arrhenius differ by 0.0002,
i.e. **not at all**.

**Sequence stability, consolidated.** Hits survived out of 17.

| Billet | res | thermal | flip | Hits | Failure |
|---|---|---|---|---|---|
| 1000–1273 K | 7 | on | 0.97 | 1 | Thermal Detonation |
| 1273 K | 10 | on | 0.97 | 1 | NaN Detected |
| 600 K | 7 | on | 0.97 | 1 | Thermal Detonation |
| 1200 K | 7 | on | 0.97 | 1 | Thermal Detonation |
| 293 K | 7 | on | 0.97 | 7 | Thermal Detonation |
| 293 K | 10 | on | 0.97 | 11 | Thermal Detonation |
| 1273 K | 7 | on | **0.00** | **12** | Supersonic Velocity |
| 1200 K | 7 | on | **0.00** | **13** | Supersonic Velocity |
| 1273 K | 10 | **off** | — | **14** | Supersonic Velocity |

⚠️ **Partial sequence failure is normal here** — the contact sweep's own arms complete
1/2/2/2/3/3/5/5/7/7/7/7/8/10/13/17 hits. Do not treat a non-17 run as anomalous on its own.

### 4.10 🎯 Scoring at later hits — the signal is there, we were looking at the wrong hit

**Every geometry score in this project had been taken at hit 1.** Six arms already carry
multi-hit particle clouds on disk (up to 14 hits); they had simply never been scored. Doing so
(pure CPU, no re-runs needed) changes the picture materially.

| arm | res | hit 1 | hit 5 | hit 10 | `dev_max` h1 → h10 |
|---|---|---|---|---|---|
| `m0_jc_293_res10` | 10 | 0.7691 | 0.7221 | **0.6155** | 2.10 → 14.19 mm |
| `t_1000C_nothermal` | 10 | 0.7887 | — | **0.5910** | 1.28 → 17.88 mm |
| `m0_jc_293_seq` | 7 | 0.7645 | 0.6715 | — | 1.81 → 5.26 (h5) |
| `m1_jc_1273_T1200_pic` | 7 | — | 0.6735 | **0.5707** | → 15.40 mm |
| `m1_jc_1273_T1273_f00` | 7 | — | 0.6652 | **0.5621** | → 15.37 mm |
| `m1_jc_1273_v35_f00` | 7 | — | 0.6739 | — | → 5.17 (h5) |

**Two findings.**

**(a) Agreement degrades sharply as the sequence accumulates: ~0.77 → ~0.67 → ~0.57**, and
`dev_max` grows roughly **10×** (1.8 → 15+ mm). Errors compound; they do not wash out. This had
never been measured.

**(b) 🎯 Discriminating power GROWS with hit number — by roughly 20×.** Spread across the res-7
arms:

| | hit 1 | hit 5 | hit 10 |
|---|---|---|---|
| spread across res-7 arms | **0.0004** | **0.0087** | **0.0086** |
| vs the (provisional) 0.0002 noise floor | ~2× | **~40×** | **~40×** |

At hit 1, configs differing in billet temperature or approach speed are indistinguishable. By
hit 5 the same pairs separate by ~0.009. Two examples, each differing in **one** variable:

- `T1273_f00` (0.6652) vs `v35_f00` (0.6739) — **approach speed only**, 273 vs 35 m/s → Δ 0.0087
- `T1273_f00` (0.6652) vs `T1200_pic` (0.6735) — **billet temperature only**, 1273 vs 1200 K → Δ 0.0083

🚩 **Note what this does to §4.7's press-speed conclusion.** Approach speed did not change hit
counts and did not change hit-1 geometry — but it visibly changes *accumulated* geometry. "Press
speed doesn't matter" was a hit-1 statement and should not be generalised.

### 🚩 This partially corrects §4.6 — the ceiling is a HIT-1 phenomenon

§4.6 argues hit-1 geometry is saturated at ~98% of a structural ceiling, leaving material and
thermal effects no room. **That is true at hit 1 and false later.** At hit 10 scores sit at
0.56–0.62 against a discretisation ceiling near 0.88 — **enormous headroom.**

⇒ The right response to the ceiling is **not only** "fix the stock geometry." It is **"stop
scoring exclusively at hit 1."** Later hits are cheap (the clouds exist), have room to move, and
discriminate ~20× better. Any comparison intended to detect a material or thermal effect should
be scored across hits 1 / 5 / 10 and reported as a curve, not a point.

⚠️ Cross-arm comparisons at late hits carry a caveat: arms that died earlier are absent, so the
surviving set is selected for stability. Compare like with like, and state which arms were in
the pool.

#### 🚩 The decay curve has an alternative explanation, and it has NOT been ruled out

Finding (a) above is stated as *"errors compound."* **That may be wrong.** A-7 measured that the
real bar arrives with a localised **5.6 mm flat before hit 1**, while `init_stock()` builds a
round cylinder. Sim and real therefore start from **different shapes at step zero**, and that
mismatch would also produce a monotonically growing divergence — **indistinguishable from the
curve measured here.**

So the 0.77 → 0.57 decay is consistent with *either*:

1. genuine accumulation of physics/numerical error, or
2. a fixed initial-condition mismatch propagating and amplifying through the sequence.

**Nothing here separates them.** Hit-to-hit *action* alignment is sound — `load_real_hits()`
returns hits 1-1 and in order, verified — so the divergence is in the state, not the schedule.

⇒ **Discriminating test:** run one arm initialised from the real `V_before` mesh and re-score
the curve. If the slope flattens, the decay was mostly (2) and this workstream's error budget
is dominated by the starting geometry rather than by the coupling. If it does not, (1) stands.
**Do this before treating the decay slope as a physics validation metric (§9.2).**

⚠️ Finding (b) — that *discriminating power grows* — is **not** affected by this ambiguity. It
is a within-hit comparison between arms that share the same initial condition, so any common IC
error cancels. **(b) is safe; (a) is provisional.**

---

### 4.11 🎯 The sim's thermal boundary conditions cannot reach the measured cooling rate — and emissivity is not the lever

§3.4 measures **−6.03 °C/s** median within-bout cooling, the only real thermal ground truth this
project has. A lumped-cylinder estimate at the sim's *own* shipped parameters
(`cooling_budget.py`; ε = 0.40 `options.py:717`, h = 15, `Cp` from `get_steel_cp`, 38.1 mm bar):

| case | °C/s | vs measured |
|---|---|---|
| **sim parameters** (ε 0.40, h 15) | **1.05** | **0.17×** |
| literature oxidised 316L (ε 0.80, h 15) | 1.81 | 0.30× |
| ε 0.80 + forced convection h 50 | 2.47 | 0.41× |
| ε 0.80 + 2% of surface in die contact | 2.94 | 0.49× |
| ε 0.80 + 5% of surface in die contact | 4.63 | 0.77× |
| **measured (§3.4)** | **6.03** | — |

🚨 **No emissivity closes this gap.** Solving for the value that would reach 6.03 °C/s with no
conduction gives **ε = 3.01**, against a physical maximum of 1.0. Free-surface radiation plus
convection is short by a factor of ~3 even when both are set as favourably as physics allows.

⇒ **This substantially demotes the emissivity question for cooling.** Emissivity has absorbed a
lot of effort in this project (§14: the Optris calibration hunt, the ±50 K band). It remains the
dominant uncertainty on *absolute temperature* — but it is **not** the lever on cooling *rate*,
and it cannot be made into one.

**What the candidates actually are, in order:**

1. **Die/manipulator contact conduction.** `thermal_contact_conductivity = 3000 W/(m²K)` exists
   (`options.py:705`) and a contact fraction of only ~5% at ε = 0.80 recovers 77% of the measured
   rate. This is where calibration effort belongs.
2. 🚩 **Surface versus bulk — a genuine confound in the comparison, not just in the model.** The
   camera reads a **radiometric surface** temperature; this estimate is **lumped**. Over the ~5 s
   between blows the diffusion length is √(αt) ≈ 5.5 mm against a 19 mm radius, so a real surface
   gradient develops and the surface cools faster than the bulk. **Part of the factor of 5.7 is
   therefore this mismatch rather than missing physics, and this analysis cannot separate them.**

⚠️ Note the lumped assumption biases *against* the conclusion, not toward it: internal gradients
slow surface-limited bulk cooling, which would widen the gap further. The direction is safe even
though the magnitude is not.

**Acceptance target for the coupled model (§9.2):** reproduce **−6.03 °C/s** within a bout, and
the 12-bout spread of −11.58…+2.71. A model that gets there on radiation alone is wrong even if
the number matches.

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
| Surface/contact CFL is the cause, θ = 36 | B-3, self-retracted | 1000× units error — `_particle_volume_scale` applied twice. B-3's corrected θ is 0.036; this session's independent full-cell estimate was 0.025 — same order, both far below the θ>2 divergence threshold. |
| "JC is temperature-blind, therefore not mechanics" | both B sessions, from a stale comment in `material_arms.py` | True only of the *initial* temperature. Plastic work un-clamps `T_star` on the first blow. |
| Temperature at the blow is not measured | inherited, long-standing | Measured ~960 °C from the 06-15 mcap (session `12b6fa7e`). |
| "We use grid-only contact" | repeated across sessions | Actually **grid + CPIC** — `enable_CPIC=True` is passed explicitly at `options.py:683`, overriding Genesis's own default of `False`. All results in §4.9 are grid+CPIC with `enable_particle_contact=False`. |
| The instability is the ~97× thermal-clock discrepancy | prior session | `thermal_time_scale_mode='mechanical'` was verified applied (S_T 237,014 → 1,773) and changed survival not at all. The discrepancy is real and worth fixing (§6.2) — it is simply not the instability. ⚠️ Note the **ratio is 134×, not 97×**: see the S_T note below. |
| S_T is 171,653 and the clock is ~97× off | `options.py` comment block, and `material_arms.py`'s docstring | Both predate `cfl_use_pwave = True` (`options.py:115`). `substep_dt = 0.9·dx/c` and `S_T = 0.25·dx²/(6·α·substep_dt)`, so **S_T ∝ c**. Switching from the bar-wave speed to the p-wave speed raises c by 1.3808× (4,070 → 5,620 m/s at the shipped card), and 171,653 × 1.3808 = **237,015**. The stale figures are self-consistent with each other and with an older CFL criterion; the shipped value is 237,014 and the ratio to N is **133.7×**. ✅ **B-3 dated the in-source commentary the same day (`d2d7c701`)**, so the 97× figure is no longer presented as current. |
| The sequence failure is Arrhenius-specific | prior session | Johnson-Cook dies identically at hit 2 under the same conditions. |
| "They complete 17 hits and we don't" | prior session | The contact sweep's own arms complete 1/2/2/2/3/3/5/5/7/7/7/7/8/10/13/17. Partial failure is normal; a non-17 run is not anomalous on its own. |
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
| 4 | **Thermal expansion absent** | no CTE in the solver | ~1.3% linear / 3.8% volumetric strain over a 700 K excursion — comparable to the deformation being measured. Harmless isothermal; an error the moment coupling is on | this workstream |
| 5 | **Elastic moduli temperature-independent** | `E`, `nu` constants (`options.py:50, :80`) | `E = 121.5 GPa` is the *hot* value, so this is correct isothermally and wrong under a varying field | this workstream |
| 6 | **Runtime α 1.82× the design α** | `1.1e-5` vs `6.05e-6` | `k_diff` 0.458 vs intended 0.25 — most of the margin gone | unclaimed |
| 7 | **`approach_speed` = 273.2 m/s** vs a 100 m/s intercept | `options.py:940` — uses macro `dt`, not `substep_dt` | Real defect; **not** the thermal cause | unclaimed |
| 8 | **No KE/IE assertion** | — | The validity of the whole time-scaling scheme is unmonitored | this workstream |
| 9 | Stock volume +9.9% | `real_scale.py:47` | Caps absolute geometry agreement at ~0.80 IoU | A-7 |
| 10 | "Thermal Detonation" is a misleading name | `strike_controller.py:1667` | Checks `T > 4000 K` **or** `T < 0 K` — fires on numerical collapse, not overheating | cosmetic |

---

## 8. Implementation plan

### 8.0 Dependency structure — what is actually blocked

The phases below are **not** a linear chain, and reading them as one would idle this workstream
behind another session. What is genuinely blocked is narrower than it looks:

```
A1-A5  Instrumentation ────────────┐        ours, no dependencies
B      Scaling unification ────────┤        ours, no dependencies
                                   ├──► D1a  mechanical N-invariance   ✅ RUNNABLE NOW
                                   │
A6     Coupled path (§2.4) ────────┤        ours, no dependencies -- but MISSING TODAY
                                   │
C      Gather fix (B-3's) ─────────┴──► D1b  coupled N-invariance      ⛔ needs A6 AND C
                                        D2-D4, §9.2 thermal tests      ⛔ needs A6 AND C

E      External validation ──► RUNNABLE NOW — 06-15 mcap is local (§3.2)
```

**Only the *coupled* validation is blocked on B-3** — and even that needs **A6** first, which is
ours. Phases A and B and the mechanical half of the headline test depend on nothing outside this
workstream.

🚨 **A6 is the quiet blocker.** There is no code path today that runs a forging press with
evolving temperature (§2.4). Fixing the gather makes particle temperature *usable*; A6 is what
makes it *used*. If only one of the two lands, coupled forging still does not run.

⇒ **Do A + B + D1a first.** If D1a fails — if geometry drifts with N — that is a finding large
enough to reorder everything, and it costs nothing to learn early. If it passes, the scaling
foundation is sound and the coupled work has somewhere solid to land.

⚠️ Phase B's `S_T = N` change is the one item that reaches outside this workstream: the
induction calibration is tuned against the current default. Coordinate before landing it, or
land it behind a flag with the old default preserved.

### Phase A — Instrumentation, before touching physics

Nothing here depends on the gather fix, and without it no refactor can be evaluated.

- **A1. KE/IE monitor + assertion.** Measured from the particle velocity field, not estimated.
  This is both the validity check and the ceiling on N.
- **A2. Energy conservation audit.** `Σ χ·σ:ε̇ᵖ·V` in versus `Σ ρCp·ΔT·V` + boundary flux out.
  This is the test class that catches units bugs — precisely what produced the 1000× error and
  its 16 passing mirror tests.
- **A3. Temperature-history telemetry.** Per-hit mean/min/max/std, written to the run summary,
  so thermal behaviour is observable rather than inferred from a crash code.
- **A5. ✅ DONE 2026-08-13** (§4.5): σ = 0.00024, range 0.00060, n = 5 in separate processes.
  Criterion 7 raised 0.0002 → 0.0007. **Remaining:** replicate at a late hit, where
  non-determinism may scale. Original scope follows — n ≥ 5 replicates of one config; report σ. Also
  determine whether two identical runs are **bit-identical** — if they are not, the source is
  almost certainly non-deterministic GPU atomics, and that sets an irreducible per-run floor
  every other comparison must clear. **The current 0.0002 rests on n = 2** (§4.5) and is quoted
  throughout this document.
- **A6. Build the coupled path, then smoke-test it.** 🚨 Per §2.4 there is currently **no
  non-interactive way to run a forging press with evolving temperature** — the adapter never
  enables it and the only setter lives in the teleop socket. This is a prerequisite for *all*
  coupled work, it is **ours**, and it is independent of the gather fix.
  1. Determine whether temperature evolution and induction heating are separable
     (`thermal_enabled` vs the solver's `_induction_active`), since forging wants the former
     without the latter.
  2. Plumb an explicit enable through the adapter / batch driver, defaulting **off** so nothing
     existing changes behaviour.
  3. Smoke-test: hot billet, thermal live, confirm temperature actually moves and that die chill
     appears at the contact face.
  4. Add it to the runner's read-back verification, so a run that *requested* coupling and
     silently got the frozen path **aborts** rather than reporting a result. This is the same
     failure class as the four-identical-arms incident (§5).
- **A4. ✅ Partly done (`8112a17b`).** `material_arms.py` is committed, with its FLIP read-back
  tolerance corrected — it was `1e-9` against a float32 field, so `--flip-frac 0.97` could never
  verify and the headline arms silently ran on the default (§4.1.1). `detonation_probe.py`,
  `verify_billet_temp.py` and `dump_frames.py` came with it. **Remaining:** add `--N` and `--S_T`
  for phase B and D1, and structured output. **Keep the abort-on-mismatch property** — it has now
  caught three would-be invalid experiments.

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

Validate force and cooling rate against the 06-15 mcap, which is **already local** (§3.2).
Nothing blocks this phase.

~~First target is the ~960 °C blow-#1 figure.~~ ✅ **Done 2026-08-13** (§3.3) — it reproduces,
so §4.7's "the card is in domain" argument and the commit resting on it stand.

**Next targets, in order.** (1) Per-blow temperature for all 47 blows — the within-bout signal
that separates temperature from die width, and the input a coupled cooling model would be
validated against. (2) Peak force per blow, remembering the press is force-limited at 110.2 kN
and that force is nearly blind to material. The 110.2 kN limit is itself still `12b6fa7e`'s
number and has not been re-derived.

---

## 9. Validation and metrics plan

### 9.1 Self-consistency tests — no experimental data required

These are the strongest tools for a scaling refactor, because they test implementation
correctness directly rather than agreement with a noisy measurement.

**D1 — N-invariance. The headline test.**
If S_T = N and `rate = ε̇_sim/N` are correct, **the physical answer must not depend on N.**
Geometry and temperature history must collapse onto one curve. Any systematic drift means the
scaling is wrong.

- **D1a — mechanical only (thermal off): runnable immediately.** Validates the mechanical time
  scaling independently of the gather. If geometry drifts with N, every force and material
  result to date is contaminated.
- **D1b — fully coupled:** after Phase C.

**Sweep design.** KE/IE scales as **N²**, so the sweep deliberately brackets the point where
quasi-static validity is expected to break. Runtime per run ≈ fixed scene build (~180 s, ~93%
of a short run) + stepping ∝ 1/N.

| N | vs default | predicted KE/IE | criterion 4 | est. stepping | est. total | role |
|---|---|---|---|---|---|---|
| 400 | 0.23× | ~0.2% | pass | ~215 s | ~6.5 min | most physical; reference |
| 900 | 0.51× | ~1.0% | pass | ~95 s | ~4.6 min | interior point |
| **1,773** | **1.0×** | **~3.8%** | pass (marginal) | ~48 s | ~3.8 min | **the shipped default** |
| 2,500 | 1.41× | ~7.6% | **fail** | ~34 s | ~3.5 min | just past the limit |
| 3,500 | 1.97× | ~14.8% | **fail clearly** | ~24 s | ~3.4 min | bracket / expected failure |

**≈22 min of GPU for the whole sweep.** The two high-N points are *expected to fail* criterion 4
— that is the point of including them. A sweep that only samples the valid range cannot tell you
where the valid range ends.

🎯 **The sharpest question this sweep answers is not "is the scaling right" but "is our stated
criterion right."** Compare where geometry *starts drifting* against where KE/IE *crosses 5%*:

- drift begins at the same N → the 5% criterion is validated, use it
- drift begins **earlier** → 5% is too loose; the real ceiling on N is lower than we think, and
  the shipped default at 3.8% may already be outside it
- drift begins **later** → 5% is conservative; N could be raised, buying runtime

⚠️ Also record **wall-clock and step count per run**. If low-N runs prove affordable, the
cheapest fix for the whole inertial problem is simply to stop pushing N so hard.

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

🎯 **Score every case at hits 1 / 5 / 10 (and 17 where the run survives), not at hit 1 alone.**
Per §4.10 this costs nothing — the clouds already exist — and it buys ~20× the discriminating
power, because the ceiling that saturates hit 1 does not bind at hit 10. **A single-hit score is
no longer an acceptable result format in this workstream.**

| Test | Data | Metric | Notes |
|---|---|---|---|
| Per-hit geometry | 17-hit meshes | IoU@2.0 + dev_* at fixed resolution, **as a curve over hits** | Hit 1 is ceiling-capped ~0.80; hits 5–10 sit at 0.56–0.67 with ample headroom (§4.10) |
| **Error accumulation rate** | 17-hit meshes | slope of IoU and `dev_max` vs hit number | New in §4.10: ~0.77 → 0.57 and `dev_max` ×10 across 10 hits. A coupled model that tracks reality better should flatten this slope — arguably the single most sensitive available test |
| **Isothermal prediction** | 17-hit (no temp data) | predicted ΔT over the sequence | Coupled run *should* predict near-isothermal given reheat BCs. Deviation = missing physics or wrong BCs |
| Cooling rate | 06-15 mcap | **median −6.03 °C/s within a bout**, range −11.58…+2.71 across 12 bouts (§3.4) | ✅ Extracted. The inherited ~4.8 °C/s is bout 1 only — the slowest bout in the session |
| Billet temperature | 06-15 mcap | all 47 blows: median **823.6 °C**, range **615.2–967.1** (§3.4) | ✅ **Done.** Blow #1 re-derived (§3.3) *and* the full session extracted — which showed blow #1 is the **96th percentile**, so every prior thermal number was unrepresentative |
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

### 9.4 Acceptance criteria — what "correct" looks like

State these before running, so results are judged rather than rationalised.

| # | Criterion | Threshold | Why that number |
|---|---|---|---|
| 1 | **N-invariance** of geometry | IoU spread across the N sweep **< 0.001** | **4.2σ** at the measured σ = 0.00024 (§4.5). Threshold unchanged; the old "5× the 0.0002 floor" framing was arithmetic on a 1σ figure |
| 2 | **N-invariance** of temperature history | peak ΔT spread **< 5%** | below the ±50 K emissivity uncertainty on the reference measurement |
| 3 | **Energy closure** | `abs(in − out) / in` **< 1%** per hit | tight enough to catch a units error, loose enough for accumulation noise |
| 4 | **KE/IE** | **< 5%**, reported every run | standard explicit-forming quasi-static criterion |
| 5 | **Sequence completion**, coupled, hot | **17/17** | the current best is 14 with thermal *off*; 17 coupled is the real bar |
| 6 | **Isothermal prediction** | coupled run under reheat BCs predicts ΔT within the sequence consistent with a controlled ~900–1000 °C experiment | §3.1 — this is the only thermal test the 17-hit data can support |
| 7 | **Geometry, differential** | JC vs Arrhenius separable above **0.0007 (3σ)**, or explicitly reported as indistinguishable | ⚠️ **raised from 0.0002**, which was 1σ and would call ~1 in 3 pure-noise comparisons significant (§4.5) |

🚩 **Criterion 1 is the one that matters most.** It is self-contained, needs no experimental
data, and a failure invalidates every force and material result produced under time scaling.

⚠️ Criteria 1–4 are *necessary, not sufficient*. A scheme can be perfectly self-consistent and
still wrong about the world — that is what §9.2 is for.

---

## 10. Open questions

1. ~~**Should the 06-15 mcap be retrieved now?**~~ / ~~**does the ~960 °C figure survive
   re-derivation?**~~ **Both resolved 2026-08-13.** The file was already local (§3.2) and the
   figure reproduces within ~4 °C from an independently written extraction (§3.3). What remains
   open is narrower and more useful: **extract per-blow temperature for all 47 blows.** That is
   the within-bout signal that separates temperature from die width, and it is now a matter of
   running the extraction rather than of obtaining anything.
2. **What boundary conditions reproduce the reheat?** The 17-hit bar was reheated to
   ~900–1000 °C between hits. Modelling that explicitly (vs. simply re-initialising temperature
   per hit) determines whether the isothermal-prediction test is meaningful.
3. **Does the induction calibration survive S_T = N?** Phase B forces this question.
4. **What replaces FLIP** — FMPM(k), XPIC, or an APIC-style transfer for temperature? Needs a
   survey against our constraints (GPU kernel, Quadrants DSL, existing APIC momentum path).
5. **Is the force-vs-press-speed non-convergence just the inertial budget?** §6.5 predicts it.
   D1a tests it.
6. **How much do the two missing coupling terms matter?** (§2.5) Thermal expansion and
   temperature-dependent elastic moduli are both absent. Both are harmless isothermally and
   both become error sources under dynamic coupling. Quantify before trusting coupled results —
   a cheap first estimate is an analytic bound (CTE × ΔT × billet dimension) against the
   geometry metric's own noise floor.
7. 🚨 **Are temperature evolution and induction heating separable?** (§2.4) They appear to share
   `StrikeController.thermal_enabled`, whose setter is documented as toggling induction heating.
   Forging validation wants evolution — die chill, radiation, plastic heating — **without** an
   induction source in the coil. The solver has its own `_induction_active` gate, so they may be
   separable in practice. **This gates all of A6 and therefore all coupled work.** Establish it
   first.
8. **What is the right FLIP/PIC treatment for a field that is BOTH advected and diffused?**
   Particle temperature is advected with the material *and* diffuses. FMPM(k) addresses the
   diffusion side; the advection side is why FLIP was chosen in the first place. The
   replacement must serve both.

### 10.1 Risks — what would invalidate this plan

Stated in advance so a bad result is recognised as a result rather than absorbed as noise.

| # | Risk | Consequence if it happens | Fallback |
|---|---|---|---|
| 1 | **D1a fails — geometry drifts with N** | Time scaling as implemented is invalid. Every force and material result produced under it is contaminated, including the ones in §4.9. | Lower N until invariant and pay the runtime; the sweep already measures that cost. Check whether drift tracks KE/IE — if not, the cause is not inertial and needs separate diagnosis. |
| 2 | **FMPM(k) doesn't fit the GPU kernel / Quadrants DSL** | k≥2 needs repeated grid↔particle mappings per step — k extra passes in a hot kernel. | APIC-style affine transfer for temperature (consistent with the momentum path already there), or damped FLIP with explicit smoothing. Both are weaker; document which was chosen and why. |
| 3 | **`S_T = N` is rejected** because the induction recalibration is too costly | The thermal clock stays 134× off and no coupled result can be physical. | Make the S_T mode **per-scenario** rather than global — forging uses `N`, induction keeps its calibrated value — and reconcile when induction is re-fitted. |
| 4 | **The stock-geometry fix never lands** (A-7) | Absolute geometry stays capped near 0.80; absolute agreement cannot improve. | Rely on differentials, which §4.6 already prescribes. Does not block this workstream. |
| 5 | ~~**The 06-15 mcap can't be retrieved**~~ / ~~**decoding does not reproduce ~960 °C**~~ — **both resolved** (§3.2, §3.3). Replaced by: **38% of blows sit below Song's fit floor** (§3.4) | The Arrhenius lower clamp is active on those blows, under-predicting flow stress where the metal is coldest and strongest. | Quantify the error over 615–800 °C before using 06-15 force as a material target. ⚠️ The ±50 K emissivity systematic dominates the count: at −50 K it is **30 of 47** below the floor, at +50 K only **6 of 47**. The *existence* of a sub-floor population is robust; its size is not. |
| 6 | **Coupled runs remain unstable after the gather fix** | Entirely possible: §4.1 establishes that the gather *controls* survival, **not the mechanism** (§11). | Back to diagnosis. The temperature sweep and the runtime-field poke technique are both cheap and reusable. |
| 7 | **The missing coupling terms dominate** (§2.5) | CTE and E(T) effects (~1.3% strain over 700 K) could exceed the effects being chased, biasing every coupled geometry result. | Bound them analytically first — CTE × ΔT × dimension against the metric's noise floor. Cheap, and it decides whether they must be implemented before Phase D. |

🚩 **Risk 1 and Risk 6 are the two that would genuinely reset the plan.** Both are cheap to test
early, which is the argument for the ordering in §8.0.

---

## 11. Provenance and confidence — trust at your peril

Statements here carry different weights. These specifically went in **without independent
verification**:

- ~~**The ~960 °C billet measurement** is inherited and not re-derived.~~ ✅ **Re-derived
  2026-08-13** — reproduces within ~4 °C, cooling rate 4.73 vs 4.8 °C/s, and the press-not-coil
  camera view is confirmed by rendered frames (§3.3). The isolation-method caveat is bounded at
  3.8–16.2 °C, inside the emissivity band. **What remains** is emissivity itself: ±50 K
  systematic, a property of the camera's configuration that no re-extraction can remove, and the
  blow-#1 timing, **re-derived and corrected** — contact is at 302.7 s, not ~299 s, which is
  the command timestamp; the bar reads 942.6 °C there, not 960.9 (§3.3).
- **KE/IE ≈ 3.8%** is a back-of-envelope from die speed, not a measurement of the velocity
  field. Phase A1 exists to replace it.
- **The FLIP mechanism is now half-pinned, not pinned.** ✅ The *failure* is measured: one-sided
  undershoot to negative temperature in the freshly gathered buffer, 40.6% of particles, zero
  above 4000 K (§4.1.1). ⚠️ The *route* — why FLIP grows the excursion without bound on this
  field — is still a model. Contact θ (0.036), `k_conv` (0.0001), `k_rad` (0.0005) and `k_diff`
  (0.458) all sit below their limits, so no CFL argument explains it. Do not upgrade "consistent
  with Nairn's null-space account" into "demonstrated to be Nairn's null-space account."
- 🚩 **Two diagnostics were lying, and both are worth remembering.** The runner's flip read-back
  was satisfiable only for float32-exact values, so the headline arms were unverified while
  looking verified; and the forensic trace inspects frame 0 while the trigger tests all buffers,
  so it reported "Thermal Detonation" with no particle attached. Both failed in the safe
  direction — one aborted, the other printed nothing — but both hid the answer for weeks.
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

### Diagnostic scripts

✅ **Committed to the repo root 2026-08-13** (`8112a17b`) — the four the current findings depend on:

| Script | Purpose |
|---|---|
| `material_arms.py` | The verified arm runner. Reads the built scene back and **aborts on mismatch.** ⚠️ Its FLIP read-back tolerance is `1e-6`, not `1e-9`, **on purpose** — the field is float32 and a tighter bound is satisfiable only for float32-exact values (§4.1.1). Do not re-tighten it. |
| `detonation_probe.py` | Reads `solver.particles.temp` across **every** frame buffer at a stability failure. Exists because the built-in forensic trace reads only buffer 0 (§4.1.1). |
| `verify_billet_temp.py` | Structure + per-frame temperature + isolation-method sensitivity for the 06-15 mcap (§3.3) |
| `dump_frames.py` | Frame renders and the per-frame cross-method spread (§3.3) |

Still **uncommitted, in the WSL home directory** — useful but not load-bearing for anything above:

| Script | Purpose |
|---|---|
| `~/null_floor.py` | `dev_mean` discretisation floor via perfect-lattice fill |
| `~/null_iou.py` | IoU floor, same method |
| `~/real_stock.py` | Real bar volume by two independent methods |
| `~/cfl_audit.py` | All three thermal CFL numbers + press/approach speeds |
| `~/verify_volume.py` | Watertightness, voxel-fill convergence, hull bound |
| `~/cross_section.py` | Cross-section shape (refuted the octagon hypothesis) |
| `~/viz2.py` | Extent-based comparison plots |
| `~/patch_flip.py`, `~/patch_speed.py`, `~/patch_billet_k.py` | The patchers that added `material_arms.py`'s flags — now redundant, since the runner carries them |

### Sibling sessions — shared worktree

🚨 This worktree (`aims-genesis/thermal-st-invariance`) is **shared with session B-3**
(`5dbbd727`), which owns the thermal solver. Both sessions independently proposed fixing the
same instability on 2026-08-11. **Check what the sibling is doing before editing
`base_mpm_solver.py` or `legacy_coupler.py`.** Diagnostics that need no source change —
runtime-field pokes, read-only analysis — are always the safer path.

---

## 13. Maintaining this document

This is a **living document**, and its value depends entirely on staying honest rather than
staying tidy.

**When a finding changes:**
- **Never silently delete a refuted claim.** Move it to §5 with what killed it. The refuted list
  is the most re-read part of this document precisely because it stops work being redone.
- Findings in §4 must carry the measurement that established them. A finding without a
  reproduction path belongs in §11, not §4.
- When something moves from "believed" to "measured", say which, and by what.

**When adding a number:** mark whether it was measured here, inherited, or estimated. §11 exists
because a confident number with no provenance is how this project has repeatedly gone wrong —
including a pushed commit resting on a figure nobody re-derived.

**When a defect is fixed:** strike it in §7 with the commit that fixed it, rather than removing
the row. The history of what was wrong is load-bearing context for the next person.

**Ownership:** §1 lists what belongs to other workstreams. If a change to this document implies
a change to `real_scale.py`, `forge_common`, or the induction path, that is a coordination
event, not an edit.

### Changelog

Hashes via `git log --oneline -- docs/THERMOMECHANICAL_COUPLING_AND_SCALING.md`.

| Date | Rev | Change |
|---|---|---|
| 2026-08-13 | 15 | 🚩 **Re-pointed ~10 drifted source citations** invalidated by B-3's `d2d7c701` (+12 lines in `options.py`) and `bc850e82` (+10 in `strike_controller.py`) — including `strike_controller.py:661 → :671`, the line §3.5.1's force-limit finding cites. The code was unchanged; only the pointers were wrong. Header now warns that line numbers drift in this shared worktree. 🎯 **§4.7.3 corrects §4.7.2's framing, my error:** "the two sim numbers cannot both be right" was too strong. The stop reason changes *per hit within one run* (`~/profile_g0_17.log`, complete 17 hits: **9 Max Force, 8 Target Strain**), so both workstreams can be right — and hit 1, which every §4.7.2 number comes from, is one of the hits that is *not* force-limited: first-event bias running the other way. 🎯 **The mechanism is contact width** — every hit at `W_contact` ≤ 55.2 mm saturates and the shortfall grows as contact narrows (0.078 → 0.092 → 0.136), decoupled from commanded reduction (hit 13 asks the most at 60.8 mm and lands within 0.007; hit 14 asks less at 49.6 mm and misses by 0.136). The stop preferentially truncates the narrow-contact passes. Sim saturates 53% of hits vs 19% of real blows, unmatched denominators. Also records that `max_force` is now environment-overridable in `nsf-demo` and no longer one constant across worktrees. |
| 2026-08-13 | 14 (`aad39ba2`) | ✅ **A5 done: noise floor at n = 5** — σ = 0.00024, range 0.00060, five replicates in separate processes (§4.5). The inherited 0.0002 was right as ~1σ but had been used as a *detection threshold*; acceptance criterion 7 raised 0.0002 → **0.0007 (3σ)**. No prior verdict changes: JC vs clamped-Arrhenius is 0.8σ (confirmed indistinguishable), `m3` 15.9σ, thermal-off 91.7σ. `dev_p95`/`dev_max` were identical across all five — more reproducible than IoU. |
| 2026-08-13 | 13 (`6f3cfa4c`) | 🚨 **§4.7.2: the press already terminates on `Max Force`, not on target strain** — `HOLDING (Max Force, strain=0.2130, steps=49)`, identical in three runs. The adapter intends position control against `hit.rho`; the force stop was meant as a backstop. ⇒ in this configuration hit-1 closure is set by where a numerical artifact crosses 200 kN. **Must be reconciled with workstream A**, who measure the sim landing on its command to within 0.2 mm — both cannot hold for one run. |
| 2026-08-13 | 12 (`694b97b5`) | 🎯 **§4.11: emissivity is not the lever on cooling rate and cannot be made into one.** The sim's own parameters give **1.05 °C/s** against the measured **6.03**; the emissivity needed with no conduction is **3.01** (max 1.0). Contact conduction and a surface-vs-bulk confound are the remaining candidates. Adds `cooling_budget.py`. |
| 2026-08-13 | 11 (`b5b88037`) | 🚨 **§4.7.1: "peak force" here is a numerical artifact.** Deterministic profile (two runs, 0.06%): approach impact to 89 kN → decay → **plateau ~57 kN, dies to 0.3%** → terminal spike to **192 kN with 5× die asymmetry**. Real blow #1 is monotonic to 66.5 kN. The plateau is **0.86×** real; the over-prediction is the transient. ⇒ every force calibration in this project (2.56×, 1.8×, retracted 1.34×) was peak-based and blow-1-based. |
| 2026-08-13 | 10 (`7114a482`) | **Corrected rev 9, same day, both errors mine.** The sim is *not* missing a force limit — `strike_controller.py:671` implements it, threshold 200 kN. And its force cannot calibrate that threshold (1.34–2.9× spread, 5× die disagreement), so `max_force = 110200` must not be set. The material→geometry route was **oversold**: it scales with the same 1.1%-per-10% coupling. |
| 2026-08-13 | 9 (`deb4337d`) | **§3.5: the die-closure shortfall is a force limit, not a closure cap.** 38 of 47 blows land within 0.13 mm of command; the 9 that miss are exactly the 9 saturating 110.2 kN, clean separation (105.9 vs 110.1 kN). A fixed 7.94 mm cap would under-close 33 of the 38 the real press completed. |
| 2026-08-13 | 8 | 🎯 **§4.1.1: measured what "Thermal Detonation" actually is — one-sided undershoot to negative temperature, not runaway.** 1,284 of 3,160 particles below 0 K in the freshly gathered buffer, **zero** above 4000 K, no NaN, frame 0 pristine. Closes §4.2's loop: undershoot needs a gradient, and a 293 K billet has none, which is why it was the only surviving temperature. §11 downgraded from "mechanism not pinned" to **half-pinned** — the failure is measured, the route is still a model. Also found **two lying diagnostics**: the runner's flip read-back used a `1e-9` tolerance on a float32 field, so 0.97 could never verify and the headline arms silently ran unverified on the default; and the forensic trace inspects frame 0 while the trigger tests all buffers, so it reported the failure with no particle attached. Reconciled the repo's two S_T figures — the in-source 171,653 / ~97× predates `cfl_use_pwave`, and 171,653 × 1.3808 = 237,015. |
| 2026-08-13 | 7 (`fadfd527`) | ✅ **§3.3: re-derived the ~960 °C blow-#1 billet temperature independently** — the oldest outstanding item in the document, requested by two prior handoffs and never done. Reproduces within ~4 °C, cooling rate 4.73 vs 4.8 °C/s, and the press-not-coil camera view is confirmed by rendered frames. **New:** isolation-method sensitivity bounded at 3.8–16.2 °C, i.e. inside the ±50 K emissivity band, so the original's own risk assessment was right. §11 no longer carries it as unverified; risk 5 and open question 1 both resolve; phase E re-ordered onto per-blow extraction. |
| 2026-08-13 | 6 (`e94070ba`) | 🚨 **§3.2: corrected the 06-15 mcap location — it was local the whole time.** The file had been recorded as pCloud-only and its retrieval listed as a blocker in five places, attributed to the user. It has been staged on Windows since 2026-07-23, SHA-1 verified 2026-08-06, and documented in `WHERE-IS-THE-DATA.md` since. Nothing was blocked; phase E was runnable throughout. The failure is kept alongside the fix: the check that produced it searched only the WSL filesystem and reported a confident negative from the wrong vantage. |
| 2026-08-12 | 1 (`bbc1c1ca`) | Created. Findings, refutations, scaling theory, plan, provenance. |
| 2026-08-12 | 4 | 🎯 **§4.10: scored the existing multi-hit clouds for the first time.** Every geometry score in this project had been taken at hit 1; six arms already carried up to 14 hits on disk. Agreement degrades ~0.77 → 0.67 → 0.57 across hits 1/5/10 with `dev_max` growing ~10×, and **discriminating power grows ~20×** (arm spread 0.0004 → 0.0087). Variables invisible at hit 1 — approach speed, billet temperature — separate clearly by hit 5. **This partially corrects §4.6:** the saturation ceiling is a hit-1 phenomenon; at hit 10 there is large headroom. Multi-hit scoring is now required in §9.2. 🚨 **Also recorded workstream A's measured refutation of §4.6's proposed fix:** shrinking `REAL_STOCK_RADIUS_MM` moves IoU the *wrong way* (0.7672 → 0.7077), because IoU@2.0 rewards over-filling; and the bar arrives with a localised 5.6 mm flat, so no cylinder is a valid IC. My radius recommendation is withdrawn, and my "smooth oval" cross-section is reconciled as a mid-length sample rather than the struck region. |
| 2026-08-12 | 3 | 🚨 **§2.4: established that there is currently NO code path for coupled forging** — `thermal_enabled` defaults False, its only setter is in the teleop socket, and the adapter documents "cold (no-heating) runs for now". Every result in this document was produced with temperature frozen. Added A6 (build the coupled path) as a prerequisite ours, not B-3's. Added §8.0 dependency structure; D1 sweep design with predicted KE/IE and cost, framed as validating the *criterion* not just the scaling; §10.1 risk register; the n=2 caveat on the 0.0002 noise floor plus A5 to establish it properly; 4 more refuted claims (grid-only contact, the 97× clock, Arrhenius-specific failure, "they complete 17 hits"). |
| 2026-08-12 | 2 | Added notation table; §2.4 the two thermal switches; §2.5 coupling-term inventory (thermal expansion and temperature-dependent moduli both absent); §4.7 force blindness; §4.8 card in-domain at 960 °C; §4.9 primary data tables; §9.4 acceptance criteria; open questions 6–8; §13 maintenance; §14 sources. Corrected: a paraphrase had been presented as a direct quotation; θ agreement with B-3 was overstated ("matching" → same order); ceiling percentage given false precision. |

---

## 14. Sources

Marked by how they were used. **Anything below that a decision rests on should be read at
source before relying on it** — §11 exists because that has not always happened here.

### Numerical method

| Source | Used for | Confidence |
|---|---|---|
| Nairn, *Coupling Transport Equations to Mechanics in the Material Point Method* | The FLIP-for-transport failure mode and the FMPM(k) fix (§4.1) | **Read in summary form only.** Load-bearing for Phase C — read in full before implementing. Full citation still to be added. |
| Jiang, Schroeder, Teran et al., *An angular momentum conserving affine-particle-in-cell method* | APIC's filtering property vs FLIP null modes; context for why momentum already uses APIC | Read in summary |
| Brackbill & Ruppel, *FLIP: a low-dissipation particle-in-cell method* (1986/1988) | Origin of the FLIP scheme in use | Reference only |
| LS-DYNA thermal-mechanical metal-forming notes (Oswald; ANSYS) | Industrial practice: **implicit thermal solver alongside explicit mechanics**, thermal subcycling ~1:10, mass scaling validated against implicit | Read in summary |
| Stampack / IPPT, explicit FE formulation for bulk metal forming | Fully-explicit coupled thermomechanical staggering (isothermal split), validated against implicit quasi-static (§6.6) | Read in summary |

### Material and process

| Source | Used for | Confidence |
|---|---|---|
| Song2020 | The Arrhenius/Zener-Hollomon 316L fit, 800–1000 °C (§4.8) | See `docs/316L_MECHANICAL_PROPERTIES.md` |
| Ryan & McQueen (1989/1990) | Activation energy `Q = 454 kJ/mol`; the (C, m) pairs | 🚩 **Q verified; (C, m) pairs UNVERIFIED with no path** — thesis is a pure scan, no OCR. Functional form does not match the published equation. |
| 06-15 T4 bulk mcap | Billet ~960 °C at blow #1; ~4.8 °C/s cooling; press force-limited 110.2 kN | ✅ Temperature and cooling rate **re-derived independently 2026-08-13** (§3.3). The 110.2 kN force limit is still `12b6fa7e`'s, **not** re-derived |
| 07-17 mcap + Colton's emails | Coil geometry, heating curve, pixel scale 3.8791 px/mm, coil 250 kHz | B-3's workstream |
| Colton (direct) | 17-hit sequence reheated to ~900–1000 °C between hits, fast hits | User-relayed, 2026-08-12 (§3.1) |

⚠️ **Camera emissivity remains unresolved** and every absolute temperature carries it. The
Optris "default for metals" preset pulls calibration from Optris servers at runtime and is not
in Colton's repo — stop looking for it there. Estimated effect ±50 K, which changes no
conclusion in this document but does bound the acceptance criterion in §9.4 row 2.
