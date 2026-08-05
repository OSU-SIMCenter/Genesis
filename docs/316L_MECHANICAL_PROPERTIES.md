# 316L mechanical properties at forging temperature

Research output backing `agforge/material_properties_mechanical.py`. Replaces the
hand-set `# Johnson-Cook Parameters (Hot Steel ~1200C)` block in
`agforge/options.py:38`.

Scope: **data**. Which constitutive model the solver should use is left to the
sim-accuracy workstream — but the evidence bearing on that choice is recorded
below, because it is unambiguous and it changes what the data is *for*.

---

## 1. The headline: "properties at 1000 °C" is underdetermined

1000 °C is in the **dynamic recrystallization (DRX)** regime for 316L. Flow stress
is strongly rate-dependent there, so a single number is not a well-posed answer.

| strain rate | peak flow stress @ 1000 °C |
|---|---|
| 0.1 s⁻¹ | 157 MPa |
| **1 s⁻¹** | **213 MPa** |
| **10 s⁻¹** | **275 MPa** |
| 100 s⁻¹ | 339 MPa |

A 100× change in rate moves flow stress **1.75×**; 1000× moves it **2.15×**.
Forging rates (~1–100 s⁻¹) are far from quasi-static, so quoting a
handbook/quasi-static number here would be wrong by roughly a factor of two.

The curves also **peak near strain 0.30 and then soften** — the DRX signature.

## 2. What this says about the current simulation

| quantity | current | this work @ 1000 °C | error |
|---|---|---|---|
| flow stress @ 1 s⁻¹ | 86–121 MPa | 154–213 MPa | **1.8–1.9× too soft** |
| flow stress @ 10 s⁻¹ | 86–121 MPa | 192–275 MPa | **2.2–2.4× too soft** |
| Young's modulus `E` | 50 GPa | ~120 GPa | **2.4× too low** |
| density `rho` | 8000 kg/m³ | 7334 kg/m³ *(band 7330–7570)* | **+6–9% too high** |
| `t_melt` (in `environment.py:290`) | 1793 K | 1675 K | **+7.0%, and it is 4340's value** |

The "too soft" ratio is remarkably flat across strain (1.79–1.90 at 1 s⁻¹), so it
behaves like a clean scale factor, not a shape error.

**Testable prediction for the sim-accuracy workstream:** if the material is ~2×
too soft, the simulated forging force on the real 17-hit sequence should be
correspondingly *low*, not high. That is a discriminating check against measured
per-hit force.

### The stability coupling — measured, and less alarming than expected

`E` and the timestep are linked through CFL: `dt ∝ dx/√(E/ρ)`. Correcting E moves
the wave speed **2500 → 4070 m/s**, so the timestep shrinks **1.63×**.

**This is not a stability hazard.** `substep_dt` is *derived* in
`MaterialOptions.model_post_init` as `0.90 × dx/c`, so it self-adjusts to any
`E`/`rho` and the CFL assertions still pass — verified: ratio 0.9000,
`substep_dt = 1.208e-6 s` against `dt_cfl = 1.342e-6 s`. Nothing to violate.

What it actually costs is **wall-clock: 1.63× more substeps** for the same
physical time.

⚠️ Two corrections to earlier statements in this document's own history: the cost
is 1.63×, not the 1.56× first quoted — that figure used √(E_new/E_old) and ignored
that `rho` fell as well; the correct ratio is √((E/ρ)_new /(E/ρ)_old). And the
inherited "~1.6×" estimate it appeared to confirm was right for the wrong reason.

Separately, the material is now ~2× stiffer in *flow stress*, which does not enter
the CFL bound at all but does change forces and the return-mapping. Whether the
17-hit sequence still completes is an empirical question needing a run — but it is
a physics question, not a stability one.

## 3. Why Johnson-Cook is the wrong model here — quantified

Original JC is `σ_y = (A + B·εₚⁿ)(1 + C·ln ε̇*)(1 − T*ᵐ)`: three **uncoupled**
multiplicative terms, with a **monotonically increasing** hardening branch. It
therefore cannot represent a peak-then-soften DRX curve at all.

Fitting JC to our 1000 °C / 1 s⁻¹ curve:

| region | JC accuracy |
|---|---|
| strain 0.05–0.30 (pre-peak) | **excellent, 1.55% RMS** |
| strain 0.35 | +6.9% |
| strain 0.40 | +11.4% |
| strain 0.45 | **+26.0%** |

So JC is usable *if and only if* plastic strain stays below ~0.30. Past the DRX
peak it diverges fast, in the direction of a spuriously **stiffening** billet.

This matches the literature consensus. For 316-family austenitics in the
hot-working domain, strain-compensated Arrhenius (hyperbolic-sine /
Zener-Hollomon) models track the data well and original JC does not — stated
explicitly by Mirzadeh (2015), Zhang et al. (2014), and the 316H comparison study
(2020), and found again for other alloys by Samantaray et al. (2009).

### ⚠️ Correction to an inherited claim, and its proper replacement

An earlier session carried the figure **"~36% error for JC vs ~13% for modified
Zerilli-Armstrong"** for 316L. That specific pairing could not be substantiated;
the closest-matching study is **Samantaray et al. (2009), which is modified
9Cr-1Mo steel — a ferritic/martensitic alloy, not 316L**. Treat 36/13 as **not
applicable**.

**The claim was directionally right, though, and there is a real 316L number.**
Esmaeilpour & Abedi (2022), hot compression on SLM 316L at 973–1273 K and
0.001–0.1 s⁻¹ — i.e. genuinely inside the DRX regime — report over 420 data points:

| model | R | RMSE | AARE |
|---|---|---|---|
| **original Johnson-Cook** | 0.964 | 73.2 MPa | **48.0%** |
| Arrhenius (hyperbolic-sine) | 0.960 | 27.2 MPa | **7.7%** |
| Arrhenius, excluding 973 K | 0.992 | 8.8 MPa | 3.7% |
| ANN | 0.9997 | 1.5 MPa | 2.8% |

They note JC is acceptable only near its reference condition and "cannot predict
the flow softening behavior". So in the DRX regime JC is off by roughly **6×**
more than an Arrhenius fit. Caveat: this is **additively manufactured** 316L, not
wrought.

#### The trap next to it

The nearest 316-family *comparison* study is **Gupta et al. (2012)**, reporting
correlation coefficients JC 0.9423 / modified ZA 0.9879 / modified Arrhenius
0.9852 / ANN 0.9930, with AARE 6.63% / 3.32% / 3.34%. Those numbers look like a
direct answer and are **not one**: Gupta tested at **323–623 K and 10⁻⁴–10⁻¹ s⁻¹**
— 50–350 °C, quasi-static, the *dynamic-strain-aging* regime. No DRX anywhere in
it. JC scoring 6.63% there says nothing about JC at forging temperature, where the
same model scores 48%. Do not import Gupta's parameters or its verdict.

### ⚠️ The `jc_C` term is currently dead code

`agforge/materials.py` declares `C` and `eps0`, `options.py:43-44` exposes them,
and `environment.py:289` passes them through — but the kernel body never
references them, and its signature `(J, F_tmp, U, S, V, Jp, temp)` carries no rate
information, so it *cannot*. The class docstring advertises the rate term the code
omits, and omits the thermal term the code has.

Given that rate sensitivity is worth **1.75× over two decades** here, this is the
single largest missing physics in the material model. Enabling it means changing
the dispatch signature at `genesis/engine/solvers/base_mpm_solver.py:620` — which
is under the repo-wide edit freeze, so it is a deliberate decision, not a
drive-by fix.

## 4. Sources and confidence

| quantity | source | method | confidence |
|---|---|---|---|
| flow stress, 800–1000 °C | **Song 2020**, *Materials* 13(17):3766, Table 1 | strain-compensated Arrhenius fit | HIGH |
| independent cross-check | **Zhou 2023**, *Mater. Res. Express* 10:115604 (316H) | Gleeble-3800, 900–1200 °C, 0.01–10 s⁻¹ | HIGH |
| high-rate bound | **Benč 2023**, METAL (3D-printed 316L) | measured peak stress incl. 100 s⁻¹ | MEDIUM |
| `E(T)`, `G(T)` | **BAM 2023**, Zenodo 7813836 | dynamic resonance, ASTM E1875, RT–900 °C | HIGH |
| `E` near 1000 °C | Andrews via **ISIJ Int.** 33(4):508 | ultrasonic, 118.7 GPa @ 1270 K | MEDIUM |
| `ρ(T)` | **NIST** SRM 1155a, PMC8193647 | ohmic pulse heating + DSC | HIGH |
| Poisson's ratio | INCO/BSSA tables, ISIJ | scattered, non-monotonic | **LOW** |

### The cross-validation is the strongest evidence here

Two **independent** studies — different alloys (316L vs 316H), different labs,
different fitting procedures, different test rigs — predict the same 1000 °C flow
stress to within **0.9–2.4%** across four decades of strain rate:

| rate | 316L (Song) | 316H (Zhou) | difference |
|---|---|---|---|
| 0.1 s⁻¹ | 157.2 MPa | 161.0 MPa | 2.4% |
| 1 s⁻¹ | 213.4 MPa | 216.7 MPa | 1.5% |
| 10 s⁻¹ | 274.7 MPa | 277.7 MPa | 1.1% |
| 100 s⁻¹ | 338.5 MPa | 341.4 MPa | 0.9% |

### Song's data is TENSILE; forging is compression

Song 2020 is a hot **tensile** test. 316L is known to be tension/compression
asymmetric — El-Tahawy et al. (2020) measured tensile flow stress ~40 MPa above
compressive at low strain and ~200 MPa above by 50% strain, at **room
temperature**, driven by strain-induced martensite.

That mechanism does not operate at 1000 °C: austenite is stable, there is no
martensite to form. And the cross-check above settles it empirically — Zhou 2023
is Gleeble **compression** and agrees with Song's tension-derived curve to within
**2.4%** at forging temperature. So the asymmetry is negligible here, but the
agreement is doing real work and should not be discarded.

### One ambiguity inside the primary source

Song 2020 states its strain-rate range **inconsistently**: the abstract says
2×10⁻³–2×10⁻¹ s⁻¹, the methods section says 0.0002 / 0.002 / 0.02 s⁻¹. This module
assumes the methods section (ceiling 2×10⁻² s⁻¹), which is the *more*
conservative reading — it makes the extrapolation to forging rates look worse, not
better. If the abstract is correct, the extrapolation is one decade less severe.

### Honest bound on the extrapolation

Song's fit domain tops out at 2×10⁻² s⁻¹, so forging rates extrapolate it by
2–4 orders of magnitude. Checked against Benč's direct measurements:

- 1173 K, 100 s⁻¹ — predicted 435.6 MPa vs **measured 381 MPa** → **+14%**
- 1523 K, 0.1 s⁻¹ — predicted 50.0 MPa vs **measured 65 MPa** → **−23%**

Both checks sit at the far corners of the domain. A third check lands *inside* it:
an independently reported measurement of **111 MPa at 1000 °C / 0.01 s⁻¹** against
this module's **109.5 MPa** — **−1.3%**, which confirms the implementation itself
is faithful.

So the error budget is:

| condition | error |
|---|---|
| in-domain (1000 °C, 0.01 s⁻¹) | **−1.3%** |
| cross-model agreement (0.1–100 s⁻¹) | **0.9–2.4%** |
| extrapolated to 100 s⁻¹ at 900 °C | **+14%** |
| far corner (1250 °C, 0.1 s⁻¹) | **−23%** |

At 1000 °C and 1–10 s⁻¹ — inside the temperature range and much nearer the rate
range — the error should be small, but **±15% is the honest bar** for anything at
100 s⁻¹.

## 5. The delivered card

`isothermal_card(strain_rate=1.0)` at 1273.15 K:

| parameter | value |
|---|---|
| `E` | 121.5 GPa |
| `G` | 44.0 GPa |
| `nu` | 0.329 *(low confidence)* |
| `rho` | 7334 kg/m³ |
| `t_melt_k` | 1675 K |
| peak flow stress | 213.4 MPa |
| `jc_A` | 100.3 MPa |
| `jc_B` | 195.0 MPa |
| `jc_n` | 0.417 |
| valid to | strain 0.30 |

### Why `jc_A` is pinned rather than fitted

The least-squares residual is **nearly flat in A** — 0.85% RMS at A = 1 MPa vs
1.55% at A = 100 MPa — so the optimizer slides A to whatever bound it is given
and pays almost nothing.

That matters because the solver evaluates `A + B·εₚⁿ` starting from `εₚ = 0`,
which is **outside the fit domain** (the source data begins at strain 0.05). At
`εₚ = 0` the expression collapses to `A`, so **A is the initial yield stress the
simulation actually sees**. A free fit returns A ≈ 1 MPa — a perfect fit over
0.05–0.30, and a billet that yields at essentially zero load.

So A is pinned at 47% of peak stress (≈100 MPa at 1 s⁻¹), physically defensible
for hot 316L, costing ~0.7 points of RMS. `fit_johnson_cook(a_fixed_mpa=...)`
overrides it.

## 6. Independent cross-checks from a separate search stack

The numbers above were re-derived through a second, unrelated research pipeline
(three Perplexity runs: Gemini 3.1 Pro on plain search, Deep Research, and
Kimi K3). Useful outcomes:

- **The Song constant table was independently reproduced**, digit for digit, at
  strains 0.1 and 0.2 — confirming the transcription here is faithful.
- **A cross-alloy sanity check.** AISI 321 (Ti-stabilised austenitic) measured at
  1000 °C gives 99 MPa @ 0.01 s⁻¹, 139 @ 0.1, 188 @ 1 s⁻¹. This module's 316L
  values run 10–13% higher (109.5 / 157.2 / 213.4). Different alloy, so exact
  agreement is not expected; the offset direction is plausible (316L carries Mo),
  though one source asserted the opposite ordering, so do not lean on it.
- **The density spread was caught this way** — two independent runs both returned
  ~7550–7600 kg/m³ via the room-temperature-plus-CTE route, against NIST's
  directly measured 7334. That disagreement is now documented rather than hidden.
- **`E` at 1000 °C survived**: independent estimates of "115–130 GPa" bracket the
  121.5 GPa used here. One run claimed the modulus "degrades well below 100 GPa",
  which the BAM measurements (129 GPa at 900 °C) directly refute.

Worth recording: the plain-search run reproduced the Gupta figures **without**
noting that Gupta tested at 323–623 K, presenting them as an answer about the DRX
regime. The deep-research run caught it. The regime caveat is the whole ballgame
here, so a confident-looking citation is not sufficient — check the test window.

## 7. Open items

1. **Pick the nominal strain rate** — recommended default **1 s⁻¹**, sensitivity
   bound 10 s⁻¹.

   It is *derivable, not a lookup*: **ε̇ ≈ v / h**, ram velocity over billet
   height. The telemetry already exists — `hmr/press/state` in the mcap carries
   **`live_force_kn` + `live_stroke_mm`** (`live_position_mm` too) at ~634 Hz
   measured, and `ForgeMcap.state_series` already reads it. Differentiate stroke
   for `v`, use force to bracket the contact window.

   ⚠️ **But not in the 07-17 file.** Checked directly: across all 363,496 press
   samples `live_stroke_mm` and `live_position_mm` have **range 0.0000** — the ram
   is parked at 227.30 mm with a constant ~2.4 kN load-cell tare. That capture is
   pure heating, no blows. This needs one mcap recorded *during a forging run*.
   (Note: the 420 Hz figure in `mcap_thermal.py:304` does not match the ~634 Hz
   measured here.)

   Until then 1 s⁻¹ is the defensible default: across the whole plausible
   1–10 s⁻¹ band flow stress moves only **1.29×**, against the **1.9–2.4×** error
   the sim carries today — so this choice is second-order and should not block
   the correction. `isothermal_card(strain_rate=...)` regenerates everything.

   The sim has no physical rate to recover, incidentally: `options.py` sets
   `pressing_speed = 25.0 m/s` and derives `approach_speed` from
   `target_cfl_ratio * (dx / dt)` — a stability number, not a kinematic one.
   Same trap as `E = 50 GPa`.
2. **Where the card lives.** Written to `agforge/` because that is rebase-free and
   testable, but with **zero repo imports** so it can move to
   `forge_common/materials/` verbatim. That placement is an architecture call.
3. ✅ **FIXED — the 4340 constants at the JC call site.** `environment.py` used to
   hardcode `T_melt=1793.0, jc_m=1.03` (both AISI 4340) for a 316L billet, at the
   call site rather than in `MaterialOptions` — which is exactly why the 316L
   conversion missed them. They are now plumbed as `jc_T_ref` / `jc_T_melt` /
   `jc_m`, and **`jc_T_melt` reads `ACTIVE_MATERIAL.t_melt_k`**, so the melting
   point can no longer drift from the material module at all. Guarded by
   `test_no_4340_constants_hardcoded_at_the_jc_call_site`.
   **Still open: `jc_m` itself.** 1.03 is 4340's exponent and no 316L value was
   traced. It is inert under isothermal running (T* ≈ 0), so this is a landmine
   rather than a live bug — but it must be sourced before thermal softening is
   switched on.
4. ✅ **APPLIED — every parameter in `MaterialOptions` is now sourced.**

   | parameter | was | now | note |
   |---|---|---|---|
   | `E` | 50 GPa | **121.5 GPa** | 1.63× more substeps; NOT a CFL hazard, see below |
   | `rho` | 8000 | **7334** | |
   | `nu` | 0.28 | **0.329** | low confidence |
   | `jc_A` | 40 MPa | **100.3 MPa** | pinned, not fitted |
   | `jc_B` | 100 MPa | **195.0 MPa** | |
   | `jc_n` | 0.26 | **0.417** | 0.26 was 4340's |
   | `jc_C` | 0.014 | **0.120** | 0.014 was 4340's *room-temperature* value, ~9× low |
   | `jc_eps0` | 1.0 | **1.0** | now explicitly the calibration rate |
   | `jc_T_ref` | 293.15 | **1273.15** | so the card is not softened twice |
   | `jc_T_melt` | 1793 | **1675** | reads `ACTIVE_MATERIAL.t_melt_k` |
   | `jc_m` | 1.03 | **1.0** | the one unsourced value left |
   | `von_mises_yield_stress` | 19 MPa | **213.4 MPa** | was ~11× low |

   A joint 5-parameter JC fit across 800–1000 °C × 0.1–100 s⁻¹ was attempted and
   **rejected**: 24.7% worst-case error with `m` driven to its bound. That is the
   structural failure reproducing itself, not a tuning problem. Calibrating *at
   the operating point* instead makes the model exact where the sim runs.
5. **Poisson's ratio at forging temperature is genuinely unresolved.** Published
   values scatter non-monotonically. Low impact, but do not treat 0.329 as solid.
6. ✅ **RESOLVED — the billet is 316L.** Confirmed directly by the project owner.
   An earlier note claiming the forged material was *not* 316L is **wrong** and
   should not be propagated.

7. 🔴 **The real fix: replace Johnson-Cook with hyperbolic-sine Arrhenius in the
   kernel.** Everything above is JC calibrated to be exact at one operating
   point, which is the best JC can do. Three concrete failures remain, all
   structural:

   - **No strengthening on cooling.** `T*` clamps at 0 below `T_ref`, so the
     billet does not stiffen as it cools. Real 316L gains **~43% per 100 °C**
     drop. Inert while `thermal_enabled = False`; wrong the moment it is not.
   - **Rate dependence is a dead term.** `jc_C` is now correct but the kernel
     never reads it (see item 8), and even when wired, JC's `1 + C·ln(ε̇*)` only
     holds locally — the true C runs 0.155 → 0.101 across 0.1–100 s⁻¹.
   - **No DRX softening.** JC hardening is monotonic; past strain 0.30 it
     diverges to **+26%** while the real material softens.

   The Arrhenius form `σ = (1/α)·asinh[(Z/A)^(1/n)]`, `Z = ε̇·exp(Q/RT)` fixes all
   three at once — it couples temperature and rate inherently, reproduces the
   peak-then-soften shape, and is the model the literature endorses for 316L
   (**7.7% AARE vs 48%**). `material_properties_mechanical` already implements
   and validates it on CPU; porting it into the MPM kernel is the work.

8. **`jc_C` is still dead code in the kernel.** The value is now right, but
   `_update_F_S_Jp_jc` never reads `self.C` and its signature
   `(J, F_tmp, U, S, V, Jp, temp)` carries no rate information. Wiring it needs
   the velocity gradient (or `delta_gamma/dt`) plumbed through the dispatch in
   `base_mpm_solver.py`.
7. **One lead worth chasing.** A 2025 MDPI paper, "Experimental and Numerical Study
   of Behavior of Additively Manufactured 316L Steel Under Challenging Conditions",
   reports hot compression at **900 / 1000 / 1100 / 1250 °C × 0.1 / 1 / 10 / 100 s⁻¹**
   — precisely the grid this card extrapolates into. Its values appear to be in
   plots rather than tables (only the ~380 MPa maximum at 900 °C / 100 s⁻¹ is
   quoted in text), so it would need digitising, but it could convert the
   high-rate numbers here from extrapolation to measurement. It is AM material,
   like Esmaeilpour & Abedi.
