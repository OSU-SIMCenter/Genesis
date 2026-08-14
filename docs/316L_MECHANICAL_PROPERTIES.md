# 316L mechanical properties at forging temperature

Research output backing `agforge/material_properties_mechanical.py`. Replaces the
hand-set `# Johnson-Cook Parameters (Hot Steel ~1200C)` block in
`agforge/options.py:38`.

Scope: **data**. Which constitutive model the solver should use is left to the
sim-accuracy workstream — but the evidence bearing on that choice is recorded
below, because it is unambiguous and it changes what the data is *for*.

---

## 0. UPDATE 2026-08-07 — read this first, it changes what the data is *for*

Four things landed that alter how the rest of this document should be read.

### 0.1 The clamp — a real fix, but NOT load-bearing for this forge

> **🚩 SUPERSEDED IN PART, 2026-08-11 (same day).** This section originally read
> *"The clamp, not the fit, was the biggest material error — 2.3×"*. That was
> written while the billet temperature was believed to be 1150–1260 °C. The
> billet has since been **measured at ~960 °C at blow #1** (thermal camera in the
> 06-15 mcap views the press, not the coil). At 960 °C = 1233.15 K the material
> is **inside Song's fitted 800–1000 °C window and inside the kernel clamp
> either way** — nothing clamps, nothing extrapolates.
>
> ⇒ **the 2.3× clamp error only occurs above 1000 °C, which this forge never
> reaches.** Raising `T_fit_max` remains correct in general and is kept, but it
> changes nothing at the real operating point. Any experiment run at 1200 °C to
> test it (as one was) is testing a regime the process does not use.
>
> The genuinely useful reading of the measurement is the opposite and better
> news: the card is calibrated at 1000 °C, the bar runs at ~960 °C, and
> **the card is in domain.** Flow stress 212.4 MPa at 960 °C vs 181.1 at 1000 °C.

[Song2020] is fitted over 800–1000 °C and `ArrheniusPlasticity` clamped its
ceiling at 1273.15 K. Standard references give a 316L forging window of
**1150–1260 °C**, which is what motivated this change — but see the measurement
above: this particular process runs cooler than the generic window.

| at 1200 °C, 0.41 s⁻¹, ε = 0.207 | MPa |
|---|---|
| [RyanMcQueen1990] reference bracket | **38.5 – 67.2** |
| Song **extrapolated** | **77.8** — ~16% above the bracket top |
| Song **clamped** (what the kernel used) | **181.1** |

So Song's *functional form* extrapolates correctly; only the clamp was wrong. It
shows the same modest ~5–10% stiff offset at 1000 °C where both fits apply.
`T_fit_max` is now **1473.15 K**, stopping exactly where Ryan & McQueen's measured
window stops. Activation energy corroborates across studies: R&M **454**,
DeAlmeida & Barbosa 450 ± 20, Song's own per-strain table 426–477 kJ/mol.
(Ferreira 2020 gets 347 and is the outlier — also the only δ-ferrite-free study.)

> **🚩 Corrected 2026-08-11 — these numbers moved.** This section originally read
> a bracket of **41.6 – 74.3 MPa** and an activation energy of **460 kJ/mol**,
> with Song landing a tidy 4.7% above the bracket. Going to the paper itself
> showed 460 is the **mean of a 21-study literature survey in its Table 1**, not
> Ryan & McQueen's own measurement. Their measured values are **454 kJ/mol
> (worked)** and **402 (as-cast)**; our bar is worked. The bracket drops ~9% and
> Song's margin widens from ~5% to ~16%.
>
> **The decision is unaffected** — the clamped 181.1 MPa is still **2.7×** the
> bracket top, and 2.3× Song's extrapolation, either way. But the corroboration
> is looser than this section first claimed.
>
> ⚠️ **Deeper caveat: the bracket's own constants are unverified.** The four
> (C, m) pairs in `RM_STATES` came from an LLM research pass rather than the PDF,
> their functional form does **not** match the paper's published eqn. (4), and
> the values could not be found in the paper's text. The same pass invented a
> non-existent paper and got a citation year wrong. What *is* read straight off
> the paper — and safe — is α = 1.2 × 10⁻² MPa⁻¹, n = 4.5, Q = 454/402, and that
> Q is constant across 900–1200 °C. See the provenance note in
> `material_properties_mechanical.py`.

New in `material_properties_mechanical.py`: `RM_STATES`, `rm_stress_mpa`,
`rm_bracket_mpa`. Tests in `tests/test_ryan_mcqueen.py`.

⚠️ The full Ryan & McQueen **kernel port is not done** — but the reason recorded
here was wrong, and it is worth being precise about. Of the four published stress
states only two carry a strain (ε = 0 and ε = 0.1); the other two are *limits* —
DRV saturation and DRX steady state — so placing them on the strain axis needs
the critical and peak strains.

**The paper publishes those.** It derives ε_c and ε_p and reports
`ε_c = 0.64 ε_p` for the worked condition. They were simply absent from the
secondary extraction this module was built on. So the port is blocked by **our
source, not by the literature** — obtaining the PDF would unblock a real flow
rule. Inventing the strains still would not.

### 0.2 🚨 Peak press force CANNOT validate a flow rule here — measured

| | flow stress at ε = 0.207 | measured peak force |
|---|---|---|
| Johnson-Cook | 201.4 MPa | 119.52 kN |
| Arrhenius @ 0.41 s⁻¹ | 181.2 MPa | 118.21 kN |
| ratio | **0.90** (10% apart) | **0.99** (1.1% apart) |

A 10% flow-stress change moves the force **1.1%**. By contrast, raising `nu` from
0.329 to 0.383 — a purely **elastic** change — moved it **10%**. The reading
responds an order of magnitude more strongly to elastic and contact stiffness
than to the plasticity it is supposed to measure.

**Do not tune this card against press tonnage.** Shape and volume metrics are the
ones that can discriminate a constitutive law.

### 0.3 🚨 The sim is NOT quasi-static — the 1773× time compression is not free

The press runs at 25 m/s against a real die speed of 14.1 mm/s. That is only
defensible if the answer is invariant to press speed. It is not (grid-only
contact, single press to ε = 0.187; single-press noise floor is 0.0021 kN and
0.0002 mm, so all of this is ~10⁴× signal):

| press speed | vs real | peak force | lateral spread |
|---|---|---|---|
| 50 m/s | 3546× | 200.3 kN | — |
| **25 m/s** (shipped) | 1773× | **119.5 kN** | baseline |
| 6.25 m/s | 443× | 72.4 kN | +0.088 mm |
| 3.125 m/s | 222× | 66.6 kN | +0.297 mm |
| 1.5625 m/s | 111× | 61.2 kN | +0.556 mm |

Force spans 3.3× over that range and **has not converged even at 111× real
speed**. Lateral spread grows monotonically, so the *deformation mode* changes
too — this is not a force-only artifact. This is the root reason 0.2 happens: the
response is rate- and inertia-dominated, so it cannot see the material.

⚠️ 3.125 m/s landing on 66.6 kN next to the real 66.5 kN is a **coincidence** —
the trend walks straight past it.

### 0.4 Other corrections

- **`nu` is now sourced**: 0.383, from `nu = E/(2G) − 1` fitted across [BAM2023]'s
  own E and G. Was 0.329, an interpolation invented for this repo and held down by
  stability rather than data. Adopting it required `cfl_use_pwave`, since
  `substep_dt` came from `sqrt(E/rho)`, which does not depend on `nu` at all.
- **Contact was a hybrid** — a grid pass plus a particle pass that hard-projected
  positions out of the die and applied friction a second time. The teleport was
  reporting force **~23% low**. Now switchable via
  `MPMOptions.enable_particle_contact`; grid-only is the trustworthy path.
- **The billet is frozen isothermal during a press** (`thermal_enabled = False`
  snapshots and restores temperatures around every physics step), so a press has
  **no die chill and no adiabatic heating**, and this card's temperature coupling
  is static during a blow.

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

## 7.5 AISI 4340 is promoted to a first-class alternative — decided 2026-08-14

**This reverses the drift toward treating 316L as the only material.** 4340 was never actually
purged — `material_properties.py` already carries `MATERIALS = {"316L": ..., "4340": ...}` and its
docstring says *"Both are kept."* But retention had decayed to thermal-only, and nothing exercised
the 4340 path.

### Why keep it, stated plainly

**316L and 4340 answer different questions, and we currently can only ask one of them.**

- **316L is the validation-against-reality material.** Every measurement we have — the mcap, the
  thermal camera, all 47 blows — is 316L. It is the only material that can tell us whether the sim
  matches *the machine*.
- **4340 is the validation-against-literature material.** It is the canonical benchmark for this
  model class: Johnson & Cook's original 1983 paper used it, and MTS parameters are published for
  multiple tempers (Banerjee 2005, §8.3). Its published constitutive data is far denser than 316L's.

⇒ **Today, when a 316L run disagrees with reality, we cannot separate "the solver is wrong" from
"the card is wrong."** A material whose card is independently, densely published breaks that
ambiguity. That is the entire argument, and it is a numerical-verification argument, not a
process-fidelity one. **4340 is not a substitute for 316L validation and must not be reported as
one.**

### Current state, checked at `e489a7bd` — three layers, not one

| layer | 4340 state |
|---|---|
| **Thermal properties** | ✅ **Complete.** `AISI_4340` carries k, cp, ρ_e, μ_r, Curie behaviour; `ACTIVE_MATERIAL` is a one-line switch |
| **Mechanical card** | ❌ **Overwritten in place.** The Johnson & Cook (1983) 4340 set survives only as a *comment* — `options.py:128`, A=792 MPa, B=510, n=0.26, C=0.014 — with each replaced value annotated at `:175`, `:178`, `:189`, `:204` |
| **Solver kernels** | ⚠️ **Compiled 316L literals.** `material_properties.py` warns: *"the GPU kernel curves in `base_mpm_solver.py` do NOT read this — changing this alone silently desynchronises them"* |

### 🚩 The trap nobody has hit yet: the test suite asserts 316L, not `ACTIVE_MATERIAL`

`tests/test_material_property_consistency.py` is written to assert **316L specifically**, and
deliberately so — `test_conductivity_rises_with_temperature` documents its own failure mode as
*"here means someone reinstated an AISI 4340-shaped curve"*, and `test_alpha_worst_is_at_the_hot_end_for_316l`
encodes the same asymmetry.

⇒ **Switching `ACTIVE_MATERIAL` to 4340 today makes the consistency suite fail — correctly, but
for the wrong reason.** Those tests must become **material-parametrised** (assert the shape that
the *active* material declares) before 4340 is runnable. This is the real work in the task and it
is not large; it is just not the one-line switch the registry makes it look like.

### Scope, in dependency order

1. **Parametrise the consistency tests** over `MATERIALS` so each asserts its own material's
   declared shape rather than 316L's. Nothing else can land first.
2. **Restore the 4340 mechanical card as selectable** — the JC constants are already in the
   comment at `options.py:128`; source them properly to Johnson & Cook (1983) rather than to a
   comment, and note that 4340's constants are **room-temperature referenced** while 316L's `A`/`B`
   are already the 1000 °C values with `T_ref = 1273.15`. **The two cards use different reference
   conventions and mixing them silently double-softens or under-softens.** This is the single
   highest-risk detail in the task.
3. **Make the kernel curves read the registry**, or add a guard that refuses to run when
   `ACTIVE_MATERIAL` and the compiled literals disagree. The warning in the source is currently
   the only protection and it is a comment.
4. **Retrieve and cite 4340 reference data** — JC and MTS parameter sets with sources. Good
   delegation candidate (fetch URLs and verbatim values; do not let a delegated agent *summarise*
   parameters, per `AGENT_WORK_DISTRIBUTION.md` §2).
5. **Run one arm each way** and report both against the same geometry metric.

⚠️ **Do not let 4340 quietly become the default.** The real forge runs 316L. 4340 exists here to
test the solver, and every result produced under it must say so.

## 8. Extrapolation beyond the fit domain — what we do, and what we should

**Added 2026-08-14.** Every constitutive fit here is valid on a box in (T, ε̇, ε). The forge does
not stay in that box. This section states what happens at the walls, why the current answer is
the wrong default, and what the alternatives actually are.

### 8.1 What the code does today: it clamps, on three axes independently

`ArrheniusPlasticity._flow_stress_pa` clamps **all three inputs** before evaluating:

| axis | clamp | Song2020 fit domain | where the real process sits |
|---|---|---|---|
| temperature | `[1073.15, 1473.15] K` = 800–1200 °C | 800–**1000** °C | 615–967 °C measured (§ coupling doc 3.4) |
| **strain rate** | `[1e-4, 1e3] /s` | **2e-4 … 2e-2 /s** | **0.136–1.472 /s — 0 of 47 blows in domain** |
| plastic strain | `[0.05, 0.45]` | tabulated at 9 nodes, 0.05 spacing | exceeds 0.45 in later hits |

🚨 **The rate axis is the real extrapolation problem, and it is the one nobody has been treating
as one.** The temperature question gets the attention because the clamp is visible and was once
wrong by 2.3×. But on rate, **the entire operating regime is out of domain** — the session median
is 19× above the top of the fit — and the guard is set at `1e3`, which is **5 orders of magnitude
above the fitted maximum** and therefore never binds on anything physical. It is not a guard.

⚠️ Worse, the *simulated* rate is ~656 /s (N = 1,773 time compression), which is **3×10⁴ above
the fit** and still slips under `rate_max`. `arrhenius_process_strain_rate = 0.41` exists to
substitute the physical rate and is **opt-in**; when it is off, the law is evaluated four decades
outside its domain with no diagnostic. That is defect 3 in the coupling doc's inventory.

### 8.2 Why clamping is the wrong default

Clamping is not a neutral fallback. It makes three specific claims, all false:

1. **Zero sensitivity beyond the wall.** A clamped input means ∂σ/∂T = 0 and ∂σ/∂ε̇ = 0 outside
   the box. For a *rate- and temperature-coupled* return map that is a strong physical assertion —
   the material abruptly stops responding to the very variables the coupling exists to model.
2. **A kink at the wall.** The result is C⁰ but not C¹. In an explicit solver, a derivative
   discontinuity in the yield surface is a localisation seed: every particle sitting on the clamp
   responds identically and stops differentiating.
3. **Direction-dependent error, and one direction is dangerous.** Clamping *below* `T_fit_min`
   returns the 800 °C flow stress for a colder billet, which **understates** strength — the
   unsafe direction. Clamping *above* overstates it. This project has already paid for this once:
   clamping at 1000 °C when the billet was at forging heat gave **181.1 MPa against 77.8**, a
   **2.3× error entirely from the clamp** (see the `T_fit_max` note in `materials.py`).
4. **It is silent.** Nothing reports how much of the billet was on a clamp. A run in which 90% of
   particle-evaluations saturated the rate clamp is indistinguishable, in the output, from one
   that never touched it.

### 8.3 What the literature actually recommends

**The sinh form extrapolates better than Johnson-Cook — which is an argument for extrapolating
it, not clamping it.** A direct FEA-oriented comparison of JC against Arrhenius-type hyperbolic
sine found the A-type model more accurate in flow-stress prediction **"even outside of the fit
domain"** ([Evaluation on prediction abilities of constitutive models considering FEA
application](https://journal.hep.com.cn/jocsu/EN/10.1007/s11771-018-3822-8), 2018). This is
exactly what was done successfully for `T_fit_max`: extrapolate the *form*, then check the
extrapolated point against an independent in-domain source (Ryan & McQueen 1990, type 316 torsion
measured over 900–1200 °C). **That method — extrapolate, then corroborate out-of-band — is the
one to generalise, and it has not been applied to the rate axis at all.**

**Physics gives a validity check on the parameters themselves.** In the sinh law `Q` is an
apparent activation energy and `n` a stress exponent; the standard check is to compare them with
the values creep theory expects — lattice self-diffusion in austenite (~270–280 kJ/mol) and
n ≈ 5 for dislocation-climb creep. This is the recommended practice in the hot-working review
literature ([A review of hot deformation behavior and constitutive models to predict flow stress
of high-entropy alloys](https://consensus.app/papers/details/daad9ae3c02556478b1e853857f6d157/),
Savaedi et al. 2022). 🚩 **Our tabulated `Q` runs 426–477 kJ/mol — well above the self-diffusion
value.** That is not unusual for solute-bearing austenitic steels, but a high `Q` means very
strong temperature sensitivity, so **extrapolation error on the T axis compounds fast**. Worth an
explicit check before leaning further on the extrapolated band.

**For extrapolating strain specifically, DRX kinetics beat curve-fitting.** Incomplete flow
curves are routinely extended to higher strain using an Avrami description of dynamic
recrystallisation rather than by extending the fit ([Extrapolation of flow curves at hot working
conditions](https://consensus.app/papers/details/1b605a817f515a7ab5178e6b5902dba6/), Mirzadeh
et al. 2010). This is directly applicable to our ε > 0.45 clamp.

**On blending models — the literature is clear that naive patching is not the answer, and that
pure data-driven models must not extrapolate.** An ANN predicts interpolated *and* extrapolated
strains at R² > 0.98, but **both ANN and strain-compensated Sellars fail under new deformation
conditions**; a *hybrid* of the two generalises more widely ([Application of Constitutive Models
and Machine Learning Models to Predict the Elevated Temperature Flow Behavior of TiAl
Alloy](https://consensus.app/papers/details/4b12542bad1857fa8ceee418df1b6d9a/), Zhao et al.
2023). The working recipe is to feed the *physical* model's prediction and characteristic stress
points in as **inputs** to the network rather than replacing it: a pure network is "rather
uncertain outside the training data range", while the integrated model predicts accurately
([Development of Constitutive Models for Extrapolative Prediction of Nb–Ti Micro Alloyed
Steel](https://consensus.app/papers/details/f237239e1dfe527eb3db8ea713473839/), Wu et al. 2017).
⚠️ **A bare neural network is the worst possible choice for this project specifically** — it
fails silently and confidently outside its training box, which is the exact failure mode this
codebase keeps getting caught by.

**If the range genuinely has to widen, change the model class rather than patch the fit.** The
Mechanical Threshold Stress model is an internal-state-variable formulation built on
thermal-activation kinetics precisely to span wide temperature and rate ranges; a JC-vs-MTS
comparison under strongly varying thermal history found MTS gave more realistic plastic-strain
evolution ([Physics-based and phenomenological plasticity models for thermomechanical simulation
in laser powder bed fusion additive
manufacturing](https://consensus.app/papers/details/d106b0d8c1dd5a269268993301ee5d0c/),
Promoppatum et al. 2021), and MTS parameters have been published for AISI 4340 — the steel this
project used before 316L ([The Mechanical Threshold Stress model for various tempers of AISI 4340
steel](https://consensus.app/papers/details/fa267079fe2a5ae299b498c36fcce9a1/), Banerjee 2005).
⚠️ MTS has its own wall: it breaks where the mechanism turns over from thermal activation to
phonon drag at very high rate ([Modification of the MTS model for high strain-rate behavior of
Ti-6Al-4V](https://consensus.app/papers/details/ff83fa3c42585e9fbe0b74f3672042d2/), Allen et al.
2023). **Every model has a domain; the goal is to know where it is, not to find one without.**

### 8.3.1 🎯 MEASURED 2026-08-14 — and it re-ranks §8.4

`clamp_probe.py` (repo root) evaluates every clamp against the measured envelope. **The result
inverts the emphasis this section had when it was written.**

| axis | blows inside the **fit** | blows inside the **clamp** | does the clamp bind? |
|---|---|---|---|
| temperature | 29 of 47 | 29 of 47 | 🚨 **YES — 18 of 47 below `T_fit_min`** |
| **strain rate** | **0 of 47** | **47 of 47** | ❌ **never** |
| plastic strain (per-blow) | — | 40 of 47 | yes, 7 above the table end |

**The distinction §8.2 blurred, and it matters: *being extrapolated* and *being clamped* are
different failures.**

- **Rate is 100% extrapolated and 0% clamped.** Every blow is outside Song's fitted rate window,
  but every blow is also comfortably inside `[1e-4, 1e3]`, so the clamp never fires. Widening or
  tightening `rate_max` changes **nothing** about today's numbers. It is a guard that does not
  guard — a latent risk, not an active distortion.
- **Temperature is the axis that is actively distorting results, and via its FLOOR.** 18 of 47
  blows sit below `T_fit_min`; **zero** reach even Song's 1000 °C ceiling. And the floor errs in
  the unsafe direction: at the measured minimum (888.4 K / 615 °C), clamped gives **391.6 MPa
  against an extrapolated 673.6** — **ratio 0.58, understating strength by 42%.**

🚩 **A consequence for the 2026-08-07 `T_fit_max` change.** Raising the ceiling 1273.15 → 1473.15 K
corrected a real 2.33× error (`clamp_probe.py` reproduces it exactly: 181.1 vs 77.8 MPa). But on
**this** 47-blow envelope **no blow reaches even the old ceiling**, so that fix moves none of these
evaluations. It was motivated by a billet "at real forging heat (1150–1260 °C)" while the camera
reads 615–967 °C. ⚠️ **That is a ~200–300 °C disagreement the ±50 K emissivity band cannot
close**, and it is the same surface-versus-bulk question §4.11 of the coupling doc could not
separate. **Open — do not resolve it by picking whichever number suits the argument.**

⇒ **Revised priority: the temperature floor is the clamp costing us accuracy today.** The rate
work is still worth doing, but for a different reason than stated below — not because the clamp
binds, but because the *fit* does not cover the regime at all.

### 8.4 What to do here, in order

1. **Instrument the clamps** (~20 lines, no physics). Report per hit the fraction of
   particle-evaluations that saturated each of the three clamps, and the min/max of each input
   seen. **We are currently blind to our own extrapolation.** Everything below is guesswork until
   this exists, and it is the cheapest item in this document.
2. 🚨 **Deal with the temperature FLOOR — re-ranked to second by §8.3.1's measurement.** It
   binds on 18 of 47 blows and understates strength by up to 42%, which is the unsafe direction.
   The `T_fit_max` precedent applies directly: extrapolate the sinh form downward and corroborate
   the extrapolated point against an independent source that measured *below* 800 °C. Until then,
   every result touching those 18 blows carries a soft-material bias.
3. **Make the physical rate the default for Arrhenius arms**, not opt-in — the fictional 656 /s
   is 3×10⁴ outside the fit. ⚠️ Note this is **not** about the clamp, which never fires
   (§8.3.1); it is about what the law is *evaluated at*. Use the measured median **0.373 /s**
   (§7.1), or 0.41 which is within 9%.
4. **Make the rate guard actually guard.** `rate_max = 1e3` sits 5 orders above the fit and
   catches nothing — the sim's 656 /s slips under it. Set it near the physical envelope
   (measured max 1.47 /s) and make exceeding it **abort or warn**, matching the arm runner's
   existing contract. Low urgency for today's numbers, high value as a tripwire.
4. **Corroborate the rate extrapolation out-of-band**, the same way `T_fit_max` was corroborated.
   Ryan & McQueen 1990 covers 900–1200 °C in torsion and reaches higher rates than Song; it is
   already cited here and may bracket the 0.1–1.5 /s band directly.
5. **Check `Q` and `n` against creep theory** before trusting the extrapolated temperature band.
6. **Only then** consider MTS or a physical+data hybrid. Both are real work and neither is
   justified while the clamps are uninstrumented and the rate default is wrong.

⚠️ **What this section does not claim.** No extrapolation method here has been tested in this
codebase. The ranking is from the literature and from the one in-project precedent that worked
(`T_fit_max`). Item 1 exists because the honest answer to "how much does clamping cost us today"
is **we do not know, and we cannot know until we measure it.**

## 7. Open items

1. ✅ **RESOLVED 2026-08-14 — measured, all 47 blows. The recommendation below (1 s⁻¹) is
   superseded and was 2.7× too high.** The mcap this item says is missing has been local since
   07-23 and extracted since 08-13 (`outputs/t4_press_blows.npz`); `blow_rates.py` derives the
   rate from the die-gap trace for every blow. True strain rate `ln(h0/h1)/dt`:

   | | min | p25 | **median** | p75 | max |
   |---|---|---|---|---|---|
   | rate [1/s] | 0.136 | 0.268 | **0.373** | 0.554 | 1.472 |
   | ram [mm/s] | 4.42 | 7.19 | 11.70 | 15.68 | 22.65 |

   🎯 **The hardcoded `arrhenius_process_strain_rate = 0.41` is right** — within **9%** of the
   session median. It was derived from blow 1 alone, which was luck rather than method, but the
   number stands. The 1 s⁻¹ recommended here is **2.7× the median** and should not be adopted.

   🎯 **And blow #1 is the 57th percentile on rate** — essentially median — while it is the
   **96th percentile on temperature**. ⇒ **the first-event bias is property-specific, not a
   blanket property of blow 1.** Check it per quantity; do not assume it transfers.

   🚨 **0 of 47 blows fall inside the Song2020 fit domain** (2e-4 … 2e-2 /s). The median sits
   **19× above its top**. Rate extrapolation is not an edge case here — it is the entire
   operating regime. See §8.

   ~~Original item follows.~~ **Pick the nominal strain rate** — recommended default **1 s⁻¹**,
   sensitivity bound 10 s⁻¹.

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
   | `jc_m` | 1.03 | **1.0** | still unsourced; see §0 — inert below `jc_T_ref`, and now REACHABLE since the ceiling moved to 1200 °C |
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
