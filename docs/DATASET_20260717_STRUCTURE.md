# The 2026-07-17 induction dataset — what is actually in it

**Status 2026-08-04.** Everything here was derived from `20260717_135009.mcap` directly. It
supersedes the earlier reading of this file, which was based on sampling one thermal frame every
~4.7 s and got several things wrong. Claims are tagged **[measured]**, **[inferred]** or **[open]**.

Companion docs: [`AS_BUILT_AGILITY_FORGE.md`](./AS_BUILT_AGILITY_FORGE.md) (geometry/material
provenance), [`measured_heating_curve_20260717.csv`](./measured_heating_curve_20260717.csv)
(the sampled curve — **note the caveats in §3**).

---

## 0. UPDATE 2026-08-04 — Colton answered, and it changes three things

His reply (`EMAIL-0804`) identified the cameras and the drive frequency. Read this before the rest.

### The camera is short-wave, not LWIR — the dominant uncertainty is retired

**Optris PI 1M**, 0.85–1.1 µm, 382×288 @ 27 Hz (a documented sub-frame mode — an exact match to
these frames), range 450–1800 °C. A PI 640i (LWIR, ≤900 °C) is also connected but cannot be the
source: our data reaches 1015 °C.

The whole calibration was gated on the fear that this was LWIR, where a 2× emissivity error moves
the reading ~438 K (37%) and could inflate fitted `q_peak` ~2×. **At 1 µm the same error moves it
~65 K (5.5%)** — 6.7× less sensitive, which is exactly why Optris sells this camera for hot metal.
Emissivity is now a **~5% effect**. Colton uses "the default settings for metals"; the exact value
is in his HMR repo at `HMR/cpp/optris_cam`.

### The 450 °C floor explains the "camera floor" — and kills episode 1

The PI 1M cannot measure below **450 °C**. The 280.95 °C value on 672/1212 sampled frames is an
**out-of-range indicator, not a temperature**. Consequences:

- **Episode 1 peaked at 447 °C ⇒ it is entirely at/below the floor and is not a usable
  measurement at all.**
- Any reading below ~450 °C anywhere must be discarded, including the very start of episode 3
  (484 °C, only just above).
- ⚠️ It also means *"no hot metal in view"* and *"rod present but below 450 °C"* are
  indistinguishable in the thermal feed. The episode structure in §2 survives only because the
  **Tormach X axis** independently confirms the rod was retracted — see §2.

### 🚨 The drive frequency was wrong by 83×, and it retracts a headline

Colton: *"coil frequency is adjusted by Ambrell's controller dynamically and it tends to sit
around 250kHz for a 1.5" piece of 316L."* We had assumed **3 kHz**.

| | 3 kHz (assumed) | **250 kHz (actual)** |
|---|---|---|
| skin depth δ @1273 K | 10.22 mm | **1.12 mm** |
| R/δ | 1.9 | **17** |
| fraction of cross-section heated directly | 38% | **5.3%** |
| skin→bulk equilibration | 19.5 s | **0.23 s** |

This is a **thin-skin** process, not through-heating. And **the grid cannot resolve it**:
dx = 5.46 mm makes the skin ~4.9× finer than one cell. `options.py`'s own note warned this scheme
"would be inadequate at a much higher frequency" — that condition is now met.

**RETRACTED: "`q_peak` is 14.3× too hot".** That compared the *surface* deposition rate
`q_peak/(ρCp)` against a *bulk* measured rise — fair only when δ ~ R, which 3 kHz implied. At
250 kHz the skin equilibrates in 0.23 s, so the bar heats at the **volume-averaged** rate:

- volume-averaged rate at `q_peak = 2.5e8`: **2.93 K/s** vs measured **3.84 K/s** ⇒ **0.76×**
- i.e. the committed value is ~24% **too LOW**, not 14× too high
- implied `q_peak` ≈ **3.28e8**, which lands close to the independent physics-derived ~3e8 from the
  2026-06-19 review

A good reminder that a confident ratio is only as good as the regime assumption underneath it.

### The resolution problem, demonstrated rather than argued [measured]

Re-running the induction smoke test with the corrected frequency:

| | δ = 10.22 mm (assumed) | δ = 1.12 mm (actual) |
|---|---|---|
| heating rate | 12.9 K/s | **1.01 K/s** |
| ΔT over 80 steps | 1796 K | **134 K** |

A **13× drop**. Roughly 9.1× of that is genuine physics (the effective absorption depth is δ/2).
The remainder is **discretization**: particle spacing is ~2.7 mm, so the outermost particle centres
sit ~1.37 mm below the surface, where `w_skin = exp(−2·1.37/1.12) ≈ 0.087`. **No particle ever
samples the peak of the exponential**, so the deposition is under-integrated — a crude
layer estimate puts that loss near 2×, which combined with the 9.1× is the right order for the
observed 13×.

So the model now *under*-heats for a numerical reason, on top of `q_peak` itself being ~24% low.
Do not tune `q_peak` upward to compensate — that would bake a discretization error into a physical
constant. Fix the sampling first. (The layer estimate is analytic; actual particle depths were not
instrumented.)

---

## 1. The file is not just a thermal feed [measured]

| topic | rate | payload |
|---|---|---|
| `hmr/sensors/thermalcam` | ~27 Hz | 382×288 `16UC1`, **deci-kelvin** |
| `hmr/press/state` | ~420 Hz | `live_force_kn`, `live_stroke_mm`, `live_position_mm` |
| `hmr/ard/state` | ~35 Hz | `heater_on/ready/fault`, door, e-stop, air pressure, ram position |
| `hmr/torm/state` | ~22 Hz | `actual_position` = **[X, Y, Z, A, …]** + live G-code `command` |
| `forge/state/x_offset_mm` | **1 message** | static config: **128.2 mm** |

Two consequences worth stating plainly:

- **The manipulator state is logged.** Rod pose including the **A rotary axis** is available at
  22 Hz, so a replay of the manipulation is a data problem, not a guessing problem.
- **`forge/state/x_offset_mm = 128.2`** is almost certainly the same quantity as our
  `coil_offset_x = 129.89 mm`. They agree to **1.3%** — the first independent corroboration of a
  number that had never been checked. [inferred]

---

## 2. There are THREE heating episodes, not one [measured]

Scanning all chunks at ~1.5 Hz (1212 frames), classifying a frame as "hot metal in view" when
>1% of pixels exceed 400 °C:

| | window | duration | p99 range | **Tormach X** |
|---|---|---|---|---|
| **Episode 1** | 21.5 – 72.8 s | 51 s | 400 → 447 °C | ~290–316 mm |
| *gap A — rod withdrawn* | 74 – 225 s | 151 s | — | **0.0 mm** |
| **Episode 2** | 226.1 – 285.0 s | 59 s | 402 → **941 °C** | **290.5 mm** |
| *gap B — rod withdrawn* | 286 – 370 s | 84 s | — | **0.0 mm** |
| **Episode 3** — Colton's window | 371.2 – 568.4 s | 197 s | 484 → 925 °C | **350.0 mm** |

The X column is what makes this reading safe. Since the PI 1M cannot see below 450 °C (§0), the
thermal feed alone cannot distinguish "rod removed" from "rod present but cool" — but the Tormach
log puts X at **exactly 0.0 mm** through both gaps, so they are genuine withdrawals.

⚠️ **Episode 2 was at a DIFFERENT axial position: X = 290.5 mm against episode 3's 350.0 mm —
59.5 mm less insertion.** An earlier version of this doc called episode 2 a second dataset at the
same drive state on the strength of the shared `heater_on` block. The heater state *is* shared, but
the geometry is not, so it is **not a replicate**. That may be a feature — different coil/rod
overlap samples `f_axial` differently and carries independent information — but it must be modelled
as a different configuration, not averaged with episode 3. **Episode 1 is unusable regardless**
(entirely below the camera floor).

The two gaps are the ~235 s of "black" frames: the rod is simply not in view. The earlier
characterisation of everything before 380 s as *"unrelated activity"* undersold it — **episode 2
is a full heat to essentially the same peak temperature as episode 3.**

### The heater was on continuously across both [measured]

`heater_on` transitions 10 times over the file, but the last rising edge is at **213.9 s** and it
stays true until **571.2 s**. So episodes 2 and 3 sit inside **one continuous heater-on block**,
i.e. the same drive state. Constant power across Colton's window is now *measured*, not assumed.

- `heater_fault`: 222 samples, **all between 113.4 and 126.2 s** — inside gap A, before both
  usable episodes. Neither episode is affected. [measured]
- `heater_ready`: toggles at **exactly 50.0% duty**, constant to ±0.1% across every 20 s bin of
  episode 3 while the rod climbs 513 → 925 °C. A load-regulating signal would vary; this does not.
  Read as a heartbeat/handshake bit, **not** a 50% power derate. [inferred — worth confirming]

---

## 3. Colton's "380 s to the end" — the start is right, the end is not [measured]

Reading every frame at full rate over the last 15 s: p99 falls **925 → 400 °C between 568.4 and
570.4 s**, then sits flat at ~400 °C until the file ends at 573.3 s.

That collapse is **~200 K/s**. Radiative cooling of a 38.1 mm 316L rod at 925 °C is
`εσ(T⁴−T∞⁴)·(2/r)/(ρCp)` ≈ **1 K/s** — about **200× too slow** to explain it. The rod is leaving
the field of view; the flat tail is background, not metal. Then `heater_on` drops at 571.2 s.

**Order of events: rod withdrawn → then power off.** This is *not* the power-off cooling curve
worth requesting from Colton — far too fast to carry loss information.

⚠️ `measured_heating_curve_20260717.csv` still spans the whole file and therefore **contains one
contaminated sample at t = 571.07 s reading 418 °C.** Anything fitting "to the end" will be
dragged down ~500 °C by it. Use `HEAT_START_S` / `HEAT_END_S` from `agforge/mcap_thermal.py`.

---

## 4. Motion during episode 3 [measured]

From `hmr/torm/state`, 4114 samples in 380.0–568.0 s:

| axis | behaviour during the heat |
|---|---|
| X | **exactly 350.0000 mm, zero moves** |
| Y | −104.0 mm, **never moves** |
| Z | 0.0, **never moves** |
| **A** | 0 → 360°, **2962 moves** |

X is *literally* constant — Colton's "I did not move the rod in x direction anytime during the
heat" is exact, not approximate. The A-axis activity is "we scroll the bar in the coil (theta) so
that the heat is even". The two large p99 drops at 372.6–375.3 s (hot fraction collapsing
0.25 → 0.06) are the rod being **inserted**, before the heat proper — which is why 380 s is a
good start point.

### The withdrawal is commanded in G-code — independent confirmation [measured]

The §3 conclusion was reached from thermal data alone. The motion log confirms it from a
completely different sensor:

```
t=567.684 s   G53 G1 X191.3000 Y-104.0000 Z0.0000 A16.0000 F3000.0
              X ramps 350.0 -> 191.3 mm over 567.68 -> 570.89 s
```

A 158.7 mm retract at F3000 (= 50 mm/s) predicts **3.17 s**; the log shows **3.21 s**. The
thermal collapse starts at 568.4 s, ~0.7 s into that move. So the rod is unambiguously being
withdrawn, not cooling — and `heater_on` only drops afterwards, at 571.2 s.

Useful by-products: **X = 350.0 mm is the "in the coil" position and X = 191.3 mm is the park/
retract position** (the same destination appears in the very first logged command). Together with
`forge/state/x_offset_mm = 128.2` these are the numbers to use when pinning rod↔coil registration
rather than inferring it from pixels.

**Replaying the rotation would change our predicted temperature by ~0%.** `f_axial` is an
axisymmetric, peak-normalised profile, so rotating a cylinder about its own axis is a thermal
no-op in this model. Colton rotates the bar precisely to average out the *real* coil's azimuthal
non-uniformity — which is what makes our axisymmetric approximation fair in the first place.
Note this is an argument about *this* model, not a general one. [inferred]

---

## 5. Three corrections to previously-held beliefs

1. **The camera floor is ~281 °C, not ~450 °C.** 672 of 1212 sampled frames have their median
   sitting exactly on 280.95 °C. The "450 °C floor" appears to have come from misreading episode
   1's *peak* (451 °C) as a floor. [measured]
2. **`p99` is not a clean rod-surface temperature.** By t≈480 s onward, **98–100% of the frame
   exceeds 400 °C** — the coil itself glows and the background is hot. We are calibrating against
   a statistic that mixes rod, coil and background. This also explains why self-calibrating the
   pixel scale from "rod edges" is threshold-sensitive to ±25%: there is no clean rod edge to
   find, only bright gaps between coil turns. [measured]
3. **Episode 3 does not start from a cold or uniform rod.** It begins **84 s** after episode 2
   peaked at 941 °C. The surface has fallen back below 400 °C but the bulk cannot have. The
   initial condition is a rod carrying an unknown internal gradient — **not** a uniform 498 °C.
   This is currently the largest un-modelled error source in any fit to episode 3. [measured
   structure, inferred consequence]

### What the frames look like

Rod vertical; coil turns cross it as bands. The bright horizontal bars are rod glimpsed *between*
turns, and there are consistently **3 bright bars = the gaps between 4 turns** — an independent
visual confirmation of the 4-turn count in `AS_BUILT_AGILITY_FORGE.md`. In the rod-out frames the
bare helix is visible as diagonal zigzags.

---

## 5b. The axial profile, and a better pixel scale [measured]

Committed as [`measured_axial_profile_20260717.csv`](./measured_axial_profile_20260717.csv):
16 profiles at 12 s intervals across episode 3, 288 rows each (one per image row).

**Orientation was measured, not assumed.** The row-wise profile carries a strong periodic
signature (autocorrelation **+0.47 at ~103 px**); the column-wise profile carries none. So image
rows are the rod axis, and the periodic dips are coil turns occluding the rod.

**Pixel scale, from the frame rather than from guessing at rod edges:**

| quantity | value | spread over 6 frames |
|---|---|---|
| coil-turn dip spacing | **102.5 px** | 102–103 |
| rod width (largest contiguous run above half-range) | **159 px** | 157–163 |

## ✅ RESOLVED 2026-08-05: the pixel scale is 3.8791 px/mm, from Colton

Verified at source, not inferred. `pcloud:AgilityForge/2026-06-15_T4_bulk/README.md` quotes his
2026-06-29 email verbatim:

> This is 1.5" 316L Stainless Steel.
> The thermal camera has about **3.8791 pixels per mm** in the plane of the workpiece.
> Note that the coil is **~3" in length** and you can only see **1–1.5"** in the thermal frame.

⚠️ Stated for the **06-15** run. Applying it to 07-17 assumes the optics did not move between
June and July — unconfirmed, but nothing in the frames contradicts it.

**This retires the previous 4.173 px/mm figure, which was over-claimed three separate times.**
The true value is the *bottom* of the 3.88–4.17 bracket. Two consequences follow, and both settle
questions this document previously left open:

**1. The "159 px band is the rod" identification is CONFIRMED, and the coil-bore alternative is
refuted.** At 3.8791 px/mm the band measures **41.0 mm** against a rod of exactly 38.1 mm — a 7.6%
overshoot, which is what thermal blooming should do, and it is the same direction the old §5b
warning predicted. The alternative reading (that the band was the ~45 mm coil bore) would require
3.53 px/mm and is now **excluded** by ~9%.

**2. The coil pitch is now direct, and it was biased LOW.** It no longer needs the rod as a ruler:

| | old (scale-free ratio) | corrected (direct) |
|---|---|---|
| coil-turn pitch | 24.56 mm | **26.42 mm** (= 102.5 px ÷ 3.8791) |

The old ratio method silently assumed the 159 px band was *exactly* 38.1 mm, so it inherited the
blooming error and came out 7.6% low — exactly the observed gap (24.56 × 1.076 = 26.43 ✓).

**26.42 mm = 1.04 in per turn independently matches the user's photo read** ("a little over an inch
per turn"), which is a genuine third-party confirmation of both the pitch and the scale.

### ⚠️ But `coil_length` is now MORE uncertain, not less — and the sign is disputed

The convention matters, and the two defensible ones straddle the committed value:

| reading | coil length | vs Colton's "~3 in" (76.2 mm) |
|---|---|---|
| first-to-last turn centre, 4 turns = 3 × 26.42 | **79.3 mm** (3.12 in) | **+4% — agrees** |
| committed | 88.9 mm (3.50 in) | +17% |
| current-sheet smearing, 4 × 26.42 | **105.7 mm** (4.16 in) | +39% |

The kernel (`base_mpm_solver.py:517`) models a **continuous current sheet** of full length
`coil_length`, and the rigorous smearing of N discrete turns onto a sheet is N × pitch — which
would argue for 105.7 mm. But that disagrees with Colton's physical "~3 inch" by 39%, whereas the
turn-centre span agrees to 4%.

**Committed `coil_length` = 88.9 mm is left unchanged**, because it sits near the middle of a
76–106 mm bracket that the evidence genuinely does not close. Moving it to either end would be
over-claiming again, which is the exact failure this section is correcting. `q_peak` fits roughly
1:1 against `L_eff`, so this is a **±17% band on any fitted `q_peak`** and is now the dominant
*geometric* uncertainty. [open — needs Colton's pixel-mapping code, or a turn count from the photos
at known scale]

### "You can only see 1–1.5 inch in the thermal frame"

Colton's own words, and they constrain the observation model hard. The 382 px frame spans
**98.5 mm** at 3.8791 px/mm, yet only **25–38 mm** of *workpiece* is visible — so **most of the
frame is coil and background, not rod**. This independently corroborates §5.2 (from ~480 s onward
98–100% of the frame exceeds 400 °C) and the two failed motion checks below, and it is why a
synthetic-camera observation model is mandatory rather than optional.

### Two attempts to confirm the scale from motion — both FAILED [measured]

The withdrawal is a *known* 158.7 mm displacement at a logged 50 mm/s, which looked like a way to
get the scale with no dependence on edges at all. Recording the failures because each one says
something about the scene, and because the temptation is to keep fishing until a number appears:

1. **Whole-pattern cross-correlation of consecutive frames: no shift at all.** The axial profile
   does not move while X travels 86 mm. Expected in hindsight — a uniform cylinder sliding **along
   its own axis** is translationally invariant, so there is nothing to correlate. It also confirms
   the periodic dips are **static**, consistent with them being the coil rather than the rod.
2. **Tracking the rod's trailing end as a sweeping thermal edge: R² = 0.31, residual 14 px.** No
   clean edge crosses the frame. Instead `hot` falls 620 → 411 °C while `cold` falls 430 → 379 °C:
   the whole scene **dims together** rather than an edge sweeping through it.

Taken together these say the field of view is dominated by the coil and its surroundings, with the
rod seen through it — which reinforces §5.2, and which Colton's "you can only see 1–1.5 inch in the
thermal frame" independently confirms.

~~The identification of the 159 px band as *the rod* is plausible but unconfirmed. If that band is
instead the coil bore (~45 mm) the scale would be ~3.5 px/mm.~~ **✅ SETTLED 2026-08-05 — the band
is the rod.** Colton's 3.8791 px/mm puts it at 41.0 mm against the rod's 38.1 mm (blooming), and
excludes a 45 mm coil bore by ~9%. See the resolved-scale section above.

**The profile itself.** Temperature falls monotonically from the top of the frame downward; the
peak sits at or above y=0, i.e. **the coil centre is outside the field of view** and we are seeing
the downhill side plus, presumably, part of the 50 mm of rod Colton says protrudes.

| | top of frame | bottom of frame | drop across the 74.2 mm in view |
|---|---|---|---|
| start of window (380 s) | 498 °C | 281 °C | ~~217 °C~~ **INVALID — see the floor warning below** |
| end of window (568 s) | 908 °C | 583 °C | **324 °C** (valid) |

This is the measurement that constrains **axial conduction**, and it is a far stronger constraint
than the single-point heating curve — but using it needs one thing we do not yet have: **where the
image window sits relative to the coil centre.** That registration offset is a new free parameter.
`forge/state/x_offset_mm = 128.2` and the Tormach `X = 350.0` heat position are the numbers most
likely to pin it. [open]

⚠️ Rows within ±18 px of a detected coil dip are **interpolated**, not measured — the CSV carries
the filled values. Dips sit near y≈54 and y≈157.

### 🚨 HALF THE AXIAL PROFILES ARE PARTLY BELOW THE CAMERA FLOOR

Found during the 08-04 closeout, *after* this CSV was committed. The profile was extracted before
we knew the PI 1M cannot read below **450 °C** (§0), so the cold end of each early profile is the
out-of-range floor, not the rod:

| t [s] | top °C | bottom °C | rows < 450 °C | valid rows |
|---|---|---|---|---|
| 380 | 498 | 281 | **233** | 55 |
| 428 | 685 | 378 | 53 | 235 |
| 464 | 786 | 440 | 20 | 268 |
| **476 →** | 802 | 465 | **0** | **288** |

**Only 8 of the 16 profiles (t ≥ 476 s) are fully valid.** At t = 380 s, **81% of the profile is
floor**, not measurement.

⇒ The "axial drop of **217 °C** at the start of the window" quoted above is **wrong** — most of
that span is clamped at the floor. The **324 °C at the end of the window is valid.** Any fit
against this file must either restrict to t ≥ 476 s or mask cells below 450 °C; the CSV stores raw
values with no mask, deliberately, so the successor can choose.

By contrast `measured_heating_curve_20260717.csv` is **unaffected inside the usable window**: 0 of
its 69 samples in 380–568.4 s fall below the floor (54% of the *whole-file* samples do, but those
are outside the window anyway).

## 6. What this means for calibration

- **Episode 2 is a genuine second dataset**, sharing a continuous heater-on block with episode 3.
  Its initial condition is *better constrained*: it is preceded by 151 s of cooling from an episode
  that only reached 447 °C. It is shorter (59 s) but spans nearly the same temperature range.
- That enables real validation rather than a self-consistency check: **fit on one episode, predict
  the other.**
- The observation model is not optional. Comparing a simulated surface temperature to a p99 over a
  frame that is ~100% hot metal, coil and background is not a like-for-like residual. Sim output
  should be projected through an emissivity/band/occlusion camera model and the *same* statistic
  computed on both sides.

### Still open

- ✅ ~~Camera make/model~~ — **answered**: Optris PI 1M, 0.85–1.1 µm. See §0.
- ✅ ~~Coil drive frequency~~ — **answered**: ~250 kHz. See §0.
- **The exact emissivity setting** — Colton uses "the default settings for metals" and the
  calibration is pulled from Optris servers at runtime. The value is recoverable from his HMR repo
  (`HMR/cpp/optris_cam`, access granted 2026-08-04). Now a ~5% effect rather than a ~2× one, so
  worth having but no longer blocking.
- **🚨 Grid resolution vs a 1.12 mm skin** — new, and the most serious modelling issue on the
  table. dx = 5.46 mm. Deposition is effectively a surface flux; a volumetric source on this grid
  cannot represent it. Options: refine locally, or reformulate as a surface boundary condition.
- **Was episode 2 at the same 420.2 A?** Colton quotes that current for "this" run. `heater_on`
  proves the *enable* was continuous, not that the *setpoint* was unchanged — and we now know the
  rod position differed, so episode 2 needs its own treatment either way.
- **Same rod throughout?** Nothing in the log says so.
- Confirmation that `heater_ready` is a heartbeat rather than a power signal.
- The frequency "tends to sit around" 250 kHz — it is a dynamically tracked resonance, not a
  setpoint, so δ drifts with load temperature and coupling during the run.

---

## 7. Reproducing this

`agforge/mcap_thermal.py` carries the thermal decoder plus `state_series()` for the JSON topics.
The episode table comes from sweeping `thermal_frames(chunk_stride=1, frames_per_chunk=3)` and
thresholding the fraction of pixels above 400 °C; the heater and Tormach series come from
`state_series("hmr/ard/state")` and `state_series("hmr/torm/state")`.
