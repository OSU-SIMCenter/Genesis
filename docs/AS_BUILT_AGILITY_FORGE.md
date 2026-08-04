# As-built Agility Forge — digital twin parameters

Every geometric, material and operating parameter the simulation uses to represent the
real Agility Forge, with its **source** and an honest **confidence**. The goal is that
nobody has to re-derive where a number came from, and that anything resting on a guess
is visibly marked as one.

Written 2026-07-31 while converting the simulation from a generic 1-inch AISI 4340
billet to the actual machine.

## Sources

| Tag | What |
|---|---|
| `EMAIL-0629` | Colton Wright, "Re: Agility Forge Induction Coil Heating Data/Benchmark/Video", 2026-06-29 11:04 |
| `EMAIL-0717` | Same thread, 2026-07-17 14:21 (accompanies the 07-17 dataset) |
| `MCAP-0717` | `20260717_135009.mcap`, SHA-1 `33f66b9e8a042984c8f6ac51fd9db68dbf1d9020` |
| `PHOTO` | `IMG_9854/9855/9856.jpeg` — tape-measure photos of the coil, shipped with the 07-17 dataset |
| `LIT` | Published measurements, cited in `agforge/material_properties.py` |

## Billet

| Parameter | Value | Source | Confidence |
|---|---|---|---|
| Material | 316L austenitic stainless | `EMAIL-0629`, `EMAIL-0717` | high |
| Diameter | **38.1 mm** (1.5 in) | `EMAIL-0717` "316L 38.1mm rod" | high |
| Simulated length | 152.4 mm (8 × radius) | convention | **low — see below** |
| Protrusion past coil, far side | 50 mm | `EMAIL-0717` | high |
| Initial temperature | 293 K (ambient) | rod enters cold | medium |

**Length is genuinely unresolved.** `EMAIL-0717` describes a rod fully through a ~76 mm
coil with 50 mm protruding, so the physical rod is ≥126 mm plus whatever the chuck holds.
But `MCAP-0717`'s own attachment `999.jsonl` says `workpiece_length_mm: 75.0`. Those
cannot both describe the same object; the 75 mm figure is probably a modelled/forgeable
segment rather than the bar. The simulation models a *truncated* domain — the far end is a
cut plane with a Robin BC (`enable_fixed_end_bc`) representing conduction into the
unsimulated remainder — so `cylinder_height` is the simulated portion, not the rod. 8×R =
152.4 mm covers the coil plus the protrusion, which is the region that matters thermally.

Override with `RobotOptions.billet_length_m` once the real number is known.

## Induction coil

| Parameter | Value | Source | Confidence |
|---|---|---|---|
| Length | **88.9 mm** (3.5 in) | `PHOTO` direct count (see below); `EMAIL-0629` says "~3\"" | medium |
| Turns | **4** (possibly 3.5) | `PHOTO` direct count, corroborated by thermal-frame pitch | medium |
| Outer diameter | ~58–64 mm | `PHOTO` (IMG_9856 tape ~58–61; later direct read ~2.5 in) | medium |
| Tube outer diameter | ~1/4–3/8 in | `PHOTO` | low |
| Bore | ~45 mm | derived: OD − 2 × tube OD | medium |
| Current-path radius | **26 mm** | derived: outer radius − tube radius | medium |
| `coil_radius_multiplier` | **1.365** | 26.0 / 19.05 | medium |
| Coil current | **420.2 A** | `EMAIL-0717` | high |
| Drive frequency | **UNKNOWN** | — | **none — assumed 3 kHz** |
| Power rating (kW) | **UNKNOWN** | — | none |

The photos are handheld, with the tape at a different depth than the coil and no
orthogonal reference. Treat all photo-derived dimensions as **±10–15%**.

✅ **The turn count is now resolved: 4 turns** (possibly 3.5, leaning 4), read directly off the
coil photos, spanning ~3.5 in — a pitch of ~22 mm. That independently matches the thermal
frames, where the turns appear as cool bars occluding the rod spaced **~24 mm** apart. An
earlier reading of ~5 turns at ~17 mm pitch, taken by counting copper crossings against the
tape in IMG_9854, is **refuted** — it was the outlier of the three estimates.

Turn count still **does not enter the simulation**: `f_axial` is a *peak-normalised*
Biot–Savart profile, so turns, current and kW all cancel into `q_peak`. It would only matter
for predicting `q_peak` from first principles instead of fitting it. What the count bought us
is the **length**, via pitch × turns.

⚠️ **Coil length is now the dominant geometric uncertainty.** Colton's email says "~3\"";
the direct photo count gives ~3.5" for the current-sheet extent (4 × 22 mm), which is what
the finite-solenoid `f_axial` actually models — his "~3" may describe the helical body
without leads. `coil_length` is set to **3.5"**. The difference is not cosmetic: going
3.0 → 3.5 in raises the effective heated length (∫`f_axial` along the rod) by **15%**, and
fitted `q_peak` scales as ~1/L_eff, so it moves the fit by 15%. Worth confirming with Colton.

By contrast the coil *radius* barely matters: sweeping the current-path radius across the
full 26.0–28.6 mm range implied by both OD readings moves L_eff by **<1%**.

`coil_radius_multiplier` was **2.0**, which places the bore at 200% of workpiece OD while
the comment beside it cited ASM practice of 110–125%. The measured bore/OD is ~1.18,
inside that range; the code value was not.

**Frequency is the single most consequential unknown.** It is not in the mcap —
`hmr/ard/state` exposes the induction supply only as the digital flags `heater_on`,
`heater_ready`, `heater_fault`, with analog channels limited to air pressure and ram
position. There is no current, power or frequency telemetry anywhere in the file. It has
to come from Colton. 3 kHz is inferred from design practice for a 38.1 mm bar (the
d/δ ≈ 4 through-heating efficiency knee), **not measured**.

At 3 kHz in 316L, δ ≈ 10.2 mm against a 19.05 mm radius — d/δ ≈ 3.7. The deposition is
therefore fairly volumetric rather than a thin surface skin, which is why ~3.5 grid cells
from surface to axis is coarse but not obviously inadequate. That would stop being true at
a much higher frequency.

## Kinematics

| Parameter | Value | Source | Confidence |
|---|---|---|---|
| Rod rotated in θ during heat | yes, for even heating | `EMAIL-0717`, confirmed by `MCAP-0717` A-axis motion | high |
| X translation during heat | **none** | `EMAIL-0717`, confirmed by Tormach X pinned at 350 mm | high |
| `x_offset_mm` | 128.2 | `MCAP-0717` `forge/state/x_offset_mm` | high |
| `coil_offset_x` | −129.89 mm | pre-existing sim value | medium |

`coil_offset_x` (−129.89 mm) and the mcap's `x_offset_mm` (128.2) are suspiciously close.
They are plausibly the same quantity under opposite sign conventions, which would mean the
sim's coil offset was already derived from real forge data. **Not verified** — worth
confirming before relying on it.

## Material properties

Curves, sources and uncertainties live in `agforge/material_properties.py`. Summary:

| Property | Source | Note |
|---|---|---|
| k(T) | Ho & Chu, CINDAS recommended values for AISI 316 | **rises** 13.3 → 30.6 W/m·K; opposite slope to 4340 |
| cp(T) | Pichler et al., NIST SRM 1155a (DSC) | flat-ish; no transformation, no Curie point |
| ρ_e(T) | Ho & Chu | saturating, not linear |
| ρ | 7980 kg/m³ | |
| μ_r | 1.0 at **all** temperatures | 316L is paramagnetic; no Curie transition to model |
| ε(T) | Balat-Pichelin et al.; Hunnewell et al. | **0.25 → 0.70 across 1100–1500 K as oxide forms** |

**Emissivity is the largest calibration uncertainty and it is not a constant.**
`calibration.json` assumed 0.80; over the range the 07-17 run reached the real value is
nearer 0.3–0.45, so radiative loss was overstated roughly 2×. The solver currently takes a
scalar (now 0.40, representative near the hot end where T⁴ dominates). Making the radiation
term take ε(T) is the proper fix and needs a kernel change.

Worse, it contaminates the *measurement*: a radiometric IR camera converts radiance to
temperature using whatever emissivity it was configured with. If the camera assumed a fixed
value while the true surface emissivity climbed during the run, part of the apparent
temperature rise is an artefact — potentially ~12% of the observed ratio. **The camera's
emissivity setting is not recorded in the mcap and needs to come from Colton.**

## Thermal camera

| Parameter | Value | Source | Confidence |
|---|---|---|---|
| Resolution | 382 × 288 | `MCAP-0717` | high |
| Encoding | `16UC1`, deci-kelvin | `EMAIL-0717` + `MCAP-0717` | high |
| Frame rate | ~26 Hz | `MCAP-0717` | high |
| Pixel scale | 3.8791 px/mm | `EMAIL-0629` | **low for the 07-17 run** |
| Emissivity setting | **UNKNOWN** | — | none |

The pixel scale comes from the **June** email. The framing changed between runs —
`EMAIL-0629` says "you can only see 1-1.5\" in the thermal frame", `EMAIL-0717` says "most
of the coil is in view". Do not assume it carries over.

`EMAIL-0629` also contains an offer that was never taken up: *"I can send you a code that
maps the thermal feed pixels onto the workpiece given some config & tormach state if you
would like."* That would settle the pixel mapping outright.

### RawImage wire format — a trap

This stream's RawImage numbers its fields `1=timestamp 2=width 3=height 4=encoding
5=step 6=data`: **no `frame_id`**, everything shifted down one from stock
`foxglove.RawImage`, and width/height/step are `fixed32` (wire type 5), not varint. The
repo's `sample_mcap.py` assumes the stock layout *and* varint, so it silently extracts
nothing. Verified self-consistent: step 764 = 2 × 382, data 220032 = 382 × 288 × 2.

## What the 07-17 run actually is

**Not a steady-state run**, despite the label that attached to it downstream. Across
Colton's own clip window (380 s → end, 193 s) the surface temperature climbs
**513 °C → 928 °C monotonically**, with the rate decaying **~3.7 → ~1.2 °C/s**. It is
approaching a plateau — extrapolating to ~1000–1050 °C — but never reaches one. Colton's
email says only "clip toward the end of the mcap"; he never called it steady state.

For calibration this is an advantage: a decaying transient constrains absorbed power and
the loss terms independently, where a plateau would only give the loss balance. But it
makes the job a **curve fit against real seconds**, which puts weight on the
`thermal_time_scale` (S_T) mapping between simulated and physical time.

## Open questions

1. **Coil drive frequency and kW rating.** Sets δ directly; not in the data.
2. **Thermal camera emissivity setting and temperature range.** Scales the calibration target.
3. **Pixel scale for the 07-17 framing**, or Colton's pixel-mapping code.
4. **Physical rod length**, and what `workpiece_length_mm: 75.0` refers to.
5. **Coil turns count** — 5 or 6 changes the field profile.
6. Whether `coil_offset_x` and `x_offset_mm` are the same quantity.
7. Whether the ~43% of every thermal frame sitting in a narrow 454–492 °C band is a sensor
   floor, an emissivity artefact, or genuinely hot surroundings.
