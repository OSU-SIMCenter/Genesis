# The 2026-07-17 induction dataset — what is actually in it

**Status 2026-08-04.** Everything here was derived from `20260717_135009.mcap` directly. It
supersedes the earlier reading of this file, which was based on sampling one thermal frame every
~4.7 s and got several things wrong. Claims are tagged **[measured]**, **[inferred]** or **[open]**.

Companion docs: [`AS_BUILT_AGILITY_FORGE.md`](./AS_BUILT_AGILITY_FORGE.md) (geometry/material
provenance), [`measured_heating_curve_20260717.csv`](./measured_heating_curve_20260717.csv)
(the sampled curve — **note the caveats in §3**).

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

| | window | duration | p99 range |
|---|---|---|---|
| **Episode 1** | 21.5 – 72.8 s | 51 s | 400 → 447 °C |
| *gap A — rod out of frame* | 74 – 225 s | 151 s | — |
| **Episode 2** | 226.1 – 285.0 s | 59 s | 402 → **941 °C** |
| *gap B — rod out of frame* | 286 – 370 s | 84 s | — |
| **Episode 3** — Colton's window | 371.2 – 568.4 s | 197 s | 484 → 925 °C |

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

- **Camera make/model + whether the deci-kelvin matrix is emissivity-corrected** — still the
  biggest single lever (LWIR worst case would make the true rise only 52–69% of measured).
- **Coil drive frequency** — sets δ; currently assumed 3 kHz.
- **Were episodes 1 and 2 at the same 420.2 A?** Colton quotes that current for "this" run.
  `heater_on` proves the *enable* was continuous, not that the *setpoint* was unchanged.
- **Same rod throughout?** Nothing in the log says so.
- Confirmation that `heater_ready` is a heartbeat rather than a power signal.

---

## 7. Reproducing this

`agforge/mcap_thermal.py` carries the thermal decoder plus `state_series()` for the JSON topics.
The episode table comes from sweeping `thermal_frames(chunk_stride=1, frames_per_chunk=3)` and
thresholding the fraction of pixels above 400 °C; the heater and Tormach series come from
`state_series("hmr/ard/state")` and `state_series("hmr/torm/state")`.
