# 6063-T52 mechanical card for roll1 (cold Agility Forge)

Operating point: **6063-T52 aluminum, room temperature, isothermal, ~1 /s**.
Dataset: Colton Wright `agf/HMR/datalog/2026-08-21/roll1` (README: Ø 38.1 mm, L0 75 mm, `heat_after: false`).

This is **not** the hot 316L 17-hit card. Do not overwrite `ACTIVE_MATERIAL = STEEL_316L` or the 1000 °C Johnson-Cook numbers with these. Add a second material and switch it for this replay.

Last sourced: 2026-09-10. Tables below were re-read from the PDFs/pages named in each row, not from search snippets.

## What is already pinned by the dataset

| Knob | Value | Where |
| --- | --- | --- |
| Alloy / temper | 6063-T52 | README ("6063 Al" + "6063 (T52)") |
| Temperature | room temp, no heat | README; every jsonl action has `heat_after: false` |
| Billet | Ø 38.1 mm (R0 = 19.05 mm), L0 = 75 mm | README; jsonl `workpiece_length_mm` starts at 75.0 |
| Tools | anvil 2, hammer 2 | jsonl |
| Constitutive class | isothermal elasto-plastic (JC at T*=0) | implied: no thermal, no DRX window |

Genesis `jc_C` is **dead code** in the MPM kernel (see in-repo `docs/316L_MECHANICAL_PROPERTIES.md`). The rate term will not fire even if we fill it. Calibrate `A, B, n` **at the forge rate** so the missing `C` is not a second error.

## Recommended Genesis `MaterialOptions` (replay default)

| Field | Value | Confidence | Why |
| --- | --- | --- | --- |
| `E` | 68.9e9 Pa | high | 10e6 psi class value. [MakeItFrom 6063-T52](https://www.makeitfrom.com/material-properties/6063-T52-Aluminum) lists 68 GPa (rounded). |
| `nu` | 0.33 | high | Same MakeItFrom page. Milder than the 316L 0.383 P-wave issue. |
| `rho` | 2700 kg/m³ | high | MakeItFrom 2.7 g/cm³. |
| `von_mises_yield_stress` | 137.9e6 Pa | high typical | Ilsco Table II typical 0.2 % YS = 20.0 ksi. Spec is a **band**, not a point (below). |
| `use_johnson_cook` | True | — | Same kernel as 316L; T*=0 so thermal term is off. |
| `jc_A` | 137.9e6 Pa | high typical | Pin to typical T52 yield. Same rule as 316L: A **is** initial yield; do not free-fit it to ~0. |
| `jc_B` | 173.6e6 Pa | medium-low | **No T52 flow curve found.** Fitted so σ(ε_p=0.08) ≈ true stress of typical UTS 26 ksi at 8 % (179.3×1.08 = 193.6 MPa). Exact B is `(σ_8 − A) / 0.08^0.45` in `al_6063_t52_mechanical.py`. |
| `jc_n` | 0.45 | medium-low | Typical Al Ludwik exponent; not measured on this bar. |
| `jc_C` | 0.0 | n/a (dead) | Kernel ignores C. If it ever starts reading C, 0.013 from AA6063-T6 QS tensile is the 6063-family number in the **press** rate window, not the SHPB 0.054. |
| `jc_eps0` | 1.0 /s | high (convention) | Same operating-point convention as the 316L card. Neutral rate term at ~1 /s. |
| `jc_T_ref` | 293.15 K | high | Room temp. T*=0. Do **not** copy 316L's 1273.15 K. |
| `jc_T_melt` | 933.15 K | high | 660 °C, conventional Al Tm used by Wang 2023. MakeItFrom liquidus 650 °C / solidus 620 °C. Inert while isothermal at T_ref. |
| `jc_m` | 1.0 | unused | Inert at T*=0. Do not enable thermal softening from this number. |

Thermal / induction: **off**. `AGF_BILLET_TEMP_K = 293.15` is the correct value for this dataset (it is the wrong value for hot 316L).

### Flow stress of the recommended card

σ = A + B ε_p^n  (T*=0, C unused)

| ε_p | σ (MPa) |
| ---: | ---: |
| 0.00 | 138 |
| 0.05 | 183 |
| 0.08 | 194 |
| 0.15 | 212 |
| 0.30 | 239 |

Valid as a **cold, shallow-bite** card (Colton: max ΔD ≈ 4 mm). Do not use past ~0.30 without a measured T52 curve; JC is monotonic and there is no DRX peak to miss at room temperature.

## Handbook strengths (T52 is not T6)

[Ilsco extrusions mechanical-property matrix](https://ilscoextrusions.com/wp-content/uploads/2019/07/IEI-Mechanical-Property-Matrix-for-new-website.pdf) (copies Aluminum Association / ASTM B221 style limits):

| | ksi | MPa |
| --- | ---: | ---: |
| T52 spec yield (up thru 1.000 in) | 16.0 min – 25.0 max | 110 – 172 |
| T52 spec UTS | 22.0 min – 30.0 max | 152 – 207 |
| T52 spec elongation | 8 % min | — |
| T52 **typical** yield / UTS / elong (Table II, not for design) | 20 / 26 / 12 % | 138 / 179 |
| T5 typical yield (same table) | 21 ksi | 145 |
| T6 typical yield | 30 ksi | 207 |
| T6 spec min yield (0.125–1.000 in) | 25 ksi | 172 |

[MakeItFrom 6063-T52](https://www.makeitfrom.com/material-properties/6063-T52-Aluminum): E 68 GPa, ν 0.33, ρ 2.7 g/cm³, typical YS 140 MPa, UTS 180 MPa, elong 8 %, liquidus 650 °C.

**Spec gap:** the bar is Ø 38.1 mm = 1.50 in. The T52 row is written "up thru 1.000 in". I did not find a 1.5 in T52 row. Using typical 20 ksi anyway; a mill cert for this bar would beat every number here.

**Same trap as Gupta 2012 for 316L:** T6 / T5 published JC parameters are the wrong **temper**, even when the alloy digits match.

## Published JC that must not be dropped onto this bar

### Rejected: IRJET 2020 AA6063-**T6** (student tensile, RT, 1e-4–1e-1 /s)

Source: [Rajput et al., IRJET 7(5) 2020, Table 2](https://www.irjet.net/archives/V7/i5/IRJET-V7I5290.pdf)

A=270 MPa, B=498 MPa, n=1.828, C=0.013, m=1.003 (m taken from Taylor-impact literature, not measured). ė0 = 1e-3 /s.

T6 typical yield is 207 MPa; their A=270 is already above handbook T6. n>1 is unphysical for Ludwik hardening of this alloy. Their own Fig. 9 / offset construction is internally messy. **Do not use.**

The only salvageable number is **C≈0.013** as a 6063-family QS rate coefficient in the press window (not SHPB).

### Rejected as a T52 card: Wang et al. 2023 6063-**T5**

Preprint tables re-read: [ResearchSquare rs-3070175](https://assets-eu.researchsquare.com/files/rs-3070175/v1_covered_f8a3f048-134b-4e2b-b5ba-38f68492f804.pdf). VoR: [Int. J. Thermophys. 44:133 (2023)](https://doi.org/10.1007/s10765-023-03239-6) (DOI noted; PDF of the journal version was not fetched this pass).

Their QS tensile at 22.4 °C: **Rp0 = 225.7 MPa** (Table 2) — above T52 spec **max** 172 MPa, in T6 territory. Traditional JC (Table 4, ė0=0.001 /s, T0=25 °C, Tm=660 °C): A=226, B=453, n=0.8984, C=0.054, m=1.377.

Their **C=0.054** and the modified JC (Table 5: A=352 at ė0=500 /s) are SHPB 500–6000 /s. Hydraulic forge is ~0.1–1 /s (`ε̇ ≈ v/h`). Wrong rate window.

Use Wang only as a **too-strong sensitivity case**, never as the T52 default.

### Rejected: ultrasonic-vibration 6063 JC, semi-solid 6063, hot radial-forging 6063

Wrong process and/or temperature.

## Geometry and hits (not material, but required to replay)

Canonical `forge_common.Hit.rho` is **full die gap** (mm), not radius. The jsonl `rho` values (15.76–18.20 mm) are **half-gap / remaining-radius**. Same logging convention as `2026-06-29.pt` `u[:,0]` before `_REAL_RHO_SCALE = 2`.

```
Hit.rho = 2 * jsonl.rho          # mm die gap
Hit.phi = jsonl.phi * pi / 180   # jsonl is degrees; Hit wants radians
Hit.z   = jsonl.z                # mm from the pinned face
```

Starting gap = 38.1 mm. A jsonl rho of 17.07 mm → gap 34.14 mm → ΔD = 3.96 mm, which matches Colton's "largest cold 6063-T52 ΔD is 4 mm". If you skip the ×2, the sim crushes the bar to a ~17 mm gap and the sequence is meaningless.

**Do not reuse** `forge_common.real_scale.REAL_STOCK_RADIUS_MM = 20.0` and `REAL_STOCK_LENGTH_MM = 59.0` (those were measured from the hot 316L mesh). Override to **19.05 mm / 75 mm**.

Phi: Genesis adapter already does `hinge = -hit.phi`. Convert degrees→radians **before** that. I have not opened `roll1.pt`, so I do not know whether that file already stores radians.

Which 12 hits ran: manifest has 12 timestamp triples; jsonl has 25 generated + 1 no-op. **Unverified** that executed = jsonl lines 2–13. Several of those 12 violate the README `MIN_DELTA_R=1`, `MAX_DELTA_R=2` mm bounds if R0=19.05 mm (hit 6 ΔR=2.69 mm; hits 7 and 9 ΔR<1 mm). Resolve against `roll1.pt` / the mcap before treating the first 12 as ground truth.

## Knobs that must not be copied from the 316L ship

| 316L (thermal-st-invariance / hot 17-hit) | roll1 |
| --- | --- |
| E=121.5 GPa, rho=7334, nu=0.383 | E=68.9 GPa, rho=2700, nu=0.33 |
| jc_A=100.3 MPa @ 1000 °C | jc_A=137.9 MPa @ 20 °C (cold Al is **stronger** than hot 316L at yield) |
| jc_T_ref=1273.15 K | 293.15 K |
| Arrhenius / DRX | off — room temp, no DRX |
| stock 20 mm radius × 59 mm | 19.05 × 75 mm |
| `max_force` tuned on a 316L blow that saturated the cap | do not copy a kN ceiling from 316L; sim "peak force" has been an artifact before |
| `pressing_speed` ~25 m/s | CFL artifact, not a material property; out of scope of this card |

Bar-wave speed at this card: sqrt(E/rho) ≈ 5050 m/s (316L ship ~4070). P-wave with ν=0.33 ≈ 6110 m/s. Timestep will shrink vs a 50 GPa toy steel, similar order to the sourced 316L card. Keep `cfl_use_pwave=True` if that is on.

## Scans / metrics (lower priority)

`roll1_sim_metrics.json` reports chamfer 0.29→0.43 mm and Hausdorff 1.44→~2.49 mm for actions 0–11, but **does not name which simulator wrote it**, so it cannot be cited as a Genesis result. The scan data for this sequence is distributed as `roll1.pt` (per-hit point clouds and the commanded actions), `roll1.h5`, and the raw press `.mcap`; `roll1.pt` is the one that carries the commanded `(rho, phi, z)` per hit.

## What I could not determine

- A measured 6063-**T52** flow curve (A, B, n from this bar or this temper). No published curve for this temper was found; see the rejected tables above.
- Mill cert / actual YS of Colton's extrusion.
- Whether `roll1.pt` already applied ×2 and deg→rad.
- Which 12 of the 25 jsonl actions were executed.
- Physical press speed / strain rate from this mcap (not downloaded). Bound remains ~0.1–1 /s.

## Sources

- [MakeItFrom 6063-T52](https://www.makeitfrom.com/material-properties/6063-T52-Aluminum) (page dated 2020-05-30; scraped 2026-09-10)
- [Ilsco mechanical property matrix](https://ilscoextrusions.com/wp-content/uploads/2019/07/IEI-Mechanical-Property-Matrix-for-new-website.pdf) (scraped 2026-09-10) — T52 spec band + typical 20/26 ksi
- [Rajput et al. IRJET 7(5) 2020](https://www.irjet.net/archives/V7/i5/IRJET-V7I5290.pdf) — rejected T6 JC; C=0.013 only
- [Wang et al. ResearchSquare rs-3070175](https://assets-eu.researchsquare.com/files/rs-3070175/v1_covered_f8a3f048-134b-4e2b-b5ba-38f68492f804.pdf) — T5 JC tables; VoR https://doi.org/10.1007/s10765-023-03239-6
- Colton roll1 README + `000_random_20260821_131234.jsonl` + `manifest.json` (local copies under this folder)
- Genesis `forge_common.hit.Hit` (rho = full die gap); `forge_common.real_scale` (×2 on logged rho); `agforge/options.py` MaterialOptions on `thermal-st-invariance`
