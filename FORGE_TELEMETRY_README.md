# Press-telemetry analysis for the Agility Forge mcap datasets

Companion to `inspect_mcap.py` / `sample_mcap.py` / `stats_mcap.py` (structure and
sampling) and `agforge/mcap_thermal.py` (the thermal decoder). These scripts cover the
**press/force side**, which is what validates the mechanical model.

**Run everything through pixi.** The scripts shell out to the `zstd` CLI, which exists
only inside the pixi envs -- outside them they die with "zstd CLI missing":

```bash
pixi run python extract_press_mcap.py <file.mcap> <out.npz>
```

WSL can read the staged file in place at `/mnt/c/...`; there is no need to copy an 8.58 GB
mcap into the WSL filesystem, and doing so over `\\wsl.localhost\` risks silent
zero-padding.

## Pipeline

| step | script | notes |
|---|---|---|
| 1 | `extract_press_mcap.py <mcap> <npz>` | one pass, ~195 s for 8.58 GB. Skips thermal frames without parsing them (they are ~99% of the volume) and caches every JSON topic. Also writes `<npz stem>_utaken.json`. |
| 2 | `analyze_press_mcap.py <npz> <utaken.json>` | characterises the signal, segments blows, writes `<stem>_blows.npz` |
| 3 | `align_blows_mcap.py <npz> <blows.npz> <utaken.json>` | pairs blows to commands **by timestamp** |
| 4 | `blow_detail_mcap.py`, `decode_press_mcap.py`, `invert_die_width_mcap.py` | the checks that established the decode |
| - | `dump_attachments_mcap.py <mcap> <dir>` | writes embedded attachments byte-for-byte |

Verify step 1 by checking the reported message counts against the summary index that
`inspect_mcap.py` prints -- they should match exactly.

## What this established on `20260615_180456_T4_bulk.mcap`

- `live_position_mm` **is the die gap**: contact on blow 1 begins at ~38.2 mm against a
  known 38.1 mm billet.
- `live_position_mm + live_stroke_mm == 227.3 mm` exactly, over all 858,164 samples.
  Stroke is therefore **not independent data**.
- The plan/`u_taken` `rho` is a **half**-thickness: achieved gap equals `2*rho` on 38 of
  47 blows, median error 0.04 mm.
- The press is **force-limited near 110.2 kN**. The 9 blows that miss their commanded gap
  are exactly the 9 that reach that limit. The force reading itself is genuine (all values
  distinct, no dwell at the maximum) -- it is a control stop, not a clipped sensor.
- The file holds **two forging programs**, not one: `seq` restarts, so program 1 runs plan
  lines 0-14 and program 2 runs lines 5-43. **Do not align blows to plan lines by index.**

⚠️ `align_blows_mcap.py` exists because index alignment silently breaks the moment a
retry or a second program inserts an extra episode. Align on time.
