# agforge — Agility Forge MPM adapter

Hot-forging simulation of the Agility Forge press, driven from recorded robot data.

## Running a forging sequence

Requires WSL with an NVIDIA GPU.

**Run from a login shell.** The CUDA driver lives in `/usr/lib/wsl/lib`, and only a login shell
puts it on `LD_LIBRARY_PATH`. Without it the unversioned `libcuda.so` that the backend `dlopen()`s
is invisible, and the simulation falls back to CPU **without reporting it** — a silent wrong answer
rather than an error.

```bash
wsl.exe -d my-ubuntu -- bash -lc '
  cd ~/GitHub/Genesis/aims-genesis/<your-worktree>
  AGF_CONTACT_RUNTIME_SWITCH=1 \
  AGF_BILLET_MESH=~/GitHub/Genesis/forge_common/main/outputs/real_meshes/billet_hit01_before_d8000.obj \
  python -m agforge.analysis.batch_arms --n-hits 17
'
```

`agforge.analysis.batch_arms` **exits 2** unless `AGF_CONTACT_RUNTIME_SWITCH=1`. Without it the
contact mode is baked into the kernels at scene build and every arm after the first would silently
run the first arm's configuration.

`AGF_BILLET_MESH` is required for real-forge geometry. It is produced by
`agforge/analysis/extract_real_meshes.py` and is **not stored in this repository** — it is a
generated artifact living in the `forge_common` outputs tree. Without it the simulation runs on a
built-in nominal cylinder and will not match recorded results.

The 316L card is fitted at **1000 C / 1 /s** (`jc_eps0 = 1.0` in `options.py`). A 0.373 /s figure
that appeared in the release commit message is the measured median *press* strain rate, not the
rate the card was calibrated at.

This README does **not** claim a verified 17/17-hit run on these defaults.

## Replaying a recorded toolpath

A forge run is handed over as a `*_toolpath.jsonl`: one record per action, each carrying the
commanded half-gap `rho`, the die angle `phi` in degrees, the distance `z` from the chuck face,
and the workpiece length at that point. `agforge.scripts.record_toolpath_hits` presses that
sequence.

This needs **forge_common**, which is a separate repository
([OSU-SIMCenter/forge_common](https://github.com/OSU-SIMCenter/forge_common), the released runs
used commit `73930cb`). Clone it beside this one and point `PYTHONPATH` at its `main` directory.
The same login-shell requirement as above applies: without `/usr/lib/wsl/lib` on
`LD_LIBRARY_PATH` the simulation falls back to CPU **without saying so**.

### One press, to time it

```bash
wsl.exe -d my-ubuntu -- bash -lc '
  cd ~/GitHub/Genesis/aims-genesis/<your-worktree>
  PYTHONPATH=~/GitHub/Genesis/forge_common/main \
  AGF_MATERIAL=6063T52 \
  AGF_CELLS_PER_DIAMETER=13 \
  python -u -m agforge.scripts.record_toolpath_hits \
    --jsonl /path/to/roll2_toolpath.jsonl --n-hits 1 \
    --length-from-toolpath --hit-dir /tmp/smoke
'
```

Scene construction dominates a short run — it is most of the wall clock for one press, and is paid
once per process rather than per press. At 13 cells across a 38.1 mm bar a press itself is of order
ten seconds on a laptop RTX 3060; expect the first press to look much slower than the rest.

The driver prints the grid, the stock geometry, the material and the press settings it resolved
before it starts, then verifies that the material card selected by `AGF_MATERIAL` is the card the
solver actually got. That check exists because setting part of a flow curve once left an aluminium
bar running on hot-steel constants, 27% too strong at yield, and the run completed and looked fine.

### The whole sequence

Drop `--n-hits` (or pass `0`) to run every press in the file, and give `--hit-dir` a directory to
keep: it writes `hit_NN.npz` per press plus `hit_00.npz` for the bar before the first press, and
releases the recorder buffer between presses so a 25-press sequence fits in 6 GB of device memory.

### Stock held in the chuck

The toolpath's length is the **exposed** length. The real bar continues into the chuck, and
simulating only the exposed length puts a rigid clamp directly behind the first blow.

```bash
  AGF_PIN_EXTRA_MM=25 AGF_CLAMP_DEPTH_MM=25 ... --z-shift-mm 25
```

`AGF_PIN_EXTRA_MM` adds the held stock on the pinned side and leaves the free end where it was.
`AGF_CLAMP_DEPTH_MM` grips that much and no more, so the exposed bar is free over its whole length.
`--z-shift-mm` moves every `z` by the same amount, because `z` is measured from the chuck face and
the chuck face has moved.

All three go together. The trap is the clamp: it defaults to a *fraction* of the billet, so on a
100 mm bar it grips 17.50 mm back from the pinned face while the innermost blows of the aluminium
programs are at z = 15.68 and 17.87 mm — the die lands on rigid particles and that material cannot
flow. `tests/test_stock_geometry.py` pins those numbers.

### Point clouds

```bash
  python -m agforge.scripts.export_run_clouds \
    --hit-dir /tmp/run --out /tmp/series --cells 13 --exposed-mm 100 --held-mm 25
```

Writes `cloudN.ply` (the reconstructed outer surface), `particles/particlesN.ply` (particle
centres) and `params.json`, in the frame the scans and the slab-model clouds use: x in millimetres
from the chuck face toward the free end, bar axis at the origin.

**Compare surfaces to surfaces.** Particle centres sit up to one particle width inside the material,
so scoring centres against a scanned surface reads consistently small. `agforge/analysis/surface_recon.py`
documents the reconstruction and the stretch beyond which its gap-closing stops working.

### Reproducing the published point clouds

The published clouds were produced on a different line of development, so the settings do not map
one-to-one. Checked on press 1 of the Roll2 series, 13 cells, Tool2 mesh, 400 C card:

| | published | this branch | difference |
|---|---|---|---|
| free end | 100.627 mm | 100.627 mm | 0.000 mm |
| barrel radius, median over x = 10-60 mm | 19.043 mm | 19.043 mm | 0.001 mm |
| half-thickness at the press, z = 77.06 mm | 16.671 mm | 16.678 mm | 0.007 mm |

The files are **not** byte-identical: the 150,000 points are subsampled from a slightly different
vertex set. The geometry agrees to microns, which is three orders of magnitude inside the 0.79 mm
95th-percentile error against the scan, and inside the scanner's own 0.12-0.19 mm offset.

Settings used, which differ from the published run's in two ways worth knowing. `AGF_PIN_EXTRA_MM`
**adds** to the length here, so the exposed length is passed, not the total: the published run passed
a 125 mm length with 25 mm held, and the equivalent here is `--length-from-toolpath` on a file that
records 100 mm. And four knobs that were compiled in on the old line are defaults here, so they are
set explicitly:

```bash
  AGF_MATERIAL=6063_400C AGF_CELLS_PER_DIAMETER=13 AGF_CFL_SAFETY=0.45   AGF_MAX_FORCE=1e12 AGF_PIN_EXTRA_MM=25 AGF_CLAMP_DEPTH_MM=25   AGF_DIE_MESH=/path/to/Tool2.stl   AGF_ENABLE_CPIC=1 AGF_MPM_X_PAD_LOWER=0.85 AGF_APPROACH_CFL_RATIO=0.0205   python -u -m agforge.scripts.record_toolpath_hits     --jsonl roll2_toolpath.jsonl --n-hits 1 --length-from-toolpath     --z-shift-mm 25 --hit-dir out/roll2_p1
```

One press took **49 s** at 13 cells and 44,678 particles on an RTX 3060 Laptop, after about two
minutes of one-time scene construction and kernel compilation. The press stopped on Target Strain at
strain 0.1943 with a peak force near 56 kN, inside the 54.9-61.2 kN the published run reported.

On exit, a headless run raises `OpenGL.error.Error: Attempt to retrieve context when no valid
context` from an `atexit` handler in the bundled pyrender. It happens after the presses are finished
and written, and does not affect results.

## Materials

`AGF_MATERIAL` selects a card from `agforge/material_presets.py`. Unset is the shipped 316L, and
the 316L entry is an empty dict on purpose, so the declared field defaults remain the single source
of truth and an unset run is byte-identical to one from before presets existed.

| `AGF_MATERIAL` | card |
|---|---|
| unset / `316L` | hot steel, thermal on, as shipped |
| `6063T52` | cold 6063-T52 aluminium, isothermal, 137.9 + 173.6 eps^0.45 MPa |
| `6063_400C` | 6063 at 400 C, isothermal, perfectly plastic at 75 MPa |

`docs/6063_T52_MECHANICAL_PROPERTIES.md` gives the sources for the cold card and states its limits:
there is no measured T52 flow curve, `jc_n` is a typical aluminium exponent rather than a fitted
one, and the card should not be used past about 0.30 strain without one. The 400 C card is a
**literature estimate** from two published laws, not a measured property of this stock, and any
write-up has to say so. An unknown name raises rather than quietly falling back to steel.

## Contact names (tags are not modes)

`AGF_CONTACT_MODE` accepts only the short coupler names:

`grid` | `particle` | `fluidlab` | `penalty` | `none`

The six academic **tags** in `agforge/analysis/batch_arms.py` are not those mode names. Each tag
is a `(mode, teleport, refinements)` tuple:

| tag | `AGF_CONTACT_MODE` | teleport (`mech` / `AGF_PARTICLE_CONTACT`) |
|---|---|---|
| `grid` | `grid` | off |
| `grid_position_correction` | `grid` | on |
| `grid_fluidlab` | `fluidlab` | off |
| `grid_particle_sdf` | `particle` | off |
| `grid_penalty` | `penalty` | off |
| `no_contact` | `none` | off |

Two tags share mode `grid`. They are different arms. There is **no alias** that maps a long tag
onto a short mode; `AGF_CONTACT_MODE=grid_penalty` is an error (use `penalty`).

Teleport (`apply_particle_contact` mechanical projection) is in that academic set as
`grid_position_correction`. **It is off by default since 2026-09-16**, so a scene built with mode
`grid` and no other contact knobs is tag `grid`. Ask for the teleport with
`python -m agforge.analysis.batch_arms --arms grid_position_correction` (or
`AGF_PARTICLE_CONTACT_MECH=1` on a single-scene build). Die-to-billet heat transfer is the other,
independent job of the same pass and is NOT affected: it follows `AGF_PARTICLE_CONTACT_THERMAL`,
which still defaults on. Setting the old master `AGF_PARTICLE_CONTACT` still drives both jobs.

## Checking the build

Every module that reads a knob should import cleanly. This is a three-second check and it catches
the class of error where a knob is rewritten but its helper is not in scope:

```bash
python -c "import agforge.options, agforge.environment, agforge.strike_controller"
```

## Knobs

Configuration is read from `AGF_*` environment variables. In `options.py`, `environment.py` and
`strike_controller.py` these go through one typed helper, so a malformed value fails immediately and
names the variable responsible. Knobs read inside `genesis/engine/solvers/` are **not** yet routed
that way and still coerce directly.

`AGF_ROBOT_TIME_TO_SECONDS` must never be set to an empty string. Blank is rejected rather than
silently falling back to the derived value, because a blank pin during a CFL sweep would unpin the
controller without any indication.

Knobs added for recorded-toolpath replay:

| knob | default | what it does |
|---|---|---|
| `AGF_MATERIAL` | `316L` | selects a material card; see **Materials** |
| `AGF_PIN_EXTRA_MM` | `0` | stock held in the chuck, added on the pinned side |
| `AGF_CLAMP_DEPTH_MM` | unset | clamp grip as an absolute length instead of a fraction |
| `AGF_CLAMP_FRACTION` | `0.35` | clamp grip as a fraction of the billet |
| `AGF_DIE_MESH` | unset | tool mesh as the die contact surface instead of the box |
| `AGF_MAX_RIGID_SPEED` | `35.0` | bound on the implied jaw speed while repositioning |

`AGF_CLAMP_DEPTH_MM` and `AGF_CLAMP_FRACTION` are the same quantity in different units, so setting
both raises rather than resolving a precedence question. `AGF_DIE_MESH` pointing at a file that does
not exist also raises: falling back to the box would record in the log as a mesh run.

## What the defaults are

The defaults in `options.py` are the values the recorded results were produced with, taken from the
run provenance of the 2026-08-20 batches. Two exceptions, both deliberate:

- `AGF_MAX_FORCE` ships as a **runaway backstop**, not the value the measurements used. The
  measurement runs disabled the force stop entirely; a default should still halt a divergent run.
  It is set above every peak force ever observed, so recorded results reproduce unchanged.
- `AGF_FORCE_IMBALANCE_THRESHOLD` stays at its guard value. The measurement runs raised it to
  disable the speed-modulation throttle; that is a diagnostic lever, not a default. Lower
  `AGF_FORCE_BALANCE_GAIN` instead if you need a quieter controller.

Setting either to the measurement value reproduces the banked geometry more closely on the
elongation axis, at the cost of running without that guard.
