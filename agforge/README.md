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
