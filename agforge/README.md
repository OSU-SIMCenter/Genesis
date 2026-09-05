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
  AGF_BILLET_MESH=~/GitHub/Genesis/forge_common/main/outputs/real_meshes/billet_hit01_before_d8000.obj \
  python -m agforge.analysis.batch_arms --n-hits 17
'
```

`AGF_BILLET_MESH` is the only variable you must set. It is real-forge geometry produced by
`agforge/analysis/extract_real_meshes.py` and is **not stored in this repository** — it is a
generated artifact living in the `forge_common` outputs tree. Without it the simulation runs on a
built-in nominal cylinder and will not match recorded results.

Every other default now matches the configuration the recorded results were produced with, so a run
with nothing else set is the reproduction case. Setting any other `AGF_*` variable changes that.

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
