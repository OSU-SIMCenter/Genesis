"""Decompose the ~197 s per-run overhead before trying to remove it.

Measured across round 1: each run costs ~197 s of startup against ~76 s of actual
simulation -- about 70% of wall-clock. But "startup" is a bag containing pixi resolution,
python imports, scene construction, MJCF robot load, SDF build for the dies, particle
sampling, AND kernel compilation. Only the last of those is what a runtime-switchable
contact port would remove.

So this splits it, using the REAL adapter path (build_adapter -> init_stock -> apply_hit)
rather than a reimplementation, and reports where the time actually goes:

  import        `import genesis` and friends
  build         build_adapter + init_stock: scene, robot MJCF, die SDFs, particle sampling
  hit 1         first apply_hit -- includes JIT, since kernels compile on first launch
  hit 2, 3      steady-state hits, no compilation

JIT is then approximately (hit 1 - mean(hit 2, hit 3)). If that number is small, the
runtime-switchable port is not worth building and the overhead lives somewhere else.
"""
import os
import sys
import time

T0 = time.time()
STAMPS = [("process start", 0.0)]


def mark(label):
    STAMPS.append((label, time.time() - T0))


sys.path.insert(0, os.path.expanduser("~/GitHub/Genesis/forge_common/main"))

from forge_common.adapter_build import build_adapter          # noqa: E402
from forge_common.real_data import load_real_hits_for_sim     # noqa: E402
from forge_common.real_scale import (                          # noqa: E402
    REAL_STOCK_RADIUS_MM, REAL_STOCK_LENGTH_MM)
mark("imports (forge_common)")

import genesis  # noqa: E402,F401
mark("import genesis")

hits = load_real_hits_for_sim("genesis", 3)
adapter = build_adapter("genesis")
mark("build_adapter")

state = adapter.init_stock(radius_mm=REAL_STOCK_RADIUS_MM, length_mm=REAL_STOCK_LENGTH_MM)
mark("init_stock (scene+robot+SDF+sampling)")

for i, h in enumerate(hits, 1):
    state = adapter.apply_hit(state, h)
    mark("hit %d%s" % (i, "  <-- includes JIT" if i == 1 else ""))

print()
print("=" * 74)
print("STARTUP DECOMPOSITION")
print("=" * 74)
print("%-42s %10s %10s" % ("phase", "delta s", "cumul s"))
print("-" * 74)
prev = 0.0
deltas = {}
for label, t in STAMPS[1:]:
    print("%-42s %10.1f %10.1f" % (label, t - prev, t))
    deltas[label] = t - prev
    prev = t

h1 = deltas.get("hit 1  <-- includes JIT", 0.0)
steady = [v for k, v in deltas.items() if k.startswith("hit ") and "JIT" not in k]
mean_steady = sum(steady) / len(steady) if steady else 0.0
setup = sum(v for k, v in deltas.items() if not k.startswith("hit "))

print("-" * 74)
print("setup (imports + scene + SDF + sampling) : %6.1f s" % setup)
print("first hit                                : %6.1f s" % h1)
print("steady-state hit (mean of later hits)    : %6.1f s" % mean_steady)
print("=> JIT approx (first hit - steady)       : %6.1f s" % (h1 - mean_steady))
print()
print("A runtime-switchable contact port removes only the JIT line, and only for runs")
print("after the first in a shared process. If setup dominates, the fix is batching arms")
print("into one process (or a persistent kernel cache), NOT runtime-switchable modes.")
