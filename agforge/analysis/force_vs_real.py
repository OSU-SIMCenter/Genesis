"""First-ever comparison of simulated forging force against the real machine's F.

The real dataset's `F` is in ARBITRARY MACHINE UNITS (forge_common/hit.py says so, and no
adapter reads it), so absolute magnitude is not comparable without a calibration nobody has
done. What IS comparable, unit-free, is the SHAPE of the sequence: real F varies 46.7-93.9
across the 17 hits, a factor of 2.0. If the sim is responding correctly to hit-to-hit
variation in commanded reduction, its per-hit force should track that pattern.

So this reports:
  - Pearson and Spearman correlation of sim pressing-force vs real F across hits
  - the implied unit scale (sim kN per real F unit) and how tightly it holds
  - a per-hit table

It also tests workstream B's explicit prediction. Their 316L work found the current material
card is ~1.9x too soft at 1/s, and predicted: "if the material is ~2x too soft, simulated
forging force on the 17-hit sequence should be LOW, not high." That is a discriminating
check, and the force tap now makes it answerable -- but ONLY in relative terms unless the
unit scale can be pinned independently.

Force metric is pressing-MEAN, not peak: the no-contact control showed peak reads ~128 kN
with nothing touching the bar (gripper actuation during approach).
"""
import json
import os
import pathlib

import numpy as np
import torch

OUT = pathlib.Path.home() / "GitHub/Genesis/forge_common/main/outputs"
PT = (pathlib.Path.home()
      / "GitHub/Genesis/forge_common/models/forge-net/forge_net/data/datasets/agf_data/2026-06-29.pt")


def sim_force(tag):
    """Per-strike (L+R) mean pressing force, kN, indexed by hit."""
    p = OUT / (tag + ".diag.jsonl")
    if not p.exists():
        return {}
    out = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        fl, fr = r.get("force_L_press_mean"), r.get("force_R_press_mean")
        if fl is not None and fr is not None:
            out[r["strike"]] = (fl + fr) / 1000.0
    return out


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    real = torch.load(PT, map_location="cpu", weights_only=False)["F"].numpy()
    arms = [("g1_grid+teleport", ["velo_mx_g1_grid_prod_r%d" % i for i in (1, 2, 3)]),
            ("g0_grid_alone", ["velo_mx_g0_grid_only_r%d" % i for i in (1, 2, 3)]),
            ("h1_grid+cinj", ["velo_mx_h1_grid_cinj_r%d" % i for i in (1, 2, 3)])]

    print("=" * 86)
    print("SIM FORCE vs REAL MACHINE F  (real F is in arbitrary units -- shape, not magnitude)")
    print("=" * 86)
    print("real F : min %.1f  max %.1f  mean %.1f  (spread %.2fx)"
          % (real.min(), real.max(), real.mean(), real.max() / real.min()))
    print()
    print("%-20s %6s %10s %10s %14s %16s" % (
        "arm", "hits", "pearson", "spearman", "mean kN", "kN per F unit"))
    print("-" * 86)

    series = {}
    for name, tags in arms:
        per = [sim_force(t) for t in tags]
        common = sorted(set.intersection(*[set(d) for d in per if d]) if any(per) else [])
        if not common:
            print("%-20s   no diag" % name)
            continue
        f = np.array([np.mean([d[h] for d in per if h in d]) for h in common])
        r = real[[h - 1 for h in common]]
        series[name] = (common, f, r)
        pear = float(np.corrcoef(f, r)[0, 1])
        scale = f / r
        print("%-20s %6d %10.3f %10.3f %14.1f %16s" % (
            name, len(common), pear, spearman(f, r), f.mean(),
            "%.2f+-%.2f" % (scale.mean(), scale.std())))

    name = "g0_grid_alone" if "g0_grid_alone" in series else next(iter(series))
    common, f, r = series[name]
    print()
    print("per-hit detail (%s):" % name)
    print("%5s %12s %12s %12s" % ("hit", "real F", "sim kN", "kN / F"))
    for h, fs, rr in zip(common, f, r):
        print("%5d %12.1f %12.1f %12.2f" % (h, rr, fs, fs / rr))

    print()
    print("READ WITH CARE:")
    print("  * A high correlation means the sim TRACKS hit-to-hit variation. It says nothing")
    print("    about whether the absolute level is right -- that needs the unit calibration.")
    print("  * A tight 'kN per F unit' across hits would itself be evidence the two are")
    print("    proportional, i.e. that a single scale factor is the whole story. A loose one")
    print("    means the sim's error is hit-dependent and no single factor will fix it.")
    print("  * Workstream B predicts the current card is ~1.9x too soft => sim force LOW.")
    print("    That can only be checked once one real F unit is pinned to newtons.")


if __name__ == "__main__":
    main()
