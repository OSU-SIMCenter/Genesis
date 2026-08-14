#!/usr/bin/env python3
"""How much do ArrheniusPlasticity's input clamps actually bind?

docs/316L_MECHANICAL_PROPERTIES.md §8 argues the clamps in
`ArrheniusPlasticity._flow_stress_pa` (temperature, strain rate, plastic strain)
are the wrong default, and that the first thing to do is measure them: nothing
in the solver reports how often they saturate or what they cost in flow stress.
This script is that measurement. It does not change the clamps.

(1) Declared domains, read from agforge/materials.py — clamp vs Song2020 fit.
(2) Where the measured T4 envelope sits relative to each domain.
(3) Cost of each clamp wall: clamped vs unclamped-extrapolated sinh, other
    inputs held at measured medians.
(4) Derivative jump at each wall (finite difference; outside, the clamp is
    exactly flat).

No GPU, no scene, no genesis import. Numpy plus the standard library.
"""
from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
DEFAULT_MATERIALS = ROOT / "agforge" / "materials.py"
DEFAULT_TEMP = Path("/home/timothy/GitHub/Genesis/forge_common/main/outputs/t4_per_blow_temp.npz")
DEFAULT_BLOWS = Path("/home/timothy/GitHub/Genesis/forge_common/main/outputs/t4_press_blows.npz")


# ---------------------------------------------------------------------------
# Read the kernel source. Do not import agforge.materials (that pulls genesis).
# ---------------------------------------------------------------------------

def _literal_assigns(tree):
    out = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            t = node.targets[0]
            if isinstance(t, ast.Name):
                try:
                    out[t.id] = ast.literal_eval(node.value)
                except (ValueError, TypeError):
                    pass
    return out


def _class_field_defaults(tree, cls_name):
    out = {}
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == cls_name:
            for stmt in node.body:
                if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name) and stmt.value is not None:
                    try:
                        out[stmt.target.id] = ast.literal_eval(stmt.value)
                    except (ValueError, TypeError):
                        pass
    return out


def load_kernel_source(path):
    src = Path(path).read_text(encoding="utf-8")
    tree = ast.parse(src)
    consts = _literal_assigns(tree)
    fields = _class_field_defaults(tree, "ArrheniusPlasticity")

    missing = [k for k in ("_R_GAS", "_ARR_STRAIN0", "_ARR_DSTRAIN",
                           "_ARR_ALPHA", "_ARR_N", "_ARR_Q", "_ARR_A") if k not in consts]
    if missing:
        sys.exit("could not read %s from %s" % (", ".join(missing), path))
    for k in ("T_fit_min", "T_fit_max", "rate_min", "rate_max"):
        if k not in fields:
            sys.exit("could not read ArrheniusPlasticity.%s from %s" % (k, path))

    m_t = re.search(r"calibrated over\s+(\d+)\s*-\s*(\d+)\s*C", src)
    if not m_t:
        sys.exit("could not parse Song2020 temperature fit domain from %s" % path)
    t_fit_c = (float(m_t.group(1)), float(m_t.group(2)))

    m_r = re.search(r"fitted domain of\s+([0-9.eE+-]+)\s*\.\.\s*([0-9.eE+-]+)\s*/s", src)
    if not m_r:
        sys.exit("could not parse Song2020 rate fit domain from %s" % path)
    rate_fit = (float(m_r.group(1)), float(m_r.group(2)))

    m_e = re.search(r"clamps plastic_strain into \[([0-9.]+),\s*([0-9.]+)\]", src)
    if not m_e:
        sys.exit("could not parse strain clamp comment from %s" % path)
    strain_comment = (float(m_e.group(1)), float(m_e.group(2)))

    m_n = re.search(r"for k in qd\.static\(range\((\d+)\)\)", src)
    n_loop = int(m_n.group(1)) if m_n else None

    m_old = re.search(r"Raised\s+([0-9.]+)\s*->\s*([0-9.]+)", src)
    former_t_max = float(m_old.group(1)) if m_old else None

    m_pt = re.search(
        r"([0-9.]+)\s*MPa at eps\s*([0-9.]+),\s*([0-9.]+)\s*/s", src)
    comment_mpa = comment_eps = comment_rate = None
    if m_pt:
        comment_mpa = float(m_pt.group(1))
        comment_eps = float(m_pt.group(2))
        comment_rate = float(m_pt.group(3))
    m_ex = re.search(r"extrapolating Song's own form gives\s+([0-9.]+)\s*MPa", src)
    comment_extrap_mpa = float(m_ex.group(1)) if m_ex else None

    m_sim = re.search(r"nominal rate of v/D\s*=\s*([0-9.]+)\s*/s", src)
    sim_rate = float(m_sim.group(1)) if m_sim else None

    n_nodes = len(consts["_ARR_ALPHA"])
    strain0 = float(consts["_ARR_STRAIN0"])
    dstrain = float(consts["_ARR_DSTRAIN"])
    strain_from_grid = (strain0, strain0 + (n_nodes - 1) * dstrain)

    return {
        "path": str(path),
        "R": float(consts["_R_GAS"]),
        "strain0": strain0,
        "dstrain": dstrain,
        "n_nodes": n_nodes,
        "n_loop": n_loop,
        "alpha": np.asarray(consts["_ARR_ALPHA"], dtype=float),
        "n_exp": np.asarray(consts["_ARR_N"], dtype=float),
        "Q": np.asarray(consts["_ARR_Q"], dtype=float),
        "A": np.asarray(consts["_ARR_A"], dtype=float),
        "T_fit_min": float(fields["T_fit_min"]),
        "T_fit_max": float(fields["T_fit_max"]),
        "rate_min": float(fields["rate_min"]),
        "rate_max": float(fields["rate_max"]),
        "t_fit_c": t_fit_c,
        "t_fit_k": (t_fit_c[0] + 273.15, t_fit_c[1] + 273.15),
        "rate_fit": rate_fit,
        "strain_comment": strain_comment,
        "strain_grid": strain_from_grid,
        "former_t_max": former_t_max,
        "comment_mpa": comment_mpa,
        "comment_eps": comment_eps,
        "comment_rate": comment_rate,
        "comment_extrap_mpa": comment_extrap_mpa,
        "sim_rate": sim_rate,
    }


# ---------------------------------------------------------------------------
# Sinh evaluation. Matches _flow_stress_pa: hat-function sum on a uniform
# strain grid, which is numpy.interp on that grid. Clamps can be bypassed
# independently so the unclamped form is the same function.
# ---------------------------------------------------------------------------

def _nodal_sigma_mpa(rate, temp, k):
    z = rate * np.exp(k["Q"] / (k["R"] * temp))
    x = (z / k["A"]) ** (1.0 / k["n_exp"])
    return np.arcsinh(x) / k["alpha"]


def _interp_extrap(x, xp, fp):
    x = float(x)
    if x < xp[0]:
        return float(fp[0] + (fp[1] - fp[0]) / (xp[1] - xp[0]) * (x - xp[0]))
    if x > xp[-1]:
        return float(fp[-1] + (fp[-1] - fp[-2]) / (xp[-1] - xp[-2]) * (x - xp[-1]))
    return float(np.interp(x, xp, fp))


def flow_stress_mpa(eps_p, rate, temp, k, clamp_temp=True, clamp_rate=True, clamp_strain=True):
    """Arrhenius flow stress in MPa. Returns nan if the unclamped form overflows."""
    tk = float(np.clip(temp, k["T_fit_min"], k["T_fit_max"])) if clamp_temp else float(temp)
    r = float(np.clip(rate, k["rate_min"], k["rate_max"])) if clamp_rate else float(rate)
    if tk <= 0.0 or r <= 0.0:
        return float("nan")
    nodes = _nodal_sigma_mpa(r, tk, k)
    if not np.all(np.isfinite(nodes)):
        return float("nan")
    strains = k["strain0"] + k["dstrain"] * np.arange(k["n_nodes"], dtype=float)
    if clamp_strain:
        e = float(np.clip(eps_p, strains[0], strains[-1]))
        return float(np.interp(e, strains, nodes))
    return _interp_extrap(eps_p, strains, nodes)


def pct(a, p):
    return float(np.percentile(a, p))


def summarize(name, a, unit):
    print("%-22s %10.4f %10.4f %10.4f %10.4f %10.4f  %s"
          % (name, a.min(), pct(a, 25), pct(a, 50), pct(a, 75), a.max(), unit))


def count_in(a, lo, hi):
    n = int(((a >= lo) & (a <= hi)).sum())
    return n, len(a) - n, len(a)


def print_cost_table(title, xs, labels, eval_clamped, eval_free):
    print("\n" + title)
    print("%-22s %14s %16s %8s" % ("input", "clamped MPa", "unclamped MPa", "ratio"))
    for x, lab in zip(xs, labels):
        sc = eval_clamped(x)
        su = eval_free(x)
        if np.isfinite(sc) and np.isfinite(su) and su != 0.0:
            ratio = sc / su
            rs = "%8.3f" % ratio
        else:
            ratio = float("nan")
            rs = "%8s" % "n/a"
        tag = "  " + lab if lab else ""
        print("%-22s %14.4f %16s %s%s"
              % (_fmt(x), sc,
                 ("%.4f" % su) if np.isfinite(su) else "overflow",
                 rs, tag))


def _fmt(x):
    ax = abs(x)
    if ax != 0.0 and (ax < 1e-2 or ax >= 1e4):
        return "%.4e" % x
    return "%.4f" % x


def wall_derivatives(name, wall, inside_dir, h, f_clamped, f_free):
    """inside_dir = +1 if the in-domain side is wall+h (a min wall), -1 for a max wall."""
    x_in = wall + inside_dir * h
    x_out = wall - inside_dir * h
    s_wall = f_clamped(wall)
    s_in = f_clamped(x_in)
    s_out = f_clamped(x_out)
    d_in = (s_in - s_wall) / (inside_dir * h)
    d_out = (s_out - s_wall) / (-inside_dir * h)
    d_free_out = (f_free(x_out) - f_free(wall)) / (-inside_dir * h)
    jump = d_in - d_out
    return {
        "name": name,
        "wall": wall,
        "d_in": d_in,
        "d_out": d_out,
        "d_free_out": d_free_out,
        "jump": jump,
        "s_wall": s_wall,
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--materials", default=str(DEFAULT_MATERIALS),
                   help="ArrheniusPlasticity source (default: agforge/materials.py)")
    p.add_argument("--temp-npz", default=str(DEFAULT_TEMP),
                   help="per-blow temperature table")
    p.add_argument("--blows-npz", default=str(DEFAULT_BLOWS),
                   help="per-blow press positions / durations")
    args = p.parse_args(argv)

    k = load_kernel_source(args.materials)
    if k["n_loop"] is not None and k["n_loop"] != k["n_nodes"]:
        print("WARNING: hat-function loop is range(%d) but %d tabulated nodes"
              % (k["n_loop"], k["n_nodes"]))

    # --- measured envelope ------------------------------------------------
    blows = np.load(args.blows_npz, allow_pickle=True)
    h0 = blows["pos_start"].astype(float)
    h1 = blows["pos_min"].astype(float)
    dur = blows["dur"].astype(float)
    ok = (h0 > 0) & (h1 > 0) & (h1 < h0) & (dur > 0)
    n_drop = int((~ok).sum())
    h0, h1, dur = h0[ok], h1[ok], dur[ok]
    rate = np.log(h0 / h1) / dur
    # Per-blow true compressive strain. Accumulated plastic strain Jp is not
    # in either npz; this is the only strain derivable from the same traces.
    eps_blow = np.log(h0 / h1)

    temp_npz = np.load(args.temp_npz, allow_pickle=True)
    cols = [str(c) for c in temp_npz["columns"]]
    table = temp_npz["table"]
    if "T_pre_p50" not in cols:
        sys.exit("t4_per_blow_temp.npz has no T_pre_p50 column; columns=%s" % cols)
    t_c = table[:, cols.index("T_pre_p50")].astype(float)
    present = np.ones(len(t_c), dtype=bool)
    if "present" in cols:
        present = table[:, cols.index("present")] > 0
    t_ok = np.isfinite(t_c) & present
    t_c = t_c[t_ok]
    t_k = t_c + 273.15

    t_med = pct(t_k, 50)
    r_med = pct(rate, 50)
    e_med = pct(eps_blow, 50)

    print("clamp_probe — constitutive-domain diagnostic")
    print("kernel source: %s" % k["path"])
    print("temp npz:      %s" % args.temp_npz)
    print("blows npz:     %s" % args.blows_npz)
    if n_drop:
        print("dropped %d of %d blows (non-closing or zero duration)" % (n_drop, n_drop + len(rate)))
    print()

    # ===================================================================
    # (1) Declared domains
    # ===================================================================
    print("=" * 78)
    print("(1) DECLARED DOMAINS  (read from %s, not hardcoded)" % Path(k["path"]).name)
    print("=" * 78)
    print("T_fit_min = %.2f K" % k["T_fit_min"])
    print("T_fit_max = %.2f K" % k["T_fit_max"])
    print("rate_min  = %.4g /s" % k["rate_min"])
    print("rate_max  = %.4g /s" % k["rate_max"])
    print("strain clamp (comment) = [%.2f, %.2f]" % k["strain_comment"])
    print("strain clamp (grid)    = [%.2f, %.2f]  (%d nodes, spacing %.2f)"
          % (k["strain_grid"][0], k["strain_grid"][1], k["n_nodes"], k["dstrain"]))
    print()
    print("%-16s %-32s %s" % ("axis", "clamp (what the kernel uses)", "Song2020 fit (comments)"))
    print("%-16s %-32s %s"
          % ("temperature",
             "[%.2f, %.2f] K" % (k["T_fit_min"], k["T_fit_max"]),
             "[%.0f, %.0f] C = [%.2f, %.2f] K" % (k["t_fit_c"][0], k["t_fit_c"][1],
                                                  k["t_fit_k"][0], k["t_fit_k"][1])))
    print("%-16s %-32s %s"
          % ("strain rate",
             "[%.4g, %.4g] /s" % (k["rate_min"], k["rate_max"]),
             "[%.4g, %.4g] /s" % (k["rate_fit"][0], k["rate_fit"][1])))
    print("%-16s %-32s %s"
          % ("plastic strain",
             "[%.2f, %.2f]" % k["strain_grid"],
             "tabulated at %d nodes, %.2f spacing (table IS the fit)"
             % (k["n_nodes"], k["dstrain"])))
    print()
    print("gap: T clamp extends %.0f K above the fit ceiling; rate clamp ceiling is %.0e /s"
          % (k["T_fit_max"] - k["t_fit_k"][1], k["rate_max"]))
    print("     vs a fitted maximum of %.0e /s (%.0f orders of magnitude)."
          % (k["rate_fit"][1], round(np.log10(k["rate_max"] / k["rate_fit"][1]))))
    print("     strain clamp coincides with the table ends.")

    # ===================================================================
    # (2) Operating envelope
    # ===================================================================
    print()
    print("=" * 78)
    print("(2) OPERATING ENVELOPE vs EACH DOMAIN")
    print("=" * 78)
    print("n = %d blows  (temperature: T_pre_p50, %d present and finite; rate: ln(h0/h1)/dur)"
          % (len(rate), len(t_k)))
    print("%-22s %10s %10s %10s %10s %10s"
          % ("", "min", "p25", "median", "p75", "max"))
    summarize("T_pre_p50", t_c, "C")
    summarize("T_pre_p50", t_k, "K")
    summarize("TRUE rate", rate, "/s")
    summarize("per-blow true strain", eps_blow, "ln(h0/h1); not accumulated Jp")

    n_t_fit, n_t_fit_out, n_t = count_in(t_k, k["t_fit_k"][0], k["t_fit_k"][1])
    n_t_cl, n_t_cl_out, _ = count_in(t_k, k["T_fit_min"], k["T_fit_max"])
    n_t_below = int((t_k < k["T_fit_min"]).sum())
    n_t_above_fit = int((t_k > k["t_fit_k"][1]).sum())
    n_t_above_cl = int((t_k > k["T_fit_max"]).sum())

    n_r_fit, n_r_fit_out, n_r = count_in(rate, k["rate_fit"][0], k["rate_fit"][1])
    n_r_cl, n_r_cl_out, _ = count_in(rate, k["rate_min"], k["rate_max"])

    n_e_cl, n_e_cl_out, n_e = count_in(eps_blow, k["strain_grid"][0], k["strain_grid"][1])

    print()
    print("temperature: %d of %d blows inside Song2020 fit [%.2f, %.2f] K; %d outside"
          % (n_t_fit, n_t, k["t_fit_k"][0], k["t_fit_k"][1], n_t_fit_out))
    print("             %d of %d blows inside clamp     [%.2f, %.2f] K; %d outside"
          % (n_t_cl, n_t, k["T_fit_min"], k["T_fit_max"], n_t_cl_out))
    print("             outside split: %d below T_fit_min, %d above Song ceiling, %d above T_fit_max"
          % (n_t_below, n_t_above_fit, n_t_above_cl))
    print("strain rate: %d of %d blows inside Song2020 fit [%.4g, %.4g] /s; %d outside"
          % (n_r_fit, n_r, k["rate_fit"][0], k["rate_fit"][1], n_r_fit_out))
    print("             %d of %d blows inside clamp     [%.4g, %.4g] /s; %d outside"
          % (n_r_cl, n_r, k["rate_min"], k["rate_max"], n_r_cl_out))
    print("median measured true strain rate = %.2g /s   (two significant figures; exact %.4f /s)"
          % (r_med, r_med))
    print("per-blow true strain: %d of %d inside strain clamp [%.2f, %.2f]; %d outside"
          % (n_e_cl, n_e, k["strain_grid"][0], k["strain_grid"][1], n_e_cl_out))
    print("NOTE: accumulated plastic strain Jp is not in the npz files. The strain")
    print("      numbers above are per-blow ln(pos_start/pos_min) only.")

    # ===================================================================
    # (3) Cost of each clamp wall
    # ===================================================================
    print()
    print("=" * 78)
    print("(3) COST OF EACH CLAMP WALL")
    print("=" * 78)
    print("held at measured medians: T = %.2f K (%.1f C), rate = %.4f /s, eps = %.4f"
          % (t_med, t_med - 273.15, r_med, e_med))
    print("(eps = median per-blow true strain; see note above)")
    print("ratio = sigma_clamped / sigma_unclamped.  1.0 means the clamp did not bind.")

    t_points = sorted(set([
        float(t_k.min()),
        k["T_fit_min"] - 1.0,
        k["T_fit_min"],
        k["T_fit_min"] + 1.0,
        t_med,
        float(t_k.max()),
        k["t_fit_k"][1],
        k["T_fit_max"] - 1.0,
        k["T_fit_max"],
        k["T_fit_max"] + 1.0,
        1423.15,
        1533.15,
        1675.0,
    ]))
    t_labels = []
    for x in t_points:
        lab = []
        if abs(x - t_k.min()) < 1e-9:
            lab.append("measured min")
        if abs(x - t_med) < 1e-9:
            lab.append("measured median")
        if abs(x - t_k.max()) < 1e-9:
            lab.append("measured max")
        if abs(x - k["T_fit_min"]) < 1e-9:
            lab.append("T_fit_min wall")
        if abs(x - k["T_fit_max"]) < 1e-9:
            lab.append("T_fit_max wall")
        if abs(x - k["t_fit_k"][1]) < 1e-9:
            lab.append("Song 1000 C ceiling")
        if abs(x - 1423.15) < 1e-9:
            lab.append("1150 C forging")
        if abs(x - 1533.15) < 1e-9:
            lab.append("1260 C forging")
        if abs(x - 1675.0) < 1e-9:
            lab.append("solidus")
        t_labels.append(", ".join(lab))

    print_cost_table(
        "temperature  [current clamp T_fit_min..T_fit_max; rate and eps at medians]",
        t_points, t_labels,
        lambda T: flow_stress_mpa(e_med, r_med, T, k, True, True, True),
        lambda T: flow_stress_mpa(e_med, r_med, T, k, False, True, True),
    )

    # Documented 2.3x: former ceiling 1273.15 vs extrapolating, at the comment's
    # (eps, rate), not at session medians. Current T_fit_max is already 1473.15,
    # so this is the historical wall the comment records, not today's kernel.
    if k["former_t_max"] is not None and k["comment_eps"] is not None and k["comment_rate"] is not None:
        e_c, r_c = k["comment_eps"], k["comment_rate"]
        t_hist = sorted(set([
            k["former_t_max"],
            k["T_fit_max"],
            1423.15,
            1533.15,
        ]))

        def _clamp_former(T):
            tk = float(np.clip(T, k["T_fit_min"], k["former_t_max"]))
            return flow_stress_mpa(e_c, r_c, tk, k, False, False, True)

        def _free_hist(T):
            return flow_stress_mpa(e_c, r_c, T, k, False, False, True)

        hist_labels = []
        for x in t_hist:
            lab = []
            if abs(x - k["former_t_max"]) < 1e-9:
                lab.append("former T_fit_max (1000 C)")
            if abs(x - k["T_fit_max"]) < 1e-9:
                lab.append("current T_fit_max / 1200 C")
            if abs(x - 1423.15) < 1e-9:
                lab.append("1150 C")
            if abs(x - 1533.15) < 1e-9:
                lab.append("1260 C")
            hist_labels.append(", ".join(lab))
        print_cost_table(
            "temperature  [FORMER wall at %.2f K as in the T_fit_max comment; "
            "eps=%.3f, rate=%.2f /s from that comment]"
            % (k["former_t_max"], e_c, r_c),
            t_hist, hist_labels, _clamp_former, _free_hist,
        )
        s_old = _clamp_former(k["T_fit_max"])
        s_new = _free_hist(k["T_fit_max"])
        ratio = s_old / s_new if s_new else float("nan")
        print("  reproduction: clamp@%.2f K vs extrapolate@%.2f K = %.2f / %.2f MPa = %.2fx"
              % (k["former_t_max"], k["T_fit_max"], s_old, s_new, ratio))
        if k["comment_mpa"] is not None and k["comment_extrap_mpa"] is not None:
            print("  comment quoted %.1f vs %.1f MPa (%.2fx). this run: %.1f vs %.1f MPa."
                  % (k["comment_mpa"], k["comment_extrap_mpa"],
                     k["comment_mpa"] / k["comment_extrap_mpa"], s_old, s_new))

    r_points = []
    r_labels_map = {}

    def _radd(x, lab):
        if x is None or not np.isfinite(x) or x <= 0:
            return
        r_points.append(float(x))
        r_labels_map.setdefault(float(x), []).append(lab)

    _radd(k["rate_min"] * 0.5, "below rate_min")
    _radd(k["rate_min"], "rate_min wall")
    _radd(k["rate_min"] * 2, "just inside rate_min")
    _radd(k["rate_fit"][0], "Song fit floor")
    _radd(k["rate_fit"][1], "Song fit ceiling")
    _radd(float(rate.min()), "measured min")
    _radd(r_med, "measured median")
    _radd(float(rate.max()), "measured max")
    _radd(k["comment_rate"], "comment 0.41 /s")
    _radd(k["sim_rate"], "sim v/D (still under rate_max)")
    _radd(k["rate_max"] * 0.5, "just inside rate_max")
    _radd(k["rate_max"], "rate_max wall")
    _radd(k["rate_max"] * 2, "beyond rate_max")
    r_xs = sorted(set(r_points))
    r_labs = [", ".join(r_labels_map.get(x, [])) for x in r_xs]

    print_cost_table(
        "strain rate  [current clamp rate_min..rate_max; T and eps at medians]",
        r_xs, r_labs,
        lambda r: flow_stress_mpa(e_med, r, t_med, k, True, True, True),
        lambda r: flow_stress_mpa(e_med, r, t_med, k, True, False, True),
    )

    e_points = []
    e_map = {}

    def _eadd(x, lab):
        e_points.append(float(x))
        e_map.setdefault(float(x), []).append(lab)

    _eadd(0.0, "below table")
    _eadd(k["strain_grid"][0], "strain min wall")
    _eadd(k["comment_eps"] if k["comment_eps"] is not None else 0.207, "comment eps")
    _eadd(e_med, "median per-blow true strain")
    _eadd(k["strain_grid"][1], "strain max wall")
    _eadd(k["strain_grid"][1] + k["dstrain"], "one node past table")
    _eadd(float(eps_blow.max()), "measured max per-blow strain")
    e_xs = sorted(set(e_points))
    e_labs = [", ".join(e_map.get(x, [])) for x in e_xs]
    print()
    print("strain unclamped = linear extrapolation of the 9 nodal stresses.")
    print("The sinh constants are only tabulated; the hat-function sum is zero")
    print("outside t in [0, 8], so bypassing that clamp without extrapolation")
    print("would return 0, which is not a physical extrapolation of the form.")
    print_cost_table(
        "plastic strain  [current table-end clamp; T and rate at medians]",
        e_xs, e_labs,
        lambda e: flow_stress_mpa(e, r_med, t_med, k, True, True, True),
        lambda e: flow_stress_mpa(e, r_med, t_med, k, True, True, False),
    )

    # ===================================================================
    # (4) Derivative discontinuity
    # ===================================================================
    print()
    print("=" * 78)
    print("(4) DERIVATIVE DISCONTINUITY AT EACH WALL")
    print("=" * 78)
    print("finite difference of the CLAMPED function. inside: one-sided toward")
    print("the open interval; outside: one-sided into the clamp (must be ~0).")
    print("held at the same measured medians as (3).")
    print()

    rows = [
        wall_derivatives("T_fit_min", k["T_fit_min"], +1, 0.05,
                         lambda T: flow_stress_mpa(e_med, r_med, T, k, True, True, True),
                         lambda T: flow_stress_mpa(e_med, r_med, T, k, False, True, True)),
        wall_derivatives("T_fit_max", k["T_fit_max"], -1, 0.05,
                         lambda T: flow_stress_mpa(e_med, r_med, T, k, True, True, True),
                         lambda T: flow_stress_mpa(e_med, r_med, T, k, False, True, True)),
        wall_derivatives("rate_min", k["rate_min"], +1, k["rate_min"] * 1e-3,
                         lambda r: flow_stress_mpa(e_med, r, t_med, k, True, True, True),
                         lambda r: flow_stress_mpa(e_med, r, t_med, k, True, False, True)),
        wall_derivatives("rate_max", k["rate_max"], -1, k["rate_max"] * 1e-3,
                         lambda r: flow_stress_mpa(e_med, r, t_med, k, True, True, True),
                         lambda r: flow_stress_mpa(e_med, r, t_med, k, True, False, True)),
        wall_derivatives("strain_min", k["strain_grid"][0], +1, 1e-4,
                         lambda e: flow_stress_mpa(e, r_med, t_med, k, True, True, True),
                         lambda e: flow_stress_mpa(e, r_med, t_med, k, True, True, False)),
        wall_derivatives("strain_max", k["strain_grid"][1], -1, 1e-4,
                         lambda e: flow_stress_mpa(e, r_med, t_med, k, True, True, True),
                         lambda e: flow_stress_mpa(e, r_med, t_med, k, True, True, False)),
    ]
    # former T wall, at the comment's (eps, rate), for the 2.3x kink
    if k["former_t_max"] is not None and k["comment_eps"] is not None:
        e_c, r_c = k["comment_eps"], k["comment_rate"]

        def _cf(T):
            tk = float(np.clip(T, k["T_fit_min"], k["former_t_max"]))
            return flow_stress_mpa(e_c, r_c, tk, k, False, False, True)

        rows.append(wall_derivatives(
            "former T_fit_max", k["former_t_max"], -1, 0.05,
            _cf, lambda T: flow_stress_mpa(e_c, r_c, T, k, False, False, True)))

    print("%-18s %12s %14s %14s %14s %s"
          % ("wall", "wall value", "d(in)", "d(out) clamp", "jump in-out", "unit"))
    units = {
        "T_fit_min": "MPa/K",
        "T_fit_max": "MPa/K",
        "former T_fit_max": "MPa/K",
        "rate_min": "MPa/(1/s)",
        "rate_max": "MPa/(1/s)",
        "strain_min": "MPa/strain",
        "strain_max": "MPa/strain",
    }
    for row in rows:
        print("%-18s %12s %14.4f %14.4e %14.4f %s"
              % (row["name"], _fmt(row["wall"]), row["d_in"], row["d_out"],
                 row["jump"], units[row["name"]]))
        print("%-18s %12s unclamped d(out) = %.4f   (what the form would do past the wall)"
              % ("", "", row["d_free_out"]))

    print()
    print("inside each wall d(sigma)/d(input) is nonzero; the clamp forces it to")
    print("zero outside. The jump equals the interior derivative. That is C0 but")
    print("not C1 — the localisation seed §8.2 warns about.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
