#!/usr/bin/env python3
"""Thermal coefficient calibration CLI.

Modes
-----
analyze   Print geometry, timing, literature defaults, and heating/cooling estimates.
predict   Same as analyze (analytical energy balance only, no GPU).
fit       Recommend q_peak and L_eff (analytical; optional --gpu grid search).
verify    Run headless idle-heating benchmark against targets (requires GPU).

Examples
--------
    pixi run python -m agforge.scripts.thermal_calibrate analyze
    pixi run python -m agforge.scripts.thermal_calibrate fit --output calibration.json
    pixi run python -m agforge.scripts.thermal_calibrate fit --gpu
    pixi run python -m agforge.scripts.thermal_calibrate verify
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from agforge.thermal_calibration import (
    CalibrationTargets,
    ThermalCalibrationConfig,
    analyze,
    apply_fit_to_options_dict,
    fit_coefficients_analytical,
    format_analysis_report,
    load_config_from_teleop,
    save_calibration_json,
)


def _parse_targets(args: argparse.Namespace) -> CalibrationTargets:
    return CalibrationTargets(
        initial_temp_k=args.initial_temp,
        target_surface_temp_k=args.target_temp,
        target_heating_time_thermal_s=args.target_time,
        target_held_end_temp_k=args.held_end_max,
        ambient_temp_k=args.ambient,
    )


def _parse_config(args: argparse.Namespace) -> ThermalCalibrationConfig:
    cfg = load_config_from_teleop()
    if args.q_peak is not None:
        cfg.q_peak_w_m3 = args.q_peak
    if args.l_eff is not None:
        cfg.fixed_end_conduction_length_m = args.l_eff
    if args.h_air is not None:
        cfg.h_air_w_m2k = args.h_air
    if args.h_contact is not None:
        cfg.h_contact_w_m2k = args.h_contact
    return cfg


def cmd_analyze(args: argparse.Namespace) -> int:
    cfg = _parse_config(args)
    targets = _parse_targets(args)
    report = analyze(cfg, targets)
    print(format_analysis_report(report))
    if args.output:
        from agforge.thermal_calibration import FitResult

        dummy_fit = FitResult(
            q_peak_w_m3=cfg.q_peak_w_m3,
            fixed_end_conduction_length_m=cfg.fixed_end_conduction_length_m,
            h_contact_w_m2k=cfg.h_contact_w_m2k,
            score=0.0,
            metrics={},
            method="analyze-only",
        )
        save_calibration_json(args.output, dummy_fit, report)
        print(f"\nWrote {args.output}")
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    return cmd_analyze(args)


def cmd_fit(args: argparse.Namespace) -> int:
    cfg = _parse_config(args)
    targets = _parse_targets(args)

    if args.gpu:
        from agforge.thermal_benchmark_harness import fit_coefficients_simulation

        print("Running GPU grid search (this may take a few minutes)...")
        best_q, best_l, trials = fit_coefficients_simulation(cfg, targets)
        from agforge.thermal_calibration import FitResult

        fit = FitResult(
            q_peak_w_m3=best_q,
            fixed_end_conduction_length_m=best_l,
            h_contact_w_m2k=cfg.h_contact_w_m2k,
            score=0.0,
            metrics={"n_trials": len(trials)},
            method="simulation_grid_search",
        )
        print(f"GPU fit: q_peak={best_q:.3e} W/m³, L_eff={best_l*1e3:.1f} mm")
        passed = sum(1 for t in trials if t.success)
        print(f"  Trials: {len(trials)}, successes: {passed}")
    else:
        fit = fit_coefficients_analytical(cfg, targets)
        print("Analytical fit (fast, no GPU):")
        print(f"  q_peak = {fit.q_peak_w_m3:.3e} W/m³")
        print(f"  L_eff  = {fit.fixed_end_conduction_length_m*1e3:.1f} mm")
        print(f"  h_contact = {fit.h_contact_w_m2k:.0f} W/m²K (unchanged)")
        for k, v in fit.metrics.items():
            print(f"  {k}: {v}")

    final_cfg = ThermalCalibrationConfig(
        **{
            **asdict(cfg),
            "q_peak_w_m3": fit.q_peak_w_m3,
            "fixed_end_conduction_length_m": fit.fixed_end_conduction_length_m,
        }
    )
    report = analyze(final_cfg, targets)
    print()
    print(format_analysis_report(report))

    patch = apply_fit_to_options_dict(fit)
    print("\n--- Suggested option overrides ---")
    print(json.dumps(patch, indent=2))

    if args.output:
        save_calibration_json(args.output, fit, report)
        print(f"\nWrote {args.output}")

    if args.apply:
        _write_options_patch(args.apply, patch)
        print(f"Wrote patch snippet to {args.apply}")

    return 0


def _write_options_patch(path: str, patch: dict) -> None:
    lines = [
        "# Thermal calibration patch — merge into agforge/options.py or teleop config",
        f"heating_power = {patch['heating_power']:.6e}  # W/m³",
        f"fixed_end_conduction_length = {patch['mpm']['fixed_end_conduction_length']:.6f}  # m",
        f"thermal_contact_conductivity = {patch['mpm']['thermal_contact_conductivity']:.1f}  # W/m²K",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def cmd_verify(args: argparse.Namespace) -> int:
    from agforge.thermal_benchmark_harness import (
        run_air_cooling_benchmark,
        run_fixed_end_gradient_benchmark,
        run_idle_heating_verify,
        run_induction_heating_benchmark,
    )

    cfg = _parse_config(args)
    targets = _parse_targets(args)

    benchmarks = [
        ("Air cooling", lambda: run_air_cooling_benchmark(cfg, n_steps=args.steps)),
        ("Induction heating", lambda: run_induction_heating_benchmark(cfg, n_steps=args.steps)),
        ("Fixed-end gradient", lambda: run_fixed_end_gradient_benchmark(cfg, n_steps=args.steps)),
        ("Idle heating verify", lambda: run_idle_heating_verify(cfg, targets, n_steps=args.steps)),
    ]

    all_ok = True
    for label, fn in benchmarks:
        print(f"\n=== {label} ===")
        try:
            res = fn()
            print(f"  {res.message}")
            for k, v in res.metrics.items():
                if isinstance(v, float):
                    print(f"  {k}: {v:.4f}")
                else:
                    print(f"  {k}: {v}")
            if not res.success:
                all_ok = False
        except Exception as exc:
            print(f"  ERROR: {exc}")
            all_ok = False

    return 0 if all_ok else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Thermal coefficient calibration for AgForge")
    sub = p.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--q-peak", type=float, default=None, help="Override q_peak [W/m³]")
    common.add_argument("--l-eff", type=float, default=None, help="Override L_eff [m]")
    common.add_argument("--h-air", type=float, default=None, help="Override h_air [W/m²K]")
    common.add_argument("--h-contact", type=float, default=None, help="Override h_contact [W/m²K]")
    common.add_argument("--target-temp", type=float, default=1200.0, help="Target surface temp [K]")
    common.add_argument("--target-time", type=float, default=45.0, help="Target heating time [thermal s]")
    common.add_argument("--initial-temp", type=float, default=293.15, help="Initial temp [K]")
    common.add_argument("--held-end-max", type=float, default=400.0, help="Max held-end temp [K]")
    common.add_argument("--ambient", type=float, default=293.15, help="Ambient/bulk temp [K]")
    common.add_argument("-o", "--output", type=str, default=None, help="Write JSON report")

    sub.add_parser("analyze", parents=[common], help="Print analysis report")
    sub.add_parser("predict", parents=[common], help="Analytical predict (alias for analyze)")

    fit_p = sub.add_parser("fit", parents=[common], help="Fit q_peak and L_eff")
    fit_p.add_argument("--gpu", action="store_true", help="Use GPU grid search (slow, accurate)")
    fit_p.add_argument(
        "--apply",
        type=str,
        default=None,
        help="Write a .patch.txt snippet with recommended values",
    )

    ver_p = sub.add_parser("verify", parents=[common], help="Run GPU benchmark suite")
    ver_p.add_argument(
        "--steps",
        type=int,
        default=350,
        help="Steps per benchmark (~350 ≈ 50 s thermal at default S_T)",
    )

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "analyze":
        return cmd_analyze(args)
    if args.command == "predict":
        return cmd_predict(args)
    if args.command == "fit":
        return cmd_fit(args)
    if args.command == "verify":
        return cmd_verify(args)
    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
