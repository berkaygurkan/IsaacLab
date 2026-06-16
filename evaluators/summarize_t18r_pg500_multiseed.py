#!/usr/bin/env python3
"""Summarize T18-R-PG500 closed-loop residual-ablation seed results.

This utility reads existing T17-style closed-loop result roots and writes
paper-grade package tables. It does not launch Isaac, train, evaluate, modify
checkpoints, or modify datasets.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_ROOTS = [
    "papers/conference/results/t18r_pg500_closed_loop_eval_50hz_h50_a2h_a5_alpha_sweep_seed0",
    "papers/conference/results/t18r_pg500_closed_loop_eval_50hz_h50_a2h_a5_alpha_sweep_seed1",
    "papers/conference/results/t18r_pg500_closed_loop_eval_50hz_h50_a2h_a5_alpha_sweep_seed2",
]
DEFAULT_OUTPUT_DIR = "papers/conference/results/t18r_pg500_closed_loop_report_package_50hz_h50"

METRIC_FIELDS = [
    "mean_vel_x_post_fault",
    "mean_abs_vx_error_post_fault",
    "median_abs_vx_error_post_fault",
    "p90_abs_vx_error_post_fault",
    "mean_abs_yaw_error",
    "post_fault_alive_sample_fraction",
    "post_fault_done_sample_rate",
]

PER_SEED_FIELDS = [
    "seed",
    "source_root",
    "protocol",
    "velocity_mode",
    "policy",
    "alpha",
    "control_frequency_hz",
    "physics_frequency_hz",
    "control_dt_s",
    "sim_dt_s",
    "decimation",
    *METRIC_FIELDS,
    "fallback_used",
    "pd_surrogate_used",
    "simulation_override_applied",
    "command_mode_valid",
    "no_nan_inf",
    "output_dir",
]

AGG_FIELDS = [
    "protocol",
    "velocity_mode",
    "policy",
    "alpha",
    "seed_count",
    "seeds",
    "control_frequency_hz",
    "physics_frequency_hz",
    "control_dt_s",
    "sim_dt_s",
    "decimation",
]
for metric_name in METRIC_FIELDS:
    AGG_FIELDS.extend([f"{metric_name}_mean", f"{metric_name}_std"])


def resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def repo_relative(path_value: str | Path) -> str:
    path = resolve_path(path_value)
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: format_value(row.get(field, "")) for field in fieldnames})


def write_md_table(path: Path, rows: list[dict[str, Any]], fieldnames: list[str], title: str) -> None:
    lines = [
        f"# {title}",
        "",
        "T18-R-PG500 paper-grade corrected simulation tracking table.",
        "",
        "| " + " | ".join(fieldnames) + " |",
        "| " + " | ".join("---" for _ in fieldnames) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(format_value(row.get(field, "")) for field in fieldnames) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    try:
        parsed = float(text)
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


def normalize_alpha(value: Any) -> str:
    parsed = parse_float(value)
    return "" if parsed is None else f"{parsed:g}"


def format_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        return f"{value:.9g}"
    text = str(value)
    return "" if text.lower() == "none" else text


def infer_seed(root: Path, fallback_index: int) -> int:
    text = root.name
    if "seed" in text:
        suffix = text.rsplit("seed", 1)[-1]
        digits = "".join(char for char in suffix if char.isdigit())
        if digits:
            return int(digits)
    return fallback_index


def load_seed_rows(input_roots: list[str]) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    for fallback_index, root_text in enumerate(input_roots):
        root = resolve_path(root_text)
        source_root = repo_relative(root)
        seed = infer_seed(root, fallback_index)
        per_run_rows = read_csv(root / "per_run_metrics.csv")
        if not per_run_rows:
            warnings.append(f"missing or empty per_run_metrics.csv under {source_root}")
            continue
        for row in per_run_rows:
            policy = row.get("policy", "")
            if policy == "a2_single_step":
                continue
            rows.append(
                {
                    "seed": seed,
                    "source_root": source_root,
                    "protocol": row.get("protocol", ""),
                    "velocity_mode": row.get("velocity_mode", ""),
                    "policy": policy,
                    "alpha": normalize_alpha(row.get("alpha", "")),
                    "control_frequency_hz": row.get("control_frequency_hz", ""),
                    "physics_frequency_hz": row.get("physics_frequency_hz", ""),
                    "control_dt_s": row.get("control_dt_s", ""),
                    "sim_dt_s": row.get("sim_dt_s", ""),
                    "decimation": row.get("decimation", ""),
                    **{metric: row.get(metric, "") for metric in METRIC_FIELDS},
                    "fallback_used": row.get("fallback_used", ""),
                    "pd_surrogate_used": row.get("pd_surrogate_used", ""),
                    "simulation_override_applied": row.get("simulation_override_applied", ""),
                    "command_mode_valid": row.get("command_mode_valid", ""),
                    "no_nan_inf": row.get("no_nan_inf", ""),
                    "output_dir": row.get("output_dir", ""),
                }
            )
    return rows, warnings


def aggregate_rows(per_seed_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for row in per_seed_rows:
        key = (
            str(row.get("protocol", "")),
            str(row.get("velocity_mode", "")),
            str(row.get("policy", "")),
            normalize_alpha(row.get("alpha", "")),
        )
        groups.setdefault(key, []).append(row)

    aggregate: list[dict[str, Any]] = []
    for (protocol, velocity_mode, policy, alpha), rows in sorted(groups.items()):
        seeds = sorted({int(row["seed"]) for row in rows})
        output: dict[str, Any] = {
            "protocol": protocol,
            "velocity_mode": velocity_mode,
            "policy": policy,
            "alpha": alpha,
            "seed_count": len(seeds),
            "seeds": ",".join(str(seed) for seed in seeds),
            "control_frequency_hz": rows[0].get("control_frequency_hz", ""),
            "physics_frequency_hz": rows[0].get("physics_frequency_hz", ""),
            "control_dt_s": rows[0].get("control_dt_s", ""),
            "sim_dt_s": rows[0].get("sim_dt_s", ""),
            "decimation": rows[0].get("decimation", ""),
        }
        for metric_name in METRIC_FIELDS:
            values = [value for row in rows if (value := parse_float(row.get(metric_name))) is not None]
            output[f"{metric_name}_mean"] = statistics.mean(values) if values else None
            output[f"{metric_name}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0 if values else None
        aggregate.append(output)
    return aggregate


def write_protocol_readme(path: Path, warnings: list[str]) -> None:
    lines = [
        "# T18-R-PG500 Protocol",
        "",
        "- physics frequency: 500 Hz",
        "- sim_dt: 0.002 s",
        "- control frequency: 50 Hz",
        "- decimation: 10",
        "- control_dt: 0.02 s",
        "- history_len: 50",
        "- history duration: 1.0 s",
        "- policies: A2-history H50, A5-H50 alpha sweep, privileged A1-F teacher reference",
        "- A2 single-step is intentionally excluded from this residual-ablation table.",
        "- deployment-facing policies receive no selected joint id, q_lock vector, fault-active flag, health token, UQ, or CBF output.",
        "",
        "## Warnings",
        "",
    ]
    lines.extend(f"- {warning}" for warning in warnings)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize T18-R-PG500 multi-seed closed-loop results.")
    parser.add_argument("--input_roots", nargs="+", default=DEFAULT_INPUT_ROOTS)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = resolve_path(args.output_dir)
    per_seed_rows, warnings = load_seed_rows(args.input_roots)
    aggregate = aggregate_rows(per_seed_rows)
    print("[T18R-PG500 SUMMARY]")
    print(f"input_roots: {len(args.input_roots)}")
    print(f"per_seed_rows: {len(per_seed_rows)}")
    print(f"aggregate_rows: {len(aggregate)}")
    print(f"output_dir: {repo_relative(output_dir)}")
    if warnings:
        print("warnings:")
        for warning in warnings:
            print(f"  - {warning}")
    if args.dry_run:
        print("dry_run: no files written")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "table_t18r_pg500_50hz_h50_tracking_error_per_seed.csv", per_seed_rows, PER_SEED_FIELDS)
    write_md_table(
        output_dir / "table_t18r_pg500_50hz_h50_tracking_error_per_seed.md",
        per_seed_rows,
        PER_SEED_FIELDS,
        "T18-R-PG500 Per-Seed Tracking Error",
    )
    write_csv(output_dir / "table_t18r_pg500_50hz_h50_tracking_error.csv", aggregate, AGG_FIELDS)
    write_md_table(
        output_dir / "table_t18r_pg500_50hz_h50_tracking_error.md",
        aggregate,
        AGG_FIELDS,
        "T18-R-PG500 Mean/Std Tracking Error",
    )
    timing_metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "phase": "T18-R-PG500",
        "physics_frequency_hz": 500.0,
        "sim_dt_s": 0.002,
        "control_frequency_hz": 50.0,
        "decimation": 10,
        "control_dt_s": 0.02,
        "history_len": 50,
        "input_roots": [repo_relative(root) for root in args.input_roots],
        "warnings": warnings,
    }
    (output_dir / "timing_metadata.json").write_text(json.dumps(timing_metadata, indent=2, sort_keys=True) + "\n")
    write_protocol_readme(output_dir / "paper_grade_protocol_readme.md", warnings)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
