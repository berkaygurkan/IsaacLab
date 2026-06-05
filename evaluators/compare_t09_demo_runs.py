#!/usr/bin/env python3
"""Compare two T09 advisor-demo run folders.

This helper is offline and stdlib-only. It reads summary_metrics.json when
available, falls back to step_metrics.csv, and writes a compact Markdown table.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = "runs/t09_demo_play/comparisons"


def resolve_repo_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def repo_relative(path: str | Path) -> str:
    resolved = resolve_repo_path(path).resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(resolved)


def _numeric(value: Any) -> float | None:
    if value in (None, "", "NA"):
        return None
    try:
        float_value = float(value)
    except (TypeError, ValueError):
        return None
    return float_value if math.isfinite(float_value) else None


def _mean(values: list[Any]) -> float | None:
    numeric_values = [value for value in (_numeric(value) for value in values) if value is not None]
    if not numeric_values:
        return None
    return sum(numeric_values) / len(numeric_values)


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"true", "1", "yes"}


def _read_step_csv_summary(run_dir: Path) -> dict[str, Any]:
    csv_path = run_dir / "step_metrics.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"missing summary_metrics.json and step_metrics.csv in {repo_relative(run_dir)}")
    with csv_path.open("r", encoding="utf-8", newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))
    return {
        "fault_profile": rows[0].get("fault_profile") if rows else "unknown",
        "rollout_steps_executed": len(rows),
        "overall_reward_mean": _mean([row.get("reward_mean") for row in rows]),
        "overall_base_lin_vel_x_mean": _mean([row.get("base_lin_vel_x_mean") for row in rows]),
        "overall_action_l2_mean": _mean([row.get("action_l2_mean") for row in rows]),
        "total_done_count": sum(int(_numeric(row.get("done_count")) or 0) for row in rows),
        "fault_window_reached": any(_bool(row.get("fault_active")) for row in rows),
        "runtime_smoke_status": "derived_from_step_csv",
    }


def read_run_summary(run_dir_value: str) -> dict[str, Any]:
    run_dir = resolve_repo_path(run_dir_value)
    summary_path = run_dir / "summary_metrics.json"
    if summary_path.is_file():
        with summary_path.open("r", encoding="utf-8") as summary_file:
            summary = json.load(summary_file)
    else:
        summary = _read_step_csv_summary(run_dir)
    summary["run_dir"] = repo_relative(run_dir)
    return summary


def _metric(summary: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if summary.get(key) not in (None, "NA", ""):
            return summary.get(key)
    return None


def _format_value(value: Any) -> str:
    numeric_value = _numeric(value)
    if numeric_value is not None:
        return f"{numeric_value:.4f}"
    if value is None:
        return "NA"
    return str(value)


def _format_delta(f0_value: Any, p4_value: Any) -> str:
    f0_numeric = _numeric(f0_value)
    p4_numeric = _numeric(p4_value)
    if f0_numeric is None or p4_numeric is None:
        return "NA"
    return f"{p4_numeric - f0_numeric:.4f}"


def _safe_name(value: str) -> str:
    safe = []
    for char in value:
        if char.isalnum() or char in {"-", "_"}:
            safe.append(char)
        else:
            safe.append("_")
    return "".join(safe).strip("_") or "demo"


def _default_label(summary: dict[str, Any]) -> str:
    ablation_id = summary.get("ablation_id") or "demo"
    fault_profile = summary.get("fault_profile") or "unknown_fault"
    torque_scale = summary.get("torque_scale")
    if torque_scale not in (None, "", "NA"):
        return f"{ablation_id}_{fault_profile}_torque{torque_scale}"
    return f"{ablation_id}_{fault_profile}"


def build_markdown(
    left_summary: dict[str, Any],
    right_summary: dict[str, Any],
    *,
    left_label: str | None = None,
    right_label: str | None = None,
) -> str:
    left_label = left_label or _default_label(left_summary)
    right_label = right_label or _default_label(right_summary)
    rows = [
        (
            "rollout_steps",
            _metric(left_summary, "rollout_steps_executed"),
            _metric(right_summary, "rollout_steps_executed"),
        ),
        (
            "reward_mean",
            _metric(left_summary, "overall_reward_mean", "pre_fault_reward_mean"),
            _metric(right_summary, "overall_reward_mean"),
        ),
        (
            "base_lin_vel_x_mean",
            _metric(left_summary, "overall_base_lin_vel_x_mean", "pre_fault_base_lin_vel_x_mean"),
            _metric(right_summary, "overall_base_lin_vel_x_mean"),
        ),
        (
            "action_l2_mean",
            _metric(left_summary, "overall_action_l2_mean", "pre_fault_action_l2_mean"),
            _metric(right_summary, "overall_action_l2_mean"),
        ),
        (
            "done_count",
            _metric(left_summary, "total_done_count"),
            _metric(right_summary, "total_done_count"),
        ),
        (
            "fault_window_reached",
            _metric(left_summary, "fault_window_reached"),
            _metric(right_summary, "fault_window_reached"),
        ),
        (
            "runtime_smoke_status",
            _metric(left_summary, "runtime_smoke_status"),
            _metric(right_summary, "runtime_smoke_status"),
        ),
    ]
    lines = [
        "# T09 Demo Comparison",
        "",
        "Scope: advisor-demo interpretation only; not paper-grade evaluation.",
        "",
        f"{left_label} run: `{left_summary.get('run_dir')}`",
        f"{right_label} run: `{right_summary.get('run_dir')}`",
        "",
        f"| metric | {left_label} | {right_label} | delta right-left |",
        "| --- | ---: | ---: | ---: |",
    ]
    for metric_name, left_value, right_value in rows:
        lines.append(
            f"| {metric_name} | {_format_value(left_value)} | {_format_value(right_value)} | "
            f"{_format_delta(left_value, right_value)} |"
        )
    lines.extend(["", "Note: advisor demo only, not paper-grade result.", ""])
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare two T09 demo run folders.")
    parser.add_argument("--left_run_dir", help="Left/baseline demo run folder.")
    parser.add_argument("--right_run_dir", help="Right/comparison demo run folder.")
    parser.add_argument("--left_label", help="Optional left column label.")
    parser.add_argument("--right_label", help="Optional right column label.")
    parser.add_argument("--f0_run_dir", help="Backward-compatible alias for --left_run_dir.")
    parser.add_argument("--p4_run_dir", help="Backward-compatible alias for --right_run_dir.")
    parser.add_argument("--output_root", default=DEFAULT_OUTPUT_ROOT)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    left_run_dir = args.left_run_dir or args.f0_run_dir
    right_run_dir = args.right_run_dir or args.p4_run_dir
    if not left_run_dir or not right_run_dir:
        raise SystemExit("error: provide --left_run_dir/--right_run_dir or --f0_run_dir/--p4_run_dir.")
    left_summary = read_run_summary(left_run_dir)
    right_summary = read_run_summary(right_run_dir)
    left_label = args.left_label or ("F0_none" if args.f0_run_dir and not args.left_run_dir else None)
    right_label = args.right_label or ("P4_torque_degradation" if args.p4_run_dir and not args.right_run_dir else None)
    markdown = build_markdown(left_summary, right_summary, left_label=left_label, right_label=right_label)
    output_root = resolve_repo_path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    output_left = _safe_name(left_label or _default_label(left_summary))
    output_right = _safe_name(right_label or _default_label(right_summary))
    output_path = output_root / f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{output_left}_vs_{output_right}.md"
    output_path.write_text(markdown, encoding="utf-8")
    print(markdown)
    print(f"comparison_path: {repo_relative(output_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
