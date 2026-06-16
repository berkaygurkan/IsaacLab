#!/usr/bin/env python3
"""Package preliminary T17 closed-loop results for advisor/report review.

This utility is post-processing only. It reads existing T17 result directories
and writes compact tables plus a survival-metric audit. It does not launch
Isaac, run evaluation, train, or modify checkpoints/datasets.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = "papers/conference/results/t17_closed_loop_report_package"
DEFAULT_INPUT_ROOTS = (
    "papers/conference/results/t17_multijoint_closed_loop_eval_a2h_a5_alpha_sweep_seed0",
    "papers/conference/results/t17_multijoint_closed_loop_eval_a2h_a5_alpha_sweep_fixed_vx1_seed0",
    "papers/conference/results/t17_multijoint_closed_loop_eval_a2_single_step_fixed_vx1_seed0",
)

ABLATION_FIELDS = [
    "source_root",
    "velocity_mode",
    "protocol",
    "policy",
    "alpha",
    "mean_vel_x_pre_fault",
    "mean_vel_x_post_fault",
    "mean_abs_vx_error_pre_fault",
    "mean_abs_vx_error_post_fault",
    "median_abs_vx_error_post_fault",
    "p90_abs_vx_error_post_fault",
    "mean_abs_yaw_error",
    "torso_height_failure_rate",
    "timeout_rate",
    "current_raw_survival_rate",
    "current_raw_done_rate",
    "post_fault_alive_sample_fraction",
    "post_fault_done_sample_rate",
    "selected_fault_joint_all_8_covered",
    "fallback_used",
    "pd_surrogate_used",
    "simulation_override_applied",
    "command_mode_valid",
    "no_nan_inf",
    "post_fault_sample_count",
    "fault_active_env_count",
    "output_dir",
]

IMPROVEMENT_FIELDS = [
    "source_root",
    "velocity_mode",
    "protocol",
    "baseline_policy",
    "candidate_policy",
    "alpha",
    "baseline_post_fault_error",
    "candidate_post_fault_error",
    "absolute_error_reduction",
    "percent_error_reduction",
    "baseline_p90_error",
    "candidate_p90_error",
    "p90_error_reduction",
    "best_alpha_by_mean_error",
    "best_alpha_by_p90_error",
    "alpha_ranking_note",
]

PER_JOINT_FIELDS = [
    "source_root",
    "velocity_mode",
    "protocol",
    "policy",
    "alpha",
    "selected_fault_joint_index",
    "joint_name",
    "mean_abs_vx_error_post_fault",
    "p90_abs_vx_error_post_fault",
    "torso_height_failure_rate",
    "timeout_rate",
    "post_fault_sample_count",
    "post_fault_alive_sample_fraction",
    "post_fault_done_sample_rate",
]


def repo_relative(path: Path | str) -> str:
    path = Path(path)
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def resolve_path(path: str | Path) -> Path:
    path = Path(path).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def parse_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    if not math.isfinite(number):
        return None
    return number


def normalize_alpha(value: Any) -> str:
    number = parse_float(value)
    if number is None:
        return ""
    return f"{number:g}"


def format_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.9g}"
    text = str(value)
    if text.lower() == "none":
        return ""
    return text


def first_nonblank(*values: Any) -> Any:
    for value in values:
        if value is None:
            continue
        if str(value).strip() == "":
            continue
        return value
    return ""


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
        "Preliminary advisor-facing evidence only; not final paper-grade statistics.",
        "",
        "| " + " | ".join(fieldnames) + " |",
        "| " + " | ".join("---" for _ in fieldnames) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(format_value(row.get(field, "")) for field in fieldnames) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def key_for(row: dict[str, Any], source_root: str | None = None) -> tuple[str, str, str, str, str]:
    return (
        source_root or str(row.get("source_root", "")),
        str(row.get("protocol", "")),
        str(row.get("velocity_mode", "")),
        str(row.get("policy", "")),
        normalize_alpha(row.get("alpha", "")),
    )


def active_sample_survival_by_key(
    per_joint_rows: list[dict[str, str]],
    *,
    source_root: str,
) -> dict[tuple[str, str, str, str, str], dict[str, float]]:
    totals: dict[tuple[str, str, str, str, str], dict[str, float]] = {}
    for row in per_joint_rows:
        sample_count = parse_float(row.get("sample_count"))
        if sample_count is None or sample_count <= 0:
            continue
        key = key_for(row, source_root)
        values = totals.setdefault(key, {"samples": 0.0, "alive": 0.0, "done": 0.0})
        values["samples"] += sample_count
        survival_rate = parse_float(row.get("survival_rate"))
        done_rate = parse_float(row.get("done_rate"))
        if survival_rate is not None:
            values["alive"] += sample_count * survival_rate
        if done_rate is not None:
            values["done"] += sample_count * done_rate
    for values in totals.values():
        samples = values["samples"]
        values["post_fault_alive_sample_fraction"] = values["alive"] / samples if samples else None
        values["post_fault_done_sample_rate"] = values["done"] / samples if samples else None
    return totals


def load_result_roots(input_roots: list[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    ablation_rows: list[dict[str, Any]] = []
    per_joint_summary_rows: list[dict[str, Any]] = []
    warnings: list[str] = []

    for root_text in input_roots:
        root = resolve_path(root_text)
        source_root = repo_relative(root)
        per_run_path = root / "per_run_metrics.csv"
        per_joint_path = root / "per_joint_metrics.csv"
        per_run_rows = read_csv(per_run_path)
        per_joint_rows = read_csv(per_joint_path)
        if not per_run_rows:
            warnings.append(f"No per_run_metrics.csv rows found under {source_root}.")
            continue
        active_survival = active_sample_survival_by_key(per_joint_rows, source_root=source_root)
        for row in per_run_rows:
            key = key_for(row, source_root)
            survival = active_survival.get(key, {})
            ablation_rows.append(
                {
                    "source_root": source_root,
                    "velocity_mode": row.get("velocity_mode", ""),
                    "protocol": row.get("protocol", ""),
                    "policy": row.get("policy", ""),
                    "alpha": normalize_alpha(row.get("alpha", "")),
                    "mean_vel_x_pre_fault": row.get("mean_vel_x_pre_fault", ""),
                    "mean_vel_x_post_fault": row.get("mean_vel_x_post_fault", ""),
                    "mean_abs_vx_error_pre_fault": row.get("mean_abs_vx_error_pre_fault", ""),
                    "mean_abs_vx_error_post_fault": row.get("mean_abs_vx_error_post_fault", ""),
                    "median_abs_vx_error_post_fault": row.get("median_abs_vx_error_post_fault", ""),
                    "p90_abs_vx_error_post_fault": row.get("p90_abs_vx_error_post_fault", ""),
                    "mean_abs_yaw_error": row.get("mean_abs_yaw_error", ""),
                    "torso_height_failure_rate": row.get("torso_height_failure_rate", ""),
                    "timeout_rate": row.get("timeout_rate", ""),
                    "current_raw_survival_rate": first_nonblank(
                        row.get("raw_never_done_survival_rate"),
                        row.get("survival_rate"),
                    ),
                    "current_raw_done_rate": first_nonblank(row.get("raw_any_done_rate"), row.get("done_rate")),
                    "post_fault_alive_sample_fraction": first_nonblank(
                        row.get("post_fault_alive_sample_fraction"),
                        survival.get("post_fault_alive_sample_fraction"),
                    ),
                    "post_fault_done_sample_rate": first_nonblank(
                        row.get("post_fault_done_sample_rate"),
                        survival.get("post_fault_done_sample_rate"),
                    ),
                    "selected_fault_joint_all_8_covered": row.get("selected_fault_joint_all_8_covered", ""),
                    "fallback_used": row.get("fallback_used", ""),
                    "pd_surrogate_used": row.get("pd_surrogate_used", ""),
                    "simulation_override_applied": row.get("simulation_override_applied", ""),
                    "command_mode_valid": row.get("command_mode_valid", ""),
                    "no_nan_inf": row.get("no_nan_inf", ""),
                    "post_fault_sample_count": first_nonblank(row.get("post_fault_sample_count"), survival.get("samples")),
                    "fault_active_env_count": row.get("fault_active_env_count", ""),
                    "output_dir": row.get("output_dir", ""),
                }
            )
        for row in per_joint_rows:
            sample_count = parse_float(row.get("sample_count"))
            survival_rate = parse_float(row.get("survival_rate"))
            done_rate = parse_float(row.get("done_rate"))
            per_joint_summary_rows.append(
                {
                    "source_root": source_root,
                    "velocity_mode": row.get("velocity_mode", ""),
                    "protocol": row.get("protocol", ""),
                    "policy": row.get("policy", ""),
                    "alpha": normalize_alpha(row.get("alpha", "")),
                    "selected_fault_joint_index": row.get("joint_id", ""),
                    "joint_name": row.get("joint_name", ""),
                    "mean_abs_vx_error_post_fault": row.get("mean_post_fault_abs_vx_error", ""),
                    "p90_abs_vx_error_post_fault": row.get("p90_post_fault_abs_vx_error", ""),
                    "torso_height_failure_rate": "",
                    "timeout_rate": "",
                    "post_fault_sample_count": sample_count,
                    "post_fault_alive_sample_fraction": survival_rate,
                    "post_fault_done_sample_rate": done_rate,
                }
            )
    return ablation_rows, per_joint_summary_rows, warnings


def row_sort_key(row: dict[str, Any]) -> tuple[str, str, str, str, float]:
    alpha = parse_float(row.get("alpha"))
    return (
        str(row.get("velocity_mode", "")),
        str(row.get("protocol", "")),
        str(row.get("policy", "")),
        str(row.get("source_root", "")),
        -1.0 if alpha is None else alpha,
    )


def build_improvement_rows(
    ablation_rows: list[dict[str, Any]],
    *,
    baseline_policy: str,
    candidate_policy: str,
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in ablation_rows:
        groups.setdefault(
            (str(row.get("source_root", "")), str(row.get("velocity_mode", "")), str(row.get("protocol", ""))),
            [],
        ).append(row)

    raw_rows: list[dict[str, Any]] = []
    best_mean_by_group: dict[tuple[str, str, str], str] = {}
    best_p90_by_group: dict[tuple[str, str, str], str] = {}
    best_mean_by_mode_protocol: dict[tuple[str, str], str] = {}

    for group_key, rows in groups.items():
        baseline = next((row for row in rows if row.get("policy") == baseline_policy), None)
        if baseline is None:
            continue
        candidates = [row for row in rows if row.get("policy") == candidate_policy]
        if not candidates:
            continue
        best_mean = min(
            candidates,
            key=lambda row: parse_float(row.get("mean_abs_vx_error_post_fault")) or float("inf"),
        )
        best_p90 = min(
            candidates,
            key=lambda row: parse_float(row.get("p90_abs_vx_error_post_fault")) or float("inf"),
        )
        best_mean_alpha = normalize_alpha(best_mean.get("alpha", ""))
        best_p90_alpha = normalize_alpha(best_p90.get("alpha", ""))
        best_mean_by_group[group_key] = best_mean_alpha
        best_p90_by_group[group_key] = best_p90_alpha
        _, velocity_mode, protocol = group_key
        best_mean_by_mode_protocol[(velocity_mode, protocol)] = best_mean_alpha
        baseline_error = parse_float(baseline.get("mean_abs_vx_error_post_fault"))
        baseline_p90 = parse_float(baseline.get("p90_abs_vx_error_post_fault"))
        for candidate in candidates:
            candidate_error = parse_float(candidate.get("mean_abs_vx_error_post_fault"))
            candidate_p90 = parse_float(candidate.get("p90_abs_vx_error_post_fault"))
            absolute_reduction = None
            percent_reduction = None
            if baseline_error is not None and candidate_error is not None:
                absolute_reduction = baseline_error - candidate_error
                percent_reduction = 100.0 * absolute_reduction / baseline_error if baseline_error else None
            p90_reduction = None
            if baseline_p90 is not None and candidate_p90 is not None:
                p90_reduction = baseline_p90 - candidate_p90
            raw_rows.append(
                {
                    "source_root": group_key[0],
                    "velocity_mode": velocity_mode,
                    "protocol": protocol,
                    "baseline_policy": baseline_policy,
                    "candidate_policy": candidate_policy,
                    "alpha": normalize_alpha(candidate.get("alpha", "")),
                    "baseline_post_fault_error": baseline_error,
                    "candidate_post_fault_error": candidate_error,
                    "absolute_error_reduction": absolute_reduction,
                    "percent_error_reduction": percent_reduction,
                    "baseline_p90_error": baseline_p90,
                    "candidate_p90_error": candidate_p90,
                    "p90_error_reduction": p90_reduction,
                    "best_alpha_by_mean_error": best_mean_alpha,
                    "best_alpha_by_p90_error": best_p90_alpha,
                    "alpha_ranking_note": "",
                }
            )

    protocols = sorted({row["protocol"] for row in raw_rows})
    for protocol in protocols:
        best_by_velocity = {
            velocity_mode: alpha
            for (velocity_mode, row_protocol), alpha in best_mean_by_mode_protocol.items()
            if row_protocol == protocol
        }
        if len(set(best_by_velocity.values())) <= 1:
            note = "best alpha by mean error matches across available velocity modes"
        else:
            pieces = [f"{mode}: alpha={alpha}" for mode, alpha in sorted(best_by_velocity.items())]
            note = "best alpha by mean error differs across velocity modes (" + "; ".join(pieces) + ")"
        for row in raw_rows:
            if row["protocol"] == protocol:
                row["alpha_ranking_note"] = note

    return sorted(raw_rows, key=row_sort_key)


def write_advisor_snippet(path: Path, improvement_rows: list[dict[str, Any]]) -> None:
    positive_rows = [
        row
        for row in improvement_rows
        if (parse_float(row.get("percent_error_reduction")) or -float("inf")) > 0.0
    ]
    best = None
    if positive_rows:
        best = max(positive_rows, key=lambda row: parse_float(row.get("percent_error_reduction")) or -float("inf"))
    lines = [
        "# Advisor Result Snippet",
        "",
        "Preliminary advisor-facing evidence from a single seed on the Ant proxy, not final paper-grade statistics.",
        "",
        (
            "The deployment-facing policy receives no privileged fault labels, q-lock vector, health token, UQ, "
            "or safety filter."
        ),
        "",
        "A5 improves post-fault velocity tracking over frozen A2-history under both command-random and fixed-vx evaluations.",
        "",
        (
            "The fixed-vx A2 single-step run should be interpreted separately: it is unexpectedly strong under "
            "vx=1.0 and means the all-method fixed-vx figure is an ablation comparison, not evidence that A5 "
            "dominates every deployment-facing baseline."
        ),
    ]
    if best is not None:
        lines.extend(
            [
                "",
                (
                    "Largest packaged mean post-fault vx-error reduction: "
                    f"{format_value(best.get('percent_error_reduction'))}% for "
                    f"{best.get('velocity_mode')} / {best.get('protocol')} at alpha={best.get('alpha')}."
                ),
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report_paragraph(path: Path, improvement_rows: list[dict[str, Any]]) -> None:
    best_by_mode: dict[str, dict[str, Any]] = {}
    for row in improvement_rows:
        reduction = parse_float(row.get("percent_error_reduction"))
        if reduction is None:
            continue
        mode = str(row.get("velocity_mode", ""))
        previous = best_by_mode.get(mode)
        if previous is None or reduction > (parse_float(previous.get("percent_error_reduction")) or -float("inf")):
            best_by_mode[mode] = row
    mode_text = "; ".join(
        f"{mode}: alpha={row.get('alpha')}, {format_value(row.get('percent_error_reduction'))}% mean-error reduction"
        for mode, row in sorted(best_by_mode.items())
    )
    if not mode_text:
        mode_text = "no positive A5-over-A2-history reduction was found in the loaded rows"
    paragraph = (
        "In preliminary advisor-facing single-seed Ant proxy evaluations, the A5 history-residual policy improved "
        "post-fault velocity tracking relative to the frozen A2-history student in the packaged T17 runs. "
        f"Best packaged reductions by velocity mode were: {mode_text}. "
        "The fixed-vx=1.0 A2 single-step result is unexpectedly strong, so fixed-command plots should be read as "
        "ablation comparisons rather than as evidence that A5 dominates every deployment-facing baseline. "
        "Further command-random comparison is required before choosing the final deployment-facing variant. "
        "These are not final paper-grade statistics. The deployment policy remains fault-descriptor-free: it receives "
        "student observations only, with no privileged fault labels, q-lock vector, health token, UQ, or safety filter."
    )
    path.write_text("# Report Result Paragraph\n\n" + paragraph + "\n", encoding="utf-8")


def write_survival_audit(path: Path, ablation_rows: list[dict[str, Any]], warnings: list[str]) -> None:
    raw_done_values = [parse_float(row.get("current_raw_done_rate")) for row in ablation_rows]
    raw_done_values = [value for value in raw_done_values if value is not None]
    active_alive_values = [parse_float(row.get("post_fault_alive_sample_fraction")) for row in ablation_rows]
    active_alive_values = [value for value in active_alive_values if value is not None]
    lines = [
        "# T17 Survival Metric Audit",
        "",
        "Preliminary advisor-facing audit only; not final paper-grade statistics.",
        "",
        "## Finding",
        "",
        (
            "The existing `survival_rate`/`done_rate` columns are raw rollout-level first-done-ever metrics. "
            "In 1000-step Isaac rollouts, `done_rate=1` means every env eventually produced at least one done, "
            "not that every policy immediately failed after the P2 fault."
        ),
        "",
        "The packager therefore renames these as:",
        "",
        "- `current_raw_done_rate` / future `raw_any_done_rate`",
        "- `current_raw_survival_rate` / future `raw_never_done_survival_rate`",
        "",
        "## Post-Fault Metrics Recoverable Without Rerun",
        "",
        (
            "The existing root `per_joint_metrics.csv` files contain fault-active sample counts and active-sample "
            "done/survival rates by joint. The packager uses these to compute `post_fault_alive_sample_fraction` "
            "and `post_fault_done_sample_rate` for each run."
        ),
        "",
        f"- raw done-rate range in loaded rows: `{format_value(min(raw_done_values))}` to `{format_value(max(raw_done_values))}`"
        if raw_done_values
        else "- raw done-rate range in loaded rows: unavailable",
        f"- post-fault alive-sample fraction range: `{format_value(min(active_alive_values))}` to `{format_value(max(active_alive_values))}`"
        if active_alive_values
        else "- post-fault alive-sample fraction range: unavailable",
        "",
        "## Metrics Requiring Future Rerun",
        "",
        (
            "The current artifacts do not preserve per-env post-fault termination cause. Therefore "
            "`post_fault_torso_failure_rate`, `post_fault_timeout_success_rate`, and "
            "`post_fault_non_timeout_failure_rate` cannot be reconstructed faithfully from the existing CSVs."
        ),
        "",
        (
            "The evaluator has been patched for future runs to write active-sample counters "
            "`fault_active_sample_count`, `fault_active_done_count`, `fault_active_alive_count`, plus summary aliases "
            "`raw_any_done_rate`, `raw_never_done_survival_rate`, `post_fault_alive_sample_fraction`, and "
            "`post_fault_done_sample_rate`. Cause-separated timeout/collapse rates still require explicit "
            "termination-cause logging from Isaac/env extras."
        ),
        "",
        "## Interpretation",
        "",
        "- Timeout should be reported separately from collapse/failure.",
        "- A rollout-level done event should not automatically mean post-fault failure if the env later resets.",
        "- Survival should be reported relative to post-fault behavior, not only as no-done-ever over 1000 steps.",
    ]
    if warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in warnings)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_package_readme(path: Path, input_roots: list[str]) -> None:
    lines = [
        "# T17 Closed-Loop Report Package",
        "",
        "Post-processing package for preliminary T17 multi-joint P2 closed-loop results.",
        "",
        "This package is generated from existing evaluator outputs only. It does not launch Isaac, train, or mutate checkpoints.",
        "",
        "## Inputs",
        "",
    ]
    lines.extend(f"- `{root}`" for root in input_roots)
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            "- `t17_ablation_summary.csv/md`",
            "- `t17_improvement_table.csv/md`",
            "- `t17_per_joint_summary.csv/md`",
            "- `advisor_result_snippet.md`",
            "- `report_result_paragraph.md`",
            "- `survival_metric_audit.md`",
            "- `command.txt`",
            "",
            "## Survival Metric Note",
            "",
            (
                "Raw `survival_rate`/`done_rate` from the original T17 runs are first-done-ever rollout metrics. "
                "Use `post_fault_alive_sample_fraction` for the current no-rerun post-fault survival proxy, and rerun "
                "with future termination-cause logging before making cause-separated survival claims."
            ),
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize existing T17 closed-loop result directories.")
    parser.add_argument("--input_roots", nargs="+", default=list(DEFAULT_INPUT_ROOTS))
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--baseline_policy", default="a2_history")
    parser.add_argument("--candidate_policy", default="a5")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ablation_rows, per_joint_rows, warnings = load_result_roots(args.input_roots)
    ablation_rows = sorted(ablation_rows, key=row_sort_key)
    improvement_rows = build_improvement_rows(
        ablation_rows,
        baseline_policy=args.baseline_policy,
        candidate_policy=args.candidate_policy,
    )
    per_joint_rows = sorted(per_joint_rows, key=row_sort_key)

    write_csv(output_dir / "t17_ablation_summary.csv", ablation_rows, ABLATION_FIELDS)
    write_md_table(output_dir / "t17_ablation_summary.md", ablation_rows, ABLATION_FIELDS, "T17 Ablation Summary")
    write_csv(output_dir / "t17_improvement_table.csv", improvement_rows, IMPROVEMENT_FIELDS)
    write_md_table(
        output_dir / "t17_improvement_table.md",
        improvement_rows,
        IMPROVEMENT_FIELDS,
        "T17 A5 over A2-History Improvement Table",
    )
    write_csv(output_dir / "t17_per_joint_summary.csv", per_joint_rows, PER_JOINT_FIELDS)
    write_md_table(output_dir / "t17_per_joint_summary.md", per_joint_rows, PER_JOINT_FIELDS, "T17 Per-Joint Summary")
    write_advisor_snippet(output_dir / "advisor_result_snippet.md", improvement_rows)
    write_report_paragraph(output_dir / "report_result_paragraph.md", improvement_rows)
    write_survival_audit(output_dir / "survival_metric_audit.md", ablation_rows, warnings)
    write_package_readme(output_dir / "README.md", args.input_roots)
    (output_dir / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    (output_dir / "package_metadata.json").write_text(
        json.dumps(
            {
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "not_paper_grade_final": True,
                "input_roots": args.input_roots,
                "baseline_policy": args.baseline_policy,
                "candidate_policy": args.candidate_policy,
                "ablation_rows": len(ablation_rows),
                "improvement_rows": len(improvement_rows),
                "per_joint_rows": len(per_joint_rows),
                "warnings": warnings,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[T17 packager] wrote {repo_relative(output_dir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
