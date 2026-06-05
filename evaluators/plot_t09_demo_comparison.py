#!/usr/bin/env python3
"""Offline A0 advisor-demo plot generator.

This script reads existing T09 demo run folders and saves PNG figures only. It
does not launch Isaac Sim, run play, train, or write result manifests.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = "papers/conference/demo/advisor_meeting_2026_06_03/figures"
DEMO_NOTE = "advisor demo only, not paper-grade result"


@dataclass
class DemoRun:
    label: str
    run_dir: Path
    summary: dict[str, Any]
    steps: list[dict[str, Any]]


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


def numeric(value: Any) -> float | None:
    if value in (None, "", "NA"):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def bool_numeric(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    lowered = str(value).lower()
    if lowered in {"true", "1", "yes"}:
        return 1.0
    if lowered in {"false", "0", "no"}:
        return 0.0
    return numeric(value)


def read_summary(run_dir: Path) -> dict[str, Any]:
    summary_path = run_dir / "summary_metrics.json"
    if not summary_path.is_file():
        print(f"[T09-DEMO-H WARNING] missing summary_metrics.json: {repo_relative(summary_path)}")
        return {}
    with summary_path.open("r", encoding="utf-8") as summary_file:
        return json.load(summary_file)


def read_steps(run_dir: Path) -> list[dict[str, Any]]:
    csv_path = run_dir / "step_metrics.csv"
    if not csv_path.is_file():
        print(f"[T09-DEMO-H WARNING] missing step_metrics.csv: {repo_relative(csv_path)}")
        return []
    with csv_path.open("r", encoding="utf-8", newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def label_from_summary(default_label: str, summary: dict[str, Any]) -> str:
    fault_profile = summary.get("fault_profile")
    torque_scale = summary.get("torque_scale")
    if fault_profile == "F0_none":
        return "F0_none"
    if torque_scale not in (None, "", "NA"):
        return f"P4 torque_scale={torque_scale}"
    return default_label


def load_run(path_value: str, default_label: str) -> DemoRun:
    run_dir = resolve_repo_path(path_value)
    if not run_dir.is_dir():
        raise FileNotFoundError(f"run folder missing: {repo_relative(run_dir)}")
    summary = read_summary(run_dir)
    steps = read_steps(run_dir)
    return DemoRun(
        label=label_from_summary(default_label, summary),
        run_dir=run_dir,
        summary=summary,
        steps=steps,
    )


def import_pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("matplotlib is required for T09 demo plotting but is not available.") from exc
    return plt


def step_series(run: DemoRun, metric: str, *, bool_values: bool = False) -> tuple[list[float], list[float]]:
    xs = []
    ys = []
    converter = bool_numeric if bool_values else numeric
    for row in run.steps:
        x_value = numeric(row.get("step"))
        y_value = converter(row.get(metric))
        if x_value is None or y_value is None:
            continue
        xs.append(x_value)
        ys.append(y_value)
    return xs, ys


def plot_step_metric(plt: Any, runs: list[DemoRun], metric: str, output_path: Path, title: str, ylabel: str) -> Path | None:
    plotted = False
    plt.figure(figsize=(10, 5.5))
    for run in runs:
        xs, ys = step_series(run, metric)
        if not ys:
            print(f"[T09-DEMO-H WARNING] metric {metric} unavailable for {run.label}; skipping that line.")
            continue
        plt.plot(xs, ys, label=run.label, linewidth=1.6)
        plotted = True
    if not plotted:
        print(f"[T09-DEMO-H WARNING] metric {metric} unavailable for all runs; plot skipped.")
        plt.close()
        return None
    plt.title(f"{title}\n{DEMO_NOTE}")
    plt.xlabel("step")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
    return output_path


def plot_done_metric(plt: Any, runs: list[DemoRun], output_path: Path, title: str) -> Path | None:
    metric = "done_count"
    bool_values = False
    if not any(step_series(run, metric)[1] for run in runs):
        metric = "dones_any"
        bool_values = True
    plotted = False
    plt.figure(figsize=(10, 5.5))
    for run in runs:
        xs, ys = step_series(run, metric, bool_values=bool_values)
        if not ys:
            print(f"[T09-DEMO-H WARNING] metric {metric} unavailable for {run.label}; skipping that line.")
            continue
        plt.plot(xs, ys, label=run.label, linewidth=1.6)
        plotted = True
    if not plotted:
        print("[T09-DEMO-H WARNING] done_count/dones_any unavailable for all runs; plot skipped.")
        plt.close()
        return None
    plt.title(f"{title}\n{DEMO_NOTE}")
    plt.xlabel("step")
    plt.ylabel(metric)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
    return output_path


def plot_summary_bars(plt: Any, runs: list[DemoRun], output_path: Path, title: str) -> Path | None:
    metrics = [
        ("overall_reward_mean", "reward mean"),
        ("overall_base_lin_vel_x_mean", "base lin vel x mean"),
        ("overall_action_l2_mean", "action L2 mean"),
        ("total_done_count", "done count"),
    ]
    plotted_any = False
    figure, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    axes_flat = [axis for row in axes for axis in row]
    for axis, (metric, label) in zip(axes_flat, metrics):
        values = []
        labels = []
        for run in runs:
            value = numeric(run.summary.get(metric))
            if value is None:
                continue
            labels.append(run.label)
            values.append(value)
        if not values:
            axis.set_title(f"{label} unavailable")
            axis.axis("off")
            print(f"[T09-DEMO-H WARNING] summary metric {metric} unavailable; subplot skipped.")
            continue
        axis.bar(range(len(values)), values)
        axis.set_title(label)
        axis.set_xticks(range(len(values)))
        axis.set_xticklabels(labels, rotation=20, ha="right")
        axis.grid(True, axis="y", alpha=0.3)
        plotted_any = True
    if not plotted_any:
        print("[T09-DEMO-H WARNING] no summary metrics available; summary bar plot skipped.")
        plt.close(figure)
        return None
    figure.suptitle(f"{title}\n{DEMO_NOTE}")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return output_path


def write_report(output_dir: Path, runs: list[DemoRun], generated: list[Path], title: str) -> Path:
    report_path = output_dir / "a0_demo_plot_report.md"
    lines = [
        "# A0 Demo Plot Report",
        "",
        f"Title: {title}",
        "",
        f"Note: {DEMO_NOTE}",
        "",
        "## Run Folders",
        "",
    ]
    for run in runs:
        lines.extend(
            [
                f"- {run.label}: `{repo_relative(run.run_dir)}`",
                f"  - checkpoint_path: `{run.summary.get('checkpoint_path', 'NA')}`",
                f"  - resolved_checkpoint_path: `{run.summary.get('resolved_checkpoint_path', 'NA')}`",
            ]
        )
    lines.extend(["", "## Generated Figures", ""])
    if generated:
        for path in generated:
            lines.append(f"- `{repo_relative(path)}`")
    else:
        lines.append("- No figures generated; required metrics were unavailable.")
    lines.extend(
        [
            "",
            "## Interpretation Placeholders",
            "",
            "- Reward trend: TBD after advisor-demo inspection.",
            "- Forward velocity trend: TBD after advisor-demo inspection.",
            "- Action magnitude trend: TBD after advisor-demo inspection.",
            "- Done/fall behavior: TBD after advisor-demo inspection.",
            "",
            "These plots are advisor-demo artifacts only, not paper-grade figures.",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot offline A0 T09 demo run comparisons.")
    parser.add_argument("--f0_run_dir", required=True)
    parser.add_argument("--p4_run_dir", required=True)
    parser.add_argument("--stress_run_dir")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--title", default="A0 Demo: F0 vs P4")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output_dir = resolve_repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    runs = [
        load_run(args.f0_run_dir, "F0_none"),
        load_run(args.p4_run_dir, "P4 torque_scale=0.5"),
    ]
    if args.stress_run_dir:
        runs.append(load_run(args.stress_run_dir, "P4 visual stress torque_scale=0.0"))

    plt = import_pyplot()
    generated = []
    plot_specs = [
        ("reward_mean", "reward_mean_vs_step.png", "Reward Mean vs Step", "reward_mean"),
        ("base_lin_vel_x_mean", "base_lin_vel_x_mean_vs_step.png", "Base Linear Velocity X vs Step", "base_lin_vel_x_mean"),
        ("action_l2_mean", "action_l2_mean_vs_step.png", "Action L2 Mean vs Step", "action_l2_mean"),
    ]
    for metric, filename, title, ylabel in plot_specs:
        path = plot_step_metric(plt, runs, metric, output_dir / filename, f"{args.title}: {title}", ylabel)
        if path is not None:
            generated.append(path)
    done_path = plot_done_metric(
        plt,
        runs,
        output_dir / "done_count_or_dones_vs_step.png",
        f"{args.title}: Done Count or Dones vs Step",
    )
    if done_path is not None:
        generated.append(done_path)
    summary_path = plot_summary_bars(plt, runs, output_dir / "summary_bar_metrics.png", f"{args.title}: Summary Metrics")
    if summary_path is not None:
        generated.append(summary_path)

    report_path = write_report(output_dir, runs, generated, args.title)
    print("[T09-DEMO-H] plot generation complete")
    for path in generated:
        print(f"  figure: {repo_relative(path)}")
    print(f"  report: {repo_relative(report_path)}")
    print(f"  note: {DEMO_NOTE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
