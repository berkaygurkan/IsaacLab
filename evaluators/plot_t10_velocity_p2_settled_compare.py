#!/usr/bin/env python3
"""Plot three-way settled-regime T10 P2 velocity comparisons.

This utility reads existing evaluation artifacts only. It does not launch Isaac
Sim, run evaluation, train, modify checkpoints, edit task configs, or touch the
P2 wrapper.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_A0_DIR = "papers/conference/results/t10_velocity_p2_settled_eval_seed0/a0_single"
DEFAULT_RANDOM_TEACHER_DIR = (
    "papers/conference/results/t10_velocity_p2_settled_eval_seed0/a1f_random_teacher_single"
)
DEFAULT_CURRICULUM_TEACHER_DIR = (
    "papers/conference/results/t10_velocity_p2_settled_eval_seed0/a1f_curriculum_teacher_single"
)
DEFAULT_OUTPUT_DIR = "papers/conference/results/t10_velocity_p2_settled_eval_seed0/plots_compare"
PRE_SETTLED_START = 250
PRE_SETTLED_END = 299
POST_FAULT_START = 300
POST_FAULT_END = 1000


class PlotError(ValueError):
    """Raised for invalid plotting inputs."""


@dataclass(frozen=True)
class PolicyInput:
    key: str
    label: str
    directory: Path
    color: str


@dataclass
class PolicySeries:
    spec: PolicyInput
    steps: list[int]
    mean_vel_x: list[float]
    mean_abs_vx_error: list[float]
    csv_policy_labels: list[str]
    summary: dict[str, Any]


def resolve_repo_path(path_value: str | Path) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def repo_relative(path_value: str | Path) -> str:
    resolved = resolve_repo_path(path_value).resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(resolved)


def parse_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"none", "null", "na", "nan"}:
        return None
    try:
        parsed = float(text)
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot three-way settled-regime T10 P2 velocity comparison.")
    parser.add_argument("--a0_dir", default=DEFAULT_A0_DIR)
    parser.add_argument("--random_teacher_dir", default=DEFAULT_RANDOM_TEACHER_DIR)
    parser.add_argument("--curriculum_teacher_dir", default=DEFAULT_CURRICULUM_TEACHER_DIR)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--fault_onset_step", type=int, default=300)
    parser.add_argument("--target_vx", type=float, default=1.0)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.fault_onset_step < 0:
        raise PlotError("--fault_onset_step must be non-negative.")
    if not math.isfinite(args.target_vx):
        raise PlotError("--target_vx must be finite.")


def require_matplotlib() -> Any:
    try:
        os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise PlotError("matplotlib is required for this plotter; install matplotlib and retry.") from exc
    return plt


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise PlotError(f"summary.json not found: {repo_relative(path)}")
    return json.loads(path.read_text(encoding="utf-8"))


def read_timeseries(path: Path, *, fallback_label: str, target_vx: float) -> tuple[list[int], list[float], list[float], list[str]]:
    if not path.is_file():
        raise PlotError(f"velocity_timeseries.csv not found: {repo_relative(path)}")
    steps: list[int] = []
    mean_vel_x: list[float] = []
    mean_abs_vx_error: list[float] = []
    labels: set[str] = set()
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise PlotError(f"CSV has no header: {repo_relative(path)}")
        fieldnames = set(reader.fieldnames)
        missing = {"step", "mean_vel_x"} - fieldnames
        if missing:
            raise PlotError(f"{repo_relative(path)} missing required columns: {sorted(missing)}")
        has_error = "mean_abs_vx_error" in fieldnames
        for row in reader:
            step_value = parse_float(row.get("step"))
            vx_value = parse_float(row.get("mean_vel_x"))
            if step_value is None or vx_value is None:
                continue
            error_value = parse_float(row.get("mean_abs_vx_error")) if has_error else None
            if error_value is None:
                error_value = abs(vx_value - target_vx)
            label = str(row.get("policy_label") or fallback_label).strip() or fallback_label
            labels.add(label)
            steps.append(int(step_value))
            mean_vel_x.append(float(vx_value))
            mean_abs_vx_error.append(float(error_value))
    if not steps:
        raise PlotError(f"no usable timeseries rows found: {repo_relative(path)}")
    order = sorted(range(len(steps)), key=lambda index: steps[index])
    return (
        [steps[index] for index in order],
        [mean_vel_x[index] for index in order],
        [mean_abs_vx_error[index] for index in order],
        sorted(labels),
    )


def load_policy_series(spec: PolicyInput, *, target_vx: float) -> PolicySeries:
    csv_path = spec.directory / "velocity_timeseries.csv"
    summary_path = spec.directory / "summary.json"
    steps, velocities, errors, labels = read_timeseries(csv_path, fallback_label=spec.label, target_vx=target_vx)
    summary = load_json(summary_path)
    return PolicySeries(
        spec=spec,
        steps=steps,
        mean_vel_x=velocities,
        mean_abs_vx_error=errors,
        csv_policy_labels=labels,
        summary=summary,
    )


def mean_window(steps: list[int], values: list[float], *, start: int, end: int) -> float:
    selected = [value for step, value in zip(steps, values) if start <= step <= end and math.isfinite(value)]
    if not selected:
        raise PlotError(f"no values in requested window [{start}, {end}].")
    return sum(selected) / len(selected)


def first_policy_summary(summary: dict[str, Any]) -> dict[str, Any]:
    policy_summary = summary.get("policy_summary")
    if isinstance(policy_summary, dict):
        return policy_summary
    policies = summary.get("policies")
    if isinstance(policies, dict):
        for value in policies.values():
            if isinstance(value, dict):
                return value
    return summary


def summary_rate(summary: dict[str, Any], key: str) -> float | None:
    for container in (summary, first_policy_summary(summary)):
        value = parse_float(container.get(key))
        if value is not None:
            return value
    return None


def settled_summary(series: PolicySeries) -> dict[str, float | None]:
    mean_vel_x_pre = mean_window(
        series.steps,
        series.mean_vel_x,
        start=PRE_SETTLED_START,
        end=PRE_SETTLED_END,
    )
    mean_vel_x_post = mean_window(
        series.steps,
        series.mean_vel_x,
        start=POST_FAULT_START,
        end=POST_FAULT_END,
    )
    mean_abs_vx_error_pre = mean_window(
        series.steps,
        series.mean_abs_vx_error,
        start=PRE_SETTLED_START,
        end=PRE_SETTLED_END,
    )
    mean_abs_vx_error_post = mean_window(
        series.steps,
        series.mean_abs_vx_error,
        start=POST_FAULT_START,
        end=POST_FAULT_END,
    )
    velocity_retention = mean_vel_x_post / mean_vel_x_pre if abs(mean_vel_x_pre) > 1.0e-12 else None
    return {
        "mean_vel_x_pre_settled": mean_vel_x_pre,
        "mean_vel_x_post_fault": mean_vel_x_post,
        "mean_abs_vx_error_pre_settled": mean_abs_vx_error_pre,
        "mean_abs_vx_error_post_fault": mean_abs_vx_error_post,
        "velocity_retention": velocity_retention,
        "torso_height_failure_rate": summary_rate(series.summary, "torso_height_failure_rate"),
        "timeout_rate": summary_rate(series.summary, "timeout_rate"),
    }


def add_onset_line(ax: Any, *, fault_onset_step: int) -> None:
    ax.axvline(fault_onset_step, color="#444444", linestyle="--", linewidth=1.2, alpha=0.85)
    ymin, ymax = ax.get_ylim()
    ax.text(
        fault_onset_step + 8,
        ymax - 0.07 * (ymax - ymin),
        "P2 onset",
        color="#333333",
        fontsize=9,
        va="top",
    )


def style_axis(ax: Any, *, ylabel: str, fault_onset_step: int) -> None:
    ax.set_xlabel("step")
    ax.set_ylabel(ylabel)
    ax.grid(True, color="#d7dce2", linewidth=0.7, alpha=0.8)
    add_onset_line(ax, fault_onset_step=fault_onset_step)


def save_figure(fig: Any, output_dir: Path, stem: str) -> list[Path]:
    output_paths = [output_dir / f"{stem}.png", output_dir / f"{stem}.pdf"]
    for path in output_paths:
        fig.savefig(path, bbox_inches="tight", dpi=220)
    return output_paths


def plot_velocity(plt: Any, series_items: list[PolicySeries], *, output_dir: Path, fault_onset_step: int, target_vx: float) -> list[Path]:
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    for series in series_items:
        ax.plot(series.steps, series.mean_vel_x, label=series.spec.label, color=series.spec.color, linewidth=1.8)
    ax.axhline(target_vx, color="#2e7d32", linestyle=":", linewidth=1.4, label="vx_cmd = 1.0 m/s")
    ax.set_title("Settled-regime P2 fault response, vx_cmd = 1.0 m/s", pad=14)
    fig.text(0.5, 0.01, "mean over 128 parallel environments", ha="center", fontsize=9, color="#444444")
    style_axis(ax, ylabel="mean_vel_x", fault_onset_step=fault_onset_step)
    ax.legend(loc="best", frameon=False)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    paths = save_figure(fig, output_dir, "vel_x_time_three_way")
    plt.close(fig)
    return paths


def plot_error(plt: Any, series_items: list[PolicySeries], *, output_dir: Path, fault_onset_step: int) -> list[Path]:
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    for series in series_items:
        ax.plot(
            series.steps,
            series.mean_abs_vx_error,
            label=series.spec.label,
            color=series.spec.color,
            linewidth=1.8,
        )
    ax.set_title("Velocity tracking error after settled-regime P2 fault", pad=14)
    style_axis(ax, ylabel="mean_abs_vx_error", fault_onset_step=fault_onset_step)
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    paths = save_figure(fig, output_dir, "vx_error_time_three_way")
    plt.close(fig)
    return paths


def plot_combined(
    plt: Any,
    series_items: list[PolicySeries],
    *,
    output_dir: Path,
    fault_onset_step: int,
    target_vx: float,
) -> list[Path]:
    fig, axes = plt.subplots(2, 1, figsize=(9.2, 7.4), sharex=True)
    ax_vel, ax_error = axes
    for series in series_items:
        ax_vel.plot(series.steps, series.mean_vel_x, label=series.spec.label, color=series.spec.color, linewidth=1.7)
        ax_error.plot(
            series.steps,
            series.mean_abs_vx_error,
            label=series.spec.label,
            color=series.spec.color,
            linewidth=1.7,
        )
    ax_vel.axhline(target_vx, color="#2e7d32", linestyle=":", linewidth=1.3, label="vx_cmd = 1.0 m/s")
    ax_vel.set_title("Settled-regime P2 fault response, vx_cmd = 1.0 m/s", pad=10)
    style_axis(ax_vel, ylabel="mean_vel_x", fault_onset_step=fault_onset_step)
    style_axis(ax_error, ylabel="mean_abs_vx_error", fault_onset_step=fault_onset_step)
    ax_error.set_title("Velocity tracking error after settled-regime P2 fault", pad=10)
    handles, labels = ax_vel.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.text(0.5, 0.01, "mean over 128 parallel environments", ha="center", fontsize=9, color="#444444")
    fig.tight_layout(rect=(0, 0.03, 1, 0.93))
    paths = save_figure(fig, output_dir, "vel_x_and_vx_error_three_way")
    plt.close(fig)
    return paths


def format_float(value: float | None) -> str:
    if value is None:
        return "NA"
    return f"{value:.6f}"


def markdown_table(summary_rows: list[tuple[str, dict[str, float | None]]]) -> list[str]:
    lines = [
        "| policy | mean_vel_x_pre_settled | mean_vel_x_post_fault | mean_abs_vx_error_pre_settled | mean_abs_vx_error_post_fault | velocity_retention | torso_height_failure_rate | timeout_rate |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, values in summary_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    label,
                    format_float(values["mean_vel_x_pre_settled"]),
                    format_float(values["mean_vel_x_post_fault"]),
                    format_float(values["mean_abs_vx_error_pre_settled"]),
                    format_float(values["mean_abs_vx_error_post_fault"]),
                    format_float(values["velocity_retention"]),
                    format_float(values["torso_height_failure_rate"]),
                    format_float(values["timeout_rate"]),
                ]
            )
            + " |"
        )
    return lines


def write_readme(
    output_dir: Path,
    *,
    args: argparse.Namespace,
    series_items: list[PolicySeries],
    summary_rows: list[tuple[str, dict[str, float | None]]],
    generated_paths: list[Path],
) -> Path:
    lines = [
        "# T10 Settled-Regime P2 Velocity Three-Way Comparison",
        "",
        "## Input Directories",
        "",
        f"- A0 Healthy PPO: `{repo_relative(args.a0_dir)}`",
        f"- A1-F Random Teacher: `{repo_relative(args.random_teacher_dir)}`",
        f"- A1-F Curriculum Teacher: `{repo_relative(args.curriculum_teacher_dir)}`",
        "",
        "## Protocol Summary",
        "",
        f"- `vx_cmd`: `{args.target_vx}` m/s",
        "- `num_envs`: `128`",
        "- `num_steps`: `1000`",
        "- `seed`: `0`",
        "- P2 fault: `front_left_foot` joint lock",
        f"- fault onset: step `{args.fault_onset_step}`",
        "- evaluator onset mode: `random_uniform` with `min=max=300`",
        "- semantics: `simulation_joint_state_override_lock`",
        "- fallback: disabled",
        "",
        "## Interpretation Notes",
        "",
        "- `velocity_timeseries.csv` stores per-step aggregate metrics.",
        "- `mean_vel_x` curves are averaged across 128 parallel environments per step, not single-trajectory traces.",
        "- The rollout is 1000 steps over 128 parallel environments; it is not 1000 independent tests.",
        "- A0 is the deployment-facing healthy PPO baseline.",
        "- A1-F teachers are privileged/reference policies and are not deployment-facing.",
        "",
        "## CSV Policy Labels",
        "",
    ]
    for series in series_items:
        lines.append(f"- {series.spec.label}: `{', '.join(series.csv_policy_labels)}`")
    lines.extend(
        [
            "",
            "## Settled-Window Summary",
            "",
            f"Pre-fault settled window: steps `{PRE_SETTLED_START}-{PRE_SETTLED_END}`.",
            f"Post-fault window: steps `{POST_FAULT_START}-{POST_FAULT_END}`.",
            "",
            *markdown_table(summary_rows),
            "",
            "## Generated Files",
            "",
        ]
    )
    for path in generated_paths:
        lines.append(f"- `{path.name}`")
    readme_path = output_dir / "README.md"
    readme_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return readme_path


def main() -> int:
    args = build_parser().parse_args()
    validate_args(args)
    plt = require_matplotlib()
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
    output_dir = resolve_repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    specs = [
        PolicyInput("a0", "A0 Healthy PPO", resolve_repo_path(args.a0_dir), "#1f77b4"),
        PolicyInput("random_teacher", "A1-F Random Teacher", resolve_repo_path(args.random_teacher_dir), "#d62728"),
        PolicyInput(
            "curriculum_teacher",
            "A1-F Curriculum Teacher",
            resolve_repo_path(args.curriculum_teacher_dir),
            "#2ca02c",
        ),
    ]
    series_items = [load_policy_series(spec, target_vx=args.target_vx) for spec in specs]
    generated_paths: list[Path] = []
    generated_paths.extend(
        plot_velocity(
            plt,
            series_items,
            output_dir=output_dir,
            fault_onset_step=args.fault_onset_step,
            target_vx=args.target_vx,
        )
    )
    generated_paths.extend(
        plot_error(
            plt,
            series_items,
            output_dir=output_dir,
            fault_onset_step=args.fault_onset_step,
        )
    )
    generated_paths.extend(
        plot_combined(
            plt,
            series_items,
            output_dir=output_dir,
            fault_onset_step=args.fault_onset_step,
            target_vx=args.target_vx,
        )
    )
    summary_rows = [(series.spec.label, settled_summary(series)) for series in series_items]
    readme_path = write_readme(
        output_dir,
        args=args,
        series_items=series_items,
        summary_rows=summary_rows,
        generated_paths=generated_paths,
    )
    print("[T10 SETTLED PLOTTER] Generated files:")
    for path in [*generated_paths, readme_path]:
        print(f"  {repo_relative(path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
