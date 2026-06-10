#!/usr/bin/env python3
"""Offline T09-R2l P2 quick sanity velocity plotter.

This utility reads existing quick-demo CSV files and saves advisor-facing
figures only. It does not launch Isaac Sim, train, freeze checkpoints, or update
paper-grade result manifests.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = "papers/conference/demo/advisor_meeting_2026_06_09/figures"
VELOCITY_COLUMNS = ("mean_vel_x", "mean_base_lin_vel_x", "base_lin_vel_x_mean")


class PlotError(ValueError):
    """Raised for invalid offline velocity plotting inputs."""


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
        return float(text)
    except ValueError:
        return None


def read_velocity_rows(csv_path: Path, label: str) -> tuple[list[int], list[float], str]:
    if not csv_path.is_file():
        raise PlotError(f"{label} CSV does not exist: {repo_relative(csv_path)}")
    with csv_path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise PlotError(f"{label} CSV has no header: {repo_relative(csv_path)}")
        velocity_column = next((name for name in VELOCITY_COLUMNS if name in reader.fieldnames), None)
        if velocity_column is None:
            raise PlotError(
                f"{label} CSV has no velocity column. Expected one of {VELOCITY_COLUMNS}; "
                f"found {reader.fieldnames}."
            )
        steps: list[int] = []
        velocities: list[float] = []
        missing_count = 0
        for row in reader:
            step_value = parse_float(row.get("step"))
            velocity_value = parse_float(row.get(velocity_column))
            if step_value is None:
                continue
            if velocity_value is None:
                missing_count += 1
                continue
            steps.append(int(step_value))
            velocities.append(float(velocity_value))
    if not velocities:
        raise PlotError(f"{label} CSV velocity column {velocity_column!r} contains no numeric values.")
    if missing_count:
        print(f"[T09-R2l WARNING] {label}: skipped {missing_count} rows with missing velocity.")
    return steps, velocities, velocity_column


def mean_for_window(steps: list[int], values: list[float], *, fault_onset_step: int, post_fault: bool) -> float | None:
    selected = [
        value
        for step, value in zip(steps, values)
        if (step >= fault_onset_step if post_fault else step < fault_onset_step)
    ]
    if not selected:
        return None
    return float(statistics.mean(selected))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot A0 vs A1-F quick P2 forward velocity demo curves.")
    parser.add_argument("--a0_csv", required=True)
    parser.add_argument("--a1f_csv", required=True)
    parser.add_argument("--a0_label", default="A0 healthy PPO under P2")
    parser.add_argument("--a1f_label", default="A1-F random-onset privileged teacher under P2")
    parser.add_argument("--fault_onset_step", type=int, default=50)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output_name", default="quick_p2_velocity_a0_vs_a1f")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.fault_onset_step < 0:
        raise PlotError("--fault_onset_step must be non-negative.")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    a0_csv = resolve_repo_path(args.a0_csv)
    a1f_csv = resolve_repo_path(args.a1f_csv)
    output_dir = resolve_repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    a0_steps, a0_velocities, a0_velocity_column = read_velocity_rows(a0_csv, args.a0_label)
    a1f_steps, a1f_velocities, a1f_velocity_column = read_velocity_rows(a1f_csv, args.a1f_label)

    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    ax.plot(a0_steps, a0_velocities, label=args.a0_label, linewidth=2.0)
    ax.plot(a1f_steps, a1f_velocities, label=args.a1f_label, linewidth=2.0)
    ax.axvline(
        args.fault_onset_step,
        color="black",
        linestyle="--",
        linewidth=1.2,
        label=f"P2 onset step {args.fault_onset_step}",
    )
    ax.set_xlabel("simulation step")
    ax.set_ylabel("mean forward velocity vx")
    ax.set_title(
        "Quick P2 Sanity Demo: Forward Velocity Before/After Joint Lock\n"
        "quick sanity only, not paper-grade; A1-F is privileged teacher"
    )
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()

    png_path = output_dir / f"{args.output_name}.png"
    pdf_path = output_dir / f"{args.output_name}.pdf"
    summary_path = output_dir / f"{args.output_name}_summary.json"
    fig.savefig(png_path, dpi=180)
    pdf_written = False
    try:
        fig.savefig(pdf_path)
        pdf_written = True
    except Exception as exc:
        print(f"[T09-R2l WARNING] PDF export failed: {exc}")
    plt.close(fig)

    a0_pre = mean_for_window(a0_steps, a0_velocities, fault_onset_step=args.fault_onset_step, post_fault=False)
    a0_post = mean_for_window(a0_steps, a0_velocities, fault_onset_step=args.fault_onset_step, post_fault=True)
    a1f_pre = mean_for_window(a1f_steps, a1f_velocities, fault_onset_step=args.fault_onset_step, post_fault=False)
    a1f_post = mean_for_window(a1f_steps, a1f_velocities, fault_onset_step=args.fault_onset_step, post_fault=True)
    summary = {
        "demo_scope": "quick_p2_velocity_sanity_plot",
        "not_paper_grade": True,
        "note": "advisor-facing mini demo only; A1-F is privileged and not deployment-facing",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "fault_onset_step": args.fault_onset_step,
        "a0_csv": repo_relative(a0_csv),
        "a1f_csv": repo_relative(a1f_csv),
        "a0_label": args.a0_label,
        "a1f_label": args.a1f_label,
        "a0_velocity_column": a0_velocity_column,
        "a1f_velocity_column": a1f_velocity_column,
        "a0_mean_vel_x_pre_fault": a0_pre,
        "a0_mean_vel_x_post_fault": a0_post,
        "a0_delta_vel_x_post_minus_pre": None if a0_pre is None or a0_post is None else a0_post - a0_pre,
        "a1f_mean_vel_x_pre_fault": a1f_pre,
        "a1f_mean_vel_x_post_fault": a1f_post,
        "a1f_delta_vel_x_post_minus_pre": None if a1f_pre is None or a1f_post is None else a1f_post - a1f_pre,
        "png_path": repo_relative(png_path),
        "pdf_path": repo_relative(pdf_path) if pdf_written else None,
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("[T09-R2l] velocity plot written")
    print(f"  png_path: {repo_relative(png_path)}")
    print(f"  pdf_path: {repo_relative(pdf_path) if pdf_written else 'not_written'}")
    print(f"  summary_path: {repo_relative(summary_path)}")
    print("  not_paper_grade: True")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
