#!/usr/bin/env python3
"""Offline controlled P2 velocity mini-demo plotter.

Reads controlled_single_rollout CSV files and saves advisor-facing velocity
figures only. It does not launch Isaac Sim or update any paper-grade artifact.
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
PLOT_MODES = ("representative_env", "mean")


class ControlledPlotError(ValueError):
    """Raised for invalid controlled velocity plotting inputs."""


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


def read_rows(csv_path: Path, label: str, *, plot_mode: str) -> tuple[list[int], list[float], float | None, str]:
    if not csv_path.is_file():
        raise ControlledPlotError(f"{label} CSV does not exist: {repo_relative(csv_path)}")
    with csv_path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ControlledPlotError(f"{label} CSV has no header: {repo_relative(csv_path)}")
        value_column = "representative_env_vel_x" if plot_mode == "representative_env" else "mean_vel_x"
        if value_column not in reader.fieldnames:
            raise ControlledPlotError(
                f"{label} CSV missing required column {value_column!r}; found {reader.fieldnames}."
            )
        steps: list[int] = []
        values: list[float] = []
        target_vx_values: list[float] = []
        missing_count = 0
        for row in reader:
            step = parse_float(row.get("step"))
            velocity = parse_float(row.get(value_column))
            if step is None:
                continue
            if velocity is None:
                missing_count += 1
                continue
            steps.append(int(step))
            values.append(float(velocity))
            target_vx = parse_float(row.get("target_vx"))
            if target_vx is not None:
                target_vx_values.append(target_vx)
    if not values:
        raise ControlledPlotError(f"{label} CSV column {value_column!r} contains no numeric values.")
    if missing_count:
        print(f"[T09-R2m WARNING] {label}: skipped {missing_count} rows with missing {value_column}.")
    target_vx_value = None
    if target_vx_values:
        unique = sorted(set(round(value, 8) for value in target_vx_values))
        if len(unique) == 1:
            target_vx_value = float(target_vx_values[0])
        else:
            print(f"[T09-R2m WARNING] {label}: target_vx varies in CSV; horizontal target line skipped.")
    return steps, values, target_vx_value, value_column


def mean_window(steps: list[int], values: list[float], *, start_step: int, end_step: int) -> float | None:
    selected = [value for step, value in zip(steps, values) if start_step <= step < end_step]
    if not selected:
        return None
    return float(statistics.mean(selected))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot controlled A0/A1-F P2 velocity traces around joint lock.")
    parser.add_argument("--a0_csv", required=True)
    parser.add_argument("--a1f_csv", required=True)
    parser.add_argument("--fault_onset_step", type=int, default=250)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output_name", default="controlled_p2_velocity_a0_vs_a1f")
    parser.add_argument("--plot_mode", default="representative_env", choices=PLOT_MODES)
    parser.add_argument("--a0_label", default="A0 healthy PPO")
    parser.add_argument("--a1f_label", default="A1-F privileged teacher")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.fault_onset_step < 0:
        raise ControlledPlotError("--fault_onset_step must be non-negative.")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    a0_csv = resolve_repo_path(args.a0_csv)
    a1f_csv = resolve_repo_path(args.a1f_csv)
    output_dir = resolve_repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    a0_steps, a0_values, a0_target_vx, a0_column = read_rows(a0_csv, args.a0_label, plot_mode=args.plot_mode)
    a1f_steps, a1f_values, a1f_target_vx, a1f_column = read_rows(a1f_csv, args.a1f_label, plot_mode=args.plot_mode)
    target_vx = a0_target_vx if a0_target_vx is not None and a0_target_vx == a1f_target_vx else None

    fig, ax = plt.subplots(figsize=(9.2, 5.3))
    ax.plot(a0_steps, a0_values, linewidth=2.0, label=args.a0_label)
    ax.plot(a1f_steps, a1f_values, linewidth=2.0, label=args.a1f_label)
    ax.axvline(
        args.fault_onset_step,
        color="black",
        linestyle="--",
        linewidth=1.2,
        label=f"P2 onset step {args.fault_onset_step}",
    )
    if target_vx is not None:
        ax.axhline(
            target_vx,
            color="gray",
            linestyle=":",
            linewidth=1.2,
            label=f"target vx {target_vx:g}",
        )
    ax.set_xlabel("simulation step")
    ax.set_ylabel("forward velocity vx")
    ax.set_title(
        "Controlled P2 Demo: Forward Velocity Around Joint Lock\n"
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
        print(f"[T09-R2m WARNING] PDF export failed: {exc}")
    plt.close(fig)

    pre_start = max(1, args.fault_onset_step - 50)
    pre_end = args.fault_onset_step
    post_start = args.fault_onset_step
    post_end = args.fault_onset_step + 100
    a0_pre = mean_window(a0_steps, a0_values, start_step=pre_start, end_step=pre_end)
    a0_post = mean_window(a0_steps, a0_values, start_step=post_start, end_step=post_end)
    a1f_pre = mean_window(a1f_steps, a1f_values, start_step=pre_start, end_step=pre_end)
    a1f_post = mean_window(a1f_steps, a1f_values, start_step=post_start, end_step=post_end)
    summary = {
        "demo_scope": "controlled_p2_velocity_sanity_plot",
        "not_paper_grade": True,
        "note": "advisor-facing mini demo only; A1-F is privileged and not deployment-facing",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "plot_mode": args.plot_mode,
        "fault_onset_step": args.fault_onset_step,
        "target_vx_plotted": target_vx,
        "a0_csv": repo_relative(a0_csv),
        "a1f_csv": repo_relative(a1f_csv),
        "a0_column": a0_column,
        "a1f_column": a1f_column,
        "pre_fault_window": [pre_start, pre_end],
        "post_fault_window": [post_start, post_end],
        "a0_mean_vel_x_pre_fault_window": a0_pre,
        "a0_mean_vel_x_post_fault_window": a0_post,
        "a0_delta_vel_x_post_minus_pre_window": None if a0_pre is None or a0_post is None else a0_post - a0_pre,
        "a1f_mean_vel_x_pre_fault_window": a1f_pre,
        "a1f_mean_vel_x_post_fault_window": a1f_post,
        "a1f_delta_vel_x_post_minus_pre_window": None
        if a1f_pre is None or a1f_post is None
        else a1f_post - a1f_pre,
        "png_path": repo_relative(png_path),
        "pdf_path": repo_relative(pdf_path) if pdf_written else None,
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("[T09-R2m] controlled velocity plot written")
    print(f"  png_path: {repo_relative(png_path)}")
    print(f"  pdf_path: {repo_relative(pdf_path) if pdf_written else 'not_written'}")
    print(f"  summary_path: {repo_relative(summary_path)}")
    print("  not_paper_grade: True")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
