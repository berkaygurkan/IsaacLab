#!/usr/bin/env python3
"""Plot T12 current-fault final-eval velocity traces."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_ROOT = Path("papers/conference/results/t12_current_fault_final_eval")
TARGET_VX = 1.0
DEFAULT_RANDOM_ONSET_MIN = 30
DEFAULT_RANDOM_ONSET_MAX = 150

PROTOCOL_DIRS = {
    "random": "random_onset",
    "settled": "settled_onset",
}

POLICY_LABELS = {
    "a0": "A0 Healthy",
    "a1f_teacher": "A1-F Teacher privileged",
    "a2_single_step": "A2 Single-step",
    "a2_history_h16": "A2-History H16",
    "a5_alpha_000": "A5 alpha=0",
    "a5_alpha_025": "A5 alpha=0.25",
    "a5_alpha_050": "A5 alpha=0.5",
    "a5_alpha_075": "A5 alpha=0.75",
    "a5_alpha_100": "A5 alpha=1",
    "a7_alpha_000": "A7 alpha=0",
    "a7_alpha_001": "A7 alpha=0.01",
    "a7_alpha_0025": "A7 alpha=0.025",
    "a7_alpha_005": "A7 alpha=0.05",
    "a7_alpha_010": "A7 alpha=0.10",
    "a7_alpha_025": "A7 alpha=0.25",
}

BASE_CURVES = ("a0", "a1f_teacher", "a2_history_h16")


@dataclass
class Curve:
    protocol: str
    policy_dir: str
    label: str
    steps: list[float]
    vx: list[float]
    vx_error: list[float]
    residual_norm: list[float]


def resolve_repo_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot T12 current-fault final-eval traces.")
    parser.add_argument("--input_root", default=DEFAULT_INPUT_ROOT.as_posix())
    parser.add_argument("--protocol", default="both", choices=("random", "settled", "both"))
    parser.add_argument("--include_all", action="store_true")
    parser.add_argument("--output_dir", default=None)
    return parser


def selected_protocols(protocol: str) -> list[str]:
    if protocol == "both":
        return ["random", "settled"]
    return [protocol]


def read_metrics(input_root: Path) -> list[dict[str, str]]:
    metrics_path = input_root / "tables" / "metrics_summary.csv"
    if not metrics_path.is_file():
        return []
    with metrics_path.open(newline="") as file:
        return list(csv.DictReader(file))


def best_residual_dirs(metrics: list[dict[str, str]], protocol_dir: str) -> set[str]:
    selected: set[str] = set()
    for family in ("a5", "a7"):
        candidates = []
        for row in metrics:
            if row.get("protocol") != protocol_dir or row.get("policy") != family:
                continue
            try:
                error = float(row.get("mean_abs_vx_error_post_fault", "nan"))
            except ValueError:
                continue
            output_dir = row.get("output_dir", "")
            if output_dir:
                candidates.append((error, Path(output_dir).name))
        if candidates:
            candidates.sort(key=lambda item: item[0])
            selected.add(candidates[0][1])
    return selected


def wanted_policy_dirs(input_root: Path, protocol_dir: str, include_all: bool, metrics: list[dict[str, str]]) -> list[str]:
    protocol_root = input_root / protocol_dir
    if include_all:
        return [path.name for path in sorted(protocol_root.iterdir()) if path.is_dir()]
    wanted = set(BASE_CURVES)
    wanted.update(best_residual_dirs(metrics, protocol_dir))
    return [name for name in POLICY_LABELS if name in wanted]


def numeric(row: dict[str, str], *names: str) -> float | None:
    for name in names:
        value = row.get(name)
        if value in (None, ""):
            continue
        try:
            return float(value)
        except ValueError:
            continue
    return None


def read_curve(input_root: Path, protocol_dir: str, policy_dir: str) -> Curve | None:
    csv_path = input_root / protocol_dir / policy_dir / "velocity_timeseries.csv"
    if not csv_path.is_file():
        return None
    steps: list[float] = []
    vx: list[float] = []
    vx_error: list[float] = []
    residual_norm: list[float] = []
    with csv_path.open(newline="") as file:
        for row in csv.DictReader(file):
            step = numeric(row, "step")
            mean_vx = numeric(row, "mean_vel_x")
            mean_error = numeric(row, "mean_abs_vx_error")
            residual = numeric(row, "residual_action_mean_norm", "mean_residual_action_norm", "residual_norm")
            if step is None or mean_vx is None or mean_error is None:
                continue
            steps.append(step)
            vx.append(mean_vx)
            vx_error.append(mean_error)
            if residual is not None:
                residual_norm.append(residual)
    if not steps:
        return None
    return Curve(
        protocol=protocol_dir,
        policy_dir=policy_dir,
        label=POLICY_LABELS.get(policy_dir, policy_dir),
        steps=steps,
        vx=vx,
        vx_error=vx_error,
        residual_norm=residual_norm,
    )


def load_curves(input_root: Path, protocol_key: str, include_all: bool, metrics: list[dict[str, str]]) -> list[Curve]:
    protocol_dir = PROTOCOL_DIRS[protocol_key]
    curves = []
    for policy_dir in wanted_policy_dirs(input_root, protocol_dir, include_all, metrics):
        curve = read_curve(input_root, protocol_dir, policy_dir)
        if curve is not None:
            curves.append(curve)
    return curves


def infer_random_onset_bounds(input_root: Path) -> tuple[int, int]:
    protocol_root = input_root / PROTOCOL_DIRS["random"]
    if not protocol_root.is_dir():
        return DEFAULT_RANDOM_ONSET_MIN, DEFAULT_RANDOM_ONSET_MAX

    for rollout_path in sorted(protocol_root.glob("*/rollout_metrics.csv")):
        with rollout_path.open(newline="") as file:
            for row in csv.DictReader(file):
                onset_min = numeric(row, "fault_onset_step_min")
                onset_max = numeric(row, "fault_onset_step_max")
                if onset_min is None or onset_max is None:
                    continue
                return int(onset_min), int(onset_max)
    return DEFAULT_RANDOM_ONSET_MIN, DEFAULT_RANDOM_ONSET_MAX


def apply_onset_marker(ax: object, protocol_key: str, random_onset_bounds: tuple[int, int]) -> None:
    if protocol_key == "random":
        onset_min, onset_max = random_onset_bounds
        ax.axvspan(
            onset_min,
            onset_max,
            color="0.85",
            alpha=0.5,
            label=f"P2 onset U({onset_min},{onset_max})",
        )
    else:
        ax.axvline(300, color="0.35", linestyle="--", linewidth=1.2, label="P2 settled onset step 300")


def save_figure(fig: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")


def plot_protocol(
    curves: list[Curve],
    protocol_key: str,
    output_dir: Path,
    *,
    random_onset_bounds: tuple[int, int],
) -> None:
    import matplotlib.pyplot as plt

    if not curves:
        raise RuntimeError(f"no completed curves found for protocol: {protocol_key}")
    protocol_name = PROTOCOL_DIRS[protocol_key].replace("_", " ")

    fig, ax = plt.subplots(figsize=(11, 6))
    apply_onset_marker(ax, protocol_key, random_onset_bounds)
    ax.axhline(TARGET_VX, color="black", linestyle=":", linewidth=1.2, label="target vx=1.0")
    for curve in curves:
        ax.plot(curve.steps, curve.vx, linewidth=1.6, label=curve.label)
    ax.set_title(f"T12 {protocol_name}: vx over time")
    ax.set_xlabel("step")
    ax.set_ylabel("mean vx")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    save_figure(fig, output_dir / f"{protocol_key}_vx_time_comparison.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 6))
    apply_onset_marker(ax, protocol_key, random_onset_bounds)
    ax.axhline(0.0, color="black", linestyle=":", linewidth=1.2, label="zero error")
    for curve in curves:
        ax.plot(curve.steps, curve.vx_error, linewidth=1.6, label=curve.label)
    ax.set_title(f"T12 {protocol_name}: absolute vx error over time")
    ax.set_xlabel("step")
    ax.set_ylabel("mean |vx - target|")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    save_figure(fig, output_dir / f"{protocol_key}_vx_error_time_comparison.png")
    plt.close(fig)


def plot_residual_norm(all_curves: Iterable[Curve], output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    residual_curves = [curve for curve in all_curves if curve.residual_norm]
    if not residual_curves:
        raise RuntimeError("no residual norm columns found in completed velocity_timeseries.csv files")
    fig, ax = plt.subplots(figsize=(11, 6))
    for curve in residual_curves:
        steps = curve.steps[: len(curve.residual_norm)]
        ax.plot(steps, curve.residual_norm, linewidth=1.5, label=f"{curve.protocol}: {curve.label}")
    ax.set_title("T12 residual action norm over time")
    ax.set_xlabel("step")
    ax.set_ylabel("residual action mean norm")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    save_figure(fig, output_dir / "residual_norm_time_comparison.png")
    plt.close(fig)


def main() -> int:
    args = build_parser().parse_args()
    input_root = resolve_repo_path(args.input_root)
    output_dir = resolve_repo_path(args.output_dir) if args.output_dir else input_root / "plots"
    try:
        import matplotlib  # noqa: F401
    except ImportError as exc:
        raise SystemExit("matplotlib is required for T12 plotting; no fake plots were created") from exc

    metrics = read_metrics(input_root)
    random_onset_bounds = infer_random_onset_bounds(input_root)
    all_curves: list[Curve] = []
    for protocol_key in selected_protocols(args.protocol):
        curves = load_curves(input_root, protocol_key, args.include_all, metrics)
        plot_protocol(curves, protocol_key, output_dir, random_onset_bounds=random_onset_bounds)
        all_curves.extend(curves)
    plot_residual_norm(all_curves, output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
