#!/usr/bin/env python3
"""Plot T17 fixed-vx multi-joint P2 fault-response velocity traces.

The plotter reads existing result artifacts only. It does not launch Isaac,
run evaluation, train, or modify checkpoints/datasets.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_INPUT_ROOT = "papers/conference/results/t17_multijoint_closed_loop_eval_a2h_a5_alpha_sweep_fixed_vx1_seed0"
DEFAULT_A2_SINGLE_STEP_ROOT = "papers/conference/results/t17_multijoint_closed_loop_eval_a2_single_step_fixed_vx1_seed0"
DEFAULT_TEACHER_ROOT = "papers/conference/results/t13c_multijoint_teacher_eval_fixed_vx_guarded"
DEFAULT_OUTPUT_DIR = "papers/conference/results/t17_closed_loop_report_package"
DEFAULT_PROTOCOLS = ("realistic_random", "late_random")
VELOCITY_MODE = "fixed_vx_1p0"
TARGET_VX = 1.0

RUN_TABLE_FIELDS = [
    "method",
    "protocol",
    "velocity_mode",
    "alpha",
    "mean_vel_x_post_fault",
    "mean_abs_vx_error_post_fault",
    "p90_abs_vx_error_post_fault",
    "source",
    "velocity_timeseries_csv",
]


@dataclass
class MethodSpec:
    method: str
    label: str
    alpha: str
    kind: str
    run_subdir: str
    color: str
    linestyle: str
    linewidth: float
    zorder: int


@dataclass
class RunArtifact:
    spec: MethodSpec
    protocol: str
    source: str
    run_dir: Path
    velocity_csv: Path
    rollout_csv: Path | None
    summary_json: Path | None
    table_metrics: dict[str, str]
    summary: dict[str, Any]
    time_rows: list[dict[str, str]]


METHOD_SPECS = [
    MethodSpec("teacher_reference", "A1-F Teacher (priv.)", "", "teacher", "", "#333333", "--", 2.0, 7),
    MethodSpec("a2_single_step", "A2 Single-Step", "", "a2_single_step", "a2_single_step", "#2ca02c", "-.", 2.4, 7),
    MethodSpec("a2_history", "A2-History H16", "", "student", "a2_history", "#1f77b4", "-", 2.0, 4),
    MethodSpec("a5_alpha_0p25", "A5 alpha=0.25", "0.25", "student", "a5_alpha_0p25", "#ffb000", "-", 2.0, 5),
    MethodSpec("a5_alpha_0p5", "A5 alpha=0.5", "0.5", "student", "a5_alpha_0p5", "#ff7f0e", "-", 2.2, 6),
    MethodSpec("a5_alpha_1", "A5 alpha=1.0", "1.0", "student", "a5_alpha_1", "#d62728", "-", 3.0, 8),
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


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def format_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.9g}"
    text = str(value)
    if text.lower() == "none":
        return ""
    return text


def write_md_table(path: Path, rows: list[dict[str, Any]], fields: list[str], title: str) -> None:
    lines = [
        f"# {title}",
        "",
        "Preliminary advisor-facing figure support table; not final paper-grade statistics.",
        "",
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join("---" for _ in fields) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(format_value(row.get(field, "")) for field in fields) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def normalize_alpha(value: Any) -> str:
    text = str(value if value is not None else "").strip()
    if text in {"", "None", "none"}:
        return ""
    try:
        return f"{float(text):g}"
    except ValueError:
        return text


def metric_key(method: str, protocol: str, velocity_mode: str, alpha: str) -> tuple[str, str, str, str]:
    return (method, protocol, velocity_mode, normalize_alpha(alpha))


def load_t17_metrics(input_root: Path) -> dict[tuple[str, str, str, str], dict[str, str]]:
    rows = read_csv(input_root / "per_run_metrics.csv")
    metrics: dict[tuple[str, str, str, str], dict[str, str]] = {}
    for row in rows:
        policy = row.get("policy", "")
        alpha = normalize_alpha(row.get("alpha", ""))
        if policy == "a5":
            method = f"a5_alpha_{alpha.replace('.', 'p')}"
        else:
            method = policy
        key = metric_key(method, row.get("protocol", ""), row.get("velocity_mode", ""), alpha)
        metrics[key] = row
    return metrics


def load_teacher_metrics(teacher_root: Path) -> dict[tuple[str, str, str, str], dict[str, str]]:
    rows = read_csv(teacher_root / "tables" / "metrics_summary.csv")
    metrics: dict[tuple[str, str, str, str], dict[str, str]] = {}
    for row in rows:
        key = metric_key("teacher_reference", row.get("protocol", ""), row.get("velocity_mode", ""), "")
        metrics[key] = row
    return metrics


def run_dir_for(
    spec: MethodSpec,
    protocol: str,
    input_root: Path,
    a2_single_step_root: Path,
    teacher_root: Path,
) -> tuple[Path, str]:
    if spec.kind == "teacher":
        embedded_t17_dir = teacher_root / protocol / VELOCITY_MODE / "teacher_reference"
        if embedded_t17_dir.is_dir():
            return embedded_t17_dir, repo_relative(teacher_root)
        return teacher_root / protocol / VELOCITY_MODE, repo_relative(teacher_root)
    if spec.kind == "a2_single_step":
        return a2_single_step_root / protocol / VELOCITY_MODE / spec.run_subdir, repo_relative(a2_single_step_root)
    return input_root / protocol / VELOCITY_MODE / spec.run_subdir, repo_relative(input_root)


def load_run_artifacts(
    input_root: Path,
    a2_single_step_root: Path,
    teacher_root: Path,
    protocols: list[str],
    method_specs: list[MethodSpec] | None = None,
) -> tuple[list[RunArtifact], list[str]]:
    method_specs = method_specs or METHOD_SPECS
    t17_metrics = load_t17_metrics(input_root)
    a2_single_step_metrics = load_t17_metrics(a2_single_step_root)
    teacher_metrics = load_teacher_metrics(teacher_root)
    artifacts: list[RunArtifact] = []
    missing: list[str] = []
    for protocol in protocols:
        for spec in method_specs:
            run_dir, source = run_dir_for(spec, protocol, input_root, a2_single_step_root, teacher_root)
            velocity_csv = run_dir / "velocity_timeseries.csv"
            rollout_csv = run_dir / "rollout_metrics.csv"
            summary_json = run_dir / "summary.json"
            if not velocity_csv.is_file():
                missing.append(f"{spec.method} / {protocol}: missing {repo_relative(velocity_csv)}")
                continue
            if spec.kind == "teacher":
                table_metrics = teacher_metrics.get(
                    metric_key(spec.method, protocol, VELOCITY_MODE, ""),
                    t17_metrics.get(metric_key(spec.method, protocol, VELOCITY_MODE, ""), {}),
                )
            elif spec.kind == "a2_single_step":
                table_metrics = a2_single_step_metrics.get(metric_key(spec.method, protocol, VELOCITY_MODE, ""), {})
            else:
                table_metrics = t17_metrics.get(metric_key(spec.method, protocol, VELOCITY_MODE, spec.alpha), {})
            artifacts.append(
                RunArtifact(
                    spec=spec,
                    protocol=protocol,
                    source=source,
                    run_dir=run_dir,
                    velocity_csv=velocity_csv,
                    rollout_csv=rollout_csv if rollout_csv.is_file() else None,
                    summary_json=summary_json if summary_json.is_file() else None,
                    table_metrics=table_metrics,
                    summary=read_json(summary_json),
                    time_rows=read_csv(velocity_csv),
                )
            )
    return artifacts, missing


def method_specs_for_args(args: argparse.Namespace) -> list[MethodSpec]:
    specs: list[MethodSpec] = []
    for spec in METHOD_SPECS:
        if args.exclude_teacher_reference and spec.kind == "teacher":
            continue
        if args.exclude_a2_single_step and spec.kind == "a2_single_step":
            continue
        if spec.method == "a2_history":
            specs.append(
                MethodSpec(
                    spec.method,
                    f"A2-History {args.history_label}",
                    spec.alpha,
                    spec.kind,
                    spec.run_subdir,
                    spec.color,
                    spec.linestyle,
                    spec.linewidth,
                    spec.zorder,
                )
            )
            continue
        specs.append(spec)
    return specs


def velocity_value(row: dict[str, str]) -> float | None:
    value = row.get("mean_vel_x", "")
    if value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def step_value(row: dict[str, str]) -> int | None:
    value = row.get("step", "")
    try:
        return int(float(value))
    except ValueError:
        return None


def inspect_alignment(artifacts: list[RunArtifact]) -> dict[str, Any]:
    varied_runs: list[str] = []
    synchronized_runs: list[str] = []
    per_env_onset_available = False
    per_env_velocity_available = False
    onset_ranges: dict[str, tuple[int | None, int | None]] = {}

    for artifact in artifacts:
        summary = artifact.summary
        onset_values = summary.get("initial_onset_step_by_env")
        if isinstance(onset_values, list) and onset_values:
            per_env_onset_available = True
            numeric = [int(value) for value in onset_values if isinstance(value, (int, float))]
            if numeric and len(set(numeric)) > 1:
                varied_runs.append(f"{artifact.spec.method}/{artifact.protocol}")
            elif numeric:
                synchronized_runs.append(f"{artifact.spec.method}/{artifact.protocol}")
        onset_min = summary.get("fault_onset_step_min")
        onset_max = summary.get("fault_onset_step_max")
        try:
            onset_min_int = int(float(onset_min))
        except (TypeError, ValueError):
            onset_min_int = None
        try:
            onset_max_int = int(float(onset_max))
        except (TypeError, ValueError):
            onset_max_int = None
        if onset_min_int is not None and onset_max_int is not None:
            onset_ranges[artifact.protocol] = (onset_min_int, onset_max_int)
            if onset_min_int != onset_max_int:
                varied_runs.append(f"{artifact.spec.method}/{artifact.protocol}")
        if artifact.time_rows:
            headers = set(artifact.time_rows[0].keys())
            if {"env_id", "velocity_x"}.issubset(headers) or {"env_index", "velocity_x"}.issubset(headers):
                per_env_velocity_available = True

    fault_onset_varies = bool(varied_runs)
    fault_aligned_supported = bool(fault_onset_varies and per_env_onset_available and per_env_velocity_available)
    plot_mode = "fault_aligned" if fault_aligned_supported else "absolute_time"
    if fault_onset_varies and not per_env_velocity_available:
        limitation = (
            "Fault onset varies across environments, and per-env onset values are present in summary.json, "
            "but velocity_timeseries.csv/rollout_metrics.csv contain aggregate mean traces rather than per-env "
            "velocity trajectories. Fault-aligned recovery curves cannot be reconstructed from current artifacts."
        )
    elif not fault_onset_varies:
        limitation = "Fault onset appears synchronized or fixed; absolute-time plotting is acceptable."
    else:
        limitation = "Fault-aligned plotting is supported by current artifacts."
    return {
        "plot_mode": plot_mode,
        "fault_onset_varies": fault_onset_varies,
        "per_env_onset_available": per_env_onset_available,
        "per_env_velocity_available": per_env_velocity_available,
        "fault_aligned_supported": fault_aligned_supported,
        "varied_runs": sorted(set(varied_runs)),
        "synchronized_runs": sorted(set(synchronized_runs)),
        "onset_ranges": onset_ranges,
        "limitation": limitation,
    }


def best_fixed_vx_alpha(rows: list[dict[str, Any]], protocol: str, metric: str = "mean_abs_vx_error_post_fault") -> str:
    candidates = [
        row
        for row in rows
        if row.get("protocol") == protocol and row.get("method", "").startswith("a5_alpha")
    ]
    if not candidates:
        return ""
    def score(row: dict[str, Any]) -> float:
        try:
            return float(row.get(metric, ""))
        except (TypeError, ValueError):
            return float("inf")
    return str(min(candidates, key=score).get("alpha", ""))


def best_deployment_method(rows: list[dict[str, Any]], protocol: str) -> str:
    candidates = [
        row
        for row in rows
        if row.get("protocol") == protocol and row.get("method") != "teacher_reference"
    ]
    if not candidates:
        return ""

    def score(row: dict[str, Any]) -> float:
        try:
            return float(row.get("mean_abs_vx_error_post_fault", ""))
        except (TypeError, ValueError):
            return float("inf")

    best = min(candidates, key=score)
    return f"{best.get('method', '')} (alpha={best.get('alpha', '')})".replace(" (alpha=)", "")


def build_plotted_run_rows(artifacts: list[RunArtifact]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for artifact in artifacts:
        metrics = artifact.table_metrics
        rows.append(
            {
                "method": artifact.spec.method,
                "protocol": artifact.protocol,
                "velocity_mode": VELOCITY_MODE,
                "alpha": artifact.spec.alpha,
                "mean_vel_x_post_fault": metrics.get("mean_vel_x_post_fault", ""),
                "mean_abs_vx_error_post_fault": metrics.get("mean_abs_vx_error_post_fault", ""),
                "p90_abs_vx_error_post_fault": metrics.get("p90_abs_vx_error_post_fault", ""),
                "source": artifact.source,
                "velocity_timeseries_csv": repo_relative(artifact.velocity_csv),
            }
        )
    return rows


def write_notes(
    *,
    output_dir: Path,
    artifacts: list[RunArtifact],
    missing: list[str],
    alignment: dict[str, Any],
    plotted_rows: list[dict[str, Any]],
    phase_label: str,
    notes_basename: str,
    summary_basename: str,
    include_a2_single_step: bool,
) -> None:
    included = [f"{artifact.spec.method}/{artifact.protocol}" for artifact in artifacts]
    best_by_protocol = {
        protocol: best_fixed_vx_alpha(plotted_rows, protocol)
        for protocol in DEFAULT_PROTOCOLS
    }
    best_deployment_by_protocol = {
        protocol: best_deployment_method(plotted_rows, protocol)
        for protocol in DEFAULT_PROTOCOLS
    }
    strongest_a5 = "A5 alpha=1.0"
    lines = [
        f"# {phase_label} Fixed-vx Fault Response Figure Notes",
        "",
        "Figure scope: preliminary advisor-facing multi-joint P2 fixed-vx velocity tracking; not final paper-grade statistics.",
        "",
        "## Included Methods",
        "",
    ]
    lines.extend(f"- {item}" for item in included)
    lines.extend(["", "## Missing Methods", ""])
    lines.extend(f"- {item}" for item in missing)
    lines.extend(
        [
            "",
            "## Alignment Audit",
            "",
            f"- plot_mode: `{alignment['plot_mode']}`",
            f"- fault_onset_varies: `{str(alignment['fault_onset_varies']).lower()}`",
            f"- per_env_onset_available: `{str(alignment['per_env_onset_available']).lower()}`",
            f"- per_env_velocity_available: `{str(alignment['per_env_velocity_available']).lower()}`",
            f"- fault_aligned_supported: `{str(alignment['fault_aligned_supported']).lower()}`",
            f"- limitation: {alignment['limitation']}",
            "",
            "## A5 Readout",
            "",
            f"- visually strongest A5 curve: `{strongest_a5}`",
            f"- best A5 alpha by fixed-vx mean post-fault error, realistic_random: `{best_by_protocol.get('realistic_random', '')}`",
            f"- best A5 alpha by fixed-vx mean post-fault error, late_random: `{best_by_protocol.get('late_random', '')}`",
            "",
            "## Interpretation / Caption Guidance",
            "",
            "- A5 improves the history-student baseline.",
            "- The A1-F teacher is privileged and non-deployment-facing.",
            f"- best deployment-facing method by fixed-vx mean post-fault error, realistic_random: `{best_deployment_by_protocol.get('realistic_random', '')}`",
            f"- best deployment-facing method by fixed-vx mean post-fault error, late_random: `{best_deployment_by_protocol.get('late_random', '')}`",
        ]
    )
    if include_a2_single_step:
        lines.extend(
            [
                "- A2 single-step is unexpectedly strong under fixed-vx=1.0.",
                "- The all-method fixed-vx figure should be interpreted as an ablation comparison, not as evidence that A5 dominates every deployment-facing baseline.",
                "- Suggested caption wording: A5 improves the history-student baseline, while A2 single-step remains a strong fixed-command baseline; further command-random comparison is required before choosing the final deployment-facing variant.",
            ]
        )
    else:
        lines.extend(
            [
                "- A2 single-step is intentionally excluded from this corrected residual-ablation figure.",
                "- Suggested caption wording: corrected 50 Hz / H50 comparison of A2-history and A5 residual policies under fixed-command multi-joint P2 faults; the privileged A1-F teacher is shown only as a non-deployment reference.",
            ]
        )
    (output_dir / f"{notes_basename}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    summary_lines = [
        f"{phase_label} fixed-vx fault-response figure summary",
        f"plot_mode: {alignment['plot_mode']}",
        f"fault_onset_varies: {alignment['fault_onset_varies']}",
        f"fault_aligned_supported: {alignment['fault_aligned_supported']}",
        f"visual_best_a5: {strongest_a5}",
        f"best_a5_alpha_fixed_vx_realistic_random: {best_by_protocol.get('realistic_random', '')}",
        f"best_a5_alpha_fixed_vx_late_random: {best_by_protocol.get('late_random', '')}",
        f"best_deployment_fixed_vx_realistic_random: {best_deployment_by_protocol.get('realistic_random', '')}",
        f"best_deployment_fixed_vx_late_random: {best_deployment_by_protocol.get('late_random', '')}",
        "missing: " + "; ".join(missing),
    ]
    if include_a2_single_step:
        summary_lines.append(
            "interpretation: A5 improves A2-history, but A2 single-step is the strongest fixed-command deployment-facing baseline in the current artifacts."
        )
    else:
        summary_lines.append(
            "interpretation: A2 single-step excluded; this package is the corrected history/residual ablation."
        )
    (output_dir / f"{summary_basename}.txt").write_text(
        "\n".join(summary_lines) + "\n",
        encoding="utf-8",
    )


def plot_figure(
    *,
    output_dir: Path,
    artifacts: list[RunArtifact],
    alignment: dict[str, Any],
    protocols: list[str],
    figure_basename: str,
    title: str | None,
    max_step: int | None,
) -> tuple[Path, Path]:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required to generate the T17 fixed-vx figure.") from exc

    fig, axes = plt.subplots(1, len(protocols), figsize=(12.0, 4.6), sharey=True)
    if len(protocols) == 1:
        axes = [axes]
    by_protocol: dict[str, list[RunArtifact]] = {protocol: [] for protocol in protocols}
    for artifact in artifacts:
        if artifact.protocol in by_protocol:
            by_protocol[artifact.protocol].append(artifact)

    for index, protocol in enumerate(protocols):
        ax = axes[index]
        protocol_artifacts = by_protocol.get(protocol, [])
        onset_min, onset_max = alignment.get("onset_ranges", {}).get(protocol, (None, None))
        if onset_min is not None and onset_max is not None:
            ax.axvspan(onset_min, onset_max, color="#d9d9d9", alpha=0.35, label=f"P2 onset U({onset_min},{onset_max})")
        ax.axhline(TARGET_VX, color="black", linestyle=":", linewidth=1.2, label="target vx=1.0")
        for artifact in protocol_artifacts:
            xs: list[int] = []
            ys: list[float] = []
            for row in artifact.time_rows:
                step = step_value(row)
                value = velocity_value(row)
                if step is None or value is None:
                    continue
                if max_step is not None and step > max_step:
                    continue
                xs.append(step)
                ys.append(value)
            if not xs:
                continue
            ax.plot(
                xs,
                ys,
                label=artifact.spec.label,
                color=artifact.spec.color,
                linestyle=artifact.spec.linestyle,
                linewidth=artifact.spec.linewidth,
                zorder=artifact.spec.zorder,
            )
        subplot_label = "(a)" if index == 0 else "(b)"
        ax.set_title(subplot_label)
        ax.set_xlabel("simulation step")
        ax.grid(True, color="#eeeeee", linewidth=0.8)
        ax.set_xlim(left=0)
        if max_step is not None:
            ax.set_xlim(right=max_step)
    axes[0].set_ylabel("mean forward velocity vx (m/s)")
    handles = []
    labels = []
    for ax in axes:
        ax_handles, ax_labels = ax.get_legend_handles_labels()
        handles.extend(ax_handles)
        labels.extend(ax_labels)
    dedup: dict[str, Any] = {}
    for handle, label in zip(handles, labels):
        dedup.setdefault(label, handle)
    fig.legend(
        dedup.values(),
        dedup.keys(),
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )
    if title:
        fig.suptitle(title, y=0.98)
        fig.tight_layout(rect=(0.0, 0.08, 1.0, 0.93))
    else:
        fig.tight_layout(rect=(0.0, 0.08, 1.0, 0.98))
    png_path = output_dir / f"{figure_basename}.png"
    pdf_path = output_dir / f"{figure_basename}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot T17 fixed-vx multi-joint P2 fault-response velocity traces.")
    parser.add_argument("--input_root", default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--a2_single_step_root", default=DEFAULT_A2_SINGLE_STEP_ROOT)
    parser.add_argument("--teacher_root", default=DEFAULT_TEACHER_ROOT)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--protocols", nargs="+", default=list(DEFAULT_PROTOCOLS), choices=DEFAULT_PROTOCOLS)
    parser.add_argument("--phase_label", default="T17")
    parser.add_argument("--history_label", default="H16")
    parser.add_argument("--figure_basename", default="t17_fixed_vx_fault_response")
    parser.add_argument("--tracking_table_basename", default="t17_fixed_vx_plotted_runs")
    parser.add_argument("--notes_basename", default="t17_fixed_vx_fault_response_notes")
    parser.add_argument("--summary_basename", default="t17_fixed_vx_fault_response_summary")
    parser.add_argument("--title", default=None)
    parser.add_argument("--max_step", type=int, default=None)
    parser.add_argument("--exclude_a2_single_step", action="store_true")
    parser.add_argument("--exclude_teacher_reference", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_root = resolve_path(args.input_root)
    a2_single_step_root = resolve_path(args.a2_single_step_root)
    teacher_root = resolve_path(args.teacher_root)
    output_dir = resolve_path(args.output_dir)
    method_specs = method_specs_for_args(args)
    artifacts, missing = load_run_artifacts(input_root, a2_single_step_root, teacher_root, args.protocols, method_specs)
    alignment = inspect_alignment(artifacts)
    plotted_rows = build_plotted_run_rows(artifacts)

    print("[T17 fixed-vx plot]")
    print(f"input_root: {repo_relative(input_root)}")
    print(f"a2_single_step_root: {repo_relative(a2_single_step_root)}")
    print(f"teacher_root: {repo_relative(teacher_root)}")
    print(f"output_dir: {repo_relative(output_dir)}")
    print(f"available_run_count: {len(artifacts)}")
    print(f"missing_run_count: {len(missing)}")
    print(f"phase_label: {args.phase_label}")
    print(f"history_label: {args.history_label}")
    print(f"exclude_a2_single_step: {args.exclude_a2_single_step}")
    print(f"exclude_teacher_reference: {args.exclude_teacher_reference}")
    print(f"plot_mode: {alignment['plot_mode']}")
    print(f"fault_aligned_supported: {alignment['fault_aligned_supported']}")
    print(f"alignment_note: {alignment['limitation']}")
    if missing:
        print("missing:")
        for item in missing:
            print(f"  - {item}")
    if args.dry_run:
        print("dry_run: no files written")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    png_path, pdf_path = plot_figure(
        output_dir=output_dir,
        artifacts=artifacts,
        alignment=alignment,
        protocols=args.protocols,
        figure_basename=args.figure_basename,
        title=args.title,
        max_step=args.max_step,
    )
    write_csv(output_dir / f"{args.tracking_table_basename}.csv", plotted_rows, RUN_TABLE_FIELDS)
    write_md_table(
        output_dir / f"{args.tracking_table_basename}.md",
        plotted_rows,
        RUN_TABLE_FIELDS,
        f"{args.phase_label} Fixed-vx Plotted Runs",
    )
    write_notes(
        output_dir=output_dir,
        artifacts=artifacts,
        missing=missing,
        alignment=alignment,
        plotted_rows=plotted_rows,
        phase_label=args.phase_label,
        notes_basename=args.notes_basename,
        summary_basename=args.summary_basename,
        include_a2_single_step=not args.exclude_a2_single_step,
    )
    (output_dir / f"{args.figure_basename}_command.txt").write_text(
        " ".join(sys.argv) + "\n",
        encoding="utf-8",
    )
    (output_dir / f"{args.figure_basename}_metadata.json").write_text(
        json.dumps(
            {
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "phase_label": args.phase_label,
                "history_label": args.history_label,
                "input_root": repo_relative(input_root),
                "a2_single_step_root": repo_relative(a2_single_step_root),
                "teacher_root": repo_relative(teacher_root),
                "exclude_a2_single_step": args.exclude_a2_single_step,
                "exclude_teacher_reference": args.exclude_teacher_reference,
                "plot_mode": alignment["plot_mode"],
                "max_step": args.max_step,
                "fault_aligned_supported": alignment["fault_aligned_supported"],
                "missing": missing,
                "figure_png": repo_relative(png_path),
                "figure_pdf": repo_relative(pdf_path),
                "not_paper_grade_final": True,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote: {repo_relative(png_path)}")
    print(f"wrote: {repo_relative(pdf_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
