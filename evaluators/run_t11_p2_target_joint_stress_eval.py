#!/usr/bin/env python3
"""Batch runner for T11 P2 target-joint stress tests.

This script orchestrates existing T10 deployment-facing evaluators. It does not
implement policy logic, train, or modify checkpoints/task configs/P2 semantics.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_OUTPUT_ROOT = Path("papers/conference/results/t11_p2_target_joint_stress")
DEFAULT_NUM_ENVS = 128
DEFAULT_NUM_STEPS = 1000
DEFAULT_SEED = 0
DEFAULT_DEVICE = "cpu"
DEFAULT_HISTORY_LEN = 16

A0_CHECKPOINT = (
    "logs/rsl_rl/healthy_baseline_velocity__rlm1_stripped__none/"
    "2026-06-09_23-58-25_a0_velocity_candidate1000__seed0/model_999.pt"
)
A2_HISTORY_CHECKPOINT = "papers/conference/results/t10_a2_student_history_distill_full_h16_seed0/a2_student_history.pt"
A5_CHECKPOINT = "papers/conference/results/t10_a5_history_residual_distill_full_h16_seed0/a5_history_residual.pt"
A7_CHECKPOINT = "papers/conference/results/t10_a7_a0_residual_distill_full_h16_seed0/a7_a0_residual.pt"

TARGET_JOINTS = (
    "front_left_foot",
    "front_right_foot",
    "left_back_foot",
    "right_back_foot",
    "front_left_leg",
    "front_right_leg",
    "left_back_leg",
    "right_back_leg",
)

PROTOCOLS = {
    "random": {
        "dir_name": "random_p2",
        "fault_onset_step_min": 30,
        "fault_onset_step_max": 150,
    },
    "settled": {
        "dir_name": "settled_p2",
        "fault_onset_step_min": 300,
        "fault_onset_step_max": 300,
    },
}

SUMMARY_FIELDS = [
    "protocol",
    "target_joint",
    "policy",
    "alpha",
    "checkpoint",
    "mean_vel_x_post_fault",
    "mean_abs_vx_error_post_fault",
    "mean_abs_yaw_error",
    "timeout_rate",
    "torso_height_failure_rate",
    "residual_action_mean_norm",
    "residual_action_max_norm",
    "fallback_used",
    "simulation_override_applied",
    "p2_fault_became_active",
    "no_nan_inf",
    "output_dir",
]


class T11StressEvalError(ValueError):
    """Raised for invalid T11 stress-runner state."""


@dataclass(frozen=True)
class RunSpec:
    protocol: str
    protocol_dir: str
    target_joint: str
    policy: str
    alpha: float | None
    output_dir: Path
    checkpoint: str
    command: list[str]


def repo_relative(path: str | Path) -> str:
    path = Path(path)
    if not path.is_absolute():
        return path.as_posix()
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def resolve_repo_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def alpha_label(alpha: float) -> str:
    labels = {
        0.0: "000",
        0.01: "001",
        0.025: "0025",
        0.05: "005",
        0.25: "025",
        0.5: "050",
    }
    for key, label in labels.items():
        if abs(alpha - key) < 1.0e-12:
            return label
    text = f"{alpha:.6f}".rstrip("0").rstrip(".")
    return text.replace(".", "p").replace("-", "m")


def shell_command(command: list[str]) -> str:
    display = ["python" if index == 0 else value for index, value in enumerate(command)]
    return "PYTHONUNBUFFERED=1 TERM=xterm " + shlex.join(display)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plan or execute T11 P2 target-joint stress tests through existing T10 evaluators."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--execute", action="store_true", help="Run planned evaluator commands sequentially.")
    mode.add_argument("--dry_run", action="store_true", help="Print planned commands without launching Isaac.")
    parser.add_argument("--num_envs", type=int, default=DEFAULT_NUM_ENVS)
    parser.add_argument("--num_steps", type=int, default=DEFAULT_NUM_STEPS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--protocol", default="both", choices=("random", "settled", "both"))
    parser.add_argument("--joint", default="all", choices=("all", *TARGET_JOINTS))
    parser.add_argument("--policy_set", default="default", choices=("default",))
    parser.add_argument("--output_root", default=DEFAULT_OUTPUT_ROOT.as_posix())
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.num_envs <= 0:
        raise T11StressEvalError("--num_envs must be > 0.")
    if args.num_steps <= 0:
        raise T11StressEvalError("--num_steps must be > 0.")
    if args.seed < 0:
        raise T11StressEvalError("--seed must be non-negative.")
    for checkpoint in (A0_CHECKPOINT, A2_HISTORY_CHECKPOINT, A5_CHECKPOINT, A7_CHECKPOINT):
        if not resolve_repo_path(checkpoint).is_file():
            raise T11StressEvalError(f"checkpoint does not exist: {checkpoint}")


def selected_protocols(protocol: str) -> list[str]:
    if protocol == "both":
        return ["random", "settled"]
    return [protocol]


def selected_joints(joint: str) -> list[str]:
    if joint == "all":
        return list(TARGET_JOINTS)
    return [joint]


def common_eval_args(args: argparse.Namespace, protocol: str, target_joint: str, output_dir: Path) -> list[str]:
    protocol_cfg = PROTOCOLS[protocol]
    return [
        "--execute_eval",
        "--headless",
        "--output_dir",
        repo_relative(output_dir),
        "--num_envs",
        str(args.num_envs),
        "--num_steps",
        str(args.num_steps),
        "--seed",
        str(args.seed),
        "--fault_onset_mode",
        "random_uniform",
        "--fault_onset_step_min",
        str(protocol_cfg["fault_onset_step_min"]),
        "--fault_onset_step_max",
        str(protocol_cfg["fault_onset_step_max"]),
        "--target_joint",
        target_joint,
        "--history_len",
        str(DEFAULT_HISTORY_LEN),
        "--device",
        args.device,
    ]


def make_specs(args: argparse.Namespace) -> list[RunSpec]:
    output_root = resolve_repo_path(args.output_root)
    specs: list[RunSpec] = []
    python = sys.executable

    for protocol in selected_protocols(args.protocol):
        protocol_dir = PROTOCOLS[protocol]["dir_name"]
        for target_joint in selected_joints(args.joint):
            base_dir = output_root / protocol_dir / target_joint

            output_dir = base_dir / "a2_history"
            command = [
                python,
                "evaluators/run_t10_a2_student_p2_eval.py",
                "--policy_kind",
                "history",
                "--checkpoint",
                A2_HISTORY_CHECKPOINT,
                *common_eval_args(args, protocol, target_joint, output_dir),
            ]
            specs.append(
                RunSpec(
                    protocol=protocol_dir,
                    protocol_dir=protocol_dir,
                    target_joint=target_joint,
                    policy="a2_history",
                    alpha=None,
                    output_dir=output_dir,
                    checkpoint=A2_HISTORY_CHECKPOINT,
                    command=command,
                )
            )

            for alpha in (0.0, 0.25, 0.5):
                output_dir = base_dir / f"a5_alpha_{alpha_label(alpha)}"
                command = [
                    python,
                    "evaluators/run_t10_a5_history_residual_p2_eval.py",
                    "--a2_checkpoint",
                    A2_HISTORY_CHECKPOINT,
                    "--a5_checkpoint",
                    A5_CHECKPOINT,
                    "--alpha",
                    str(alpha),
                    *common_eval_args(args, protocol, target_joint, output_dir),
                ]
                specs.append(
                    RunSpec(
                        protocol=protocol_dir,
                        protocol_dir=protocol_dir,
                        target_joint=target_joint,
                        policy="a5_history_residual",
                        alpha=alpha,
                        output_dir=output_dir,
                        checkpoint=A5_CHECKPOINT,
                        command=command,
                    )
                )

            for alpha in (0.0, 0.01, 0.025, 0.05):
                output_dir = base_dir / f"a7_alpha_{alpha_label(alpha)}"
                command = [
                    python,
                    "evaluators/run_t10_a7_a0_residual_p2_eval.py",
                    "--a0_checkpoint",
                    A0_CHECKPOINT,
                    "--a7_checkpoint",
                    A7_CHECKPOINT,
                    "--alpha",
                    str(alpha),
                    *common_eval_args(args, protocol, target_joint, output_dir),
                ]
                specs.append(
                    RunSpec(
                        protocol=protocol_dir,
                        protocol_dir=protocol_dir,
                        target_joint=target_joint,
                        policy="a7_a0_residual",
                        alpha=alpha,
                        output_dir=output_dir,
                        checkpoint=A7_CHECKPOINT,
                        command=command,
                    )
                )

    return specs


def summary_value(summary: dict[str, Any], key: str, default: Any = "") -> Any:
    value = summary.get(key, default)
    if value is None:
        return default
    return value


def checkpoint_from_summary(spec: RunSpec, summary: dict[str, Any]) -> str:
    for key in ("checkpoint_path", "a5_checkpoint", "a7_checkpoint", "a2_checkpoint", "a0_checkpoint"):
        value = summary.get(key)
        if isinstance(value, str) and value:
            if spec.policy == "a5_history_residual" and key != "a5_checkpoint":
                continue
            if spec.policy == "a7_a0_residual" and key != "a7_checkpoint":
                continue
            return value
    return spec.checkpoint


def row_from_summary(spec: RunSpec) -> dict[str, Any]:
    summary_path = spec.output_dir / "summary.json"
    summary = json.loads(summary_path.read_text())
    simulation_override = summary.get("P2/simulation_override_applied", summary.get("simulation_override_applied", ""))
    fallback_used = summary.get("fallback_used", summary.get("P2/fallback_used", ""))
    return {
        "protocol": spec.protocol,
        "target_joint": summary.get("target_joint", spec.target_joint),
        "policy": spec.policy,
        "alpha": "" if spec.alpha is None else spec.alpha,
        "checkpoint": checkpoint_from_summary(spec, summary),
        "mean_vel_x_post_fault": summary_value(summary, "mean_vel_x_post_fault"),
        "mean_abs_vx_error_post_fault": summary_value(summary, "mean_abs_vx_error_post_fault"),
        "mean_abs_yaw_error": summary_value(summary, "mean_abs_yaw_error"),
        "timeout_rate": summary_value(summary, "timeout_rate"),
        "torso_height_failure_rate": summary_value(summary, "torso_height_failure_rate"),
        "residual_action_mean_norm": summary_value(summary, "residual_action_mean_norm"),
        "residual_action_max_norm": summary_value(summary, "residual_action_max_norm"),
        "fallback_used": fallback_used,
        "simulation_override_applied": simulation_override,
        "p2_fault_became_active": summary_value(summary, "p2_fault_became_active"),
        "no_nan_inf": summary_value(summary, "no_nan_inf"),
        "output_dir": repo_relative(spec.output_dir),
    }


def format_md_value(value: Any) -> str:
    if value == "":
        return ""
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        return f"{value:.9g}"
    return str(value)


def write_summary(output_root: Path, specs: list[RunSpec]) -> tuple[Path, Path, int]:
    rows: list[dict[str, Any]] = []
    for spec in specs:
        if (spec.output_dir / "summary.json").is_file():
            rows.append(row_from_summary(spec))

    csv_path = output_root / "t11_p2_target_joint_stress_summary.csv"
    md_path = output_root / "t11_p2_target_joint_stress_summary.md"
    output_root.mkdir(parents=True, exist_ok=True)

    with csv_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# T11 P2 Target-Joint Stress Summary",
        "",
        "Candidate-level stress-test aggregation only; not paper-grade final.",
        "",
        f"- aggregated successful runs: `{len(rows)}`",
        "",
        "| " + " | ".join(SUMMARY_FIELDS) + " |",
        "| " + " | ".join("---" for _ in SUMMARY_FIELDS) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(format_md_value(row[field]) for field in SUMMARY_FIELDS) + " |")
    md_path.write_text("\n".join(lines) + "\n")
    return csv_path, md_path, len(rows)


def require_run_outputs(spec: RunSpec) -> None:
    missing = [name for name in ("summary.json", "velocity_timeseries.csv") if not (spec.output_dir / name).is_file()]
    if missing:
        joined = ", ".join(missing)
        raise T11StressEvalError(f"run did not produce required files ({joined}): {repo_relative(spec.output_dir)}")


def run_dry_run(args: argparse.Namespace, specs: list[RunSpec]) -> int:
    output_root = resolve_repo_path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    command_path = output_root / "dry_run_commands.txt"
    commands = [shell_command(spec.command) for spec in specs]
    command_path.write_text("\n\n".join(commands) + "\n")
    print("[T11-P2-STRESS DRY RUN]")
    print(f"planned_command_count: {len(specs)}")
    print(f"dry_run_commands: {repo_relative(command_path)}")
    print("no_isaac_sim_launched: true")
    print("no_fake_summary_metrics_written: true")
    for command in commands:
        print(command)
    return 0


def run_execute(args: argparse.Namespace, specs: list[RunSpec]) -> int:
    output_root = resolve_repo_path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    completed: list[RunSpec] = []
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["TERM"] = env.get("TERM", "xterm")

    print("[T11-P2-STRESS EXECUTE]")
    print(f"planned_command_count: {len(specs)}")
    print("stop_on_failure: true")
    for index, spec in enumerate(specs, start=1):
        spec.output_dir.mkdir(parents=True, exist_ok=True)
        terminal_log = spec.output_dir / "terminal.log"
        print(f"[{index}/{len(specs)}] {spec.protocol}/{spec.target_joint}/{spec.output_dir.name}")
        print(f"  command: {shell_command(spec.command)}")
        with terminal_log.open("w") as log_file:
            log_file.write(shell_command(spec.command) + "\n\n")
            log_file.flush()
            result = subprocess.run(
                spec.command,
                cwd=REPO_ROOT,
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        if result.returncode != 0:
            raise T11StressEvalError(
                f"command failed with exit code {result.returncode}; see {repo_relative(terminal_log)}"
            )
        require_run_outputs(spec)
        completed.append(spec)
        csv_path, md_path, row_count = write_summary(output_root, completed)
        print(f"  terminal_log: {repo_relative(terminal_log)}")
        print(f"  aggregate_rows: {row_count}")
        print(f"  summary_csv: {repo_relative(csv_path)}")
        print(f"  summary_md: {repo_relative(md_path)}")
    return 0


def main() -> int:
    try:
        args = build_parser().parse_args()
        validate_args(args)
        specs = make_specs(args)
        if args.dry_run:
            return run_dry_run(args, specs)
        return run_execute(args, specs)
    except T11StressEvalError as exc:
        print(f"[T11-P2-STRESS ERROR] {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("[T11-P2-STRESS ERROR] interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
