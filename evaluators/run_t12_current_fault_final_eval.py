#!/usr/bin/env python3
"""T12 current-fault final-eval orchestration scaffold.

This is a batch wrapper around existing T10 evaluators. It does not implement
new policy logic, train policies, modify checkpoints, modify task configs, or
change P2 wrapper semantics.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = Path("papers/conference/results/t12_current_fault_final_eval")
SELECTED_TEACHER_DOC = Path("configs/train/selected_teacher_p2_velocity.md")

TASK = "Isaac-Ant-Velocity-Flat-v0"
TEACHER_TASK = "Isaac-Ant-Teacher-Velocity-Flat-v0"
A0_LABEL = "A0_Vel_healthy_train_eval_P2_random"
A1F_LABEL = "A1F_Vel_P2_random_train_eval_P2_random"
FAULT_PROFILE = "P2_locked_joint"
TARGET_JOINT = "front_left_foot"
REQUESTED_SEMANTICS = "simulation_joint_state_override_lock"
TARGET_VX = 1.0
DEFAULT_NUM_ENVS = 128
DEFAULT_NUM_STEPS = 1000
DEFAULT_SEED = 0
DEFAULT_DEVICE = "cuda"
DEFAULT_HISTORY_LEN = 16
DEFAULT_RANDOM_ONSET_STEP_MIN = 30
DEFAULT_RANDOM_ONSET_STEP_MAX = 150

A0_CHECKPOINT = (
    "logs/rsl_rl/healthy_baseline_velocity__rlm1_stripped__none/"
    "2026-06-09_23-58-25_a0_velocity_candidate1000__seed0/model_999.pt"
)
A2_SINGLE_STEP_CHECKPOINT = "papers/conference/results/t10_a2_student_distill_full_seed0/a2_student.pt"
A2_HISTORY_CHECKPOINT = "papers/conference/results/t10_a2_student_history_distill_full_h16_seed0/a2_student_history.pt"
A5_CHECKPOINT = "papers/conference/results/t10_a5_history_residual_distill_full_h16_seed0/a5_history_residual.pt"
A7_CHECKPOINT = "papers/conference/results/t10_a7_a0_residual_distill_full_h16_seed0/a7_a0_residual.pt"

PROTOCOLS = {
    "random": {
        "dir_name": "random_onset",
        "fault_onset_step_min": DEFAULT_RANDOM_ONSET_STEP_MIN,
        "fault_onset_step_max": DEFAULT_RANDOM_ONSET_STEP_MAX,
        "onset_note": "random_uniform [30, 150]",
    },
    "settled": {
        "dir_name": "settled_onset",
        "fault_onset_step_min": 300,
        "fault_onset_step_max": 300,
        "onset_note": "random_uniform [300, 300]",
    },
}

POLICY_DIRS = {
    "a0": "a0",
    "a1f_teacher": "a1f_teacher",
    "a2_single_step": "a2_single_step",
    "a2_history_h16": "a2_history_h16",
}

A5_ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
A7_ALPHAS = (0.0, 0.01, 0.025, 0.05, 0.10, 0.25)

SUMMARY_FIELDS = [
    "protocol",
    "policy",
    "alpha",
    "checkpoint",
    "target_joint",
    "num_envs",
    "num_steps",
    "seed",
    "mean_vel_x_pre_fault",
    "mean_vel_x_post_fault",
    "mean_abs_vx_error_pre_fault",
    "mean_abs_vx_error_post_fault",
    "mean_abs_yaw_error",
    "timeout_rate",
    "torso_height_failure_rate",
    "mean_post_fault_survival_steps",
    "residual_action_mean_norm",
    "residual_action_max_norm",
    "fallback_used",
    "simulation_override_applied",
    "p2_fault_became_active",
    "no_nan_inf",
    "velocity_timeseries_csv",
    "output_dir",
]


class T12EvalError(ValueError):
    """Raised for invalid T12 orchestration state."""


@dataclass(frozen=True)
class RunSpec:
    protocol_key: str
    protocol_dir: str
    policy: str
    alpha: float | None
    output_dir: Path
    checkpoint: str
    command: list[str]
    reference_policy: str | None = None


def str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in ("1", "true", "yes", "y", "on"):
        return True
    if lowered in ("0", "false", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value}")


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


def load_teacher_checkpoint() -> str:
    doc_path = resolve_repo_path(SELECTED_TEACHER_DOC)
    if not doc_path.is_file():
        raise T12EvalError(f"selected teacher doc not found: {SELECTED_TEACHER_DOC.as_posix()}")
    text = doc_path.read_text()
    matches = re.findall(r"(logs/[^\s`]+/model_[^\s`]+\.pt)", text)
    if not matches:
        raise T12EvalError(f"could not find selected teacher checkpoint in {SELECTED_TEACHER_DOC.as_posix()}")
    return matches[0]


def alpha_label(alpha: float) -> str:
    labels = {
        0.0: "000",
        0.01: "001",
        0.025: "0025",
        0.05: "005",
        0.10: "010",
        0.25: "025",
        0.5: "050",
        0.75: "075",
        1.0: "100",
    }
    for value, label in labels.items():
        if abs(alpha - value) < 1.0e-12:
            return label
    return f"{alpha:.6f}".rstrip("0").rstrip(".").replace(".", "p")


def display_command(command: list[str]) -> str:
    display_tokens = list(command)
    if display_tokens:
        display_tokens[0] = "python"
    return "PYTHONUNBUFFERED=1 TERM=xterm " + shlex.join(display_tokens)


def flag_value(flag: str, value: object) -> list[str]:
    """Return a CLI flag/value pair as two subprocess-safe tokens."""
    return [flag, str(value)]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="T12 current-fault final evaluation orchestration scaffold.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry_run", action="store_true", help="Print and save planned commands without launching Isaac.")
    mode.add_argument("--execute", action="store_true", help="Run planned commands sequentially.")
    parser.add_argument("--protocol", default="both", choices=("random", "settled", "both"))
    parser.add_argument(
        "--policy",
        default="all",
        choices=("all", "a0", "a1f_teacher", "a2_single_step", "a2_history_h16", "a5", "a7"),
    )
    parser.add_argument("--num_envs", type=int, default=DEFAULT_NUM_ENVS)
    parser.add_argument("--num_steps", type=int, default=DEFAULT_NUM_STEPS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--random_onset_step_min", type=int, default=DEFAULT_RANDOM_ONSET_STEP_MIN)
    parser.add_argument("--random_onset_step_max", type=int, default=DEFAULT_RANDOM_ONSET_STEP_MAX)
    parser.add_argument("--output_root", default=DEFAULT_OUTPUT_ROOT.as_posix())
    parser.add_argument("--stop_on_failure", type=str_to_bool, nargs="?", const=True, default=True)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.num_envs <= 0:
        raise T12EvalError("--num_envs must be > 0.")
    if args.num_steps <= 0:
        raise T12EvalError("--num_steps must be > 0.")
    if args.seed < 0:
        raise T12EvalError("--seed must be non-negative.")
    if args.random_onset_step_min < 0 or args.random_onset_step_max < 0:
        raise T12EvalError("random onset bounds must be non-negative.")
    if args.random_onset_step_min > args.random_onset_step_max:
        raise T12EvalError("--random_onset_step_min must be <= --random_onset_step_max.")
    checkpoint_paths = (
        A0_CHECKPOINT,
        A2_SINGLE_STEP_CHECKPOINT,
        A2_HISTORY_CHECKPOINT,
        A5_CHECKPOINT,
        A7_CHECKPOINT,
        load_teacher_checkpoint(),
    )
    for checkpoint in checkpoint_paths:
        if not resolve_repo_path(checkpoint).is_file():
            raise T12EvalError(f"checkpoint not found: {checkpoint}")


def selected_protocols(protocol: str) -> list[str]:
    if protocol == "both":
        return ["random", "settled"]
    return [protocol]


def selected_policy_groups(policy: str) -> list[str]:
    if policy == "all":
        return ["a0", "a1f_teacher", "a2_single_step", "a2_history_h16", "a5", "a7"]
    return [policy]


def onset_bounds(args: argparse.Namespace, protocol_key: str) -> tuple[int, int]:
    if protocol_key == "random":
        return args.random_onset_step_min, args.random_onset_step_max
    protocol = PROTOCOLS[protocol_key]
    return protocol["fault_onset_step_min"], protocol["fault_onset_step_max"]


def common_eval_args(args: argparse.Namespace, protocol_key: str, output_dir: Path) -> list[str]:
    onset_min, onset_max = onset_bounds(args, protocol_key)
    return [
        "--execute_eval",
        "--headless",
        *flag_value("--output_dir", repo_relative(output_dir)),
        *flag_value("--num_envs", args.num_envs),
        *flag_value("--num_steps", args.num_steps),
        *flag_value("--seed", args.seed),
        *flag_value("--fault_onset_mode", "random_uniform"),
        *flag_value("--fault_onset_step_min", onset_min),
        *flag_value("--fault_onset_step_max", onset_max),
        *flag_value("--device", args.device),
    ]


def student_common_args(args: argparse.Namespace, protocol_key: str, output_dir: Path) -> list[str]:
    return [
        *common_eval_args(args, protocol_key, output_dir),
        *flag_value("--target_joint", TARGET_JOINT),
        *flag_value("--history_len", DEFAULT_HISTORY_LEN),
    ]


def reference_common_args(
    args: argparse.Namespace,
    protocol_key: str,
    output_dir: Path,
    *,
    policy_label: str,
    task: str,
    checkpoint: str,
) -> list[str]:
    onset_min, onset_max = onset_bounds(args, protocol_key)
    return [
        "--execute_eval",
        "--headless",
        *flag_value("--single_policy_label", policy_label),
        *flag_value("--single_policy_task", task),
        *flag_value("--single_policy_checkpoint", checkpoint),
        *flag_value("--single_policy_output_dir", repo_relative(output_dir)),
        *flag_value("--num_envs", args.num_envs),
        *flag_value("--num_steps", args.num_steps),
        *flag_value("--seed", args.seed),
        *flag_value("--fault_onset_mode", "random_uniform"),
        *flag_value("--fault_onset_step_min", onset_min),
        *flag_value("--fault_onset_step_max", onset_max),
        *flag_value("--device", args.device),
    ]


def make_specs(args: argparse.Namespace) -> list[RunSpec]:
    output_root = resolve_repo_path(args.output_root)
    teacher_checkpoint = load_teacher_checkpoint()
    specs: list[RunSpec] = []
    python = sys.executable

    for protocol_key in selected_protocols(args.protocol):
        protocol_dir = PROTOCOLS[protocol_key]["dir_name"]
        protocol_root = output_root / protocol_dir
        for group in selected_policy_groups(args.policy):
            if group == "a0":
                output_dir = protocol_root / "a0"
                command = [
                    python,
                    "evaluators/run_t10_velocity_p2_random_eval_compare.py",
                    *reference_common_args(
                        args,
                        protocol_key,
                        output_dir,
                        policy_label=A0_LABEL,
                        task=TASK,
                        checkpoint=A0_CHECKPOINT,
                    ),
                ]
                specs.append(
                    RunSpec(protocol_key, protocol_dir, "a0", None, output_dir, A0_CHECKPOINT, command, "a0")
                )
            elif group == "a1f_teacher":
                output_dir = protocol_root / "a1f_teacher"
                command = [
                    python,
                    "evaluators/run_t10_velocity_p2_random_eval_compare.py",
                    *reference_common_args(
                        args,
                        protocol_key,
                        output_dir,
                        policy_label=A1F_LABEL,
                        task=TEACHER_TASK,
                        checkpoint=teacher_checkpoint,
                    ),
                ]
                specs.append(
                    RunSpec(
                        protocol_key,
                        protocol_dir,
                        "a1f_teacher",
                        None,
                        output_dir,
                        teacher_checkpoint,
                        command,
                        "a1f",
                    )
                )
            elif group == "a2_single_step":
                output_dir = protocol_root / POLICY_DIRS[group]
                command = [
                    python,
                    "evaluators/run_t10_a2_student_p2_eval.py",
                    *flag_value("--policy_kind", "single_step"),
                    *flag_value("--checkpoint", A2_SINGLE_STEP_CHECKPOINT),
                    *student_common_args(args, protocol_key, output_dir),
                ]
                specs.append(
                    RunSpec(protocol_key, protocol_dir, group, None, output_dir, A2_SINGLE_STEP_CHECKPOINT, command)
                )
            elif group == "a2_history_h16":
                output_dir = protocol_root / POLICY_DIRS[group]
                command = [
                    python,
                    "evaluators/run_t10_a2_student_p2_eval.py",
                    *flag_value("--policy_kind", "history"),
                    *flag_value("--checkpoint", A2_HISTORY_CHECKPOINT),
                    *student_common_args(args, protocol_key, output_dir),
                ]
                specs.append(
                    RunSpec(protocol_key, protocol_dir, group, None, output_dir, A2_HISTORY_CHECKPOINT, command)
                )
            elif group == "a5":
                for alpha in A5_ALPHAS:
                    output_dir = protocol_root / f"a5_alpha_{alpha_label(alpha)}"
                    command = [
                        python,
                        "evaluators/run_t10_a5_history_residual_p2_eval.py",
                        *flag_value("--a2_checkpoint", A2_HISTORY_CHECKPOINT),
                        *flag_value("--a5_checkpoint", A5_CHECKPOINT),
                        *flag_value("--alpha", alpha),
                        *student_common_args(args, protocol_key, output_dir),
                    ]
                    specs.append(
                        RunSpec(protocol_key, protocol_dir, "a5", alpha, output_dir, A5_CHECKPOINT, command)
                    )
            elif group == "a7":
                for alpha in A7_ALPHAS:
                    output_dir = protocol_root / f"a7_alpha_{alpha_label(alpha)}"
                    command = [
                        python,
                        "evaluators/run_t10_a7_a0_residual_p2_eval.py",
                        *flag_value("--a0_checkpoint", A0_CHECKPOINT),
                        *flag_value("--a7_checkpoint", A7_CHECKPOINT),
                        *flag_value("--alpha", alpha),
                        *student_common_args(args, protocol_key, output_dir),
                    ]
                    specs.append(
                        RunSpec(protocol_key, protocol_dir, "a7", alpha, output_dir, A7_CHECKPOINT, command)
                    )
    return specs


def write_command_markdown(output_root: Path, specs: list[RunSpec]) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    command_path = output_root / "t12_run_commands.md"
    lines = [
        "# T12 Current-Fault Final-Eval Commands",
        "",
        "Generated by `evaluators/run_t12_current_fault_final_eval.py --dry_run`.",
        "",
        "These commands are candidate-level evaluation commands only. They launch Isaac only when run manually.",
        "",
    ]
    for index, spec in enumerate(specs, start=1):
        lines.extend(
            [
                f"## {index}. {spec.protocol_dir} / {spec.policy}"
                + ("" if spec.alpha is None else f" / alpha={spec.alpha}"),
                "",
                "```bash",
                display_command(spec.command),
                "```",
                "",
            ]
        )
    command_path.write_text("\n".join(lines))
    return command_path


def normalize_reference_outputs(spec: RunSpec) -> None:
    if spec.reference_policy is None:
        return
    if (spec.output_dir / "summary.json").is_file() and (spec.output_dir / "velocity_timeseries.csv").is_file():
        return
    needles = ("a0",) if spec.reference_policy == "a0" else ("a1f", "teacher")
    candidate_dirs = [
        path.parent
        for path in spec.output_dir.rglob("summary.json")
        if path.parent != spec.output_dir and any(needle in path.parent.name.lower() for needle in needles)
    ]
    if not candidate_dirs:
        return
    source_dir = candidate_dirs[0]
    for filename in ("summary.json", "velocity_timeseries.csv", "README.md", "command.txt"):
        source = source_dir / filename
        if source.is_file() and not (spec.output_dir / filename).is_file():
            shutil.copy2(source, spec.output_dir / filename)


def require_completed_outputs(spec: RunSpec) -> None:
    normalize_reference_outputs(spec)
    missing = [name for name in ("summary.json", "velocity_timeseries.csv") if not (spec.output_dir / name).is_file()]
    if missing:
        raise T12EvalError(f"missing {', '.join(missing)} in {repo_relative(spec.output_dir)}")


def unwrap_summary(spec: RunSpec, summary: dict[str, Any]) -> dict[str, Any]:
    if "mean_vel_x_post_fault" in summary:
        return summary
    policies = summary.get("policies")
    if not isinstance(policies, dict) or not policies:
        return summary
    if spec.reference_policy == "a0":
        for key, value in policies.items():
            if "A0" in key and isinstance(value, dict):
                return value
    if spec.reference_policy == "a1f":
        for key, value in policies.items():
            if "A1F" in key and isinstance(value, dict):
                return value
    first_value = next(iter(policies.values()))
    return first_value if isinstance(first_value, dict) else summary


def metric(summary: dict[str, Any], key: str) -> Any:
    value = summary.get(key, "")
    return "" if value is None else value


def parse_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value in (None, ""):
        return None
    text = str(value).strip().lower()
    if text in ("1", "1.0", "true", "yes", "y"):
        return True
    if text in ("0", "0.0", "false", "no", "n"):
        return False
    return None


def mean_or_blank(values: list[float]) -> float | str:
    if not values:
        return ""
    return sum(values) / len(values)


def max_or_blank(values: list[float]) -> float | str:
    if not values:
        return ""
    return max(values)


def first_nonblank(*values: Any) -> Any:
    for value in values:
        if value not in (None, ""):
            return value
    return ""


def summary_csv_path(spec: RunSpec, raw_summary: dict[str, Any], summary: dict[str, Any], key: str, filename: str) -> Path:
    for candidate_summary in (summary, raw_summary):
        value = candidate_summary.get(key)
        if isinstance(value, str) and value:
            path = Path(value)
            if path.is_absolute():
                return path
            return resolve_repo_path(path)
    return spec.output_dir / filename


def rollout_p2_fault_became_active(rollout_path: Path) -> bool | str:
    if not rollout_path.is_file():
        return ""
    saw_fault_value = False
    with rollout_path.open(newline="") as file:
        for row in csv.DictReader(file):
            fault_applied = parse_float(row.get("P2/fault_applied"))
            if fault_applied is None:
                continue
            saw_fault_value = True
            if fault_applied >= 0.5:
                return True
    return False if saw_fault_value else ""


def rollout_metrics_fallback(spec: RunSpec, raw_summary: dict[str, Any], summary: dict[str, Any]) -> dict[str, Any]:
    rollout_path = summary_csv_path(spec, raw_summary, summary, "rollout_metrics_csv", "rollout_metrics.csv")
    if not rollout_path.is_file():
        return {}

    with rollout_path.open(newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        return {}

    pre_rows: list[dict[str, str]] = []
    post_rows: list[dict[str, str]] = []
    p2_fault_became_active = False
    for row in rows:
        fault_applied = parse_float(row.get("P2/fault_applied"))
        if fault_applied is None:
            continue
        if fault_applied >= 0.5:
            p2_fault_became_active = True
            post_rows.append(row)
        else:
            pre_rows.append(row)

    def row_mean(source_rows: list[dict[str, str]], column: str) -> float | str:
        return mean_or_blank([value for row in source_rows if (value := parse_float(row.get(column))) is not None])

    fallback_values = [value for row in rows if (value := parse_float(row.get("P2/fallback_used"))) is not None]
    override_values = [
        value for row in rows if (value := parse_float(row.get("P2/simulation_override_applied"))) is not None
    ]
    no_nan_values = [value for row in rows if (value := parse_bool(row.get("no_nan_inf"))) is not None]
    post_fault_yaw_rows = post_rows if post_rows else rows

    return {
        "target_joint": first_nonblank(rows[0].get("target_joint"), TARGET_JOINT),
        "seed": first_nonblank(rows[0].get("seed")),
        "mean_vel_x_pre_fault": row_mean(pre_rows, "mean_vel_x"),
        "mean_vel_x_post_fault": row_mean(post_rows, "mean_vel_x"),
        "mean_abs_vx_error_pre_fault": row_mean(pre_rows, "mean_abs_vx_error"),
        "mean_abs_vx_error_post_fault": row_mean(post_rows, "mean_abs_vx_error"),
        "mean_abs_yaw_error": row_mean(post_fault_yaw_rows, "mean_abs_yaw_error"),
        "fallback_used": max_or_blank(fallback_values),
        "simulation_override_applied": max_or_blank(override_values),
        "p2_fault_became_active": p2_fault_became_active,
        "no_nan_inf": all(no_nan_values) if no_nan_values else "",
    }


def merged_metric(summary: dict[str, Any], fallback: dict[str, Any], key: str) -> Any:
    return first_nonblank(metric(summary, key), fallback.get(key, ""))


def checkpoint_from_summary(spec: RunSpec, summary: dict[str, Any]) -> str:
    if spec.policy == "a5":
        return summary.get("a5_checkpoint") or spec.checkpoint
    if spec.policy == "a7":
        return summary.get("a7_checkpoint") or spec.checkpoint
    return summary.get("checkpoint_path") or summary.get("checkpoint") or spec.checkpoint


def row_from_spec(spec: RunSpec) -> dict[str, Any]:
    raw_summary = json.loads((spec.output_dir / "summary.json").read_text())
    summary = unwrap_summary(spec, raw_summary)
    rollout_csv = summary_csv_path(spec, raw_summary, summary, "rollout_metrics_csv", "rollout_metrics.csv")
    fallback = {}
    if metric(summary, "mean_vel_x_post_fault") == "":
        fallback = rollout_metrics_fallback(spec, raw_summary, summary)
    velocity_csv = summary_csv_path(spec, raw_summary, summary, "velocity_timeseries_csv", "velocity_timeseries.csv")
    p2_fault_became_active = first_nonblank(
        metric(summary, "p2_fault_became_active"),
        fallback.get("p2_fault_became_active"),
        rollout_p2_fault_became_active(rollout_csv),
    )
    return {
        "protocol": spec.protocol_dir,
        "policy": spec.policy,
        "alpha": "" if spec.alpha is None else spec.alpha,
        "checkpoint": checkpoint_from_summary(spec, summary),
        "target_joint": first_nonblank(summary.get("target_joint"), fallback.get("target_joint"), TARGET_JOINT),
        "num_envs": first_nonblank(summary.get("num_envs"), raw_summary.get("num_envs")),
        "num_steps": first_nonblank(summary.get("num_steps"), raw_summary.get("num_steps")),
        "seed": first_nonblank(summary.get("seed"), fallback.get("seed"), raw_summary.get("seed")),
        "mean_vel_x_pre_fault": merged_metric(summary, fallback, "mean_vel_x_pre_fault"),
        "mean_vel_x_post_fault": merged_metric(summary, fallback, "mean_vel_x_post_fault"),
        "mean_abs_vx_error_pre_fault": merged_metric(summary, fallback, "mean_abs_vx_error_pre_fault"),
        "mean_abs_vx_error_post_fault": merged_metric(summary, fallback, "mean_abs_vx_error_post_fault"),
        "mean_abs_yaw_error": merged_metric(summary, fallback, "mean_abs_yaw_error"),
        "timeout_rate": metric(summary, "timeout_rate"),
        "torso_height_failure_rate": metric(summary, "torso_height_failure_rate"),
        "mean_post_fault_survival_steps": metric(summary, "mean_post_fault_survival_steps"),
        "residual_action_mean_norm": metric(summary, "residual_action_mean_norm"),
        "residual_action_max_norm": metric(summary, "residual_action_max_norm"),
        "fallback_used": first_nonblank(
            summary.get("fallback_used"), summary.get("P2/fallback_used"), fallback.get("fallback_used")
        ),
        "simulation_override_applied": first_nonblank(
            summary.get("P2/simulation_override_applied"),
            summary.get("simulation_override_applied"),
            fallback.get("simulation_override_applied"),
        ),
        "p2_fault_became_active": p2_fault_became_active,
        "no_nan_inf": merged_metric(summary, fallback, "no_nan_inf"),
        "velocity_timeseries_csv": repo_relative(velocity_csv),
        "output_dir": repo_relative(spec.output_dir),
    }


def format_md(value: Any) -> str:
    if value == "":
        return ""
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        return f"{value:.9g}"
    return str(value)


def aggregate_completed(output_root: Path, specs: list[RunSpec]) -> tuple[Path, Path, int]:
    tables_dir = output_root / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    rows = [row_from_spec(spec) for spec in specs if (spec.output_dir / "summary.json").is_file()]
    csv_path = tables_dir / "metrics_summary.csv"
    md_path = tables_dir / "metrics_summary.md"
    with csv_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# T12 Current-Fault Metrics Summary",
        "",
        "Candidate-level aggregation only; not paper-grade final.",
        "",
        f"- completed runs: `{len(rows)}`",
        "",
        "| " + " | ".join(SUMMARY_FIELDS) + " |",
        "| " + " | ".join("---" for _ in SUMMARY_FIELDS) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(format_md(row[field]) for field in SUMMARY_FIELDS) + " |")
    md_path.write_text("\n".join(lines) + "\n")
    return csv_path, md_path, len(rows)


def run_dry_run(args: argparse.Namespace, specs: list[RunSpec]) -> int:
    output_root = resolve_repo_path(args.output_root)
    command_path = write_command_markdown(output_root, specs)
    print("[T12 DRY RUN]")
    print(f"planned_command_count: {len(specs)}")
    print(f"commands: {repo_relative(command_path)}")
    print("no_isaac_sim_launched: true")
    print("no_fake_metrics_written: true")
    for spec in specs:
        print(display_command(spec.command))
    return 0


def run_execute(args: argparse.Namespace, specs: list[RunSpec]) -> int:
    output_root = resolve_repo_path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    write_command_markdown(output_root, specs)
    completed: list[RunSpec] = []
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["TERM"] = env.get("TERM", "xterm")
    for index, spec in enumerate(specs, start=1):
        spec.output_dir.mkdir(parents=True, exist_ok=True)
        terminal_log = spec.output_dir / "terminal.log"
        print(f"[T12 {index}/{len(specs)}] {spec.protocol_dir}/{spec.output_dir.name}")
        print(display_command(spec.command))
        with terminal_log.open("w") as log_file:
            log_file.write(display_command(spec.command) + "\n\n")
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
            message = f"command failed with exit code {result.returncode}; see {repo_relative(terminal_log)}"
            if args.stop_on_failure:
                raise T12EvalError(message)
            print(f"[T12 WARNING] {message}", file=sys.stderr)
            continue
        require_completed_outputs(spec)
        completed.append(spec)
        csv_path, md_path, count = aggregate_completed(output_root, completed)
        print(f"aggregate_rows: {count}")
        print(f"metrics_csv: {repo_relative(csv_path)}")
        print(f"metrics_md: {repo_relative(md_path)}")
    return 0


def main() -> int:
    try:
        args = build_parser().parse_args()
        validate_args(args)
        specs = make_specs(args)
        if args.dry_run:
            return run_dry_run(args, specs)
        return run_execute(args, specs)
    except T12EvalError as exc:
        print(f"[T12 ERROR] {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("[T12 ERROR] interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
