#!/usr/bin/env python3
"""T17 closed-loop multi-joint P2 evaluator scaffold for A2/A5 policies.

This evaluator is execution-gated. Dry-run prints the planned policy/protocol
matrix without launching Isaac. Execute mode runs per-policy Isaac rollouts in
separate subprocesses and aggregates completed outputs for preliminary
advisor-facing evidence. It does not train, mutate checkpoints, mutate
datasets, modify task configs, or change P2 wrapper semantics.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shlex
import statistics
import subprocess
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


EVALUATORS_DIR = Path(__file__).resolve().parent
if str(EVALUATORS_DIR) not in sys.path:
    sys.path.insert(0, str(EVALUATORS_DIR))

from collect_t10_teacher_gap_dataset import (
    policy_tensor,
    resolve_forward_velocity_tensor,
    target_vx_tensor,
)
from run_t09_p2_quick_demo_compare import (
    optional_float,
    prepare_agent_cfg,
    repo_relative,
    resolve_repo_path,
    scalar,
)
from run_t10_velocity_p2_random_eval_compare import (
    build_random_onset_survival_summary,
    current_onset_steps,
    first_matching_log_value,
    resolve_yaw_rate_metrics,
)
from run_t13c_multijoint_teacher_eval import (
    command_stats_from_rows,
    command_vx_stats,
    force_fixed_velocity_command,
    per_env_yaw_rate,
    row_mean,
    rows_by_fault_fraction,
    tensor_to_int_list,
    tensors_are_finite,
    validate_command_mode,
)
from t18r_control_timing import add_control_timing_args, apply_control_timing_to_env_cfg, validate_pg500_history_len


REPO_ROOT = Path(__file__).resolve().parents[1]
TASK = "Isaac-Ant-Velocity-Flat-v0"
TEACHER_TASK = "Isaac-Ant-Teacher-Velocity-MultiJointP2-Flat-v0"
FAULT_PROFILE = "P2_locked_joint"
TARGET_JOINT_MODE = "random_per_env"
TARGET_JOINT_PLACEHOLDER = "front_left_foot"
REQUESTED_SEMANTICS = "simulation_joint_state_override_lock"
RLM_PHASE = "RLM1 stripped / conference"

DEFAULT_TEACHER_CHECKPOINT = (
    "logs/rsl_rl/teacher_p2_multijoint_velocity_curriculum_v2b__rlm1_stripped__p2_multi_joint_random/"
    "2026-06-15_18-16-03_a1f_multijoint_velocity_p2_v2b_curriculum_curriculum_s2_realistic_transition_random_p2__seed0/"
    "model_9997.pt"
)
DEFAULT_A2_SINGLE_STEP_CHECKPOINT = (
    "papers/conference/results/t15_a2_single_step_multijoint_seed0/a2_single_step_multijoint.pt"
)
DEFAULT_A2_HISTORY_CHECKPOINT = (
    "papers/conference/results/t15_a2_history_h16_multijoint_seed0/a2_history_h16_multijoint.pt"
)
DEFAULT_A5_CHECKPOINT = (
    "papers/conference/results/t16_a5_history_residual_multijoint_seed0/a5_history_residual_multijoint.pt"
)
DEFAULT_OUTPUT_DIR = "papers/conference/results/t17_multijoint_closed_loop_eval"

EXPECTED_STUDENT_OBS_DIM = 61
EXPECTED_ACTION_DIM = 8
EXPECTED_RESIDUAL_DIM = 8
DEFAULT_HISTORY_LEN = 16
DEFAULT_NUM_ENVS = 512
DEFAULT_NUM_STEPS = 1000
DEFAULT_SEED = 0
DEFAULT_DEVICE = "cuda"
DEFAULT_PROGRESS_EVERY = 100
FIXED_VX = 1.0
COMMAND_RANDOM_RANGE = (0.2, 1.5)
DEFAULT_POLICIES = ("a2_single_step", "a2_history", "a5")
POLICY_CHOICES = ("a2_single_step", "a2_history", "a5", "teacher_reference")
DEFAULT_ALPHA_VALUES = (0.25, 0.5, 1.0)

PROTOCOLS = {
    "realistic_random": {
        "fault_onset_step_min": 120,
        "fault_onset_step_max": 700,
        "description": "realistic healthy-to-fault transition",
    },
    "late_random": {
        "fault_onset_step_min": 250,
        "fault_onset_step_max": 700,
        "description": "late random P2 onset",
    },
    "stress_random": {
        "fault_onset_step_min": 30,
        "fault_onset_step_max": 700,
        "description": "early-onset stress protocol, opt-in only",
    },
}

VELOCITY_MODES = {
    "command_random": {
        "description": "command-conditioned vx_cmd in [0.2, 1.5]",
        "target_vx": None,
    },
    "fixed_vx_1p0": {
        "description": "fixed vx_cmd=1.0, vy_cmd=0.0, yaw_cmd=0.0",
        "target_vx": FIXED_VX,
    },
}

PER_RUN_FIELDS = [
    "policy",
    "run_status",
    "protocol",
    "velocity_mode",
    "alpha",
    "checkpoint",
    "base_checkpoint",
    "residual_checkpoint",
    "num_envs",
    "num_steps",
    "seed",
    "control_frequency_hz",
    "control_dt_s",
    "physics_frequency_hz",
    "sim_dt_s",
    "decimation",
    "mean_vel_x_pre_fault",
    "mean_vel_x_post_fault",
    "mean_abs_vx_error_pre_fault",
    "mean_abs_vx_error_post_fault",
    "median_abs_vx_error_post_fault",
    "p90_abs_vx_error_post_fault",
    "survival_rate",
    "done_rate",
    "raw_any_done_rate",
    "raw_never_done_survival_rate",
    "post_fault_alive_sample_fraction",
    "post_fault_done_sample_rate",
    "post_fault_torso_failure_rate",
    "post_fault_timeout_success_rate",
    "post_fault_non_timeout_failure_rate",
    "torso_height_failure_rate",
    "timeout_rate",
    "mean_abs_yaw_error",
    "vx_cmd_mean",
    "vx_cmd_min",
    "vx_cmd_max",
    "command_mode_valid",
    "command_mode_validation_error",
    "fault_active_any_ever",
    "fault_active_sample_count",
    "fault_active_env_count",
    "post_fault_sample_count",
    "fault_active_fraction_last_step",
    "insufficient_post_fault_coverage",
    "smoke_partial_fault_coverage_ok",
    "selected_fault_joint_index_min",
    "selected_fault_joint_index_max",
    "selected_fault_joint_all_8_covered",
    "supported_fault_joint_count",
    "fallback_used",
    "pd_surrogate_used",
    "simulation_override_applied",
    "p2_fault_became_active",
    "no_nan_inf",
    "error_type",
    "error_message",
    "error_summary_json",
    "output_dir",
]

ROLLOUT_FIELDS = [
    "step",
    "policy",
    "protocol",
    "velocity_mode",
    "alpha",
    "fault_onset_step_min",
    "fault_onset_step_max",
    "vx_cmd_mean",
    "vx_cmd_min",
    "vx_cmd_max",
    "reward_mean",
    "mean_vel_x",
    "mean_abs_vx_error",
    "median_abs_vx_error",
    "p90_abs_vx_error",
    "mean_yaw_rate",
    "mean_abs_yaw_error",
    "done_count",
    "total_done_count",
    "P2/fault_applied",
    "P2/simulation_override_applied",
    "P2/fallback_used",
    "P2/onset_step_mean",
    "P2/onset_step_min",
    "P2/onset_step_max",
    "P2/per_env_onset_randomization",
    "P2/selected_fault_joint_index_mean",
    "P2/selected_fault_joint_index_min",
    "P2/selected_fault_joint_index_max",
    "P2/supported_fault_joint_count",
    "P2/multi_joint_randomization",
    "fault_active_sample_count",
    "fault_active_done_count",
    "fault_active_alive_count",
    "base_action_mean_norm",
    "residual_action_mean_norm",
    "residual_action_max_norm",
    "final_action_mean_norm",
    "no_nan_inf",
]

VELOCITY_FIELDS = [
    "step",
    "policy",
    "protocol",
    "velocity_mode",
    "alpha",
    "mean_vel_x",
    "mean_abs_vx_error",
    "mean_vx_cmd",
    "p2_fault_applied",
    "selected_fault_joint_index_mean",
    "done_count",
]

PER_JOINT_FIELDS = [
    "policy",
    "protocol",
    "velocity_mode",
    "alpha",
    "joint_id",
    "joint_name",
    "sample_count",
    "env_count",
    "mean_post_fault_velocity_x",
    "mean_post_fault_abs_vx_error",
    "p90_post_fault_abs_vx_error",
    "survival_rate",
    "done_rate",
    "output_dir",
]


class T17ClosedLoopEvalError(ValueError):
    """Raised for invalid T17 closed-loop evaluation state."""


@dataclass(frozen=True)
class RunSpec:
    policy: str
    protocol: str
    velocity_mode: str
    alpha: float | None
    output_dir: Path
    command: list[str]


BOOLEAN_COMMAND_FLAGS = {
    "--execute_eval",
    "--single_run",
    "--headless",
    "--t18r_pg500_timing",
    "--require_t18r_pg500_timing",
}

VALUE_COMMAND_FLAGS = {
    "--a2_history_checkpoint",
    "--a2_single_step_checkpoint",
    "--a5_checkpoint",
    "--alpha",
    "--checkpoint",
    "--control_frequency_hz",
    "--decimation",
    "--device",
    "--episode_length_s",
    "--history_len",
    "--num_envs",
    "--num_steps",
    "--output_dir",
    "--output_root",
    "--progress_every",
    "--protocol",
    "--require_control_frequency_hz",
    "--run_output_dir",
    "--seed",
    "--sim_dt",
    "--task",
    "--teacher_checkpoint",
    "--velocity_mode",
}

VARIADIC_COMMAND_FLAGS = {
    "--policies",
    "--alpha_values",
}

ALL_COMMAND_FLAGS = BOOLEAN_COMMAND_FLAGS | VALUE_COMMAND_FLAGS | VARIADIC_COMMAND_FLAGS

MALFORMED_RENDER_SUBSTRINGS = (
    "--progress_every100",
    "--velocity_modecommand_random",
    "--alpha 1.0--headless",
    "--alpha 0.5--headless",
    "--alpha 0.25--headless",
    "--a2_history_checkpointpapers",
    "--a2_single_step_checkpointpapers",
    "--a5_checkpointpapers",
    "--teacher_checkpointlogs",
    "--execute_eval--single_run",
)

MERGED_RENDER_PATTERNS = (
    (
        re.compile(r"--[A-Za-z0-9_]+(?:papers/|logs/)"),
        "path flag is immediately followed by a path value",
    ),
    (
        re.compile(r"--[A-Za-z0-9_]+(?:command_random|fixed_vx_1p0|cuda|cpu|\d+(?:\.\d+)?)\b"),
        "flag is immediately followed by a scalar or enum value",
    ),
    (
        re.compile(r"(?:^|\s)\d+(?:\.\d+)?--[A-Za-z0-9_]"),
        "numeric value is immediately followed by another flag",
    ),
    (
        re.compile(r"--[A-Za-z0-9_]+--[A-Za-z0-9_]"),
        "two CLI flags are merged into one rendered token",
    ),
)


def build_parser(*, add_app_launcher_args: bool = False) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="T17 multi-joint closed-loop A2/A5 evaluator scaffold.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry_run", action="store_true", help="Print planned runs without launching Isaac.")
    mode.add_argument("--execute_eval", action="store_true", help="Run planned eval commands.")
    parser.add_argument("--single_run", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--policies", nargs="+", default=list(DEFAULT_POLICIES), choices=POLICY_CHOICES)
    parser.add_argument("--alpha_values", nargs="+", type=float, default=list(DEFAULT_ALPHA_VALUES))
    parser.add_argument("--alpha", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--protocol",
        default="default",
        choices=("default", "realistic_random", "late_random", "stress_random", "all"),
        help="default runs realistic_random and late_random; stress_random is opt-in.",
    )
    parser.add_argument("--velocity_mode", default="command_random", choices=("command_random", "fixed_vx_1p0", "both"))
    parser.add_argument("--a2_single_step_checkpoint", default=DEFAULT_A2_SINGLE_STEP_CHECKPOINT)
    parser.add_argument("--a2_history_checkpoint", default=DEFAULT_A2_HISTORY_CHECKPOINT)
    parser.add_argument("--a5_checkpoint", default=DEFAULT_A5_CHECKPOINT)
    parser.add_argument("--teacher_checkpoint", default=DEFAULT_TEACHER_CHECKPOINT)
    parser.add_argument("--history_len", type=int, default=DEFAULT_HISTORY_LEN)
    parser.add_argument("--num_envs", type=int, default=DEFAULT_NUM_ENVS)
    parser.add_argument("--num_steps", type=int, default=DEFAULT_NUM_STEPS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run_output_dir", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--progress_every", type=int, default=DEFAULT_PROGRESS_EVERY)
    parser.add_argument("--debug_max_runs", type=int, default=0)
    parser.add_argument("--stop_on_failure", action="store_true", default=True)
    add_control_timing_args(parser)
    if not add_app_launcher_args:
        parser.add_argument("--headless", action="store_true", default=True)
        parser.add_argument("--device", default=DEFAULT_DEVICE)
    if add_app_launcher_args:
        from isaaclab.app import AppLauncher

        AppLauncher.add_app_launcher_args(parser)
    return parser


def parse_args() -> argparse.Namespace:
    pre_parser = build_parser(add_app_launcher_args=False)
    pre_args, _ = pre_parser.parse_known_args()
    if pre_args.execute_eval and pre_args.single_run:
        parser = build_parser(add_app_launcher_args=True)
        args, _ = parser.parse_known_args()
        args.headless = True
        return args
    return pre_args


def validate_args(args: argparse.Namespace) -> None:
    if args.num_envs <= 0:
        raise T17ClosedLoopEvalError("--num_envs must be > 0.")
    if args.num_steps <= 0:
        raise T17ClosedLoopEvalError("--num_steps must be > 0.")
    if args.seed < 0:
        raise T17ClosedLoopEvalError("--seed must be non-negative.")
    if args.history_len < 1:
        raise T17ClosedLoopEvalError("--history_len must be >= 1.")
    if args.progress_every <= 0:
        raise T17ClosedLoopEvalError("--progress_every must be > 0.")
    if args.debug_max_runs < 0:
        raise T17ClosedLoopEvalError("--debug_max_runs must be >= 0.")
    if not args.alpha_values:
        raise T17ClosedLoopEvalError("--alpha_values must contain at least one value.")
    for alpha in args.alpha_values:
        if not math.isfinite(float(alpha)):
            raise T17ClosedLoopEvalError("--alpha_values must be finite.")
    if args.single_run and len(args.policies) != 1:
        raise T17ClosedLoopEvalError("--single_run requires exactly one --policies value.")
    if args.single_run and args.protocol in {"default", "all"}:
        raise T17ClosedLoopEvalError("--single_run requires one concrete protocol.")
    if args.single_run and args.velocity_mode == "both":
        raise T17ClosedLoopEvalError("--single_run requires one concrete velocity_mode.")
    if args.single_run and args.policies[0] == "a5" and args.alpha is None:
        raise T17ClosedLoopEvalError("--single_run policy a5 requires --alpha.")
    if (args.t18r_pg500_timing or args.require_t18r_pg500_timing) and any(
        policy in {"a2_history", "a5"} for policy in args.policies
    ):
        try:
            validate_pg500_history_len(args.history_len)
        except ValueError as exc:
            raise T17ClosedLoopEvalError(str(exc)) from exc
    if args.execute_eval:
        requested = set(args.policies)
        checkpoint_pairs = []
        if "a2_single_step" in requested:
            checkpoint_pairs.append(("A2 single-step", args.a2_single_step_checkpoint))
        if "a2_history" in requested or "a5" in requested:
            checkpoint_pairs.append(("A2-history", args.a2_history_checkpoint))
        if "a5" in requested:
            checkpoint_pairs.append(("A5", args.a5_checkpoint))
        if "teacher_reference" in requested:
            checkpoint_pairs.append(("teacher", args.teacher_checkpoint))
        for label, checkpoint in checkpoint_pairs:
            path = resolve_repo_path(checkpoint)
            if not path.is_file():
                raise T17ClosedLoopEvalError(f"{label} checkpoint does not exist: {repo_relative(path)}")


def selected_protocols(args: argparse.Namespace) -> list[str]:
    if args.protocol == "default":
        return ["realistic_random", "late_random"]
    if args.protocol == "all":
        return ["realistic_random", "late_random", "stress_random"]
    return [args.protocol]


def selected_velocity_modes(args: argparse.Namespace) -> list[str]:
    if args.velocity_mode == "both":
        return ["command_random", "fixed_vx_1p0"]
    return [args.velocity_mode]


def alpha_slug(alpha: float | None) -> str:
    if alpha is None:
        return "no_alpha"
    text = f"{float(alpha):.6g}".replace(".", "p").replace("-", "m")
    return f"alpha_{text}"


def policy_output_name(policy: str, alpha: float | None) -> str:
    if policy == "a5":
        return f"a5_{alpha_slug(alpha)}"
    return policy


def flag_value(flag: str, value: object) -> list[str]:
    return [flag, str(value)]


def append_flag_value(command: list[str], flag: str, value: object) -> None:
    command.extend(flag_value(flag, value))


def validate_command_tokens(command_tokens: list[str]) -> None:
    if not isinstance(command_tokens, list) or not all(isinstance(token, str) for token in command_tokens):
        raise RuntimeError("T17 command must be a list[str].")
    if len(command_tokens) < 3:
        raise RuntimeError("T17 command is too short to be a subprocess invocation.")
    for index, token in enumerate(command_tokens):
        if token == "":
            raise RuntimeError(f"T17 command contains an empty token at index {index}.")
        if token.startswith("--"):
            has_merged_flag_marker = "--" in token[2:]
        else:
            has_merged_flag_marker = "--" in token
        if has_merged_flag_marker:
            raise RuntimeError(f"T17 command token contains merged flags: {token!r}.")
        if token.startswith("--"):
            if token not in ALL_COMMAND_FLAGS:
                for known_flag in sorted(ALL_COMMAND_FLAGS, key=len, reverse=True):
                    if token.startswith(known_flag) and token != known_flag:
                        raise RuntimeError(
                            f"T17 command token merges flag and value: {token!r}; expected {known_flag!r} as its own token."
                        )
                raise RuntimeError(f"T17 command contains unknown CLI flag token: {token!r}.")

    index = 0
    while index < len(command_tokens):
        token = command_tokens[index]
        if not token.startswith("--"):
            index += 1
            continue
        if token in BOOLEAN_COMMAND_FLAGS:
            index += 1
            continue
        if token in VALUE_COMMAND_FLAGS:
            value_index = index + 1
            if value_index >= len(command_tokens):
                raise RuntimeError(f"T17 command flag {token!r} is missing its value token.")
            value = command_tokens[value_index]
            if value.startswith("--"):
                raise RuntimeError(f"T17 command flag {token!r} value is missing before {value!r}.")
            index += 2
            continue
        if token in VARIADIC_COMMAND_FLAGS:
            value_index = index + 1
            if value_index >= len(command_tokens) or command_tokens[value_index].startswith("--"):
                raise RuntimeError(f"T17 command flag {token!r} needs at least one value token.")
            index = value_index
            while index < len(command_tokens) and not command_tokens[index].startswith("--"):
                index += 1
            continue
        raise RuntimeError(f"T17 command contains unhandled flag token: {token!r}.")


def render_command(command_tokens: list[str]) -> str:
    validate_command_tokens(command_tokens)
    rendered = shlex.join(command_tokens)
    for malformed in MALFORMED_RENDER_SUBSTRINGS:
        if malformed in rendered:
            raise RuntimeError(f"T17 rendered command contains malformed token sequence: {malformed}")
    for pattern, description in MERGED_RENDER_PATTERNS:
        match = pattern.search(rendered)
        if match:
            raise RuntimeError(
                f"T17 rendered command failed sanity check: {description}: {match.group(0)!r}"
            )
    return rendered


def display_command(command_tokens: list[str]) -> str:
    return render_command(command_tokens)


def build_self_single_run_command(
    *,
    python: str,
    args: argparse.Namespace,
    policy: str,
    protocol: str,
    velocity_mode: str,
    alpha: float | None,
    run_output_dir: Path,
) -> list[str]:
    command = [
        python,
        "evaluators/run_t17_multijoint_closed_loop_eval.py",
        "--execute_eval",
        "--single_run",
    ]
    command.extend(["--policies", policy])
    append_flag_value(command, "--protocol", protocol)
    append_flag_value(command, "--velocity_mode", velocity_mode)
    append_flag_value(command, "--output_dir", args.output_dir)
    append_flag_value(command, "--run_output_dir", repo_relative(run_output_dir))
    append_flag_value(command, "--a2_single_step_checkpoint", args.a2_single_step_checkpoint)
    append_flag_value(command, "--a2_history_checkpoint", args.a2_history_checkpoint)
    append_flag_value(command, "--a5_checkpoint", args.a5_checkpoint)
    append_flag_value(command, "--teacher_checkpoint", args.teacher_checkpoint)
    append_flag_value(command, "--history_len", args.history_len)
    append_flag_value(command, "--num_envs", args.num_envs)
    append_flag_value(command, "--num_steps", args.num_steps)
    append_flag_value(command, "--seed", args.seed)
    append_flag_value(command, "--progress_every", args.progress_every)
    append_optional_timing_flags(command, args)
    if alpha is not None:
        append_flag_value(command, "--alpha", alpha)
    if args.headless:
        command.append("--headless")
    append_flag_value(command, "--device", args.device)
    validate_command_tokens(command)
    return command


def build_teacher_reference_command(
    *,
    python: str,
    args: argparse.Namespace,
    protocol: str,
    velocity_mode: str,
    run_output_dir: Path,
) -> list[str]:
    command = [
        python,
        "evaluators/run_t13c_multijoint_teacher_eval.py",
        "--execute_eval",
        "--single_run",
    ]
    append_flag_value(command, "--checkpoint", args.teacher_checkpoint)
    append_flag_value(command, "--task", TEACHER_TASK)
    append_flag_value(command, "--protocol", protocol)
    append_flag_value(command, "--velocity_mode", velocity_mode)
    append_flag_value(command, "--output_root", args.output_dir)
    append_flag_value(command, "--output_dir", repo_relative(run_output_dir))
    append_flag_value(command, "--num_envs", args.num_envs)
    append_flag_value(command, "--num_steps", args.num_steps)
    append_flag_value(command, "--seed", args.seed)
    append_optional_timing_flags(command, args)
    if args.headless:
        command.append("--headless")
    append_flag_value(command, "--device", args.device)
    validate_command_tokens(command)
    return command


def make_specs(args: argparse.Namespace) -> list[RunSpec]:
    output_root = resolve_repo_path(args.output_dir)
    python = sys.executable
    specs: list[RunSpec] = []
    for protocol in selected_protocols(args):
        for velocity_mode in selected_velocity_modes(args):
            for policy in args.policies:
                alphas: list[float | None] = list(args.alpha_values) if policy == "a5" else [None]
                if args.single_run:
                    alphas = [args.alpha] if policy == "a5" else [None]
                for alpha in alphas:
                    run_output_dir = output_root / protocol / velocity_mode / policy_output_name(policy, alpha)
                    if policy == "teacher_reference":
                        command = build_teacher_reference_command(
                            python=python,
                            args=args,
                            protocol=protocol,
                            velocity_mode=velocity_mode,
                            run_output_dir=run_output_dir,
                        )
                    else:
                        command = build_self_single_run_command(
                            python=python,
                            args=args,
                            policy=policy,
                            protocol=protocol,
                            velocity_mode=velocity_mode,
                            alpha=alpha,
                            run_output_dir=run_output_dir,
                        )
                    specs.append(RunSpec(policy, protocol, velocity_mode, alpha, run_output_dir, command))
    if args.debug_max_runs > 0:
        return specs[: args.debug_max_runs]
    return specs


def append_optional_timing_flags(command: list[str], args: argparse.Namespace) -> None:
    if getattr(args, "t18r_pg500_timing", False):
        command.append("--t18r_pg500_timing")
    if getattr(args, "require_t18r_pg500_timing", False):
        command.append("--require_t18r_pg500_timing")
    for flag, attr_name in (
        ("--control_frequency_hz", "control_frequency_hz"),
        ("--sim_dt", "sim_dt"),
        ("--decimation", "decimation"),
        ("--episode_length_s", "episode_length_s"),
        ("--require_control_frequency_hz", "require_control_frequency_hz"),
    ):
        value = getattr(args, attr_name, None)
        if value is not None:
            append_flag_value(command, flag, value)


def write_csv(path: Path, rows: list[dict[str, Any]], *, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def write_json(path: Path, values: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(values, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def format_md(value: Any) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        return f"{value:.9g}"
    return str(value)


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return float(values[0])
    sorted_values = sorted(float(value) for value in values)
    position = (len(sorted_values) - 1) * q
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def maybe_mean(values: list[float]) -> float | None:
    return float(statistics.mean(values)) if values else None


def row_value(summary: dict[str, Any], key: str) -> Any:
    value = summary.get(key, "")
    return "" if value is None else value


def run_row_from_summary(spec: RunSpec) -> dict[str, Any]:
    summary_path = spec.output_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if spec.policy == "a5":
        checkpoint = summary.get("a5_checkpoint", DEFAULT_A5_CHECKPOINT)
        base_checkpoint = summary.get("a2_history_checkpoint", DEFAULT_A2_HISTORY_CHECKPOINT)
        residual_checkpoint = summary.get("a5_checkpoint", DEFAULT_A5_CHECKPOINT)
    elif spec.policy == "a2_history":
        checkpoint = summary.get("checkpoint", summary.get("a2_history_checkpoint", DEFAULT_A2_HISTORY_CHECKPOINT))
        base_checkpoint = ""
        residual_checkpoint = ""
    elif spec.policy == "a2_single_step":
        checkpoint = summary.get("checkpoint", summary.get("a2_single_step_checkpoint", DEFAULT_A2_SINGLE_STEP_CHECKPOINT))
        base_checkpoint = ""
        residual_checkpoint = ""
    else:
        checkpoint = summary.get("checkpoint", DEFAULT_TEACHER_CHECKPOINT)
        base_checkpoint = ""
        residual_checkpoint = ""
    return {
        "policy": spec.policy,
        "run_status": "completed",
        "protocol": spec.protocol,
        "velocity_mode": spec.velocity_mode,
        "alpha": "" if spec.alpha is None else float(spec.alpha),
        "checkpoint": checkpoint,
        "base_checkpoint": base_checkpoint,
        "residual_checkpoint": residual_checkpoint,
        "num_envs": row_value(summary, "num_envs"),
        "num_steps": row_value(summary, "num_steps"),
        "seed": row_value(summary, "seed"),
        "control_frequency_hz": row_value(summary, "control_frequency_hz"),
        "control_dt_s": row_value(summary, "control_dt_s"),
        "physics_frequency_hz": row_value(summary, "physics_frequency_hz"),
        "sim_dt_s": row_value(summary, "sim_dt_s"),
        "decimation": row_value(summary, "decimation"),
        "mean_vel_x_pre_fault": row_value(summary, "mean_vel_x_pre_fault"),
        "mean_vel_x_post_fault": row_value(summary, "mean_vel_x_post_fault"),
        "mean_abs_vx_error_pre_fault": row_value(summary, "mean_abs_vx_error_pre_fault"),
        "mean_abs_vx_error_post_fault": row_value(summary, "mean_abs_vx_error_post_fault"),
        "median_abs_vx_error_post_fault": row_value(summary, "median_abs_vx_error_post_fault"),
        "p90_abs_vx_error_post_fault": row_value(summary, "p90_abs_vx_error_post_fault"),
        "survival_rate": row_value(summary, "survival_rate"),
        "done_rate": row_value(summary, "done_rate"),
        "raw_any_done_rate": row_value(summary, "raw_any_done_rate"),
        "raw_never_done_survival_rate": row_value(summary, "raw_never_done_survival_rate"),
        "post_fault_alive_sample_fraction": row_value(summary, "post_fault_alive_sample_fraction"),
        "post_fault_done_sample_rate": row_value(summary, "post_fault_done_sample_rate"),
        "post_fault_torso_failure_rate": row_value(summary, "post_fault_torso_failure_rate"),
        "post_fault_timeout_success_rate": row_value(summary, "post_fault_timeout_success_rate"),
        "post_fault_non_timeout_failure_rate": row_value(summary, "post_fault_non_timeout_failure_rate"),
        "torso_height_failure_rate": row_value(summary, "torso_height_failure_rate"),
        "timeout_rate": row_value(summary, "timeout_rate"),
        "mean_abs_yaw_error": row_value(summary, "mean_abs_yaw_error"),
        "vx_cmd_mean": row_value(summary, "vx_cmd_mean"),
        "vx_cmd_min": row_value(summary, "vx_cmd_min"),
        "vx_cmd_max": row_value(summary, "vx_cmd_max"),
        "command_mode_valid": row_value(summary, "command_mode_valid"),
        "command_mode_validation_error": row_value(summary, "command_mode_validation_error"),
        "fault_active_any_ever": row_value(summary, "fault_active_any_ever"),
        "fault_active_sample_count": row_value(summary, "fault_active_sample_count"),
        "fault_active_env_count": row_value(summary, "fault_active_env_count"),
        "post_fault_sample_count": row_value(summary, "post_fault_sample_count"),
        "fault_active_fraction_last_step": row_value(summary, "fault_active_fraction_last_step"),
        "insufficient_post_fault_coverage": row_value(summary, "insufficient_post_fault_coverage"),
        "smoke_partial_fault_coverage_ok": row_value(summary, "smoke_partial_fault_coverage_ok"),
        "selected_fault_joint_index_min": row_value(summary, "selected_fault_joint_index_min"),
        "selected_fault_joint_index_max": row_value(summary, "selected_fault_joint_index_max"),
        "selected_fault_joint_all_8_covered": row_value(summary, "selected_fault_joint_all_8_covered"),
        "supported_fault_joint_count": row_value(summary, "supported_fault_joint_count"),
        "fallback_used": row_value(summary, "fallback_used"),
        "pd_surrogate_used": row_value(summary, "pd_surrogate_used"),
        "simulation_override_applied": row_value(summary, "simulation_override_applied"),
        "p2_fault_became_active": row_value(summary, "p2_fault_became_active"),
        "no_nan_inf": row_value(summary, "no_nan_inf"),
        "error_type": "",
        "error_message": "",
        "error_summary_json": "",
        "output_dir": repo_relative(spec.output_dir),
    }


def failed_row_from_error_summary(spec: RunSpec) -> dict[str, Any]:
    error_path = spec.output_dir / "error_summary.json"
    error_summary = json.loads(error_path.read_text(encoding="utf-8"))
    summary_path = spec.output_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}

    def recovered_value(key: str) -> Any:
        value = row_value(summary, key)
        if value != "":
            return value
        return row_value(error_summary, key)

    return {
        "policy": spec.policy,
        "run_status": "failed",
        "protocol": spec.protocol,
        "velocity_mode": spec.velocity_mode,
        "alpha": "" if spec.alpha is None else float(spec.alpha),
        "checkpoint": recovered_value("checkpoint"),
        "base_checkpoint": row_value(summary, "a2_history_checkpoint"),
        "residual_checkpoint": row_value(summary, "a5_checkpoint"),
        "num_envs": recovered_value("num_envs"),
        "num_steps": recovered_value("num_steps"),
        "seed": recovered_value("seed"),
        "control_frequency_hz": recovered_value("control_frequency_hz"),
        "control_dt_s": recovered_value("control_dt_s"),
        "physics_frequency_hz": recovered_value("physics_frequency_hz"),
        "sim_dt_s": recovered_value("sim_dt_s"),
        "decimation": recovered_value("decimation"),
        "mean_vel_x_pre_fault": recovered_value("mean_vel_x_pre_fault"),
        "mean_vel_x_post_fault": recovered_value("mean_vel_x_post_fault"),
        "mean_abs_vx_error_pre_fault": recovered_value("mean_abs_vx_error_pre_fault"),
        "mean_abs_vx_error_post_fault": recovered_value("mean_abs_vx_error_post_fault"),
        "median_abs_vx_error_post_fault": recovered_value("median_abs_vx_error_post_fault"),
        "p90_abs_vx_error_post_fault": recovered_value("p90_abs_vx_error_post_fault"),
        "survival_rate": recovered_value("survival_rate"),
        "done_rate": recovered_value("done_rate"),
        "raw_any_done_rate": recovered_value("raw_any_done_rate"),
        "raw_never_done_survival_rate": recovered_value("raw_never_done_survival_rate"),
        "post_fault_alive_sample_fraction": recovered_value("post_fault_alive_sample_fraction"),
        "post_fault_done_sample_rate": recovered_value("post_fault_done_sample_rate"),
        "post_fault_torso_failure_rate": recovered_value("post_fault_torso_failure_rate"),
        "post_fault_timeout_success_rate": recovered_value("post_fault_timeout_success_rate"),
        "post_fault_non_timeout_failure_rate": recovered_value("post_fault_non_timeout_failure_rate"),
        "torso_height_failure_rate": recovered_value("torso_height_failure_rate"),
        "timeout_rate": recovered_value("timeout_rate"),
        "mean_abs_yaw_error": recovered_value("mean_abs_yaw_error"),
        "vx_cmd_mean": recovered_value("vx_cmd_mean"),
        "vx_cmd_min": recovered_value("vx_cmd_min"),
        "vx_cmd_max": recovered_value("vx_cmd_max"),
        "command_mode_valid": recovered_value("command_mode_valid"),
        "command_mode_validation_error": recovered_value("command_mode_validation_error"),
        "fault_active_any_ever": recovered_value("fault_active_any_ever"),
        "fault_active_sample_count": recovered_value("fault_active_sample_count"),
        "fault_active_env_count": recovered_value("fault_active_env_count"),
        "post_fault_sample_count": recovered_value("post_fault_sample_count"),
        "fault_active_fraction_last_step": recovered_value("fault_active_fraction_last_step"),
        "insufficient_post_fault_coverage": recovered_value("insufficient_post_fault_coverage"),
        "smoke_partial_fault_coverage_ok": recovered_value("smoke_partial_fault_coverage_ok"),
        "selected_fault_joint_index_min": recovered_value("selected_fault_joint_index_min"),
        "selected_fault_joint_index_max": recovered_value("selected_fault_joint_index_max"),
        "selected_fault_joint_all_8_covered": recovered_value("selected_fault_joint_all_8_covered"),
        "supported_fault_joint_count": recovered_value("supported_fault_joint_count"),
        "fallback_used": recovered_value("fallback_used"),
        "pd_surrogate_used": recovered_value("pd_surrogate_used"),
        "simulation_override_applied": recovered_value("simulation_override_applied"),
        "p2_fault_became_active": recovered_value("p2_fault_became_active"),
        "no_nan_inf": recovered_value("no_nan_inf"),
        "error_type": error_summary.get("error_type", ""),
        "error_message": error_summary.get("error_message", ""),
        "error_summary_json": repo_relative(error_path),
        "output_dir": repo_relative(spec.output_dir),
    }


def write_root_readme(output_root: Path) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    readme_path = output_root / "README.md"
    readme_path.write_text(
        "\n".join(
            [
                "# T17 Multi-Joint Closed-Loop Evaluation",
                "",
                "Preliminary advisor-facing closed-loop scaffold for RLM1 stripped / conference A2 and A5 policies.",
                "",
                "## Scope",
                "",
                "- This is evaluation only, not offline training and not RL training.",
                "- This is candidate-level evidence, not paper-grade final reporting.",
                "- No checkpoint, dataset, task config, or P2 wrapper mutation is performed.",
                "- Latent z_t t-SNE/UMAP is deferred to T18.",
                "",
                "## Policies",
                "",
                f"- A2 single-step: `{DEFAULT_A2_SINGLE_STEP_CHECKPOINT}`",
                f"- A2-history H16: `{DEFAULT_A2_HISTORY_CHECKPOINT}`",
                f"- A5 residual: `{DEFAULT_A5_CHECKPOINT}`",
                "- A5 action rule: `final_action = A2_history(history) + alpha * A5_residual(history)`",
                "- Optional teacher reference delegates to T13C if `teacher_reference` is requested.",
                "",
                "## Fault Semantics",
                "",
                "- one selected locked joint per env/episode",
                "- selected joint random over all 8 Ant actuated joints",
                "- q_lock captured from current selected joint position at onset",
                "- direct `simulation_joint_state_override_lock`",
                "- fallback disabled; PD surrogate disabled",
                "- health token OFF; no explicit fault token into deployment-facing student policies",
                "",
                "## Protocols",
                "",
                "- `realistic_random`: onset U(120,700)",
                "- `late_random`: onset U(250,700)",
                "- `stress_random`: onset U(30,700), opt-in via `--protocol stress_random` or `--protocol all`",
                "",
                "## Velocity Modes",
                "",
                "- `command_random`: primary mode, vx_cmd in [0.2, 1.5]",
                "- `fixed_vx_1p0`: optional comparability mode with guarded vx=1.0 forcing",
                "",
                "## Optional T18-R Timing",
                "",
                "- Pass `--control_frequency_hz 50 --sim_dt 0.01 --decimation 2 --require_control_frequency_hz 50`",
                "  for the corrected 50 Hz / H50 phase.",
                "- Default behavior remains the task timing, currently `sim.dt=1/120` and `decimation=2` in the Ant config.",
                "",
                "## Outputs",
                "",
                "- `results_summary.md`",
                "- `results_summary.json`",
                "- `per_run_metrics.csv`",
                "- `per_joint_metrics.csv`",
                "- `advisor_update_snippet.txt`",
                "- per-run `summary.json`, `rollout_metrics.csv`, `velocity_timeseries.csv`, `per_joint_metrics.csv`, `terminal.log`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return readme_path


def write_command_txt(output_root: Path, command: str) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / "command.txt"
    path.write_text(command + "\n", encoding="utf-8")
    return path


def write_advisor_snippet(output_root: Path, rows: list[dict[str, Any]]) -> Path:
    path = output_root / "advisor_update_snippet.txt"
    completed = sum(1 for row in rows if row.get("run_status", "completed") == "completed")
    failed = sum(1 for row in rows if row.get("run_status") == "failed")
    best = None
    numeric_rows = [
        row
        for row in rows
        if optional_float(row.get("mean_abs_vx_error_post_fault")) is not None
    ]
    if numeric_rows:
        best = min(numeric_rows, key=lambda row: optional_float(row.get("mean_abs_vx_error_post_fault")) or float("inf"))
    lines = [
        "T17 preliminary closed-loop multi-joint P2 evaluation scaffold output.",
        f"Completed runs aggregated: {completed}. Failed runs recorded: {failed}.",
        "Scope: advisor-facing candidate evidence only, not paper-grade final.",
    ]
    if best is not None:
        lines.append(
            "Best observed post-fault vx-error row: "
            f"policy={best.get('policy')} protocol={best.get('protocol')} "
            f"velocity_mode={best.get('velocity_mode')} alpha={best.get('alpha')} "
            f"mean_abs_vx_error_post_fault={best.get('mean_abs_vx_error_post_fault')}."
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_results_summary(output_root: Path, rows: list[dict[str, Any]], per_joint_rows: list[dict[str, Any]]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    csv_path = output_root / "per_run_metrics.csv"
    per_joint_path = output_root / "per_joint_metrics.csv"
    md_path = output_root / "results_summary.md"
    json_path = output_root / "results_summary.json"
    write_csv(csv_path, rows, fieldnames=PER_RUN_FIELDS)
    write_csv(per_joint_path, per_joint_rows, fieldnames=PER_JOINT_FIELDS)
    lines = [
        "# T17 Multi-Joint Closed-Loop Results Summary",
        "",
        "Preliminary advisor-facing candidate evidence only; not paper-grade final.",
        "",
        f"- aggregated rows: `{len(rows)}`",
        f"- completed runs: `{sum(1 for row in rows if row.get('run_status', 'completed') == 'completed')}`",
        f"- failed runs: `{sum(1 for row in rows if row.get('run_status') == 'failed')}`",
        "",
        "| " + " | ".join(PER_RUN_FIELDS) + " |",
        "| " + " | ".join("---" for _ in PER_RUN_FIELDS) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(format_md(row.get(field, "")) for field in PER_RUN_FIELDS) + " |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(
        json_path,
        {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "rlm_phase": RLM_PHASE,
            "not_paper_grade_final": True,
            "aggregated_runs": len(rows),
            "completed_runs": sum(1 for row in rows if row.get("run_status", "completed") == "completed"),
            "failed_runs": sum(1 for row in rows if row.get("run_status") == "failed"),
            "per_run_metrics_csv": repo_relative(csv_path),
            "per_joint_metrics_csv": repo_relative(per_joint_path),
            "rows": rows,
        },
    )
    write_advisor_snippet(output_root, rows)


def aggregate_completed(output_root: Path, specs: list[RunSpec]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    per_joint_rows: list[dict[str, Any]] = []
    for spec in specs:
        summary_path = spec.output_dir / "summary.json"
        error_path = spec.output_dir / "error_summary.json"
        completed_outputs = all(
            (spec.output_dir / name).is_file()
            for name in ("summary.json", "rollout_metrics.csv", "velocity_timeseries.csv", "per_joint_metrics.csv")
        )
        summary_has_error = False
        if summary_path.is_file():
            try:
                summary_has_error = bool(json.loads(summary_path.read_text(encoding="utf-8")).get("error_type"))
            except json.JSONDecodeError:
                summary_has_error = True
        if summary_path.is_file() and completed_outputs and not summary_has_error:
            rows.append(run_row_from_summary(spec))
        elif error_path.is_file():
            rows.append(failed_row_from_error_summary(spec))
        per_joint_path = spec.output_dir / "per_joint_metrics.csv"
        if per_joint_path.is_file():
            with per_joint_path.open(newline="", encoding="utf-8") as stream:
                for row in csv.DictReader(stream):
                    normalized = {field: row.get(field, "") for field in PER_JOINT_FIELDS}
                    normalized["policy"] = normalized.get("policy") or spec.policy
                    normalized["protocol"] = normalized.get("protocol") or spec.protocol
                    normalized["velocity_mode"] = normalized.get("velocity_mode") or spec.velocity_mode
                    normalized["alpha"] = normalized.get("alpha") or ("" if spec.alpha is None else str(spec.alpha))
                    per_joint_rows.append(normalized)
    write_results_summary(output_root, rows, per_joint_rows)
    return rows, per_joint_rows


def add_trainers_to_path() -> None:
    trainers_dir = REPO_ROOT / "trainers"
    if str(trainers_dir) not in sys.path:
        sys.path.insert(0, str(trainers_dir))


def torch_load_checkpoint(path: Path, *, map_location: str) -> dict[str, Any]:
    import torch

    try:
        checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=map_location)
    if not isinstance(checkpoint, dict):
        raise T17ClosedLoopEvalError(f"checkpoint must be a dict, got {type(checkpoint).__name__}.")
    return checkpoint


def assert_finite_tensor(name: str, tensor: Any) -> None:
    import torch

    if not torch.isfinite(tensor).all():
        raise T17ClosedLoopEvalError(f"{name} contains NaN or Inf.")


def assert_action_shape(name: str, tensor: Any, *, num_envs: int) -> None:
    if tensor.ndim != 2 or int(tensor.shape[0]) != num_envs or int(tensor.shape[1]) != EXPECTED_ACTION_DIM:
        raise T17ClosedLoopEvalError(
            f"{name} expected shape [{num_envs}, {EXPECTED_ACTION_DIM}], got {tuple(tensor.shape)}."
        )


def checkpoint_guardrails(
    checkpoint: dict[str, Any],
    *,
    checkpoint_name: str,
    require_residual_head: bool,
    require_no_residual_head: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    metrics = checkpoint.get("metrics")
    if not isinstance(metrics, dict):
        raise T17ClosedLoopEvalError(f"{checkpoint_name} checkpoint missing metrics metadata.")
    guardrails = checkpoint.get("guardrails") if isinstance(checkpoint.get("guardrails"), dict) else {}
    forbidden_keys = (
        "teacher_obs_used_as_input",
        "true_fault_state_used",
        "selected_fault_joint_index_used_as_input",
        "selected_fault_joint_one_hot_used_as_input",
        "q_lock_vector_used_as_input",
        "p2_fault_active_used_as_input",
        "health_token_used",
        "a0_action_used",
    )
    for key in forbidden_keys:
        if bool(metrics.get(key, False)) or bool(guardrails.get(key, False)):
            raise T17ClosedLoopEvalError(f"{checkpoint_name} checkpoint guardrail violated: {key}=true")
    if metrics.get("no_nan_inf") is False:
        raise T17ClosedLoopEvalError(f"{checkpoint_name} checkpoint metrics.no_nan_inf is false.")
    if require_no_residual_head and bool(metrics.get("residual_head_used", False)):
        raise T17ClosedLoopEvalError(f"{checkpoint_name} must not have residual_head_used=true.")
    if require_residual_head and bool(metrics.get("residual_head_used")) is not True:
        raise T17ClosedLoopEvalError(f"{checkpoint_name} must have residual_head_used=true.")
    return metrics, guardrails


def load_a2_model(*, policy: str, checkpoint_path: Path, history_len: int, device: Any) -> tuple[Any, dict[str, Any]]:
    import torch

    add_trainers_to_path()
    from train_t15_multijoint_a2_student_distill import build_history_model, build_single_step_model

    checkpoint = torch_load_checkpoint(checkpoint_path, map_location="cpu")
    metrics, guardrails = checkpoint_guardrails(
        checkpoint,
        checkpoint_name=policy,
        require_residual_head=False,
        require_no_residual_head=True,
    )
    mode = checkpoint.get("mode", metrics.get("mode"))
    if policy == "a2_single_step" and mode != "single_step":
        raise T17ClosedLoopEvalError(f"A2 single-step checkpoint mode expected single_step, got {mode!r}.")
    if policy == "a2_history" and mode != "history":
        raise T17ClosedLoopEvalError(f"A2-history checkpoint mode expected history, got {mode!r}.")
    input_dim = checkpoint.get("input_dim", metrics.get("input_dim"))
    target_dim = checkpoint.get("target_dim", metrics.get("target_dim", metrics.get("action_dim")))
    if int(input_dim) != EXPECTED_STUDENT_OBS_DIM or int(target_dim) != EXPECTED_ACTION_DIM:
        raise T17ClosedLoopEvalError(f"{policy} checkpoint dims expected 61 -> 8.")
    if policy == "a2_history":
        checkpoint_history_len = checkpoint.get("history_len", metrics.get("history_len"))
        if int(checkpoint_history_len) != int(history_len):
            raise T17ClosedLoopEvalError(
                f"A2-history checkpoint history_len expected {history_len}, got {checkpoint_history_len}."
            )
        model = build_history_model(history_len, EXPECTED_STUDENT_OBS_DIM, EXPECTED_ACTION_DIM)
    else:
        model = build_single_step_model(EXPECTED_STUDENT_OBS_DIM, EXPECTED_ACTION_DIM)
    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(state_dict, dict):
        raise T17ClosedLoopEvalError(f"{policy} checkpoint missing model_state_dict.")
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    with torch.inference_mode():
        dummy = (
            torch.zeros((1, history_len, EXPECTED_STUDENT_OBS_DIM), dtype=torch.float32, device=device)
            if policy == "a2_history"
            else torch.zeros((1, EXPECTED_STUDENT_OBS_DIM), dtype=torch.float32, device=device)
        )
        action = model(dummy)
        assert_action_shape(f"{policy}_dummy_action", action, num_envs=1)
        assert_finite_tensor(f"{policy}_dummy_action", action)
    return model, {"checkpoint_metrics": metrics, "checkpoint_guardrails": guardrails}


def load_a5_model(*, checkpoint_path: Path, history_len: int, device: Any) -> tuple[Any, dict[str, Any]]:
    import torch

    add_trainers_to_path()
    from train_t16_multijoint_a5_history_residual import RESIDUAL_INPUT_MODE, build_residual_model

    checkpoint = torch_load_checkpoint(checkpoint_path, map_location="cpu")
    metrics, guardrails = checkpoint_guardrails(
        checkpoint,
        checkpoint_name="A5",
        require_residual_head=True,
        require_no_residual_head=False,
    )
    checkpoint_history_len = checkpoint.get("history_len", metrics.get("history_len"))
    if int(checkpoint_history_len) != int(history_len):
        raise T17ClosedLoopEvalError(
            f"A5 checkpoint history_len expected {history_len}, got {checkpoint_history_len}."
        )
    residual_input_mode = checkpoint.get("residual_input_mode", metrics.get("residual_input_mode"))
    if residual_input_mode != RESIDUAL_INPUT_MODE:
        raise T17ClosedLoopEvalError(
            f"A5 residual_input_mode expected {RESIDUAL_INPUT_MODE!r}, got {residual_input_mode!r}."
        )
    if bool(metrics.get("base_action_used_as_residual_input", False)) or bool(
        checkpoint.get("base_action_used_as_residual_input", False)
    ):
        raise T17ClosedLoopEvalError("T16 A5 residual must not use base_action as residual input.")
    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(state_dict, dict):
        raise T17ClosedLoopEvalError("A5 checkpoint missing model_state_dict.")
    model = build_residual_model(history_len, EXPECTED_STUDENT_OBS_DIM, EXPECTED_RESIDUAL_DIM)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    with torch.inference_mode():
        dummy = torch.zeros((1, history_len, EXPECTED_STUDENT_OBS_DIM), dtype=torch.float32, device=device)
        residual = model(dummy)
        assert_action_shape("a5_dummy_residual", residual, num_envs=1)
        assert_finite_tensor("a5_dummy_residual", residual)
    return model, {
        "a5_checkpoint_metrics": metrics,
        "a5_checkpoint_guardrails": guardrails,
        "residual_input_mode": residual_input_mode,
    }


def select_policy_device(args: argparse.Namespace, env_device: Any) -> Any:
    import torch

    requested = getattr(args, "device", None) or "cuda"
    if str(requested).startswith("cuda") and not torch.cuda.is_available():
        print("[T17 WARNING] CUDA unavailable; using CPU for policy inference.", flush=True)
        requested = "cpu"
    return torch.device(requested if requested is not None else env_device)


def safe_student_obs(obs: Any) -> Any:
    student_obs = policy_tensor(obs).detach().float()
    if student_obs.ndim != 2 or student_obs.shape[-1] != EXPECTED_STUDENT_OBS_DIM:
        raise T17ClosedLoopEvalError(
            f"student-safe observation dim expected {EXPECTED_STUDENT_OBS_DIM}, got {tuple(student_obs.shape)}."
        )
    assert_finite_tensor("student_obs", student_obs)
    return student_obs


def done_mask_tensor(dones: Any, *, num_envs: int, device: Any) -> Any:
    import torch

    tensor = torch.as_tensor(dones, device=device)
    if tensor.ndim == 0:
        tensor = tensor.repeat(num_envs)
    if tensor.ndim > 1:
        tensor = tensor.reshape(tensor.shape[0], -1).any(dim=1)
    if int(tensor.shape[0]) != num_envs:
        raise T17ClosedLoopEvalError(f"done tensor length mismatch: expected {num_envs}, got {int(tensor.shape[0])}.")
    return tensor.to(dtype=torch.bool)


def configure_velocity_commands(env_cfg: Any, velocity_mode: str) -> dict[str, Any]:
    try:
        ranges = env_cfg.commands.base_velocity.ranges
        if velocity_mode == "command_random":
            ranges.lin_vel_x = COMMAND_RANDOM_RANGE
            ranges.lin_vel_y = (0.0, 0.0)
            ranges.ang_vel_z = (0.0, 0.0)
            ranges.heading = (0.0, 0.0)
            return {"configured_vx_cmd_range": list(COMMAND_RANDOM_RANGE), "configured_fixed_command": False}
        ranges.lin_vel_x = (FIXED_VX, FIXED_VX)
        ranges.lin_vel_y = (0.0, 0.0)
        ranges.ang_vel_z = (0.0, 0.0)
        ranges.heading = (0.0, 0.0)
        return {"configured_vx_cmd_range": [FIXED_VX, FIXED_VX], "configured_fixed_command": True}
    except Exception as exc:
        raise T17ClosedLoopEvalError(f"failed to configure velocity command ranges: {exc}") from exc


def per_joint_rows_from_samples(
    *,
    policy: str,
    protocol: str,
    velocity_mode: str,
    alpha: float | None,
    joint_names: tuple[str, ...],
    output_dir: Path,
    samples: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for joint_id, joint_name in enumerate(joint_names):
        values = samples.get(
            joint_id,
            {"vel_x": [], "abs_vx_error": [], "done": [], "env_ids": set()},
        )
        done_values = values.get("done", [])
        done_rate = maybe_mean([float(done) for done in done_values])
        rows.append(
            {
                "policy": policy,
                "protocol": protocol,
                "velocity_mode": velocity_mode,
                "alpha": "" if alpha is None else float(alpha),
                "joint_id": joint_id,
                "joint_name": joint_name,
                "sample_count": len(values.get("vel_x", [])),
                "env_count": len(values.get("env_ids", set())),
                "mean_post_fault_velocity_x": maybe_mean(values.get("vel_x", [])),
                "mean_post_fault_abs_vx_error": maybe_mean(values.get("abs_vx_error", [])),
                "p90_post_fault_abs_vx_error": percentile(values.get("abs_vx_error", []), 0.9),
                "survival_rate": None if done_rate is None else 1.0 - done_rate,
                "done_rate": done_rate,
                "output_dir": repo_relative(output_dir),
            }
        )
    return rows


def run_single_policy_rollout(args: argparse.Namespace) -> int:
    import gymnasium as gym
    import torch

    import isaaclab_tasks  # noqa: F401
    from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    trainers_dir = REPO_ROOT / "trainers"
    if str(trainers_dir) not in sys.path:
        sys.path.insert(0, str(trainers_dir))
    from p2_joint_lock_training_wrapper import P2JointLockActionMaskWrapper

    policy = args.policies[0]
    alpha = float(args.alpha) if policy == "a5" else None
    output_dir = resolve_repo_path(args.run_output_dir or args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")

    protocol_cfg = PROTOCOLS[args.protocol]
    rows: list[dict[str, Any]] = []
    velocity_rows: list[dict[str, Any]] = []
    post_fault_abs_errors: list[float] = []
    per_joint_samples: dict[int, dict[str, Any]] = {}
    selected_joint_ids_seen: set[int] = set()
    env = None
    vec_env = None
    p2_wrapper = None
    summary: dict[str, Any] = {
        "eval_scope": "t17_multijoint_closed_loop_eval",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "not_paper_grade_final": True,
        "rlm_phase": RLM_PHASE,
        "policy": policy,
        "protocol": args.protocol,
        "velocity_mode": args.velocity_mode,
        "alpha": alpha,
        "task": TASK,
        "num_envs": args.num_envs,
        "num_steps": args.num_steps,
        "seed": args.seed,
        "fault_profile": FAULT_PROFILE,
        "target_joint_mode": TARGET_JOINT_MODE,
        "requested_semantics": REQUESTED_SEMANTICS,
        "fault_onset_mode": "random_uniform",
        "fault_onset_step_min": protocol_cfg["fault_onset_step_min"],
        "fault_onset_step_max": protocol_cfg["fault_onset_step_max"],
        "fallback_allowed": False,
        "pd_surrogate_allowed": False,
        "health_token_enabled": False,
        "explicit_fault_token_into_student_policy": False,
        "student_policy_obs_dim": EXPECTED_STUDENT_OBS_DIM,
        "action_dim": EXPECTED_ACTION_DIM,
        "history_len": args.history_len if policy in {"a2_history", "a5"} else None,
        "metric_notes": [],
    }

    try:
        print("[T17 WARNING] Preliminary advisor-facing evaluation only; not paper-grade final.", flush=True)
        print(f"[T17] single_run policy={policy} protocol={args.protocol} velocity_mode={args.velocity_mode}", flush=True)
        env_cfg = load_cfg_from_registry(TASK, "env_cfg_entry_point")
        agent_cfg = load_cfg_from_registry(TASK, "rsl_rl_cfg_entry_point")
        env_cfg.scene.num_envs = args.num_envs
        env_cfg.seed = args.seed
        if getattr(args, "device", None):
            env_cfg.sim.device = args.device
        env_cfg.log_dir = str(output_dir / "isaac_logs")
        summary.update(configure_velocity_commands(env_cfg, args.velocity_mode))
        control_timing = apply_control_timing_to_env_cfg(env_cfg, args)
        summary.update(control_timing)
        print(
            "[T17] control timing "
            f"frequency_hz={control_timing['control_frequency_hz']:.9g} "
            f"physics_frequency_hz={control_timing['physics_frequency_hz']:.9g} "
            f"control_dt_s={control_timing['control_dt_s']:.9g} "
            f"sim_dt_s={control_timing['sim_dt_s']:.9g} "
            f"decimation={control_timing['decimation']}",
            flush=True,
        )

        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

        print("[T17] gym.make start", flush=True)
        env = gym.make(TASK, cfg=env_cfg)
        if isinstance(env.unwrapped, DirectMARLEnv):
            env = multi_agent_to_single_agent(env)
        print("[T17] gym.make done", flush=True)

        print("[T17] P2 wrapper attach start", flush=True)
        p2_wrapper = P2JointLockActionMaskWrapper(
            env,
            target_joint=TARGET_JOINT_PLACEHOLDER,
            target_joint_mode=TARGET_JOINT_MODE,
            fault_onset_step=protocol_cfg["fault_onset_step_min"],
            fault_onset_mode="random_uniform",
            fault_onset_step_min=protocol_cfg["fault_onset_step_min"],
            fault_onset_step_max=protocol_cfg["fault_onset_step_max"],
            expected_action_dim=EXPECTED_ACTION_DIM,
            requested_semantics=REQUESTED_SEMANTICS,
            allow_fallback=False,
            velocity_override=0.0,
            debug=True,
        )
        env = p2_wrapper
        if p2_wrapper.mapping.semantics != REQUESTED_SEMANTICS:
            raise T17ClosedLoopEvalError(f"actual P2 semantics must be {REQUESTED_SEMANTICS}.")
        if p2_wrapper.mapping.target_joint_mode != TARGET_JOINT_MODE:
            raise T17ClosedLoopEvalError(f"target_joint_mode must be {TARGET_JOINT_MODE}.")
        if p2_wrapper.mapping.allow_fallback or p2_wrapper.fallback_used:
            raise T17ClosedLoopEvalError("fallback must remain disabled.")
        if len(p2_wrapper.mapping.supported_target_joints) != EXPECTED_ACTION_DIM:
            raise T17ClosedLoopEvalError("multi-joint eval must cover all 8 supported actuated joints.")
        print("[T17] P2 wrapper attach done", flush=True)

        if args.velocity_mode == "fixed_vx_1p0":
            fixed_info = force_fixed_velocity_command(env, FIXED_VX)
            summary.update(fixed_info)
            if not fixed_info.get("target_vx_available"):
                raise T17ClosedLoopEvalError("fixed_vx_1p0 requires writable command tensor.")
        else:
            summary.update(
                {
                    "target_vx_available": False,
                    "target_vx": None,
                    "command_forced_each_step": False,
                    "target_vx_mode_note": "command_random leaves the configured command range active",
                }
            )

        agent_cfg, _ = prepare_agent_cfg(agent_cfg)
        vec_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        if int(vec_env.num_actions) != EXPECTED_ACTION_DIM:
            raise T17ClosedLoopEvalError(f"action dim expected {EXPECTED_ACTION_DIM}, got {int(vec_env.num_actions)}.")
        tracking_num_envs = int(vec_env.num_envs)
        tracking_device = vec_env.unwrapped.device
        policy_device = select_policy_device(args, tracking_device)

        a2_single_model = None
        a2_history_model = None
        a5_model = None
        if policy == "a2_single_step":
            a2_single_model, model_metadata = load_a2_model(
                policy=policy,
                checkpoint_path=resolve_repo_path(args.a2_single_step_checkpoint),
                history_len=args.history_len,
                device=policy_device,
            )
            summary["checkpoint"] = repo_relative(args.a2_single_step_checkpoint)
            summary["policy_metadata"] = model_metadata
        elif policy == "a2_history":
            a2_history_model, model_metadata = load_a2_model(
                policy=policy,
                checkpoint_path=resolve_repo_path(args.a2_history_checkpoint),
                history_len=args.history_len,
                device=policy_device,
            )
            summary["checkpoint"] = repo_relative(args.a2_history_checkpoint)
            summary["a2_history_checkpoint"] = repo_relative(args.a2_history_checkpoint)
            summary["policy_metadata"] = model_metadata
        elif policy == "a5":
            a2_history_model, a2_metadata = load_a2_model(
                policy="a2_history",
                checkpoint_path=resolve_repo_path(args.a2_history_checkpoint),
                history_len=args.history_len,
                device=policy_device,
            )
            a5_model, a5_metadata = load_a5_model(
                checkpoint_path=resolve_repo_path(args.a5_checkpoint),
                history_len=args.history_len,
                device=policy_device,
            )
            summary["checkpoint"] = repo_relative(args.a5_checkpoint)
            summary["a2_history_checkpoint"] = repo_relative(args.a2_history_checkpoint)
            summary["a5_checkpoint"] = repo_relative(args.a5_checkpoint)
            summary["action_rule"] = "base_action = A2_history(history); final_action = base_action + alpha * A5_residual(history)"
            summary["policy_metadata"] = {"a2": a2_metadata, "a5": a5_metadata}
        else:
            raise T17ClosedLoopEvalError(f"single-run policy {policy!r} is not handled by this rollout path.")

        if args.velocity_mode == "fixed_vx_1p0":
            force_fixed_velocity_command(env, FIXED_VX)
        obs = vec_env.get_observations()
        if args.velocity_mode == "fixed_vx_1p0":
            force_fixed_velocity_command(env, FIXED_VX)
            obs = vec_env.get_observations()
        student_obs = safe_student_obs(obs)
        if int(student_obs.shape[0]) != tracking_num_envs:
            raise T17ClosedLoopEvalError("student_obs env count mismatch.")
        history_buffer = None
        if policy in {"a2_history", "a5"}:
            history_buffer = student_obs.to(policy_device).unsqueeze(1).repeat(1, args.history_len, 1)
            assert_finite_tensor("history_buffer", history_buffer)

        p2_wrapper._ensure_lock_buffers()
        initial_onset_steps = current_onset_steps(p2_wrapper, num_envs=tracking_num_envs)
        first_done_step = torch.full((tracking_num_envs,), -1, dtype=torch.long, device=tracking_device)
        p2_action_term_name = p2_wrapper.mapping.action_term_name
        total_done_count = 0
        reward_sum = 0.0
        reward_count = 0
        no_nan_inf = tensors_are_finite(obs)
        timeout_values: list[float] = []
        torso_height_failure_values: list[float] = []
        fallback_values: list[float] = []
        override_values: list[float] = []
        p2_fault_values: list[float] = []
        fault_active_any_ever = False
        fault_active_sample_count = 0
        fault_active_done_sample_count = 0
        fault_active_alive_sample_count = 0
        fault_active_env_ids: set[int] = set()
        fault_active_fraction_last_step = 0.0
        velocity_source = None
        yaw_source = None
        residual_mean_norm_values: list[float] = []
        residual_max_norm_values: list[float] = []
        base_mean_norm_values: list[float] = []
        final_mean_norm_values: list[float] = []

        print("[T17] rollout start", flush=True)
        for step_index in range(args.num_steps):
            step_number = step_index + 1
            if args.velocity_mode == "fixed_vx_1p0":
                force_fixed_velocity_command(env, FIXED_VX)
                obs = vec_env.get_observations()
            student_obs = safe_student_obs(obs)
            with torch.inference_mode():
                if policy == "a2_single_step":
                    assert a2_single_model is not None
                    final_action = a2_single_model(student_obs.to(policy_device)).detach().float()
                    base_action = None
                    residual_action = None
                elif policy == "a2_history":
                    assert a2_history_model is not None and history_buffer is not None
                    final_action = a2_history_model(history_buffer).detach().float()
                    base_action = None
                    residual_action = None
                else:
                    assert a2_history_model is not None and a5_model is not None and history_buffer is not None
                    base_action = a2_history_model(history_buffer).detach().float()
                    residual_action = a5_model(history_buffer).detach().float()
                    final_action = base_action + float(alpha) * residual_action
                assert_action_shape("final_action", final_action, num_envs=tracking_num_envs)
                assert_finite_tensor("final_action", final_action)
                if base_action is not None:
                    assert_action_shape("base_action", base_action, num_envs=tracking_num_envs)
                    assert_finite_tensor("base_action", base_action)
                    base_norm = base_action.norm(dim=1)
                    base_mean_norm_values.append(float(base_norm.mean().detach().cpu().item()))
                if residual_action is not None:
                    assert_action_shape("residual_action", residual_action, num_envs=tracking_num_envs)
                    assert_finite_tensor("residual_action", residual_action)
                    residual_norm = residual_action.norm(dim=1)
                    residual_mean_norm_values.append(float(residual_norm.mean().detach().cpu().item()))
                    residual_max_norm_values.append(float(residual_norm.max().detach().cpu().item()))
                final_norm = final_action.norm(dim=1)
                final_mean_norm_values.append(float(final_norm.mean().detach().cpu().item()))
                next_obs, rewards, dones, extras = vec_env.step(final_action.to(tracking_device))
                if args.velocity_mode == "fixed_vx_1p0":
                    force_fixed_velocity_command(env, FIXED_VX)
                    next_obs = vec_env.get_observations()

            reward_tensor = torch.as_tensor(rewards, device=tracking_device).detach().float()
            no_nan_inf = no_nan_inf and tensors_are_finite(reward_tensor) and tensors_are_finite(next_obs)
            reward_sum += float(reward_tensor.sum().cpu().item())
            reward_count += int(reward_tensor.numel())
            done_mask = done_mask_tensor(dones, num_envs=tracking_num_envs, device=tracking_device)
            step_done_count = int(done_mask.sum().detach().cpu().item())
            total_done_count += step_done_count
            new_done = torch.logical_and(done_mask, first_done_step < 0)
            first_done_step[new_done] = step_number

            next_student_obs = safe_student_obs(next_obs)
            if history_buffer is not None:
                next_student_obs_policy = next_student_obs.to(policy_device)
                history_buffer = torch.roll(history_buffer, shifts=-1, dims=1)
                history_buffer[:, -1, :] = next_student_obs_policy
                if bool(done_mask.any().item()):
                    done_policy = done_mask.to(policy_device)
                    history_buffer[done_policy] = next_student_obs_policy[done_policy].unsqueeze(1).repeat(
                        1,
                        args.history_len,
                        1,
                    )
                assert_finite_tensor("history_buffer", history_buffer)

            log_values = extras.get("log", {}) if isinstance(extras, dict) else {}
            if isinstance(log_values, dict):
                timeout_value = first_matching_log_value(log_values, ("time", "out"))
                if timeout_value is not None:
                    timeout_values.append(timeout_value)
                torso_value = first_matching_log_value(log_values, ("torso", "height"))
                if torso_value is not None:
                    torso_height_failure_values.append(torso_value)

            fallback_value = scalar(log_values.get("P2/fallback_used"))
            override_value = scalar(log_values.get("P2/simulation_override_applied"))
            fallback_values.append(fallback_value)
            override_values.append(override_value)
            if p2_wrapper.fallback_used or fallback_value > 0.0:
                raise T17ClosedLoopEvalError("P2 fallback was used during evaluation.")

            fault_mask = getattr(p2_wrapper, "last_fault_applied_mask", None)
            if fault_mask is None:
                fault_mask = torch.zeros(tracking_num_envs, dtype=torch.bool, device=tracking_device)
            else:
                fault_mask = torch.as_tensor(fault_mask, device=tracking_device).to(dtype=torch.bool).reshape(
                    tracking_num_envs
                )
            p2_fault_active_mean = float(fault_mask.float().mean().detach().cpu().item())
            p2_fault_values.append(p2_fault_active_mean)
            fault_active_fraction_last_step = p2_fault_active_mean
            fault_active_count_this_step = int(fault_mask.sum().detach().cpu().item())
            if fault_active_count_this_step > 0:
                fault_active_any_ever = True
                fault_active_sample_count += fault_active_count_this_step
                fault_active_done_count_this_step = int(
                    torch.logical_and(fault_mask, done_mask).sum().detach().cpu().item()
                )
                fault_active_done_sample_count += fault_active_done_count_this_step
                fault_active_alive_sample_count += fault_active_count_this_step - fault_active_done_count_this_step
                active_env_indices = torch.nonzero(fault_mask, as_tuple=False).reshape(-1).detach().cpu().tolist()
                fault_active_env_ids.update(int(index) for index in active_env_indices)
            else:
                fault_active_done_count_this_step = 0
            if bool(fault_mask.any().item()) and override_value <= 0.0:
                raise T17ClosedLoopEvalError("P2 simulation override was not applied after onset.")

            velocity_x, step_velocity_source = resolve_forward_velocity_tensor(
                env,
                extras,
                action_term_name=p2_action_term_name,
                num_envs=tracking_num_envs,
            )
            target_vx, _ = target_vx_tensor(env, num_envs=tracking_num_envs)
            yaw_metrics = resolve_yaw_rate_metrics(env, extras, action_term_name=p2_action_term_name)
            if velocity_source is None:
                velocity_source = step_velocity_source
            if yaw_source is None and yaw_metrics["yaw_metric_source"] is not None:
                yaw_source = yaw_metrics["yaw_metric_source"]
            assert_finite_tensor("velocity_x", velocity_x)
            assert_finite_tensor("target_vx", target_vx)
            vx_abs_error = (velocity_x - target_vx).abs()
            assert_finite_tensor("vx_abs_error", vx_abs_error)

            cmd_stats = command_vx_stats(env)
            selected_indices = tensor_to_int_list(p2_wrapper.per_env_target_action_index)
            fault_mask_list = tensor_to_int_list(fault_mask)
            yaw_per_env = per_env_yaw_rate(env, action_term_name=p2_action_term_name)
            velocity_list = [float(value) for value in velocity_x.detach().cpu().tolist()]
            error_list = [float(value) for value in vx_abs_error.detach().cpu().tolist()]
            selected_index_mean = optional_float(log_values.get("P2/selected_fault_joint_index_mean"))
            selected_index_min = optional_float(log_values.get("P2/selected_fault_joint_index_min"))
            selected_index_max = optional_float(log_values.get("P2/selected_fault_joint_index_max"))
            if selected_indices is not None:
                selected_joint_ids_seen.update(int(value) for value in selected_indices)

            if selected_indices is not None and fault_mask_list is not None:
                for env_index, is_fault_active in enumerate(fault_mask_list):
                    if env_index >= len(selected_indices):
                        continue
                    joint_index = int(selected_indices[env_index])
                    values = per_joint_samples.setdefault(
                        joint_index,
                        {"vel_x": [], "abs_vx_error": [], "done": [], "env_ids": set()},
                    )
                    values["env_ids"].add(env_index)
                    if not is_fault_active:
                        continue
                    if env_index < len(velocity_list):
                        values["vel_x"].append(velocity_list[env_index])
                    if env_index < len(error_list):
                        values["abs_vx_error"].append(error_list[env_index])
                        post_fault_abs_errors.append(error_list[env_index])
                    values["done"].append(bool(done_mask[env_index].detach().cpu().item()))
                    if yaw_per_env is not None and env_index < len(yaw_per_env):
                        values.setdefault("abs_yaw_error", []).append(abs(yaw_per_env[env_index]))

            row_abs_errors = error_list
            row = {
                "step": step_number,
                "policy": policy,
                "protocol": args.protocol,
                "velocity_mode": args.velocity_mode,
                "alpha": "" if alpha is None else float(alpha),
                "fault_onset_step_min": protocol_cfg["fault_onset_step_min"],
                "fault_onset_step_max": protocol_cfg["fault_onset_step_max"],
                "vx_cmd_mean": cmd_stats["vx_cmd_mean"],
                "vx_cmd_min": cmd_stats["vx_cmd_min"],
                "vx_cmd_max": cmd_stats["vx_cmd_max"],
                "reward_mean": scalar(rewards),
                "mean_vel_x": float(velocity_x.mean().detach().cpu().item()),
                "mean_abs_vx_error": float(vx_abs_error.mean().detach().cpu().item()),
                "median_abs_vx_error": percentile(row_abs_errors, 0.5),
                "p90_abs_vx_error": percentile(row_abs_errors, 0.9),
                "mean_yaw_rate": yaw_metrics["mean_yaw_rate"],
                "mean_abs_yaw_error": yaw_metrics["mean_abs_yaw_error"],
                "done_count": step_done_count,
                "total_done_count": total_done_count,
                "P2/fault_applied": p2_fault_active_mean,
                "P2/simulation_override_applied": override_value,
                "P2/fallback_used": fallback_value,
                "P2/onset_step_mean": optional_float(log_values.get("P2/onset_step_mean")),
                "P2/onset_step_min": optional_float(log_values.get("P2/onset_step_min")),
                "P2/onset_step_max": optional_float(log_values.get("P2/onset_step_max")),
                "P2/per_env_onset_randomization": optional_float(log_values.get("P2/per_env_onset_randomization")),
                "P2/selected_fault_joint_index_mean": selected_index_mean,
                "P2/selected_fault_joint_index_min": selected_index_min,
                "P2/selected_fault_joint_index_max": selected_index_max,
                "P2/supported_fault_joint_count": optional_float(log_values.get("P2/supported_fault_joint_count")),
                "P2/multi_joint_randomization": optional_float(log_values.get("P2/multi_joint_randomization")),
                "fault_active_sample_count": fault_active_count_this_step,
                "fault_active_done_count": fault_active_done_count_this_step,
                "fault_active_alive_count": fault_active_count_this_step - fault_active_done_count_this_step,
                "base_action_mean_norm": maybe_mean(base_mean_norm_values[-1:]),
                "residual_action_mean_norm": maybe_mean(residual_mean_norm_values[-1:]),
                "residual_action_max_norm": max(residual_max_norm_values[-1:]) if residual_max_norm_values else None,
                "final_action_mean_norm": maybe_mean(final_mean_norm_values[-1:]),
                "no_nan_inf": bool(no_nan_inf),
            }
            rows.append(row)
            velocity_rows.append(
                {
                    "step": step_number,
                    "policy": policy,
                    "protocol": args.protocol,
                    "velocity_mode": args.velocity_mode,
                    "alpha": "" if alpha is None else float(alpha),
                    "mean_vel_x": row["mean_vel_x"],
                    "mean_abs_vx_error": row["mean_abs_vx_error"],
                    "mean_vx_cmd": cmd_stats["vx_cmd_mean"],
                    "p2_fault_applied": p2_fault_active_mean,
                    "selected_fault_joint_index_mean": selected_index_mean,
                    "done_count": step_done_count,
                }
            )
            obs = next_obs
            if step_number <= 3 or step_number % args.progress_every == 0 or step_number == args.num_steps:
                print(
                    f"[T17] step={step_number}/{args.num_steps} policy={policy} "
                    f"mean_vel_x={row['mean_vel_x']:.4f} mean_abs_vx_error={row['mean_abs_vx_error']:.4f} "
                    f"fault_active={p2_fault_active_mean:.4f}",
                    flush=True,
                )

        first_done_steps = [int(value) for value in first_done_step.detach().cpu().tolist()]
        survival_summary = build_random_onset_survival_summary(
            first_done_steps=first_done_steps,
            initial_onset_steps=initial_onset_steps,
            num_steps=args.num_steps,
        )
        done_count = sum(1 for value in first_done_steps if value >= 0)
        done_rate = done_count / float(tracking_num_envs)
        pre_rows = rows_by_fault_fraction(rows, post_fault=False)
        post_rows = rows_by_fault_fraction(rows, post_fault=True)
        partial_post_rows = [
            row
            for row in rows
            if (optional_float(row.get("P2/fault_applied")) or 0.0) > 0.0
        ]
        post_rows_for_metrics = post_rows if post_rows else partial_post_rows
        command_mode_stats = command_stats_from_rows(rows)
        command_mode_valid, command_mode_validation_error = validate_command_mode(args.velocity_mode, command_mode_stats)
        p2_fault_became_active = bool(fault_active_any_ever)
        post_fault_sample_count = int(fault_active_sample_count)
        post_fault_done_sample_rate = (
            fault_active_done_sample_count / float(post_fault_sample_count) if post_fault_sample_count else None
        )
        post_fault_alive_sample_fraction = (
            fault_active_alive_sample_count / float(post_fault_sample_count) if post_fault_sample_count else None
        )
        short_before_max_onset = int(args.num_steps) < int(protocol_cfg["fault_onset_step_max"])
        insufficient_post_fault_coverage = bool(
            p2_fault_became_active
            and (short_before_max_onset or post_fault_sample_count < tracking_num_envs)
        )
        smoke_partial_fault_coverage_ok = bool(insufficient_post_fault_coverage and short_before_max_onset)
        partial_fault_warning = (
            "Short smoke run had partial P2 activation; metrics are for pipeline validation only."
        )
        fallback_used = bool(max(fallback_values) >= 0.5) if fallback_values else bool(p2_wrapper.fallback_used)
        simulation_override_applied = bool(max(override_values) >= 0.5) if override_values else False
        pd_surrogate_used = p2_wrapper.mapping.semantics == "pd_position_hold_surrogate"
        per_joint_rows = per_joint_rows_from_samples(
            policy=policy,
            protocol=args.protocol,
            velocity_mode=args.velocity_mode,
            alpha=alpha,
            joint_names=p2_wrapper.mapping.supported_target_joints,
            output_dir=output_dir,
            samples=per_joint_samples,
        )

        rollout_csv = output_dir / "rollout_metrics.csv"
        velocity_csv = output_dir / "velocity_timeseries.csv"
        per_joint_csv = output_dir / "per_joint_metrics.csv"
        summary.update(
            {
                "device": str(policy_device),
                "actual_semantics": p2_wrapper.mapping.semantics,
                "supported_fault_joint_names": list(p2_wrapper.mapping.supported_target_joints),
                "supported_fault_joint_count": len(p2_wrapper.mapping.supported_target_joints),
                "selected_fault_joint_index_min": min(selected_joint_ids_seen) if selected_joint_ids_seen else None,
                "selected_fault_joint_index_max": max(selected_joint_ids_seen) if selected_joint_ids_seen else None,
                "selected_fault_joint_all_8_covered": selected_joint_ids_seen == set(range(EXPECTED_ACTION_DIM)),
                "mean_reward": reward_sum / reward_count if reward_count else None,
                "mean_vel_x_pre_fault": row_mean(pre_rows, "mean_vel_x"),
                "mean_vel_x_post_fault": row_mean(post_rows_for_metrics, "mean_vel_x"),
                "mean_abs_vx_error_pre_fault": row_mean(pre_rows, "mean_abs_vx_error"),
                "mean_abs_vx_error_post_fault": row_mean(post_rows_for_metrics, "mean_abs_vx_error"),
                "median_abs_vx_error_post_fault": percentile(post_fault_abs_errors, 0.5),
                "p90_abs_vx_error_post_fault": percentile(post_fault_abs_errors, 0.9),
                "survival_rate": 1.0 - done_rate,
                "done_rate": done_rate,
                "raw_any_done_rate": done_rate,
                "raw_never_done_survival_rate": 1.0 - done_rate,
                "post_fault_alive_sample_fraction": post_fault_alive_sample_fraction,
                "post_fault_done_sample_rate": post_fault_done_sample_rate,
                "post_fault_torso_failure_rate": None,
                "post_fault_timeout_success_rate": None,
                "post_fault_non_timeout_failure_rate": None,
                "survival_metric_audit_note": (
                    "survival_rate/raw_never_done_survival_rate is first-done-ever over the whole rollout; "
                    "post_fault_alive_sample_fraction is computed from fault-active env-step samples. "
                    "Post-fault timeout/collapse separation needs explicit termination-cause logging."
                ),
                "done_count": done_count,
                "timeout_rate": maybe_mean(timeout_values),
                "torso_height_failure_rate": maybe_mean(torso_height_failure_values),
                "mean_abs_yaw_error": row_mean(post_rows_for_metrics if post_rows_for_metrics else rows, "mean_abs_yaw_error"),
                **command_mode_stats,
                "command_mode_valid": command_mode_valid,
                "command_mode_validation_error": command_mode_validation_error,
                "fault_active_any_ever": p2_fault_became_active,
                "fault_active_sample_count": fault_active_sample_count,
                "fault_active_env_count": len(fault_active_env_ids),
                "post_fault_sample_count": post_fault_sample_count,
                "fault_active_fraction_last_step": fault_active_fraction_last_step,
                "insufficient_post_fault_coverage": insufficient_post_fault_coverage,
                "smoke_partial_fault_coverage_ok": smoke_partial_fault_coverage_ok,
                "smoke_partial_fault_coverage_warning": partial_fault_warning if smoke_partial_fault_coverage_ok else "",
                "fallback_used": fallback_used,
                "pd_surrogate_used": pd_surrogate_used,
                "simulation_override_applied": simulation_override_applied,
                "p2_fault_became_active": p2_fault_became_active,
                "no_nan_inf": bool(no_nan_inf),
                "base_action_mean_norm": maybe_mean(base_mean_norm_values),
                "residual_action_mean_norm": maybe_mean(residual_mean_norm_values),
                "residual_action_max_norm": max(residual_max_norm_values) if residual_max_norm_values else None,
                "final_action_mean_norm": maybe_mean(final_mean_norm_values),
                "velocity_metric_source": velocity_source,
                "yaw_metric_source": yaw_source,
                "rollout_metrics_csv": repo_relative(rollout_csv),
                "velocity_timeseries_csv": repo_relative(velocity_csv),
                "per_joint_metrics_csv": repo_relative(per_joint_csv),
                "output_dir": repo_relative(output_dir),
                "post_fault_row_definition": "P2/fault_applied >= 0.5, falling back to > 0.0 for short partial-activation smoke runs",
                **survival_summary,
            }
        )
        if not command_mode_valid:
            summary["metric_notes"].append(f"invalid command mode: {command_mode_validation_error}")
        if smoke_partial_fault_coverage_ok:
            summary["metric_notes"].append(partial_fault_warning)
        if not p2_fault_became_active:
            write_json(
                output_dir / "error_summary.json",
                {
                    "created_at": datetime.now().isoformat(timespec="seconds"),
                    "error_type": "T17ClosedLoopEvalError",
                    "error_message": "P2 fault never became active in any env/sample.",
                    "policy": policy,
                    "protocol": args.protocol,
                    "velocity_mode": args.velocity_mode,
                    "alpha": alpha,
                    "num_envs": args.num_envs,
                    "num_steps": args.num_steps,
                    "fault_onset_step_min": protocol_cfg["fault_onset_step_min"],
                    "fault_onset_step_max": protocol_cfg["fault_onset_step_max"],
                    "fault_active_sample_count": fault_active_sample_count,
                    "fault_active_env_count": len(fault_active_env_ids),
                    "fault_active_fraction_last_step": fault_active_fraction_last_step,
                    "control_frequency_hz": summary.get("control_frequency_hz"),
                    "control_dt_s": summary.get("control_dt_s"),
                    "physics_frequency_hz": summary.get("physics_frequency_hz"),
                    "sim_dt_s": summary.get("sim_dt_s"),
                    "decimation": summary.get("decimation"),
                    "output_dir": repo_relative(output_dir),
                },
            )
            raise T17ClosedLoopEvalError("P2 fault never became active in any env/sample.")
        if fallback_used:
            raise T17ClosedLoopEvalError("P2 fallback was used.")
        if pd_surrogate_used:
            raise T17ClosedLoopEvalError("PD surrogate was used.")
        if not simulation_override_applied:
            raise T17ClosedLoopEvalError("simulation override was never observed after onset.")
        if not no_nan_inf:
            raise T17ClosedLoopEvalError("NaN or Inf observed.")

        write_csv(rollout_csv, rows, fieldnames=ROLLOUT_FIELDS)
        write_csv(velocity_csv, velocity_rows, fieldnames=VELOCITY_FIELDS)
        write_csv(per_joint_csv, per_joint_rows, fieldnames=PER_JOINT_FIELDS)
        write_json(output_dir / "summary.json", summary)
        (output_dir / "README.md").write_text(
            "\n".join(
                [
                    "# T17 Closed-Loop Run Output",
                    "",
                    f"- policy: `{policy}`",
                    f"- protocol: `{args.protocol}`",
                    f"- velocity_mode: `{args.velocity_mode}`",
                    f"- alpha: `{alpha}`",
                    "- direct simulation-state override required",
                    "- fallback and PD surrogate disabled",
                    "- no teacher observation or explicit fault token enters deployment-facing student policy",
                    "- preliminary advisor-facing evidence only",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        print(f"[T17] summary: {repo_relative(output_dir / 'summary.json')}", flush=True)
        if not command_mode_valid:
            print(f"[T17 ERROR] invalid command mode: {command_mode_validation_error}", file=sys.stderr, flush=True)
            return 1
        return 0
    except Exception as exc:
        summary["error_type"] = type(exc).__name__
        summary["error_message"] = str(exc)
        summary["traceback"] = traceback.format_exc()
        write_json(
            output_dir / "error_summary.json",
            {
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback": summary["traceback"],
                "policy": summary.get("policy"),
                "protocol": summary.get("protocol"),
                "velocity_mode": summary.get("velocity_mode"),
                "alpha": summary.get("alpha"),
                "num_envs": summary.get("num_envs"),
                "num_steps": summary.get("num_steps"),
                "fault_onset_step_min": summary.get("fault_onset_step_min"),
                "fault_onset_step_max": summary.get("fault_onset_step_max"),
                "fault_active_sample_count": summary.get("fault_active_sample_count"),
                "fault_active_env_count": summary.get("fault_active_env_count"),
                "fault_active_fraction_last_step": summary.get("fault_active_fraction_last_step"),
                "control_frequency_hz": summary.get("control_frequency_hz"),
                "control_dt_s": summary.get("control_dt_s"),
                "physics_frequency_hz": summary.get("physics_frequency_hz"),
                "sim_dt_s": summary.get("sim_dt_s"),
                "decimation": summary.get("decimation"),
                "summary_json": repo_relative(output_dir / "summary.json"),
                "output_dir": repo_relative(output_dir),
            },
        )
        write_json(output_dir / "summary.json", summary)
        print(f"[T17 ERROR] {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        print(summary["traceback"], file=sys.stderr, flush=True)
        return 1
    finally:
        if vec_env is not None:
            vec_env.close()
        elif env is not None:
            env.close()


def require_run_outputs(spec: RunSpec) -> None:
    missing = [
        name
        for name in ("summary.json", "rollout_metrics.csv", "velocity_timeseries.csv", "per_joint_metrics.csv")
        if not (spec.output_dir / name).is_file()
    ]
    if missing:
        raise T17ClosedLoopEvalError(f"missing {', '.join(missing)} in {repo_relative(spec.output_dir)}")


def run_dry_run(args: argparse.Namespace, specs: list[RunSpec]) -> int:
    print("[T17 DRY RUN]")
    print("no_isaac_sim_launched: true")
    print("no_eval: true")
    print("no_training: true")
    print("no_checkpoint_modification: true")
    print("no_dataset_modification: true")
    print("no_task_config_modification: true")
    print("no_p2_wrapper_modification: true")
    print("not_paper_grade_final: true")
    print(f"output_dir: {repo_relative(args.output_dir)}")
    print(f"a2_single_step_checkpoint: {args.a2_single_step_checkpoint}")
    print(f"a2_history_checkpoint: {args.a2_history_checkpoint}")
    print(f"a5_checkpoint: {args.a5_checkpoint}")
    print(f"teacher_checkpoint: {args.teacher_checkpoint}")
    print(f"history_len: {args.history_len}")
    print(f"requested_control_frequency_hz: {args.control_frequency_hz}")
    print(f"requested_sim_dt: {args.sim_dt}")
    print(f"requested_decimation: {args.decimation}")
    print(f"required_control_frequency_hz: {args.require_control_frequency_hz}")
    print(f"t18r_pg500_timing: {args.t18r_pg500_timing}")
    print(f"require_t18r_pg500_timing: {args.require_t18r_pg500_timing}")
    print("fault_semantics: random_per_env one locked joint, direct simulation_joint_state_override_lock, no fallback")
    print("student_policy_inputs: 61-D student_obs only; no teacher_obs, q_lock, selected joint id/one-hot, P2-active flag, or health token")
    print("expected_output_files: results_summary.md, results_summary.json, per_run_metrics.csv, per_joint_metrics.csv, advisor_update_snippet.txt, README.md, command.txt")
    print(f"planned_run_count: {len(specs)}")
    for index, spec in enumerate(specs, start=1):
        rendered_command = render_command(spec.command)
        print(
            f"[{index:03d}] policy={spec.policy} protocol={spec.protocol} "
            f"velocity_mode={spec.velocity_mode} alpha={spec.alpha} output={repo_relative(spec.output_dir)}"
        )
        print(rendered_command)
    return 0


def run_execute(args: argparse.Namespace, specs: list[RunSpec]) -> int:
    output_root = resolve_repo_path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    write_root_readme(output_root)
    write_command_txt(output_root, " ".join(sys.argv))
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["TERM"] = env.get("TERM", "xterm")
    for index, spec in enumerate(specs, start=1):
        spec.output_dir.mkdir(parents=True, exist_ok=True)
        terminal_log = spec.output_dir / "terminal.log"
        print(
            f"[T17 {index}/{len(specs)}] policy={spec.policy} protocol={spec.protocol} "
            f"velocity_mode={spec.velocity_mode} alpha={spec.alpha}"
        )
        validate_command_tokens(spec.command)
        rendered_command = render_command(spec.command)
        print(rendered_command)
        with terminal_log.open("w", encoding="utf-8") as log_file:
            log_file.write(rendered_command + "\n\n")
            log_file.flush()
            validate_command_tokens(spec.command)
            render_command(spec.command)
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
            rows, per_joint_rows = aggregate_completed(output_root, specs)
            print(f"[T17] aggregate_completed_runs={len(rows)} per_joint_rows={len(per_joint_rows)}", flush=True)
            if args.stop_on_failure:
                raise T17ClosedLoopEvalError(message)
            print(f"[T17 WARNING] {message}", file=sys.stderr, flush=True)
            continue
        require_run_outputs(spec)
        rows, per_joint_rows = aggregate_completed(output_root, specs)
        print(f"[T17] aggregate_completed_runs={len(rows)} per_joint_rows={len(per_joint_rows)}", flush=True)
    aggregate_completed(output_root, specs)
    return 0


def main() -> int:
    try:
        args = parse_args()
        validate_args(args)
        if args.execute_eval and args.single_run:
            from isaaclab.app import AppLauncher

            args.headless = True
            app_launcher = AppLauncher(args)
            simulation_app = app_launcher.app
            try:
                return run_single_policy_rollout(args)
            finally:
                simulation_app.close()
        specs = make_specs(args)
        if args.dry_run:
            return run_dry_run(args, specs)
        return run_execute(args, specs)
    except T17ClosedLoopEvalError as exc:
        print(f"[T17 ERROR] {exc}", file=sys.stderr, flush=True)
        return 2
    except KeyboardInterrupt:
        print("[T17 ERROR] interrupted", file=sys.stderr, flush=True)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
