#!/usr/bin/env python3
"""T13C multi-joint P2 privileged teacher evaluation scaffold.

This evaluator is guarded: it does not train, modify checkpoints, edit task
configs, or change P2 semantics. Isaac is launched only through ``--execute_eval``.
The default mode evaluates the final T13-B v2b privileged teacher on late and
realistic random multi-joint P2 protocols with fixed ``vx_cmd = 1.0``.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import statistics
import subprocess
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from run_t09_p2_quick_demo_compare import (
    done_mask_tensor,
    optional_float,
    prepare_agent_cfg,
    repo_relative,
    resolve_forward_velocity_metrics,
    resolve_repo_path,
    scalar,
)
from run_t10_velocity_p2_random_eval_compare import (
    build_random_onset_survival_summary,
    current_onset_steps,
    first_matching_log_value,
    resolve_yaw_rate_metrics,
)
from t18r_control_timing import add_control_timing_args, apply_control_timing_to_env_cfg


REPO_ROOT = Path(__file__).resolve().parents[1]

TASK = "Isaac-Ant-Teacher-Velocity-MultiJointP2-Flat-v0"
POLICY_LABEL = "A1F_multijoint_P2_teacher_v2b"
FAULT_PROFILE = "P2_locked_joint"
TARGET_JOINT_MODE = "random_per_env"
TARGET_JOINT_PLACEHOLDER = "front_left_foot"
REQUESTED_SEMANTICS = "simulation_joint_state_override_lock"
DEFAULT_CHECKPOINT = (
    "logs/rsl_rl/teacher_p2_multijoint_velocity_curriculum_v2b__rlm1_stripped__p2_multi_joint_random/"
    "2026-06-15_18-16-03_a1f_multijoint_velocity_p2_v2b_curriculum_curriculum_s2_realistic_transition_random_p2__seed0/"
    "model_9997.pt"
)
DEFAULT_OUTPUT_ROOT = "papers/conference/results/t13c_multijoint_teacher_eval"
DEFAULT_NUM_ENVS = 512
DEFAULT_NUM_STEPS = 1000
DEFAULT_SEED = 0
DEFAULT_DEVICE = "cuda"
EXPECTED_ACTION_DIM = 8
EXPECTED_TEACHER_OBS_DIM = 77
FIXED_VX = 1.0
COMMAND_RANDOM_RANGE = (0.2, 1.5)
MALFORMED_EXEC_SINGLE_RUN = "--execute_eval--single_run"

PROTOCOLS = {
    "late_random": {
        "dir_name": "late_random",
        "fault_onset_step_min": 250,
        "fault_onset_step_max": 700,
        "description": "late random P2 onset, matching v2b S1",
    },
    "realistic_random": {
        "dir_name": "realistic_random",
        "fault_onset_step_min": 120,
        "fault_onset_step_max": 700,
        "description": "realistic healthy-to-fault transition, matching v2b S2",
    },
    "stress_random": {
        "dir_name": "stress_random",
        "fault_onset_step_min": 30,
        "fault_onset_step_max": 700,
        "description": "early-onset robustness stress protocol, opt-in only",
    },
}

VELOCITY_MODES = {
    "fixed_vx_1p0": {
        "dir_name": "fixed_vx_1p0",
        "target_vx": FIXED_VX,
        "description": "fixed vx_cmd = 1.0 for T12 comparability",
    },
    "command_random": {
        "dir_name": "command_random",
        "target_vx": None,
        "description": "task command range vx_cmd in [0.2, 1.5]",
    },
}

SUMMARY_FIELDS = [
    "protocol",
    "velocity_mode",
    "checkpoint",
    "task",
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
    "mean_abs_yaw_error",
    "timeout_rate",
    "torso_height_failure_rate",
    "p2_fault_became_active",
    "fallback_used",
    "simulation_override_applied",
    "no_nan_inf",
    "vx_cmd_mean",
    "vx_cmd_min",
    "vx_cmd_max",
    "command_mode_valid",
    "command_mode_validation_error",
    "selected_fault_joint_index_mean",
    "selected_fault_joint_index_min",
    "selected_fault_joint_index_max",
    "supported_fault_joint_count",
    "multi_joint_randomization",
    "onset_step_mean",
    "onset_step_min",
    "onset_step_max",
    "teacher_policy_obs_dim",
    "p2_fault_joint_one_hot_dim",
    "p2_fault_q_lock_vector_dim",
    "rollout_metrics_csv",
    "velocity_timeseries_csv",
    "output_dir",
]

ROLLOUT_FIELDS = [
    "step",
    "protocol",
    "velocity_mode",
    "policy_label",
    "task",
    "seed",
    "fault_profile",
    "target_joint_mode",
    "fault_onset_mode",
    "fault_onset_step_min",
    "fault_onset_step_max",
    "vx_cmd_mode",
    "vx_cmd",
    "vx_cmd_mean",
    "vx_cmd_min",
    "vx_cmd_max",
    "reward_mean",
    "mean_vel_x",
    "mean_abs_vx_error",
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
    "velocity_metric_source",
    "yaw_metric_source",
    "no_nan_inf",
]

VELOCITY_FIELDS = [
    "step",
    "policy_label",
    "protocol",
    "velocity_mode",
    "mean_vel_x",
    "mean_abs_vx_error",
    "mean_vx_cmd",
    "p2_fault_applied",
    "selected_fault_joint_index_mean",
    "selected_fault_joint_index_min",
    "selected_fault_joint_index_max",
]

PER_JOINT_FIELDS = [
    "protocol",
    "velocity_mode",
    "joint_id",
    "joint_name",
    "count",
    "post_fault_mean_vel_x",
    "post_fault_mean_abs_vx_error",
    "post_fault_mean_abs_yaw_error",
    "timeout_rate",
    "torso_height_failure_rate",
    "output_dir",
]


class T13CEvalError(ValueError):
    """Raised for invalid T13C evaluation state."""


@dataclass(frozen=True)
class RunSpec:
    protocol: str
    velocity_mode: str
    output_dir: Path
    checkpoint: str
    command: list[str]


def build_parser(*, add_app_launcher_args: bool = False) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="T13C multi-joint privileged teacher evaluation scaffold.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry_run", action="store_true", help="Print and save planned commands without Isaac.")
    mode.add_argument("--execute_eval", action="store_true", help="Run planned eval commands.")
    parser.add_argument("--single_run", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--task", default=TASK)
    parser.add_argument(
        "--protocol",
        default="default",
        choices=("default", "late_random", "realistic_random", "stress_random", "all"),
        help="default runs late_random and realistic_random; stress_random is opt-in.",
    )
    parser.add_argument(
        "--velocity_mode",
        default="fixed_vx_1p0",
        choices=("fixed_vx_1p0", "command_random", "both"),
    )
    parser.add_argument("--include_stress", action="store_true", help="Include stress_random in the default protocol set.")
    parser.add_argument("--output_root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output_dir", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--num_envs", type=int, default=DEFAULT_NUM_ENVS)
    parser.add_argument("--num_steps", type=int, default=DEFAULT_NUM_STEPS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
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
        raise T13CEvalError("--num_envs must be > 0.")
    if args.num_steps <= 0:
        raise T13CEvalError("--num_steps must be > 0.")
    if args.seed < 0:
        raise T13CEvalError("--seed must be non-negative.")
    if args.task != TASK:
        raise T13CEvalError(f"T13C is guarded for {TASK}; got {args.task!r}.")
    if args.single_run and args.velocity_mode == "both":
        raise T13CEvalError("--single_run requires one concrete velocity_mode.")
    if args.single_run and args.protocol in {"default", "all"}:
        raise T13CEvalError("--single_run requires one concrete protocol.")
    if args.execute_eval and resolve_repo_path(args.checkpoint).is_file() is False:
        raise T13CEvalError(f"checkpoint does not exist: {args.checkpoint}")


def selected_protocols(args: argparse.Namespace) -> list[str]:
    if args.protocol == "default":
        protocols = ["late_random", "realistic_random"]
        if args.include_stress:
            protocols.append("stress_random")
        return protocols
    if args.protocol == "all":
        return ["late_random", "realistic_random", "stress_random"]
    return [args.protocol]


def selected_velocity_modes(args: argparse.Namespace) -> list[str]:
    if args.velocity_mode == "both":
        return ["fixed_vx_1p0", "command_random"]
    return [args.velocity_mode]


def guard_single_run_command_tokens(command_tokens: list[str]) -> None:
    if not isinstance(command_tokens, list) or not all(isinstance(token, str) for token in command_tokens):
        raise RuntimeError("T13C command must be represented as a list[str] token sequence.")
    if "--execute_eval" not in command_tokens:
        raise RuntimeError("T13C single-run command is missing the --execute_eval token.")
    if "--single_run" not in command_tokens:
        raise RuntimeError("T13C single-run command is missing the --single_run token.")
    if MALFORMED_EXEC_SINGLE_RUN in command_tokens:
        raise RuntimeError(f"T13C command contains malformed fused token: {MALFORMED_EXEC_SINGLE_RUN}")
    for token in command_tokens:
        if MALFORMED_EXEC_SINGLE_RUN in token:
            raise RuntimeError(f"T13C command token contains malformed fused flags: {token}")


def render_command(command_tokens: list[str]) -> str:
    guard_single_run_command_tokens(command_tokens)
    rendered = shlex.join(command_tokens)
    if MALFORMED_EXEC_SINGLE_RUN in rendered:
        raise RuntimeError(f"T13C rendered command contains malformed fused flags: {MALFORMED_EXEC_SINGLE_RUN}")
    return rendered


def display_command(command_tokens: list[str]) -> str:
    rendered = render_command(command_tokens)
    display = "PYTHONUNBUFFERED=1 TERM=xterm " + rendered
    if MALFORMED_EXEC_SINGLE_RUN in display:
        raise RuntimeError(f"T13C displayed command contains malformed fused flags: {MALFORMED_EXEC_SINGLE_RUN}")
    return display


def flag_value(flag: str, value: object) -> list[str]:
    return [flag, str(value)]


def append_flag_value(command: list[str], flag: str, value: object) -> None:
    command.extend(flag_value(flag, value))


def build_single_run_command(
    *,
    python: str,
    args: argparse.Namespace,
    protocol: str,
    velocity_mode: str,
    output_root: Path,
    output_dir: Path,
) -> list[str]:
    command = [
        python,
        "evaluators/run_t13c_multijoint_teacher_eval.py",
    ]
    command.append("--execute_eval")
    command.append("--single_run")
    append_flag_value(command, "--checkpoint", args.checkpoint)
    append_flag_value(command, "--task", args.task)
    append_flag_value(command, "--protocol", protocol)
    append_flag_value(command, "--velocity_mode", velocity_mode)
    append_flag_value(command, "--output_root", repo_relative(output_root))
    append_flag_value(command, "--output_dir", repo_relative(output_dir))
    append_flag_value(command, "--num_envs", args.num_envs)
    append_flag_value(command, "--num_steps", args.num_steps)
    append_flag_value(command, "--seed", args.seed)
    append_optional_timing_flags(command, args)
    command.append("--headless")
    append_flag_value(command, "--device", args.device)
    guard_single_run_command_tokens(command)
    return command


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


def make_specs(args: argparse.Namespace) -> list[RunSpec]:
    output_root = resolve_repo_path(args.output_root)
    specs: list[RunSpec] = []
    python = sys.executable
    for protocol in selected_protocols(args):
        protocol_cfg = PROTOCOLS[protocol]
        for velocity_mode in selected_velocity_modes(args):
            mode_cfg = VELOCITY_MODES[velocity_mode]
            output_dir = output_root / protocol_cfg["dir_name"] / mode_cfg["dir_name"]
            command = build_single_run_command(
                python=python,
                args=args,
                protocol=protocol,
                velocity_mode=velocity_mode,
                output_root=output_root,
                output_dir=output_dir,
            )
            specs.append(RunSpec(protocol, velocity_mode, output_dir, args.checkpoint, command))
    return specs


def write_json(path: Path, values: dict[str, Any]) -> None:
    path.write_text(json.dumps(values, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], *, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_root_readme(output_root: Path) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    readme_path = output_root / "README.md"
    readme_path.write_text(
        "\n".join(
            [
                "# T13C Multi-Joint A1-F Teacher Evaluation",
                "",
                "Guarded evaluator scaffold for the T13-B v2b privileged multi-joint P2 teacher.",
                "",
                "This is evaluation-only infrastructure. It does not train, modify checkpoints, modify task configs,",
                "or change P2 wrapper semantics.",
                "",
                "## Policy",
                "",
                f"- task: `{TASK}`",
                f"- checkpoint: `{DEFAULT_CHECKPOINT}`",
                "- teacher observation: `77 = 61 base + 8 selected-joint one-hot + 8 q_lock vector`",
                "- selected locked joint: one random actuated joint per env/episode",
                "- semantics: direct `simulation_joint_state_override_lock`",
                "- fallback/surrogate: disabled",
                "- A7 terminology in this pipeline: A0-anchored residual variant / healthy-base residual variant",
                "",
                "## Default Protocols",
                "",
                "- `late_random`: onset U(250,700)",
                "- `realistic_random`: onset U(120,700)",
                "- `stress_random`: onset U(30,700), opt-in only",
                "",
                "## Velocity Modes",
                "",
                "- `fixed_vx_1p0`: forces command tensor to vx = 1.0, vy = 0, yaw = 0 before policy inference and after each env step",
                "- `command_random`: leaves the command-conditioned task range active, vx in [0.2, 1.5]",
                "",
                "## Optional T18-R Timing",
                "",
                "- Pass `--control_frequency_hz 50 --sim_dt 0.01 --decimation 2 --require_control_frequency_hz 50`",
                "  to evaluate the privileged reference at 50 Hz control.",
                "- For paper-grade PG500 timing, pass `--t18r_pg500_timing --require_t18r_pg500_timing`",
                "  to enforce 500 Hz physics and 50 Hz control.",
                "- Default behavior remains the task timing, currently `sim.dt=1/120` and `decimation=2` in the Ant config.",
                "",
                "## Command-Mode Note",
                "",
                "T13C fixed-vx outputs generated before the command-enforcement patch were command-unforced and",
                "must not be used as fixed-vx evidence. They may be kept only as preliminary command-conditioned",
                "or random-command evidence.",
                "",
                "## Outputs",
                "",
                "- per-run: `summary.json`, `rollout_metrics.csv`, `velocity_timeseries.csv`, `per_joint_metrics.csv`, `command.txt`",
                "- aggregate: `tables/metrics_summary.csv`, `tables/metrics_summary.md`, `tables/per_joint_metrics.csv`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return readme_path


def write_command_markdown(output_root: Path, specs: list[RunSpec]) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    command_path = output_root / "t13c_run_commands.md"
    lines = [
        "# T13C Multi-Joint Teacher Eval Commands",
        "",
        "Generated by `evaluators/run_t13c_multijoint_teacher_eval.py --dry_run`.",
        "",
        "These commands launch Isaac only when run manually.",
        "",
    ]
    for index, spec in enumerate(specs, start=1):
        guard_single_run_command_tokens(spec.command)
        lines.extend(
            [
                f"## {index}. {spec.protocol} / {spec.velocity_mode}",
                "",
                "```bash",
                display_command(spec.command),
                "```",
                "",
            ]
        )
    command_path.write_text("\n".join(lines), encoding="utf-8")
    return command_path


def maybe_mean(values: list[float]) -> float | None:
    return float(statistics.mean(values)) if values else None


def mean_or_blank(values: list[float]) -> float | str:
    return float(statistics.mean(values)) if values else ""


def max_or_blank(values: list[float]) -> float | str:
    return max(values) if values else ""


def min_or_blank(values: list[float]) -> float | str:
    return min(values) if values else ""


def parse_json_float_list(value: Any) -> list[float] | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, list):
        return None
    values: list[float] = []
    for item in parsed:
        parsed_float = optional_float(item)
        if parsed_float is None:
            return None
        values.append(parsed_float)
    return values


def tensor_to_float_list(value: Any) -> list[float] | None:
    try:
        import torch

        tensor = torch.as_tensor(value).detach().float()
        if tensor.ndim == 0:
            tensor = tensor.reshape(1)
        if not torch.isfinite(tensor).all():
            return None
        return [float(item) for item in tensor.reshape(-1).cpu().tolist()]
    except Exception:
        return None


def tensor_to_int_list(value: Any) -> list[int] | None:
    try:
        import torch

        tensor = torch.as_tensor(value).detach().long()
        if tensor.ndim == 0:
            tensor = tensor.reshape(1)
        return [int(item) for item in tensor.reshape(-1).cpu().tolist()]
    except Exception:
        return None


def tensors_are_finite(value: Any) -> bool:
    try:
        import torch

        if isinstance(value, dict):
            return all(tensors_are_finite(item) for item in value.values())
        if isinstance(value, (tuple, list)):
            return all(tensors_are_finite(item) for item in value)
        tensor = torch.as_tensor(value).detach().float()
        return bool(torch.isfinite(tensor).all().item())
    except Exception:
        return True


def observation_dim(obs: Any) -> int | None:
    try:
        import torch

        candidate = obs
        if isinstance(obs, dict):
            candidate = obs.get("policy")
            if candidate is None and obs:
                candidate = next(iter(obs.values()))
        tensor = torch.as_tensor(candidate)
        if tensor.ndim == 0:
            return 1
        return int(tensor.shape[-1])
    except Exception:
        return None


def command_vx_tensor(env: Any):
    try:
        import torch

        command_manager = getattr(env.unwrapped, "command_manager", None)
        terms = getattr(command_manager, "_terms", None)
        if not isinstance(terms, dict):
            return None
        for term in terms.values():
            command = getattr(term, "command", None)
            if command is None:
                command = getattr(term, "_command", None)
            if command is None:
                continue
            tensor = torch.as_tensor(command).detach().float()
            if tensor.ndim == 2 and tensor.shape[1] >= 1 and torch.isfinite(tensor).all():
                return tensor[:, 0]
    except Exception:
        return None
    return None


def force_fixed_velocity_command(env: Any, target_vx: float) -> dict[str, Any]:
    """Force the command manager tensor to vx=target, vy=0, yaw=0 for all envs."""
    command_manager = getattr(env.unwrapped, "command_manager", None)
    if command_manager is None:
        return {
            "target_vx_available": False,
            "target_vx": None,
            "target_vx_source": None,
            "target_vx_requested": float(target_vx),
            "target_vx_unavailable_reason": "env.unwrapped.command_manager is unavailable",
        }
    terms = getattr(command_manager, "_terms", None)
    if not isinstance(terms, dict):
        return {
            "target_vx_available": False,
            "target_vx": None,
            "target_vx_source": None,
            "target_vx_requested": float(target_vx),
            "target_vx_unavailable_reason": "command_manager terms are unavailable",
        }

    for term_name, term in terms.items():
        for attr_name in ("command", "_command"):
            command = getattr(term, attr_name, None)
            if command is None:
                continue
            try:
                if command.ndim != 2 or command.shape[1] < 1:
                    continue
                command[:, 0] = float(target_vx)
                if command.shape[1] > 1:
                    command[:, 1:] = 0.0
                return {
                    "target_vx_available": True,
                    "target_vx": float(target_vx),
                    "target_vx_source": f"command_manager.{term_name}.{attr_name}[:, 0:3]",
                    "target_vx_requested": float(target_vx),
                    "target_vy": 0.0,
                    "target_yaw": 0.0,
                    "command_forced_each_step": True,
                }
            except Exception:
                continue
    return {
        "target_vx_available": False,
        "target_vx": None,
        "target_vx_source": None,
        "target_vx_requested": float(target_vx),
        "target_vx_unavailable_reason": "no writable velocity command tensor found",
    }


def command_vx_stats(env: Any) -> dict[str, Any]:
    tensor = command_vx_tensor(env)
    if tensor is None:
        return {"vx_cmd_mean": None, "vx_cmd_min": None, "vx_cmd_max": None, "per_env_vx_cmd": None}
    return {
        "vx_cmd_mean": float(tensor.mean().cpu().item()),
        "vx_cmd_min": float(tensor.min().cpu().item()),
        "vx_cmd_max": float(tensor.max().cpu().item()),
        "per_env_vx_cmd": [float(item) for item in tensor.cpu().tolist()],
    }


def command_stats_from_rows(rows: list[dict[str, Any]]) -> dict[str, float | str]:
    mean_values = [value for row in rows if (value := optional_float(row.get("vx_cmd_mean"))) is not None]
    min_values = [value for row in rows if (value := optional_float(row.get("vx_cmd_min"))) is not None]
    max_values = [value for row in rows if (value := optional_float(row.get("vx_cmd_max"))) is not None]
    return {
        "vx_cmd_mean": mean_or_blank(mean_values),
        "vx_cmd_min": min_or_blank(min_values),
        "vx_cmd_max": max_or_blank(max_values),
    }


def validate_command_mode(velocity_mode: str, stats: dict[str, Any]) -> tuple[bool, str]:
    vx_mean = optional_float(stats.get("vx_cmd_mean"))
    vx_min = optional_float(stats.get("vx_cmd_min"))
    vx_max = optional_float(stats.get("vx_cmd_max"))
    if vx_mean is None or vx_min is None or vx_max is None:
        return False, "velocity command statistics are unavailable"

    if velocity_mode == "fixed_vx_1p0":
        tolerance = 1.0e-4
        for name, value in (("vx_cmd_mean", vx_mean), ("vx_cmd_min", vx_min), ("vx_cmd_max", vx_max)):
            if abs(value - FIXED_VX) > tolerance:
                return False, f"{name}={value:.9g} is not fixed at {FIXED_VX:.1f}"
        return True, ""

    if velocity_mode == "command_random":
        lower, upper = COMMAND_RANDOM_RANGE
        tolerance = 1.0e-4
        if vx_min < lower - tolerance or vx_max > upper + tolerance:
            return False, (
                f"command_random vx range [{vx_min:.9g}, {vx_max:.9g}] is outside "
                f"configured range [{lower:.1f}, {upper:.1f}]"
            )
        if (vx_max - vx_min) < 0.05:
            return False, (
                f"command_random vx range [{vx_min:.9g}, {vx_max:.9g}] does not show "
                "meaningful command variation"
            )
        return True, ""

    return False, f"unsupported velocity_mode={velocity_mode!r}"


def per_env_yaw_rate(env: Any, *, action_term_name: str | None) -> list[float] | None:
    try:
        import torch

        action_manager = getattr(env.unwrapped, "action_manager", None)
        terms = getattr(action_manager, "_terms", None)
        term = terms.get(action_term_name) if isinstance(terms, dict) and action_term_name else None
        if term is None and isinstance(terms, dict) and terms:
            term = next(iter(terms.values()))
        asset = getattr(term, "_asset", None)
        data = getattr(asset, "data", None)
        for attr_name in (
            "root_ang_vel_w",
            "root_link_ang_vel_w",
            "root_com_ang_vel_w",
            "root_ang_vel_b",
            "root_link_ang_vel_b",
            "root_com_ang_vel_b",
        ):
            value = getattr(data, attr_name, None)
            if value is None:
                continue
            tensor = torch.as_tensor(value).detach().float()
            if tensor.ndim >= 2 and tensor.shape[-1] >= 3 and torch.isfinite(tensor).all():
                return [float(item) for item in tensor[:, -1].cpu().tolist()]
    except Exception:
        return None
    return None


def mean_abs_vx_error(
    *,
    mean_vel_x: float | None,
    per_env_vel_x: list[float] | None,
    per_env_vx_cmd: list[float] | None,
    fixed_target_vx: float | None,
) -> float | None:
    if per_env_vel_x is not None and per_env_vx_cmd is not None and len(per_env_vel_x) == len(per_env_vx_cmd):
        return maybe_mean([abs(vel - cmd) for vel, cmd in zip(per_env_vel_x, per_env_vx_cmd)])
    if fixed_target_vx is not None and mean_vel_x is not None:
        return abs(float(mean_vel_x) - fixed_target_vx)
    return None


def collect_dim(value: Any) -> int | None:
    try:
        tensor = getattr(value, "shape", None)
        if tensor is not None and len(value.shape) >= 2:
            return int(value.shape[-1])
    except Exception:
        return None
    return None


def row_mean(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [value for row in rows if (value := optional_float(row.get(key))) is not None]
    return maybe_mean(values)


def rows_by_fault_fraction(rows: list[dict[str, Any]], *, post_fault: bool) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for row in rows:
        fault_fraction = optional_float(row.get("P2/fault_applied"))
        if fault_fraction is None:
            continue
        if post_fault and fault_fraction >= 0.5:
            selected.append(row)
        if not post_fault and fault_fraction < 0.5:
            selected.append(row)
    return selected


def format_md(value: Any) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        return f"{value:.9g}"
    return str(value)


def summary_metric(summary: dict[str, Any], key: str) -> Any:
    value = summary.get(key, "")
    return "" if value is None else value


def row_from_summary(summary_path: Path) -> dict[str, Any]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return {field: summary_metric(summary, field) for field in SUMMARY_FIELDS}


def aggregate_completed(output_root: Path, specs: list[RunSpec]) -> tuple[Path, Path, Path, int]:
    tables_dir = output_root / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    per_joint_rows: list[dict[str, Any]] = []
    for spec in specs:
        summary_path = spec.output_dir / "summary.json"
        if summary_path.is_file():
            rows.append(row_from_summary(summary_path))
        per_joint_path = spec.output_dir / "per_joint_metrics.csv"
        if per_joint_path.is_file():
            with per_joint_path.open(newline="", encoding="utf-8") as stream:
                per_joint_rows.extend(csv.DictReader(stream))

    csv_path = tables_dir / "metrics_summary.csv"
    md_path = tables_dir / "metrics_summary.md"
    per_joint_csv_path = tables_dir / "per_joint_metrics.csv"
    write_csv(csv_path, rows, fieldnames=SUMMARY_FIELDS)
    write_csv(per_joint_csv_path, per_joint_rows, fieldnames=PER_JOINT_FIELDS)

    lines = [
        "# T13C Multi-Joint Teacher Metrics Summary",
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
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path, md_path, per_joint_csv_path, len(rows)


def per_joint_rows_from_samples(
    *,
    protocol: str,
    velocity_mode: str,
    joint_names: tuple[str, ...],
    output_dir: Path,
    samples: dict[int, dict[str, list[float]]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for joint_id, joint_name in enumerate(joint_names):
        values = samples.get(joint_id, {"vel_x": [], "abs_vx_error": [], "abs_yaw_error": []})
        rows.append(
            {
                "protocol": protocol,
                "velocity_mode": velocity_mode,
                "joint_id": joint_id,
                "joint_name": joint_name,
                "count": len(values.get("vel_x", [])),
                "post_fault_mean_vel_x": mean_or_blank(values.get("vel_x", [])),
                "post_fault_mean_abs_vx_error": mean_or_blank(values.get("abs_vx_error", [])),
                "post_fault_mean_abs_yaw_error": mean_or_blank(values.get("abs_yaw_error", [])),
                "timeout_rate": "",
                "torso_height_failure_rate": "",
                "output_dir": repo_relative(output_dir),
            }
        )
    return rows


def evaluate_single_run(args: argparse.Namespace) -> int:
    import gymnasium as gym
    import torch
    from rsl_rl.runners import DistillationRunner, OnPolicyRunner

    import isaaclab_tasks  # noqa: F401
    from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    trainers_dir = REPO_ROOT / "trainers"
    if str(trainers_dir) not in sys.path:
        sys.path.insert(0, str(trainers_dir))
    from p2_joint_lock_training_wrapper import P2JointLockActionMaskWrapper

    output_dir = resolve_repo_path(args.output_dir) if args.output_dir else (
        resolve_repo_path(args.output_root) / PROTOCOLS[args.protocol]["dir_name"] / VELOCITY_MODES[args.velocity_mode]["dir_name"]
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")

    protocol_cfg = PROTOCOLS[args.protocol]
    mode_cfg = VELOCITY_MODES[args.velocity_mode]
    checkpoint_path = resolve_repo_path(args.checkpoint)
    rows: list[dict[str, Any]] = []
    velocity_rows: list[dict[str, Any]] = []
    per_joint_samples: dict[int, dict[str, list[float]]] = {}
    env = None
    vec_env = None
    p2_wrapper = None
    summary: dict[str, Any] = {
        "eval_scope": "t13c_multijoint_privileged_teacher_eval",
        "not_paper_grade_final_evaluation": True,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "policy_label": POLICY_LABEL,
        "task": args.task,
        "checkpoint": repo_relative(checkpoint_path),
        "output_dir": repo_relative(output_dir),
        "protocol": args.protocol,
        "velocity_mode": args.velocity_mode,
        "num_envs": args.num_envs,
        "num_steps": args.num_steps,
        "seed": args.seed,
        "fault_profile": FAULT_PROFILE,
        "target_joint_mode": TARGET_JOINT_MODE,
        "requested_semantics": REQUESTED_SEMANTICS,
        "fallback_allowed": False,
        "pd_surrogate_allowed": False,
        "fault_onset_mode": "random_uniform",
        "fault_onset_step_min": protocol_cfg["fault_onset_step_min"],
        "fault_onset_step_max": protocol_cfg["fault_onset_step_max"],
        "vx_cmd_mode": args.velocity_mode,
        "vx_cmd": mode_cfg["target_vx"],
        "command_random_vx_range": list(COMMAND_RANDOM_RANGE),
        "expected_teacher_policy_obs_dim": EXPECTED_TEACHER_OBS_DIM,
        "expected_student_safe_obs_dim": 61,
        "student_fault_vector_excluded": True,
        "student_q_lock_vector_excluded": True,
        "health_token_enabled": False,
        "metric_notes": [],
    }

    try:
        env_cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
        agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
        env_cfg.scene.num_envs = args.num_envs
        env_cfg.seed = args.seed
        if getattr(args, "device", None):
            env_cfg.sim.device = args.device
        env_cfg.log_dir = str(output_dir / "isaac")
        control_timing = apply_control_timing_to_env_cfg(env_cfg, args)
        summary.update(control_timing)
        print(
            "[T13C] control timing "
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

        env = gym.make(args.task, cfg=env_cfg)
        if isinstance(env.unwrapped, DirectMARLEnv):
            env = multi_agent_to_single_agent(env)
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
            raise T13CEvalError(f"actual P2 semantics must be {REQUESTED_SEMANTICS}.")
        if p2_wrapper.mapping.target_joint_mode != TARGET_JOINT_MODE:
            raise T13CEvalError(f"target_joint_mode must be {TARGET_JOINT_MODE}.")
        if p2_wrapper.mapping.allow_fallback:
            raise T13CEvalError("multi-joint teacher eval must not allow fallback.")

        if args.velocity_mode == "fixed_vx_1p0":
            target_vx_info = force_fixed_velocity_command(env, FIXED_VX)
        else:
            target_vx_info = {
                "target_vx_available": False,
                "target_vx": None,
                "target_vx_source": None,
                "target_vx_requested": None,
                "command_forced_each_step": False,
                "target_vx_mode_note": "command_random leaves the command-conditioned task range active",
            }
        summary.update(target_vx_info)
        if args.velocity_mode == "fixed_vx_1p0" and not target_vx_info.get("target_vx_available"):
            raise T13CEvalError("fixed_vx_1p0 requires writable command tensor for vx_cmd = 1.0.")

        agent_cfg, agent_cfg_dict = prepare_agent_cfg(agent_cfg)
        vec_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

        if agent_cfg.class_name == "OnPolicyRunner":
            runner = OnPolicyRunner(vec_env, agent_cfg_dict, log_dir=None, device=agent_cfg.device)
        elif agent_cfg.class_name == "DistillationRunner":
            runner = DistillationRunner(vec_env, agent_cfg_dict, log_dir=None, device=agent_cfg.device)
        else:
            raise T13CEvalError(f"unsupported runner class {agent_cfg.class_name!r}.")
        runner.load(str(checkpoint_path))
        policy = runner.get_inference_policy(device=vec_env.unwrapped.device)

        if args.velocity_mode == "fixed_vx_1p0":
            force_fixed_velocity_command(env, FIXED_VX)
        obs = vec_env.get_observations()
        if args.velocity_mode == "fixed_vx_1p0":
            force_fixed_velocity_command(env, FIXED_VX)
            obs = vec_env.get_observations()
        teacher_policy_obs_dim = observation_dim(obs)
        p2_wrapper._ensure_lock_buffers()
        p2_fault_joint_one_hot_dim = collect_dim(p2_wrapper.p2_fault_joint_one_hot)
        p2_fault_q_lock_vector_dim = collect_dim(p2_wrapper.p2_fault_q_lock_vector)
        if teacher_policy_obs_dim != EXPECTED_TEACHER_OBS_DIM:
            summary["metric_notes"].append(f"teacher policy obs dim expected 77, observed {teacher_policy_obs_dim}")

        tracking_num_envs = int(getattr(vec_env, "num_envs", args.num_envs))
        tracking_device = vec_env.unwrapped.device
        initial_onset_steps = current_onset_steps(p2_wrapper, num_envs=tracking_num_envs)
        first_done_step = torch.full((tracking_num_envs,), -1, dtype=torch.long, device=tracking_device)
        p2_action_term_name = p2_wrapper.mapping.action_term_name
        total_done_count = 0
        reward_sum = 0.0
        reward_count = 0
        no_nan_inf = tensors_are_finite(obs)
        timeout_values: list[float] = []
        torso_height_failure_values: list[float] = []
        last_log_values: dict[str, Any] = {}
        velocity_metric_source = None
        yaw_metric_source = None

        for step_index in range(args.num_steps):
            step_number = step_index + 1
            with torch.inference_mode():
                if args.velocity_mode == "fixed_vx_1p0":
                    force_fixed_velocity_command(env, FIXED_VX)
                    obs = vec_env.get_observations()
                actions = policy(obs)
                no_nan_inf = no_nan_inf and tensors_are_finite(actions)
                obs, rewards, dones, extras = vec_env.step(actions)
                if args.velocity_mode == "fixed_vx_1p0":
                    force_fixed_velocity_command(env, FIXED_VX)
                    obs = vec_env.get_observations()
                no_nan_inf = no_nan_inf and tensors_are_finite(obs) and tensors_are_finite(rewards)
                if hasattr(policy, "reset"):
                    policy.reset(dones)

            reward_tensor = torch.as_tensor(rewards).detach().float()
            reward_sum += float(reward_tensor.sum().cpu().item())
            reward_count += int(reward_tensor.numel())

            done_mask = done_mask_tensor(dones, num_envs=tracking_num_envs, device=tracking_device)
            step_done_count = int(done_mask.sum().detach().cpu().item())
            total_done_count += step_done_count
            new_done = torch.logical_and(done_mask, first_done_step < 0)
            first_done_step[new_done] = step_number

            log_values = extras.get("log", {}) if isinstance(extras, dict) else {}
            if isinstance(log_values, dict):
                last_log_values = log_values
                timeout_value = first_matching_log_value(log_values, ("time", "out"))
                if timeout_value is not None:
                    timeout_values.append(timeout_value)
                torso_value = first_matching_log_value(log_values, ("torso", "height"))
                if torso_value is not None:
                    torso_height_failure_values.append(torso_value)

            velocity_metrics = resolve_forward_velocity_metrics(
                env,
                extras,
                action_term_name=p2_action_term_name,
                representative_env_index=0,
            )
            if velocity_metrics["velocity_available"] and velocity_metric_source is None:
                velocity_metric_source = velocity_metrics["velocity_metric_source"]
            yaw_metrics = resolve_yaw_rate_metrics(env, extras, action_term_name=p2_action_term_name)
            if yaw_metrics["yaw_metric_source"] and yaw_metric_source is None:
                yaw_metric_source = yaw_metrics["yaw_metric_source"]

            cmd_stats = command_vx_stats(env)
            per_env_vel_x = parse_json_float_list(velocity_metrics.get("per_env_vel_x"))
            mean_abs_error = mean_abs_vx_error(
                mean_vel_x=velocity_metrics["mean_vel_x"],
                per_env_vel_x=per_env_vel_x,
                per_env_vx_cmd=cmd_stats["per_env_vx_cmd"],
                fixed_target_vx=mode_cfg["target_vx"],
            )
            selected_index_mean = optional_float(log_values.get("P2/selected_fault_joint_index_mean"))
            selected_index_min = optional_float(log_values.get("P2/selected_fault_joint_index_min"))
            selected_index_max = optional_float(log_values.get("P2/selected_fault_joint_index_max"))
            p2_fault_applied = optional_float(log_values.get("P2/fault_applied"))
            row = {
                "step": step_number,
                "protocol": args.protocol,
                "velocity_mode": args.velocity_mode,
                "policy_label": POLICY_LABEL,
                "task": args.task,
                "seed": args.seed,
                "fault_profile": FAULT_PROFILE,
                "target_joint_mode": TARGET_JOINT_MODE,
                "fault_onset_mode": "random_uniform",
                "fault_onset_step_min": protocol_cfg["fault_onset_step_min"],
                "fault_onset_step_max": protocol_cfg["fault_onset_step_max"],
                "vx_cmd_mode": args.velocity_mode,
                "vx_cmd": mode_cfg["target_vx"],
                "vx_cmd_mean": cmd_stats["vx_cmd_mean"],
                "vx_cmd_min": cmd_stats["vx_cmd_min"],
                "vx_cmd_max": cmd_stats["vx_cmd_max"],
                "reward_mean": scalar(rewards),
                "mean_vel_x": velocity_metrics["mean_vel_x"],
                "mean_abs_vx_error": mean_abs_error,
                "mean_yaw_rate": yaw_metrics["mean_yaw_rate"],
                "mean_abs_yaw_error": yaw_metrics["mean_abs_yaw_error"],
                "done_count": step_done_count,
                "total_done_count": total_done_count,
                "P2/fault_applied": p2_fault_applied,
                "P2/simulation_override_applied": optional_float(log_values.get("P2/simulation_override_applied")),
                "P2/fallback_used": optional_float(log_values.get("P2/fallback_used")),
                "P2/onset_step_mean": optional_float(log_values.get("P2/onset_step_mean")),
                "P2/onset_step_min": optional_float(log_values.get("P2/onset_step_min")),
                "P2/onset_step_max": optional_float(log_values.get("P2/onset_step_max")),
                "P2/per_env_onset_randomization": optional_float(log_values.get("P2/per_env_onset_randomization")),
                "P2/selected_fault_joint_index_mean": selected_index_mean,
                "P2/selected_fault_joint_index_min": selected_index_min,
                "P2/selected_fault_joint_index_max": selected_index_max,
                "P2/supported_fault_joint_count": optional_float(log_values.get("P2/supported_fault_joint_count")),
                "P2/multi_joint_randomization": optional_float(log_values.get("P2/multi_joint_randomization")),
                "velocity_metric_source": velocity_metrics["velocity_metric_source"],
                "yaw_metric_source": yaw_metrics["yaw_metric_source"],
                "no_nan_inf": no_nan_inf,
            }
            rows.append(row)
            velocity_rows.append(
                {
                    "step": step_number,
                    "policy_label": POLICY_LABEL,
                    "protocol": args.protocol,
                    "velocity_mode": args.velocity_mode,
                    "mean_vel_x": velocity_metrics["mean_vel_x"],
                    "mean_abs_vx_error": mean_abs_error,
                    "mean_vx_cmd": cmd_stats["vx_cmd_mean"],
                    "p2_fault_applied": p2_fault_applied,
                    "selected_fault_joint_index_mean": selected_index_mean,
                    "selected_fault_joint_index_min": selected_index_min,
                    "selected_fault_joint_index_max": selected_index_max,
                }
            )

            selected_indices = tensor_to_int_list(p2_wrapper.per_env_target_action_index)
            fault_mask = tensor_to_int_list(p2_wrapper.last_fault_applied_mask)
            yaw_per_env = per_env_yaw_rate(env, action_term_name=p2_action_term_name)
            if selected_indices is not None and fault_mask is not None and per_env_vel_x is not None:
                for env_index, is_fault_active in enumerate(fault_mask):
                    if not is_fault_active or env_index >= len(selected_indices) or env_index >= len(per_env_vel_x):
                        continue
                    joint_index = selected_indices[env_index]
                    per_joint = per_joint_samples.setdefault(
                        joint_index, {"vel_x": [], "abs_vx_error": [], "abs_yaw_error": []}
                    )
                    per_joint["vel_x"].append(per_env_vel_x[env_index])
                    if cmd_stats["per_env_vx_cmd"] is not None and env_index < len(cmd_stats["per_env_vx_cmd"]):
                        per_joint["abs_vx_error"].append(abs(per_env_vel_x[env_index] - cmd_stats["per_env_vx_cmd"][env_index]))
                    elif mode_cfg["target_vx"] is not None:
                        per_joint["abs_vx_error"].append(abs(per_env_vel_x[env_index] - mode_cfg["target_vx"]))
                    if yaw_per_env is not None and env_index < len(yaw_per_env):
                        per_joint["abs_yaw_error"].append(abs(yaw_per_env[env_index]))

        first_done_steps = [int(value) for value in first_done_step.detach().cpu().tolist()]
        survival_summary = build_random_onset_survival_summary(
            first_done_steps=first_done_steps,
            initial_onset_steps=initial_onset_steps,
            num_steps=args.num_steps,
        )
        pre_rows = rows_by_fault_fraction(rows, post_fault=False)
        post_rows = rows_by_fault_fraction(rows, post_fault=True)
        p2_fault_values = [value for row in rows if (value := optional_float(row.get("P2/fault_applied"))) is not None]
        fallback_values = [value for row in rows if (value := optional_float(row.get("P2/fallback_used"))) is not None]
        override_values = [
            value for row in rows if (value := optional_float(row.get("P2/simulation_override_applied"))) is not None
        ]
        selected_mean_values = [
            value for row in rows if (value := optional_float(row.get("P2/selected_fault_joint_index_mean"))) is not None
        ]
        selected_min_values = [
            value for row in rows if (value := optional_float(row.get("P2/selected_fault_joint_index_min"))) is not None
        ]
        selected_max_values = [
            value for row in rows if (value := optional_float(row.get("P2/selected_fault_joint_index_max"))) is not None
        ]
        supported_count_values = [
            value for row in rows if (value := optional_float(row.get("P2/supported_fault_joint_count"))) is not None
        ]
        multi_joint_values = [
            value for row in rows if (value := optional_float(row.get("P2/multi_joint_randomization"))) is not None
        ]

        rollout_csv = output_dir / "rollout_metrics.csv"
        velocity_csv = output_dir / "velocity_timeseries.csv"
        per_joint_csv = output_dir / "per_joint_metrics.csv"
        readme_path = output_dir / "README.md"
        per_joint_rows = per_joint_rows_from_samples(
            protocol=args.protocol,
            velocity_mode=args.velocity_mode,
            joint_names=p2_wrapper.mapping.supported_target_joints,
            output_dir=output_dir,
            samples=per_joint_samples,
        )
        command_mode_stats = command_stats_from_rows(rows)
        command_mode_valid, command_mode_validation_error = validate_command_mode(
            args.velocity_mode,
            command_mode_stats,
        )

        summary.update(
            {
                "actual_semantics": p2_wrapper.mapping.semantics,
                "fallback_used": bool(max(fallback_values) >= 0.5) if fallback_values else bool(p2_wrapper.fallback_used),
                "simulation_override_applied": max_or_blank(override_values),
                "p2_fault_became_active": any(value > 0.0 for value in p2_fault_values),
                "mean_reward": reward_sum / reward_count if reward_count else None,
                "timeout_rate": maybe_mean(timeout_values),
                "torso_height_failure_rate": maybe_mean(torso_height_failure_values),
                "mean_vel_x_pre_fault": row_mean(pre_rows, "mean_vel_x"),
                "mean_vel_x_post_fault": row_mean(post_rows, "mean_vel_x"),
                "mean_abs_vx_error_pre_fault": row_mean(pre_rows, "mean_abs_vx_error"),
                "mean_abs_vx_error_post_fault": row_mean(post_rows, "mean_abs_vx_error"),
                "mean_abs_yaw_error": row_mean(post_rows if post_rows else rows, "mean_abs_yaw_error"),
                "selected_fault_joint_index_mean": mean_or_blank(selected_mean_values),
                "selected_fault_joint_index_min": min_or_blank(selected_min_values),
                "selected_fault_joint_index_max": max_or_blank(selected_max_values),
                "supported_fault_joint_count": max_or_blank(supported_count_values),
                "supported_fault_joint_names": list(p2_wrapper.mapping.supported_target_joints),
                "multi_joint_randomization": bool(max(multi_joint_values) >= 0.5) if multi_joint_values else "",
                "onset_step_mean": row_mean(rows, "P2/onset_step_mean"),
                "onset_step_min": min_or_blank(
                    [value for row in rows if (value := optional_float(row.get("P2/onset_step_min"))) is not None]
                ),
                "onset_step_max": max_or_blank(
                    [value for row in rows if (value := optional_float(row.get("P2/onset_step_max"))) is not None]
                ),
                "teacher_policy_obs_dim": teacher_policy_obs_dim,
                "p2_fault_joint_one_hot_dim": p2_fault_joint_one_hot_dim,
                "p2_fault_q_lock_vector_dim": p2_fault_q_lock_vector_dim,
                "velocity_metric_source": velocity_metric_source,
                "yaw_metric_source": yaw_metric_source,
                "no_nan_inf": bool(no_nan_inf),
                **command_mode_stats,
                "command_mode_valid": command_mode_valid,
                "command_mode_validation_error": command_mode_validation_error,
                "post_fault_row_definition": "P2/fault_applied >= 0.5",
                "rollout_metrics_csv": repo_relative(rollout_csv),
                "velocity_timeseries_csv": repo_relative(velocity_csv),
                "per_joint_metrics_csv": repo_relative(per_joint_csv),
                **survival_summary,
            }
        )
        if summary["timeout_rate"] is None:
            summary["metric_notes"].append("timeout_rate unavailable")
        if summary["torso_height_failure_rate"] is None:
            summary["metric_notes"].append("torso_height_failure_rate unavailable")
        if summary["mean_abs_vx_error_post_fault"] is None:
            summary["metric_notes"].append("post-fault vx error unavailable")
        if not command_mode_valid:
            summary["metric_notes"].append(f"invalid command mode: {command_mode_validation_error}")

        write_csv(rollout_csv, rows, fieldnames=ROLLOUT_FIELDS)
        write_csv(velocity_csv, velocity_rows, fieldnames=VELOCITY_FIELDS)
        write_csv(per_joint_csv, per_joint_rows, fieldnames=PER_JOINT_FIELDS)
        write_json(output_dir / "summary.json", summary)
        readme_path.write_text(
            "\n".join(
                [
                    "# T13C Run Output",
                    "",
                    f"- protocol: `{args.protocol}`",
                    f"- velocity_mode: `{args.velocity_mode}`",
                    f"- checkpoint: `{repo_relative(checkpoint_path)}`",
                    "- direct simulation-state override required",
                    "- fallback/surrogate disabled",
                    "- privileged teacher/reference only",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        print(f"[T13C] summary: {repo_relative(output_dir / 'summary.json')}", flush=True)
        print(f"[T13C] rollout_metrics: {repo_relative(rollout_csv)}", flush=True)
        print(f"[T13C] velocity_timeseries: {repo_relative(velocity_csv)}", flush=True)
        if not command_mode_valid:
            print(f"[T13C ERROR] invalid command mode: {command_mode_validation_error}", file=sys.stderr, flush=True)
            return 1
        return 0
    except Exception as exc:
        summary["error_type"] = type(exc).__name__
        summary["error_message"] = str(exc)
        summary["traceback"] = traceback.format_exc()
        write_json(output_dir / "summary.json", summary)
        print(f"[T13C ERROR] {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
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
        for name in ("summary.json", "rollout_metrics.csv", "velocity_timeseries.csv")
        if not (spec.output_dir / name).is_file()
    ]
    if missing:
        raise T13CEvalError(f"missing {', '.join(missing)} in {repo_relative(spec.output_dir)}")


def run_dry_run(args: argparse.Namespace, specs: list[RunSpec]) -> int:
    output_root = resolve_repo_path(args.output_root)
    readme_path = write_root_readme(output_root)
    command_path = write_command_markdown(output_root, specs)
    print("[T13C DRY RUN]")
    print(f"planned_command_count: {len(specs)}")
    print(f"commands: {repo_relative(command_path)}")
    print(f"readme: {repo_relative(readme_path)}")
    print("no_isaac_sim_launched: true")
    print("no_training: true")
    print("no_checkpoint_modification: true")
    print("no_fake_metrics_written: true")
    for spec in specs:
        guard_single_run_command_tokens(spec.command)
        print(display_command(spec.command))
    return 0


def run_execute(args: argparse.Namespace, specs: list[RunSpec]) -> int:
    output_root = resolve_repo_path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    write_root_readme(output_root)
    write_command_markdown(output_root, specs)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["TERM"] = env.get("TERM", "xterm")
    completed: list[RunSpec] = []
    for index, spec in enumerate(specs, start=1):
        spec.output_dir.mkdir(parents=True, exist_ok=True)
        terminal_log = spec.output_dir / "terminal.log"
        guard_single_run_command_tokens(spec.command)
        render_command(spec.command)
        print(f"[T13C {index}/{len(specs)}] {spec.protocol}/{spec.velocity_mode}")
        print(display_command(spec.command))
        with terminal_log.open("w", encoding="utf-8") as log_file:
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
                raise T13CEvalError(message)
            print(f"[T13C WARNING] {message}", file=sys.stderr)
            continue
        require_run_outputs(spec)
        completed.append(spec)
        csv_path, md_path, per_joint_path, row_count = aggregate_completed(output_root, specs)
        print(f"aggregate_rows: {row_count}")
        print(f"metrics_csv: {repo_relative(csv_path)}")
        print(f"metrics_md: {repo_relative(md_path)}")
        print(f"per_joint_csv: {repo_relative(per_joint_path)}")
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
                return evaluate_single_run(args)
            finally:
                simulation_app.close()
        specs = make_specs(args)
        if args.dry_run:
            return run_dry_run(args, specs)
        return run_execute(args, specs)
    except T13CEvalError as exc:
        print(f"[T13C ERROR] {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("[T13C ERROR] interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
