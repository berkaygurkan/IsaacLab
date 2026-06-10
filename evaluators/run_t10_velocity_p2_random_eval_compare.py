#!/usr/bin/env python3
"""T10-IL-05 controlled P2-random velocity candidate comparison.

This runner evaluates A0-Vel and A1-F-Vel candidate checkpoints under the same
P2 random-onset joint-lock protocol. It is a guarded candidate comparison only:
no training, checkpoint freezing, pointer edits, or paper-grade manifest writes.
Runtime execution requires ``--execute_eval``.
"""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import statistics
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

from run_t09_p2_quick_demo_compare import (
    _asset_from_action_term,
    done_mask_tensor,
    maybe_set_target_vx,
    optional_float,
    prepare_agent_cfg,
    repo_relative,
    resolve_forward_velocity_metrics,
    resolve_repo_path,
    scalar,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
A0_LABEL = "A0_Vel_healthy_train_eval_P2_random"
A1F_LABEL = "A1F_Vel_P2_random_train_eval_P2_random"
A0_TASK = "Isaac-Ant-Velocity-Flat-v0"
A1F_TASK = "Isaac-Ant-Teacher-Velocity-Flat-v0"
POLICY_SPECS = (
    ("A0", A0_LABEL, A0_TASK, "a0_checkpoint"),
    ("A1-F", A1F_LABEL, A1F_TASK, "a1f_checkpoint"),
)
FAULT_PROFILE = "P2_locked_joint"
TARGET_JOINT = "front_left_foot"
REQUESTED_SEMANTICS = "simulation_joint_state_override_lock"
DEFAULT_NUM_ENVS = 128
DEFAULT_NUM_STEPS = 1000
DEFAULT_SEED = 0
DEFAULT_ONSET_MODE = "random_uniform"
DEFAULT_ONSET_STEP = 50
DEFAULT_ONSET_MIN = 30
DEFAULT_ONSET_MAX = 150
VX_CMD = 1.0
ROLLOUT_FIELDS = [
    "step",
    "policy_label",
    "task",
    "seed",
    "fault_profile",
    "fault_onset_mode",
    "fault_onset_step_min",
    "fault_onset_step_max",
    "target_joint",
    "vx_cmd",
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
    "velocity_metric_source",
    "yaw_metric_source",
    "no_nan_inf",
]
VELOCITY_FIELDS = ["step", "policy_label", "mean_vel_x", "mean_abs_vx_error", "p2_fault_applied"]


class T10EvalError(ValueError):
    """Raised for invalid T10 controlled velocity evaluation configuration."""


def build_parser(*, add_app_launcher_args: bool = False) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="T10-IL-05 controlled P2-random eval for A0-Vel vs A1-F-Vel candidate checkpoints."
    )
    parser.add_argument("--execute_eval", action="store_true", help="Launch Isaac Sim and run the guarded evaluation.")
    parser.add_argument("--a0_checkpoint")
    parser.add_argument("--a1f_checkpoint")
    parser.add_argument("--output_dir")
    parser.add_argument("--single_policy_label")
    parser.add_argument("--single_policy_task")
    parser.add_argument("--single_policy_checkpoint")
    parser.add_argument("--single_policy_output_dir")
    parser.add_argument("--num_envs", type=int, default=DEFAULT_NUM_ENVS)
    parser.add_argument("--num_steps", type=int, default=DEFAULT_NUM_STEPS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--fault_onset_mode", default=DEFAULT_ONSET_MODE, choices=(DEFAULT_ONSET_MODE,))
    parser.add_argument("--fault_onset_step_min", type=int, default=DEFAULT_ONSET_MIN)
    parser.add_argument("--fault_onset_step_max", type=int, default=DEFAULT_ONSET_MAX)
    parser.add_argument(
        "--allow_partial",
        action="store_true",
        help="Write partial outputs and exit zero if one policy fails. Default is fail nonzero.",
    )
    if not add_app_launcher_args:
        parser.add_argument("--headless", action="store_true")
        parser.add_argument("--device", default=None)
    if add_app_launcher_args:
        from isaaclab.app import AppLauncher

        AppLauncher.add_app_launcher_args(parser)
    return parser


def parse_args() -> argparse.Namespace:
    pre_parser = build_parser(add_app_launcher_args=False)
    pre_args, _ = pre_parser.parse_known_args()
    if pre_args.execute_eval and is_single_policy_mode(pre_args):
        parser = build_parser(add_app_launcher_args=True)
        args, _ = parser.parse_known_args()
        return args
    return pre_args


def is_single_policy_mode(args: argparse.Namespace) -> bool:
    return any(
        getattr(args, name, None)
        for name in (
            "single_policy_label",
            "single_policy_task",
            "single_policy_checkpoint",
            "single_policy_output_dir",
        )
    )


def validate_args(args: argparse.Namespace) -> None:
    if args.num_envs <= 0:
        raise T10EvalError("--num_envs must be > 0.")
    if args.num_steps <= 0:
        raise T10EvalError("--num_steps must be > 0.")
    if args.seed < 0:
        raise T10EvalError("--seed must be non-negative.")
    if args.fault_onset_step_min < 0 or args.fault_onset_step_max < 0:
        raise T10EvalError("fault onset bounds must be non-negative.")
    if args.fault_onset_step_min > args.fault_onset_step_max:
        raise T10EvalError("--fault_onset_step_min must be <= --fault_onset_step_max.")
    if args.fault_onset_mode != DEFAULT_ONSET_MODE:
        raise T10EvalError("T10-IL-05 supports only P2 random_uniform onset.")
    if is_single_policy_mode(args):
        missing = [
            name
            for name in (
                "single_policy_label",
                "single_policy_task",
                "single_policy_checkpoint",
                "single_policy_output_dir",
            )
            if not getattr(args, name, None)
        ]
        if missing:
            raise T10EvalError(f"single-policy mode requires all single-policy args; missing {missing}.")
        if args.single_policy_label not in {A0_LABEL, A1F_LABEL}:
            raise T10EvalError(f"unsupported single_policy_label: {args.single_policy_label!r}.")
        if args.single_policy_task not in {A0_TASK, A1F_TASK}:
            raise T10EvalError(f"unsupported single_policy_task: {args.single_policy_task!r}.")
        if args.execute_eval:
            checkpoint_path = resolve_repo_path(args.single_policy_checkpoint)
            if not checkpoint_path.is_file():
                raise T10EvalError(f"single-policy checkpoint does not exist: {repo_relative(checkpoint_path)}")
        return

    missing = [name for name in ("a0_checkpoint", "a1f_checkpoint", "output_dir") if not getattr(args, name, None)]
    if missing:
        raise T10EvalError(f"compare mode requires {missing}.")
    if args.execute_eval:
        for label, checkpoint in (("A0", args.a0_checkpoint), ("A1-F", args.a1f_checkpoint)):
            checkpoint_path = resolve_repo_path(checkpoint)
            if not checkpoint_path.is_file():
                raise T10EvalError(f"{label} checkpoint does not exist: {repo_relative(checkpoint_path)}")


def policy_output_dir(parent_output_dir: Path, short_label: str) -> Path:
    if short_label == "A0":
        return parent_output_dir / "a0_single"
    if short_label == "A1-F":
        return parent_output_dir / "a1f_single"
    return parent_output_dir / f"{short_label.lower()}_single"


def build_single_policy_command(
    args: argparse.Namespace,
    *,
    short_label: str,
    policy_label: str,
    task: str,
    checkpoint: str,
    output_dir: Path,
) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--execute_eval",
        "--single_policy_label",
        policy_label,
        "--single_policy_task",
        task,
        "--single_policy_checkpoint",
        checkpoint,
        "--single_policy_output_dir",
        str(output_dir),
        "--num_envs",
        str(args.num_envs),
        "--num_steps",
        str(args.num_steps),
        "--seed",
        str(args.seed),
        "--fault_onset_mode",
        args.fault_onset_mode,
        "--fault_onset_step_min",
        str(args.fault_onset_step_min),
        "--fault_onset_step_max",
        str(args.fault_onset_step_max),
    ]
    if getattr(args, "headless", False):
        command.append("--headless")
    if getattr(args, "device", None):
        command.extend(["--device", str(args.device)])
    return command


def planned_subprocess_commands(args: argparse.Namespace) -> list[tuple[str, str, Path, list[str]]]:
    parent_output_dir = resolve_repo_path(args.output_dir)
    commands: list[tuple[str, str, Path, list[str]]] = []
    for short_label, policy_label, task, checkpoint_attr in POLICY_SPECS:
        child_output_dir = policy_output_dir(parent_output_dir, short_label)
        checkpoint = getattr(args, checkpoint_attr)
        commands.append(
            (
                short_label,
                policy_label,
                child_output_dir,
                build_single_policy_command(
                    args,
                    short_label=short_label,
                    policy_label=policy_label,
                    task=task,
                    checkpoint=checkpoint,
                    output_dir=child_output_dir,
                ),
            )
        )
    return commands


def print_preview(args: argparse.Namespace) -> None:
    output_dir = resolve_repo_path(args.output_dir)
    print("[T10-IL-05 CONTROLLED P2-RANDOM EVAL PREVIEW]")
    print("  not_paper_grade_final_evaluation: True")
    print("  no_isaac_sim_launched: True")
    print("  no_training: True")
    print("  no_checkpoint_freeze: True")
    print("  no_checkpoint_pointer_edit: True")
    print("  no_paper_grade_manifest_update: True")
    print(f"  output_dir: {repo_relative(output_dir)}")
    print(f"  num_envs: {args.num_envs}")
    print(f"  num_steps: {args.num_steps}")
    print(f"  seed: {args.seed}")
    print(f"  target_joint: {TARGET_JOINT}")
    print(f"  requested_semantics: {REQUESTED_SEMANTICS}")
    print("  fallback_allowed: False")
    print(f"  fault_onset_mode: {args.fault_onset_mode}")
    print(f"  fault_onset_step_min: {args.fault_onset_step_min}")
    print(f"  fault_onset_step_max: {args.fault_onset_step_max}")
    print(f"  vx_cmd: {VX_CMD}")
    print("  policies:")
    print(f"    {A0_LABEL}: task={A0_TASK}, checkpoint={args.a0_checkpoint}")
    print(f"    {A1F_LABEL}: task={A1F_TASK}, checkpoint={args.a1f_checkpoint}")
    print(f"  allow_partial: {args.allow_partial}")
    print("  planned_subprocess_commands:")
    for short_label, policy_label, child_output_dir, command in planned_subprocess_commands(args):
        print(f"    {short_label}:")
        print(f"      policy_label: {policy_label}")
        print(f"      output_dir: {repo_relative(child_output_dir)}")
        print(f"      command: {shlex.join(command)}")
    print("  execute_eval_required: True")


def write_json(path: Path, values: dict[str, Any]) -> None:
    path.write_text(json.dumps(values, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], *, fieldnames: list[str] | None = None) -> None:
    if fieldnames is None:
        fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        return list(reader)


def write_outputs(
    *,
    output_dir: Path,
    summary: dict[str, Any],
    all_rows: list[dict[str, Any]],
    all_velocity_rows: list[dict[str, Any]],
) -> None:
    rollout_csv = output_dir / "rollout_metrics.csv"
    velocity_csv = output_dir / "velocity_timeseries.csv"
    write_csv(rollout_csv, all_rows, fieldnames=ROLLOUT_FIELDS)
    write_csv(velocity_csv, all_velocity_rows, fieldnames=VELOCITY_FIELDS)
    summary["rollout_metrics_csv"] = repo_relative(rollout_csv)
    summary["velocity_timeseries_csv"] = repo_relative(velocity_csv)
    write_json(output_dir / "summary.json", summary)


def maybe_mean(values: list[float]) -> float | None:
    return float(statistics.mean(values)) if values else None


def maybe_median(values: list[float]) -> float | None:
    return float(statistics.median(values)) if values else None


def mean_row_value(
    rows: list[dict[str, Any]],
    key: str,
    *,
    start_step: int | None = None,
    end_step: int | None = None,
) -> float | None:
    values: list[float] = []
    for row in rows:
        step = int(row.get("step") or 0)
        if start_step is not None and step < start_step:
            continue
        if end_step is not None and step >= end_step:
            continue
        value = optional_float(row.get(key))
        if value is not None:
            values.append(value)
    return maybe_mean(values)


def mean_abs_vx_error_window(
    rows: list[dict[str, Any]],
    *,
    start_step: int | None = None,
    end_step: int | None = None,
) -> float | None:
    values: list[float] = []
    for row in rows:
        step = int(row.get("step") or 0)
        if start_step is not None and step < start_step:
            continue
        if end_step is not None and step >= end_step:
            continue
        vel_x = optional_float(row.get("mean_vel_x"))
        if vel_x is not None:
            values.append(abs(vel_x - VX_CMD))
    return maybe_mean(values)


def resolve_yaw_rate_metrics(env: Any, extras: Any, *, action_term_name: str | None = None) -> dict[str, Any]:
    import torch

    candidates: list[tuple[str, Any]] = []
    asset = _asset_from_action_term(env, action_term_name)
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
        if value is not None:
            candidates.append((f"robot.data.{attr_name}", value))
    for state_name in ("root_state_w", "root_link_state_w"):
        state = getattr(data, state_name, None)
        if state is not None:
            try:
                candidates.append((f"robot.data.{state_name}[:, 10:13]", state[:, 10:13]))
            except Exception:
                pass
    if isinstance(extras, dict):
        for container_name in ("log", "episode", "metrics"):
            container = extras.get(container_name)
            if not isinstance(container, dict):
                continue
            for key in ("base_ang_vel_z", "base_ang_vel_z_mean", "mean_yaw_rate", "yaw_rate"):
                if key in container:
                    candidates.append((f"extras.{container_name}.{key}", container[key]))

    for source, value in candidates:
        try:
            tensor = torch.as_tensor(value).detach().float()
            if tensor.numel() == 0 or not torch.isfinite(tensor).all():
                continue
            if tensor.ndim == 0:
                yaw = tensor.reshape(1)
            elif tensor.ndim == 1:
                yaw = tensor if tensor.numel() == 1 else tensor[-1].reshape(1)
            else:
                yaw = tensor[:, -1]
            return {
                "mean_yaw_rate": float(yaw.mean().cpu().item()),
                "mean_abs_yaw_error": float(yaw.abs().mean().cpu().item()),
                "yaw_metric_source": source,
            }
        except Exception:
            continue
    return {"mean_yaw_rate": None, "mean_abs_yaw_error": None, "yaw_metric_source": None}


def first_matching_log_value(log_values: dict[str, Any], tokens: tuple[str, ...]) -> float | None:
    for key, value in log_values.items():
        lowered = str(key).lower()
        if all(token in lowered for token in tokens):
            parsed = optional_float(value)
            if parsed is not None:
                return parsed
    return None


def current_onset_steps(p2_wrapper: Any, *, num_envs: int) -> list[int] | None:
    try:
        p2_wrapper._ensure_lock_buffers()
        onset = getattr(p2_wrapper, "per_env_fault_onset_step", None)
        if onset is None:
            return None
        values = [int(value) for value in onset.detach().cpu().tolist()]
        return values if len(values) == num_envs else None
    except Exception:
        return None


def build_random_onset_survival_summary(
    *,
    first_done_steps: list[int],
    initial_onset_steps: list[int] | None,
    num_steps: int,
) -> dict[str, Any]:
    if initial_onset_steps is None:
        return {
            "initial_onset_step_by_env": None,
            "survived_to_fault_onset_rate": None,
            "failed_before_fault_count": None,
            "reached_fault_count": None,
            "mean_first_done_step": maybe_mean([float(step) for step in first_done_steps if step >= 0]),
            "median_first_done_step": maybe_median([float(step) for step in first_done_steps if step >= 0]),
            "mean_post_fault_survival_steps": None,
            "median_post_fault_survival_steps": None,
            "survival_metric_note": "initial per-env random onset steps unavailable",
        }

    failed_before_fault: list[bool] = []
    reached_fault: list[bool] = []
    post_fault_survival_steps: list[float] = []
    for first_done_step, onset_step in zip(first_done_steps, initial_onset_steps):
        failed = 0 <= first_done_step < onset_step
        reached = (not failed) and num_steps >= onset_step
        failed_before_fault.append(failed)
        reached_fault.append(reached)
        if not reached:
            continue
        if first_done_step >= onset_step:
            post_fault_survival_steps.append(float(max(0, first_done_step - onset_step)))
        else:
            post_fault_survival_steps.append(float(max(0, num_steps - onset_step)))

    observed_done_steps = [float(step) for step in first_done_steps if step >= 0]
    reached_count = sum(1 for item in reached_fault if item)
    failed_count = sum(1 for item in failed_before_fault if item)
    env_count = len(first_done_steps)
    return {
        "initial_onset_step_by_env": initial_onset_steps,
        "survived_to_fault_onset_rate": reached_count / env_count if env_count else None,
        "failed_before_fault_count": failed_count,
        "reached_fault_count": reached_count,
        "mean_first_done_step": maybe_mean(observed_done_steps),
        "median_first_done_step": maybe_median(observed_done_steps),
        "mean_post_fault_survival_steps": maybe_mean(post_fault_survival_steps),
        "median_post_fault_survival_steps": maybe_median(post_fault_survival_steps),
        "survival_metric_note": "computed from first observed done per env and initial sampled random onset",
    }


def set_torch_seed(seed: int) -> None:
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_one_policy(
    *,
    args: argparse.Namespace,
    policy_label: str,
    task: str,
    checkpoint_path: str,
    output_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
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

    env = None
    vec_env = None
    p2_wrapper = None
    rows: list[dict[str, Any]] = []
    velocity_rows: list[dict[str, Any]] = []
    resolved_checkpoint = resolve_repo_path(checkpoint_path)
    if not resolved_checkpoint.is_file():
        raise T10EvalError(f"{policy_label}: checkpoint does not exist: {repo_relative(resolved_checkpoint)}")
    summary: dict[str, Any] = {
        "policy_label": policy_label,
        "task": task,
        "checkpoint_path": repo_relative(resolved_checkpoint),
        "fault_profile": FAULT_PROFILE,
        "target_joint": TARGET_JOINT,
        "requested_semantics": REQUESTED_SEMANTICS,
        "fallback_allowed": False,
        "fault_onset_mode": args.fault_onset_mode,
        "fault_onset_step_min": args.fault_onset_step_min,
        "fault_onset_step_max": args.fault_onset_step_max,
        "vx_cmd": VX_CMD,
        "metric_notes": [],
    }

    try:
        print(f"[T10-IL-05] {policy_label}: config load start", flush=True)
        env_cfg = load_cfg_from_registry(task, "env_cfg_entry_point")
        agent_cfg = load_cfg_from_registry(task, "rsl_rl_cfg_entry_point")
        env_cfg.scene.num_envs = args.num_envs
        env_cfg.seed = args.seed
        if getattr(args, "device", None) is not None:
            env_cfg.sim.device = args.device
        env_cfg.log_dir = str(output_dir / policy_label)
        set_torch_seed(args.seed)
        print(f"[T10-IL-05] {policy_label}: config load done", flush=True)

        print(f"[T10-IL-05] {policy_label}: gym.make start", flush=True)
        env = gym.make(task, cfg=env_cfg)
        print(f"[T10-IL-05] {policy_label}: gym.make done", flush=True)
        if isinstance(env.unwrapped, DirectMARLEnv):
            env = multi_agent_to_single_agent(env)

        print(f"[T10-IL-05] {policy_label}: P2 wrapper attach start", flush=True)
        p2_wrapper = P2JointLockActionMaskWrapper(
            env,
            target_joint=TARGET_JOINT,
            fault_onset_step=DEFAULT_ONSET_STEP,
            fault_onset_mode=args.fault_onset_mode,
            fault_onset_step_min=args.fault_onset_step_min,
            fault_onset_step_max=args.fault_onset_step_max,
            expected_action_dim=8,
            requested_semantics=REQUESTED_SEMANTICS,
            allow_fallback=False,
            velocity_override=0.0,
            debug=True,
        )
        env = p2_wrapper
        if p2_wrapper.mapping.semantics != REQUESTED_SEMANTICS:
            raise T10EvalError(f"{policy_label}: actual P2 semantics must be {REQUESTED_SEMANTICS}.")
        print(f"[T10-IL-05] {policy_label}: P2 wrapper attach done", flush=True)

        target_vx_info = maybe_set_target_vx(env, VX_CMD)
        summary.update(target_vx_info)
        agent_cfg, agent_cfg_dict = prepare_agent_cfg(agent_cfg)
        vec_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

        print(f"[T10-IL-05] {policy_label}: runner construction start", flush=True)
        if agent_cfg.class_name == "OnPolicyRunner":
            runner = OnPolicyRunner(vec_env, agent_cfg_dict, log_dir=None, device=agent_cfg.device)
        elif agent_cfg.class_name == "DistillationRunner":
            runner = DistillationRunner(vec_env, agent_cfg_dict, log_dir=None, device=agent_cfg.device)
        else:
            raise T10EvalError(f"{policy_label}: unsupported runner class {agent_cfg.class_name!r}.")
        print(f"[T10-IL-05] {policy_label}: runner construction done", flush=True)

        print(f"[T10-IL-05] {policy_label}: checkpoint load start {repo_relative(resolved_checkpoint)}", flush=True)
        runner.load(str(resolved_checkpoint))
        print(f"[T10-IL-05] {policy_label}: checkpoint load done", flush=True)
        policy = runner.get_inference_policy(device=vec_env.unwrapped.device)

        obs = vec_env.get_observations()
        tracking_num_envs = int(getattr(vec_env, "num_envs", args.num_envs))
        tracking_device = vec_env.unwrapped.device
        set_torch_seed(args.seed)
        initial_onset_steps = current_onset_steps(p2_wrapper, num_envs=tracking_num_envs)
        first_done_step = torch.full((tracking_num_envs,), -1, dtype=torch.long, device=tracking_device)
        p2_action_term_name = p2_wrapper.mapping.action_term_name
        reward_sum = 0.0
        reward_count = 0
        total_done_count = 0
        no_nan_inf = True
        velocity_metric_source = None
        yaw_metric_source = None
        timeout_values: list[float] = []
        torso_height_failure_values: list[float] = []
        last_log_values: dict[str, Any] = {}
        warned_velocity = False
        warned_yaw = False

        print(f"[T10-IL-05] {policy_label}: rollout start", flush=True)
        for step_index in range(args.num_steps):
            step_number = step_index + 1
            with torch.inference_mode():
                actions = policy(obs)
                obs, rewards, dones, extras = vec_env.step(actions)
                if hasattr(policy, "reset"):
                    policy.reset(dones)

            reward_mean = scalar(rewards)
            reward_tensor = torch.as_tensor(rewards).detach().float()
            no_nan_inf = no_nan_inf and bool(torch.isfinite(reward_tensor).all().item())
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
                print(f"[T10-IL-05] {policy_label}: velocity_metric_source={velocity_metric_source}", flush=True)
            if not velocity_metrics["velocity_available"] and not warned_velocity:
                warned_velocity = True
                print(f"[T10-IL-05 WARNING] {policy_label}: forward velocity unavailable", flush=True)

            yaw_metrics = resolve_yaw_rate_metrics(env, extras, action_term_name=p2_action_term_name)
            if yaw_metrics["yaw_metric_source"] and yaw_metric_source is None:
                yaw_metric_source = yaw_metrics["yaw_metric_source"]
            if yaw_metrics["yaw_metric_source"] is None and not warned_yaw:
                warned_yaw = True
                summary["metric_notes"].append("mean_abs_yaw_error unavailable")

            mean_vel_x = velocity_metrics["mean_vel_x"]
            step_mean_abs_vx_error = None if mean_vel_x is None else abs(float(mean_vel_x) - VX_CMD)
            row = {
                "step": step_number,
                "policy_label": policy_label,
                "task": task,
                "seed": args.seed,
                "fault_profile": FAULT_PROFILE,
                "fault_onset_mode": args.fault_onset_mode,
                "fault_onset_step_min": args.fault_onset_step_min,
                "fault_onset_step_max": args.fault_onset_step_max,
                "target_joint": TARGET_JOINT,
                "vx_cmd": VX_CMD,
                "reward_mean": reward_mean,
                "mean_vel_x": mean_vel_x,
                "mean_abs_vx_error": step_mean_abs_vx_error,
                "mean_yaw_rate": yaw_metrics["mean_yaw_rate"],
                "mean_abs_yaw_error": yaw_metrics["mean_abs_yaw_error"],
                "done_count": step_done_count,
                "total_done_count": total_done_count,
                "P2/fault_applied": scalar(log_values.get("P2/fault_applied")),
                "P2/simulation_override_applied": scalar(log_values.get("P2/simulation_override_applied")),
                "P2/fallback_used": scalar(log_values.get("P2/fallback_used")),
                "P2/onset_step_mean": scalar(log_values.get("P2/onset_step_mean")),
                "P2/onset_step_min": scalar(log_values.get("P2/onset_step_min")),
                "P2/onset_step_max": scalar(log_values.get("P2/onset_step_max")),
                "P2/per_env_onset_randomization": scalar(log_values.get("P2/per_env_onset_randomization")),
                "velocity_metric_source": velocity_metrics["velocity_metric_source"],
                "yaw_metric_source": yaw_metrics["yaw_metric_source"],
                "no_nan_inf": no_nan_inf,
            }
            rows.append(row)
            velocity_rows.append(
                {
                    "step": step_number,
                    "policy_label": policy_label,
                    "mean_vel_x": mean_vel_x,
                    "mean_abs_vx_error": step_mean_abs_vx_error,
                    "p2_fault_applied": row["P2/fault_applied"],
                }
            )
            if step_index < 3 or step_number % 100 == 0 or step_number == args.num_steps:
                print(
                    f"[T10-IL-05] {policy_label}: step={step_number} "
                    f"reward_mean={reward_mean:.4f} done_count={step_done_count} "
                    f"p2_fault={row['P2/fault_applied']:.4f}",
                    flush=True,
                )

        first_done_steps = [int(value) for value in first_done_step.detach().cpu().tolist()]
        survival_summary = build_random_onset_survival_summary(
            first_done_steps=first_done_steps,
            initial_onset_steps=initial_onset_steps,
            num_steps=args.num_steps,
        )
        pre_start = 1
        pre_end = args.fault_onset_step_min
        post_start = args.fault_onset_step_max
        post_end = args.num_steps + 1
        timeout_rate = maybe_mean(timeout_values)
        torso_height_failure_rate = maybe_mean(torso_height_failure_values)
        if timeout_rate is None:
            summary["metric_notes"].append("timeout_rate unavailable")
        if torso_height_failure_rate is None:
            summary["metric_notes"].append("torso_height_failure_rate unavailable")
        if velocity_metric_source is None:
            summary["metric_notes"].append("forward velocity unavailable")

        summary.update(
            {
                "actual_semantics": p2_wrapper.mapping.semantics,
                "fallback_used": bool(p2_wrapper.fallback_used),
                "total_reward_mean": reward_sum / reward_count if reward_count else None,
                "mean_reward": reward_sum / reward_count if reward_count else None,
                "episode_length_mean": maybe_mean([float(step) for step in first_done_steps if step >= 0]),
                "timeout_rate": timeout_rate,
                "torso_height_failure_rate": torso_height_failure_rate,
                "mean_vel_x_pre_fault": mean_row_value(rows, "mean_vel_x", start_step=pre_start, end_step=pre_end),
                "mean_vel_x_post_fault": mean_row_value(rows, "mean_vel_x", start_step=post_start, end_step=post_end),
                "mean_abs_vx_error_pre_fault": mean_abs_vx_error_window(
                    rows,
                    start_step=pre_start,
                    end_step=pre_end,
                ),
                "mean_abs_vx_error_post_fault": mean_abs_vx_error_window(
                    rows,
                    start_step=post_start,
                    end_step=post_end,
                ),
                "mean_abs_yaw_error": mean_row_value(rows, "mean_abs_yaw_error"),
                "velocity_metric_source": velocity_metric_source,
                "yaw_metric_source": yaw_metric_source,
                "P2/fault_applied": scalar(last_log_values.get("P2/fault_applied")),
                "P2/simulation_override_applied": scalar(last_log_values.get("P2/simulation_override_applied")),
                "P2/fallback_used": scalar(last_log_values.get("P2/fallback_used")),
                "P2/onset_step_mean": scalar(last_log_values.get("P2/onset_step_mean")),
                "P2/onset_step_min": scalar(last_log_values.get("P2/onset_step_min")),
                "P2/onset_step_max": scalar(last_log_values.get("P2/onset_step_max")),
                "P2/per_env_onset_randomization": scalar(last_log_values.get("P2/per_env_onset_randomization")),
                "first_done_step_by_env": first_done_steps,
                "done_count": total_done_count,
                "total_done_count": total_done_count,
                "no_nan_inf": no_nan_inf,
                "pre_fault_window": [pre_start, pre_end],
                "post_fault_window": [post_start, post_end],
                **survival_summary,
            }
        )
        return summary, rows, velocity_rows
    finally:
        if vec_env is not None:
            vec_env.close()
        elif env is not None:
            env.close()


def build_comparison(policy_summaries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    a0 = policy_summaries.get(A0_LABEL, {})
    a1f = policy_summaries.get(A1F_LABEL, {})
    if not a0 or not a1f:
        return {}

    def delta(key: str) -> float | None:
        left = optional_float(a1f.get(key))
        right = optional_float(a0.get(key))
        if left is None or right is None:
            return None
        return left - right

    def better_by_lower(key: str) -> str | None:
        a0_value = optional_float(a0.get(key))
        a1f_value = optional_float(a1f.get(key))
        if a0_value is None or a1f_value is None:
            return None
        if a0_value < a1f_value:
            return A0_LABEL
        if a1f_value < a0_value:
            return A1F_LABEL
        return "tie"

    def better_by_higher(key: str) -> str | None:
        a0_value = optional_float(a0.get(key))
        a1f_value = optional_float(a1f.get(key))
        if a0_value is None or a1f_value is None:
            return None
        if a0_value > a1f_value:
            return A0_LABEL
        if a1f_value > a0_value:
            return A1F_LABEL
        return "tie"

    return {
        "a1f_minus_a0_total_reward_mean": delta("total_reward_mean"),
        "a1f_minus_a0_mean_vel_x_post_fault": delta("mean_vel_x_post_fault"),
        "a1f_minus_a0_mean_abs_vx_error_post_fault": delta("mean_abs_vx_error_post_fault"),
        "a1f_minus_a0_survived_to_fault_onset_rate": delta("survived_to_fault_onset_rate"),
        "delta_mean_abs_vx_error_post_fault": delta("mean_abs_vx_error_post_fault"),
        "delta_torso_height_failure_rate": delta("torso_height_failure_rate"),
        "delta_timeout_rate": delta("timeout_rate"),
        "delta_mean_post_fault_survival_steps": delta("mean_post_fault_survival_steps"),
        "better_policy_by_post_fault_vx_error": better_by_lower("mean_abs_vx_error_post_fault"),
        "better_policy_by_survival": better_by_higher("mean_post_fault_survival_steps"),
        "interpretation_note": "A1-F is privileged/reference only; this is not a fair deployment comparison.",
    }


def execute_single_policy(args: argparse.Namespace) -> int:
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    output_dir = resolve_repo_path(args.single_policy_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")

    summary: dict[str, Any] = {
        "eval_scope": "controlled_p2_random_velocity_single_policy",
        "not_paper_grade_final_evaluation": True,
        "note": "single-policy subprocess result for controlled candidate comparison only",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "output_dir": repo_relative(output_dir),
        "num_envs": args.num_envs,
        "num_steps": args.num_steps,
        "seed": args.seed,
        "fault_profile": FAULT_PROFILE,
        "target_joint": TARGET_JOINT,
        "requested_semantics": REQUESTED_SEMANTICS,
        "fallback_allowed": False,
        "fault_onset_mode": args.fault_onset_mode,
        "fault_onset_step_min": args.fault_onset_step_min,
        "fault_onset_step_max": args.fault_onset_step_max,
        "vx_cmd": VX_CMD,
        "single_policy_label": args.single_policy_label,
        "single_policy_task": args.single_policy_task,
        "single_policy_checkpoint": repo_relative(args.single_policy_checkpoint),
        "policies": {},
        "policy_errors": {},
        "comparison": {},
    }
    all_rows: list[dict[str, Any]] = []
    all_velocity_rows: list[dict[str, Any]] = []

    try:
        print("[T10-IL-05 WARNING] This is not paper-grade final evaluation.", flush=True)
        policy_summary, rows, velocity_rows = evaluate_one_policy(
            args=args,
            policy_label=args.single_policy_label,
            task=args.single_policy_task,
            checkpoint_path=args.single_policy_checkpoint,
            output_dir=output_dir,
        )
        summary["policies"][args.single_policy_label] = policy_summary
        summary["policy_summary"] = policy_summary
        all_rows.extend(rows)
        all_velocity_rows.extend(velocity_rows)
        write_outputs(output_dir=output_dir, summary=summary, all_rows=all_rows, all_velocity_rows=all_velocity_rows)
        print(f"[T10-IL-05] single summary: {repo_relative(output_dir / 'summary.json')}", flush=True)
        print(f"[T10-IL-05] single rollout_metrics_csv: {summary.get('rollout_metrics_csv')}", flush=True)
        print(f"[T10-IL-05] single velocity_timeseries_csv: {summary.get('velocity_timeseries_csv')}", flush=True)
        return 0
    except Exception as exc:
        traceback_text = traceback.format_exc()
        print(f"[T10-IL-05 ERROR] single policy failed: {args.single_policy_label}", flush=True)
        print(traceback_text, flush=True)
        summary["policy_errors"][args.single_policy_label] = {
            "task": args.single_policy_task,
            "checkpoint_path": repo_relative(args.single_policy_checkpoint),
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "traceback": traceback_text,
        }
        write_outputs(output_dir=output_dir, summary=summary, all_rows=all_rows, all_velocity_rows=all_velocity_rows)
        return 1
    finally:
        try:
            write_outputs(output_dir=output_dir, summary=summary, all_rows=all_rows, all_velocity_rows=all_velocity_rows)
        except Exception:
            pass
        simulation_app.close()


def execute_eval(args: argparse.Namespace) -> int:
    output_dir = resolve_repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")

    commands = planned_subprocess_commands(args)
    subprocess_command_lines = [shlex.join(command) for _, _, _, command in commands]
    (output_dir / "subprocess_commands.txt").write_text(
        "\n".join(subprocess_command_lines) + "\n",
        encoding="utf-8",
    )

    summary: dict[str, Any] = {
        "eval_scope": "controlled_p2_random_velocity_candidate_comparison",
        "runner_mode": "subprocess_per_policy",
        "not_paper_grade_final_evaluation": True,
        "note": "controlled candidate comparison only; A1-F is privileged and not deployment-facing",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "output_dir": repo_relative(output_dir),
        "num_envs": args.num_envs,
        "num_steps": args.num_steps,
        "seed": args.seed,
        "fault_profile": FAULT_PROFILE,
        "target_joint": TARGET_JOINT,
        "requested_semantics": REQUESTED_SEMANTICS,
        "fallback_allowed": False,
        "fault_onset_mode": args.fault_onset_mode,
        "fault_onset_step_min": args.fault_onset_step_min,
        "fault_onset_step_max": args.fault_onset_step_max,
        "vx_cmd": VX_CMD,
        "allow_partial": args.allow_partial,
        "subprocess_commands_txt": repo_relative(output_dir / "subprocess_commands.txt"),
        "policy_specs": [
            {
                "short_label": short_label,
                "policy_label": policy_label,
                "task": task,
                "checkpoint_path": repo_relative(getattr(args, checkpoint_attr)),
                "single_output_dir": repo_relative(policy_output_dir(output_dir, short_label)),
            }
            for short_label, policy_label, task, checkpoint_attr in POLICY_SPECS
        ],
        "policies": {},
        "policy_errors": {},
        "comparison": {},
    }
    all_rows: list[dict[str, Any]] = []
    all_velocity_rows: list[dict[str, Any]] = []

    try:
        print("[T10-IL-05 WARNING] This is not paper-grade final evaluation.", flush=True)
        for index, (short_label, policy_label, child_output_dir, command) in enumerate(commands, start=1):
            print(f"EVALUATING POLICY {index}/2: {short_label}...", flush=True)
            print(f"[T10-IL-05] subprocess command: {shlex.join(command)}", flush=True)
            child_output_dir.mkdir(parents=True, exist_ok=True)
            completed = subprocess.run(command, cwd=REPO_ROOT)
            child_summary_path = child_output_dir / "summary.json"
            child_summary: dict[str, Any] = {}
            if child_summary_path.is_file():
                try:
                    child_summary = json.loads(child_summary_path.read_text(encoding="utf-8"))
                except json.JSONDecodeError as exc:
                    summary["policy_errors"][policy_label] = {
                        "short_label": short_label,
                        "single_output_dir": repo_relative(child_output_dir),
                        "error_type": type(exc).__name__,
                        "error_message": f"could not parse child summary.json: {exc}",
                        "returncode": completed.returncode,
                    }
            else:
                summary["policy_errors"][policy_label] = {
                    "short_label": short_label,
                    "single_output_dir": repo_relative(child_output_dir),
                    "error_type": "MissingChildSummary",
                    "error_message": "child summary.json was not written",
                    "returncode": completed.returncode,
                }

            child_rollout_rows = read_csv_rows(child_output_dir / "rollout_metrics.csv")
            child_velocity_rows = read_csv_rows(child_output_dir / "velocity_timeseries.csv")
            all_rows.extend(child_rollout_rows)
            all_velocity_rows.extend(child_velocity_rows)

            child_policy_summary = child_summary.get("policies", {}).get(policy_label)
            if child_policy_summary is None:
                child_policy_summary = child_summary.get("policy_summary")
            if child_policy_summary is not None:
                summary["policies"][policy_label] = child_policy_summary

            child_errors = child_summary.get("policy_errors", {})
            if child_errors:
                summary["policy_errors"].update(child_errors)
            if completed.returncode != 0:
                summary["policy_errors"].setdefault(
                    policy_label,
                    {
                        "short_label": short_label,
                        "single_output_dir": repo_relative(child_output_dir),
                        "error_type": "SubprocessFailed",
                        "error_message": f"subprocess exited with return code {completed.returncode}",
                        "returncode": completed.returncode,
                    },
                )

            write_outputs(output_dir=output_dir, summary=summary, all_rows=all_rows, all_velocity_rows=all_velocity_rows)

        expected_labels = {policy_label for _, policy_label, _, _ in POLICY_SPECS}
        observed_labels = set(summary["policies"].keys())
        missing_labels = sorted(expected_labels - observed_labels)
        if missing_labels:
            summary["missing_policy_labels"] = missing_labels
            for missing_label in missing_labels:
                summary["policy_errors"].setdefault(
                    missing_label,
                    {
                        "error_type": "MissingPolicySummary",
                        "error_message": "policy did not produce a merged summary",
                    },
                )

        summary["comparison"] = build_comparison(summary["policies"])
        if not summary["comparison"]:
            summary["comparison_error"] = "comparison is empty because both policy summaries are not available"

        write_outputs(output_dir=output_dir, summary=summary, all_rows=all_rows, all_velocity_rows=all_velocity_rows)
        print(f"[T10-IL-05] summary: {repo_relative(output_dir / 'summary.json')}", flush=True)
        print(f"[T10-IL-05] rollout_metrics_csv: {summary.get('rollout_metrics_csv')}", flush=True)
        print(f"[T10-IL-05] velocity_timeseries_csv: {summary.get('velocity_timeseries_csv')}", flush=True)
        if (summary["policy_errors"] or not summary["comparison"]) and not args.allow_partial:
            return 1
        return 0
    finally:
        try:
            write_outputs(
                output_dir=output_dir,
                summary=summary,
                all_rows=all_rows,
                all_velocity_rows=all_velocity_rows,
            )
        except Exception:
            pass


def main() -> int:
    args = parse_args()
    validate_args(args)
    if not args.execute_eval:
        print_preview(args)
        return 0
    if is_single_policy_mode(args):
        return execute_single_policy(args)
    return execute_eval(args)


if __name__ == "__main__":
    raise SystemExit(main())
