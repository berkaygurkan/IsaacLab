#!/usr/bin/env python3
"""Isaac-side A2 student P2 evaluation scaffold.

This guarded evaluator loads offline A2 student checkpoints and runs them as
deployment-facing policies on ``Isaac-Ant-Velocity-Flat-v0`` under the repo-owned
P2 locked-joint wrapper. Runtime execution requires ``--execute_eval``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import traceback
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
    maybe_set_target_vx,
    optional_float,
    prepare_agent_cfg,
    repo_relative,
    resolve_repo_path,
    scalar,
)
from run_t10_velocity_p2_random_eval_compare import (
    first_matching_log_value,
    resolve_yaw_rate_metrics,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
TASK = "Isaac-Ant-Velocity-Flat-v0"
FAULT_PROFILE = "P2_locked_joint"
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
DEFAULT_TARGET_JOINT = "front_left_foot"
REQUESTED_SEMANTICS = "simulation_joint_state_override_lock"
DEFAULT_ONSET_MODE = "random_uniform"
DEFAULT_ONSET_STEP = 50
DEFAULT_ONSET_MIN = 30
DEFAULT_ONSET_MAX = 150
DEFAULT_NUM_ENVS = 16
DEFAULT_NUM_STEPS = 200
DEFAULT_SEED = 0
DEFAULT_HISTORY_LEN = 16
EXPECTED_OBS_DIM = 61
EXPECTED_ACTION_DIM = 8
TARGET_VX = 1.0
VELOCITY_FIELDS = [
    "step",
    "mean_vel_x",
    "mean_abs_vx_error",
    "p2_fault_active_mean",
    "mean_abs_yaw_error",
    "done_count",
    "no_nan_inf",
]


class A2StudentEvalError(ValueError):
    """Raised for invalid A2 student evaluation state."""


def build_parser(*, add_app_launcher_args: bool = False) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run guarded Isaac-side A2 student P2 evaluation.")
    parser.add_argument("--execute_eval", action="store_true", help="Launch Isaac Sim and run the guarded eval.")
    parser.add_argument("--policy_kind", default="single_step", choices=("single_step", "history"))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_envs", type=int, default=DEFAULT_NUM_ENVS)
    parser.add_argument("--num_steps", type=int, default=DEFAULT_NUM_STEPS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--fault_onset_mode", default=DEFAULT_ONSET_MODE, choices=(DEFAULT_ONSET_MODE,))
    parser.add_argument("--fault_onset_step_min", type=int, default=DEFAULT_ONSET_MIN)
    parser.add_argument("--fault_onset_step_max", type=int, default=DEFAULT_ONSET_MAX)
    parser.add_argument("--target_joint", default=DEFAULT_TARGET_JOINT, choices=TARGET_JOINTS)
    parser.add_argument("--history_len", type=int, default=DEFAULT_HISTORY_LEN)
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
    if pre_args.execute_eval:
        parser = build_parser(add_app_launcher_args=True)
        args, _ = parser.parse_known_args()
        return args
    args, _ = pre_parser.parse_known_args()
    return args


def validate_args(args: argparse.Namespace) -> None:
    if args.num_envs <= 0:
        raise A2StudentEvalError("--num_envs must be > 0.")
    if args.num_steps <= 0:
        raise A2StudentEvalError("--num_steps must be > 0.")
    if args.seed < 0:
        raise A2StudentEvalError("--seed must be non-negative.")
    if args.history_len < 1:
        raise A2StudentEvalError("--history_len must be >= 1.")
    if args.fault_onset_step_min < 0 or args.fault_onset_step_max < 0:
        raise A2StudentEvalError("fault onset bounds must be non-negative.")
    if args.fault_onset_step_min > args.fault_onset_step_max:
        raise A2StudentEvalError("--fault_onset_step_min must be <= --fault_onset_step_max.")
    if args.fault_onset_mode != DEFAULT_ONSET_MODE:
        raise A2StudentEvalError("Only random_uniform P2 onset is supported for this scaffold.")
    checkpoint = resolve_repo_path(args.checkpoint)
    if args.execute_eval and not checkpoint.is_file():
        raise A2StudentEvalError(f"checkpoint does not exist: {repo_relative(checkpoint)}")


def print_preview(args: argparse.Namespace) -> None:
    print("[T10-A2-P2 EVAL PREVIEW]")
    print("  execute_eval_required: True")
    print("  no_isaac_sim_launched: True")
    print("  no_training: True")
    print("  no_checkpoint_modification: True")
    print("  no_task_config_modification: True")
    print("  no_p2_wrapper_modification: True")
    print("  no_checkpoint_pointer_update: True")
    print("  not_paper_grade_final: True")
    print(f"  policy_kind: {args.policy_kind}")
    print(f"  task: {TASK}")
    print(f"  checkpoint: {args.checkpoint}")
    print(f"  output_dir: {repo_relative(args.output_dir)}")
    print(f"  num_envs: {args.num_envs}")
    print(f"  num_steps: {args.num_steps}")
    print(f"  seed: {args.seed}")
    print(f"  fault_profile: {FAULT_PROFILE}")
    print(f"  target_joint: {args.target_joint}")
    print(f"  semantics: {REQUESTED_SEMANTICS}")
    print("  fallback_allowed: False")
    print(f"  onset: {args.fault_onset_mode} [{args.fault_onset_step_min}, {args.fault_onset_step_max}]")
    print(f"  expected_observation_dim: {EXPECTED_OBS_DIM}")
    print(f"  expected_action_dim: {EXPECTED_ACTION_DIM}")
    if args.policy_kind == "history":
        print(f"  history_len: {args.history_len}")
        print("  history_reset_initialization: repeat_first_observation")


def set_torch_seed(seed: int) -> None:
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def torch_load_checkpoint(path: Path, *, map_location: str) -> dict[str, Any]:
    import torch

    try:
        checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=map_location)
    if not isinstance(checkpoint, dict):
        raise A2StudentEvalError(f"checkpoint must be a dict, got {type(checkpoint).__name__}.")
    return checkpoint


def assert_finite_tensor(name: str, tensor: Any) -> None:
    import torch

    if not torch.isfinite(tensor).all():
        raise A2StudentEvalError(f"{name} contains NaN or Inf.")


def assert_finite_value(name: str, value: Any) -> None:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, int):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise A2StudentEvalError(f"{name} is NaN or Inf: {value}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            assert_finite_value(f"{name}[{index}]", item)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            assert_finite_value(f"{name}.{key}", item)
        return
    raise A2StudentEvalError(f"{name} has unsupported metric type {type(value).__name__}.")


def fail_if_checkpoint_guardrail_violated(checkpoint: dict[str, Any], *, checkpoint_path: Path) -> None:
    metrics = checkpoint.get("metrics")
    if not isinstance(metrics, dict):
        raise A2StudentEvalError(f"checkpoint missing metrics metadata: {repo_relative(checkpoint_path)}")
    guardrails = checkpoint.get("guardrails") if isinstance(checkpoint.get("guardrails"), dict) else {}
    for key in ("teacher_obs_used_as_input", "true_fault_state_used", "health_token_used", "residual_head_used"):
        if bool(metrics.get(key, False)):
            raise A2StudentEvalError(f"checkpoint guardrail violated: metrics.{key}=true")
    if bool(guardrails.get("no_teacher_obs_input")) is False:
        raise A2StudentEvalError("checkpoint guardrail missing/false: no_teacher_obs_input")
    if bool(guardrails.get("no_true_fault_state")) is False:
        raise A2StudentEvalError("checkpoint guardrail missing/false: no_true_fault_state")
    if metrics.get("input_dim") not in (None, EXPECTED_OBS_DIM):
        raise A2StudentEvalError(f"checkpoint input_dim expected {EXPECTED_OBS_DIM}, got {metrics.get('input_dim')}")
    if metrics.get("action_dim") not in (None, EXPECTED_ACTION_DIM):
        raise A2StudentEvalError(f"checkpoint action_dim expected {EXPECTED_ACTION_DIM}, got {metrics.get('action_dim')}")


def load_student_policy(
    *,
    policy_kind: str,
    checkpoint_path: Path,
    history_len: int,
    device: Any,
) -> tuple[Any, dict[str, Any]]:
    import torch

    checkpoint = torch_load_checkpoint(checkpoint_path, map_location="cpu")
    fail_if_checkpoint_guardrail_violated(checkpoint, checkpoint_path=checkpoint_path)
    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(state_dict, dict):
        raise A2StudentEvalError("checkpoint missing model_state_dict.")

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    if policy_kind == "single_step":
        from trainers.train_t10_a2_student_distill import StudentMLP

        model = StudentMLP(EXPECTED_OBS_DIM, EXPECTED_ACTION_DIM)
    elif policy_kind == "history":
        from trainers.train_t10_a2_student_history_distill import build_history_student_mlp

        checkpoint_history_len = checkpoint.get("history_len") or checkpoint.get("metrics", {}).get("history_len")
        if int(checkpoint_history_len) != int(history_len):
            raise A2StudentEvalError(
                f"history checkpoint history_len expected {history_len}, got {checkpoint_history_len}."
            )
        model = build_history_student_mlp(history_len, EXPECTED_OBS_DIM, EXPECTED_ACTION_DIM)
    else:
        raise A2StudentEvalError(f"unsupported policy_kind: {policy_kind}")

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    metadata = {
        "checkpoint_model_class": checkpoint.get("model_class"),
        "checkpoint_history_len": checkpoint.get("history_len") or checkpoint.get("metrics", {}).get("history_len"),
        "checkpoint_metrics": checkpoint.get("metrics", {}),
    }
    with torch.inference_mode():
        if policy_kind == "single_step":
            dummy = torch.zeros((1, EXPECTED_OBS_DIM), dtype=torch.float32, device=device)
        else:
            dummy = torch.zeros((1, history_len, EXPECTED_OBS_DIM), dtype=torch.float32, device=device)
        dummy_action = model(dummy)
        if dummy_action.ndim != 2 or dummy_action.shape[-1] != EXPECTED_ACTION_DIM:
            raise A2StudentEvalError(f"model action dim expected {EXPECTED_ACTION_DIM}, got {tuple(dummy_action.shape)}.")
        assert_finite_tensor("dummy_action", dummy_action)
    return model, metadata


def select_policy_device(args: argparse.Namespace, env_device: Any) -> Any:
    import torch

    requested = getattr(args, "device", None)
    if requested is None:
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if str(requested).startswith("cuda") and not torch.cuda.is_available():
        print("[T10-A2-P2 WARNING] CUDA unavailable; using CPU for A2 policy inference.", flush=True)
        requested = "cpu"
    return torch.device(requested if requested is not None else env_device)


def safe_student_obs(obs: Any) -> Any:
    student_obs = policy_tensor(obs).detach().float()
    if student_obs.ndim != 2 or student_obs.shape[-1] != EXPECTED_OBS_DIM:
        raise A2StudentEvalError(f"student-safe observation dim expected {EXPECTED_OBS_DIM}, got {tuple(student_obs.shape)}.")
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
        raise A2StudentEvalError(f"done tensor length mismatch: expected {num_envs}, got {int(tensor.shape[0])}.")
    return tensor.to(dtype=torch.bool)


def maybe_mean(values: list[float]) -> float | None:
    return float(statistics.mean(values)) if values else None


def mean_row_value(rows: list[dict[str, Any]], key: str, *, start_step: int | None = None, end_step: int | None = None) -> float | None:
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


def write_json(path: Path, values: dict[str, Any]) -> None:
    assert_finite_value("summary", values)
    path.write_text(json.dumps(values, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], *, fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_readme(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# T10 A2 Student P2 Eval",
        "",
        "## Purpose",
        "",
        "Isaac-side evaluation for an offline-distilled A2 student under the P2 locked-joint fault.",
        "",
        "## Policy",
        "",
        f"- policy_kind: `{summary['policy_kind']}`",
        f"- checkpoint: `{summary['checkpoint_path']}`",
        f"- task: `{summary['task']}`",
        f"- observation_dim: `{summary['observation_dim']}`",
        f"- action_dim: `{summary['action_dim']}`",
    ]
    if summary.get("history_len") is not None:
        lines.extend(
            [
                f"- history_len: `{summary['history_len']}`",
                "- history reset initialization: `repeat_first_observation`",
            ]
        )
    lines.extend(
        [
            "",
            "## Fault",
            "",
            f"- fault_profile: `{summary['fault_profile']}`",
            f"- target_joint: `{summary['target_joint']}`",
            f"- semantics: `{summary['semantics']}`",
            f"- fallback_used: `{summary['fallback_used']}`",
            f"- p2_fault_became_active: `{summary['p2_fault_became_active']}`",
            f"- simulation_override_applied: `{summary['simulation_override_applied']}`",
            "",
            "## Scope",
            "",
            "- This is not paper-grade final evidence.",
            "- No training, residual correction, teacher observations, true fault state, health token, UQ, CBF, P3, or P4 are used.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def execute_rollout(args: argparse.Namespace) -> int:
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

    output_dir = resolve_repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    checkpoint_path = resolve_repo_path(args.checkpoint)

    env = None
    vec_env = None
    p2_wrapper = None
    rows: list[dict[str, Any]] = []
    no_nan_inf = True
    p2_fault_became_active = False
    simulation_override_applied = False
    fallback_used = False
    timeout_values: list[float] = []
    torso_height_failure_values: list[float] = []
    last_log_values: dict[str, Any] = {}
    velocity_source = None
    yaw_source = None
    model_metadata: dict[str, Any] = {}

    try:
        print("[T10-A2-P2 WARNING] Candidate-level evaluation only; not paper-grade final.", flush=True)
        print("[T10-A2-P2] config load start", flush=True)
        env_cfg = load_cfg_from_registry(TASK, "env_cfg_entry_point")
        agent_cfg = load_cfg_from_registry(TASK, "rsl_rl_cfg_entry_point")
        env_cfg.scene.num_envs = args.num_envs
        env_cfg.seed = args.seed
        if getattr(args, "device", None) is not None:
            env_cfg.sim.device = args.device
        env_cfg.log_dir = str(output_dir / "isaac_logs")
        set_torch_seed(args.seed)
        print("[T10-A2-P2] config load done", flush=True)

        print("[T10-A2-P2] gym.make start", flush=True)
        env = gym.make(TASK, cfg=env_cfg)
        print("[T10-A2-P2] gym.make done", flush=True)
        if isinstance(env.unwrapped, DirectMARLEnv):
            env = multi_agent_to_single_agent(env)

        print("[T10-A2-P2] P2 wrapper attach start", flush=True)
        p2_wrapper = P2JointLockActionMaskWrapper(
            env,
            target_joint=args.target_joint,
            fault_onset_step=DEFAULT_ONSET_STEP,
            fault_onset_mode=args.fault_onset_mode,
            fault_onset_step_min=args.fault_onset_step_min,
            fault_onset_step_max=args.fault_onset_step_max,
            expected_action_dim=EXPECTED_ACTION_DIM,
            requested_semantics=REQUESTED_SEMANTICS,
            allow_fallback=False,
            velocity_override=0.0,
            debug=True,
        )
        env = p2_wrapper
        if p2_wrapper.mapping.semantics != REQUESTED_SEMANTICS:
            raise A2StudentEvalError(f"P2 actual semantics must be {REQUESTED_SEMANTICS}.")
        if p2_wrapper.fallback_used:
            raise A2StudentEvalError("P2 fallback was used during wrapper attachment.")
        print("[T10-A2-P2] P2 wrapper attach done", flush=True)

        target_vx_info = maybe_set_target_vx(env, TARGET_VX)
        agent_cfg, _ = prepare_agent_cfg(agent_cfg)
        vec_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        if int(vec_env.num_actions) != EXPECTED_ACTION_DIM:
            raise A2StudentEvalError(f"action dim expected {EXPECTED_ACTION_DIM}, got {int(vec_env.num_actions)}.")

        tracking_num_envs = int(vec_env.num_envs)
        tracking_device = vec_env.unwrapped.device
        policy_device = select_policy_device(args, tracking_device)
        model, model_metadata = load_student_policy(
            policy_kind=args.policy_kind,
            checkpoint_path=checkpoint_path,
            history_len=args.history_len,
            device=policy_device,
        )

        obs = vec_env.get_observations()
        student_obs = safe_student_obs(obs)
        if int(student_obs.shape[0]) != tracking_num_envs:
            raise A2StudentEvalError(
                f"student_obs env count expected {tracking_num_envs}, got {int(student_obs.shape[0])}."
            )

        history_buffer = None
        if args.policy_kind == "history":
            history_buffer = student_obs.to(policy_device).unsqueeze(1).repeat(1, args.history_len, 1)
            assert_finite_tensor("history_buffer", history_buffer)

        first_done_step = torch.full((tracking_num_envs,), -1, dtype=torch.long, device=tracking_device)
        p2_action_term_name = p2_wrapper.mapping.action_term_name
        print("[T10-A2-P2] rollout start", flush=True)
        for step_index in range(args.num_steps):
            step_number = step_index + 1
            student_obs = safe_student_obs(obs)
            with torch.inference_mode():
                if args.policy_kind == "single_step":
                    model_input = student_obs.to(policy_device)
                else:
                    if history_buffer is None:
                        raise A2StudentEvalError("history buffer was not initialized.")
                    model_input = history_buffer
                assert_finite_tensor("model_input", model_input)
                actions = model(model_input).detach().float()
                if actions.ndim != 2 or actions.shape[-1] != EXPECTED_ACTION_DIM:
                    raise A2StudentEvalError(f"action dim expected {EXPECTED_ACTION_DIM}, got {tuple(actions.shape)}.")
                assert_finite_tensor("actions", actions)
                next_obs, rewards, dones, extras = vec_env.step(actions.to(tracking_device))

            reward_tensor = torch.as_tensor(rewards, device=tracking_device).detach().float()
            assert_finite_tensor("rewards", reward_tensor)
            done_mask = done_mask_tensor(dones, num_envs=tracking_num_envs, device=tracking_device)
            new_done = torch.logical_and(done_mask, first_done_step < 0)
            first_done_step[new_done] = step_number

            next_student_obs = safe_student_obs(next_obs)
            if args.policy_kind == "history":
                assert history_buffer is not None
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
                last_log_values = log_values
                timeout_value = first_matching_log_value(log_values, ("time", "out"))
                if timeout_value is not None:
                    timeout_values.append(timeout_value)
                torso_value = first_matching_log_value(log_values, ("torso", "height"))
                if torso_value is not None:
                    torso_height_failure_values.append(torso_value)

            fallback_value = scalar(log_values.get("P2/fallback_used"))
            if p2_wrapper.fallback_used or fallback_value > 0.0:
                raise A2StudentEvalError("P2 fallback was used during evaluation.")
            fallback_used = fallback_used or bool(p2_wrapper.fallback_used) or fallback_value > 0.0

            fault_mask = getattr(p2_wrapper, "last_fault_applied_mask", None)
            if fault_mask is None:
                fault_mask = torch.zeros(tracking_num_envs, dtype=torch.bool, device=tracking_device)
            else:
                fault_mask = torch.as_tensor(fault_mask, device=tracking_device).to(dtype=torch.bool).reshape(
                    tracking_num_envs
                )
            p2_fault_active_mean = float(fault_mask.float().mean().detach().cpu().item())
            if bool(fault_mask.any().item()):
                p2_fault_became_active = True
                if scalar(log_values.get("P2/simulation_override_applied")) <= 0.0:
                    raise A2StudentEvalError("P2 simulation override was not applied after onset.")
                simulation_override_applied = True

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
                print(f"[T10-A2-P2] velocity_source={velocity_source}", flush=True)
            if yaw_source is None and yaw_metrics["yaw_metric_source"] is not None:
                yaw_source = yaw_metrics["yaw_metric_source"]

            vx_error = velocity_x - target_vx
            for name, tensor in (
                ("velocity_x", velocity_x),
                ("target_vx", target_vx),
                ("vx_error", vx_error),
            ):
                assert_finite_tensor(name, tensor)
            mean_vel_x = float(velocity_x.mean().detach().cpu().item())
            mean_abs_vx_error = float(vx_error.abs().mean().detach().cpu().item())
            mean_abs_yaw_error = yaw_metrics["mean_abs_yaw_error"]
            row = {
                "step": step_number,
                "mean_vel_x": mean_vel_x,
                "mean_abs_vx_error": mean_abs_vx_error,
                "p2_fault_active_mean": p2_fault_active_mean,
                "mean_abs_yaw_error": mean_abs_yaw_error,
                "done_count": int(done_mask.sum().detach().cpu().item()),
                "no_nan_inf": True,
            }
            assert_finite_value("velocity_row", row)
            rows.append(row)
            obs = next_obs

            if step_index < 3 or step_number % 50 == 0 or step_number == args.num_steps:
                print(
                    f"[T10-A2-P2] step={step_number} mean_vel_x={mean_vel_x:.4f} "
                    f"mean_abs_vx_error={mean_abs_vx_error:.4f} p2_fault_active={p2_fault_active_mean:.4f}",
                    flush=True,
                )

        if not p2_fault_became_active:
            raise A2StudentEvalError("P2 fault never became active during evaluation.")
        if not simulation_override_applied:
            raise A2StudentEvalError("P2 simulation override was never observed after onset.")
        if not no_nan_inf:
            raise A2StudentEvalError("NaN or Inf was observed during evaluation.")

        pre_start = 1
        pre_end = args.fault_onset_step_min
        post_start = args.fault_onset_step_max
        post_end = args.num_steps + 1
        first_done_steps = [int(value) for value in first_done_step.detach().cpu().tolist()]
        timeout_rate = maybe_mean(timeout_values)
        torso_height_failure_rate = maybe_mean(torso_height_failure_values)
        summary = {
            "eval_scope": "t10_a2_student_p2_eval",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "policy_kind": args.policy_kind,
            "checkpoint_path": repo_relative(checkpoint_path),
            "task": TASK,
            "num_envs": args.num_envs,
            "num_steps": args.num_steps,
            "seed": args.seed,
            "device": str(policy_device),
            "observation_dim": EXPECTED_OBS_DIM,
            "action_dim": EXPECTED_ACTION_DIM,
            "history_len": args.history_len if args.policy_kind == "history" else None,
            "history_reset_initialization": "repeat_first_observation" if args.policy_kind == "history" else None,
            "fault_profile": FAULT_PROFILE,
            "fault_onset_mode": args.fault_onset_mode,
            "fault_onset_step_min": args.fault_onset_step_min,
            "fault_onset_step_max": args.fault_onset_step_max,
            "target_joint": args.target_joint,
            "semantics": REQUESTED_SEMANTICS,
            "fallback_allowed": False,
            "fallback_used": bool(fallback_used),
            "simulation_override_applied": bool(simulation_override_applied),
            "p2_fault_became_active": bool(p2_fault_became_active),
            "mean_vel_x_pre_fault": mean_row_value(rows, "mean_vel_x", start_step=pre_start, end_step=pre_end),
            "mean_vel_x_post_fault": mean_row_value(rows, "mean_vel_x", start_step=post_start, end_step=post_end),
            "mean_abs_vx_error_pre_fault": mean_row_value(
                rows,
                "mean_abs_vx_error",
                start_step=pre_start,
                end_step=pre_end,
            ),
            "mean_abs_vx_error_post_fault": mean_row_value(
                rows,
                "mean_abs_vx_error",
                start_step=post_start,
                end_step=post_end,
            ),
            "mean_abs_yaw_error": mean_row_value(rows, "mean_abs_yaw_error"),
            "timeout_rate": timeout_rate,
            "torso_height_failure_rate": torso_height_failure_rate,
            "no_nan_inf": bool(no_nan_inf),
            "not_paper_grade_final": True,
            "target_vx": TARGET_VX,
            "target_vx_info": target_vx_info,
            "velocity_metric_source": velocity_source,
            "yaw_metric_source": yaw_source,
            "first_done_step_by_env": first_done_steps,
            "done_count": sum(1 for value in first_done_steps if value >= 0),
            "P2/fault_applied": scalar(last_log_values.get("P2/fault_applied")),
            "P2/simulation_override_applied": scalar(last_log_values.get("P2/simulation_override_applied")),
            "P2/fallback_used": scalar(last_log_values.get("P2/fallback_used")),
            "P2/onset_step_mean": scalar(last_log_values.get("P2/onset_step_mean")),
            "P2/onset_step_min": scalar(last_log_values.get("P2/onset_step_min")),
            "P2/onset_step_max": scalar(last_log_values.get("P2/onset_step_max")),
            "P2/per_env_onset_randomization": scalar(last_log_values.get("P2/per_env_onset_randomization")),
            "model_metadata": model_metadata,
            "guardrails": {
                "no_training": True,
                "no_checkpoint_modification": True,
                "no_task_config_modification": True,
                "no_p2_wrapper_modification": True,
                "no_checkpoint_pointer_update": True,
                "no_teacher_obs": True,
                "no_true_fault_state": True,
                "no_health_token": True,
                "no_residual_correction": True,
            },
        }
        if summary["fallback_used"]:
            raise A2StudentEvalError("summary indicates fallback was used.")
        if not summary["simulation_override_applied"]:
            raise A2StudentEvalError("summary indicates simulation override was not applied.")
        if not summary["p2_fault_became_active"]:
            raise A2StudentEvalError("summary indicates P2 fault never became active.")
        if not summary["no_nan_inf"]:
            raise A2StudentEvalError("summary indicates NaN/Inf was observed.")

        velocity_csv = output_dir / "velocity_timeseries.csv"
        summary_json = output_dir / "summary.json"
        readme_path = output_dir / "README.md"
        write_csv(velocity_csv, rows, fieldnames=VELOCITY_FIELDS)
        summary["velocity_timeseries_csv"] = repo_relative(velocity_csv)
        write_json(summary_json, summary)
        write_readme(readme_path, summary)
        for required_path in (summary_json, velocity_csv, output_dir / "command.txt", readme_path):
            if not required_path.is_file():
                raise A2StudentEvalError(f"required output was not written: {repo_relative(required_path)}")
        print(f"[T10-A2-P2] summary: {repo_relative(summary_json)}", flush=True)
        print(f"[T10-A2-P2] velocity_timeseries: {repo_relative(velocity_csv)}", flush=True)
        print(f"[T10-A2-P2] README: {repo_relative(readme_path)}", flush=True)
        return 0
    finally:
        if vec_env is not None:
            vec_env.close()
        elif env is not None:
            env.close()


def execute_eval(args: argparse.Namespace) -> int:
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    try:
        return execute_rollout(args)
    finally:
        simulation_app.close()


def main() -> int:
    try:
        args = parse_args()
        validate_args(args)
        if not args.execute_eval:
            print_preview(args)
            return 0
        return execute_eval(args)
    except A2StudentEvalError as exc:
        print(f"[T10-A2-P2 ERROR] {exc}", file=sys.stderr, flush=True)
        return 2
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
