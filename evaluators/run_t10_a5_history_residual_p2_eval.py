#!/usr/bin/env python3
"""Isaac-side A5 history residual P2 evaluation scaffold.

This guarded evaluator composes a frozen A2-history base policy with an A5
teacher-gap residual head on deployment-safe student observations only:

    final_action = a2_history(history) + alpha * a5_residual(history, a2_action)

Runtime execution requires ``--execute_eval``. This is candidate-level smoke
evaluation only; it does not train, use teacher observations, use true fault
state, use A0 actions, run A7, or modify task/P2 code.
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
TARGET_JOINT = "front_left_foot"
REQUESTED_SEMANTICS = "simulation_joint_state_override_lock"
DEFAULT_ONSET_MODE = "random_uniform"
DEFAULT_ONSET_STEP = 50
DEFAULT_ONSET_MIN = 30
DEFAULT_ONSET_MAX = 150
DEFAULT_NUM_ENVS = 16
DEFAULT_NUM_STEPS = 200
DEFAULT_SEED = 0
DEFAULT_HISTORY_LEN = 16
DEFAULT_ALPHA = 1.0
EXPECTED_OBS_DIM = 61
EXPECTED_ACTION_DIM = 8
EXPECTED_RESIDUAL_DIM = 8
EXPECTED_RESIDUAL_INPUT_MODE = "history_plus_base_action"
TARGET_VX = 1.0
VELOCITY_FIELDS = [
    "step",
    "mean_vel_x",
    "mean_abs_vx_error",
    "p2_fault_active_mean",
    "residual_action_mean_norm",
    "final_action_mean_norm",
    "base_action_mean_norm",
    "residual_action_max_norm",
    "mean_abs_yaw_error",
    "done_count",
    "no_nan_inf",
]


class A5ResidualEvalError(ValueError):
    """Raised for invalid A5 residual evaluation state."""


def build_parser(*, add_app_launcher_args: bool = False) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run guarded Isaac-side A5 history residual P2 evaluation.")
    parser.add_argument("--execute_eval", action="store_true", help="Launch Isaac Sim and run the guarded eval.")
    parser.add_argument("--a2_checkpoint", required=True)
    parser.add_argument("--a5_checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_envs", type=int, default=DEFAULT_NUM_ENVS)
    parser.add_argument("--num_steps", type=int, default=DEFAULT_NUM_STEPS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--fault_onset_mode", default=DEFAULT_ONSET_MODE, choices=(DEFAULT_ONSET_MODE,))
    parser.add_argument("--fault_onset_step_min", type=int, default=DEFAULT_ONSET_MIN)
    parser.add_argument("--fault_onset_step_max", type=int, default=DEFAULT_ONSET_MAX)
    parser.add_argument("--history_len", type=int, default=DEFAULT_HISTORY_LEN)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    if not add_app_launcher_args:
        parser.add_argument("--headless", action="store_true")
        parser.add_argument("--device", default=None, help="Policy/env device. Default is cuda if available else cpu.")
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
        raise A5ResidualEvalError("--num_envs must be > 0.")
    if args.num_steps <= 0:
        raise A5ResidualEvalError("--num_steps must be > 0.")
    if args.seed < 0:
        raise A5ResidualEvalError("--seed must be non-negative.")
    if args.history_len < 1:
        raise A5ResidualEvalError("--history_len must be >= 1.")
    if not math.isfinite(float(args.alpha)):
        raise A5ResidualEvalError("--alpha must be finite.")
    if args.fault_onset_step_min < 0 or args.fault_onset_step_max < 0:
        raise A5ResidualEvalError("fault onset bounds must be non-negative.")
    if args.fault_onset_step_min > args.fault_onset_step_max:
        raise A5ResidualEvalError("--fault_onset_step_min must be <= --fault_onset_step_max.")
    if args.fault_onset_mode != DEFAULT_ONSET_MODE:
        raise A5ResidualEvalError("Only random_uniform P2 onset is supported for this scaffold.")
    if args.execute_eval:
        for label, checkpoint in (("A2", args.a2_checkpoint), ("A5", args.a5_checkpoint)):
            checkpoint_path = resolve_repo_path(checkpoint)
            if not checkpoint_path.is_file():
                raise A5ResidualEvalError(f"{label} checkpoint does not exist: {repo_relative(checkpoint_path)}")


def print_preview(args: argparse.Namespace) -> None:
    print("[T10-A5-HISTORY-RESIDUAL-P2 EVAL PREVIEW]")
    print("  execute_eval_required: True")
    print("  no_isaac_sim_launched: True")
    print("  no_training: True")
    print("  no_checkpoint_modification: True")
    print("  no_task_config_modification: True")
    print("  no_p2_wrapper_modification: True")
    print("  no_checkpoint_pointer_update: True")
    print("  not_paper_grade_final: True")
    print("  policy_kind: a5_history_residual")
    print(f"  task: {TASK}")
    print(f"  a2_checkpoint: {args.a2_checkpoint}")
    print(f"  a5_checkpoint: {args.a5_checkpoint}")
    print(f"  output_dir: {repo_relative(args.output_dir)}")
    print(f"  num_envs: {args.num_envs}")
    print(f"  num_steps: {args.num_steps}")
    print(f"  seed: {args.seed}")
    print(f"  history_len: {args.history_len}")
    print(f"  alpha: {args.alpha}")
    print(f"  residual_input_mode: {EXPECTED_RESIDUAL_INPUT_MODE}")
    print(f"  fault_profile: {FAULT_PROFILE}")
    print(f"  target_joint: {TARGET_JOINT}")
    print(f"  semantics: {REQUESTED_SEMANTICS}")
    print("  fallback_allowed: False")
    print(f"  onset: {args.fault_onset_mode} [{args.fault_onset_step_min}, {args.fault_onset_step_max}]")
    print(f"  expected_observation_dim: {EXPECTED_OBS_DIM}")
    print(f"  expected_action_dim: {EXPECTED_ACTION_DIM}")
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
        raise A5ResidualEvalError(f"checkpoint must be a dict, got {type(checkpoint).__name__}.")
    return checkpoint


def assert_finite_tensor(name: str, tensor: Any) -> None:
    import torch

    if not torch.isfinite(tensor).all():
        raise A5ResidualEvalError(f"{name} contains NaN or Inf.")


def assert_finite_value(name: str, value: Any) -> None:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, int):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise A5ResidualEvalError(f"{name} is NaN or Inf: {value}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            assert_finite_value(f"{name}[{index}]", item)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            assert_finite_value(f"{name}.{key}", item)
        return
    raise A5ResidualEvalError(f"{name} has unsupported metric type {type(value).__name__}.")


def flag_true(container: dict[str, Any], key: str) -> bool:
    return bool(container.get(key, False))


def assert_dim_metadata(
    *,
    checkpoint: dict[str, Any],
    metrics: dict[str, Any],
    checkpoint_name: str,
    expected_history_len: int,
    expected_input_dim: int = EXPECTED_OBS_DIM,
    expected_action_dim: int = EXPECTED_ACTION_DIM,
    expected_residual_dim: int | None = None,
) -> int:
    checkpoint_history_len = checkpoint.get("history_len", metrics.get("history_len"))
    if checkpoint_history_len is None:
        raise A5ResidualEvalError(f"{checkpoint_name} checkpoint missing history_len metadata.")
    if int(checkpoint_history_len) != int(expected_history_len):
        raise A5ResidualEvalError(
            f"{checkpoint_name} checkpoint history_len expected {expected_history_len}, got {checkpoint_history_len}."
        )
    for source_name, source in (("checkpoint", checkpoint), ("metrics", metrics)):
        input_dim = source.get("input_dim")
        action_dim = source.get("action_dim")
        residual_dim = source.get("residual_dim")
        if input_dim is not None and int(input_dim) != expected_input_dim:
            raise A5ResidualEvalError(
                f"{checkpoint_name} {source_name}.input_dim expected {expected_input_dim}, got {input_dim}."
            )
        if action_dim is not None and int(action_dim) != expected_action_dim:
            raise A5ResidualEvalError(
                f"{checkpoint_name} {source_name}.action_dim expected {expected_action_dim}, got {action_dim}."
            )
        if expected_residual_dim is not None and residual_dim is not None and int(residual_dim) != expected_residual_dim:
            raise A5ResidualEvalError(
                f"{checkpoint_name} {source_name}.residual_dim expected {expected_residual_dim}, got {residual_dim}."
            )
    return int(checkpoint_history_len)


def fail_if_common_guardrail_violated(
    checkpoint: dict[str, Any],
    *,
    checkpoint_path: Path,
    checkpoint_name: str,
    require_no_residual_head: bool,
    require_residual_head: bool,
    require_no_a0_action_guardrail: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    metrics = checkpoint.get("metrics")
    if not isinstance(metrics, dict):
        raise A5ResidualEvalError(f"{checkpoint_name} checkpoint missing metrics metadata: {repo_relative(checkpoint_path)}")
    guardrails = checkpoint.get("guardrails") if isinstance(checkpoint.get("guardrails"), dict) else {}
    if metrics.get("no_nan_inf") is False:
        raise A5ResidualEvalError(f"{checkpoint_name} checkpoint metrics.no_nan_inf is false.")
    for key in ("teacher_obs_used_as_input", "true_fault_state_used", "health_token_used", "a0_action_used"):
        if flag_true(metrics, key):
            raise A5ResidualEvalError(f"{checkpoint_name} checkpoint guardrail violated: metrics.{key}=true")
    if flag_true(guardrails, "uses_teacher_obs") or flag_true(guardrails, "uses_true_fault_state"):
        raise A5ResidualEvalError(f"{checkpoint_name} checkpoint guardrail indicates unsafe observations.")
    required_true = {
        "no_teacher_obs_input": "teacher_obs",
        "no_true_fault_state": "true_fault_state",
        "no_health_token": "health_token",
    }
    for key, label in required_true.items():
        if bool(guardrails.get(key)) is not True:
            raise A5ResidualEvalError(f"{checkpoint_name} checkpoint missing/false guardrail: {key} ({label}).")
    if require_no_a0_action_guardrail and bool(guardrails.get("no_a0_action")) is not True:
        raise A5ResidualEvalError(f"{checkpoint_name} checkpoint missing/false guardrail: no_a0_action.")
    if require_no_residual_head:
        if flag_true(metrics, "residual_head_used"):
            raise A5ResidualEvalError(f"{checkpoint_name} checkpoint guardrail violated: residual_head_used=true")
        if bool(guardrails.get("no_residual_head")) is not True:
            raise A5ResidualEvalError(f"{checkpoint_name} checkpoint missing/false guardrail: no_residual_head.")
    if require_residual_head:
        if bool(metrics.get("residual_head_used")) is not True:
            raise A5ResidualEvalError(f"{checkpoint_name} checkpoint metrics.residual_head_used must be true.")
        if bool(guardrails.get("residual_head")) is not True:
            raise A5ResidualEvalError(f"{checkpoint_name} checkpoint missing/false guardrail: residual_head.")
    return metrics, guardrails


def add_trainers_to_path() -> None:
    trainers_dir = REPO_ROOT / "trainers"
    if str(trainers_dir) not in sys.path:
        sys.path.insert(0, str(trainers_dir))


def load_a2_history_policy(
    *,
    checkpoint_path: Path,
    history_len: int,
    device: Any,
) -> tuple[Any, dict[str, Any]]:
    import torch

    add_trainers_to_path()
    from train_t10_a2_student_history_distill import build_history_student_mlp

    checkpoint = torch_load_checkpoint(checkpoint_path, map_location="cpu")
    metrics, guardrails = fail_if_common_guardrail_violated(
        checkpoint,
        checkpoint_path=checkpoint_path,
        checkpoint_name="A2",
        require_no_residual_head=True,
        require_residual_head=False,
        require_no_a0_action_guardrail=False,
    )
    checkpoint_history_len = assert_dim_metadata(
        checkpoint=checkpoint,
        metrics=metrics,
        checkpoint_name="A2",
        expected_history_len=history_len,
    )
    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(state_dict, dict):
        raise A5ResidualEvalError("A2 checkpoint missing model_state_dict.")

    model = build_history_student_mlp(history_len, EXPECTED_OBS_DIM, EXPECTED_ACTION_DIM)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    with torch.inference_mode():
        dummy = torch.zeros((1, history_len, EXPECTED_OBS_DIM), dtype=torch.float32, device=device)
        dummy_action = model(dummy)
        if dummy_action.ndim != 2 or dummy_action.shape[-1] != EXPECTED_ACTION_DIM:
            raise A5ResidualEvalError(f"A2 action dim expected {EXPECTED_ACTION_DIM}, got {tuple(dummy_action.shape)}.")
        assert_finite_tensor("a2_dummy_action", dummy_action)

    metadata = {
        "a2_checkpoint_model_class": checkpoint.get("model_class"),
        "a2_checkpoint_history_len": checkpoint_history_len,
        "a2_checkpoint_guardrails_ok": True,
        "a2_checkpoint_metrics": metrics,
        "a2_checkpoint_guardrails": guardrails,
    }
    return model, metadata


def load_a5_residual_policy(
    *,
    checkpoint_path: Path,
    history_len: int,
    device: Any,
) -> tuple[Any, dict[str, Any]]:
    import torch

    add_trainers_to_path()
    from train_t10_a5_history_residual_distill import build_residual_mlp

    checkpoint = torch_load_checkpoint(checkpoint_path, map_location="cpu")
    metrics, guardrails = fail_if_common_guardrail_violated(
        checkpoint,
        checkpoint_path=checkpoint_path,
        checkpoint_name="A5",
        require_no_residual_head=False,
        require_residual_head=True,
        require_no_a0_action_guardrail=True,
    )
    checkpoint_history_len = assert_dim_metadata(
        checkpoint=checkpoint,
        metrics=metrics,
        checkpoint_name="A5",
        expected_history_len=history_len,
        expected_residual_dim=EXPECTED_RESIDUAL_DIM,
    )
    residual_input_mode = checkpoint.get("residual_input_mode", metrics.get("residual_input_mode"))
    if residual_input_mode != EXPECTED_RESIDUAL_INPUT_MODE:
        raise A5ResidualEvalError(
            f"A5 residual_input_mode expected {EXPECTED_RESIDUAL_INPUT_MODE!r}, got {residual_input_mode!r}."
        )
    if bool(metrics.get("a2_checkpoint_guardrails_ok")) is not True:
        raise A5ResidualEvalError("A5 checkpoint metrics.a2_checkpoint_guardrails_ok must be true.")
    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(state_dict, dict):
        raise A5ResidualEvalError("A5 checkpoint missing model_state_dict.")

    model = build_residual_mlp(
        history_len,
        EXPECTED_OBS_DIM,
        EXPECTED_ACTION_DIM,
        residual_input_mode=EXPECTED_RESIDUAL_INPUT_MODE,
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    with torch.inference_mode():
        dummy_history = torch.zeros((1, history_len, EXPECTED_OBS_DIM), dtype=torch.float32, device=device)
        dummy_base = torch.zeros((1, EXPECTED_ACTION_DIM), dtype=torch.float32, device=device)
        dummy_residual = model(dummy_history, dummy_base)
        if dummy_residual.ndim != 2 or dummy_residual.shape[-1] != EXPECTED_RESIDUAL_DIM:
            raise A5ResidualEvalError(
                f"A5 residual dim expected {EXPECTED_RESIDUAL_DIM}, got {tuple(dummy_residual.shape)}."
            )
        assert_finite_tensor("a5_dummy_residual", dummy_residual)

    metadata = {
        "a5_checkpoint_model_class": checkpoint.get("model_class"),
        "a5_checkpoint_history_len": checkpoint_history_len,
        "a5_checkpoint_guardrails_ok": True,
        "a5_checkpoint_metrics": metrics,
        "a5_checkpoint_guardrails": guardrails,
        "residual_input_mode": residual_input_mode,
    }
    return model, metadata


def select_policy_device(args: argparse.Namespace, env_device: Any) -> Any:
    import torch

    requested = getattr(args, "device", None)
    if requested is None:
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if str(requested).startswith("cuda") and not torch.cuda.is_available():
        print("[T10-A5-P2 WARNING] CUDA unavailable; using CPU for A2/A5 policy inference.", flush=True)
        requested = "cpu"
    return torch.device(requested if requested is not None else env_device)


def safe_student_obs(obs: Any) -> Any:
    student_obs = policy_tensor(obs).detach().float()
    if student_obs.ndim != 2 or student_obs.shape[-1] != EXPECTED_OBS_DIM:
        raise A5ResidualEvalError(
            f"student-safe observation dim expected {EXPECTED_OBS_DIM}, got {tuple(student_obs.shape)}."
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
        raise A5ResidualEvalError(f"done tensor length mismatch: expected {num_envs}, got {int(tensor.shape[0])}.")
    return tensor.to(dtype=torch.bool)


def assert_action_shape(name: str, tensor: Any, *, num_envs: int) -> None:
    if tensor.ndim != 2 or int(tensor.shape[0]) != num_envs or int(tensor.shape[1]) != EXPECTED_ACTION_DIM:
        raise A5ResidualEvalError(
            f"{name} expected shape [{num_envs}, {EXPECTED_ACTION_DIM}], got {tuple(tensor.shape)}."
        )


def maybe_mean(values: list[float]) -> float | None:
    return float(statistics.mean(values)) if values else None


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
        "# T10 A5 History Residual P2 Eval",
        "",
        "## Purpose",
        "",
        "Isaac-side smoke evaluation for the offline-distilled A5 history residual policy under the P2 locked-joint fault.",
        "",
        "## Policy",
        "",
        f"- policy_kind: `{summary['policy_kind']}`",
        f"- A2 checkpoint: `{summary['a2_checkpoint']}`",
        f"- A5 checkpoint: `{summary['a5_checkpoint']}`",
        f"- task: `{summary['task']}`",
        f"- observation_dim: `{summary['observation_dim']}`",
        f"- action_dim: `{summary['action_dim']}`",
        f"- history_len: `{summary['history_len']}`",
        f"- alpha: `{summary['alpha']}`",
        f"- residual_input_mode: `{summary['residual_input_mode']}`",
        "- composition: `final_action = a2_action + alpha * residual_action`",
        "- history reset initialization: `repeat_first_observation`",
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
        "- No training, A0 action, teacher observations, true fault state, health token, UQ, CBF, P3, P4, or A7 are used.",
    ]
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
    a2_checkpoint_path = resolve_repo_path(args.a2_checkpoint)
    a5_checkpoint_path = resolve_repo_path(args.a5_checkpoint)

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
    residual_mean_norm_values: list[float] = []
    residual_max_norm_values: list[float] = []
    base_mean_norm_values: list[float] = []
    final_mean_norm_values: list[float] = []
    last_log_values: dict[str, Any] = {}
    velocity_source = None
    yaw_source = None
    policy_metadata: dict[str, Any] = {}

    try:
        print("[T10-A5-P2 WARNING] Candidate-level evaluation only; not paper-grade final.", flush=True)
        print("[T10-A5-P2] config load start", flush=True)
        env_cfg = load_cfg_from_registry(TASK, "env_cfg_entry_point")
        agent_cfg = load_cfg_from_registry(TASK, "rsl_rl_cfg_entry_point")
        env_cfg.scene.num_envs = args.num_envs
        env_cfg.seed = args.seed
        if getattr(args, "device", None) is not None:
            env_cfg.sim.device = args.device
        env_cfg.log_dir = str(output_dir / "isaac_logs")
        set_torch_seed(args.seed)
        print("[T10-A5-P2] config load done", flush=True)

        print("[T10-A5-P2] gym.make start", flush=True)
        env = gym.make(TASK, cfg=env_cfg)
        print("[T10-A5-P2] gym.make done", flush=True)
        if isinstance(env.unwrapped, DirectMARLEnv):
            env = multi_agent_to_single_agent(env)

        print("[T10-A5-P2] P2 wrapper attach start", flush=True)
        p2_wrapper = P2JointLockActionMaskWrapper(
            env,
            target_joint=TARGET_JOINT,
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
            raise A5ResidualEvalError(f"P2 actual semantics must be {REQUESTED_SEMANTICS}.")
        if p2_wrapper.fallback_used:
            raise A5ResidualEvalError("P2 fallback was used during wrapper attachment.")
        print("[T10-A5-P2] P2 wrapper attach done", flush=True)

        target_vx_info = maybe_set_target_vx(env, TARGET_VX)
        agent_cfg, _ = prepare_agent_cfg(agent_cfg)
        vec_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        if int(vec_env.num_actions) != EXPECTED_ACTION_DIM:
            raise A5ResidualEvalError(f"action dim expected {EXPECTED_ACTION_DIM}, got {int(vec_env.num_actions)}.")

        tracking_num_envs = int(vec_env.num_envs)
        tracking_device = vec_env.unwrapped.device
        policy_device = select_policy_device(args, tracking_device)
        a2_model, a2_metadata = load_a2_history_policy(
            checkpoint_path=a2_checkpoint_path,
            history_len=args.history_len,
            device=policy_device,
        )
        a5_model, a5_metadata = load_a5_residual_policy(
            checkpoint_path=a5_checkpoint_path,
            history_len=args.history_len,
            device=policy_device,
        )
        policy_metadata = {
            **a2_metadata,
            **a5_metadata,
        }

        obs = vec_env.get_observations()
        student_obs = safe_student_obs(obs)
        if int(student_obs.shape[0]) != tracking_num_envs:
            raise A5ResidualEvalError(
                f"student_obs env count expected {tracking_num_envs}, got {int(student_obs.shape[0])}."
            )

        history_buffer = student_obs.to(policy_device).unsqueeze(1).repeat(1, args.history_len, 1)
        assert_finite_tensor("history_buffer", history_buffer)

        first_done_step = torch.full((tracking_num_envs,), -1, dtype=torch.long, device=tracking_device)
        p2_action_term_name = p2_wrapper.mapping.action_term_name
        print("[T10-A5-P2] rollout start", flush=True)
        for step_index in range(args.num_steps):
            step_number = step_index + 1
            student_obs = safe_student_obs(obs)
            with torch.inference_mode():
                model_input = history_buffer
                assert_finite_tensor("model_input", model_input)
                a2_action = a2_model(model_input).detach().float()
                assert_action_shape("a2_action", a2_action, num_envs=tracking_num_envs)
                assert_finite_tensor("a2_action", a2_action)
                residual_action = a5_model(model_input, a2_action).detach().float()
                assert_action_shape("residual_action", residual_action, num_envs=tracking_num_envs)
                assert_finite_tensor("residual_action", residual_action)
                final_action = a2_action + float(args.alpha) * residual_action
                assert_action_shape("final_action", final_action, num_envs=tracking_num_envs)
                assert_finite_tensor("final_action", final_action)
                next_obs, rewards, dones, extras = vec_env.step(final_action.to(tracking_device))

            reward_tensor = torch.as_tensor(rewards, device=tracking_device).detach().float()
            assert_finite_tensor("rewards", reward_tensor)
            done_mask = done_mask_tensor(dones, num_envs=tracking_num_envs, device=tracking_device)
            new_done = torch.logical_and(done_mask, first_done_step < 0)
            first_done_step[new_done] = step_number

            next_student_obs = safe_student_obs(next_obs)
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

            base_norm = a2_action.norm(dim=1)
            residual_norm = residual_action.norm(dim=1)
            final_norm = final_action.norm(dim=1)
            for name, tensor in (
                ("base_action_norm", base_norm),
                ("residual_action_norm", residual_norm),
                ("final_action_norm", final_norm),
            ):
                assert_finite_tensor(name, tensor)
            base_action_mean_norm = float(base_norm.mean().detach().cpu().item())
            residual_action_mean_norm = float(residual_norm.mean().detach().cpu().item())
            residual_action_max_norm = float(residual_norm.max().detach().cpu().item())
            final_action_mean_norm = float(final_norm.mean().detach().cpu().item())
            base_mean_norm_values.append(base_action_mean_norm)
            residual_mean_norm_values.append(residual_action_mean_norm)
            residual_max_norm_values.append(residual_action_max_norm)
            final_mean_norm_values.append(final_action_mean_norm)

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
                raise A5ResidualEvalError("P2 fallback was used during evaluation.")
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
                    raise A5ResidualEvalError("P2 simulation override was not applied after onset.")
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
                print(f"[T10-A5-P2] velocity_source={velocity_source}", flush=True)
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
                "residual_action_mean_norm": residual_action_mean_norm,
                "final_action_mean_norm": final_action_mean_norm,
                "base_action_mean_norm": base_action_mean_norm,
                "residual_action_max_norm": residual_action_max_norm,
                "mean_abs_yaw_error": mean_abs_yaw_error,
                "done_count": int(done_mask.sum().detach().cpu().item()),
                "no_nan_inf": True,
            }
            assert_finite_value("velocity_row", row)
            rows.append(row)
            obs = next_obs

            if step_index < 3 or step_number % 50 == 0 or step_number == args.num_steps:
                print(
                    f"[T10-A5-P2] step={step_number} mean_vel_x={mean_vel_x:.4f} "
                    f"mean_abs_vx_error={mean_abs_vx_error:.4f} "
                    f"residual_norm={residual_action_mean_norm:.4f} p2_fault_active={p2_fault_active_mean:.4f}",
                    flush=True,
                )

        if not p2_fault_became_active:
            raise A5ResidualEvalError("P2 fault never became active during evaluation.")
        if not simulation_override_applied:
            raise A5ResidualEvalError("P2 simulation override was never observed after onset.")
        if not no_nan_inf:
            raise A5ResidualEvalError("NaN or Inf was observed during evaluation.")

        pre_start = 1
        pre_end = args.fault_onset_step_min
        post_start = args.fault_onset_step_max
        post_end = args.num_steps + 1
        first_done_steps = [int(value) for value in first_done_step.detach().cpu().tolist()]
        timeout_rate = maybe_mean(timeout_values)
        torso_height_failure_rate = maybe_mean(torso_height_failure_values)
        summary = {
            "eval_scope": "t10_a5_history_residual_p2_eval",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "policy_kind": "a5_history_residual",
            "a2_checkpoint": repo_relative(a2_checkpoint_path),
            "a5_checkpoint": repo_relative(a5_checkpoint_path),
            "task": TASK,
            "num_envs": args.num_envs,
            "num_steps": args.num_steps,
            "seed": args.seed,
            "device": str(policy_device),
            "observation_dim": EXPECTED_OBS_DIM,
            "action_dim": EXPECTED_ACTION_DIM,
            "history_len": args.history_len,
            "alpha": float(args.alpha),
            "residual_input_mode": EXPECTED_RESIDUAL_INPUT_MODE,
            "action_composition": "final_action = a2_action + alpha * residual_action",
            "history_reset_initialization": "repeat_first_observation",
            "fault_profile": FAULT_PROFILE,
            "fault_onset_mode": args.fault_onset_mode,
            "fault_onset_step_min": args.fault_onset_step_min,
            "fault_onset_step_max": args.fault_onset_step_max,
            "target_joint": TARGET_JOINT,
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
            "residual_action_mean_norm": maybe_mean(residual_mean_norm_values),
            "residual_action_max_norm": max(residual_max_norm_values) if residual_max_norm_values else None,
            "base_action_mean_norm": maybe_mean(base_mean_norm_values),
            "final_action_mean_norm": maybe_mean(final_mean_norm_values),
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
            "policy_metadata": policy_metadata,
            "guardrails": {
                "no_training": True,
                "no_checkpoint_modification": True,
                "no_task_config_modification": True,
                "no_p2_wrapper_modification": True,
                "no_checkpoint_pointer_update": True,
                "no_teacher_obs": True,
                "no_true_fault_state": True,
                "no_health_token": True,
                "no_a0_action": True,
                "no_a7": True,
                "residual_learning_active": True,
            },
        }
        if summary["observation_dim"] != EXPECTED_OBS_DIM:
            raise A5ResidualEvalError("summary observation_dim mismatch.")
        if summary["action_dim"] != EXPECTED_ACTION_DIM:
            raise A5ResidualEvalError("summary action_dim mismatch.")
        if summary["fallback_used"]:
            raise A5ResidualEvalError("summary indicates fallback was used.")
        if not summary["simulation_override_applied"]:
            raise A5ResidualEvalError("summary indicates simulation override was not applied.")
        if not summary["p2_fault_became_active"]:
            raise A5ResidualEvalError("summary indicates P2 fault never became active.")
        if not summary["no_nan_inf"]:
            raise A5ResidualEvalError("summary indicates NaN/Inf was observed.")

        velocity_csv = output_dir / "velocity_timeseries.csv"
        summary_json = output_dir / "summary.json"
        readme_path = output_dir / "README.md"
        write_csv(velocity_csv, rows, fieldnames=VELOCITY_FIELDS)
        summary["velocity_timeseries_csv"] = repo_relative(velocity_csv)
        write_json(summary_json, summary)
        write_readme(readme_path, summary)
        for required_path in (summary_json, velocity_csv, output_dir / "command.txt", readme_path):
            if not required_path.is_file():
                raise A5ResidualEvalError(f"required output was not written: {repo_relative(required_path)}")
        print(f"[T10-A5-P2] summary: {repo_relative(summary_json)}", flush=True)
        print(f"[T10-A5-P2] velocity_timeseries: {repo_relative(velocity_csv)}", flush=True)
        print(f"[T10-A5-P2] README: {repo_relative(readme_path)}", flush=True)
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
    except A5ResidualEvalError as exc:
        print(f"[T10-A5-P2 ERROR] {exc}", file=sys.stderr, flush=True)
        return 2
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
