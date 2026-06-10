#!/usr/bin/env python3
"""T09-R2j quick P2 checkpoint sanity comparison evaluator.

This is a guarded demo/sanity runner only. It does not train, freeze
checkpoints, update checkpoint pointers, or write paper-grade observed results.
Runtime execution requires ``--execute_demo``.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
ALLOWED_FAULT_PROFILES = ("F0_none", "P2_locked_joint")
ALLOWED_TASKS = ("Isaac-Ant-v0", "Isaac-Ant-Teacher-v0")
DEFAULT_A0_CHECKPOINT = (
    "logs/rsl_rl/healthy_baseline__rlm1_stripped__canonical/"
    "2026-06-05_21-42-13_a0_canonical__seed0/model_1999.pt"
)
DEFAULT_A1F_FIXED_CHECKPOINT = (
    "logs/rsl_rl/teacher_p2__rlm1_stripped__canonical/"
    "2026-06-09_14-58-55_a1f_p2_teacher_canonical__seed0/model_1999.pt"
)
DEFAULT_A1F_RANDOM_CHECKPOINT = (
    "logs/rsl_rl/teacher_p2_random_onset__rlm1_stripped__canonical/"
    "2026-06-09_15-30-02_a1f_p2_teacher_random_onset_canonical__seed0/model_1999.pt"
)
DEFAULT_TARGET_JOINT = "front_left_foot"
DEFAULT_FAULT_ONSET_STEP = 50
DEFAULT_NUM_ENVS = 64
DEFAULT_NUM_STEPS = 1000
CONTROLLED_FAULT_ONSET_STEP = 250
CONTROLLED_NUM_ENVS = 1
CONTROLLED_NUM_STEPS = 600
DEFAULT_OUTPUT_ROOT = "runs/t09_quick_p2_compare"
DEPRECATED_MLP_KWARGS = ("stochastic", "init_noise_std", "noise_std_type", "state_dependent_std")
DEMO_MODES = ("quick_mean", "controlled_single_rollout")


class QuickP2Error(ValueError):
    """Raised for invalid quick P2 demo configuration."""


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


def safe_name(value: str) -> str:
    chars = [char if char.isalnum() or char in {"-", "_"} else "_" for char in value]
    return "".join(chars).strip("_") or "p2_quick_demo"


def default_output_dir(args: argparse.Namespace) -> Path:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    label = safe_name(args.policy_label)
    fault = safe_name(args.fault_profile)
    return resolve_repo_path(DEFAULT_OUTPUT_ROOT) / f"{timestamp}_{label}_{fault}"


def build_parser(add_app_launcher_args: bool = False) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="T09-R2j quick P2 checkpoint sanity comparison evaluator.")
    parser.add_argument("--execute_demo", action="store_true", help="Launch Isaac Sim and execute guarded sanity rollout.")
    parser.add_argument("--dry_run", action="store_true", help="Preview only. This is the default behavior.")
    parser.add_argument("--demo_mode", default="quick_mean", choices=DEMO_MODES)
    parser.add_argument("--policy_label", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--fault_profile", required=True)
    parser.add_argument("--target_joint", default=DEFAULT_TARGET_JOINT)
    parser.add_argument("--fault_onset_step", type=int, default=None)
    parser.add_argument("--num_envs", type=int, default=None)
    parser.add_argument("--num_steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--representative_env_index", type=int, default=0)
    parser.add_argument("--target_vx", type=float, default=None)
    parser.add_argument("--output_dir", default=None)
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
    if pre_args.execute_demo:
        parser = build_parser(add_app_launcher_args=True)
        args, _ = parser.parse_known_args()
        return args
    args, _ = pre_parser.parse_known_args()
    return args


def normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    if args.fault_onset_step is None:
        args.fault_onset_step = (
            CONTROLLED_FAULT_ONSET_STEP
            if args.demo_mode == "controlled_single_rollout"
            else DEFAULT_FAULT_ONSET_STEP
        )
    if args.num_envs is None:
        args.num_envs = CONTROLLED_NUM_ENVS if args.demo_mode == "controlled_single_rollout" else DEFAULT_NUM_ENVS
    if args.num_steps is None:
        args.num_steps = (
            CONTROLLED_NUM_STEPS
            if args.demo_mode == "controlled_single_rollout"
            else DEFAULT_NUM_STEPS
        )
    return args


def validate_args(args: argparse.Namespace) -> None:
    if args.dry_run and args.execute_demo:
        raise QuickP2Error("Use --dry_run without --execute_demo.")
    if args.fault_profile not in ALLOWED_FAULT_PROFILES:
        raise QuickP2Error(f"T09-R2j allows only fault profiles {ALLOWED_FAULT_PROFILES}.")
    if args.task not in ALLOWED_TASKS:
        raise QuickP2Error(f"T09-R2j supports only tasks {ALLOWED_TASKS}.")
    if args.target_joint != DEFAULT_TARGET_JOINT:
        raise QuickP2Error(f"T09-R2j P2 target_joint is fixed to {DEFAULT_TARGET_JOINT}.")
    if args.fault_onset_step < 0:
        raise QuickP2Error("--fault_onset_step must be non-negative.")
    if args.num_envs <= 0:
        raise QuickP2Error("--num_envs must be > 0.")
    if args.num_steps <= 0:
        raise QuickP2Error("--num_steps must be > 0.")
    if args.seed < 0:
        raise QuickP2Error("--seed must be non-negative.")
    if args.representative_env_index < 0:
        raise QuickP2Error("--representative_env_index must be non-negative.")
    if args.representative_env_index >= args.num_envs:
        raise QuickP2Error("--representative_env_index must be < --num_envs.")
    if args.execute_demo:
        checkpoint_path = resolve_repo_path(args.checkpoint_path)
        if not checkpoint_path.is_file():
            raise QuickP2Error(f"checkpoint_path does not exist: {repo_relative(checkpoint_path)}")


def print_dry_run(args: argparse.Namespace) -> None:
    output_dir = resolve_repo_path(args.output_dir) if args.output_dir else default_output_dir(args)
    print("[T09-R2j P2 QUICK DEMO DRY-RUN]")
    print("  demo_scope: quick_p2_checkpoint_sanity")
    print(f"  demo_mode: {args.demo_mode}")
    print("  not_paper_grade: True")
    print("  no_isaac_sim_launched: True")
    print("  no_training: True")
    print("  no_checkpoint_freeze: True")
    print("  no_checkpoint_pointer_edit: True")
    print("  no_observed_manifest_update: True")
    print(f"  policy_label: {args.policy_label}")
    print(f"  task: {args.task}")
    print(f"  checkpoint_path: {args.checkpoint_path}")
    print(f"  fault_profile: {args.fault_profile}")
    print(f"  target_joint: {args.target_joint}")
    print(f"  fault_onset_step: {args.fault_onset_step}")
    print(f"  seed: {args.seed}")
    print(f"  representative_env_index: {args.representative_env_index}")
    print(f"  target_vx_requested: {args.target_vx}")
    print("  target_vx_available: false in dry_run")
    print("  requested_semantics: simulation_joint_state_override_lock")
    print("  fallback_allowed: False")
    print(f"  num_envs: {args.num_envs}")
    print(f"  num_steps: {args.num_steps}")
    print(f"  output_dir: {repo_relative(output_dir)}")
    execute_parts = [
        "python evaluators/run_t09_p2_quick_demo_compare.py --execute_demo",
        f"--demo_mode {args.demo_mode}",
        f"--policy_label {args.policy_label}",
        f"--task {args.task}",
        f"--checkpoint_path {args.checkpoint_path}",
        f"--fault_profile {args.fault_profile}",
        f"--target_joint {args.target_joint}",
        f"--fault_onset_step {args.fault_onset_step}",
        f"--num_envs {args.num_envs}",
        f"--num_steps {args.num_steps}",
        f"--seed {args.seed}",
        f"--representative_env_index {args.representative_env_index}",
        f"--output_dir {repo_relative(output_dir)}",
        "--headless",
    ]
    if args.target_vx is not None:
        execute_parts.insert(-2, f"--target_vx {args.target_vx}")
    print(f"  execute_command: {' '.join(execute_parts)}")


def _sanitize_model_cfg(model_cfg: Any, *, name: str) -> list[str]:
    if not isinstance(model_cfg, dict):
        raise QuickP2Error(f"agent {name} config must be a dict, got {type(model_cfg).__name__}.")
    removed: list[str] = []
    for key in DEPRECATED_MLP_KWARGS:
        if key in model_cfg:
            model_cfg.pop(key)
            removed.append(key)
    return removed


def prepare_agent_cfg(agent_cfg: Any) -> tuple[Any, dict[str, Any]]:
    import importlib.metadata as metadata

    from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg

    installed_version = metadata.version("rsl-rl-lib")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
    agent_cfg.obs_groups = {"actor": ["policy"], "critic": ["policy"]}
    cfg_dict = agent_cfg.to_dict()
    removed = {
        "actor": _sanitize_model_cfg(cfg_dict.get("actor"), name="actor"),
        "critic": _sanitize_model_cfg(cfg_dict.get("critic"), name="critic"),
    }
    actor_cfg = cfg_dict.get("actor")
    critic_cfg = cfg_dict.get("critic")
    if not actor_cfg.get("class_name"):
        raise QuickP2Error("OnPolicyRunner config missing actor.class_name after compatibility conversion.")
    if not critic_cfg.get("class_name"):
        raise QuickP2Error("OnPolicyRunner config missing critic.class_name after compatibility conversion.")
    if cfg_dict.get("obs_groups") != {"actor": ["policy"], "critic": ["policy"]}:
        raise QuickP2Error(f"obs_groups must map actor/critic to policy only, got {cfg_dict.get('obs_groups')!r}.")
    print("[T09-R2j AGENT CFG]")
    print(f"  class_name: {cfg_dict.get('class_name')}", flush=True)
    print(f"  obs_groups: {cfg_dict.get('obs_groups')}", flush=True)
    print(f"  removed_deprecated_actor_keys: {removed['actor']}", flush=True)
    print(f"  removed_deprecated_critic_keys: {removed['critic']}", flush=True)
    return agent_cfg, cfg_dict


def scalar(value: Any, default: float = 0.0) -> float:
    try:
        import torch

        if isinstance(value, torch.Tensor):
            if value.numel() == 0:
                return default
            return float(value.detach().float().mean().cpu().item())
    except ImportError:
        pass
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def optional_float(value: Any) -> float | None:
    try:
        import torch

        if isinstance(value, torch.Tensor):
            if value.numel() == 0:
                return None
            return float(value.detach().float().mean().cpu().item())
    except ImportError:
        pass
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def done_count(dones: Any) -> int:
    try:
        import torch

        tensor = torch.as_tensor(dones)
        return int(tensor.to(dtype=torch.bool).sum().detach().cpu().item())
    except Exception:
        return 0


def _asset_from_action_term(env: Any, action_term_name: str | None = None) -> Any | None:
    action_manager = getattr(env.unwrapped, "action_manager", None)
    terms = getattr(action_manager, "_terms", None)
    if not isinstance(terms, dict) or not terms:
        return None
    term = terms.get(action_term_name) if action_term_name is not None else next(iter(terms.values()))
    if term is None:
        return None
    return getattr(term, "_asset", None)


def _candidate_velocity_tensors(env: Any, extras: Any, *, action_term_name: str | None = None) -> list[tuple[str, Any]]:
    candidates: list[tuple[str, Any]] = []
    asset = _asset_from_action_term(env, action_term_name)
    data = getattr(asset, "data", None)
    for attr_name in (
        "root_lin_vel_w",
        "root_link_lin_vel_w",
        "root_com_lin_vel_w",
        "root_lin_vel_b",
        "root_link_lin_vel_b",
        "root_com_lin_vel_b",
    ):
        value = getattr(data, attr_name, None)
        if value is not None:
            candidates.append((f"robot.data.{attr_name}", value))
    for state_name in ("root_state_w", "root_link_state_w"):
        state = getattr(data, state_name, None)
        if state is not None:
            try:
                candidates.append((f"robot.data.{state_name}[:, 7:10]", state[:, 7:10]))
            except Exception:
                pass

    if isinstance(extras, dict):
        for container_name in ("log", "episode", "metrics"):
            container = extras.get(container_name)
            if not isinstance(container, dict):
                continue
            for key in (
                "base_lin_vel_x",
                "base_lin_vel_x_mean",
                "mean_base_lin_vel_x",
                "mean_vel_x",
                "root_lin_vel_x",
            ):
                if key in container:
                    candidates.append((f"extras.{container_name}.{key}", container[key]))

    return candidates


def maybe_set_target_vx(env: Any, target_vx: float | None) -> dict[str, Any]:
    if target_vx is None:
        return {
            "target_vx_available": False,
            "target_vx": None,
            "target_vx_source": None,
            "target_vx_requested": None,
        }

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
        command = getattr(term, "command", None)
        source_attr = "command"
        if command is None:
            command = getattr(term, "_command", None)
            source_attr = "_command"
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
                "target_vx_source": f"command_manager.{term_name}.{source_attr}[:, 0]",
                "target_vx_requested": float(target_vx),
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


def resolve_forward_velocity_metrics(
    env: Any,
    extras: Any,
    *,
    action_term_name: str | None = None,
    representative_env_index: int = 0,
) -> dict[str, Any]:
    import torch

    for source, value in _candidate_velocity_tensors(env, extras, action_term_name=action_term_name):
        try:
            tensor = torch.as_tensor(value).detach().float()
            if tensor.numel() == 0:
                continue
            if not torch.isfinite(tensor).all():
                continue
            if tensor.ndim == 0:
                vx = tensor.reshape(1)
            elif tensor.ndim == 1:
                vx = tensor
            else:
                vx = tensor[:, 0]
            if vx.numel() == 0:
                continue
            if representative_env_index >= vx.numel():
                raise QuickP2Error(
                    f"representative_env_index={representative_env_index} exceeds velocity env count {vx.numel()}."
                )
            per_env_vel_x = [float(item) for item in vx.detach().cpu().tolist()]
            return {
                "mean_vel_x": float(vx.mean().cpu().item()),
                "representative_env_vel_x": float(vx[representative_env_index].cpu().item()),
                "std_base_lin_vel_x": float(vx.std(unbiased=False).cpu().item()) if vx.numel() > 1 else 0.0,
                "mean_abs_base_lin_vel_x": float(vx.abs().mean().cpu().item()),
                "per_env_vel_x": json.dumps(per_env_vel_x),
                "velocity_metric_source": source,
                "velocity_available": True,
            }
        except Exception:
            continue
    return {
        "mean_vel_x": None,
        "representative_env_vel_x": None,
        "std_base_lin_vel_x": None,
        "mean_abs_base_lin_vel_x": None,
        "per_env_vel_x": None,
        "velocity_metric_source": None,
        "velocity_available": False,
    }


def done_mask_tensor(dones: Any, *, num_envs: int, device: Any):
    import torch

    tensor = torch.as_tensor(dones, device=device)
    if tensor.ndim == 0:
        tensor = tensor.repeat(num_envs)
    if tensor.ndim > 1:
        tensor = tensor.reshape(tensor.shape[0], -1).any(dim=1)
    if tensor.shape[0] != num_envs:
        raise QuickP2Error(f"done tensor length mismatch: expected {num_envs}, got {tensor.shape[0]}.")
    return tensor.to(dtype=torch.bool)


def maybe_mean(values: list[int]) -> float | None:
    if not values:
        return None
    return float(statistics.mean(values))


def maybe_median(values: list[int]) -> float | None:
    if not values:
        return None
    return float(statistics.median(values))


def mean_numeric_rows(rows: list[dict[str, Any]], key: str, *, before_fault: bool | None = None, fault_onset_step: int = 0) -> float | None:
    values: list[float] = []
    for row in rows:
        if before_fault is not None:
            step = int(row.get("step") or 0)
            if before_fault and step >= fault_onset_step:
                continue
            if not before_fault and step < fault_onset_step:
                continue
        value = optional_float(row.get(key))
        if value is not None:
            values.append(value)
    if not values:
        return None
    return float(statistics.mean(values))


def mean_numeric_window(rows: list[dict[str, Any]], key: str, *, start_step: int, end_step: int) -> float | None:
    values: list[float] = []
    for row in rows:
        step = int(row.get("step") or 0)
        if step < start_step or step >= end_step:
            continue
        value = optional_float(row.get(key))
        if value is not None:
            values.append(value)
    if not values:
        return None
    return float(statistics.mean(values))


def mean_abs_tracking_error(rows: list[dict[str, Any]], *, start_step: int, end_step: int) -> float | None:
    values: list[float] = []
    for row in rows:
        step = int(row.get("step") or 0)
        if step < start_step or step >= end_step:
            continue
        vel_x = optional_float(row.get("mean_vel_x"))
        target_vx = optional_float(row.get("target_vx"))
        if vel_x is None or target_vx is None:
            continue
        values.append(abs(vel_x - target_vx))
    if not values:
        return None
    return float(statistics.mean(values))


def build_p2_survival_summary(
    *,
    first_done_steps: list[int],
    fault_onset_step: int,
    num_steps: int,
) -> dict[str, Any]:
    fault_window_available = num_steps >= fault_onset_step
    failed_before_fault = [step >= 0 and step < fault_onset_step for step in first_done_steps]
    reached_fault = [fault_window_available and not failed for failed in failed_before_fault]
    survived_to_fault_onset = list(reached_fault)
    post_fault_survival_steps: list[int | None] = []
    for first_done_step, reached in zip(first_done_steps, reached_fault):
        if not reached:
            post_fault_survival_steps.append(None)
            continue
        if first_done_step >= fault_onset_step:
            post_fault_survival_steps.append(max(0, first_done_step - fault_onset_step))
        else:
            post_fault_survival_steps.append(max(0, num_steps - fault_onset_step))

    observed_first_done_steps = [step for step in first_done_steps if step >= 0]
    observed_post_fault_steps = [step for step in post_fault_survival_steps if step is not None]
    reached_fault_count = sum(1 for value in reached_fault if value)
    failed_before_fault_count = sum(1 for value in failed_before_fault if value)
    env_count = len(first_done_steps)
    return {
        "first_done_step_by_env": first_done_steps,
        "survived_to_fault_onset_by_env": survived_to_fault_onset,
        "failed_before_fault_by_env": failed_before_fault,
        "reached_fault_by_env": reached_fault,
        "post_fault_survival_steps_by_env": post_fault_survival_steps,
        "survival_to_fault_onset_rate": reached_fault_count / env_count if env_count else None,
        "failed_before_fault_count": failed_before_fault_count,
        "reached_fault_count": reached_fault_count,
        "mean_first_done_step": maybe_mean(observed_first_done_steps),
        "median_first_done_step": maybe_median(observed_first_done_steps),
        "mean_post_fault_survival_steps": maybe_mean(observed_post_fault_steps),
        "median_post_fault_survival_steps": maybe_median(observed_post_fault_steps),
    }


def empty_p2_survival_summary() -> dict[str, Any]:
    return {
        "first_done_step_by_env": None,
        "survived_to_fault_onset_by_env": None,
        "failed_before_fault_by_env": None,
        "reached_fault_by_env": None,
        "post_fault_survival_steps_by_env": None,
        "survival_to_fault_onset_rate": None,
        "failed_before_fault_count": None,
        "reached_fault_count": None,
        "mean_first_done_step": None,
        "median_first_done_step": None,
        "mean_post_fault_survival_steps": None,
        "median_post_fault_survival_steps": None,
    }


def write_json(path: Path, values: dict[str, Any]) -> None:
    path.write_text(json.dumps(values, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def execute_demo(args: argparse.Namespace) -> int:
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    output_dir = resolve_repo_path(args.output_dir) if args.output_dir else default_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")

    env = None
    vec_env = None
    p2_wrapper = None
    summary: dict[str, Any] = {
        "demo_scope": "quick_p2_checkpoint_sanity",
        "demo_mode": args.demo_mode,
        "not_paper_grade": True,
        "policy_label": args.policy_label,
        "task": args.task,
        "checkpoint_path": repo_relative(args.checkpoint_path),
        "fault_profile": args.fault_profile,
        "target_joint": args.target_joint,
        "fault_onset_step": args.fault_onset_step,
        "requested_semantics": "simulation_joint_state_override_lock",
        "actual_semantics": "none" if args.fault_profile == "F0_none" else "pending",
        "fallback_used": False,
        "num_envs": args.num_envs,
        "num_steps": args.num_steps,
        "seed": args.seed,
        "representative_env_index": args.representative_env_index,
        "target_vx_requested": args.target_vx,
        "target_vx_available": False,
        "target_vx": None,
        "target_vx_source": None,
        "note": "quick P2 demo/sanity only, not paper-grade result",
    }

    try:
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

        print("[T09-R2j] env/agent config load start", flush=True)
        env_cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
        agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
        env_cfg.scene.num_envs = args.num_envs
        env_cfg.seed = args.seed
        if args.device is not None:
            env_cfg.sim.device = args.device
        env_cfg.log_dir = str(output_dir)
        print("[T09-R2j] env/agent config load done", flush=True)

        print("[T09-R2j] gym.make start", flush=True)
        env = gym.make(args.task, cfg=env_cfg)
        print("[T09-R2j] gym.make done", flush=True)
        if isinstance(env.unwrapped, DirectMARLEnv):
            env = multi_agent_to_single_agent(env)

        if args.fault_profile == "P2_locked_joint":
            print("[T09-R2j] P2 wrapper attach start", flush=True)
            p2_wrapper = P2JointLockActionMaskWrapper(
                env,
                target_joint=args.target_joint,
                fault_onset_step=args.fault_onset_step,
                fault_onset_mode="fixed",
                expected_action_dim=8,
                requested_semantics="simulation_joint_state_override_lock",
                allow_fallback=False,
                velocity_override=0.0,
                debug=True,
            )
            env = p2_wrapper
            if p2_wrapper.mapping.semantics != "simulation_joint_state_override_lock":
                raise QuickP2Error(f"P2 actual_semantics must be simulation_joint_state_override_lock, got {p2_wrapper.mapping.semantics}.")
            print("[T09-R2j] P2 wrapper attach done", flush=True)

        target_vx_info = maybe_set_target_vx(env, args.target_vx)
        summary.update(target_vx_info)
        print("[T09-R2m] target_vx command status", flush=True)
        print(f"  target_vx_available: {target_vx_info.get('target_vx_available')}", flush=True)
        print(f"  target_vx: {target_vx_info.get('target_vx')}", flush=True)
        print(f"  target_vx_source: {target_vx_info.get('target_vx_source')}", flush=True)
        if target_vx_info.get("target_vx_unavailable_reason"):
            print(f"  target_vx_unavailable_reason: {target_vx_info['target_vx_unavailable_reason']}", flush=True)

        agent_cfg, agent_cfg_dict = prepare_agent_cfg(agent_cfg)
        vec_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        checkpoint_path = str(resolve_repo_path(args.checkpoint_path))

        print("[T09-R2j] runner construction start", flush=True)
        if agent_cfg.class_name == "OnPolicyRunner":
            runner = OnPolicyRunner(vec_env, agent_cfg_dict, log_dir=None, device=agent_cfg.device)
        elif agent_cfg.class_name == "DistillationRunner":
            runner = DistillationRunner(vec_env, agent_cfg_dict, log_dir=None, device=agent_cfg.device)
        else:
            raise QuickP2Error(f"Unsupported runner class: {agent_cfg.class_name}")
        print("[T09-R2j] runner construction done", flush=True)

        print(f"[T09-R2j] checkpoint load start: {repo_relative(checkpoint_path)}", flush=True)
        runner.load(checkpoint_path)
        print("[T09-R2j] checkpoint load done", flush=True)
        policy = runner.get_inference_policy(device=vec_env.unwrapped.device)

        obs = vec_env.get_observations()
        tracking_num_envs = int(getattr(vec_env, "num_envs", args.num_envs))
        tracking_device = vec_env.unwrapped.device
        first_done_step = torch.full((tracking_num_envs,), -1, dtype=torch.long, device=tracking_device)
        rows: list[dict[str, Any]] = []
        reward_sum = 0.0
        reward_count = 0
        total_done_count = 0
        no_nan_inf = True
        last_p2_fault_applied = 0.0
        last_p2_sim_override = 0.0
        last_p2_fallback = 0.0
        last_velocity_warning_source = None
        velocity_metric_source = None
        p2_action_term_name = p2_wrapper.mapping.action_term_name if p2_wrapper is not None else None

        print("[T09-R2j] rollout start", flush=True)
        for step in range(args.num_steps):
            step_number = step + 1
            with torch.inference_mode():
                actions = policy(obs)
                obs, rewards, dones, extras = vec_env.step(actions)
                if hasattr(policy, "reset"):
                    policy.reset(dones)

            reward_mean = scalar(rewards)
            done_mask = done_mask_tensor(dones, num_envs=tracking_num_envs, device=tracking_device)
            step_done_count = int(done_mask.sum().detach().cpu().item())
            total_done_count += step_done_count
            new_first_done_mask = torch.logical_and(done_mask, first_done_step < 0)
            first_done_step[new_first_done_mask] = step_number
            reward_sum += float(torch.as_tensor(rewards).float().sum().detach().cpu().item())
            reward_count += int(torch.as_tensor(rewards).numel())
            no_nan_inf = no_nan_inf and bool(torch.isfinite(torch.as_tensor(rewards)).all().item())

            log_values = extras.get("log", {}) if isinstance(extras, dict) else {}
            last_p2_fault_applied = scalar(log_values.get("P2/fault_applied"))
            last_p2_sim_override = scalar(log_values.get("P2/simulation_override_applied"))
            last_p2_fallback = scalar(log_values.get("P2/fallback_used"))
            velocity_metrics = resolve_forward_velocity_metrics(
                env,
                extras,
                action_term_name=p2_action_term_name,
                representative_env_index=args.representative_env_index,
            )
            if velocity_metrics["velocity_available"] and velocity_metric_source is None:
                velocity_metric_source = velocity_metrics["velocity_metric_source"]
                print(f"[T09-R2l] velocity_metric_source: {velocity_metric_source}", flush=True)
            if not velocity_metrics["velocity_available"] and last_velocity_warning_source is None:
                last_velocity_warning_source = "unresolved"
                print(
                    "[T09-R2l WARNING] forward velocity could not be resolved; "
                    "mean_vel_x fields will be null.",
                    flush=True,
                )
            if args.fault_profile == "P2_locked_joint":
                failed_before_fault_count = int(
                    torch.logical_and(first_done_step >= 0, first_done_step < args.fault_onset_step)
                    .sum()
                    .detach()
                    .cpu()
                    .item()
                )
                reached_fault_count = 0
                survival_to_fault_onset_rate: float | str = ""
                mean_first_done_step: float | str = ""
                mean_post_fault_survival_steps: float | str = ""
                if step_number >= args.fault_onset_step:
                    reached_fault_count = tracking_num_envs - failed_before_fault_count
                    survival_to_fault_onset_rate = reached_fault_count / tracking_num_envs
                observed_done_steps = [
                    int(value)
                    for value in first_done_step.detach().cpu().tolist()
                    if int(value) >= 0
                ]
                if observed_done_steps:
                    mean_first_done_step = float(statistics.mean(observed_done_steps))
                if step_number >= args.fault_onset_step:
                    post_fault_steps: list[int] = []
                    for value in first_done_step.detach().cpu().tolist():
                        value = int(value)
                        if value >= args.fault_onset_step:
                            post_fault_steps.append(max(0, value - args.fault_onset_step))
                        elif value < 0:
                            post_fault_steps.append(max(0, step_number - args.fault_onset_step))
                    if post_fault_steps:
                        mean_post_fault_survival_steps = float(statistics.mean(post_fault_steps))
            else:
                failed_before_fault_count = ""
                reached_fault_count = ""
                survival_to_fault_onset_rate = ""
                mean_first_done_step = ""
                mean_post_fault_survival_steps = ""
            rows.append(
                {
                    "step": step_number,
                    "policy_label": args.policy_label,
                    "task": args.task,
                    "fault_profile": args.fault_profile,
                    "fault_onset_step": args.fault_onset_step,
                    "demo_mode": args.demo_mode,
                    "env_index": args.representative_env_index,
                    "fault_active": step_number >= args.fault_onset_step,
                    "reward_mean": reward_mean,
                    "mean_vel_x": velocity_metrics["mean_vel_x"],
                    "mean_base_lin_vel_x": velocity_metrics["mean_vel_x"],
                    "representative_env_vel_x": velocity_metrics["representative_env_vel_x"],
                    "per_env_vel_x": velocity_metrics["per_env_vel_x"],
                    "std_base_lin_vel_x": velocity_metrics["std_base_lin_vel_x"],
                    "mean_abs_base_lin_vel_x": velocity_metrics["mean_abs_base_lin_vel_x"],
                    "velocity_metric_source": velocity_metrics["velocity_metric_source"],
                    "target_vx_available": target_vx_info.get("target_vx_available"),
                    "target_vx": target_vx_info.get("target_vx"),
                    "target_vx_source": target_vx_info.get("target_vx_source"),
                    "velocity_tracking_error": (
                        None
                        if target_vx_info.get("target_vx") is None or velocity_metrics["mean_vel_x"] is None
                        else velocity_metrics["mean_vel_x"] - float(target_vx_info["target_vx"])
                    ),
                    "done_count": step_done_count,
                    "total_done_count": total_done_count,
                    "failed_before_fault_count": failed_before_fault_count,
                    "reached_fault_count": reached_fault_count,
                    "survival_to_fault_onset_rate": survival_to_fault_onset_rate,
                    "mean_first_done_step": mean_first_done_step,
                    "mean_post_fault_survival_steps": mean_post_fault_survival_steps,
                    "P2/fault_applied": last_p2_fault_applied,
                    "P2/simulation_override_applied": last_p2_sim_override,
                    "P2/fallback_used": last_p2_fallback,
                    "no_nan_inf": no_nan_inf,
                }
            )
            if step < 3 or (step + 1) % 100 == 0 or step + 1 == args.num_steps:
                print(
                    f"[T09-R2j] step={step + 1} reward_mean={reward_mean:.4f} "
                    f"done_count={step_done_count} p2_fault={last_p2_fault_applied:.4f}",
                    flush=True,
                )

        csv_path = output_dir / "rollout_metrics.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()) if rows else [])
            if rows:
                writer.writeheader()
                writer.writerows(rows)

        mean_reward = reward_sum / reward_count if reward_count else 0.0
        first_done_steps = [int(value) for value in first_done_step.detach().cpu().tolist()]
        pre_window_start = max(1, args.fault_onset_step - 50)
        pre_window_end = args.fault_onset_step
        post_window_start = args.fault_onset_step
        post_window_end = args.fault_onset_step + 100
        mean_vel_x_pre_window = mean_numeric_window(
            rows,
            "mean_vel_x",
            start_step=pre_window_start,
            end_step=pre_window_end,
        )
        mean_vel_x_post_window = mean_numeric_window(
            rows,
            "mean_vel_x",
            start_step=post_window_start,
            end_step=post_window_end,
        )
        survival_summary = (
            build_p2_survival_summary(
                first_done_steps=first_done_steps,
                fault_onset_step=args.fault_onset_step,
                num_steps=args.num_steps,
            )
            if args.fault_profile == "P2_locked_joint"
            else empty_p2_survival_summary()
        )
        summary.update(
            {
                "actual_semantics": p2_wrapper.mapping.semantics if p2_wrapper is not None else "none",
                "fallback_used": bool(p2_wrapper.fallback_used) if p2_wrapper is not None else False,
                "mean_reward": mean_reward,
                "done_count": total_done_count,
                "total_done_count": total_done_count,
                "no_nan_inf": no_nan_inf,
                "P2/fault_applied": last_p2_fault_applied,
                "P2/simulation_override_applied": last_p2_sim_override,
                "P2/fallback_used": last_p2_fallback,
                "pre_fault_window": [pre_window_start, pre_window_end],
                "post_fault_window": [post_window_start, post_window_end],
                "mean_vel_x_pre_fault_window": mean_vel_x_pre_window,
                "mean_vel_x_post_fault_window": mean_vel_x_post_window,
                "delta_vel_x_post_minus_pre_window": (
                    None
                    if mean_vel_x_pre_window is None or mean_vel_x_post_window is None
                    else mean_vel_x_post_window - mean_vel_x_pre_window
                ),
                "mean_abs_tracking_error_pre_fault": (
                    mean_abs_tracking_error(
                        rows,
                        start_step=pre_window_start,
                        end_step=pre_window_end,
                    )
                    if target_vx_info.get("target_vx_available")
                    else None
                ),
                "mean_abs_tracking_error_post_fault": (
                    mean_abs_tracking_error(
                        rows,
                        start_step=post_window_start,
                        end_step=post_window_end,
                    )
                    if target_vx_info.get("target_vx_available")
                    else None
                ),
                "mean_vel_x_pre_fault": mean_numeric_rows(
                    rows,
                    "mean_vel_x",
                    before_fault=True,
                    fault_onset_step=args.fault_onset_step,
                ),
                "mean_vel_x_post_fault": mean_numeric_rows(
                    rows,
                    "mean_vel_x",
                    before_fault=False,
                    fault_onset_step=args.fault_onset_step,
                ),
                "delta_vel_x_post_minus_pre": None,
                "velocity_metric_source": velocity_metric_source,
                **survival_summary,
                "rollout_metrics_csv": repo_relative(csv_path),
            }
        )
        if summary["mean_vel_x_pre_fault"] is not None and summary["mean_vel_x_post_fault"] is not None:
            summary["delta_vel_x_post_minus_pre"] = (
                summary["mean_vel_x_post_fault"] - summary["mean_vel_x_pre_fault"]
            )
        write_json(output_dir / "summary.json", summary)
        print(f"[T09-R2j] summary: {repo_relative(output_dir / 'summary.json')}", flush=True)
        return 0
    finally:
        if summary:
            try:
                write_json(output_dir / "summary.json", summary)
            except Exception:
                pass
        if vec_env is not None:
            vec_env.close()
        elif env is not None:
            env.close()
        simulation_app.close()


def main() -> int:
    args = normalize_args(parse_args())
    validate_args(args)
    if not args.execute_demo:
        print_dry_run(args)
        return 0
    return execute_demo(args)


if __name__ == "__main__":
    raise SystemExit(main())
