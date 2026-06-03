#!/usr/bin/env python3
"""Guarded T09-E1 P4 torque-degradation pilot runner.

Default behavior is dry-run preview only. Runtime mapping inspection requires
--execute_mapping_preflight. Pilot execution requires --execute_pilot and is
restricted to A0/A2/A5 with P4 only.
"""

from __future__ import annotations

import argparse
import faulthandler
import json
import math
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MATRIX = "configs/ablation/t09_conference_matrix.yaml"
DEFAULT_PROFILE = "configs/fault/torque_scale/p4_torque_degradation.yaml"
DEFAULT_RUN_ROOT = "runs/t09_p4_pilot"
ALLOWED_ROWS = ("A0", "A2", "A5")
FAULT_PROFILE = "P4_torque_degradation"
TARGET_JOINT = "front_left_foot"
TORQUE_SCALE = 0.5
FAULT_ONSET_STEP = 50
NUM_ENVS = 8
ALLOWED_EPISODES = (2, 5)
POLICY_MODE = "deterministic"
DEBUG_STEP_LOG_LIMIT = 5
DEPRECATED_MLP_KWARGS = ("stochastic", "init_noise_std", "noise_std_type", "state_dependent_std")


class PilotConfigError(ValueError):
    """Raised for invalid T09-E1 pilot configuration."""


def _debug(message: str) -> None:
    print(f"[T09-E1 DEBUG] {message}", flush=True)


def _strip_inline_comment(line: str) -> str:
    if "#" not in line:
        return line
    return line.split("#", 1)[0].rstrip()


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"none", "null"}:
        return lowered
    if value:
        try:
            return int(value)
        except ValueError:
            pass
        try:
            return float(value)
        except ValueError:
            pass
    return value


def _split_key_value(text: str, line_no: int) -> tuple[str, Any]:
    if ":" not in text:
        raise PilotConfigError(f"line {line_no}: expected 'key: value'")
    key, raw_value = text.split(":", 1)
    key = key.strip()
    if not key:
        raise PilotConfigError(f"line {line_no}: empty key")
    return key, _parse_scalar(raw_value)


def load_flat_yaml(path: Path) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    lists: dict[str, list[dict[str, Any]]] = {}
    current_list: str | None = None
    current_item: dict[str, Any] | None = None

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise PilotConfigError(f"could not read {repo_relative(path)}: {exc}") from exc

    for line_no, raw_line in enumerate(lines, start=1):
        line = _strip_inline_comment(raw_line).rstrip()
        if not line.strip():
            continue

        if not raw_line.startswith(" "):
            key, value = _split_key_value(line, line_no)
            if value == "":
                current_list = key
                current_item = None
                lists.setdefault(key, [])
            else:
                current_list = None
                current_item = None
                metadata[key] = value
            continue

        if current_list is None:
            raise PilotConfigError(f"line {line_no}: indented content without a list block")
        stripped = line.strip()
        if stripped.startswith("- "):
            if not line.startswith("  - "):
                raise PilotConfigError(f"line {line_no}: list item must use two-space indentation")
            current_item = {}
            lists[current_list].append(current_item)
            first_field = stripped[2:].strip()
            if first_field:
                key, value = _split_key_value(first_field, line_no)
                current_item[key] = value
            continue
        if not line.startswith("    "):
            raise PilotConfigError(f"line {line_no}: list field must use four-space indentation")
        if current_item is None:
            raise PilotConfigError(f"line {line_no}: list field before list item")
        key, value = _split_key_value(stripped, line_no)
        current_item[key] = value

    return {"metadata": metadata, "lists": lists}


def resolve_repo_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def repo_relative(path: str | Path) -> str:
    resolved = resolve_repo_path(path).resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(resolved)


def read_pointer(pointer_value: str) -> dict[str, Any]:
    pointer_path = resolve_repo_path(pointer_value)
    if not pointer_path.is_file():
        raise PilotConfigError(f"checkpoint pointer missing: {repo_relative(pointer_path)}")
    pointer = load_flat_yaml(pointer_path)["metadata"]
    checkpoint_value = pointer.get("checkpoint_path")
    if not checkpoint_value:
        raise PilotConfigError(f"checkpoint pointer missing checkpoint_path: {repo_relative(pointer_path)}")
    checkpoint_path = resolve_repo_path(str(checkpoint_value))
    if not checkpoint_path.is_file():
        raise PilotConfigError(f"checkpoint file missing: {repo_relative(checkpoint_path)}")
    return {
        "pointer_path": repo_relative(pointer_path),
        "resolved_checkpoint_path": repo_relative(checkpoint_path),
        "checkpoint_exists": True,
    }


def load_rows(matrix_path: str) -> dict[str, dict[str, Any]]:
    matrix = load_flat_yaml(resolve_repo_path(matrix_path))
    rows = matrix["lists"].get("rows", [])
    return {str(row.get("ablation_id")): row for row in rows}


def validate_args(args: argparse.Namespace) -> None:
    runtime_flags = [args.execute_mapping_preflight, args.execute_pilot]
    if args.dry_run and any(runtime_flags):
        raise PilotConfigError("Use --dry_run without runtime execution flags.")
    if sum(1 for flag in runtime_flags if flag) > 1:
        raise PilotConfigError("Use only one runtime flag: --execute_mapping_preflight or --execute_pilot.")
    if args.fault_profile != FAULT_PROFILE:
        raise PilotConfigError(f"T09-E1 allows only --fault_profile {FAULT_PROFILE}.")
    if args.target_joint != TARGET_JOINT:
        raise PilotConfigError(f"T09-E1 target_joint is fixed to {TARGET_JOINT}.")
    if float(args.torque_scale) != TORQUE_SCALE:
        raise PilotConfigError(f"T09-E1 torque_scale is fixed to {TORQUE_SCALE}.")
    if int(args.fault_onset_step) != FAULT_ONSET_STEP:
        raise PilotConfigError(f"T09-E1 fault_onset_step is fixed to {FAULT_ONSET_STEP}.")
    if int(args.num_envs) != NUM_ENVS:
        raise PilotConfigError(f"T09-E1 num_envs is fixed to {NUM_ENVS}.")
    if int(args.episodes) not in ALLOWED_EPISODES:
        raise PilotConfigError(f"T09-E1 episodes must be one of {ALLOWED_EPISODES}.")
    if args.policy_mode != POLICY_MODE:
        raise PilotConfigError("T09-E1 supports deterministic policy mode only.")
    if args.debug_timeout_sec < 0:
        raise PilotConfigError("--debug_timeout_sec must be >= 0.")
    if args.max_debug_steps_per_episode <= 0:
        raise PilotConfigError("--max_debug_steps_per_episode must be > 0.")
    if args.telemetry_interval_steps <= 0:
        raise PilotConfigError("--telemetry_interval_steps must be > 0.")
    if args.policy_debug_only and not args.execute_pilot:
        raise PilotConfigError("--policy_debug_only requires --execute_pilot.")
    if args.ablation_id is not None and args.ablation_id not in ALLOWED_ROWS:
        raise PilotConfigError(f"T09-E1 allows only rows {ALLOWED_ROWS}.")
    if args.execute_pilot and args.ablation_id is None:
        raise PilotConfigError("--execute_pilot requires --ablation_id.")


def validate_p4_profile(profile_path: str) -> dict[str, Any]:
    profile = load_flat_yaml(resolve_repo_path(profile_path))["metadata"]
    expected = {
        "profile_id": FAULT_PROFILE,
        "family": "torque_scale",
        "execution_enabled": False,
        "runtime_fault_injection_implemented": False,
    }
    for key, expected_value in expected.items():
        actual = profile.get(key)
        if actual != expected_value:
            raise PilotConfigError(f"P4 profile field {key} must be {expected_value!r}, got {actual!r}")
    return profile


def row_role(row_id: str) -> str:
    roles = {
        "A0": "zero-shot healthy PPO under P4",
        "A2": "frozen student under P4; no online learning",
        "A5": "frozen residual-policy adaptation under P4; no online learning",
    }
    return roles[row_id]


def checkpoint_plan_for_row(row_id: str, rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if row_id not in rows:
        raise PilotConfigError(f"matrix row missing: {row_id}")
    row = rows[row_id]
    primary = read_pointer(str(row["checkpoint_pointer"]))
    plan: dict[str, Any] = {
        "ablation_id": row_id,
        "row_name": row["name"],
        "row_role": row_role(row_id),
        "checkpoint_pointer": primary["pointer_path"],
        "resolved_checkpoint_path": primary["resolved_checkpoint_path"],
        "checkpoint_exists": primary["checkpoint_exists"],
    }
    if row_id == "A5":
        student = read_pointer(str(row["student_checkpoint_pointer"]))
        plan.update(
            {
                "student_checkpoint_pointer": student["pointer_path"],
                "resolved_student_checkpoint_path": student["resolved_checkpoint_path"],
                "residual_checkpoint_role": "pilot/development unless explicitly promoted later",
                "residual_scale": row["residual_scale"],
            }
        )
    return plan


def _contains_obs_group(obs_groups: Any, group_name: str) -> bool:
    if isinstance(obs_groups, dict):
        values = obs_groups.values()
    else:
        values = [obs_groups]
    for value in values:
        if isinstance(value, str) and value == group_name:
            return True
        if isinstance(value, (list, tuple, set)) and group_name in value:
            return True
    return False


def _print_agent_cfg_diagnostic(row_id: str, agent_cfg_dict: dict[str, Any]) -> None:
    actor_cfg = agent_cfg_dict.get("actor")
    critic_cfg = agent_cfg_dict.get("critic")
    actor_has_class = isinstance(actor_cfg, dict) and bool(actor_cfg.get("class_name"))
    critic_has_class = isinstance(critic_cfg, dict) and bool(critic_cfg.get("class_name"))
    print("[T09-E1 AGENT CFG DIAGNOSTIC]", flush=True)
    print(f"  ablation_id: {row_id}", flush=True)
    print(f"  top_level_keys: {sorted(agent_cfg_dict.keys())}", flush=True)
    print(f"  actor_exists: {isinstance(actor_cfg, dict)}", flush=True)
    print(f"  critic_exists: {isinstance(critic_cfg, dict)}", flush=True)
    print(f"  actor.class_name_exists: {actor_has_class}", flush=True)
    print(f"  critic.class_name_exists: {critic_has_class}", flush=True)
    print(f"  obs_groups: {agent_cfg_dict.get('obs_groups')}", flush=True)
    print(f"  class_name: {agent_cfg_dict.get('class_name')}", flush=True)


def _validate_a0_agent_cfg(agent_cfg_dict: dict[str, Any]) -> None:
    actor_cfg = agent_cfg_dict.get("actor")
    critic_cfg = agent_cfg_dict.get("critic")
    if not isinstance(actor_cfg, dict) or not actor_cfg.get("class_name"):
        raise ValueError("A0 OnPolicyRunner config missing actor.class_name after local compatibility conversion.")
    if not isinstance(critic_cfg, dict) or not critic_cfg.get("class_name"):
        raise ValueError("A0 OnPolicyRunner config missing critic.class_name after local compatibility conversion.")

    obs_groups = agent_cfg_dict.get("obs_groups")
    if _contains_obs_group(obs_groups, "teacher_policy"):
        raise ValueError("A0 obs_groups must not expose teacher_policy.")
    if _contains_obs_group(obs_groups, "true_fault_state"):
        raise ValueError("A0 obs_groups must not expose true_fault_state.")
    if obs_groups != {"actor": ["policy"], "critic": ["policy"]}:
        raise ValueError(f"A0 obs_groups must map actor/critic to policy only, got: {obs_groups!r}")


def _prepare_a0_agent_cfg_dict(agent_cfg: Any) -> dict[str, Any]:
    from importlib import metadata

    from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg

    installed_version = metadata.version("rsl-rl-lib")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
    agent_cfg.obs_groups = {"actor": ["policy"], "critic": ["policy"]}
    agent_cfg_dict = agent_cfg.to_dict()
    _print_agent_cfg_diagnostic("A0", agent_cfg_dict)
    _validate_a0_agent_cfg(agent_cfg_dict)
    return agent_cfg_dict


def _sanitize_mlp_model_cfg(model_cfg: Any, *, model_name: str) -> list[str]:
    if not isinstance(model_cfg, dict):
        raise ValueError(f"A5 OnPolicyRunner config {model_name} must be a dict, got {type(model_cfg).__name__}.")
    removed_keys = []
    for key in DEPRECATED_MLP_KWARGS:
        if key in model_cfg:
            removed_keys.append(key)
            model_cfg.pop(key)
    return removed_keys


def _print_a5_agent_cfg_diagnostic(agent_cfg_dict: dict[str, Any], removed_keys: dict[str, list[str]]) -> None:
    actor_cfg = agent_cfg_dict.get("actor")
    critic_cfg = agent_cfg_dict.get("critic")
    actor_keys = sorted(actor_cfg.keys()) if isinstance(actor_cfg, dict) else []
    critic_keys = sorted(critic_cfg.keys()) if isinstance(critic_cfg, dict) else []
    actor_has_class = isinstance(actor_cfg, dict) and bool(actor_cfg.get("class_name"))
    critic_has_class = isinstance(critic_cfg, dict) and bool(critic_cfg.get("class_name"))
    distribution_cfg_exists = isinstance(actor_cfg, dict) and actor_cfg.get("distribution_cfg") is not None
    print("[T09-E1 A5 AGENT CFG DIAGNOSTIC]", flush=True)
    print(f"  top_level_keys: {sorted(agent_cfg_dict.keys())}", flush=True)
    print(f"  actor_keys: {actor_keys}", flush=True)
    print(f"  critic_keys: {critic_keys}", flush=True)
    print(f"  actor.class_name_exists: {actor_has_class}", flush=True)
    print(f"  critic.class_name_exists: {critic_has_class}", flush=True)
    print(f"  distribution_cfg_exists: {distribution_cfg_exists}", flush=True)
    print(f"  removed_deprecated_actor_keys: {removed_keys.get('actor', [])}", flush=True)
    print(f"  removed_deprecated_critic_keys: {removed_keys.get('critic', [])}", flush=True)
    print(f"  obs_groups: {agent_cfg_dict.get('obs_groups')}", flush=True)


def _validate_a5_agent_cfg(agent_cfg_dict: dict[str, Any]) -> None:
    actor_cfg = agent_cfg_dict.get("actor")
    critic_cfg = agent_cfg_dict.get("critic")
    if not isinstance(actor_cfg, dict) or not actor_cfg.get("class_name"):
        raise ValueError("A5 OnPolicyRunner config missing actor.class_name after local compatibility conversion.")
    if not isinstance(critic_cfg, dict) or not critic_cfg.get("class_name"):
        raise ValueError("A5 OnPolicyRunner config missing critic.class_name after local compatibility conversion.")
    for model_name, model_cfg in (("actor", actor_cfg), ("critic", critic_cfg)):
        if "stochastic" in model_cfg:
            raise ValueError(f"A5 {model_name} config still contains deprecated stochastic kwarg.")

    obs_groups = agent_cfg_dict.get("obs_groups")
    if _contains_obs_group(obs_groups, "teacher_policy"):
        raise ValueError("A5 obs_groups must not expose teacher_policy.")
    if _contains_obs_group(obs_groups, "true_fault_state"):
        raise ValueError("A5 obs_groups must not expose true_fault_state.")
    if obs_groups != {"actor": ["policy"], "critic": ["policy"]}:
        raise ValueError(f"A5 obs_groups must map actor/critic to policy only, got: {obs_groups!r}")


def _prepare_a5_agent_cfg_dict(agent_cfg: Any) -> dict[str, Any]:
    from importlib import metadata

    from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg

    installed_version = metadata.version("rsl-rl-lib")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
    agent_cfg.obs_groups = {"actor": ["policy"], "critic": ["policy"]}
    agent_cfg_dict = agent_cfg.to_dict()
    removed_keys = {
        "actor": _sanitize_mlp_model_cfg(agent_cfg_dict.get("actor"), model_name="actor"),
        "critic": _sanitize_mlp_model_cfg(agent_cfg_dict.get("critic"), model_name="critic"),
    }
    _print_a5_agent_cfg_diagnostic(agent_cfg_dict, removed_keys)
    _validate_a5_agent_cfg(agent_cfg_dict)
    return agent_cfg_dict


def print_no_runtime_summary() -> None:
    print("[T09-E1 SUMMARY] dry-run preview only")
    print("  no_training: True")
    print("  no_evaluation_execution: True")
    print("  no_isaac_sim: True")
    print("  no_checkpoint_writes: True")
    print("  no_observed_result_updates: True")
    print("  no_paper_grade_claims: True")


def print_mapping_preview(args: argparse.Namespace) -> None:
    print("[T09-E1 ACTION MAPPING PREVIEW]")
    print(f"  fault_profile: {args.fault_profile}")
    print(f"  target_joint: {args.target_joint}")
    print(f"  expected_action_dim: 8")
    print(f"  torque_scale: {args.torque_scale}")
    print(f"  fault_onset_step: {args.fault_onset_step}")
    print("  runtime_check: --execute_pilot will require exactly one joint action term and one target action index.")
    print("  dry_run_note: no env is constructed and no Isaac Sim module is imported in dry-run mode.")


def print_preview(args: argparse.Namespace, rows: dict[str, dict[str, Any]]) -> None:
    row_ids = [args.ablation_id] if args.ablation_id is not None else list(ALLOWED_ROWS)
    for row_id in row_ids:
        plan = checkpoint_plan_for_row(row_id, rows)
        print("[T09-E1 COMMAND PREVIEW]")
        for key, value in plan.items():
            print(f"  {key}: {value}")
        print(f"  fault_profile: {args.fault_profile}")
        print(f"  num_envs: {args.num_envs}")
        print(f"  episodes: {args.episodes}")
        print(f"  target_joint: {args.target_joint}")
        print(f"  torque_scale: {args.torque_scale}")
        print(f"  fault_onset_step: {args.fault_onset_step}")
        print(f"  policy_mode: {args.policy_mode}")
        print("  execution_disabled_reason: --execute_pilot was not provided.")
        print(
            "  future_execute_command: "
            f"python evaluators/run_t09_p4_pilot_eval.py --execute_pilot --ablation_id {row_id} "
            f"--fault_profile {args.fault_profile} --num_envs {args.num_envs} --episodes {args.episodes} "
            f"--target_joint {args.target_joint} --torque_scale {args.torque_scale} "
            f"--fault_onset_step {args.fault_onset_step} --policy_mode {args.policy_mode}"
        )


def build_parser(add_app_launcher_args: bool = False):
    parser = argparse.ArgumentParser(description="T09-E1 P4 pilot evaluator")
    parser.add_argument("--matrix", default=DEFAULT_MATRIX)
    parser.add_argument("--p4_profile", default=DEFAULT_PROFILE)
    parser.add_argument("--run_root", default=DEFAULT_RUN_ROOT)
    parser.add_argument("--dry_run", action="store_true", help="Preview only. This is the default behavior.")
    parser.add_argument(
        "--execute_mapping_preflight",
        action="store_true",
        help="Construct the Ant env only to verify P4 action mapping. No checkpoints or env steps.",
    )
    parser.add_argument("--execute_pilot", action="store_true", help="Explicitly execute the guarded P4 pilot.")
    parser.add_argument("--verify_profile", action="store_true", help="Validate the P4 profile scaffold.")
    parser.add_argument("--preview_mapping", action="store_true", help="Preview runtime action mapping checks.")
    parser.add_argument("--ablation_id", choices=ALLOWED_ROWS)
    parser.add_argument("--fault_profile", default=FAULT_PROFILE)
    parser.add_argument("--num_envs", type=int, default=NUM_ENVS)
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--target_joint", default=TARGET_JOINT)
    parser.add_argument("--torque_scale", type=float, default=TORQUE_SCALE)
    parser.add_argument("--fault_onset_step", type=int, default=FAULT_ONSET_STEP)
    parser.add_argument("--policy_mode", default=POLICY_MODE)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--debug_timeout_sec", type=float, default=30.0)
    parser.add_argument("--max_debug_steps_per_episode", type=int, default=200)
    parser.add_argument("--telemetry_interval_steps", type=int, default=50)
    parser.add_argument(
        "--debug_skip_app_close_on_error",
        action="store_true",
        help="Debug only: skip SimulationApp.close after a pre-evaluation runtime exception.",
    )
    parser.add_argument(
        "--policy_debug_only",
        action="store_true",
        help="Construct/load policy after P4 mapping, then exit without stepping the env.",
    )
    if not add_app_launcher_args:
        parser.add_argument("--device", default=None)
    if add_app_launcher_args:
        from isaaclab.app import AppLauncher

        AppLauncher.add_app_launcher_args(parser)
    return parser


def parse_args() -> argparse.Namespace:
    pre_parser = build_parser(add_app_launcher_args=False)
    pre_args, _ = pre_parser.parse_known_args()
    if pre_args.execute_mapping_preflight or pre_args.execute_pilot:
        parser = build_parser(add_app_launcher_args=True)
        args, _ = parser.parse_known_args()
        return args
    args, _ = pre_parser.parse_known_args()
    return args


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _tensor_to_float(value: Any) -> float | None:
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "mean"):
        value = value.mean()
    if hasattr(value, "item"):
        return float(value.item())
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _timeout_for_env(extras: dict[str, Any], env_id: int) -> bool:
    time_outs = extras.get("time_outs")
    if time_outs is None:
        return False
    try:
        value = time_outs[env_id]
    except (IndexError, KeyError, TypeError):
        return False
    if hasattr(value, "item"):
        return bool(value.item())
    return bool(value)


def _surface_policy_load_error(exc: Exception) -> None:
    print("[T09-E1 ERROR] policy model construction/load failed before evaluation", file=sys.stderr, flush=True)
    traceback.print_exc(file=sys.stderr)
    print(f"[T09-E1 ERROR] exception_type: {type(exc).__name__}", file=sys.stderr, flush=True)
    print(f"[T09-E1 ERROR] exception_message: {exc}", file=sys.stderr, flush=True)


def _surface_eval_loop_error(
    exc: Exception,
    *,
    ablation_id: str,
    global_step: int | None,
    rollout_steps_executed: int,
    completed_episodes: int,
    requested_episodes: int,
    fault_window_reached: bool,
    no_nan_inf: bool,
    last_dones_diagnostic: dict[str, Any] | None,
) -> None:
    print("[T09-E1 ERROR] eval loop failed before clean status classification", file=sys.stderr, flush=True)
    traceback.print_exc(file=sys.stderr)
    print(f"[T09-E1 ERROR] exception_type: {type(exc).__name__}", file=sys.stderr, flush=True)
    print(f"[T09-E1 ERROR] exception_message: {exc}", file=sys.stderr, flush=True)
    print("[T09-E1 ERROR] last known rollout state:", file=sys.stderr, flush=True)
    print(f"  ablation_id: {ablation_id}", file=sys.stderr, flush=True)
    print(f"  global_step: {global_step}", file=sys.stderr, flush=True)
    print(f"  rollout_steps_executed: {rollout_steps_executed}", file=sys.stderr, flush=True)
    print(f"  completed_episodes: {completed_episodes}", file=sys.stderr, flush=True)
    print(f"  requested_episodes: {requested_episodes}", file=sys.stderr, flush=True)
    print(f"  fault_window_reached: {fault_window_reached}", file=sys.stderr, flush=True)
    print(f"  no_nan_inf: {no_nan_inf}", file=sys.stderr, flush=True)
    print(f"  last_dones_diagnostic: {last_dones_diagnostic}", file=sys.stderr, flush=True)


def _close_env_for_debug_error(env_obj: Any) -> None:
    if env_obj is None:
        return
    try:
        _debug("env close start")
        env_obj.close()
        _debug("env close done")
    except Exception as close_exc:
        print(
            f"[T09-E1 ERROR] env close failed during debug cleanup: "
            f"{type(close_exc).__name__}: {close_exc}",
            file=sys.stderr,
            flush=True,
        )


def _mask_hidden_tensor_clone(hidden_tensor: Any, dones: Any, num_envs: int) -> Any:
    import torch

    if not hasattr(hidden_tensor, "shape"):
        raise ValueError(f"Unsupported student hidden-state object: {type(hidden_tensor).__name__}")
    if hidden_tensor.shape[-2] != num_envs:
        raise ValueError(
            "Invalid A2 student hidden-state shape for safe reset: "
            f"hidden_state={tuple(hidden_tensor.shape)}, num_envs={num_envs}"
        )
    keep = (~dones.to(device=hidden_tensor.device, dtype=torch.bool)).to(dtype=hidden_tensor.dtype).view(
        1, num_envs, 1
    )
    return hidden_tensor.detach().clone() * keep


def _safe_reset_student_hidden_on_done(student_model: Any, dones: Any, num_envs: int, done_ids: list[int]) -> None:
    hidden_state = student_model.get_hidden_state() if hasattr(student_model, "get_hidden_state") else None
    strategy = "clone_mask_reset_hidden_state"
    success = False
    try:
        if hidden_state is None:
            strategy = "no_hidden_state_available"
            success = True
            return

        if not callable(getattr(student_model, "reset", None)):
            raise ValueError("A2 student model does not expose reset(hidden_state=...) for safe hidden-state replacement.")

        if isinstance(hidden_state, tuple):
            masked_hidden_state = tuple(_mask_hidden_tensor_clone(state, dones, num_envs) for state in hidden_state)
        else:
            masked_hidden_state = _mask_hidden_tensor_clone(hidden_state, dones, num_envs)
        student_model.reset(hidden_state=masked_hidden_state)
        success = True
    except TypeError as exc:
        raise ValueError(
            "A2 student model reset API does not accept hidden_state replacement; "
            "refusing unsafe in-place hidden-state mutation."
        ) from exc
    finally:
        print("[T09-E1 HIDDEN RESET]", flush=True)
        print(f"  done_count: {len(done_ids)}", flush=True)
        print(f"  done_ids: {[int(env_id) for env_id in done_ids]}", flush=True)
        print(f"  reset_strategy: {strategy}", flush=True)
        print(f"  hidden_reset_success: {success}", flush=True)


def _runtime_smoke_status(
    *,
    no_nan_inf: bool,
    rollout_steps_executed: int,
    fault_window_reached: bool,
    eval_loop_exception: bool,
) -> str:
    if eval_loop_exception:
        if fault_window_reached and no_nan_inf:
            return "partial_error_after_fault_window"
        if not no_nan_inf:
            return "fail_nonfinite"
        if rollout_steps_executed <= 0:
            return "fail_no_steps"
        return "partial_error_before_fault_window"
    if not no_nan_inf:
        return "fail_nonfinite"
    if rollout_steps_executed <= 0:
        return "fail_no_steps"
    if not fault_window_reached:
        return "partial_fault_window_not_reached"
    return "pass"


def _print_status_block(
    *,
    runtime_smoke_status: str,
    pilot_status: str,
    fault_window_reached: bool,
    rollout_steps_executed: int,
    partial_rollout_step_limit: bool,
    telemetry_interval_steps: int,
    completed_episodes: int,
    requested_episodes: int,
    run_dir: Path,
) -> None:
    print("[T09-E1 STATUS]", flush=True)
    print(f"  runtime_smoke_status: {runtime_smoke_status}", flush=True)
    print(f"  pilot_status: {pilot_status}", flush=True)
    print(f"  fault_window_reached: {fault_window_reached}", flush=True)
    print(f"  rollout_steps_executed: {rollout_steps_executed}", flush=True)
    print(f"  partial_rollout_step_limit: {partial_rollout_step_limit}", flush=True)
    print(f"  telemetry_interval_steps: {telemetry_interval_steps}", flush=True)
    print("  no_checkpoint_or_manifest_written: True", flush=True)
    if completed_episodes >= requested_episodes:
        print(f"[T09-E1] pilot complete: {repo_relative(run_dir)}")
    else:
        print(f"[T09-E1] pilot incomplete: {pilot_status}: {repo_relative(run_dir)}")


def _execute_mapping_preflight(args: argparse.Namespace, rows: dict[str, dict[str, Any]]) -> int:
    from isaaclab.app import AppLauncher

    _debug("app launcher creation start")
    app_launcher = AppLauncher(args)
    _debug("app launcher creation done")
    simulation_app = app_launcher.app

    try:
        return _execute_mapping_preflight_with_app(args, rows)
    finally:
        _debug("app close start")
        simulation_app.close()
        _debug("app close done")


def _execute_mapping_preflight_with_app(args: argparse.Namespace, rows: dict[str, dict[str, Any]]) -> int:
    import gymnasium as gym

    import isaaclab_tasks  # noqa: F401
    from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
    from isaaclab_tasks.utils import parse_env_cfg

    from p4_action_degradation_wrapper import resolve_p4_action_mapping

    row_ids = [args.ablation_id] if args.ablation_id is not None else list(ALLOWED_ROWS)
    for row_id in row_ids:
        row = rows[row_id]
        task = str(row["task"])
        env = None
        try:
            _debug(f"env config parse start row={row_id} task={task}")
            env_cfg = parse_env_cfg(task, device=args.device, num_envs=args.num_envs)
            env_cfg.seed = args.seed
            _debug(f"env config parse done row={row_id}")
            _debug(f"gym.make start row={row_id}")
            env = gym.make(task, cfg=env_cfg)
            _debug(f"gym.make done row={row_id}")
            if isinstance(env.unwrapped, DirectMARLEnv):
                env = multi_agent_to_single_agent(env)

            _debug(f"action mapping validation start row={row_id}")
            mapping = resolve_p4_action_mapping(
                env,
                target_joint=args.target_joint,
                torque_scale=args.torque_scale,
                fault_onset_step=args.fault_onset_step,
                expected_action_dim=8,
            )
            _debug(f"action mapping validation done row={row_id}")
            print("[T09-E1-B MAPPING PREFLIGHT]")
            print(f"  ablation_id: {row_id}")
            print(f"  task: {task}")
            print(f"  target_joint: {mapping.target_joint}")
            print(f"  resolved_action_index: {mapping.target_action_index}")
            print(f"  full_joint_action_order: {list(mapping.joint_names)}")
            print(f"  action_dim: {mapping.action_dim}")
            print(f"  torque_scale: {mapping.torque_scale}")
            print(f"  fault_onset_step: {mapping.fault_onset_step}")
            print("  no_policy_checkpoint_loaded: True")
            print("  no_env_step_or_evaluation_run: True")
            print("  no_checkpoint_or_manifest_written: True")
        finally:
            if env is not None:
                _debug(f"env close start row={row_id}")
                env.close()
                _debug(f"env close done row={row_id}")
    print("[T09-E1-B SUMMARY] runtime mapping preflight only")
    print("  no_training: True")
    print("  no_policy_checkpoint_loading: True")
    print("  no_evaluation_execution: True")
    print("  no_env_step: True")
    print("  no_checkpoint_writes: True")
    print("  no_observed_result_updates: True")
    return 0


def _execute_pilot(args: argparse.Namespace, rows: dict[str, dict[str, Any]]) -> int:
    from isaaclab.app import AppLauncher

    _debug("app launcher creation start")
    app_launcher = AppLauncher(args)
    _debug("app launcher creation done")
    simulation_app = app_launcher.app

    close_app = True
    try:
        return _execute_pilot_with_app(args, rows)
    except Exception:
        if args.debug_skip_app_close_on_error and (
            not getattr(args, "_t09_eval_started", False) or getattr(args, "_t09_eval_loop_error", False)
        ):
            close_app = False
            _debug("app close skipped due to debug_skip_app_close_on_error")
        raise
    finally:
        if close_app:
            _debug("app close start")
            simulation_app.close()
            _debug("app close done")


def _execute_pilot_with_app(args: argparse.Namespace, rows: dict[str, dict[str, Any]]) -> int:
    import sys as _sys

    trainers_dir = REPO_ROOT / "trainers"
    if str(trainers_dir) not in _sys.path:
        _sys.path.insert(0, str(trainers_dir))

    import gymnasium as gym
    import torch
    from rsl_rl.runners import OnPolicyRunner

    import isaaclab_tasks  # noqa: F401
    from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
    from isaaclab_tasks.manager_based.classic.ant.agents.rsl_rl_ppo_cfg import AntPPORunnerCfg
    from isaaclab_tasks.manager_based.classic.ant.agents.rsl_rl_residual_ppo_cfg import AntResidualPPORunnerCfg
    from isaaclab_tasks.utils import parse_env_cfg
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    from p4_action_degradation_wrapper import P4ActionDegradationWrapper
    from residual_action_wrapper import ResidualActionWrapper, _build_student_model

    setattr(args, "_t09_eval_started", False)
    row = rows[args.ablation_id]
    task = str(row["task"])
    _debug("env config parse start")
    env_cfg = parse_env_cfg(task, device=args.device, num_envs=args.num_envs)
    env_cfg.seed = args.seed
    _debug("env config parse done")
    _debug("gym.make start")
    env = gym.make(task, cfg=env_cfg)
    _debug("gym.make done")
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    _debug("P4 wrapper creation start")
    _debug("action mapping validation start")
    p4_wrapper = P4ActionDegradationWrapper(
        env,
        target_joint=args.target_joint,
        torque_scale=args.torque_scale,
        fault_onset_step=args.fault_onset_step,
        expected_action_dim=8,
        debug=True,
    )
    _debug("action mapping validation done")
    _debug("P4 wrapper creation done")
    env = p4_wrapper
    residual_wrapper = None

    _debug("policy checkpoint resolution start")
    checkpoint_plan = checkpoint_plan_for_row(args.ablation_id, rows)
    _debug("policy checkpoint resolution done")
    if args.policy_debug_only:
        run_dir = None
        runner_log_dir = None
        _debug("pilot run dir creation skipped policy_debug_only")
    else:
        _debug("pilot run dir creation start")
        run_dir = resolve_repo_path(args.run_root) / (
            f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{args.ablation_id}_seed{args.seed}"
        )
        run_dir.mkdir(parents=True, exist_ok=False)
        runner_log_dir = str(run_dir)
        _debug("pilot run dir creation done")

    policy_dim = None
    student_model = None
    vec_env = None
    if args.ablation_id == "A5":
        try:
            _debug("policy model construction start")
            residual_wrapper = ResidualActionWrapper(
                env,
                student_checkpoint_pointer=checkpoint_plan["student_checkpoint_pointer"],
                residual_scale=float(checkpoint_plan["residual_scale"]),
                final_action_clip=None,
                device=env.unwrapped.device,
                reset_hidden_on_done=False,
                debug=False,
            )
            _debug("policy model construction done")
            env = residual_wrapper
        except Exception as exc:
            _surface_policy_load_error(exc)
            if args.debug_skip_app_close_on_error:
                _close_env_for_debug_error(env)
            raise

    _debug("env reset start")
    vec_env = RslRlVecEnvWrapper(env, clip_actions=None)
    _debug("env reset done")
    device = vec_env.device

    try:
        if args.ablation_id == "A0":
            _debug("policy model construction start")
            agent_cfg_dict = _prepare_a0_agent_cfg_dict(AntPPORunnerCfg())
            runner = OnPolicyRunner(vec_env, agent_cfg_dict, log_dir=runner_log_dir, device=device)
            _debug("policy model construction done")
            _debug("policy checkpoint load start")
            runner.load(str(resolve_repo_path(checkpoint_plan["resolved_checkpoint_path"])))
            _debug("policy checkpoint load done")
            policy = runner.get_inference_policy(device=device)
        elif args.ablation_id == "A5":
            _debug("policy model construction start")
            agent_cfg_dict = _prepare_a5_agent_cfg_dict(AntResidualPPORunnerCfg())
            runner = OnPolicyRunner(vec_env, agent_cfg_dict, log_dir=runner_log_dir, device=device)
            _debug("policy model construction done")
            _debug("policy checkpoint load start")
            runner.load(str(resolve_repo_path(checkpoint_plan["resolved_checkpoint_path"])))
            _debug("policy checkpoint load done")
            policy = runner.get_inference_policy(device=device)
            policy_dim = getattr(residual_wrapper, "policy_dim", None)
        else:
            policy_dim = vec_env.unwrapped.observation_manager.group_obs_dim["policy"][0]
            action_dim = vec_env.unwrapped.action_manager.total_action_dim
            _debug("policy model construction start")
            student_model = _build_student_model(
                num_envs=args.num_envs,
                policy_dim=policy_dim,
                action_dim=action_dim,
                device=device,
            )
            _debug("policy model construction done")
            _debug("policy checkpoint load start")
            student_checkpoint = torch.load(
                resolve_repo_path(checkpoint_plan["resolved_checkpoint_path"]),
                weights_only=False,
                map_location=device,
            )
            if "student_state_dict" not in student_checkpoint:
                raise RuntimeError("A2 checkpoint missing student_state_dict.")
            student_model.load_state_dict(student_checkpoint["student_state_dict"], strict=True)
            student_model.eval()
            _debug("policy checkpoint load done")

            def policy(obs):
                with torch.inference_mode():
                    student_obs = obs.select("policy") if hasattr(obs, "select") else obs
                    return student_model(student_obs, stochastic_output=False)
    except Exception as exc:
        _surface_policy_load_error(exc)
        if args.debug_skip_app_close_on_error:
            _close_env_for_debug_error(vec_env)
        raise

    if args.ablation_id == "A0":
        student_model = None

    if args.policy_debug_only:
        _debug("policy_debug_only complete")
        print("[T09-E1 POLICY DEBUG] policy model construction/load succeeded")
        print("  no_env_step_or_evaluation_run: True")
        print("  no_checkpoint_or_manifest_written: True")
        _debug("env close start")
        vec_env.close()
        _debug("env close done")
        return 0

    summary: dict[str, Any] = {
        **checkpoint_plan,
        "run_dir": repo_relative(run_dir),
        "fault_profile": args.fault_profile,
        "target_joint": args.target_joint,
        "target_action_index": p4_wrapper.mapping.target_action_index,
        "joint_names": list(p4_wrapper.mapping.joint_names),
        "torque_scale": args.torque_scale,
        "fault_onset_step": args.fault_onset_step,
        "num_envs": args.num_envs,
        "episodes_requested": args.episodes,
        "policy_mode": args.policy_mode,
        "action_dim": p4_wrapper.mapping.action_dim,
        "policy_dim": policy_dim,
        "load_success": True,
        "paper_grade": False,
    }
    _write_json(run_dir / "pilot_metadata.json", summary)

    _debug("env reset start")
    obs, _ = vec_env.reset()
    _debug("env reset done")
    if student_model is not None:
        _debug("student hidden full reset start")
        student_model.reset()
        _debug("student hidden full reset done")
    episode_returns = torch.zeros(args.num_envs, device=device)
    episode_lengths = torch.zeros(args.num_envs, dtype=torch.long, device=device)
    completed = 0
    no_nan_inf = True
    rollout_steps_executed = 0
    episode_path = run_dir / "episodes.jsonl"
    max_rollout_steps = int(vec_env.max_episode_length) * (math.ceil(args.episodes / args.num_envs) + 2)
    max_rollout_steps = min(max_rollout_steps, args.max_debug_steps_per_episode * args.episodes)

    _debug("eval loop start")
    setattr(args, "_t09_eval_started", True)
    last_global_step: int | None = None
    last_dones_diagnostic: dict[str, Any] | None = None
    try:
        with episode_path.open("w", encoding="utf-8") as stream:
            current_episode_ordinal = 1
            _debug(f"episode start ordinal={current_episode_ordinal}")
            for _step in range(max_rollout_steps):
                last_global_step = _step
                trace_step = _step < DEBUG_STEP_LOG_LIMIT or (_step + 1) % args.telemetry_interval_steps == 0
                if trace_step:
                    _debug(f"action inference start global_step={_step}")
                with torch.inference_mode():
                    actions = policy(obs)
                if trace_step:
                    _debug(f"action inference done global_step={_step}")
                no_nan_inf = no_nan_inf and bool(torch.isfinite(actions).all())

                if trace_step:
                    _debug(f"env.step start global_step={_step}")
                obs, rewards, dones, extras = vec_env.step(actions)
                if trace_step:
                    _debug(f"env.step done global_step={_step}")
                no_nan_inf = no_nan_inf and bool(torch.isfinite(rewards).all())
                episode_returns += rewards
                episode_lengths += 1
                rollout_steps_executed += 1

                if rollout_steps_executed % args.telemetry_interval_steps == 0:
                    print("[T09-E1 TELEMETRY]", flush=True)
                    print(f"  rollout_steps_executed: {rollout_steps_executed}", flush=True)
                    print(f"  episodes_completed: {completed}", flush=True)
                    print(f"  episodes_requested: {args.episodes}", flush=True)
                    print(f"  fault_window_reached: {bool(p4_wrapper.fault_applied)}", flush=True)
                    print(f"  no_nan_inf: {no_nan_inf}", flush=True)
                    print(f"  telemetry_interval_steps: {args.telemetry_interval_steps}", flush=True)

                done_ids = torch.nonzero(dones > 0, as_tuple=False).flatten().tolist()
                try:
                    dones_any = bool(dones.any().item())
                except Exception:
                    dones_any = bool(done_ids)
                last_dones_diagnostic = {
                    "done_count": len(done_ids),
                    "done_ids": [int(env_id) for env_id in done_ids[:16]],
                    "dones_any": dones_any,
                    "dones_shape": list(getattr(dones, "shape", [])),
                    "extras_keys": sorted(extras.keys()) if isinstance(extras, dict) else None,
                }
                if student_model is not None and done_ids:
                    _safe_reset_student_hidden_on_done(student_model, dones, args.num_envs, done_ids)
                for env_id in done_ids:
                    record = {
                        "ablation_id": args.ablation_id,
                        "env_id": int(env_id),
                        "episode_return": float(episode_returns[env_id].item()),
                        "episode_length": int(episode_lengths[env_id].item()),
                        "termination_reason": "time_out" if _timeout_for_env(extras, env_id) else "terminated",
                        "fault_applied": bool(p4_wrapper.fault_applied),
                        "post_fault_survival": max(0, int(episode_lengths[env_id].item()) - args.fault_onset_step),
                        "no_nan_inf": no_nan_inf,
                    }
                    if residual_wrapper is not None:
                        record.update(
                            {
                                "residual_scale": float(checkpoint_plan["residual_scale"]),
                                "residual_mean_abs_delta": _tensor_to_float(residual_wrapper.last_mean_abs_delta),
                                "residual_max_abs_delta": _tensor_to_float(residual_wrapper.last_max_abs_delta),
                                "residual_saturation_ratio": _tensor_to_float(residual_wrapper.last_saturation_ratio),
                                "residual_clip_fraction": _tensor_to_float(residual_wrapper.last_clip_fraction),
                            }
                        )
                    stream.write(json.dumps(record, sort_keys=True) + "\n")
                    _debug(f"episode done ordinal={completed + 1} env_id={env_id}")
                    completed += 1
                    if completed < args.episodes:
                        current_episode_ordinal = completed + 1
                        _debug(f"episode start ordinal={current_episode_ordinal}")
                    episode_returns[env_id] = 0.0
                    episode_lengths[env_id] = 0
                    if completed >= args.episodes:
                        break
                if completed >= args.episodes:
                    break
    except Exception as exc:
        setattr(args, "_t09_eval_loop_error", True)
        fault_window_reached = bool(p4_wrapper.fault_applied)
        partial_rollout_step_limit = completed < args.episodes and rollout_steps_executed >= max_rollout_steps
        runtime_smoke_status = _runtime_smoke_status(
            no_nan_inf=no_nan_inf,
            rollout_steps_executed=rollout_steps_executed,
            fault_window_reached=fault_window_reached,
            eval_loop_exception=True,
        )
        pilot_status = "eval_loop_exception"
        _surface_eval_loop_error(
            exc,
            ablation_id=args.ablation_id,
            global_step=last_global_step,
            rollout_steps_executed=rollout_steps_executed,
            completed_episodes=completed,
            requested_episodes=args.episodes,
            fault_window_reached=fault_window_reached,
            no_nan_inf=no_nan_inf,
            last_dones_diagnostic=last_dones_diagnostic,
        )
        summary.update(
            {
                "episodes_completed": completed,
                "fault_applied": bool(p4_wrapper.fault_applied),
                "fault_window_reached": fault_window_reached,
                "rollout_steps_executed": rollout_steps_executed,
                "partial_rollout_step_limit": partial_rollout_step_limit,
                "telemetry_interval_steps": args.telemetry_interval_steps,
                "runtime_smoke_status": runtime_smoke_status,
                "pilot_status": pilot_status,
                "eval_loop_exception": True,
                "last_global_step": last_global_step,
                "last_dones_diagnostic": last_dones_diagnostic,
                "no_nan_inf": no_nan_inf,
                "episode_log": repo_relative(episode_path),
                "no_checkpoint_or_manifest_written": True,
            }
        )
        try:
            _write_json(run_dir / "pilot_summary.json", summary)
        except Exception as summary_exc:
            print(
                f"[T09-E1 ERROR] failed to write partial pilot_summary.json: "
                f"{type(summary_exc).__name__}: {summary_exc}",
                file=sys.stderr,
                flush=True,
            )
        _print_status_block(
            runtime_smoke_status=runtime_smoke_status,
            pilot_status=pilot_status,
            fault_window_reached=fault_window_reached,
            rollout_steps_executed=rollout_steps_executed,
            partial_rollout_step_limit=partial_rollout_step_limit,
            telemetry_interval_steps=args.telemetry_interval_steps,
            completed_episodes=completed,
            requested_episodes=args.episodes,
            run_dir=run_dir,
        )
        if args.debug_skip_app_close_on_error:
            _close_env_for_debug_error(vec_env)
        raise

    if completed < args.episodes:
        _debug(
            f"eval loop stopped before requested episodes completed={completed} "
            f"requested={args.episodes} max_rollout_steps={max_rollout_steps}"
        )

    fault_window_reached = bool(p4_wrapper.fault_applied)
    partial_rollout_step_limit = completed < args.episodes and rollout_steps_executed >= max_rollout_steps
    runtime_smoke_status = _runtime_smoke_status(
        no_nan_inf=no_nan_inf,
        rollout_steps_executed=rollout_steps_executed,
        fault_window_reached=fault_window_reached,
        eval_loop_exception=False,
    )

    if completed >= args.episodes:
        pilot_status = "episodes_completed"
    elif partial_rollout_step_limit:
        pilot_status = "partial_rollout_step_limit"
    else:
        pilot_status = "partial_incomplete"

    summary.update(
        {
            "episodes_completed": completed,
            "fault_applied": bool(p4_wrapper.fault_applied),
            "fault_window_reached": fault_window_reached,
            "rollout_steps_executed": rollout_steps_executed,
            "partial_rollout_step_limit": partial_rollout_step_limit,
            "telemetry_interval_steps": args.telemetry_interval_steps,
            "runtime_smoke_status": runtime_smoke_status,
            "pilot_status": pilot_status,
            "no_nan_inf": no_nan_inf,
            "episode_log": repo_relative(episode_path),
            "no_checkpoint_or_manifest_written": True,
        }
    )
    _write_json(run_dir / "pilot_summary.json", summary)
    _debug("env close start")
    vec_env.close()
    _debug("env close done")
    _print_status_block(
        runtime_smoke_status=runtime_smoke_status,
        pilot_status=pilot_status,
        fault_window_reached=fault_window_reached,
        rollout_steps_executed=rollout_steps_executed,
        partial_rollout_step_limit=partial_rollout_step_limit,
        telemetry_interval_steps=args.telemetry_interval_steps,
        completed_episodes=completed,
        requested_episodes=args.episodes,
        run_dir=run_dir,
    )
    return 0


def main() -> int:
    args = parse_args()
    traceback_timer_active = False
    try:
        validate_args(args)
        if (args.execute_mapping_preflight or args.execute_pilot) and args.debug_timeout_sec > 0:
            faulthandler.enable(file=sys.stderr)
            faulthandler.dump_traceback_later(args.debug_timeout_sec, repeat=True, file=sys.stderr)
            traceback_timer_active = True
        validate_p4_profile(args.p4_profile)
        rows = load_rows(args.matrix)
        if args.verify_profile:
            print("[T09-E1 PROFILE] P4_torque_degradation scaffold validated")
        if args.preview_mapping:
            print_mapping_preview(args)
        if args.execute_mapping_preflight:
            return _execute_mapping_preflight(args, rows)
        if not args.execute_pilot:
            print_preview(args, rows)
            print_no_runtime_summary()
            return 0
        return _execute_pilot(args, rows)
    except PilotConfigError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        print_no_runtime_summary()
        return 1
    finally:
        if traceback_timer_active:
            faulthandler.cancel_dump_traceback_later()


if __name__ == "__main__":
    sys.exit(main())
