#!/usr/bin/env python3
"""T10-IL-02 Ant velocity-task preflight.

Default behavior is dry-run registry/config validation. Runtime environment
construction, reset, step, and P2 attach checks require ``--execute_smoke``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TASK = "Isaac-Ant-Velocity-Flat-v0"
DEFAULT_TEACHER_TASK = "Isaac-Ant-Teacher-Velocity-Flat-v0"
DEFAULT_TARGET_JOINT = "front_left_foot"
DEFAULT_EXPECTED_ACTION_DIM = 8
ANT_INIT = REPO_ROOT / "source" / "isaaclab_tasks" / "isaaclab_tasks" / "manager_based" / "classic" / "ant" / "__init__.py"
ANT_VELOCITY_CFG = (
    REPO_ROOT
    / "source"
    / "isaaclab_tasks"
    / "isaaclab_tasks"
    / "manager_based"
    / "classic"
    / "ant"
    / "ant_velocity_env_cfg.py"
)


class PreflightError(ValueError):
    """Raised for invalid Ant velocity-task preflight state."""


def _prepare_import_path() -> None:
    for relative in ("source/isaaclab_tasks", "source/isaaclab", "source/isaaclab_assets", "source/isaaclab_rl"):
        path = REPO_ROOT / relative
        if path.is_dir() and str(path) not in sys.path:
            sys.path.insert(0, str(path))


def build_parser(*, add_app_launcher_args: bool = False) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preflight the T10 Ant velocity-tracking task scaffold.")
    parser.add_argument("--dry_run", action="store_true", help="Registry/config validation only. This is the default.")
    parser.add_argument("--execute_smoke", action="store_true", help="Launch Isaac Sim for one reset/step and P2 attach smoke.")
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--teacher_task", default=DEFAULT_TEACHER_TASK)
    parser.add_argument("--target_joint", default=DEFAULT_TARGET_JOINT)
    parser.add_argument("--expected_action_dim", type=int, default=DEFAULT_EXPECTED_ACTION_DIM)
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--num_steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
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
    if pre_args.execute_smoke:
        try:
            parser = build_parser(add_app_launcher_args=True)
        except Exception as exc:
            print("[T10-IL-02b RUNTIME]")
            print("  runtime_mode: execute_smoke")
            print("  app_launcher_created: false")
            print("  task_registration_loaded: false")
            print("  gym_spec_found: false")
            print(f"  app_launcher_import_failed: {type(exc).__name__}: {exc}")
            print("  no_training: true")
            print("  no_checkpoint_or_manifest_written: true")
            raise
        args, _ = parser.parse_known_args()
        return args
    args, _ = pre_parser.parse_known_args()
    return args


def _public_cfg_names(obj: Any) -> list[str]:
    names = []
    for name in dir(obj):
        if name.startswith("_"):
            continue
        try:
            value = getattr(obj, name)
        except Exception:
            continue
        if callable(value):
            continue
        names.append(name)
    return sorted(names)


def _load_env_cfg(task_name: str) -> Any:
    _prepare_import_path()
    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    return load_cfg_from_registry(task_name, "env_cfg_entry_point")


def _registry_spec(task_name: str) -> Any:
    _prepare_import_path()
    import gymnasium as gym
    import isaaclab_tasks  # noqa: F401

    try:
        return gym.spec(task_name)
    except Exception as exc:
        raise PreflightError(f"task is not registered: {task_name}") from exc


def _static_validate_velocity_cfg(task_name: str, *, teacher: bool, reason: Exception) -> dict[str, Any]:
    init_text = ANT_INIT.read_text(encoding="utf-8")
    cfg_text = ANT_VELOCITY_CFG.read_text(encoding="utf-8")
    if f'id="{task_name}"' not in init_text:
        raise PreflightError(f"{task_name} not found in static Ant registry file: {ANT_INIT}") from reason
    expected_cfg = "AntTeacherVelocityFlatEnvCfg" if teacher else "AntVelocityFlatEnvCfg"
    if expected_cfg not in cfg_text:
        raise PreflightError(f"{expected_cfg} not found in {ANT_VELOCITY_CFG}") from reason
    required_tokens = (
        "UniformVelocityCommandCfg",
        "base_velocity",
        "lin_vel_x=(1.0, 1.0)",
        "lin_vel_y=(0.0, 0.0)",
        "ang_vel_z=(0.0, 0.0)",
        "velocity_commands",
        "generated_commands",
        "track_lin_vel_xy_exp",
        "track_ang_vel_z_exp",
        "ActionsCfg",
    )
    missing = [token for token in required_tokens if token not in cfg_text]
    if missing:
        raise PreflightError(f"static velocity cfg missing tokens: {missing}") from reason
    if "progress = RewTerm" in cfg_text or "move_to_target = RewTerm" in cfg_text:
        raise PreflightError("static velocity cfg still defines progress-style reward terms") from reason
    if teacher and "true_fault_state = ObsTerm" not in cfg_text:
        raise PreflightError("teacher velocity cfg missing true_fault_state") from reason

    return {
        "task": task_name,
        "entry_point": "isaaclab.envs:ManagerBasedRLEnv",
        "env_cfg_entry_point": f"static:{expected_cfg}",
        "rsl_rl_cfg_entry_point": "static:AntPPORunnerCfg",
        "command_names": ["base_velocity"],
        "velocity_command_ranges": {
            "lin_vel_x": (1.0, 1.0),
            "lin_vel_y": (0.0, 0.0),
            "ang_vel_z": (0.0, 0.0),
            "heading": (0.0, 0.0),
        },
        "observation_names": (
            [
                "base_height",
                "true_fault_state",
                "base_velocity",
                "base_ang_vel",
                "velocity_commands",
                "base_yaw_roll",
                "base_up_proj",
                "joint_pos_norm",
                "joint_vel_rel",
                "contacts",
                "actions",
            ]
            if teacher
            else [
                "base_height",
                "base_lin_vel",
                "base_ang_vel",
                "velocity_commands",
                "base_yaw_roll",
                "base_up_proj",
                "joint_pos_norm",
                "joint_vel_rel",
                "feet_body_forces",
                "actions",
            ]
        ),
        "reward_names": [
            "track_lin_vel_xy_exp",
            "track_ang_vel_z_exp",
            "alive",
            "upright",
            "action_l2",
            "energy",
            "joint_pos_limits",
        ],
        "action_names": ["joint_effort"],
        "teacher": teacher,
        "validation_mode": f"static_fallback_due_to_{type(reason).__name__}",
    }


def validate_velocity_cfg(task_name: str, *, teacher: bool = False, allow_static_fallback: bool = True) -> dict[str, Any]:
    try:
        spec = _registry_spec(task_name)
        cfg = _load_env_cfg(task_name)
    except ModuleNotFoundError as exc:
        if not allow_static_fallback:
            raise
        return _static_validate_velocity_cfg(task_name, teacher=teacher, reason=exc)

    command_names = _public_cfg_names(getattr(cfg, "commands", None))
    observation_names = _public_cfg_names(getattr(getattr(cfg, "observations", None), "policy", None))
    reward_names = _public_cfg_names(getattr(cfg, "rewards", None))
    action_names = _public_cfg_names(getattr(cfg, "actions", None))

    if "base_velocity" not in command_names:
        raise PreflightError(f"{task_name} missing commands.base_velocity.")
    base_velocity = cfg.commands.base_velocity
    ranges = getattr(base_velocity, "ranges", None)
    if ranges is None:
        raise PreflightError(f"{task_name} commands.base_velocity has no ranges.")
    if getattr(ranges, "lin_vel_x", None) is None:
        raise PreflightError(f"{task_name} commands.base_velocity missing lin_vel_x range.")
    if "velocity_commands" not in observation_names:
        raise PreflightError(f"{task_name} missing observations.policy.velocity_commands.")
    if "track_lin_vel_xy_exp" not in reward_names:
        raise PreflightError(f"{task_name} missing rewards.track_lin_vel_xy_exp.")
    if "track_ang_vel_z_exp" not in reward_names:
        raise PreflightError(f"{task_name} missing rewards.track_ang_vel_z_exp.")
    if "progress" in reward_names or "move_to_target" in reward_names:
        raise PreflightError(f"{task_name} still exposes progress-style reward terms: {reward_names}.")
    if "joint_effort" not in action_names:
        raise PreflightError(f"{task_name} must preserve actions.joint_effort.")
    if teacher and "true_fault_state" not in observation_names:
        raise PreflightError(f"{task_name} teacher variant missing true_fault_state.")
    if not teacher and "true_fault_state" in observation_names:
        raise PreflightError(f"{task_name} deployment-facing policy exposes true_fault_state.")

    return {
        "task": task_name,
        "entry_point": spec.entry_point,
        "env_cfg_entry_point": spec.kwargs.get("env_cfg_entry_point"),
        "rsl_rl_cfg_entry_point": spec.kwargs.get("rsl_rl_cfg_entry_point"),
        "command_names": command_names,
        "velocity_command_ranges": {
            "lin_vel_x": getattr(ranges, "lin_vel_x", None),
            "lin_vel_y": getattr(ranges, "lin_vel_y", None),
            "ang_vel_z": getattr(ranges, "ang_vel_z", None),
            "heading": getattr(ranges, "heading", None),
        },
        "observation_names": observation_names,
        "reward_names": reward_names,
        "action_names": action_names,
        "teacher": teacher,
    }


def print_cfg_report(report: dict[str, Any]) -> None:
    print("[T10-IL-02 Ant velocity task preflight]")
    print(f"  task: {report['task']}")
    print(f"  entry_point: {report['entry_point']}")
    print(f"  env_cfg_entry_point: {report['env_cfg_entry_point']}")
    print(f"  rsl_rl_cfg_entry_point: {report['rsl_rl_cfg_entry_point']}")
    print(f"  commands: {report['command_names']}")
    print(f"  velocity_command_ranges: {report['velocity_command_ranges']}")
    print(f"  observations.policy: {report['observation_names']}")
    print(f"  rewards: {report['reward_names']}")
    print(f"  actions: {report['action_names']}")
    print(f"  teacher_variant: {report['teacher']}")
    if report.get("validation_mode"):
        print(f"  validation_mode: {report['validation_mode']}")


def execute_smoke(args: argparse.Namespace) -> int:
    from isaaclab.app import AppLauncher

    print("[T10-IL-02b RUNTIME]")
    print("  runtime_mode: execute_smoke", flush=True)
    print("  app_launcher_creation_start: true", flush=True)
    app_launcher = AppLauncher(args)
    print("  app_launcher_created: true", flush=True)
    simulation_app = app_launcher.app
    print("  task_registration_loaded: false", flush=True)
    print("  gym_spec_found: false", flush=True)
    print("  reset_ok: false", flush=True)
    print("  step_ok: false", flush=True)
    env = None
    p2_wrapper = None
    try:
        trainers_dir = REPO_ROOT / "trainers"
        if str(trainers_dir) not in sys.path:
            sys.path.insert(0, str(trainers_dir))

        import gymnasium as gym
        import torch

        import isaaclab_tasks  # noqa: F401
        from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
        from isaaclab_tasks.utils import parse_env_cfg

        from p2_joint_lock_training_wrapper import P2JointLockActionMaskWrapper

        print("  task_registration_loaded: true", flush=True)
        spec = gym.spec(args.task)
        print("  gym_spec_found: true", flush=True)
        print(f"  env_cfg_entry_point: {spec.kwargs.get('env_cfg_entry_point')}", flush=True)

        print("[T10-IL-02b] runtime config validation start", flush=True)
        report = validate_velocity_cfg(args.task, teacher=args.task == args.teacher_task, allow_static_fallback=False)
        print_cfg_report(report)
        print("[T10-IL-02b] runtime config validation done", flush=True)

        print("[T10-IL-02b] env config parse start", flush=True)
        cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
        cfg.seed = args.seed
        print("[T10-IL-02b] env config parse done", flush=True)

        print("[T10-IL-02] gym.make start", flush=True)
        env = gym.make(args.task, cfg=cfg)
        print("[T10-IL-02] gym.make done", flush=True)

        command_manager = getattr(env.unwrapped, "command_manager", None)
        print(f"  command_manager_exists: {command_manager is not None}", flush=True)
        if command_manager is None:
            raise PreflightError("command_manager missing after env construction.")
        command_terms = getattr(command_manager, "_terms", {})
        active_command_terms = list(command_terms.keys()) if isinstance(command_terms, dict) else []
        print(f"  active_command_terms: {active_command_terms}", flush=True)
        command = command_manager.get_command("base_velocity")
        if command.shape[-1] < 3:
            raise PreflightError(f"base_velocity command shape must expose vx/vy/yaw, got {tuple(command.shape)}.")
        print(f"[T10-IL-02] base_velocity command shape: {tuple(command.shape)}", flush=True)

        obs_manager = getattr(env.unwrapped, "observation_manager", None)
        active_terms = getattr(obs_manager, "active_terms", {}) if obs_manager is not None else {}
        policy_terms = list(active_terms.get("policy", [])) if isinstance(active_terms, dict) else []
        print(f"  velocity_commands_observation_exists: {'velocity_commands' in policy_terms}", flush=True)
        print(f"  observation_terms.policy: {policy_terms}", flush=True)
        reward_manager = getattr(env.unwrapped, "reward_manager", None)
        active_reward_terms = getattr(reward_manager, "active_terms", None) if reward_manager is not None else None
        if isinstance(active_reward_terms, dict):
            reward_terms = list(active_reward_terms.keys())
        elif active_reward_terms is not None:
            reward_terms = list(active_reward_terms)
        else:
            reward_terms = []
        print(f"  reward_terms: {reward_terms}", flush=True)
        action_manager = getattr(env.unwrapped, "action_manager", None)
        action_dim = int(getattr(action_manager, "total_action_dim", -1)) if action_manager is not None else -1
        print(f"  action_dim: {action_dim}", flush=True)
        if action_dim != args.expected_action_dim:
            raise PreflightError(f"expected action_dim={args.expected_action_dim}, got {action_dim}.")

        p2_wrapper = P2JointLockActionMaskWrapper(
            env,
            target_joint=args.target_joint,
            fault_onset_step=250,
            fault_onset_mode="fixed",
            expected_action_dim=args.expected_action_dim,
            requested_semantics="simulation_joint_state_override_lock",
            allow_fallback=False,
            velocity_override=0.0,
            debug=True,
        )
        env = p2_wrapper
        if p2_wrapper.mapping.semantics != "simulation_joint_state_override_lock":
            raise PreflightError(f"P2 fallback occurred unexpectedly: {p2_wrapper.mapping.semantics}.")

        print("[T10-IL-02b] env reset start", flush=True)
        obs, _ = env.reset()
        print("  reset_ok: true", flush=True)
        action = torch.zeros((args.num_envs, args.expected_action_dim), device=env.unwrapped.device)
        print("[T10-IL-02b] step smoke start", flush=True)
        for step in range(args.num_steps):
            env.step(action)
            if step < 3 or step == args.num_steps - 1:
                print(
                    f"  smoke_step: {step + 1} p2_fault_applied: {env.last_fault_applied}",
                    flush=True,
                )
        print("  step_ok: true", flush=True)
        print("[T10-IL-02] reset/one-step smoke: pass", flush=True)
        print("  P2/fallback_used: False", flush=True)
        print(f"  target_joint: {args.target_joint}", flush=True)
        print(f"  target_action_index: {p2_wrapper.mapping.target_action_index}", flush=True)
        print(f"  actual_semantics: {p2_wrapper.mapping.semantics}", flush=True)
        print(f"  full_joint_action_order: {list(p2_wrapper.mapping.joint_names)}", flush=True)
        print(f"  observation_type: {type(obs).__name__}", flush=True)
        print("  no_training: true", flush=True)
        print("  no_checkpoint_or_manifest_written: true", flush=True)
        return 0
    except Exception as exc:
        import traceback

        print("[T10-IL-02b ERROR] runtime smoke failed", flush=True)
        traceback.print_exc()
        print(f"  exception_type: {type(exc).__name__}", flush=True)
        print(f"  exception_message: {exc}", flush=True)
        raise
    finally:
        if env is not None:
            print("[T10-IL-02b] env close start", flush=True)
            env.close()
            print("[T10-IL-02b] env close done", flush=True)
        print("[T10-IL-02b] app close start", flush=True)
        simulation_app.close()
        print("[T10-IL-02b] app close done", flush=True)


def main() -> int:
    args = parse_args()
    if args.dry_run and args.execute_smoke:
        raise PreflightError("Use --dry_run without --execute_smoke.")
    if args.num_envs <= 0:
        raise PreflightError("--num_envs must be positive.")
    if args.num_steps <= 0:
        raise PreflightError("--num_steps must be positive.")
    if args.expected_action_dim <= 0:
        raise PreflightError("--expected_action_dim must be positive.")

    if args.execute_smoke:
        return execute_smoke(args)

    base_report = validate_velocity_cfg(args.task, teacher=False, allow_static_fallback=True)
    teacher_report = validate_velocity_cfg(args.teacher_task, teacher=True, allow_static_fallback=True)
    print_cfg_report(base_report)
    print_cfg_report(teacher_report)
    print("[T10-IL-02] dry_run: no Isaac Sim, no env reset/step, no training, no checkpoint writes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
