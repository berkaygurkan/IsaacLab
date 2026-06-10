#!/usr/bin/env python3
"""T09-R2i P2 joint-lock runtime preflight.

Default behavior is dry-run only. Runtime inspection requires
``--execute_preflight`` and performs no training, checkpoint writing, freezing,
or observed-result update.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FAULT_CONFIG = "configs/fault/joint_lock/p2_locked_joint.yaml"
DEFAULT_TASK = "Isaac-Ant-Teacher-v0"
DEFAULT_TARGET_JOINT = "front_left_foot"
DEFAULT_FAULT_ONSET_STEP = 50
DEFAULT_FAULT_ONSET_MODE = "random_uniform"
DEFAULT_FAULT_ONSET_STEP_MIN = 30
DEFAULT_FAULT_ONSET_STEP_MAX = 150
DEFAULT_NUM_ENVS = 8
DEFAULT_SEED = 0


def repo_relative(path_value: str | Path) -> str:
    path = Path(path_value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def build_parser(add_app_launcher_args: bool = False) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Dry-run or execute the T09-R2i P2 joint-lock preflight.")
    parser.add_argument("--dry_run", action="store_true", help="Preview only. This is the default behavior.")
    parser.add_argument(
        "--execute_preflight",
        action="store_true",
        help="Launch Isaac Sim, construct the env, attach P2 wrapper, reset, and run a tiny action smoke.",
    )
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--fault_config", default=DEFAULT_FAULT_CONFIG)
    parser.add_argument("--target_joint", default=DEFAULT_TARGET_JOINT)
    parser.add_argument("--fault_onset_step", type=int, default=DEFAULT_FAULT_ONSET_STEP)
    parser.add_argument("--fault_onset_mode", default=DEFAULT_FAULT_ONSET_MODE, choices=("fixed", "random_uniform"))
    parser.add_argument("--fault_onset_step_min", type=int, default=DEFAULT_FAULT_ONSET_STEP_MIN)
    parser.add_argument("--fault_onset_step_max", type=int, default=DEFAULT_FAULT_ONSET_STEP_MAX)
    parser.add_argument("--num_envs", type=int, default=DEFAULT_NUM_ENVS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--expected_action_dim", type=int, default=8)
    parser.add_argument("--locked_action_value", type=float, default=0.0)
    parser.add_argument("--p2_kp", type=float, default=4.0)
    parser.add_argument("--p2_kd", type=float, default=0.4)
    parser.add_argument("--p2_action_clip", type=float, default=1.0)
    parser.add_argument(
        "--p2_requested_semantics",
        default="simulation_joint_state_override_lock",
        choices=("simulation_joint_state_override_lock", "pd_position_hold_surrogate"),
    )
    parser.add_argument("--p2_allow_fallback", action="store_true")
    parser.add_argument("--p2_velocity_override", type=float, default=0.0)
    parser.add_argument("--p2_velocity_tolerance", type=float, default=1.0e-5)
    parser.add_argument("--post_onset_steps", type=int, default=3)
    if not add_app_launcher_args:
        parser.add_argument("--device", default=None)
        parser.add_argument("--headless", action="store_true")
    if add_app_launcher_args:
        from isaaclab.app import AppLauncher

        AppLauncher.add_app_launcher_args(parser)
    return parser


def parse_args() -> argparse.Namespace:
    pre_parser = build_parser(add_app_launcher_args=False)
    pre_args, _ = pre_parser.parse_known_args()
    if pre_args.execute_preflight:
        parser = build_parser(add_app_launcher_args=True)
        args, _ = parser.parse_known_args()
        return args
    return pre_args


def _print_static_summary(args: argparse.Namespace) -> None:
    print("[T09-R2i P2 PREFLIGHT DRY-RUN]")
    print("  no_isaac_sim_launched: True")
    print("  no_training: True")
    print("  no_checkpoint_writes: True")
    print("  no_observed_manifest_update: True")
    print("  P2_runtime_hook_enabled: planned")
    print("  fault_profile: P2_locked_joint")
    print(f"  fault_config: {args.fault_config}")
    print(f"  task: {args.task}")
    print(f"  target_joint: {args.target_joint}")
    print(f"  P2_fault_onset_mode: {args.fault_onset_mode}")
    print(f"  P2_fault_onset_step_min: {args.fault_onset_step_min}")
    print(f"  P2_fault_onset_step_max: {args.fault_onset_step_max}")
    if args.fault_onset_mode == "fixed":
        print(f"  P2_fixed_fault_onset_step: {args.fault_onset_step}")
    else:
        print(f"  P2_fixed_fault_onset_step: ignored_for_random_uniform_{args.fault_onset_step}")
    print(f"  per_env_onset_randomization: {args.fault_onset_mode == 'random_uniform'}")
    print(f"  requested_semantics: {args.p2_requested_semantics}")
    print("  fallback_semantics: pd_position_hold_surrogate")
    print(f"  allow_fallback: {args.p2_allow_fallback}")
    print(f"  velocity_override: {args.p2_velocity_override}")
    print(f"  velocity_tolerance: {args.p2_velocity_tolerance}")
    print(f"  P2_kp: {args.p2_kp}")
    print(f"  P2_kd: {args.p2_kd}")
    print(f"  P2_action_clip: {args.p2_action_clip}")
    print(
        "  execute_preflight_command: "
        "python evaluators/preflight_t09_p2_joint_lock.py --execute_preflight --headless "
        f"--task {args.task} --fault_config {args.fault_config} --target_joint {args.target_joint} "
        f"--fault_onset_mode {args.fault_onset_mode} --fault_onset_step {args.fault_onset_step} "
        f"--fault_onset_step_min {args.fault_onset_step_min} "
        f"--fault_onset_step_max {args.fault_onset_step_max} --num_envs {args.num_envs} "
        f"--p2_requested_semantics {args.p2_requested_semantics} "
        f"--p2_velocity_override {args.p2_velocity_override} "
        f"--p2_velocity_tolerance {args.p2_velocity_tolerance} "
        f"--p2_kp {args.p2_kp} --p2_kd {args.p2_kd} --p2_action_clip {args.p2_action_clip}"
        + (" --p2_allow_fallback" if args.p2_allow_fallback else "")
    )


def _execute_preflight(args: argparse.Namespace) -> int:
    from isaaclab.app import AppLauncher

    print("[T09-R2i P2 PREFLIGHT] app launcher creation start", flush=True)
    app_launcher = AppLauncher(args)
    print("[T09-R2i P2 PREFLIGHT] app launcher creation done", flush=True)
    simulation_app = app_launcher.app
    try:
        return _execute_preflight_with_app(args)
    finally:
        print("[T09-R2i P2 PREFLIGHT] app close start", flush=True)
        simulation_app.close()
        print("[T09-R2i P2 PREFLIGHT] app close done", flush=True)


def _execute_preflight_with_app(args: argparse.Namespace) -> int:
    trainers_dir = REPO_ROOT / "trainers"
    if str(trainers_dir) not in sys.path:
        sys.path.insert(0, str(trainers_dir))

    import gymnasium as gym
    import torch

    import isaaclab_tasks  # noqa: F401
    from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
    from isaaclab_tasks.utils import parse_env_cfg

    from p2_joint_lock_training_wrapper import P2JointLockActionMaskWrapper

    fault_config = REPO_ROOT / args.fault_config
    if not fault_config.is_file():
        raise FileNotFoundError(f"P2 fault config not found: {repo_relative(fault_config)}")

    env = None
    try:
        print("[T09-R2i P2 PREFLIGHT] env config parse start", flush=True)
        env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
        env_cfg.seed = args.seed
        print("[T09-R2i P2 PREFLIGHT] env config parse done", flush=True)

        print("[T09-R2i P2 PREFLIGHT] gym.make start", flush=True)
        env = gym.make(args.task, cfg=env_cfg)
        print("[T09-R2i P2 PREFLIGHT] gym.make done", flush=True)
        if isinstance(env.unwrapped, DirectMARLEnv):
            env = multi_agent_to_single_agent(env)

        print("[T09-R2i P2 PREFLIGHT] P2 wrapper attach start", flush=True)
        env = P2JointLockActionMaskWrapper(
            env,
            target_joint=args.target_joint,
            fault_onset_step=args.fault_onset_step,
            fault_onset_mode=args.fault_onset_mode,
            fault_onset_step_min=args.fault_onset_step_min,
            fault_onset_step_max=args.fault_onset_step_max,
            expected_action_dim=args.expected_action_dim,
            locked_action_value=args.locked_action_value,
            kp=args.p2_kp,
            kd=args.p2_kd,
            action_clip=args.p2_action_clip,
            velocity_override=args.p2_velocity_override,
            requested_semantics=args.p2_requested_semantics,
            allow_fallback=args.p2_allow_fallback,
            debug=True,
        )
        print("[T09-R2i P2 PREFLIGHT] P2 wrapper attach done", flush=True)

        print("[T09-R2i P2 PREFLIGHT] env reset start", flush=True)
        env.reset()
        print("[T09-R2i P2 PREFLIGHT] env reset done", flush=True)

        action = torch.zeros((args.num_envs, env.mapping.action_dim), device=env.unwrapped.device)
        action[:, env.mapping.target_action_index] = 0.5
        if args.fault_onset_mode == "random_uniform":
            total_steps = args.fault_onset_step_max + max(1, args.post_onset_steps)
        else:
            total_steps = args.fault_onset_step + max(1, args.post_onset_steps)
        print("[T09-R2i P2 PREFLIGHT] step smoke start", flush=True)
        for step in range(total_steps):
            env.step(action)
            if step < 3 or step == args.fault_onset_step or step == args.fault_onset_step_max or step == total_steps - 1:
                print(
                    "[T09-R2i P2 PREFLIGHT] "
                    f"step={step + 1} fault_applied={env.last_fault_applied} "
                    f"target_index={env.mapping.target_action_index}",
                    flush=True,
                )
        print("[T09-R2i P2 PREFLIGHT] step smoke done", flush=True)

        print("[T09-R2i P2 PREFLIGHT SUMMARY]")
        print("  P2_runtime_hook_enabled: True")
        print("  fault_profile: P2_locked_joint")
        print(f"  target_joint: {env.mapping.target_joint}")
        print(f"  resolved_action_index: {env.mapping.target_action_index}")
        print(f"  full_joint_action_order: {list(env.mapping.joint_names)}")
        print(f"  action_dim: {env.mapping.action_dim}")
        print(f"  P2_fault_onset_mode: {env.mapping.fault_onset_mode}")
        print(f"  P2_fault_onset_step_min: {env.mapping.fault_onset_step_min}")
        print(f"  P2_fault_onset_step_max: {env.mapping.fault_onset_step_max}")
        if env.mapping.fault_onset_mode == "fixed":
            print(f"  P2_fixed_fault_onset_step: {env.mapping.fault_onset_step}")
        else:
            print(f"  P2_fixed_fault_onset_step: ignored_for_random_uniform_{env.mapping.fault_onset_step}")
        print(f"  per_env_onset_randomization: {env.per_env_onset_randomization}")
        onset_steps = env.per_env_fault_onset_step
        if onset_steps is None:
            print("  per_env_fault_onset_steps: NA")
            print("  onset_step_mean: NA")
            print("  onset_step_min: NA")
            print("  onset_step_max: NA")
        else:
            onset_mean, onset_min, onset_max = env._onset_stats()
            print(f"  per_env_fault_onset_steps: {onset_steps.detach().cpu().tolist()}")
            print(f"  onset_step_mean: {onset_mean}")
            print(f"  onset_step_min: {onset_min}")
            print(f"  onset_step_max: {onset_max}")
        print(f"  requested_semantics: {env.mapping.requested_semantics}")
        print(f"  actual_semantics: {env.mapping.semantics}")
        print(f"  fallback_semantics: {env.mapping.fallback_semantics}")
        print(f"  allow_fallback: {env.mapping.allow_fallback}")
        print(f"  robot_articulation_found: {env.mapping.robot_articulation_found}")
        print(f"  joint_state_read_available: {env.mapping.joint_position_readable and env.mapping.joint_velocity_readable}")
        print(f"  joint_state_write_api_found: {env.mapping.joint_state_write_api_found}")
        print(f"  target_joint_id: {env.mapping.target_joint_id}")
        print(f"  velocity_override: {env.mapping.velocity_override}")
        print(f"  lock_capture_position_available: {env.lock_capture_position_available}")
        locked_position = env.locked_joint_position
        if locked_position is None:
            print("  q_lock: NA")
        else:
            print(f"  q_lock: {locked_position.detach().cpu().tolist()}")
        print(f"  lock_logic_activated: {env.ever_fault_applied}")
        print(f"  selected_joint_override_applied: {_selected_joint_override_applied(env)}")
        print(f"  post_step_override_applied: {env.post_step_override_applied}")
        print(
            "  target_joint_velocity_after_override_abs_max: "
            f"{env.target_joint_velocity_after_override_abs_max}"
        )
        velocity_abs_max = env.target_joint_velocity_after_override_abs_max
        velocity_forced = (
            velocity_abs_max is not None
            and velocity_abs_max <= abs(env.mapping.velocity_override) + args.p2_velocity_tolerance
        )
        print(f"  target_joint_velocity_forced_near_override: {velocity_forced}")
        print(f"  fallback_used: {env.fallback_used}")
        print(f"  fallback_reason: {env.fallback_reason}")
        print("  no_training: True")
        print("  no_checkpoint_or_manifest_written: True")
        return 0
    finally:
        if env is not None:
            print("[T09-R2i P2 PREFLIGHT] env close start", flush=True)
            env.close()
            print("[T09-R2i P2 PREFLIGHT] env close done", flush=True)


def _selected_joint_override_applied(env: Any) -> bool:
    import torch

    before = getattr(env, "last_target_action_before", None)
    after = getattr(env, "last_target_action_after", None)
    if before is None or after is None:
        return False
    return bool(torch.any(torch.abs(before - after) > 1.0e-6).item())


def main() -> int:
    args = parse_args()
    if not args.execute_preflight:
        _print_static_summary(args)
        return 0
    return _execute_preflight(args)


if __name__ == "__main__":
    raise SystemExit(main())
