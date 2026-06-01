"""Repo-owned T08-B launcher for residual PPO on top of the frozen T07 student."""

from __future__ import annotations

import argparse
import logging
import os
import platform
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from isaaclab.app import AppLauncher

REPO_ROOT = Path(__file__).resolve().parents[1]
RSL_RL_SCRIPT_DIR = REPO_ROOT / "scripts" / "reinforcement_learning" / "rsl_rl"
if str(RSL_RL_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(RSL_RL_SCRIPT_DIR))

import cli_args  # isort: skip
from residual_action_wrapper import ResidualActionWrapper  # isort: skip
from rsl_rl_train import (  # isort: skip
    _arg_was_provided,
    _experiment_name,
    _load_flat_yaml,
    _repo_relative,
    _resolve_config_path,
    _run_name,
    _write_checkpoint_pointer,
)


DEFAULT_TRAIN_CONFIG = REPO_ROOT / "configs" / "train" / "residual.yaml"
DEFAULT_STAGE = "residual"
DEFAULT_METHOD = "rlm1_stripped"
DEFAULT_FAULT = "none"
DEFAULT_SEED = 0
DEFAULT_TASK = "Isaac-Ant-Student-v0"
DEFAULT_STUDENT_POINTER = "checkpoints/rlm1_stripped/student/none/seed0/latest_checkpoint.yaml"


def _load_stage_config(raw_args: list[str]) -> tuple[Path, dict[str, str]]:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", default=str(DEFAULT_TRAIN_CONFIG.relative_to(REPO_ROOT)))
    config_args, _ = config_parser.parse_known_args(raw_args)

    train_config = _resolve_config_path(config_args.config)
    if not train_config.exists():
        raise FileNotFoundError(f"Train config not found: {_repo_relative(train_config)}")
    return train_config, _load_flat_yaml(train_config)


def _optional_float(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _bool_from_config(value: str | None, *, default: bool = False) -> bool:
    if value in (None, ""):
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _build_parser(train_config: Path, config: dict[str, str], raw_args: list[str]) -> argparse.ArgumentParser:
    stage = config.get("stage", DEFAULT_STAGE)
    method = config.get("method", DEFAULT_METHOD)
    fault = config.get("fault", DEFAULT_FAULT)
    seed = int(config.get("seed", DEFAULT_SEED))

    parser = argparse.ArgumentParser(description="Launch T08-B residual PPO training with RSL-RL.")
    parser.add_argument("--config", default=str(_repo_relative(train_config)))
    parser.add_argument("--stage", default=stage)
    parser.add_argument("--method", default=method)
    parser.add_argument("--fault", default=fault)
    parser.add_argument("--task", default=config.get("task", DEFAULT_TASK))
    parser.add_argument("--seed", type=int, default=seed)
    parser.add_argument(
        "--student_checkpoint_pointer",
        default=config.get("student_checkpoint_pointer", DEFAULT_STUDENT_POINTER),
        help="Repo checkpoint pointer YAML for the frozen T07 student.",
    )
    parser.add_argument("--residual_scale", type=float, default=float(config.get("residual_scale", 0.1)))
    parser.add_argument("--final_action_clip", type=float, default=_optional_float(config.get("final_action_clip")))
    parser.add_argument(
        "--reset_hidden_on_done",
        action="store_true",
        default=_bool_from_config(config.get("reset_hidden_on_done"), default=False),
        help="Opt into wrapper-owned per-env hidden-state masking for the frozen student.",
    )
    parser.add_argument("--debug_wrapper", action="store_true", default=False)
    parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
    parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video in steps.")
    parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings in steps.")
    parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
    parser.add_argument("--max_iterations", type=int, default=None, help="Residual PPO training iterations.")
    parser.add_argument("--distributed", action="store_true", default=False, help="Run with multiple GPUs or nodes.")
    parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors.")
    parser.add_argument(
        "--ray-proc-id", "-rid", type=int, default=None, help="Automatically configured by Ray integration."
    )
    cli_args.add_rsl_rl_args(parser)
    AppLauncher.add_app_launcher_args(parser)

    identity_overridden = any(_arg_was_provided(option, raw_args) for option in ("--stage", "--method", "--fault"))
    seed_overridden = _arg_was_provided("--seed", raw_args)
    parser.set_defaults(
        _config_experiment_name=config.get("experiment_name"),
        _config_run_name=config.get("run_name"),
        _identity_overridden=identity_overridden,
        _seed_overridden=seed_overridden,
    )
    return parser


raw_args = sys.argv[1:]
train_config_path, stage_config = _load_stage_config(raw_args)
parser = _build_parser(train_config_path, stage_config, raw_args)
args_cli, _ = parser.parse_known_args()

if args_cli.experiment_name is None:
    if args_cli._config_experiment_name and not args_cli._identity_overridden:
        args_cli.experiment_name = args_cli._config_experiment_name
    else:
        args_cli.experiment_name = _experiment_name(args_cli.stage, args_cli.method, args_cli.fault)

if args_cli.run_name is None:
    if args_cli._config_run_name and not args_cli._identity_overridden and not args_cli._seed_overridden:
        args_cli.run_name = args_cli._config_run_name
    else:
        args_cli.run_name = _run_name(args_cli.stage, args_cli.method, args_cli.fault, args_cli.seed)

if args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import importlib.metadata as metadata  # noqa: E402

from packaging import version  # noqa: E402

RSL_RL_VERSION = "3.0.1"
installed_version = metadata.version("rsl-rl-lib")
if version.parse(installed_version) < version.parse(RSL_RL_VERSION):
    if platform.system() == "Windows":
        cmd = [r".\isaaclab.bat", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    else:
        cmd = ["./isaaclab.sh", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    print(
        f"Please install the correct version of RSL-RL.\nExisting version is: '{installed_version}'"
        f" and required version is: '{RSL_RL_VERSION}'.\nTo install the correct version, run:"
        f"\n\n\t{' '.join(cmd)}\n"
    )
    exit(1)

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent  # noqa: E402
from isaaclab.utils.dict import print_dict  # noqa: E402
from isaaclab.utils.io import dump_yaml  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402

import isaaclab_tasks  # noqa: E402,F401
from isaaclab_tasks.manager_based.classic.ant.agents.rsl_rl_residual_ppo_cfg import (  # noqa: E402
    AntResidualPPORunnerCfg,
)
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg  # noqa: E402


logger = logging.getLogger(__name__)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def _dim_width(group_dim: tuple[int, ...] | list[tuple[int, ...]]) -> int:
    if isinstance(group_dim, tuple) and len(group_dim) == 1:
        return group_dim[0]
    raise ValueError(f"Expected a concatenated 1D observation group, got dimension metadata: {group_dim}")


def _action_dim(env) -> int:
    if hasattr(env.unwrapped, "action_manager"):
        return env.unwrapped.action_manager.total_action_dim
    return gym.spaces.flatdim(env.unwrapped.single_action_space)


def _set_deterministic_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"[INFO] T08.6 deterministic seed: {seed}")


def _validate_residual_contract(env, agent_cfg: AntResidualPPORunnerCfg) -> tuple[int, int]:
    expected_obs_groups = {"actor": ["policy"], "critic": ["policy"]}
    if agent_cfg.obs_groups != expected_obs_groups:
        raise ValueError(f"T08-B residual PPO requires obs_groups={expected_obs_groups}, got {agent_cfg.obs_groups}")

    observation_manager = env.unwrapped.observation_manager
    active_terms = observation_manager.active_terms
    group_dims = observation_manager.group_obs_dim

    if "policy" not in active_terms:
        raise ValueError("T08-B residual PPO requires a student-safe 'policy' observation group.")
    if "teacher_policy" not in active_terms:
        raise ValueError("T08-B residual PPO expects the T07-A 'teacher_policy' group to exist but remain unused.")
    if "true_fault_state" in active_terms["policy"]:
        raise ValueError("Invalid T08-B contract: 'policy' group exposes true_fault_state.")
    if any("teacher_policy" in groups for groups in agent_cfg.obs_groups.values()):
        raise ValueError("Invalid T08-B contract: residual PPO must not map teacher_policy to actor or critic.")

    policy_dim = _dim_width(group_dims["policy"])
    action_dim = _action_dim(env)
    if policy_dim != 60:
        raise ValueError(f"Expected T07 student policy dim 60, got {policy_dim}.")
    if action_dim != 8:
        raise ValueError(f"Expected Ant action dim 8, got {action_dim}.")

    print("[INFO] T08-B residual observation contract:")
    print(f"[INFO]   obs_groups: {agent_cfg.obs_groups}")
    print(f"[INFO]   policy terms: {active_terms['policy']}")
    print(f"[INFO]   policy dim: {policy_dim}")
    print("[INFO]   teacher_policy present but unused by residual PPO.")
    print(f"[INFO]   action dim: {action_dim}")
    return policy_dim, action_dim


def _print_runtime_summary(
    *,
    task: str,
    obs_dim: int,
    action_dim: int,
    residual_scale: float,
    final_action_clip: float | None,
) -> None:
    print("[INFO] T08.6 runtime summary:")
    print(f"  task: {task}")
    print(f"  obs_dim: {obs_dim}")
    print(f"  action_dim: {action_dim}")
    print(f"  residual_scale: {residual_scale}")
    print("  recurrent_student: True")
    print("  residual_policy_recurrent: False")
    print("  privileged_obs_used: False")
    print(f"  final_action_clip: {final_action_clip}")


def main() -> None:
    agent_cfg = AntResidualPPORunnerCfg()
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    agent_cfg.max_iterations = args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    if args_cli.distributed and args_cli.device is not None and "cpu" in args_cli.device:
        raise ValueError("Distributed training is not supported when using CPU device. Please use a GPU device.")

    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    _set_deterministic_seed(agent_cfg.seed)

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)
    resume_path = None
    if agent_cfg.resume:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    if hasattr(env_cfg, "export_io_descriptors"):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors
    else:
        logger.warning("IO descriptors are only supported for manager based RL environments.")

    env_cfg.log_dir = log_dir
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    policy_dim, action_dim = _validate_residual_contract(env, agent_cfg)
    _print_runtime_summary(
        task=args_cli.task,
        obs_dim=policy_dim,
        action_dim=action_dim,
        residual_scale=args_cli.residual_scale,
        final_action_clip=args_cli.final_action_clip,
    )
    print("[INFO] T08.6 residual diagnostics enabled")
    if args_cli.final_action_clip is not None:
        print(f"[INFO] T08.6 final action clipping threshold: {args_cli.final_action_clip}")

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = ResidualActionWrapper(
        env,
        student_checkpoint_pointer=args_cli.student_checkpoint_pointer,
        residual_scale=args_cli.residual_scale,
        final_action_clip=args_cli.final_action_clip,
        device=env.unwrapped.device,
        reset_hidden_on_done=args_cli.reset_hidden_on_done,
        debug=args_cli.debug_wrapper,
    )
    print(f"[INFO] T08-B frozen student pointer: {_repo_relative(env.student_pointer_path)}")
    print(f"[INFO] T08-B frozen student checkpoint: {_repo_relative(env.student_checkpoint_path)}")
    print(f"[INFO] T08-B residual_scale: {args_cli.residual_scale}")
    print(f"[INFO] T08-B final_action_clip: {args_cli.final_action_clip}")
    print(f"[INFO] T08-B reset_hidden_on_done: {args_cli.reset_hidden_on_done}")

    start_time = time.time()
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    if agent_cfg.class_name != "OnPolicyRunner":
        raise ValueError(f"T08-B requires OnPolicyRunner, got: {agent_cfg.class_name}")

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    runner.add_git_repo_to_log(__file__)
    if resume_path is not None:
        print(f"[INFO] T08.6 resume checkpoint: {resume_path}")
        runner.load(resume_path)

    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    print(f"Training time: {round(time.time() - start_time, 2)} seconds")

    env.close()

    _write_checkpoint_pointer(
        stage=args_cli.stage,
        method=args_cli.method,
        fault=args_cli.fault,
        seed=args_cli.seed,
        task=args_cli.task,
        experiment_name=agent_cfg.experiment_name,
        run_name=agent_cfg.run_name,
    )


if __name__ == "__main__":
    main()
    simulation_app.close()
