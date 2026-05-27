"""Repo-owned T07-B launcher for RSL-RL student distillation."""

from __future__ import annotations

import argparse
import logging
import os
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher

REPO_ROOT = Path(__file__).resolve().parents[1]
RSL_RL_SCRIPT_DIR = REPO_ROOT / "scripts" / "reinforcement_learning" / "rsl_rl"
if str(RSL_RL_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(RSL_RL_SCRIPT_DIR))

import cli_args  # isort: skip
from rsl_rl_train import (  # isort: skip
    _arg_was_provided,
    _experiment_name,
    _load_flat_yaml,
    _repo_relative,
    _resolve_config_path,
    _run_name,
    _write_checkpoint_pointer,
)


DEFAULT_TRAIN_CONFIG = REPO_ROOT / "configs" / "train" / "student.yaml"
DEFAULT_STAGE = "student"
DEFAULT_METHOD = "rlm1_stripped"
DEFAULT_FAULT = "none"
DEFAULT_SEED = 0
DEFAULT_TASK = "Isaac-Ant-Student-v0"
DEFAULT_TEACHER_POINTER = "checkpoints/rlm1_stripped/teacher/none/seed0/latest_checkpoint.yaml"


def _resolve_repo_path(path_value: str) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def _load_stage_config(raw_args: list[str]) -> tuple[Path, dict[str, str]]:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", default=str(DEFAULT_TRAIN_CONFIG.relative_to(REPO_ROOT)))
    config_args, _ = config_parser.parse_known_args(raw_args)

    train_config = _resolve_config_path(config_args.config)
    if not train_config.exists():
        raise FileNotFoundError(f"Train config not found: {_repo_relative(train_config)}")
    return train_config, _load_flat_yaml(train_config)


def _build_parser(train_config: Path, config: dict[str, str], raw_args: list[str]) -> argparse.ArgumentParser:
    stage = config.get("stage", DEFAULT_STAGE)
    method = config.get("method", DEFAULT_METHOD)
    fault = config.get("fault", DEFAULT_FAULT)
    seed = int(config.get("seed", DEFAULT_SEED))

    parser = argparse.ArgumentParser(description="Launch T07-B student distillation with RSL-RL.")
    parser.add_argument("--config", default=str(_repo_relative(train_config)))
    parser.add_argument("--stage", default=stage)
    parser.add_argument("--method", default=method)
    parser.add_argument("--fault", default=fault)
    parser.add_argument("--task", default=config.get("task", DEFAULT_TASK))
    parser.add_argument("--seed", type=int, default=seed)
    parser.add_argument(
        "--teacher_checkpoint_pointer",
        default=config.get("teacher_checkpoint_pointer", DEFAULT_TEACHER_POINTER),
        help="Repo checkpoint pointer YAML for the frozen T06 teacher.",
    )
    parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
    parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video in steps.")
    parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings in steps.")
    parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
    parser.add_argument(
        "--agent",
        type=str,
        default="rsl_rl_distillation_cfg_entry_point",
        help="Name of the RSL-RL distillation config entry point.",
    )
    parser.add_argument("--max_iterations", type=int, default=None, help="Student distillation iterations.")
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
args_cli, hydra_args = parser.parse_known_args()

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

sys.argv = [sys.argv[0]] + hydra_args

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
from rsl_rl.runners import DistillationRunner  # noqa: E402

from isaaclab.envs import (  # noqa: E402
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict  # noqa: E402
from isaaclab.utils.io import dump_yaml  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402

import isaaclab_tasks  # noqa: E402,F401
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402


logger = logging.getLogger(__name__)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def _load_teacher_checkpoint_from_pointer(pointer_value: str) -> tuple[Path, Path]:
    pointer_path = _resolve_repo_path(pointer_value)
    if not pointer_path.is_file():
        raise FileNotFoundError(f"Teacher checkpoint pointer not found: {_repo_relative(pointer_path)}")

    pointer = _load_flat_yaml(pointer_path)
    checkpoint_value = pointer.get("checkpoint_path")
    if not checkpoint_value:
        raise ValueError(f"Teacher checkpoint pointer missing checkpoint_path: {_repo_relative(pointer_path)}")

    checkpoint_path = _resolve_repo_path(checkpoint_value)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Teacher checkpoint not found: {_repo_relative(checkpoint_path)}")
    return pointer_path, checkpoint_path


def _dim_width(group_dim: tuple[int, ...] | list[tuple[int, ...]]) -> int:
    if isinstance(group_dim, tuple) and len(group_dim) == 1:
        return group_dim[0]
    raise ValueError(f"Expected a concatenated 1D observation group, got dimension metadata: {group_dim}")


def _action_dim(env) -> int:
    if hasattr(env.unwrapped, "action_manager"):
        return env.unwrapped.action_manager.total_action_dim
    return gym.spaces.flatdim(env.unwrapped.single_action_space)


def _validate_observation_contract(env) -> tuple[int, int, int]:
    observation_manager = env.unwrapped.observation_manager
    active_terms = observation_manager.active_terms
    group_dims = observation_manager.group_obs_dim

    if "policy" not in active_terms:
        raise ValueError("Student observation contract missing 'policy' group.")
    if "teacher_policy" not in active_terms:
        raise ValueError("Student observation contract missing 'teacher_policy' group.")

    policy_terms = set(active_terms["policy"])
    teacher_terms = set(active_terms["teacher_policy"])
    required_teacher_terms = {"true_fault_state", "base_velocity", "contacts"}

    if "true_fault_state" in policy_terms:
        raise ValueError("Invalid T07-B contract: 'policy' group exposes true_fault_state to the student.")

    missing_teacher_terms = required_teacher_terms - teacher_terms
    if missing_teacher_terms:
        raise ValueError(
            "Invalid T07-B contract: 'teacher_policy' group missing privileged terms: "
            f"{sorted(missing_teacher_terms)}"
        )

    policy_dim = _dim_width(group_dims["policy"])
    teacher_policy_dim = _dim_width(group_dims["teacher_policy"])
    action_dim = _action_dim(env)

    print("[INFO] T07-B observation contract:")
    print(f"[INFO]   policy terms: {active_terms['policy']}")
    print(f"[INFO]   teacher_policy terms: {active_terms['teacher_policy']}")
    print(f"[INFO]   policy dim: {policy_dim}")
    print(f"[INFO]   teacher_policy dim: {teacher_policy_dim}")
    print(f"[INFO]   action dim: {action_dim}")
    return policy_dim, teacher_policy_dim, action_dim


def _validate_teacher_checkpoint_shape(checkpoint_path: Path, teacher_obs_dim: int, action_dim: int) -> None:
    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location="cpu")
    actor_state = checkpoint.get("actor_state_dict")
    if actor_state is None:
        raise ValueError(f"Teacher checkpoint has no actor_state_dict: {_repo_relative(checkpoint_path)}")

    input_weight = actor_state.get("mlp.0.weight")
    output_weight = actor_state.get("mlp.6.weight")
    std_param = actor_state.get("distribution.std_param")
    if input_weight is None or output_weight is None or std_param is None:
        raise ValueError("Teacher checkpoint actor_state_dict does not match the expected Ant MLP actor layout.")

    if input_weight.shape[1] != teacher_obs_dim:
        raise ValueError(
            f"Teacher checkpoint input dim {input_weight.shape[1]} does not match teacher_policy dim {teacher_obs_dim}."
        )
    if output_weight.shape[0] != action_dim or std_param.shape[0] != action_dim:
        raise ValueError(
            f"Teacher checkpoint action dim does not match env action dim {action_dim}: "
            f"mlp.6.weight={tuple(output_weight.shape)}, distribution.std_param={tuple(std_param.shape)}"
        )

    print("[INFO] T07-B teacher checkpoint compatibility:")
    print(f"[INFO]   actor input dim: {input_weight.shape[1]}")
    print(f"[INFO]   actor action dim: {output_weight.shape[0]}")
    print(f"[INFO]   action std dim: {std_param.shape[0]}")


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations

    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

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

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors
    else:
        logger.warning("IO descriptors are only supported for manager based RL environments.")

    env_cfg.log_dir = log_dir
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    _, teacher_obs_dim, action_dim = _validate_observation_contract(env)
    pointer_path, teacher_checkpoint = _load_teacher_checkpoint_from_pointer(args_cli.teacher_checkpoint_pointer)
    print(f"[INFO] T07-B teacher pointer: {_repo_relative(pointer_path)}")
    print(f"[INFO] T07-B teacher checkpoint: {_repo_relative(teacher_checkpoint)}")
    _validate_teacher_checkpoint_shape(teacher_checkpoint, teacher_obs_dim, action_dim)

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

    start_time = time.time()
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    if agent_cfg.class_name != "DistillationRunner":
        raise ValueError(f"T07-B requires DistillationRunner, got: {agent_cfg.class_name}")

    runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    runner.add_git_repo_to_log(__file__)
    print(f"[INFO] Loading frozen teacher from: {_repo_relative(teacher_checkpoint)}")
    runner.load(str(teacher_checkpoint), load_cfg={"teacher": True, "iteration": False}, strict=True)

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
