"""Repo-owned launcher for conference-stage RSL-RL PPO runs."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN_CONFIG = REPO_ROOT / "configs" / "train" / "healthy_baseline.yaml"
ISAACLAB_SH = REPO_ROOT / "isaaclab.sh"
UPSTREAM_TRAIN = REPO_ROOT / "scripts" / "reinforcement_learning" / "rsl_rl" / "train.py"
P2_TRAINING_WRAPPER = REPO_ROOT / "trainers" / "p2_joint_lock_training_wrapper.py"
EVALUATORS_DIR = REPO_ROOT / "evaluators"
if str(EVALUATORS_DIR) not in sys.path:
    sys.path.insert(0, str(EVALUATORS_DIR))

from t18r_control_timing import add_control_timing_args

DEFAULT_STAGE = "healthy_baseline"
DEFAULT_METHOD = "rlm1_stripped"
DEFAULT_FAULT = "none"
DEFAULT_SEED = 0
DEFAULT_TASK = "Isaac-Ant-v0"


def _strip_comment(value: str) -> str:
    return value.split("#", 1)[0].strip()


def _load_flat_yaml(path: Path) -> dict[str, str]:
    """Load the tiny flat YAML config without introducing a config framework."""
    values: dict[str, str] = {}
    if not path.exists():
        return values

    for line in path.read_text(encoding="utf-8").splitlines():
        line = _strip_comment(line)
        if not line or ":" not in line or line.startswith(" "):
            continue
        key, value = line.split(":", 1)
        values[key.strip()] = value.strip().strip("\"'")
    return values


def _arg_was_provided(option: str, raw_args: list[str]) -> bool:
    return any(arg == option or arg.startswith(f"{option}=") for arg in raw_args)


def _resolve_config_path(path_value: str) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def _run_name(stage: str, method: str, fault: str, seed: int) -> str:
    return f"{stage}__{method}__{fault}__seed{seed}"


def _experiment_name(stage: str, method: str, fault: str) -> str:
    return f"{stage}__{method}__{fault}"


def _repo_relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(resolved)


def _checkpoint_step(path: Path) -> int:
    stem = path.stem
    if stem.startswith("model_"):
        try:
            return int(stem.removeprefix("model_"))
        except ValueError:
            return -1
    return -1


def _find_latest_checkpoint(experiment_name: str, run_name: str) -> tuple[Path, Path] | None:
    log_root = REPO_ROOT / "logs" / "rsl_rl" / experiment_name
    if not log_root.exists():
        return None

    run_dirs = [path for path in log_root.glob(f"*_{run_name}") if path.is_dir()]
    run_dirs.sort(key=lambda path: path.stat().st_mtime, reverse=True)

    for run_dir in run_dirs:
        checkpoints = [path for path in run_dir.glob("model_*.pt") if path.is_file()]
        checkpoints.sort(key=lambda path: (_checkpoint_step(path), path.stat().st_mtime))
        if checkpoints:
            return run_dir, checkpoints[-1]
    return None


def _write_checkpoint_pointer(
    *,
    stage: str,
    method: str,
    fault: str,
    seed: int,
    task: str,
    experiment_name: str,
    run_name: str,
) -> Path | None:
    latest = _find_latest_checkpoint(experiment_name, run_name)
    if latest is None:
        print("[WARN] No RSL-RL model_*.pt checkpoint found; checkpoint pointer was not written.")
        return None

    run_dir, checkpoint = latest
    pointer_dir = REPO_ROOT / "checkpoints" / method / stage / fault / f"seed{seed}"
    pointer_dir.mkdir(parents=True, exist_ok=True)
    pointer_path = pointer_dir / "latest_checkpoint.yaml"
    pointer_path.write_text(
        "\n".join(
            [
                f"stage: {stage}",
                f"method: {method}",
                f"fault: {fault}",
                f"seed: {seed}",
                f"task: {task}",
                f"experiment_name: {experiment_name}",
                f"run_name: {run_name}",
                f"log_dir: {_repo_relative(run_dir)}",
                f"checkpoint_path: {_repo_relative(checkpoint)}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"[INFO] Wrote checkpoint pointer: {_repo_relative(pointer_path)}")
    return pointer_path


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    raw_args = sys.argv[1:]
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", default=str(DEFAULT_TRAIN_CONFIG.relative_to(REPO_ROOT)))
    config_args, _ = config_parser.parse_known_args(raw_args)

    train_config = _resolve_config_path(config_args.config)
    if not train_config.exists():
        raise FileNotFoundError(f"Train config not found: {_repo_relative(train_config)}")
    config = _load_flat_yaml(train_config)

    stage = config.get("stage", DEFAULT_STAGE)
    method = config.get("method", DEFAULT_METHOD)
    fault = config.get("fault", DEFAULT_FAULT)
    seed = int(config.get("seed", DEFAULT_SEED))

    parser = argparse.ArgumentParser(description="Launch repo-owned conference-stage PPO runs with RSL-RL.")
    parser.add_argument("--config", default=str(_repo_relative(train_config)))
    parser.add_argument("--stage", default=stage)
    parser.add_argument("--method", default=method)
    parser.add_argument("--fault", default=fault)
    parser.add_argument("--task", default=config.get("task", DEFAULT_TASK))
    parser.add_argument("--seed", type=int, default=seed)
    parser.add_argument("--experiment_name", default=None)
    parser.add_argument("--run_name", default=None)
    parser.add_argument(
        "--skip_checkpoint_pointer",
        action="store_true",
        help="Do not write/update repo checkpoint pointer YAML after training.",
    )
    parser.add_argument(
        "--enable_p2_joint_lock",
        action="store_true",
        help="Route training through the repo-owned P2 joint-lock runtime hook.",
    )
    parser.add_argument("--p2_fault_config", default="configs/fault/joint_lock/p2_locked_joint.yaml")
    parser.add_argument("--p2_target_joint", default="front_left_foot")
    parser.add_argument("--p2_target_joint_mode", default="single", choices=("single", "random_per_env"))
    parser.add_argument(
        "--p2_supported_target_joints",
        default="",
        help="Optional comma-separated joint-name subset for random_per_env P2 sampling.",
    )
    parser.add_argument("--p2_fault_onset_step", type=int, default=50)
    parser.add_argument("--p2_fault_onset_mode", default="fixed", choices=("fixed", "random_uniform"))
    parser.add_argument("--p2_fault_onset_step_min", type=int, default=30)
    parser.add_argument("--p2_fault_onset_step_max", type=int, default=150)
    parser.add_argument("--p2_expected_action_dim", type=int, default=8)
    parser.add_argument("--p2_locked_action_value", type=float, default=0.0)
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
    add_control_timing_args(parser)
    args, passthrough = parser.parse_known_args()

    identity_overridden = any(
        _arg_was_provided(option, raw_args) for option in ("--stage", "--method", "--fault")
    )
    seed_overridden = _arg_was_provided("--seed", raw_args)

    if args.experiment_name is None:
        if config.get("experiment_name") and not identity_overridden:
            args.experiment_name = config["experiment_name"]
        else:
            args.experiment_name = _experiment_name(args.stage, args.method, args.fault)

    if args.run_name is None:
        if config.get("run_name") and not identity_overridden and not seed_overridden:
            args.run_name = config["run_name"]
        else:
            args.run_name = _run_name(args.stage, args.method, args.fault, args.seed)

    return args, passthrough


def main() -> int:
    args, passthrough = parse_args()
    if args.enable_p2_joint_lock and not args.skip_checkpoint_pointer:
        raise ValueError("P2 joint-lock training requires --skip_checkpoint_pointer; freeze canonical checkpoints manually.")
    timing_requested = any(
        value is not None
        for value in (
            args.control_frequency_hz,
            args.sim_dt,
            args.decimation,
            args.episode_length_s,
            args.require_control_frequency_hz,
        )
    ) or args.t18r_pg500_timing or args.require_t18r_pg500_timing

    upstream_args = [
        str(UPSTREAM_TRAIN.relative_to(REPO_ROOT)),
        "--task",
        args.task,
        "--seed",
        str(args.seed),
        "--experiment_name",
        args.experiment_name,
        "--run_name",
        args.run_name,
        *passthrough,
    ]
    if args.enable_p2_joint_lock or timing_requested:
        command = [
            str(ISAACLAB_SH),
            "-p",
            str(P2_TRAINING_WRAPPER.relative_to(REPO_ROOT)),
            "--p2_fault_config",
            args.p2_fault_config,
            "--p2_target_joint",
            args.p2_target_joint,
            "--p2_target_joint_mode",
            args.p2_target_joint_mode,
            "--p2_supported_target_joints",
            args.p2_supported_target_joints,
            "--p2_fault_onset_step",
            str(args.p2_fault_onset_step),
            "--p2_fault_onset_mode",
            args.p2_fault_onset_mode,
            "--p2_fault_onset_step_min",
            str(args.p2_fault_onset_step_min),
            "--p2_fault_onset_step_max",
            str(args.p2_fault_onset_step_max),
            "--p2_expected_action_dim",
            str(args.p2_expected_action_dim),
            "--p2_locked_action_value",
            str(args.p2_locked_action_value),
            "--p2_kp",
            str(args.p2_kp),
            "--p2_kd",
            str(args.p2_kd),
            "--p2_action_clip",
            str(args.p2_action_clip),
            "--p2_requested_semantics",
            args.p2_requested_semantics,
            "--p2_velocity_override",
            str(args.p2_velocity_override),
            "--p2_task",
            args.task,
            "--",
            *upstream_args,
        ]
        if not args.enable_p2_joint_lock:
            command.insert(command.index("--"), "--p2_disable_fault_wrapper")
        if args.t18r_pg500_timing:
            command.insert(command.index("--"), "--t18r_pg500_timing")
        if args.require_t18r_pg500_timing:
            command.insert(command.index("--"), "--require_t18r_pg500_timing")
        for flag, value in (
            ("--control_frequency_hz", args.control_frequency_hz),
            ("--sim_dt", args.sim_dt),
            ("--decimation", args.decimation),
            ("--episode_length_s", args.episode_length_s),
            ("--require_control_frequency_hz", args.require_control_frequency_hz),
        ):
            if value is not None:
                insert_at = command.index("--")
                command[insert_at:insert_at] = [flag, str(value)]
        if args.p2_allow_fallback:
            command.insert(command.index("--p2_task"), "--p2_allow_fallback")
        print("[T09-R2i] P2/timing runtime hook requested.")
        print(f"  P2_runtime_hook_enabled: {args.enable_p2_joint_lock}")
        print(f"  timing_hook_enabled: {timing_requested}")
        print(f"  fault_profile: P2_locked_joint")
        print(f"  target_joint: {args.p2_target_joint}")
        print(f"  target_joint_mode: {args.p2_target_joint_mode}")
        print(f"  supported_target_joints_request: {args.p2_supported_target_joints or 'all_resolved_joints'}")
        print(f"  P2_fault_onset_mode: {args.p2_fault_onset_mode}")
        print(f"  P2_fault_onset_step_min: {args.p2_fault_onset_step_min}")
        print(f"  P2_fault_onset_step_max: {args.p2_fault_onset_step_max}")
        print(f"  fault_onset_step: {args.p2_fault_onset_step}")
        print(f"  per_env_onset_randomization: {args.p2_fault_onset_mode == 'random_uniform'}")
        print(f"  requested_semantics: {args.p2_requested_semantics}")
        print("  fallback_semantics: pd_position_hold_surrogate")
        print(f"  allow_fallback: {args.p2_allow_fallback}")
        print(f"  velocity_override: {args.p2_velocity_override}")
        print(f"  kp: {args.p2_kp}")
        print(f"  kd: {args.p2_kd}")
        print(f"  action_clip: {args.p2_action_clip}")
    else:
        command = [
            str(ISAACLAB_SH),
            "-p",
            *upstream_args,
        ]

    result = subprocess.run(command, cwd=REPO_ROOT)
    if result.returncode != 0:
        return result.returncode

    if args.skip_checkpoint_pointer:
        print("[INFO] Skipped checkpoint pointer update by request.")
        return 0

    _write_checkpoint_pointer(
        stage=args.stage,
        method=args.method,
        fault=args.fault,
        seed=args.seed,
        task=args.task,
        experiment_name=args.experiment_name,
        run_name=args.run_name,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
