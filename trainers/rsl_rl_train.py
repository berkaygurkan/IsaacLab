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

    command = [
        str(ISAACLAB_SH),
        "-p",
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

    result = subprocess.run(command, cwd=REPO_ROOT)
    if result.returncode != 0:
        return result.returncode

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
