#!/usr/bin/env python3
"""A2 history student behavior-distillation scaffold for T10 P2 velocity data.

This is an offline supervised trainer. It builds deployment-safe history windows
from ``student_obs`` only and fits a small local model to the teacher action at
the current timestep. It does not launch Isaac Sim, run RL, train residuals, or
touch task/P2 code.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_PATH = "papers/conference/datasets/t10_teacher_gap_p2_velocity_seed0/dataset.npz"
DEFAULT_CONFIG_PATH = "configs/train/a2_student_history_distill.yaml"
DEFAULT_OUTPUT_DIR = "papers/conference/results/t10_a2_student_history_distill_smoke"
EXPECTED_INPUT_DIM = 61
EXPECTED_ACTION_DIM = 8
DEFAULT_HISTORY_LEN = 16
DEFAULT_SMOKE_EPOCHS = 3
DEFAULT_BATCH_SIZE = 4096
DEFAULT_SEED = 0
DEFAULT_VAL_FRACTION = 0.1
torch: Any | None = None
nn: Any | None = None
DataLoader: Any | None = None
TensorDataset: Any | None = None


class A2HistoryDistillError(ValueError):
    """Raised for invalid A2 history distillation configuration or data."""


def require_torch() -> tuple[Any, Any, Any, Any]:
    global DataLoader, TensorDataset, nn, torch
    if torch is None:
        try:
            import torch as torch_module
            from torch import nn as nn_module
            from torch.utils.data import DataLoader as data_loader_class
            from torch.utils.data import TensorDataset as tensor_dataset_class
        except ModuleNotFoundError as exc:
            raise A2HistoryDistillError(
                "PyTorch is required for A2 history distillation training. Activate the isaaclab environment."
            ) from exc
        torch = torch_module
        nn = nn_module
        DataLoader = data_loader_class
        TensorDataset = tensor_dataset_class
    return torch, nn, DataLoader, TensorDataset


def build_history_student_mlp(history_len: int, input_dim: int, action_dim: int) -> Any:
    _, nn_module, _, _ = require_torch()

    class HistoryStudentMLP(nn_module.Module):
        """Small local flatten-history MLP for A2 smoke distillation."""

        def __init__(self) -> None:
            super().__init__()
            self.history_len = history_len
            self.input_dim = input_dim
            self.net = nn_module.Sequential(
                nn_module.Flatten(start_dim=1),
                nn_module.Linear(history_len * input_dim, 512),
                nn_module.ELU(),
                nn_module.Linear(512, 256),
                nn_module.ELU(),
                nn_module.Linear(256, 128),
                nn_module.ELU(),
                nn_module.Linear(128, action_dim),
            )

        def forward(self, obs_history: Any) -> Any:
            if obs_history.ndim != 3:
                raise A2HistoryDistillError(f"expected [batch, history, obs_dim], got {tuple(obs_history.shape)}")
            return self.net(obs_history)

    return HistoryStudentMLP()


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline A2 history student behavior-distillation trainer.")
    parser.add_argument("--dataset_path", default=DEFAULT_DATASET_PATH)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--history_len", type=int, default=DEFAULT_HISTORY_LEN)
    parser.add_argument("--max_epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--smoke", action="store_true", help="Run the small smoke-training pass.")
    return parser


def set_seed(seed: int) -> None:
    torch_module, _, _, _ = require_torch()
    random.seed(seed)
    np.random.seed(seed)
    torch_module.manual_seed(seed)
    if torch_module.cuda.is_available():
        torch_module.cuda.manual_seed_all(seed)


def select_device(device_arg: str) -> torch.device:
    torch_module, _, _, _ = require_torch()
    if device_arg == "auto":
        return torch_module.device("cuda" if torch_module.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch_module.cuda.is_available():
        raise A2HistoryDistillError("--device cuda requested but CUDA is not available.")
    return torch_module.device(device_arg)


def validate_args(args: argparse.Namespace) -> None:
    dataset_path = resolve_repo_path(args.dataset_path)
    if not dataset_path.is_file():
        raise A2HistoryDistillError(f"dataset_path does not exist: {repo_relative(dataset_path)}")
    config_path = resolve_repo_path(args.config)
    if not config_path.is_file():
        raise A2HistoryDistillError(f"config does not exist: {repo_relative(config_path)}")
    if args.history_len < 1:
        raise A2HistoryDistillError("--history_len must be >= 1.")
    if args.batch_size <= 0:
        raise A2HistoryDistillError("--batch_size must be > 0.")
    if args.seed < 0:
        raise A2HistoryDistillError("--seed must be non-negative.")
    if args.max_epochs is not None and args.max_epochs <= 0:
        raise A2HistoryDistillError("--max_epochs must be > 0 when provided.")


def check_finite_array(name: str, array: np.ndarray) -> None:
    if not np.isfinite(array).all():
        raise A2HistoryDistillError(f"{name} contains NaN or Inf.")


def assert_finite_tensor(name: str, tensor: torch.Tensor) -> None:
    torch_module, _, _, _ = require_torch()
    if not torch_module.isfinite(tensor).all():
        raise A2HistoryDistillError(f"{name} contains NaN or Inf.")


def load_history_source_arrays(dataset_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    data = np.load(dataset_path)
    input_key = "student_obs"
    target_key = "teacher_action"
    required_keys = (input_key, target_key, "done", "episode_id")
    if input_key != "student_obs" or input_key == "teacher_obs":
        raise A2HistoryDistillError("A2 history model input must be student_obs and must never be teacher_obs.")
    if "true_fault_state" in required_keys:
        raise A2HistoryDistillError("A2 history model must not use true_fault_state.")
    missing = [key for key in required_keys if key not in data.files]
    if missing:
        raise A2HistoryDistillError(f"dataset missing required arrays: {missing}")

    student_obs = np.asarray(data[input_key], dtype=np.float32)
    teacher_action = np.asarray(data[target_key], dtype=np.float32)
    done = np.asarray(data["done"], dtype=bool)
    episode_id = np.asarray(data["episode_id"])
    if student_obs.ndim != 3 or student_obs.shape[-1] != EXPECTED_INPUT_DIM:
        raise A2HistoryDistillError(f"student_obs expected shape [T, N, {EXPECTED_INPUT_DIM}], got {student_obs.shape}.")
    if teacher_action.ndim != 3 or teacher_action.shape[-1] != EXPECTED_ACTION_DIM:
        raise A2HistoryDistillError(
            f"teacher_action expected shape [T, N, {EXPECTED_ACTION_DIM}], got {teacher_action.shape}."
        )
    if done.shape != student_obs.shape[:2]:
        raise A2HistoryDistillError(f"done expected shape {student_obs.shape[:2]}, got {done.shape}.")
    if episode_id.shape != student_obs.shape[:2]:
        raise A2HistoryDistillError(f"episode_id expected shape {student_obs.shape[:2]}, got {episode_id.shape}.")
    if student_obs.shape[:2] != teacher_action.shape[:2]:
        raise A2HistoryDistillError(
            f"student_obs and teacher_action time/env dimensions differ: {student_obs.shape} vs {teacher_action.shape}."
        )
    check_finite_array("student_obs", student_obs)
    check_finite_array("teacher_action", teacher_action)
    if np.issubdtype(episode_id.dtype, np.number):
        check_finite_array("episode_id", episode_id)

    metadata = {
        "input_key": "student_obs_history",
        "target_key": target_key,
        "source_student_obs_shape": [int(dim) for dim in student_obs.shape],
        "source_teacher_action_shape": [int(dim) for dim in teacher_action.shape],
        "source_done_shape": [int(dim) for dim in done.shape],
        "source_episode_id_shape": [int(dim) for dim in episode_id.shape],
        "input_dim": EXPECTED_INPUT_DIM,
        "action_dim": EXPECTED_ACTION_DIM,
        "teacher_obs_used_as_input": False,
        "true_fault_state_used": False,
    }
    return student_obs, teacher_action, done, episode_id, metadata


def build_history_windows(
    student_obs: np.ndarray,
    teacher_action: np.ndarray,
    done: np.ndarray,
    episode_id: np.ndarray,
    *,
    history_len: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    if history_len < 1:
        raise A2HistoryDistillError("history_len must be >= 1.")
    time_steps, num_envs, input_dim = student_obs.shape
    if history_len > time_steps:
        raise A2HistoryDistillError(f"history_len {history_len} exceeds dataset time dimension {time_steps}.")

    candidate_count = (time_steps - history_len + 1) * num_envs
    valid_mask = np.zeros((time_steps - history_len + 1, num_envs), dtype=bool)
    rejected_cross_episode_windows = 0
    rejected_done_crossing_windows = 0

    for env_id in range(num_envs):
        for target_t in range(history_len - 1, time_steps):
            start_t = target_t - history_len + 1
            window_episode_id = episode_id[start_t : target_t + 1, env_id]
            crosses_episode = bool(np.any(window_episode_id != window_episode_id[0]))
            done_crossing = bool(np.any(done[start_t:target_t, env_id]))
            if crosses_episode:
                rejected_cross_episode_windows += 1
            if done_crossing:
                rejected_done_crossing_windows += 1
            if not crosses_episode and not done_crossing:
                valid_mask[start_t, env_id] = True

    valid_windows = int(valid_mask.sum())
    if valid_windows <= 0:
        raise A2HistoryDistillError("no valid history windows were produced.")

    inputs = np.empty((valid_windows, history_len, input_dim), dtype=np.float32)
    targets = np.empty((valid_windows, teacher_action.shape[-1]), dtype=np.float32)
    sample_idx = 0
    for env_id in range(num_envs):
        for target_t in range(history_len - 1, time_steps):
            start_t = target_t - history_len + 1
            if not valid_mask[start_t, env_id]:
                continue
            inputs[sample_idx] = student_obs[start_t : target_t + 1, env_id, :]
            targets[sample_idx] = teacher_action[target_t, env_id, :]
            sample_idx += 1
    if sample_idx != valid_windows:
        raise A2HistoryDistillError(f"internal window count mismatch: filled {sample_idx}, expected {valid_windows}.")
    check_finite_array("history_inputs", inputs)
    check_finite_array("history_targets", targets)

    counts = {
        "total_candidate_windows": int(candidate_count),
        "valid_windows": int(valid_windows),
        "rejected_cross_episode_windows": int(rejected_cross_episode_windows),
        "rejected_done_crossing_windows": int(rejected_done_crossing_windows),
        "rejected_total_windows": int(candidate_count - valid_windows),
    }
    return inputs, targets, counts


def split_tensors(
    obs_history: np.ndarray,
    actions: np.ndarray,
    *,
    seed: int,
    val_fraction: float = DEFAULT_VAL_FRACTION,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    num_samples = obs_history.shape[0]
    if num_samples < 2:
        raise A2HistoryDistillError("dataset must contain at least two valid history samples.")
    generator = np.random.default_rng(seed)
    indices = generator.permutation(num_samples)
    val_count = max(1, int(round(num_samples * val_fraction)))
    val_count = min(val_count, num_samples - 1)
    val_indices = indices[:val_count]
    train_indices = indices[val_count:]
    train_obs = torch.from_numpy(obs_history[train_indices].copy())
    train_actions = torch.from_numpy(actions[train_indices].copy())
    val_obs = torch.from_numpy(obs_history[val_indices].copy())
    val_actions = torch.from_numpy(actions[val_indices].copy())
    return train_obs, train_actions, val_obs, val_actions


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_mse = 0.0
    total_mae = 0.0
    total_samples = 0
    with torch.inference_mode():
        for obs_history, target in loader:
            obs_history = obs_history.to(device)
            target = target.to(device)
            assert_finite_tensor("eval_obs_history", obs_history)
            assert_finite_tensor("eval_target", target)
            pred = model(obs_history)
            assert_finite_tensor("eval_predictions", pred)
            diff = pred - target
            mse = diff.square().mean(dim=1)
            mae = diff.abs().mean(dim=1)
            assert_finite_tensor("eval_mse", mse)
            assert_finite_tensor("eval_mae", mae)
            batch_size = obs_history.shape[0]
            total_mse += float(mse.sum().detach().cpu().item())
            total_mae += float(mae.sum().detach().cpu().item())
            total_samples += int(batch_size)
    if total_samples == 0:
        raise A2HistoryDistillError("evaluation loader produced zero samples.")
    return total_mse / total_samples, total_mae / total_samples


def finite_metric_value(path: str, value: Any) -> None:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, int):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise A2HistoryDistillError(f"saved metric {path} is not finite: {value}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            finite_metric_value(f"{path}[{index}]", item)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            finite_metric_value(f"{path}.{key}", item)
        return
    raise A2HistoryDistillError(f"saved metric {path} has unsupported type {type(value).__name__}.")


def finite_metric_dict(metrics: dict[str, Any]) -> None:
    finite_metric_value("metrics", metrics)


def write_json(path: Path, values: dict[str, Any]) -> None:
    finite_metric_dict(values)
    path.write_text(json.dumps(values, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_summary(path: Path, metrics: dict[str, Any]) -> None:
    run_label = "Smoke" if metrics["smoke"] else "Full"
    run_description = "smoke" if metrics["smoke"] else "full"
    lines = [
        f"# T10 A2 History Student Distillation {run_label}",
        "",
        "## Purpose",
        "",
        f"Offline supervised behavior-distillation {run_description} run for the A2 deployment-facing history student scaffold.",
        "",
        "## Dataset",
        "",
        f"- dataset: `{metrics['dataset_path']}`",
        f"- samples: `{metrics['num_samples']}`",
        f"- train samples: `{metrics['train_samples']}`",
        f"- validation samples: `{metrics['val_samples']}`",
        "",
        "## Mapping",
        "",
        f"- input: `student_obs` history window with shape `[batch, {metrics['history_len']}, {metrics['input_dim']}]`",
        "- target: `teacher_action` at the current timestep",
        "- `teacher_obs` is not used as model input.",
        "- `true_fault_state` is not used by the student.",
        "",
        "## Window Guardrails",
        "",
        f"- total candidate windows: `{metrics['total_candidate_windows']}`",
        f"- valid windows: `{metrics['valid_windows']}`",
        f"- rejected cross-episode windows: `{metrics['rejected_cross_episode_windows']}`",
        f"- rejected done-crossing windows: `{metrics['rejected_done_crossing_windows']}`",
        "",
        "## Metrics",
        "",
        f"- train_mse: `{metrics['train_mse']}`",
        f"- val_mse: `{metrics['val_mse']}`",
        f"- train_mae: `{metrics['train_mae']}`",
        f"- val_mae: `{metrics['val_mae']}`",
        f"- no_nan_inf: `{metrics['no_nan_inf']}`",
        "",
        "## Guardrails",
        "",
        "- This is not paper-grade final evidence.",
        "- This is only a supervised history-student distillation scaffold.",
        "- A5/A7 residuals are not trained here.",
        "- Health token, UQ, CBF, P3, and P4 remain inactive.",
        "- Deployment-facing policies must not use `true_fault_state`.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def train(args: argparse.Namespace) -> int:
    validate_args(args)
    set_seed(args.seed)
    device = select_device(args.device)
    torch_module, nn_module, data_loader_class, tensor_dataset_class = require_torch()
    dataset_path = resolve_repo_path(args.dataset_path)
    output_dir = resolve_repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not output_dir.is_dir():
        raise A2HistoryDistillError(f"output_dir is missing after creation: {repo_relative(output_dir)}")
    (output_dir / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")

    max_epochs = args.max_epochs if args.max_epochs is not None else (DEFAULT_SMOKE_EPOCHS if args.smoke else 25)
    student_obs, teacher_action, done, episode_id, dataset_meta = load_history_source_arrays(dataset_path)
    history_inputs, actions, window_counts = build_history_windows(
        student_obs,
        teacher_action,
        done,
        episode_id,
        history_len=args.history_len,
    )
    train_obs, train_actions, val_obs, val_actions = split_tensors(history_inputs, actions, seed=args.seed)
    for name, tensor in (
        ("train_obs_history", train_obs),
        ("train_actions", train_actions),
        ("val_obs_history", val_obs),
        ("val_actions", val_actions),
    ):
        assert_finite_tensor(name, tensor)

    train_loader = data_loader_class(
        tensor_dataset_class(train_obs, train_actions),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        generator=torch_module.Generator().manual_seed(args.seed),
    )
    eval_train_loader = data_loader_class(
        tensor_dataset_class(train_obs, train_actions),
        batch_size=args.batch_size,
        shuffle=False,
    )
    val_loader = data_loader_class(tensor_dataset_class(val_obs, val_actions), batch_size=args.batch_size, shuffle=False)

    model = build_history_student_mlp(args.history_len, EXPECTED_INPUT_DIM, EXPECTED_ACTION_DIM).to(device)
    optimizer = torch_module.optim.AdamW(model.parameters(), lr=1.0e-3, weight_decay=1.0e-5)
    loss_fn = nn_module.MSELoss()
    epoch_metrics: list[dict[str, float]] = []
    for epoch in range(1, max_epochs + 1):
        model.train()
        batch_loss_sum = 0.0
        batch_sample_count = 0
        for batch_obs, batch_target in train_loader:
            batch_obs = batch_obs.to(device)
            batch_target = batch_target.to(device)
            assert_finite_tensor("batch_obs_history", batch_obs)
            assert_finite_tensor("batch_target", batch_target)
            pred = model(batch_obs)
            assert_finite_tensor("train_predictions", pred)
            loss = loss_fn(pred, batch_target)
            assert_finite_tensor("train_loss", loss)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            batch_size = batch_obs.shape[0]
            batch_loss_sum += float(loss.detach().cpu().item()) * batch_size
            batch_sample_count += int(batch_size)
        if batch_sample_count == 0:
            raise A2HistoryDistillError("training loader produced zero samples.")
        train_epoch_mse = batch_loss_sum / batch_sample_count
        val_epoch_mse, val_epoch_mae = evaluate(model, val_loader, device=device)
        if not all(math.isfinite(value) for value in (train_epoch_mse, val_epoch_mse, val_epoch_mae)):
            raise A2HistoryDistillError("epoch metrics contain NaN or Inf.")
        epoch_metrics.append(
            {
                "epoch": float(epoch),
                "train_epoch_mse": train_epoch_mse,
                "val_epoch_mse": val_epoch_mse,
                "val_epoch_mae": val_epoch_mae,
            }
        )
        print(
            f"[T10-A2-HISTORY] epoch={epoch}/{max_epochs} "
            f"train_mse={train_epoch_mse:.6f} val_mse={val_epoch_mse:.6f} val_mae={val_epoch_mae:.6f}",
            flush=True,
        )

    train_mse, train_mae = evaluate(model, eval_train_loader, device=device)
    val_mse, val_mae = evaluate(model, val_loader, device=device)
    metrics: dict[str, Any] = {
        "stage": "a2_student_history_distillation",
        "method": "rlm1_stripped",
        "smoke": bool(args.smoke),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_path": repo_relative(dataset_path),
        "config_path": repo_relative(args.config),
        "output_dir": repo_relative(output_dir),
        "device": str(device),
        "seed": args.seed,
        "max_epochs": max_epochs,
        "batch_size": args.batch_size,
        "history_len": args.history_len,
        "train_mse": train_mse,
        "val_mse": val_mse,
        "train_mae": train_mae,
        "val_mae": val_mae,
        "num_samples": int(history_inputs.shape[0]),
        "train_samples": int(train_obs.shape[0]),
        "val_samples": int(val_obs.shape[0]),
        "input_dim": dataset_meta["input_dim"],
        "action_dim": dataset_meta["action_dim"],
        "input_key": dataset_meta["input_key"],
        "target_key": dataset_meta["target_key"],
        "teacher_obs_used_as_input": False,
        "true_fault_state_used": False,
        "health_token_used": False,
        "residual_head_used": False,
        "no_nan_inf": True,
        "epoch_metrics": epoch_metrics,
        "source_student_obs_shape": dataset_meta["source_student_obs_shape"],
        "source_teacher_action_shape": dataset_meta["source_teacher_action_shape"],
        "source_done_shape": dataset_meta["source_done_shape"],
        "source_episode_id_shape": dataset_meta["source_episode_id_shape"],
        "model_class": "HistoryStudentMLP",
        "history_encoder": "flatten_mlp",
        "not_paper_grade_final": True,
        **window_counts,
    }
    finite_metric_dict(metrics)
    checkpoint_path = output_dir / ("a2_student_history_smoke.pt" if args.smoke else "a2_student_history.pt")
    torch_module.save(
        {
            "model_state_dict": model.state_dict(),
            "metrics": metrics,
            "model_class": "HistoryStudentMLP",
            "history_encoder": "flatten_mlp",
            "history_len": args.history_len,
            "input_dim": EXPECTED_INPUT_DIM,
            "action_dim": EXPECTED_ACTION_DIM,
            "guardrails": {
                "input": "student_obs_history",
                "target": "teacher_action",
                "no_teacher_obs_input": True,
                "no_true_fault_state": True,
                "no_health_token": True,
                "no_residual_head": True,
            },
        },
        checkpoint_path,
    )
    metrics["checkpoint_path"] = repo_relative(checkpoint_path)
    metrics_path = output_dir / "metrics.json"
    summary_path = output_dir / "summary.md"
    write_json(metrics_path, metrics)
    write_summary(summary_path, metrics)
    for required_path in (checkpoint_path, metrics_path, summary_path, output_dir / "command.txt"):
        if not required_path.is_file():
            raise A2HistoryDistillError(f"required output was not written: {repo_relative(required_path)}")
    if args.smoke and (not checkpoint_path.is_file() or not metrics_path.is_file()):
        raise A2HistoryDistillError("smoke run did not produce required checkpoint and metrics.json.")
    print(f"[T10-A2-HISTORY] checkpoint: {repo_relative(checkpoint_path)}", flush=True)
    print(f"[T10-A2-HISTORY] metrics: {repo_relative(metrics_path)}", flush=True)
    print(f"[T10-A2-HISTORY] summary: {repo_relative(summary_path)}", flush=True)
    return 0


def main() -> int:
    try:
        args = build_parser().parse_args()
        return train(args)
    except A2HistoryDistillError as exc:
        print(f"[T10-A2-HISTORY ERROR] {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
