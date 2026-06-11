#!/usr/bin/env python3
"""A2 student behavior-distillation scaffold for T10 P2 velocity data.

This is an offline supervised trainer. It loads the teacher-gap NPZ dataset and
fits a small local MLP from deployment-safe ``student_obs`` to ``teacher_action``.
It does not launch Isaac Sim, run RL, modify checkpoints, or touch task/P2 code.
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
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_PATH = "papers/conference/datasets/t10_teacher_gap_p2_velocity_seed0/dataset.npz"
DEFAULT_CONFIG_PATH = "configs/train/a2_student_distill.yaml"
DEFAULT_OUTPUT_DIR = "papers/conference/results/t10_a2_student_distill_smoke"
EXPECTED_INPUT_DIM = 61
EXPECTED_ACTION_DIM = 8
DEFAULT_SMOKE_EPOCHS = 3
DEFAULT_BATCH_SIZE = 4096
DEFAULT_SEED = 0
DEFAULT_VAL_FRACTION = 0.1


class A2DistillError(ValueError):
    """Raised for invalid A2 distillation configuration or data."""


class StudentMLP(nn.Module):
    """Small local MLP for A2 smoke distillation."""

    def __init__(self, input_dim: int, action_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


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
    parser = argparse.ArgumentParser(description="Offline A2 student behavior-distillation trainer.")
    parser.add_argument("--dataset_path", default=DEFAULT_DATASET_PATH)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--max_epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--smoke", action="store_true", help="Run the small smoke-training pass.")
    return parser


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise A2DistillError("--device cuda requested but CUDA is not available.")
    return torch.device(device_arg)


def validate_args(args: argparse.Namespace) -> None:
    dataset_path = resolve_repo_path(args.dataset_path)
    if not dataset_path.is_file():
        raise A2DistillError(f"dataset_path does not exist: {repo_relative(dataset_path)}")
    config_path = resolve_repo_path(args.config)
    if not config_path.is_file():
        raise A2DistillError(f"config does not exist: {repo_relative(config_path)}")
    if args.batch_size <= 0:
        raise A2DistillError("--batch_size must be > 0.")
    if args.seed < 0:
        raise A2DistillError("--seed must be non-negative.")
    if args.max_epochs is not None and args.max_epochs <= 0:
        raise A2DistillError("--max_epochs must be > 0 when provided.")


def check_finite_array(name: str, array: np.ndarray) -> None:
    if not np.isfinite(array).all():
        raise A2DistillError(f"{name} contains NaN or Inf.")


def load_distill_arrays(dataset_path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    data = np.load(dataset_path)
    input_key = "student_obs"
    target_key = "teacher_action"
    if input_key != "student_obs" or input_key == "teacher_obs":
        raise A2DistillError("A2 model input must be student_obs and must never be teacher_obs.")
    missing = [key for key in (input_key, target_key) if key not in data.files]
    if missing:
        raise A2DistillError(f"dataset missing required arrays: {missing}")

    student_obs = np.asarray(data[input_key], dtype=np.float32)
    teacher_action = np.asarray(data[target_key], dtype=np.float32)
    if student_obs.ndim != 3 or student_obs.shape[-1] != EXPECTED_INPUT_DIM:
        raise A2DistillError(f"student_obs expected shape [T, N, {EXPECTED_INPUT_DIM}], got {student_obs.shape}.")
    if teacher_action.ndim != 3 or teacher_action.shape[-1] != EXPECTED_ACTION_DIM:
        raise A2DistillError(
            f"teacher_action expected shape [T, N, {EXPECTED_ACTION_DIM}], got {teacher_action.shape}."
        )
    if student_obs.shape[:2] != teacher_action.shape[:2]:
        raise A2DistillError(
            f"student_obs and teacher_action time/env dimensions differ: {student_obs.shape} vs {teacher_action.shape}."
        )
    check_finite_array("student_obs", student_obs)
    check_finite_array("teacher_action", teacher_action)

    flattened_obs = student_obs.reshape(-1, EXPECTED_INPUT_DIM)
    flattened_action = teacher_action.reshape(-1, EXPECTED_ACTION_DIM)
    metadata = {
        "input_key": input_key,
        "target_key": target_key,
        "source_student_obs_shape": [int(dim) for dim in student_obs.shape],
        "source_teacher_action_shape": [int(dim) for dim in teacher_action.shape],
        "num_samples": int(flattened_obs.shape[0]),
        "input_dim": EXPECTED_INPUT_DIM,
        "action_dim": EXPECTED_ACTION_DIM,
        "teacher_obs_used_as_input": False,
    }
    return flattened_obs, flattened_action, metadata


def split_tensors(
    obs: np.ndarray,
    actions: np.ndarray,
    *,
    seed: int,
    val_fraction: float = DEFAULT_VAL_FRACTION,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    num_samples = obs.shape[0]
    if num_samples < 2:
        raise A2DistillError("dataset must contain at least two samples.")
    generator = np.random.default_rng(seed)
    indices = generator.permutation(num_samples)
    val_count = max(1, int(round(num_samples * val_fraction)))
    val_count = min(val_count, num_samples - 1)
    val_indices = indices[:val_count]
    train_indices = indices[val_count:]
    train_obs = torch.from_numpy(obs[train_indices].copy())
    train_actions = torch.from_numpy(actions[train_indices].copy())
    val_obs = torch.from_numpy(obs[val_indices].copy())
    val_actions = torch.from_numpy(actions[val_indices].copy())
    return train_obs, train_actions, val_obs, val_actions


def assert_finite_tensor(name: str, tensor: torch.Tensor) -> None:
    if not torch.isfinite(tensor).all():
        raise A2DistillError(f"{name} contains NaN or Inf.")


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
        for obs, target in loader:
            obs = obs.to(device)
            target = target.to(device)
            pred = model(obs)
            assert_finite_tensor("eval_predictions", pred)
            diff = pred - target
            mse = diff.square().mean(dim=1)
            mae = diff.abs().mean(dim=1)
            assert_finite_tensor("eval_mse", mse)
            assert_finite_tensor("eval_mae", mae)
            batch_size = obs.shape[0]
            total_mse += float(mse.sum().detach().cpu().item())
            total_mae += float(mae.sum().detach().cpu().item())
            total_samples += int(batch_size)
    if total_samples == 0:
        raise A2DistillError("evaluation loader produced zero samples.")
    return total_mse / total_samples, total_mae / total_samples


def finite_metric_dict(metrics: dict[str, Any]) -> None:
    for key, value in metrics.items():
        if isinstance(value, float) and not math.isfinite(value):
            raise A2DistillError(f"saved metric {key} is not finite: {value}")


def write_json(path: Path, values: dict[str, Any]) -> None:
    finite_metric_dict(values)
    path.write_text(json.dumps(values, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_summary(path: Path, metrics: dict[str, Any]) -> None:
    lines = [
        "# T10 A2 Student Distillation Smoke",
        "",
        "## Purpose",
        "",
        "Offline supervised behavior-distillation smoke for the A2 deployment-facing student scaffold.",
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
        "- input: `student_obs`",
        "- target: `teacher_action`",
        "- `teacher_obs` is not used as model input.",
        "- `true_fault_state` is not used by the student.",
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
        "- This is only a supervised behavior-distillation scaffold.",
        "- A5/A7 residuals are not trained here.",
        "- Health token, UQ, CBF, P3, and P4 remain inactive.",
        "- Deployment-facing policies must not use `true_fault_state`.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def train(args: argparse.Namespace) -> int:
    validate_args(args)
    set_seed(args.seed)
    device = select_device(args.device)
    dataset_path = resolve_repo_path(args.dataset_path)
    output_dir = resolve_repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not output_dir.is_dir():
        raise A2DistillError(f"output_dir is missing after creation: {repo_relative(output_dir)}")
    (output_dir / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")

    max_epochs = args.max_epochs if args.max_epochs is not None else (DEFAULT_SMOKE_EPOCHS if args.smoke else 25)
    obs, actions, dataset_meta = load_distill_arrays(dataset_path)
    train_obs, train_actions, val_obs, val_actions = split_tensors(obs, actions, seed=args.seed)
    for name, tensor in (
        ("train_obs", train_obs),
        ("train_actions", train_actions),
        ("val_obs", val_obs),
        ("val_actions", val_actions),
    ):
        assert_finite_tensor(name, tensor)

    train_loader = DataLoader(
        TensorDataset(train_obs, train_actions),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        generator=torch.Generator().manual_seed(args.seed),
    )
    eval_train_loader = DataLoader(TensorDataset(train_obs, train_actions), batch_size=args.batch_size, shuffle=False)
    val_loader = DataLoader(TensorDataset(val_obs, val_actions), batch_size=args.batch_size, shuffle=False)

    model = StudentMLP(EXPECTED_INPUT_DIM, EXPECTED_ACTION_DIM).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3, weight_decay=1.0e-5)
    loss_fn = nn.MSELoss()
    epoch_metrics: list[dict[str, float]] = []
    for epoch in range(1, max_epochs + 1):
        model.train()
        batch_loss_sum = 0.0
        batch_sample_count = 0
        for batch_obs, batch_target in train_loader:
            batch_obs = batch_obs.to(device)
            batch_target = batch_target.to(device)
            assert_finite_tensor("batch_obs", batch_obs)
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
            raise A2DistillError("training loader produced zero samples.")
        train_epoch_mse = batch_loss_sum / batch_sample_count
        val_epoch_mse, val_epoch_mae = evaluate(model, val_loader, device=device)
        if not all(math.isfinite(value) for value in (train_epoch_mse, val_epoch_mse, val_epoch_mae)):
            raise A2DistillError("epoch metrics contain NaN or Inf.")
        epoch_metrics.append(
            {
                "epoch": float(epoch),
                "train_epoch_mse": train_epoch_mse,
                "val_epoch_mse": val_epoch_mse,
                "val_epoch_mae": val_epoch_mae,
            }
        )
        print(
            f"[T10-A2] epoch={epoch}/{max_epochs} "
            f"train_mse={train_epoch_mse:.6f} val_mse={val_epoch_mse:.6f} val_mae={val_epoch_mae:.6f}",
            flush=True,
        )

    train_mse, train_mae = evaluate(model, eval_train_loader, device=device)
    val_mse, val_mae = evaluate(model, val_loader, device=device)
    metrics: dict[str, Any] = {
        "stage": "a2_student_distillation",
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
        "train_mse": train_mse,
        "val_mse": val_mse,
        "train_mae": train_mae,
        "val_mae": val_mae,
        "num_samples": dataset_meta["num_samples"],
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
        "not_paper_grade_final": True,
    }
    finite_metric_dict(metrics)
    checkpoint_path = output_dir / ("a2_student_smoke.pt" if args.smoke else "a2_student.pt")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "metrics": metrics,
            "model_class": "StudentMLP",
            "input_dim": EXPECTED_INPUT_DIM,
            "action_dim": EXPECTED_ACTION_DIM,
            "guardrails": {
                "input": "student_obs",
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
            raise A2DistillError(f"required output was not written: {repo_relative(required_path)}")
    print(f"[T10-A2] checkpoint: {repo_relative(checkpoint_path)}", flush=True)
    print(f"[T10-A2] metrics: {repo_relative(metrics_path)}", flush=True)
    print(f"[T10-A2] summary: {repo_relative(summary_path)}", flush=True)
    return 0


def main() -> int:
    try:
        args = build_parser().parse_args()
        return train(args)
    except A2DistillError as exc:
        print(f"[T10-A2 ERROR] {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
