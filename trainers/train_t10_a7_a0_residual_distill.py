#!/usr/bin/env python3
"""A7 frozen-healthy-PPO teacher-gap residual distillation scaffold.

This offline supervised trainer uses deployment-safe ``student_obs`` history
windows plus dataset-provided A0 healthy PPO actions to fit a residual model to
``teacher_action - a0_action``. It does not launch Isaac Sim, run RL, train A0,
train the teacher, train A2/A5, or use teacher observations / true fault state.
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

from train_t10_a2_student_history_distill import (
    A2HistoryDistillError,
    DEFAULT_VAL_FRACTION,
    EXPECTED_ACTION_DIM,
    EXPECTED_INPUT_DIM,
    finite_metric_dict,
    repo_relative,
    require_torch,
    resolve_repo_path,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_PATH = "papers/conference/datasets/t10_teacher_gap_p2_velocity_seed0/dataset.npz"
DEFAULT_CONFIG_PATH = "configs/train/a7_a0_residual_distill.yaml"
DEFAULT_OUTPUT_DIR = "papers/conference/results/t10_a7_a0_residual_distill_smoke"
DEFAULT_A0_CHECKPOINT = (
    "logs/rsl_rl/healthy_baseline_velocity__rlm1_stripped__none/"
    "2026-06-09_23-58-25_a0_velocity_candidate1000__seed0/model_999.pt"
)
DEFAULT_HISTORY_LEN = 16
DEFAULT_SMOKE_EPOCHS = 3
DEFAULT_BATCH_SIZE = 4096
DEFAULT_SEED = 0
DEFAULT_RESIDUAL_INPUT_MODE = "history_plus_a0_action"
TARGET_CONSISTENCY_ATOL = 1.0e-5


class A7A0ResidualDistillError(ValueError):
    """Raised for invalid A7 A0-residual distillation configuration or data."""


def build_residual_mlp(history_len: int, input_dim: int, action_dim: int, *, residual_input_mode: str) -> Any:
    _, nn_module, _, _ = require_torch()
    if residual_input_mode != DEFAULT_RESIDUAL_INPUT_MODE:
        raise A7A0ResidualDistillError(f"unsupported residual_input_mode: {residual_input_mode!r}")
    residual_input_dim = history_len * input_dim + action_dim

    class A0HistoryResidualMLP(nn_module.Module):
        """Small local flatten-history plus A0-action residual MLP."""

        def __init__(self) -> None:
            super().__init__()
            self.net = nn_module.Sequential(
                nn_module.Linear(residual_input_dim, 512),
                nn_module.ELU(),
                nn_module.Linear(512, 256),
                nn_module.ELU(),
                nn_module.Linear(256, 128),
                nn_module.ELU(),
                nn_module.Linear(128, action_dim),
            )

        def forward(self, obs_history: Any, a0_action: Any) -> Any:
            if obs_history.ndim != 3:
                raise A7A0ResidualDistillError(
                    f"expected obs_history [batch, history, obs_dim], got {tuple(obs_history.shape)}"
                )
            if a0_action.ndim != 2 or a0_action.shape[-1] != action_dim:
                raise A7A0ResidualDistillError(
                    f"expected a0_action [batch, {action_dim}], got {tuple(a0_action.shape)}"
                )
            flat_history = obs_history.flatten(start_dim=1)
            torch_module, _, _, _ = require_torch()
            return self.net(torch_module.cat((flat_history, a0_action), dim=1))

    return A0HistoryResidualMLP()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline A7 A0 teacher-gap residual distillation trainer.")
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


def select_device(device_arg: str) -> Any:
    torch_module, _, _, _ = require_torch()
    if device_arg == "auto":
        return torch_module.device("cuda" if torch_module.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch_module.cuda.is_available():
        raise A7A0ResidualDistillError("--device cuda requested but CUDA is not available.")
    return torch_module.device(device_arg)


def validate_args(args: argparse.Namespace) -> None:
    dataset_path = resolve_repo_path(args.dataset_path)
    if not dataset_path.is_file():
        raise A7A0ResidualDistillError(f"dataset_path does not exist: {repo_relative(dataset_path)}")
    config_path = resolve_repo_path(args.config)
    if not config_path.is_file():
        raise A7A0ResidualDistillError(f"config does not exist: {repo_relative(config_path)}")
    if args.history_len < 1:
        raise A7A0ResidualDistillError("--history_len must be >= 1.")
    if args.batch_size <= 0:
        raise A7A0ResidualDistillError("--batch_size must be > 0.")
    if args.seed < 0:
        raise A7A0ResidualDistillError("--seed must be non-negative.")
    if args.max_epochs is not None and args.max_epochs <= 0:
        raise A7A0ResidualDistillError("--max_epochs must be > 0 when provided.")


def assert_finite_tensor(name: str, tensor: Any) -> None:
    torch_module, _, _, _ = require_torch()
    if not torch_module.isfinite(tensor).all():
        raise A7A0ResidualDistillError(f"{name} contains NaN or Inf.")


def check_finite_array(name: str, array: np.ndarray) -> None:
    if not np.isfinite(array).all():
        raise A7A0ResidualDistillError(f"{name} contains NaN or Inf.")


def load_source_arrays(dataset_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    data = np.load(dataset_path)
    required_keys = ("student_obs", "teacher_action", "a0_action", "done", "episode_id")
    missing = [key for key in required_keys if key not in data.files]
    if missing:
        raise A7A0ResidualDistillError(f"dataset missing required arrays: {missing}")

    student_obs = np.asarray(data["student_obs"], dtype=np.float32)
    teacher_action = np.asarray(data["teacher_action"], dtype=np.float32)
    a0_action = np.asarray(data["a0_action"], dtype=np.float32)
    done = np.asarray(data["done"], dtype=bool)
    episode_id = np.asarray(data["episode_id"])

    if student_obs.ndim != 3 or student_obs.shape[-1] != EXPECTED_INPUT_DIM:
        raise A7A0ResidualDistillError(f"student_obs expected [T, N, {EXPECTED_INPUT_DIM}], got {student_obs.shape}.")
    if teacher_action.ndim != 3 or teacher_action.shape[-1] != EXPECTED_ACTION_DIM:
        raise A7A0ResidualDistillError(
            f"teacher_action expected [T, N, {EXPECTED_ACTION_DIM}], got {teacher_action.shape}."
        )
    if a0_action.ndim != 3 or a0_action.shape[-1] != EXPECTED_ACTION_DIM:
        raise A7A0ResidualDistillError(f"a0_action expected [T, N, {EXPECTED_ACTION_DIM}], got {a0_action.shape}.")
    if student_obs.shape[:2] != teacher_action.shape[:2] or student_obs.shape[:2] != a0_action.shape[:2]:
        raise A7A0ResidualDistillError(
            f"time/env dimensions differ: student_obs={student_obs.shape}, "
            f"teacher_action={teacher_action.shape}, a0_action={a0_action.shape}."
        )
    if done.shape != student_obs.shape[:2]:
        raise A7A0ResidualDistillError(f"done expected shape {student_obs.shape[:2]}, got {done.shape}.")
    if episode_id.shape != student_obs.shape[:2]:
        raise A7A0ResidualDistillError(f"episode_id expected shape {student_obs.shape[:2]}, got {episode_id.shape}.")

    for name, array in (
        ("student_obs", student_obs),
        ("teacher_action", teacher_action),
        ("a0_action", a0_action),
    ):
        check_finite_array(name, array)
    if np.issubdtype(episode_id.dtype, np.number):
        check_finite_array("episode_id", episode_id)

    teacher_minus_a0 = teacher_action - a0_action
    check_finite_array("teacher_action_minus_a0_action", teacher_minus_a0)
    a7_target_present = "a7_residual_target" in data.files
    if a7_target_present:
        residual_target = np.asarray(data["a7_residual_target"], dtype=np.float32)
        if residual_target.ndim != 3 or residual_target.shape[-1] != EXPECTED_ACTION_DIM:
            raise A7A0ResidualDistillError(
                f"a7_residual_target expected [T, N, {EXPECTED_ACTION_DIM}], got {residual_target.shape}."
            )
        if residual_target.shape != teacher_minus_a0.shape:
            raise A7A0ResidualDistillError(
                f"a7_residual_target shape {residual_target.shape} does not match teacher-a0 {teacher_minus_a0.shape}."
            )
        check_finite_array("a7_residual_target", residual_target)
        consistency_error = np.abs(residual_target - teacher_minus_a0)
        check_finite_array("a7_residual_target_consistency_error", consistency_error)
        max_consistency_error = float(consistency_error.max())
        target_matches = bool(max_consistency_error <= TARGET_CONSISTENCY_ATOL)
        if not target_matches:
            raise A7A0ResidualDistillError(
                "a7_residual_target does not match teacher_action - a0_action within "
                f"{TARGET_CONSISTENCY_ATOL}: max_abs_error={max_consistency_error}."
            )
    else:
        residual_target = teacher_minus_a0
        max_consistency_error = 0.0
        target_matches = True

    metadata = {
        "input_key": "student_obs_history_plus_a0_action",
        "target_key": "a7_residual_target" if a7_target_present else "teacher_action_minus_a0_action",
        "a7_residual_target_present": bool(a7_target_present),
        "a7_target_matches_teacher_minus_a0": bool(target_matches),
        "a7_target_max_abs_consistency_error": max_consistency_error,
        "source_student_obs_shape": [int(dim) for dim in student_obs.shape],
        "source_teacher_action_shape": [int(dim) for dim in teacher_action.shape],
        "source_a0_action_shape": [int(dim) for dim in a0_action.shape],
        "source_done_shape": [int(dim) for dim in done.shape],
        "source_episode_id_shape": [int(dim) for dim in episode_id.shape],
        "input_dim": EXPECTED_INPUT_DIM,
        "action_dim": EXPECTED_ACTION_DIM,
        "teacher_obs_used_as_input": False,
        "true_fault_state_used": False,
        "health_token_used": False,
    }
    return student_obs, teacher_action, a0_action, residual_target, done, episode_id, metadata


def build_a7_history_windows(
    student_obs: np.ndarray,
    teacher_action: np.ndarray,
    a0_action: np.ndarray,
    residual_target: np.ndarray,
    done: np.ndarray,
    episode_id: np.ndarray,
    *,
    history_len: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    if history_len < 1:
        raise A7A0ResidualDistillError("history_len must be >= 1.")
    time_steps, num_envs, input_dim = student_obs.shape
    if history_len > time_steps:
        raise A7A0ResidualDistillError(f"history_len {history_len} exceeds dataset time dimension {time_steps}.")

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
        raise A7A0ResidualDistillError("no valid history windows were produced.")

    history_inputs = np.empty((valid_windows, history_len, input_dim), dtype=np.float32)
    teacher_actions = np.empty((valid_windows, EXPECTED_ACTION_DIM), dtype=np.float32)
    base_actions = np.empty((valid_windows, EXPECTED_ACTION_DIM), dtype=np.float32)
    residual_targets = np.empty((valid_windows, EXPECTED_ACTION_DIM), dtype=np.float32)
    sample_idx = 0
    for env_id in range(num_envs):
        for target_t in range(history_len - 1, time_steps):
            start_t = target_t - history_len + 1
            if not valid_mask[start_t, env_id]:
                continue
            history_inputs[sample_idx] = student_obs[start_t : target_t + 1, env_id, :]
            teacher_actions[sample_idx] = teacher_action[target_t, env_id, :]
            base_actions[sample_idx] = a0_action[target_t, env_id, :]
            residual_targets[sample_idx] = residual_target[target_t, env_id, :]
            sample_idx += 1

    for name, array in (
        ("history_inputs", history_inputs),
        ("teacher_actions", teacher_actions),
        ("base_actions", base_actions),
        ("residual_targets", residual_targets),
    ):
        check_finite_array(name, array)

    counts = {
        "total_candidate_windows": int(candidate_count),
        "valid_windows": int(valid_windows),
        "rejected_cross_episode_windows": int(rejected_cross_episode_windows),
        "rejected_done_crossing_windows": int(rejected_done_crossing_windows),
    }
    return history_inputs, base_actions, teacher_actions, residual_targets, counts


def split_arrays(
    history_inputs: np.ndarray,
    base_actions: np.ndarray,
    teacher_actions: np.ndarray,
    residual_targets: np.ndarray,
    *,
    seed: int,
    val_fraction: float = DEFAULT_VAL_FRACTION,
) -> tuple[Any, ...]:
    torch_module, _, _, _ = require_torch()
    num_samples = history_inputs.shape[0]
    if num_samples < 2:
        raise A7A0ResidualDistillError("dataset must contain at least two valid history samples.")
    generator = np.random.default_rng(seed)
    indices = generator.permutation(num_samples)
    val_count = max(1, int(round(num_samples * val_fraction)))
    val_count = min(val_count, num_samples - 1)
    val_indices = indices[:val_count]
    train_indices = indices[val_count:]

    return (
        torch_module.from_numpy(history_inputs[train_indices].copy()),
        torch_module.from_numpy(base_actions[train_indices].copy()),
        torch_module.from_numpy(teacher_actions[train_indices].copy()),
        torch_module.from_numpy(residual_targets[train_indices].copy()),
        torch_module.from_numpy(history_inputs[val_indices].copy()),
        torch_module.from_numpy(base_actions[val_indices].copy()),
        torch_module.from_numpy(teacher_actions[val_indices].copy()),
        torch_module.from_numpy(residual_targets[val_indices].copy()),
    )


def evaluate_residual_model(model: Any, loader: Any, *, device: Any) -> tuple[float, float, float, float]:
    model.eval()
    residual_mse_sum = 0.0
    residual_mae_sum = 0.0
    reconstructed_mse_sum = 0.0
    reconstructed_mae_sum = 0.0
    sample_count = 0
    torch_module, _, _, _ = require_torch()
    with torch_module.inference_mode():
        for obs_history, base_action, teacher_action, residual_target in loader:
            obs_history = obs_history.to(device)
            base_action = base_action.to(device)
            teacher_action = teacher_action.to(device)
            residual_target = residual_target.to(device)
            for name, tensor in (
                ("eval_obs_history", obs_history),
                ("eval_a0_action", base_action),
                ("eval_teacher_action", teacher_action),
                ("eval_residual_target", residual_target),
            ):
                assert_finite_tensor(name, tensor)
            pred_residual = model(obs_history, base_action)
            assert_finite_tensor("eval_pred_residual", pred_residual)
            reconstructed_action = base_action + pred_residual
            assert_finite_tensor("eval_reconstructed_action", reconstructed_action)
            residual_diff = pred_residual - residual_target
            reconstructed_diff = reconstructed_action - teacher_action
            residual_mse = residual_diff.square().mean(dim=1)
            residual_mae = residual_diff.abs().mean(dim=1)
            reconstructed_mse = reconstructed_diff.square().mean(dim=1)
            reconstructed_mae = reconstructed_diff.abs().mean(dim=1)
            for name, tensor in (
                ("eval_residual_mse", residual_mse),
                ("eval_residual_mae", residual_mae),
                ("eval_reconstructed_mse", reconstructed_mse),
                ("eval_reconstructed_mae", reconstructed_mae),
            ):
                assert_finite_tensor(name, tensor)
            batch_size = obs_history.shape[0]
            residual_mse_sum += float(residual_mse.sum().detach().cpu().item())
            residual_mae_sum += float(residual_mae.sum().detach().cpu().item())
            reconstructed_mse_sum += float(reconstructed_mse.sum().detach().cpu().item())
            reconstructed_mae_sum += float(reconstructed_mae.sum().detach().cpu().item())
            sample_count += int(batch_size)
    if sample_count == 0:
        raise A7A0ResidualDistillError("evaluation loader produced zero samples.")
    return (
        residual_mse_sum / sample_count,
        residual_mae_sum / sample_count,
        reconstructed_mse_sum / sample_count,
        reconstructed_mae_sum / sample_count,
    )


def write_json(path: Path, values: dict[str, Any]) -> None:
    finite_metric_dict(values)
    path.write_text(json.dumps(values, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_summary(path: Path, metrics: dict[str, Any]) -> None:
    run_label = "Smoke" if metrics["smoke"] else "Full"
    run_description = "smoke" if metrics["smoke"] else "full"
    lines = [
        f"# T10 A7 A0 Residual Distillation {run_label}",
        "",
        "## Purpose",
        "",
        f"Offline supervised {run_description} run for the A7 frozen-healthy-PPO teacher-gap residual scaffold.",
        "",
        "## Dataset",
        "",
        f"- dataset: `{metrics['dataset_path']}`",
        f"- samples: `{metrics['num_samples']}`",
        f"- train samples: `{metrics['train_samples']}`",
        f"- validation samples: `{metrics['val_samples']}`",
        "",
        "## Frozen Base Concept",
        "",
        f"- base policy: `{metrics['base_policy']}`",
        f"- base checkpoint: `{metrics['base_policy_checkpoint']}`",
        "- offline base action source: dataset-provided `a0_action`",
        "",
        "## Mapping",
        "",
        f"- input: `student_obs` history window with shape `[batch, {metrics['history_len']}, {metrics['input_dim']}]` plus `a0_action`",
        f"- residual_input_mode: `{metrics['residual_input_mode']}`",
        "- residual target: `teacher_action - a0_action`",
        "- reconstruction: `a0_action + predicted_residual`",
        "",
        "## Target Consistency",
        "",
        f"- a7_residual_target present: `{metrics['a7_residual_target_present']}`",
        f"- a7 target matches teacher minus A0: `{metrics['a7_target_matches_teacher_minus_a0']}`",
        f"- max absolute consistency error: `{metrics['a7_target_max_abs_consistency_error']}`",
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
        f"- train_residual_mse: `{metrics['train_residual_mse']}`",
        f"- val_residual_mse: `{metrics['val_residual_mse']}`",
        f"- train_reconstructed_action_mse: `{metrics['train_reconstructed_action_mse']}`",
        f"- val_reconstructed_action_mse: `{metrics['val_reconstructed_action_mse']}`",
        f"- base_a0_action_mse_to_teacher: `{metrics['base_a0_action_mse_to_teacher']}`",
        f"- no_nan_inf: `{metrics['no_nan_inf']}`",
        "",
        "## Guardrails",
        "",
        "- This is not paper-grade final evidence.",
        "- This is only a supervised residual distillation scaffold.",
        "- A2/A5 residuals are not trained here.",
        "- Teacher observations, true fault state, health token, A2 action, and A5 action are not used.",
        "- Deployment-facing policies must not use `true_fault_state`.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def train(args: argparse.Namespace) -> int:
    validate_args(args)
    set_seed(args.seed)
    torch_module, nn_module, data_loader_class, tensor_dataset_class = require_torch()
    device = select_device(args.device)
    dataset_path = resolve_repo_path(args.dataset_path)
    output_dir = resolve_repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")

    max_epochs = args.max_epochs if args.max_epochs is not None else (DEFAULT_SMOKE_EPOCHS if args.smoke else 25)
    student_obs, teacher_action, a0_action, residual_target, done, episode_id, dataset_meta = load_source_arrays(
        dataset_path
    )
    history_inputs, base_actions, teacher_actions, residual_targets, window_counts = build_a7_history_windows(
        student_obs,
        teacher_action,
        a0_action,
        residual_target,
        done,
        episode_id,
        history_len=args.history_len,
    )
    base_diff = base_actions - teacher_actions
    base_sample_mse = np.mean(np.square(base_diff), axis=1)
    base_sample_mae = np.mean(np.abs(base_diff), axis=1)
    residual_norm = np.linalg.norm(residual_targets, axis=1)
    for name, array in (
        ("base_sample_mse", base_sample_mse),
        ("base_sample_mae", base_sample_mae),
        ("residual_norm", residual_norm),
    ):
        check_finite_array(name, array)

    (
        train_obs,
        train_base,
        train_teacher,
        train_residual,
        val_obs,
        val_base,
        val_teacher,
        val_residual,
    ) = split_arrays(history_inputs, base_actions, teacher_actions, residual_targets, seed=args.seed)
    for name, tensor in (
        ("train_obs_history", train_obs),
        ("train_a0_action", train_base),
        ("train_teacher_action", train_teacher),
        ("train_residual_target", train_residual),
        ("val_obs_history", val_obs),
        ("val_a0_action", val_base),
        ("val_teacher_action", val_teacher),
        ("val_residual_target", val_residual),
    ):
        assert_finite_tensor(name, tensor)

    train_loader = data_loader_class(
        tensor_dataset_class(train_obs, train_base, train_teacher, train_residual),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        generator=torch_module.Generator().manual_seed(args.seed),
    )
    eval_train_loader = data_loader_class(
        tensor_dataset_class(train_obs, train_base, train_teacher, train_residual),
        batch_size=args.batch_size,
        shuffle=False,
    )
    val_loader = data_loader_class(
        tensor_dataset_class(val_obs, val_base, val_teacher, val_residual),
        batch_size=args.batch_size,
        shuffle=False,
    )

    residual_model = build_residual_mlp(
        args.history_len,
        EXPECTED_INPUT_DIM,
        EXPECTED_ACTION_DIM,
        residual_input_mode=DEFAULT_RESIDUAL_INPUT_MODE,
    ).to(device)
    optimizer = torch_module.optim.AdamW(residual_model.parameters(), lr=1.0e-3, weight_decay=1.0e-5)
    loss_fn = nn_module.MSELoss()
    epoch_metrics: list[dict[str, float]] = []
    for epoch in range(1, max_epochs + 1):
        residual_model.train()
        loss_sum = 0.0
        sample_count = 0
        for batch_obs, batch_base, _batch_teacher, batch_residual in train_loader:
            batch_obs = batch_obs.to(device)
            batch_base = batch_base.to(device)
            batch_residual = batch_residual.to(device)
            for name, tensor in (
                ("batch_obs_history", batch_obs),
                ("batch_a0_action", batch_base),
                ("batch_residual_target", batch_residual),
            ):
                assert_finite_tensor(name, tensor)
            pred_residual = residual_model(batch_obs, batch_base)
            assert_finite_tensor("train_pred_residual", pred_residual)
            loss = loss_fn(pred_residual, batch_residual)
            assert_finite_tensor("train_loss", loss)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            batch_size = batch_obs.shape[0]
            loss_sum += float(loss.detach().cpu().item()) * batch_size
            sample_count += int(batch_size)
        if sample_count == 0:
            raise A7A0ResidualDistillError("training loader produced zero samples.")
        train_epoch_residual_mse = loss_sum / sample_count
        val_residual_mse, val_residual_mae, val_recon_mse, val_recon_mae = evaluate_residual_model(
            residual_model,
            val_loader,
            device=device,
        )
        epoch_values = (train_epoch_residual_mse, val_residual_mse, val_residual_mae, val_recon_mse, val_recon_mae)
        if not all(math.isfinite(value) for value in epoch_values):
            raise A7A0ResidualDistillError("epoch metrics contain NaN or Inf.")
        epoch_metrics.append(
            {
                "epoch": float(epoch),
                "train_epoch_residual_mse": train_epoch_residual_mse,
                "val_epoch_residual_mse": val_residual_mse,
                "val_epoch_residual_mae": val_residual_mae,
                "val_epoch_reconstructed_action_mse": val_recon_mse,
                "val_epoch_reconstructed_action_mae": val_recon_mae,
            }
        )
        print(
            f"[T10-A7-A0-RESIDUAL] epoch={epoch}/{max_epochs} "
            f"train_residual_mse={train_epoch_residual_mse:.6f} "
            f"val_residual_mse={val_residual_mse:.6f} val_recon_mse={val_recon_mse:.6f}",
            flush=True,
        )

    train_residual_mse, train_residual_mae, train_recon_mse, train_recon_mae = evaluate_residual_model(
        residual_model,
        eval_train_loader,
        device=device,
    )
    val_residual_mse, val_residual_mae, val_recon_mse, val_recon_mae = evaluate_residual_model(
        residual_model,
        val_loader,
        device=device,
    )
    metrics: dict[str, Any] = {
        "stage": "a7_a0_residual_distillation",
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
        "residual_input_mode": DEFAULT_RESIDUAL_INPUT_MODE,
        "action_composition": "a_final = a0_action + alpha * residual_action",
        "alpha_default_for_later_eval": 1.0,
        "train_residual_mse": train_residual_mse,
        "val_residual_mse": val_residual_mse,
        "train_residual_mae": train_residual_mae,
        "val_residual_mae": val_residual_mae,
        "train_reconstructed_action_mse": train_recon_mse,
        "val_reconstructed_action_mse": val_recon_mse,
        "train_reconstructed_action_mae": train_recon_mae,
        "val_reconstructed_action_mae": val_recon_mae,
        "base_a0_action_mse_to_teacher": float(base_sample_mse.mean()),
        "base_a0_action_mae_to_teacher": float(base_sample_mae.mean()),
        "residual_target_mean_norm": float(residual_norm.mean()),
        "residual_target_max_norm": float(residual_norm.max()),
        "num_samples": int(history_inputs.shape[0]),
        "train_samples": int(train_obs.shape[0]),
        "val_samples": int(val_obs.shape[0]),
        "input_dim": dataset_meta["input_dim"],
        "action_dim": dataset_meta["action_dim"],
        "residual_dim": EXPECTED_ACTION_DIM,
        "input_key": "student_obs_history_plus_a0_action",
        "target_key": dataset_meta["target_key"],
        "a7_residual_target_present": dataset_meta["a7_residual_target_present"],
        "a7_target_matches_teacher_minus_a0": dataset_meta["a7_target_matches_teacher_minus_a0"],
        "a7_target_max_abs_consistency_error": dataset_meta["a7_target_max_abs_consistency_error"],
        "teacher_obs_used_as_input": False,
        "true_fault_state_used": False,
        "health_token_used": False,
        "residual_head_used": True,
        "a0_action_used": True,
        "a2_action_used": False,
        "a5_action_used": False,
        "no_nan_inf": True,
        "epoch_metrics": epoch_metrics,
        "source_student_obs_shape": dataset_meta["source_student_obs_shape"],
        "source_teacher_action_shape": dataset_meta["source_teacher_action_shape"],
        "source_a0_action_shape": dataset_meta["source_a0_action_shape"],
        "source_done_shape": dataset_meta["source_done_shape"],
        "source_episode_id_shape": dataset_meta["source_episode_id_shape"],
        "model_class": "A0HistoryResidualMLP",
        "base_policy": "A0_healthy_velocity",
        "base_policy_checkpoint": DEFAULT_A0_CHECKPOINT,
        "not_paper_grade_final": True,
        **window_counts,
    }
    finite_metric_dict(metrics)
    checkpoint_path = output_dir / ("a7_a0_residual_smoke.pt" if args.smoke else "a7_a0_residual.pt")
    torch_module.save(
        {
            "model_state_dict": residual_model.state_dict(),
            "metrics": metrics,
            "model_class": "A0HistoryResidualMLP",
            "history_len": args.history_len,
            "input_dim": EXPECTED_INPUT_DIM,
            "action_dim": EXPECTED_ACTION_DIM,
            "residual_dim": EXPECTED_ACTION_DIM,
            "residual_input_mode": DEFAULT_RESIDUAL_INPUT_MODE,
            "base_policy": "A0_healthy_velocity",
            "base_policy_checkpoint": DEFAULT_A0_CHECKPOINT,
            "guardrails": {
                "input": "student_obs_history_plus_a0_action",
                "target": "teacher_action_minus_a0_action",
                "no_teacher_obs_input": True,
                "no_true_fault_state": True,
                "no_health_token": True,
                "residual_head": True,
                "uses_a0_action": True,
                "no_a2_action": True,
                "no_a5_action": True,
                "no_a7_isaac_eval": True,
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
            raise A7A0ResidualDistillError(f"required output was not written: {repo_relative(required_path)}")
    if args.smoke and (not checkpoint_path.is_file() or not metrics_path.is_file()):
        raise A7A0ResidualDistillError("smoke run did not produce required checkpoint and metrics.json.")
    print(f"[T10-A7-A0-RESIDUAL] checkpoint: {repo_relative(checkpoint_path)}", flush=True)
    print(f"[T10-A7-A0-RESIDUAL] metrics: {repo_relative(metrics_path)}", flush=True)
    print(f"[T10-A7-A0-RESIDUAL] summary: {repo_relative(summary_path)}", flush=True)
    return 0


def main() -> int:
    try:
        args = build_parser().parse_args()
        return train(args)
    except (A7A0ResidualDistillError, A2HistoryDistillError) as exc:
        print(f"[T10-A7-A0-RESIDUAL ERROR] {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
