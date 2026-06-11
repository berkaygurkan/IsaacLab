#!/usr/bin/env python3
"""A5 history-student residual distillation scaffold for T10 P2 velocity data.

This offline supervised trainer freezes an A2-history student, computes its
base action on deployment-safe ``student_obs`` history windows, and trains a
small residual model on ``teacher_action - a2_history_action``. It does not
launch Isaac Sim, run RL, train A7, use A0 actions, or touch task/P2 code.
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
    build_history_student_mlp,
    build_history_windows,
    finite_metric_dict,
    load_history_source_arrays,
    repo_relative,
    require_torch,
    resolve_repo_path,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_PATH = "papers/conference/datasets/t10_teacher_gap_p2_velocity_seed0/dataset.npz"
DEFAULT_A2_CHECKPOINT = "papers/conference/results/t10_a2_student_history_distill_full_h16_seed0/a2_student_history.pt"
DEFAULT_CONFIG_PATH = "configs/train/a5_history_residual_distill.yaml"
DEFAULT_OUTPUT_DIR = "papers/conference/results/t10_a5_history_residual_distill_smoke"
DEFAULT_HISTORY_LEN = 16
DEFAULT_SMOKE_EPOCHS = 3
DEFAULT_BATCH_SIZE = 4096
DEFAULT_SEED = 0
DEFAULT_RESIDUAL_INPUT_MODE = "history_plus_base_action"


class A5HistoryResidualDistillError(ValueError):
    """Raised for invalid A5 residual distillation configuration or data."""


def build_residual_mlp(history_len: int, input_dim: int, action_dim: int, *, residual_input_mode: str) -> Any:
    _, nn_module, _, _ = require_torch()
    if residual_input_mode != DEFAULT_RESIDUAL_INPUT_MODE:
        raise A5HistoryResidualDistillError(f"unsupported residual_input_mode: {residual_input_mode!r}")
    residual_input_dim = history_len * input_dim + action_dim

    class HistoryResidualMLP(nn_module.Module):
        """Small local flatten-history residual MLP for A5 smoke distillation."""

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

        def forward(self, obs_history: Any, base_action: Any) -> Any:
            if obs_history.ndim != 3:
                raise A5HistoryResidualDistillError(
                    f"expected obs_history [batch, history, obs_dim], got {tuple(obs_history.shape)}"
                )
            if base_action.ndim != 2 or base_action.shape[-1] != action_dim:
                raise A5HistoryResidualDistillError(
                    f"expected base_action [batch, {action_dim}], got {tuple(base_action.shape)}"
                )
            flat_history = obs_history.flatten(start_dim=1)
            return self.net(torch_cat((flat_history, base_action)))

    def torch_cat(values: tuple[Any, ...]) -> Any:
        torch_module, _, _, _ = require_torch()
        return torch_module.cat(values, dim=1)

    return HistoryResidualMLP()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline A5 history-student residual distillation trainer.")
    parser.add_argument("--dataset_path", default=DEFAULT_DATASET_PATH)
    parser.add_argument("--a2_checkpoint", default=DEFAULT_A2_CHECKPOINT)
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
        raise A5HistoryResidualDistillError("--device cuda requested but CUDA is not available.")
    return torch_module.device(device_arg)


def validate_args(args: argparse.Namespace) -> None:
    dataset_path = resolve_repo_path(args.dataset_path)
    if not dataset_path.is_file():
        raise A5HistoryResidualDistillError(f"dataset_path does not exist: {repo_relative(dataset_path)}")
    checkpoint_path = resolve_repo_path(args.a2_checkpoint)
    if not checkpoint_path.is_file():
        raise A5HistoryResidualDistillError(f"a2_checkpoint does not exist: {repo_relative(checkpoint_path)}")
    config_path = resolve_repo_path(args.config)
    if not config_path.is_file():
        raise A5HistoryResidualDistillError(f"config does not exist: {repo_relative(config_path)}")
    if args.history_len < 1:
        raise A5HistoryResidualDistillError("--history_len must be >= 1.")
    if args.batch_size <= 0:
        raise A5HistoryResidualDistillError("--batch_size must be > 0.")
    if args.seed < 0:
        raise A5HistoryResidualDistillError("--seed must be non-negative.")
    if args.max_epochs is not None and args.max_epochs <= 0:
        raise A5HistoryResidualDistillError("--max_epochs must be > 0 when provided.")


def assert_finite_tensor(name: str, tensor: Any) -> None:
    torch_module, _, _, _ = require_torch()
    if not torch_module.isfinite(tensor).all():
        raise A5HistoryResidualDistillError(f"{name} contains NaN or Inf.")


def check_finite_array(name: str, array: np.ndarray) -> None:
    if not np.isfinite(array).all():
        raise A5HistoryResidualDistillError(f"{name} contains NaN or Inf.")


def torch_load_checkpoint(path: Path, *, map_location: str = "cpu") -> dict[str, Any]:
    torch_module, _, _, _ = require_torch()
    try:
        checkpoint = torch_module.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        checkpoint = torch_module.load(path, map_location=map_location)
    if not isinstance(checkpoint, dict):
        raise A5HistoryResidualDistillError(f"A2 checkpoint must be a dict, got {type(checkpoint).__name__}.")
    return checkpoint


def load_frozen_a2_history_model(checkpoint_path: Path, *, history_len: int, device: Any) -> tuple[Any, dict[str, Any]]:
    checkpoint = torch_load_checkpoint(checkpoint_path)
    metrics = checkpoint.get("metrics")
    if not isinstance(metrics, dict):
        raise A5HistoryResidualDistillError("A2 checkpoint missing metrics metadata.")
    guardrails = checkpoint.get("guardrails")
    if not isinstance(guardrails, dict):
        raise A5HistoryResidualDistillError("A2 checkpoint missing guardrails metadata.")

    checkpoint_history_len = checkpoint.get("history_len", metrics.get("history_len"))
    if int(checkpoint_history_len) != int(history_len):
        raise A5HistoryResidualDistillError(
            f"A2 checkpoint history_len expected {history_len}, got {checkpoint_history_len}."
        )
    if metrics.get("input_dim") != EXPECTED_INPUT_DIM or checkpoint.get("input_dim") != EXPECTED_INPUT_DIM:
        raise A5HistoryResidualDistillError("A2 checkpoint input_dim does not match 61.")
    if metrics.get("action_dim") != EXPECTED_ACTION_DIM or checkpoint.get("action_dim") != EXPECTED_ACTION_DIM:
        raise A5HistoryResidualDistillError("A2 checkpoint action_dim does not match 8.")
    for key in ("teacher_obs_used_as_input", "true_fault_state_used", "health_token_used", "residual_head_used"):
        if bool(metrics.get(key, False)):
            raise A5HistoryResidualDistillError(f"A2 checkpoint guardrail violated: metrics.{key}=true")
    required_guardrails = {
        "no_teacher_obs_input": True,
        "no_true_fault_state": True,
        "no_health_token": True,
        "no_residual_head": True,
    }
    for key, expected in required_guardrails.items():
        if bool(guardrails.get(key)) is not expected:
            raise A5HistoryResidualDistillError(f"A2 checkpoint guardrail missing/false: {key}")

    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(state_dict, dict):
        raise A5HistoryResidualDistillError("A2 checkpoint missing model_state_dict.")
    model = build_history_student_mlp(history_len, EXPECTED_INPUT_DIM, EXPECTED_ACTION_DIM)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    metadata = {
        "a2_checkpoint_path": repo_relative(checkpoint_path),
        "a2_checkpoint_model_class": checkpoint.get("model_class"),
        "a2_checkpoint_history_len": int(checkpoint_history_len),
        "a2_checkpoint_guardrails_ok": True,
    }
    return model, metadata


def compute_base_actions(
    a2_model: Any,
    history_inputs: np.ndarray,
    *,
    batch_size: int,
    device: Any,
) -> np.ndarray:
    torch_module, _, _, _ = require_torch()
    base_actions = np.empty((history_inputs.shape[0], EXPECTED_ACTION_DIM), dtype=np.float32)
    with torch_module.inference_mode():
        for start in range(0, history_inputs.shape[0], batch_size):
            end = min(start + batch_size, history_inputs.shape[0])
            batch = torch_module.from_numpy(history_inputs[start:end]).to(device)
            assert_finite_tensor("a2_base_input_history", batch)
            pred = a2_model(batch).detach().float()
            if pred.ndim != 2 or pred.shape[-1] != EXPECTED_ACTION_DIM:
                raise A5HistoryResidualDistillError(f"A2 base action expected [batch, 8], got {tuple(pred.shape)}.")
            assert_finite_tensor("a2_base_action", pred)
            base_actions[start:end] = pred.cpu().numpy()
    check_finite_array("a2_base_actions", base_actions)
    return base_actions


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
        raise A5HistoryResidualDistillError("dataset must contain at least two valid history samples.")
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
    torch_module, _, _, _ = require_torch()
    model.eval()
    residual_mse_sum = 0.0
    residual_mae_sum = 0.0
    reconstructed_mse_sum = 0.0
    reconstructed_mae_sum = 0.0
    sample_count = 0
    with torch_module.inference_mode():
        for obs_history, base_action, teacher_action, residual_target in loader:
            obs_history = obs_history.to(device)
            base_action = base_action.to(device)
            teacher_action = teacher_action.to(device)
            residual_target = residual_target.to(device)
            for name, tensor in (
                ("eval_obs_history", obs_history),
                ("eval_base_action", base_action),
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
        raise A5HistoryResidualDistillError("evaluation loader produced zero samples.")
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
        f"# T10 A5 History Residual Distillation {run_label}",
        "",
        "## Purpose",
        "",
        f"Offline supervised {run_description} run for the A5 history-student teacher-gap residual distillation scaffold.",
        "",
        "## Dataset",
        "",
        f"- dataset: `{metrics['dataset_path']}`",
        f"- samples: `{metrics['num_samples']}`",
        f"- train samples: `{metrics['train_samples']}`",
        f"- validation samples: `{metrics['val_samples']}`",
        "",
        "## Frozen Base",
        "",
        f"- A2 checkpoint: `{metrics['a2_checkpoint_path']}`",
        f"- A2 checkpoint guardrails ok: `{metrics['a2_checkpoint_guardrails_ok']}`",
        "",
        "## Mapping",
        "",
        f"- input: `student_obs` history window with shape `[batch, {metrics['history_len']}, {metrics['input_dim']}]`",
        f"- residual_input_mode: `{metrics['residual_input_mode']}`",
        "- residual target: `teacher_action - a2_history_action`",
        "- reconstruction: `a2_history_action + predicted_residual`",
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
        f"- base_a2_action_mse_to_teacher: `{metrics['base_a2_action_mse_to_teacher']}`",
        f"- no_nan_inf: `{metrics['no_nan_inf']}`",
        "",
        "## Guardrails",
        "",
        "- This is not paper-grade final evidence.",
        "- This is only a supervised residual distillation scaffold.",
        "- A7 residuals are not trained here.",
        "- A0 actions, teacher observations, true fault state, and health token are not used.",
        "- Deployment-facing policies must not use `true_fault_state`.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def train(args: argparse.Namespace) -> int:
    validate_args(args)
    set_seed(args.seed)
    torch_module, nn_module, data_loader_class, tensor_dataset_class = require_torch()
    device = select_device(args.device)
    dataset_path = resolve_repo_path(args.dataset_path)
    a2_checkpoint_path = resolve_repo_path(args.a2_checkpoint)
    output_dir = resolve_repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")

    max_epochs = args.max_epochs if args.max_epochs is not None else (DEFAULT_SMOKE_EPOCHS if args.smoke else 25)
    a2_model, a2_metadata = load_frozen_a2_history_model(a2_checkpoint_path, history_len=args.history_len, device=device)
    student_obs, teacher_action, done, episode_id, dataset_meta = load_history_source_arrays(dataset_path)
    history_inputs, teacher_actions, window_counts = build_history_windows(
        student_obs,
        teacher_action,
        done,
        episode_id,
        history_len=args.history_len,
    )
    base_actions = compute_base_actions(a2_model, history_inputs, batch_size=args.batch_size, device=device)
    residual_targets = teacher_actions - base_actions
    check_finite_array("residual_targets", residual_targets)
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
        ("train_base_action", train_base),
        ("train_teacher_action", train_teacher),
        ("train_residual_target", train_residual),
        ("val_obs_history", val_obs),
        ("val_base_action", val_base),
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
                ("batch_base_action", batch_base),
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
            raise A5HistoryResidualDistillError("training loader produced zero samples.")
        train_epoch_residual_mse = loss_sum / sample_count
        val_residual_mse, val_residual_mae, val_recon_mse, val_recon_mae = evaluate_residual_model(
            residual_model,
            val_loader,
            device=device,
        )
        epoch_values = (train_epoch_residual_mse, val_residual_mse, val_residual_mae, val_recon_mse, val_recon_mae)
        if not all(math.isfinite(value) for value in epoch_values):
            raise A5HistoryResidualDistillError("epoch metrics contain NaN or Inf.")
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
            f"[T10-A5-HISTORY-RESIDUAL] epoch={epoch}/{max_epochs} "
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
        "stage": "a5_history_residual_distillation",
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
        "action_composition": "a_final = a2_action + alpha * residual_action",
        "alpha_default_for_later_eval": 1.0,
        "train_residual_mse": train_residual_mse,
        "val_residual_mse": val_residual_mse,
        "train_residual_mae": train_residual_mae,
        "val_residual_mae": val_residual_mae,
        "train_reconstructed_action_mse": train_recon_mse,
        "val_reconstructed_action_mse": val_recon_mse,
        "train_reconstructed_action_mae": train_recon_mae,
        "val_reconstructed_action_mae": val_recon_mae,
        "base_a2_action_mse_to_teacher": float(base_sample_mse.mean()),
        "base_a2_action_mae_to_teacher": float(base_sample_mae.mean()),
        "num_samples": int(history_inputs.shape[0]),
        "train_samples": int(train_obs.shape[0]),
        "val_samples": int(val_obs.shape[0]),
        "input_dim": dataset_meta["input_dim"],
        "action_dim": dataset_meta["action_dim"],
        "residual_dim": EXPECTED_ACTION_DIM,
        "residual_target_mean_norm": float(residual_norm.mean()),
        "residual_target_max_norm": float(residual_norm.max()),
        "input_key": "student_obs_history",
        "target_key": "teacher_action_minus_a2_history_action",
        "teacher_obs_used_as_input": False,
        "true_fault_state_used": False,
        "health_token_used": False,
        "residual_head_used": True,
        "a0_action_used": False,
        "no_nan_inf": True,
        "epoch_metrics": epoch_metrics,
        "source_student_obs_shape": dataset_meta["source_student_obs_shape"],
        "source_teacher_action_shape": dataset_meta["source_teacher_action_shape"],
        "source_done_shape": dataset_meta["source_done_shape"],
        "source_episode_id_shape": dataset_meta["source_episode_id_shape"],
        "model_class": "HistoryResidualMLP",
        "base_policy": "A2_history_H16",
        "not_paper_grade_final": True,
        **a2_metadata,
        **window_counts,
    }
    finite_metric_dict(metrics)
    checkpoint_path = output_dir / ("a5_history_residual_smoke.pt" if args.smoke else "a5_history_residual.pt")
    torch_module.save(
        {
            "model_state_dict": residual_model.state_dict(),
            "metrics": metrics,
            "model_class": "HistoryResidualMLP",
            "history_len": args.history_len,
            "input_dim": EXPECTED_INPUT_DIM,
            "action_dim": EXPECTED_ACTION_DIM,
            "residual_dim": EXPECTED_ACTION_DIM,
            "residual_input_mode": DEFAULT_RESIDUAL_INPUT_MODE,
            "a2_checkpoint_path": repo_relative(a2_checkpoint_path),
            "guardrails": {
                "input": "student_obs_history",
                "target": "teacher_action_minus_a2_history_action",
                "no_teacher_obs_input": True,
                "no_true_fault_state": True,
                "no_health_token": True,
                "residual_head": True,
                "no_a0_action": True,
                "no_a7_training": True,
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
            raise A5HistoryResidualDistillError(f"required output was not written: {repo_relative(required_path)}")
    if args.smoke and (not checkpoint_path.is_file() or not metrics_path.is_file()):
        raise A5HistoryResidualDistillError("smoke run did not produce required checkpoint and metrics.json.")
    print(f"[T10-A5-HISTORY-RESIDUAL] checkpoint: {repo_relative(checkpoint_path)}", flush=True)
    print(f"[T10-A5-HISTORY-RESIDUAL] metrics: {repo_relative(metrics_path)}", flush=True)
    print(f"[T10-A5-HISTORY-RESIDUAL] summary: {repo_relative(summary_path)}", flush=True)
    return 0


def main() -> int:
    try:
        args = build_parser().parse_args()
        return train(args)
    except (A5HistoryResidualDistillError, A2HistoryDistillError) as exc:
        print(f"[T10-A5-HISTORY-RESIDUAL ERROR] {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
