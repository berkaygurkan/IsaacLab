#!/usr/bin/env python3
"""Offline T16 A5 history residual distillation for multi-joint P2.

A5 in this conference-stage pipeline freezes the T15 A2-history H16 student,
computes its base action on deployment-facing student observation histories,
and trains a residual model against ``teacher_action - base_action``.

The residual model input is only the H x 61 student observation history. It
does not consume teacher observations, selected fault IDs, fault one-hot
vectors, q-lock vectors, P2-active flags, health tokens, A0 actions, or A7
signals. This script is offline-only and never launches Isaac Sim or RL.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_PATHS = [
    "papers/conference/datasets/t14_multijoint_teacher_v2b_realistic_seed0/dataset.npz",
    "papers/conference/datasets/t14_multijoint_teacher_v2b_late_seed0/dataset.npz",
]
DEFAULT_BASE_CHECKPOINT = (
    "papers/conference/results/t15_a2_history_h16_multijoint_seed0/a2_history_h16_multijoint.pt"
)
DEFAULT_OUTPUT_DIR = "papers/conference/results/t16_a5_history_residual_multijoint_seed0"
EXPECTED_STUDENT_OBS_DIM = 61
EXPECTED_TEACHER_OBS_DIM = 77
EXPECTED_ACTION_DIM = 8
DEFAULT_HISTORY_LEN = 16
DEFAULT_SEED = 0
DEFAULT_EPOCHS = 100
DEFAULT_BATCH_SIZE = 4096
DEFAULT_LR = 1.0e-3
DEFAULT_VAL_FRACTION = 0.1
DEFAULT_ALPHA = 1.0
RESIDUAL_INPUT_MODE = "history_only_student_obs"

FORBIDDEN_INPUT_FIELDS = [
    "teacher_obs",
    "selected_fault_joint_index",
    "selected_fault_joint_one_hot",
    "q_lock_vector",
    "p2_fault_active",
    "health_token",
]

torch: Any | None = None
nn: Any | None = None
DataLoader: Any | None = None
TensorDataset: Any | None = None
DatasetBase: Any | None = None


class T16A5ResidualError(ValueError):
    """Raised for invalid T16 A5 residual distillation state."""


@dataclass
class LoadedArrays:
    student_obs: np.ndarray
    teacher_action: np.ndarray
    done: np.ndarray
    episode_id: np.ndarray
    env_id: np.ndarray
    timestep: np.ndarray
    source_dataset_index: np.ndarray
    protocol_label: np.ndarray
    velocity_mode_label: np.ndarray
    source_counts: list[dict[str, Any]]
    source_shapes: list[dict[str, Any]]
    forbidden_fields_present: list[str]


@dataclass
class HistoryPlan:
    windows: np.ndarray
    targets: np.ndarray
    labels: np.ndarray
    train_indices: np.ndarray
    val_indices: np.ndarray
    split_metadata: dict[str, Any]
    history_metadata: dict[str, Any]


@dataclass
class PreparedResidualData:
    train_dataset: Any
    val_dataset: Any
    train_samples: int
    val_samples: int
    num_samples: int
    train_base_actions: Any
    train_teacher_actions: Any
    val_base_actions: Any
    val_teacher_actions: Any
    base_metadata: dict[str, Any]


def require_torch() -> tuple[Any, Any, Any, Any, Any]:
    global DataLoader, DatasetBase, TensorDataset, nn, torch
    if torch is None:
        try:
            import torch as torch_module
            from torch import nn as nn_module
            from torch.utils.data import DataLoader as data_loader_class
            from torch.utils.data import Dataset as dataset_class
            from torch.utils.data import TensorDataset as tensor_dataset_class
        except ModuleNotFoundError as exc:
            raise T16A5ResidualError(
                "PyTorch is required for T16 A5 residual distillation. Activate the isaaclab environment."
            ) from exc
        torch = torch_module
        nn = nn_module
        DataLoader = data_loader_class
        DatasetBase = dataset_class
        TensorDataset = tensor_dataset_class
    return torch, nn, DataLoader, TensorDataset, DatasetBase


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
    parser = argparse.ArgumentParser(
        description="Offline T16 multi-joint A5 history residual distillation trainer."
    )
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--dry_run", action="store_true", help="Inspect data and split plan without training.")
    mode_group.add_argument("--execute_train", action="store_true", help="Run offline supervised residual training.")
    parser.add_argument("--dataset_paths", nargs="+", default=DEFAULT_DATASET_PATHS)
    parser.add_argument("--base_checkpoint", default=DEFAULT_BASE_CHECKPOINT)
    parser.add_argument("--history_len", type=int, default=DEFAULT_HISTORY_LEN)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", default="cuda", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--val_fraction", type=float, default=DEFAULT_VAL_FRACTION)
    parser.add_argument("--save_every", type=int, default=0, help="Save epoch checkpoints every N epochs; 0 disables.")
    parser.add_argument(
        "--alpha",
        type=float,
        default=DEFAULT_ALPHA,
        help="Offline reconstruction alpha for metrics; closed-loop alpha sweeps are later evaluation work.",
    )
    parser.add_argument("--require_history_len", type=int, default=None)
    parser.add_argument("--require_dataset_control_frequency_hz", type=float, default=None)
    parser.add_argument("--require_dataset_physics_frequency_hz", type=float, default=None)
    return parser


def validate_dataset_timing_metadata(args: argparse.Namespace) -> None:
    for dataset_path in args.dataset_paths:
        metadata_path = resolve_repo_path(dataset_path).with_name("metadata.json")
        if args.require_dataset_control_frequency_hz is None and args.require_dataset_physics_frequency_hz is None:
            continue
        if not metadata_path.is_file():
            raise T16A5ResidualError(f"required dataset timing metadata is missing: {repo_relative(metadata_path)}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if args.require_dataset_control_frequency_hz is not None:
            observed = float(metadata.get("control_frequency_hz"))
            if abs(observed - float(args.require_dataset_control_frequency_hz)) > 1.0e-4:
                raise T16A5ResidualError(
                    f"dataset {repo_relative(dataset_path)} control_frequency_hz expected "
                    f"{args.require_dataset_control_frequency_hz}, got {observed}."
                )
        if args.require_dataset_physics_frequency_hz is not None:
            observed = float(metadata.get("physics_frequency_hz"))
            if abs(observed - float(args.require_dataset_physics_frequency_hz)) > 1.0e-4:
                raise T16A5ResidualError(
                    f"dataset {repo_relative(dataset_path)} physics_frequency_hz expected "
                    f"{args.require_dataset_physics_frequency_hz}, got {observed}."
                )


def validate_args(args: argparse.Namespace) -> None:
    if not args.dataset_paths:
        raise T16A5ResidualError("--dataset_paths must contain at least one dataset.")
    for dataset_path in args.dataset_paths:
        resolved = resolve_repo_path(dataset_path)
        if not resolved.is_file():
            raise T16A5ResidualError(f"dataset_path does not exist: {repo_relative(resolved)}")
    base_checkpoint = resolve_repo_path(args.base_checkpoint)
    if not base_checkpoint.is_file():
        raise T16A5ResidualError(f"base_checkpoint does not exist: {repo_relative(base_checkpoint)}")
    if args.history_len < 1:
        raise T16A5ResidualError("--history_len must be >= 1.")
    if args.require_history_len is not None and int(args.history_len) != int(args.require_history_len):
        raise T16A5ResidualError(f"--history_len must equal --require_history_len={args.require_history_len}.")
    if args.epochs <= 0:
        raise T16A5ResidualError("--epochs must be > 0.")
    if args.batch_size <= 0:
        raise T16A5ResidualError("--batch_size must be > 0.")
    if args.lr <= 0.0:
        raise T16A5ResidualError("--lr must be > 0.")
    if not 0.0 < args.val_fraction < 1.0:
        raise T16A5ResidualError("--val_fraction must be in (0, 1).")
    if args.save_every < 0:
        raise T16A5ResidualError("--save_every must be >= 0.")
    if not math.isfinite(args.alpha):
        raise T16A5ResidualError("--alpha must be finite.")
    validate_dataset_timing_metadata(args)


def set_seed(seed: int) -> None:
    torch_module, _, _, _, _ = require_torch()
    random.seed(seed)
    np.random.seed(seed)
    torch_module.manual_seed(seed)
    if torch_module.cuda.is_available():
        torch_module.cuda.manual_seed_all(seed)


def select_device(device_arg: str) -> Any:
    torch_module, _, _, _, _ = require_torch()
    if device_arg == "auto":
        return torch_module.device("cuda" if torch_module.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch_module.cuda.is_available():
        raise T16A5ResidualError("--device cuda requested but CUDA is not available.")
    return torch_module.device(device_arg)


def check_finite_array(name: str, array: np.ndarray) -> None:
    if array.dtype.kind in {"f", "i", "u"} and not np.isfinite(array).all():
        raise T16A5ResidualError(f"{name} contains NaN or Inf.")


def assert_finite_tensor(name: str, tensor: Any) -> None:
    torch_module, _, _, _, _ = require_torch()
    if not torch_module.isfinite(tensor).all():
        raise T16A5ResidualError(f"{name} contains NaN or Inf.")


def flatten_feature_array(array: np.ndarray, *, key: str, expected_dim: int) -> np.ndarray:
    if array.ndim == 3:
        if array.shape[-1] != expected_dim:
            raise T16A5ResidualError(f"{key} expected final dim {expected_dim}, got shape {array.shape}.")
        return array.reshape(-1, expected_dim).astype(np.float32, copy=False)
    if array.ndim == 2:
        if array.shape[-1] != expected_dim:
            raise T16A5ResidualError(f"{key} expected final dim {expected_dim}, got shape {array.shape}.")
        return array.astype(np.float32, copy=False)
    raise T16A5ResidualError(f"{key} expected 2-D or 3-D array, got shape {array.shape}.")


def flatten_scalar_array(array: np.ndarray, *, key: str) -> np.ndarray:
    if array.ndim in {1, 2}:
        return array.reshape(-1)
    raise T16A5ResidualError(f"{key} expected 1-D or 2-D array, got shape {array.shape}.")


def generated_scalar(shape_source: np.ndarray, *, key: str) -> np.ndarray:
    if shape_source.ndim == 3:
        time_steps, num_envs = shape_source.shape[:2]
        if key == "env_id":
            return np.tile(np.arange(num_envs, dtype=np.int64), time_steps)
        if key == "timestep":
            return np.repeat(np.arange(time_steps, dtype=np.int64), num_envs)
    raise T16A5ResidualError(f"{key} is required for flattened T14 datasets.")


def load_one_dataset(dataset_path: Path, *, source_index: int) -> tuple[dict[str, np.ndarray], dict[str, Any], list[str]]:
    with np.load(dataset_path, allow_pickle=False) as data:
        files = set(data.files)
        required = {"student_obs", "teacher_action", "done", "episode_id"}
        missing = sorted(required - files)
        if missing:
            raise T16A5ResidualError(f"{repo_relative(dataset_path)} missing required arrays: {missing}")

        forbidden_present = [field for field in FORBIDDEN_INPUT_FIELDS if field in files]
        student_obs_source = np.asarray(data["student_obs"])
        student_obs = flatten_feature_array(
            student_obs_source,
            key="student_obs",
            expected_dim=EXPECTED_STUDENT_OBS_DIM,
        )
        teacher_action = flatten_feature_array(
            np.asarray(data["teacher_action"]),
            key="teacher_action",
            expected_dim=EXPECTED_ACTION_DIM,
        )
        if student_obs.shape[0] != teacher_action.shape[0]:
            raise T16A5ResidualError(
                f"student_obs and teacher_action sample counts differ in {repo_relative(dataset_path)}: "
                f"{student_obs.shape[0]} vs {teacher_action.shape[0]}."
            )
        sample_count = int(student_obs.shape[0])
        done = flatten_scalar_array(np.asarray(data["done"]), key="done").astype(bool)
        episode_id = flatten_scalar_array(np.asarray(data["episode_id"]), key="episode_id")
        env_id = (
            flatten_scalar_array(np.asarray(data["env_id"]), key="env_id")
            if "env_id" in files
            else generated_scalar(student_obs_source, key="env_id")
        )
        timestep = (
            flatten_scalar_array(np.asarray(data["timestep"]), key="timestep")
            if "timestep" in files
            else generated_scalar(student_obs_source, key="timestep")
        )
        for key, array in (("done", done), ("episode_id", episode_id), ("env_id", env_id), ("timestep", timestep)):
            if int(array.shape[0]) != sample_count:
                raise T16A5ResidualError(
                    f"{key} sample count mismatch in {repo_relative(dataset_path)}: "
                    f"{array.shape[0]} vs {sample_count}."
                )
        protocol_label = (
            flatten_scalar_array(np.asarray(data["protocol_label"]), key="protocol_label").astype(str)
            if "protocol_label" in files
            else np.full(sample_count, dataset_path.parent.name, dtype="U64")
        )
        velocity_mode_label = (
            flatten_scalar_array(np.asarray(data["velocity_mode_label"]), key="velocity_mode_label").astype(str)
            if "velocity_mode_label" in files
            else np.full(sample_count, "unknown", dtype="U32")
        )
        teacher_obs_shape = list(data["teacher_obs"].shape) if "teacher_obs" in files else None
        teacher_obs_final_dim = (
            int(data["teacher_obs"].shape[-1])
            if "teacher_obs" in files and data["teacher_obs"].ndim >= 2
            else None
        )
        if teacher_obs_final_dim is not None and teacher_obs_final_dim != EXPECTED_TEACHER_OBS_DIM:
            raise T16A5ResidualError(
                f"teacher_obs final dim expected {EXPECTED_TEACHER_OBS_DIM} in {repo_relative(dataset_path)}, "
                f"got {teacher_obs_final_dim}."
            )

    for key, array in (
        ("student_obs", student_obs),
        ("teacher_action", teacher_action),
        ("episode_id", episode_id),
        ("env_id", env_id),
        ("timestep", timestep),
    ):
        check_finite_array(key, array) if np.issubdtype(array.dtype, np.number) else None

    arrays = {
        "student_obs": student_obs,
        "teacher_action": teacher_action,
        "done": done,
        "episode_id": episode_id,
        "env_id": env_id,
        "timestep": timestep,
        "source_dataset_index": np.full(sample_count, source_index, dtype=np.int64),
        "protocol_label": protocol_label,
        "velocity_mode_label": velocity_mode_label,
    }
    source_shape = {
        "dataset_path": repo_relative(dataset_path),
        "sample_count": sample_count,
        "student_obs_shape": [int(dim) for dim in student_obs.shape],
        "teacher_action_shape": [int(dim) for dim in teacher_action.shape],
        "teacher_obs_shape": teacher_obs_shape,
        "teacher_obs_final_dim": teacher_obs_final_dim,
        "forbidden_fields_present": forbidden_present,
    }
    return arrays, source_shape, forbidden_present


def load_datasets(dataset_paths: list[str]) -> LoadedArrays:
    blocks: list[dict[str, np.ndarray]] = []
    source_shapes: list[dict[str, Any]] = []
    forbidden_fields: set[str] = set()
    for source_index, dataset_path_value in enumerate(dataset_paths):
        dataset_path = resolve_repo_path(dataset_path_value)
        arrays, source_shape, forbidden_present = load_one_dataset(dataset_path, source_index=source_index)
        blocks.append(arrays)
        source_shapes.append(source_shape)
        forbidden_fields.update(forbidden_present)

    concat = {key: np.concatenate([block[key] for block in blocks], axis=0) for key in blocks[0].keys()}
    source_counts = [
        {
            "dataset_path": shape["dataset_path"],
            "sample_count": shape["sample_count"],
        }
        for shape in source_shapes
    ]
    return LoadedArrays(
        student_obs=concat["student_obs"],
        teacher_action=concat["teacher_action"],
        done=concat["done"].astype(bool),
        episode_id=concat["episode_id"],
        env_id=concat["env_id"],
        timestep=concat["timestep"],
        source_dataset_index=concat["source_dataset_index"],
        protocol_label=concat["protocol_label"],
        velocity_mode_label=concat["velocity_mode_label"],
        source_counts=source_counts,
        source_shapes=source_shapes,
        forbidden_fields_present=sorted(forbidden_fields),
    )


def group_labels(arrays: LoadedArrays) -> np.ndarray:
    labels = [
        f"{int(source)}:{str(env)}:{str(episode)}"
        for source, env, episode in zip(arrays.source_dataset_index, arrays.env_id, arrays.episode_id)
    ]
    return np.asarray(labels, dtype="U96")


def split_indices_by_group(
    labels: np.ndarray,
    *,
    seed: int,
    val_fraction: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    sample_count = int(labels.shape[0])
    if sample_count < 2:
        raise T16A5ResidualError("at least two samples/windows are required for train/val split.")
    rng = np.random.default_rng(seed)
    unique_groups = np.unique(labels)
    if unique_groups.shape[0] >= 2:
        shuffled_groups = unique_groups.copy()
        rng.shuffle(shuffled_groups)
        val_group_count = max(1, int(round(unique_groups.shape[0] * val_fraction)))
        val_group_count = min(val_group_count, unique_groups.shape[0] - 1)
        val_groups = set(shuffled_groups[:val_group_count].tolist())
        val_mask = np.asarray([label in val_groups for label in labels], dtype=bool)
        val_indices = np.nonzero(val_mask)[0]
        train_indices = np.nonzero(~val_mask)[0]
        if train_indices.size > 0 and val_indices.size > 0:
            return train_indices, val_indices, {
                "split_strategy": "group_by_source_env_episode",
                "group_count": int(unique_groups.shape[0]),
                "train_group_count": int(unique_groups.shape[0] - val_group_count),
                "val_group_count": int(val_group_count),
                "temporal_leakage_note": "All windows from a source/env/episode group are assigned to one split.",
            }

    indices = rng.permutation(sample_count)
    val_count = max(1, int(round(sample_count * val_fraction)))
    val_count = min(val_count, sample_count - 1)
    return indices[val_count:], indices[:val_count], {
        "split_strategy": "sample_fallback",
        "group_count": int(unique_groups.shape[0]),
        "train_group_count": None,
        "val_group_count": None,
        "temporal_leakage_note": "Fallback split used because group split was not feasible.",
    }


def build_history_window_indices(
    arrays: LoadedArrays,
    *,
    history_len: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    groups: dict[str, list[int]] = {}
    labels = group_labels(arrays)
    for index, label in enumerate(labels):
        groups.setdefault(str(label), []).append(index)

    window_indices: list[list[int]] = []
    target_indices: list[int] = []
    window_labels: list[str] = []
    candidate_windows = 0
    rejected_done_crossing_windows = 0
    rejected_short_groups = 0

    for label, indices in groups.items():
        sorted_indices = sorted(indices, key=lambda idx: int(arrays.timestep[idx]))
        if len(sorted_indices) < history_len:
            rejected_short_groups += 1
            continue
        candidate_windows += len(sorted_indices) - history_len + 1
        for local_target in range(history_len - 1, len(sorted_indices)):
            start = local_target - history_len + 1
            window = sorted_indices[start : local_target + 1]
            if bool(np.any(arrays.done[window[:-1]])):
                rejected_done_crossing_windows += 1
                continue
            window_indices.append(window)
            target_indices.append(sorted_indices[local_target])
            window_labels.append(label)

    if not window_indices:
        raise T16A5ResidualError("history mode produced zero valid windows.")
    windows = np.asarray(window_indices, dtype=np.int64)
    targets = np.asarray(target_indices, dtype=np.int64)
    labels_array = np.asarray(window_labels, dtype="U96")
    metadata = {
        "history_len": history_len,
        "total_groups": len(groups),
        "rejected_short_groups": int(rejected_short_groups),
        "total_candidate_windows": int(candidate_windows),
        "valid_windows": int(windows.shape[0]),
        "rejected_done_crossing_windows": int(rejected_done_crossing_windows),
        "rejected_cross_episode_windows": 0,
        "history_input_shape": [int(windows.shape[0]), history_len, EXPECTED_STUDENT_OBS_DIM],
        "target_shape": [int(targets.shape[0]), EXPECTED_ACTION_DIM],
        "window_guardrail": "Windows are built within source/env/episode groups and reject prior done crossings.",
    }
    return windows, targets, labels_array, metadata


def prepare_history_plan(args: argparse.Namespace, arrays: LoadedArrays) -> HistoryPlan:
    windows, targets, window_labels, history_metadata = build_history_window_indices(
        arrays,
        history_len=args.history_len,
    )
    train_indices, val_indices, split_metadata = split_indices_by_group(
        window_labels,
        seed=args.seed,
        val_fraction=args.val_fraction,
    )
    return HistoryPlan(
        windows=windows,
        targets=targets,
        labels=window_labels,
        train_indices=train_indices,
        val_indices=val_indices,
        split_metadata=split_metadata,
        history_metadata=history_metadata,
    )


def build_a2_history_base_model(history_len: int, input_dim: int, action_dim: int) -> Any:
    _, nn_module, _, _, _ = require_torch()

    class A2HistoryBaseMLP(nn_module.Module):
        def __init__(self) -> None:
            super().__init__()
            self.flatten = nn_module.Flatten(start_dim=1)
            self.encoder = nn_module.Sequential(
                nn_module.Linear(history_len * input_dim, 512),
                nn_module.ELU(),
                nn_module.Linear(512, 256),
                nn_module.ELU(),
                nn_module.Linear(256, 128),
                nn_module.ELU(),
            )
            self.action_head = nn_module.Linear(128, action_dim)

        def encode(self, obs_history: Any) -> Any:
            return self.encoder(self.flatten(obs_history))

        def forward(self, obs_history: Any) -> Any:
            return self.action_head(self.encode(obs_history))

    return A2HistoryBaseMLP()


def build_residual_model(history_len: int, input_dim: int, residual_dim: int) -> Any:
    _, nn_module, _, _, _ = require_torch()

    class A5HistoryResidualMLP(nn_module.Module):
        def __init__(self) -> None:
            super().__init__()
            self.flatten = nn_module.Flatten(start_dim=1)
            self.encoder = nn_module.Sequential(
                nn_module.Linear(history_len * input_dim, 512),
                nn_module.ELU(),
                nn_module.Linear(512, 256),
                nn_module.ELU(),
                nn_module.Linear(256, 128),
                nn_module.ELU(),
            )
            self.residual_head = nn_module.Linear(128, residual_dim)

        def encode(self, obs_history: Any) -> Any:
            return self.encoder(self.flatten(obs_history))

        def forward(self, obs_history: Any) -> Any:
            return self.residual_head(self.encode(obs_history))

    return A5HistoryResidualMLP()


def torch_load_checkpoint(path: Path, *, map_location: str = "cpu") -> dict[str, Any]:
    torch_module, _, _, _, _ = require_torch()
    try:
        checkpoint = torch_module.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        checkpoint = torch_module.load(path, map_location=map_location)
    if not isinstance(checkpoint, dict):
        raise T16A5ResidualError(f"base checkpoint must be a dict, got {type(checkpoint).__name__}.")
    return checkpoint


def load_frozen_a2_history_model(checkpoint_path: Path, *, history_len: int, device: Any) -> tuple[Any, dict[str, Any]]:
    checkpoint = torch_load_checkpoint(checkpoint_path)
    metrics = checkpoint.get("metrics", {})
    if not isinstance(metrics, dict):
        raise T16A5ResidualError("base checkpoint metrics must be a dict when present.")
    mode = checkpoint.get("mode", metrics.get("mode"))
    if mode != "history":
        raise T16A5ResidualError(f"base checkpoint must be a T15 A2-history model, got mode={mode!r}.")
    checkpoint_history_len = checkpoint.get("history_len", metrics.get("history_len"))
    if int(checkpoint_history_len) != int(history_len):
        raise T16A5ResidualError(
            f"base checkpoint history_len expected {history_len}, got {checkpoint_history_len}."
        )
    input_dim = checkpoint.get("input_dim", metrics.get("input_dim"))
    target_dim = checkpoint.get("target_dim", metrics.get("target_dim", metrics.get("action_dim")))
    if int(input_dim) != EXPECTED_STUDENT_OBS_DIM:
        raise T16A5ResidualError(f"base checkpoint input_dim expected 61, got {input_dim}.")
    if int(target_dim) != EXPECTED_ACTION_DIM:
        raise T16A5ResidualError(f"base checkpoint action/target dim expected 8, got {target_dim}.")
    for key in (
        "teacher_obs_used_as_input",
        "selected_fault_joint_index_used_as_input",
        "selected_fault_joint_one_hot_used_as_input",
        "q_lock_vector_used_as_input",
        "p2_fault_active_used_as_input",
        "health_token_used",
    ):
        if bool(metrics.get(key, False)):
            raise T16A5ResidualError(f"base checkpoint guardrail violated: {key}=true")
    guardrails = checkpoint.get("guardrails", {})
    if isinstance(guardrails, dict):
        if bool(guardrails.get("teacher_obs_used_as_input", False)):
            raise T16A5ResidualError("base checkpoint guardrail violated: teacher_obs_used_as_input=true")
        if bool(guardrails.get("health_token_used", False)):
            raise T16A5ResidualError("base checkpoint guardrail violated: health_token_used=true")

    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(state_dict, dict):
        raise T16A5ResidualError("base checkpoint missing model_state_dict.")
    model = build_a2_history_base_model(history_len, EXPECTED_STUDENT_OBS_DIM, EXPECTED_ACTION_DIM)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    metadata = {
        "base_checkpoint_path": repo_relative(checkpoint_path),
        "base_checkpoint_mode": mode,
        "base_checkpoint_history_len": int(checkpoint_history_len),
        "base_checkpoint_guardrails_ok": True,
        "base_policy": "T15 A2-history H16 student",
    }
    return model, metadata


def make_history_input_dataset(obs_tensor: Any, action_tensor: Any, window_indices: Any, target_indices: Any) -> Any:
    _, _, _, _, dataset_base = require_torch()

    class HistoryInputDataset(dataset_base):
        def __init__(self) -> None:
            self.obs = obs_tensor
            self.actions = action_tensor
            self.window_indices = window_indices
            self.target_indices = target_indices

        def __len__(self) -> int:
            return int(self.target_indices.shape[0])

        def __getitem__(self, index: int) -> tuple[Any, Any]:
            return self.obs[self.window_indices[index]], self.actions[self.target_indices[index]]

    return HistoryInputDataset()


def make_residual_dataset(
    obs_tensor: Any,
    window_indices: Any,
    base_actions: Any,
    teacher_actions: Any,
    residual_targets: Any,
) -> Any:
    _, _, _, _, dataset_base = require_torch()

    class ResidualWindowDataset(dataset_base):
        def __init__(self) -> None:
            self.obs = obs_tensor
            self.window_indices = window_indices
            self.base_actions = base_actions
            self.teacher_actions = teacher_actions
            self.residual_targets = residual_targets

        def __len__(self) -> int:
            return int(self.window_indices.shape[0])

        def __getitem__(self, index: int) -> tuple[Any, Any, Any, Any]:
            return (
                self.obs[self.window_indices[index]],
                self.base_actions[index],
                self.teacher_actions[index],
                self.residual_targets[index],
            )

    return ResidualWindowDataset()


def compute_base_actions(
    base_model: Any,
    history_dataset: Any,
    *,
    batch_size: int,
    device: Any,
) -> tuple[Any, Any]:
    torch_module, _, data_loader_class, _, _ = require_torch()
    loader = data_loader_class(history_dataset, batch_size=batch_size, shuffle=False)
    base_actions: list[Any] = []
    teacher_actions: list[Any] = []
    with torch_module.inference_mode():
        for obs_history, teacher_action in loader:
            obs_history = obs_history.to(device)
            assert_finite_tensor("base_model_obs_history", obs_history)
            pred = base_model(obs_history).detach().float()
            if pred.ndim != 2 or pred.shape[-1] != EXPECTED_ACTION_DIM:
                raise T16A5ResidualError(f"base action expected [batch, 8], got {tuple(pred.shape)}.")
            assert_finite_tensor("base_action", pred)
            assert_finite_tensor("teacher_action", teacher_action)
            base_actions.append(pred.cpu())
            teacher_actions.append(teacher_action.detach().float().cpu())
    if not base_actions:
        raise T16A5ResidualError("base-action loader produced zero samples.")
    base_tensor = torch_module.cat(base_actions, dim=0)
    teacher_tensor = torch_module.cat(teacher_actions, dim=0)
    assert_finite_tensor("base_actions_all", base_tensor)
    assert_finite_tensor("teacher_actions_all", teacher_tensor)
    return base_tensor, teacher_tensor


def pair_metrics(pred: Any, target: Any) -> tuple[float, float]:
    diff = pred - target
    mse = diff.square().mean(dim=1)
    mae = diff.abs().mean(dim=1)
    return float(mse.mean().item()), float(mae.mean().item())


def prepare_residual_data(
    args: argparse.Namespace,
    arrays: LoadedArrays,
    history_plan: HistoryPlan,
    base_model: Any,
    *,
    device: Any,
) -> PreparedResidualData:
    torch_module, _, _, _, _ = require_torch()
    obs_tensor = torch_module.from_numpy(arrays.student_obs.astype(np.float32, copy=False))
    action_tensor = torch_module.from_numpy(arrays.teacher_action.astype(np.float32, copy=False))
    window_tensor = torch_module.from_numpy(history_plan.windows)
    target_tensor = torch_module.from_numpy(history_plan.targets)
    train_window_tensor = window_tensor[torch_module.from_numpy(history_plan.train_indices)]
    train_target_tensor = target_tensor[torch_module.from_numpy(history_plan.train_indices)]
    val_window_tensor = window_tensor[torch_module.from_numpy(history_plan.val_indices)]
    val_target_tensor = target_tensor[torch_module.from_numpy(history_plan.val_indices)]

    train_history_dataset = make_history_input_dataset(
        obs_tensor,
        action_tensor,
        train_window_tensor,
        train_target_tensor,
    )
    val_history_dataset = make_history_input_dataset(
        obs_tensor,
        action_tensor,
        val_window_tensor,
        val_target_tensor,
    )
    train_base, train_teacher = compute_base_actions(
        base_model,
        train_history_dataset,
        batch_size=args.batch_size,
        device=device,
    )
    val_base, val_teacher = compute_base_actions(
        base_model,
        val_history_dataset,
        batch_size=args.batch_size,
        device=device,
    )
    train_residual = train_teacher - train_base
    val_residual = val_teacher - val_base
    for name, tensor in (
        ("train_base_actions", train_base),
        ("train_teacher_actions", train_teacher),
        ("train_residual_targets", train_residual),
        ("val_base_actions", val_base),
        ("val_teacher_actions", val_teacher),
        ("val_residual_targets", val_residual),
    ):
        assert_finite_tensor(name, tensor)

    train_base_mse, train_base_mae = pair_metrics(train_base, train_teacher)
    val_base_mse, val_base_mae = pair_metrics(val_base, val_teacher)
    train_residual_norm = train_residual.norm(dim=1)
    val_residual_norm = val_residual.norm(dim=1)
    base_metadata = {
        "precomputed_base_action_train_mse": train_base_mse,
        "precomputed_base_action_train_mae": train_base_mae,
        "precomputed_base_action_val_mse": val_base_mse,
        "precomputed_base_action_val_mae": val_base_mae,
        "residual_target_train_mean_norm": float(train_residual_norm.mean().item()),
        "residual_target_train_max_norm": float(train_residual_norm.max().item()),
        "residual_target_val_mean_norm": float(val_residual_norm.mean().item()),
        "residual_target_val_max_norm": float(val_residual_norm.max().item()),
    }

    train_dataset = make_residual_dataset(
        obs_tensor,
        train_window_tensor,
        train_base,
        train_teacher,
        train_residual,
    )
    val_dataset = make_residual_dataset(
        obs_tensor,
        val_window_tensor,
        val_base,
        val_teacher,
        val_residual,
    )
    return PreparedResidualData(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        train_samples=int(train_window_tensor.shape[0]),
        val_samples=int(val_window_tensor.shape[0]),
        num_samples=int(history_plan.windows.shape[0]),
        train_base_actions=train_base,
        train_teacher_actions=train_teacher,
        val_base_actions=val_base,
        val_teacher_actions=val_teacher,
        base_metadata=base_metadata,
    )


def evaluate_residual_model(model: Any, loader: Any, *, device: Any, alpha: float) -> dict[str, float]:
    torch_module, _, _, _, _ = require_torch()
    model.eval()
    residual_mse_sum = 0.0
    residual_mae_sum = 0.0
    reconstructed_mse_sum = 0.0
    reconstructed_mae_sum = 0.0
    base_mse_sum = 0.0
    base_mae_sum = 0.0
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
            pred_residual = model(obs_history)
            assert_finite_tensor("eval_pred_residual", pred_residual)
            reconstructed_action = base_action + alpha * pred_residual
            assert_finite_tensor("eval_reconstructed_action", reconstructed_action)
            residual_diff = pred_residual - residual_target
            reconstructed_diff = reconstructed_action - teacher_action
            base_diff = base_action - teacher_action
            residual_mse = residual_diff.square().mean(dim=1)
            residual_mae = residual_diff.abs().mean(dim=1)
            reconstructed_mse = reconstructed_diff.square().mean(dim=1)
            reconstructed_mae = reconstructed_diff.abs().mean(dim=1)
            base_mse = base_diff.square().mean(dim=1)
            base_mae = base_diff.abs().mean(dim=1)
            for name, tensor in (
                ("eval_residual_mse", residual_mse),
                ("eval_residual_mae", residual_mae),
                ("eval_reconstructed_mse", reconstructed_mse),
                ("eval_reconstructed_mae", reconstructed_mae),
                ("eval_base_mse", base_mse),
                ("eval_base_mae", base_mae),
            ):
                assert_finite_tensor(name, tensor)
            batch_size = int(obs_history.shape[0])
            residual_mse_sum += float(residual_mse.sum().detach().cpu().item())
            residual_mae_sum += float(residual_mae.sum().detach().cpu().item())
            reconstructed_mse_sum += float(reconstructed_mse.sum().detach().cpu().item())
            reconstructed_mae_sum += float(reconstructed_mae.sum().detach().cpu().item())
            base_mse_sum += float(base_mse.sum().detach().cpu().item())
            base_mae_sum += float(base_mae.sum().detach().cpu().item())
            sample_count += batch_size
    if sample_count <= 0:
        raise T16A5ResidualError("evaluation loader produced zero samples.")
    return {
        "residual_mse": residual_mse_sum / sample_count,
        "residual_mae": residual_mae_sum / sample_count,
        "reconstructed_action_mse": reconstructed_mse_sum / sample_count,
        "reconstructed_action_mae": reconstructed_mae_sum / sample_count,
        "base_action_mse": base_mse_sum / sample_count,
        "base_action_mae": base_mae_sum / sample_count,
    }


def finite_metric_value(path: str, value: Any) -> None:
    if value is None or isinstance(value, (bool, str, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise T16A5ResidualError(f"metric {path} is not finite: {value}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            finite_metric_value(f"{path}[{index}]", item)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            finite_metric_value(f"{path}.{key}", item)
        return
    raise T16A5ResidualError(f"metric {path} has unsupported type {type(value).__name__}.")


def finite_metric_dict(metrics: dict[str, Any]) -> None:
    finite_metric_value("metrics", metrics)


def write_json(path: Path, values: dict[str, Any]) -> None:
    finite_metric_dict(values)
    path.write_text(json.dumps(values, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_training_log_header(path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "epoch",
                "train_loss_residual_mse",
                "val_residual_mse",
                "val_residual_mae",
                "val_reconstructed_action_mse",
                "val_reconstructed_action_mae",
            ],
        )
        writer.writeheader()


def append_training_log(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "epoch",
                "train_loss_residual_mse",
                "val_residual_mse",
                "val_residual_mae",
                "val_reconstructed_action_mse",
                "val_reconstructed_action_mae",
            ],
        )
        writer.writerow(row)


def save_checkpoint(path: Path, model: Any, args: argparse.Namespace, metrics: dict[str, Any]) -> None:
    torch_module, _, _, _, _ = require_torch()
    torch_module.save(
        {
            "model_state_dict": model.state_dict(),
            "metrics": metrics,
            "model_class": "A5HistoryResidualMLP",
            "history_len": args.history_len,
            "input_dim": EXPECTED_STUDENT_OBS_DIM,
            "action_dim": EXPECTED_ACTION_DIM,
            "target_dim": EXPECTED_ACTION_DIM,
            "residual_dim": EXPECTED_ACTION_DIM,
            "residual_input_mode": RESIDUAL_INPUT_MODE,
            "base_checkpoint_path": repo_relative(args.base_checkpoint),
            "guardrails": {
                "input": "student_obs_history",
                "target": "teacher_action_minus_a2_history_base_action",
                "teacher_obs_used_as_input": False,
                "selected_fault_joint_index_used_as_input": False,
                "selected_fault_joint_one_hot_used_as_input": False,
                "q_lock_vector_used_as_input": False,
                "p2_fault_active_used_as_input": False,
                "health_token_used": False,
                "a0_action_used": False,
                "a7_training": False,
                "base_action_used_as_residual_input": False,
                "base_action_used_for_target_and_reconstruction": True,
            },
        },
        path,
    )


def write_readme(path: Path, metrics: dict[str, Any]) -> None:
    lines = [
        "# T16 Multi-Joint A5 History Residual Distillation",
        "",
        "Offline supervised residual distillation for the RLM1 stripped / conference multi-joint P2 pipeline.",
        "",
        "## Definition",
        "",
        "- A5 = frozen T15 A2-history H16 base + learned residual.",
        "- residual target: `teacher_action - base_action`.",
        f"- residual input: `{metrics['input_description']}`.",
        f"- reconstruction used for offline metrics: `base_action + {metrics['alpha_for_offline_metrics']} * residual_action`.",
        "- This is not A7; A7 would use an A0 healthy PPO base.",
        "",
        "## Frozen Base",
        "",
        f"- base checkpoint: `{metrics['base_checkpoint_path']}`",
        f"- base action validation MSE: `{metrics['base_action_val_mse']}`",
        f"- base action validation MAE: `{metrics['base_action_val_mae']}`",
        "",
        "## Datasets",
        "",
    ]
    for source in metrics["dataset_sample_counts"]:
        lines.append(f"- `{source['dataset_path']}`: `{source['sample_count']}` samples")
    lines.extend(
        [
            "",
            "## Metrics",
            "",
            f"- residual train MSE: `{metrics['train_residual_mse']}`",
            f"- residual train MAE: `{metrics['train_residual_mae']}`",
            f"- residual validation MSE: `{metrics['val_residual_mse']}`",
            f"- residual validation MAE: `{metrics['val_residual_mae']}`",
            f"- reconstructed validation MSE: `{metrics['val_reconstructed_action_mse']}`",
            f"- reconstructed validation MAE: `{metrics['val_reconstructed_action_mae']}`",
            f"- validation MSE improvement over base: `{metrics['val_reconstruction_mse_improvement_over_base']}`",
            f"- validation MAE improvement over base: `{metrics['val_reconstruction_mae_improvement_over_base']}`",
            "",
            "## Later Closed-Loop Evaluation",
            "",
            "- Offline metrics use alpha=1.0 by default.",
            "- Isaac-side deployment evaluation may later sweep alpha values such as 0.25, 0.5, and 1.0.",
            "- No closed-loop evaluation is run here.",
            "",
            "## Guardrails",
            "",
            "- Residual input uses only deployment-facing `student_obs` history.",
            "- `teacher_obs` is not used as residual input.",
            "- selected fault joint index/one-hot, q-lock vector, and P2-active flag are not used as residual input.",
            "- health token is OFF.",
            "- A0 action is not used.",
            "- This is A5, not A7.",
            "- This is offline supervised training only; no Isaac, RL, checkpoint pointer update, or dataset mutation.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def dry_run(args: argparse.Namespace) -> int:
    validate_args(args)
    arrays = load_datasets(args.dataset_paths)
    history_plan = prepare_history_plan(args, arrays)
    print("[T16-A5 DRY RUN]", flush=True)
    print("  no_training: true", flush=True)
    print("  no_checkpoint_write: true", flush=True)
    print(f"  base_checkpoint: {repo_relative(args.base_checkpoint)}", flush=True)
    print(f"  output_dir: {repo_relative(args.output_dir)}", flush=True)
    print(f"  dataset_paths: {[repo_relative(path) for path in args.dataset_paths]}", flush=True)
    print(f"  dataset_sample_counts: {arrays.source_counts}", flush=True)
    print(f"  history_input_shape_plan: [batch, {args.history_len}, {EXPECTED_STUDENT_OBS_DIM}]", flush=True)
    print(f"  valid_history_windows: {history_plan.history_metadata['valid_windows']}", flush=True)
    print(f"  train_windows: {int(history_plan.train_indices.shape[0])}", flush=True)
    print(f"  val_windows: {int(history_plan.val_indices.shape[0])}", flush=True)
    print(f"  teacher_action_target_shape_plan: [batch, {EXPECTED_ACTION_DIM}]", flush=True)
    print(f"  base_action_shape_plan: [batch, {EXPECTED_ACTION_DIM}]", flush=True)
    print(f"  residual_target_shape_plan: [batch, {EXPECTED_ACTION_DIM}]", flush=True)
    print(f"  reconstructed_action_shape_plan: [batch, {EXPECTED_ACTION_DIM}]", flush=True)
    print("  residual_target_definition: teacher_action - base_action", flush=True)
    print("  reconstructed_action_definition: base_action + alpha * residual_action", flush=True)
    print(f"  alpha_for_offline_metrics: {args.alpha}", flush=True)
    print(f"  residual_input_mode: {RESIDUAL_INPUT_MODE}", flush=True)
    print("  method: A5", flush=True)
    print("  is_a7: false", flush=True)
    print(f"  split_plan: {history_plan.split_metadata}", flush=True)
    print(f"  forbidden_fields_present_but_not_used: {arrays.forbidden_fields_present}", flush=True)
    print("  teacher_obs_used_as_input: false", flush=True)
    print("  selected_fault_joint_index_used_as_input: false", flush=True)
    print("  selected_fault_joint_one_hot_used_as_input: false", flush=True)
    print("  q_lock_vector_used_as_input: false", flush=True)
    print("  p2_fault_active_used_as_input: false", flush=True)
    print("  health_token_used: false", flush=True)
    print("  a0_action_used: false", flush=True)
    return 0


def train(args: argparse.Namespace) -> int:
    validate_args(args)
    set_seed(args.seed)
    device = select_device(args.device)
    torch_module, nn_module, data_loader_class, _, _ = require_torch()
    arrays = load_datasets(args.dataset_paths)
    history_plan = prepare_history_plan(args, arrays)
    base_checkpoint_path = resolve_repo_path(args.base_checkpoint)
    base_model, base_checkpoint_metadata = load_frozen_a2_history_model(
        base_checkpoint_path,
        history_len=args.history_len,
        device=device,
    )
    prepared = prepare_residual_data(args, arrays, history_plan, base_model, device=device)

    output_dir = resolve_repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    training_log_path = output_dir / "training_log.csv"
    write_training_log_header(training_log_path)

    train_loader = data_loader_class(
        prepared.train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        generator=torch_module.Generator().manual_seed(args.seed),
    )
    eval_train_loader = data_loader_class(prepared.train_dataset, batch_size=args.batch_size, shuffle=False)
    val_loader = data_loader_class(prepared.val_dataset, batch_size=args.batch_size, shuffle=False)

    residual_model = build_residual_model(
        args.history_len,
        EXPECTED_STUDENT_OBS_DIM,
        EXPECTED_ACTION_DIM,
    ).to(device)
    optimizer = torch_module.optim.AdamW(residual_model.parameters(), lr=args.lr, weight_decay=1.0e-5)
    loss_fn = nn_module.MSELoss()
    epoch_metrics: list[dict[str, Any]] = []

    for epoch in range(1, args.epochs + 1):
        residual_model.train()
        loss_sum = 0.0
        sample_count = 0
        for batch_obs, _batch_base, _batch_teacher, batch_residual in train_loader:
            batch_obs = batch_obs.to(device)
            batch_residual = batch_residual.to(device)
            assert_finite_tensor("batch_obs_history", batch_obs)
            assert_finite_tensor("batch_residual_target", batch_residual)
            pred_residual = residual_model(batch_obs)
            assert_finite_tensor("train_pred_residual", pred_residual)
            loss = loss_fn(pred_residual, batch_residual)
            assert_finite_tensor("train_loss", loss)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            batch_size = int(batch_obs.shape[0])
            loss_sum += float(loss.detach().cpu().item()) * batch_size
            sample_count += batch_size
        if sample_count <= 0:
            raise T16A5ResidualError("training loader produced zero samples.")
        train_loss_residual_mse = loss_sum / sample_count
        val_metrics = evaluate_residual_model(residual_model, val_loader, device=device, alpha=args.alpha)
        row = {
            "epoch": epoch,
            "train_loss_residual_mse": train_loss_residual_mse,
            "val_residual_mse": val_metrics["residual_mse"],
            "val_residual_mae": val_metrics["residual_mae"],
            "val_reconstructed_action_mse": val_metrics["reconstructed_action_mse"],
            "val_reconstructed_action_mae": val_metrics["reconstructed_action_mae"],
        }
        append_training_log(training_log_path, row)
        epoch_metrics.append(row)
        print(
            f"[T16-A5] epoch={epoch}/{args.epochs} "
            f"train_loss_residual_mse={train_loss_residual_mse:.8f} "
            f"val_residual_mse={val_metrics['residual_mse']:.8f} "
            f"val_recon_mse={val_metrics['reconstructed_action_mse']:.8f}",
            flush=True,
        )
        if args.save_every > 0 and epoch % args.save_every == 0:
            interim_metrics = {
                "stage": "t16_multijoint_a5_history_residual_distillation",
                "epoch": epoch,
                "train_loss_residual_mse": train_loss_residual_mse,
                **{f"val_{key}": value for key, value in val_metrics.items()},
            }
            save_checkpoint(output_dir / f"checkpoint_epoch_{epoch:04d}.pt", residual_model, args, interim_metrics)

    train_metrics = evaluate_residual_model(residual_model, eval_train_loader, device=device, alpha=args.alpha)
    val_metrics = evaluate_residual_model(residual_model, val_loader, device=device, alpha=args.alpha)
    metrics: dict[str, Any] = {
        "stage": "t16_multijoint_a5_history_residual_distillation",
        "rlm_phase": "RLM1 stripped / conference",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "mode": "history_residual",
        "method": "A5",
        "is_a7": False,
        "device": str(device),
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "val_fraction": args.val_fraction,
        "save_every": args.save_every,
        "dataset_paths": [repo_relative(path) for path in args.dataset_paths],
        "dataset_sample_counts": arrays.source_counts,
        "source_shapes": arrays.source_shapes,
        "num_samples": prepared.num_samples,
        "train_samples": prepared.train_samples,
        "val_samples": prepared.val_samples,
        "split_metadata": history_plan.split_metadata,
        **history_plan.history_metadata,
        **base_checkpoint_metadata,
        **prepared.base_metadata,
        "input_description": f"student_obs history [batch, {args.history_len}, 61]",
        "base_action_description": "frozen T15 A2-history action [batch, 8]",
        "target_description": "teacher_action [batch, 8]",
        "residual_target_description": "teacher_action - base_action [batch, 8]",
        "reconstructed_action_description": "base_action + alpha * residual_action [batch, 8]",
        "history_len": args.history_len,
        "input_dim": EXPECTED_STUDENT_OBS_DIM,
        "action_dim": EXPECTED_ACTION_DIM,
        "target_dim": EXPECTED_ACTION_DIM,
        "residual_dim": EXPECTED_ACTION_DIM,
        "residual_input_mode": RESIDUAL_INPUT_MODE,
        "alpha_for_offline_metrics": args.alpha,
        "train_residual_mse": train_metrics["residual_mse"],
        "train_residual_mae": train_metrics["residual_mae"],
        "val_residual_mse": val_metrics["residual_mse"],
        "val_residual_mae": val_metrics["residual_mae"],
        "train_reconstructed_action_mse": train_metrics["reconstructed_action_mse"],
        "train_reconstructed_action_mae": train_metrics["reconstructed_action_mae"],
        "val_reconstructed_action_mse": val_metrics["reconstructed_action_mse"],
        "val_reconstructed_action_mae": val_metrics["reconstructed_action_mae"],
        "base_action_train_mse": train_metrics["base_action_mse"],
        "base_action_train_mae": train_metrics["base_action_mae"],
        "base_action_val_mse": val_metrics["base_action_mse"],
        "base_action_val_mae": val_metrics["base_action_mae"],
        "val_reconstruction_mse_improvement_over_base": (
            val_metrics["base_action_mse"] - val_metrics["reconstructed_action_mse"]
        ),
        "val_reconstruction_mae_improvement_over_base": (
            val_metrics["base_action_mae"] - val_metrics["reconstructed_action_mae"]
        ),
        "input_key": "student_obs_history",
        "target_key": "teacher_action_minus_a2_history_base_action",
        "teacher_obs_used_as_input": False,
        "selected_fault_joint_index_used_as_input": False,
        "selected_fault_joint_one_hot_used_as_input": False,
        "q_lock_vector_used_as_input": False,
        "p2_fault_active_used_as_input": False,
        "health_token_used": False,
        "a0_action_used": False,
        "a7_training": False,
        "base_action_used_as_residual_input": False,
        "base_action_used_for_target_and_reconstruction": True,
        "forbidden_input_fields": FORBIDDEN_INPUT_FIELDS,
        "forbidden_fields_present_but_not_used": arrays.forbidden_fields_present,
        "residual_head_used": True,
        "no_nan_inf": True,
        "not_paper_grade_final": True,
        "closed_loop_eval_run": False,
        "later_alpha_sweep_note": "Closed-loop A5 evaluation may later sweep alpha values such as 0.25, 0.5, and 1.0.",
        "epoch_metrics": epoch_metrics,
    }
    finite_metric_dict(metrics)
    checkpoint_path = output_dir / "a5_history_residual_multijoint.pt"
    save_checkpoint(checkpoint_path, residual_model, args, metrics)
    metrics["checkpoint_path"] = repo_relative(checkpoint_path)
    metrics["training_log_csv"] = repo_relative(training_log_path)
    metrics_path = output_dir / "metrics_summary.json"
    readme_path = output_dir / "README.md"
    write_json(metrics_path, metrics)
    write_readme(readme_path, metrics)
    for required_path in (checkpoint_path, training_log_path, metrics_path, readme_path, output_dir / "command.txt"):
        if not required_path.is_file():
            raise T16A5ResidualError(f"required output missing: {repo_relative(required_path)}")
    print("[T16-A5] final metrics", flush=True)
    print(f"  train_residual_mse: {metrics['train_residual_mse']}", flush=True)
    print(f"  train_residual_mae: {metrics['train_residual_mae']}", flush=True)
    print(f"  val_residual_mse: {metrics['val_residual_mse']}", flush=True)
    print(f"  val_residual_mae: {metrics['val_residual_mae']}", flush=True)
    print(f"  base_action_val_mse: {metrics['base_action_val_mse']}", flush=True)
    print(f"  val_reconstructed_action_mse: {metrics['val_reconstructed_action_mse']}", flush=True)
    print(f"  checkpoint: {repo_relative(checkpoint_path)}", flush=True)
    print(f"  metrics: {repo_relative(metrics_path)}", flush=True)
    print(f"  readme: {repo_relative(readme_path)}", flush=True)
    return 0


def main() -> int:
    try:
        args = build_parser().parse_args()
        if args.dry_run:
            return dry_run(args)
        if args.execute_train:
            return train(args)
        raise T16A5ResidualError("select --dry_run or --execute_train.")
    except T16A5ResidualError as exc:
        print(f"[T16-A5 ERROR] {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
