#!/usr/bin/env python3
"""Offline T15 A2 student distillation for multi-joint P2 teacher datasets.

This trainer is offline-only. It consumes T14 rollout datasets and fits either:

* an A2 single-step deployment-facing student from 61-D ``student_obs`` to the
  8-D teacher action, or
* an A2-history student from an H x 61 ``student_obs`` window to the current
  8-D teacher action.

The student input is never ``teacher_obs`` and never contains selected fault
joint indices, fault one-hot vectors, q-lock vectors, P2-active flags, or a
health token. Latent ``z_t`` analysis is intentionally deferred to T18.
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
DEFAULT_SINGLE_OUTPUT_DIR = "papers/conference/results/t15_a2_single_step_multijoint_seed0"
DEFAULT_HISTORY_OUTPUT_DIR = "papers/conference/results/t15_a2_history_h16_multijoint_seed0"
EXPECTED_STUDENT_OBS_DIM = 61
EXPECTED_TEACHER_OBS_DIM = 77
EXPECTED_ACTION_DIM = 8
DEFAULT_HISTORY_LEN = 16
DEFAULT_SEED = 0
DEFAULT_EPOCHS = 50
DEFAULT_BATCH_SIZE = 4096
DEFAULT_LR = 1.0e-3
DEFAULT_VAL_FRACTION = 0.1

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


class T15DistillError(ValueError):
    """Raised for invalid T15 A2 distillation state."""


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
class PreparedData:
    train_dataset: Any
    val_dataset: Any
    train_samples: int
    val_samples: int
    num_samples: int
    split_metadata: dict[str, Any]
    prep_metadata: dict[str, Any]


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
            raise T15DistillError(
                "PyTorch is required for T15 A2 distillation. Activate the isaaclab environment."
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


def default_output_dir(mode: str) -> str:
    return DEFAULT_HISTORY_OUTPUT_DIR if mode == "history" else DEFAULT_SINGLE_OUTPUT_DIR


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline T15 multi-joint A2 student distillation trainer.")
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--dry_run", action="store_true", help="Inspect data and split plan without training.")
    mode_group.add_argument("--execute_train", action="store_true", help="Run offline supervised training.")
    parser.add_argument("--dataset_paths", nargs="+", default=DEFAULT_DATASET_PATHS)
    parser.add_argument("--mode", required=True, choices=("single_step", "history"))
    parser.add_argument("--history_len", type=int, default=DEFAULT_HISTORY_LEN)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", default="cuda", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--val_fraction", type=float, default=DEFAULT_VAL_FRACTION)
    parser.add_argument("--save_every", type=int, default=0, help="Save epoch checkpoints every N epochs; 0 disables.")
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
            raise T15DistillError(f"required dataset timing metadata is missing: {repo_relative(metadata_path)}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if args.require_dataset_control_frequency_hz is not None:
            observed = float(metadata.get("control_frequency_hz"))
            if abs(observed - float(args.require_dataset_control_frequency_hz)) > 1.0e-4:
                raise T15DistillError(
                    f"dataset {repo_relative(dataset_path)} control_frequency_hz expected "
                    f"{args.require_dataset_control_frequency_hz}, got {observed}."
                )
        if args.require_dataset_physics_frequency_hz is not None:
            observed = float(metadata.get("physics_frequency_hz"))
            if abs(observed - float(args.require_dataset_physics_frequency_hz)) > 1.0e-4:
                raise T15DistillError(
                    f"dataset {repo_relative(dataset_path)} physics_frequency_hz expected "
                    f"{args.require_dataset_physics_frequency_hz}, got {observed}."
                )


def validate_args(args: argparse.Namespace) -> None:
    if not args.dataset_paths:
        raise T15DistillError("--dataset_paths must contain at least one dataset.")
    for dataset_path in args.dataset_paths:
        resolved = resolve_repo_path(dataset_path)
        if not resolved.is_file():
            raise T15DistillError(f"dataset_path does not exist: {repo_relative(resolved)}")
    if args.history_len < 1:
        raise T15DistillError("--history_len must be >= 1.")
    if args.require_history_len is not None and int(args.history_len) != int(args.require_history_len):
        raise T15DistillError(f"--history_len must equal --require_history_len={args.require_history_len}.")
    if args.epochs <= 0:
        raise T15DistillError("--epochs must be > 0.")
    if args.batch_size <= 0:
        raise T15DistillError("--batch_size must be > 0.")
    if args.lr <= 0.0:
        raise T15DistillError("--lr must be > 0.")
    if not 0.0 < args.val_fraction < 1.0:
        raise T15DistillError("--val_fraction must be in (0, 1).")
    if args.save_every < 0:
        raise T15DistillError("--save_every must be >= 0.")
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
        raise T15DistillError("--device cuda requested but CUDA is not available.")
    return torch_module.device(device_arg)


def check_finite_array(name: str, array: np.ndarray) -> None:
    if array.dtype.kind in {"f", "i", "u"} and not np.isfinite(array).all():
        raise T15DistillError(f"{name} contains NaN or Inf.")


def flatten_feature_array(array: np.ndarray, *, key: str, expected_dim: int) -> np.ndarray:
    if array.ndim == 3:
        if array.shape[-1] != expected_dim:
            raise T15DistillError(f"{key} expected final dim {expected_dim}, got shape {array.shape}.")
        return array.reshape(-1, expected_dim).astype(np.float32, copy=False)
    if array.ndim == 2:
        if array.shape[-1] != expected_dim:
            raise T15DistillError(f"{key} expected final dim {expected_dim}, got shape {array.shape}.")
        return array.astype(np.float32, copy=False)
    raise T15DistillError(f"{key} expected 2-D or 3-D array, got shape {array.shape}.")


def flatten_scalar_array(array: np.ndarray, *, key: str) -> np.ndarray:
    if array.ndim in {1, 2}:
        return array.reshape(-1)
    raise T15DistillError(f"{key} expected 1-D or 2-D array, got shape {array.shape}.")


def generated_scalar(shape_source: np.ndarray, *, key: str) -> np.ndarray:
    if shape_source.ndim == 3:
        time_steps, num_envs = shape_source.shape[:2]
        if key == "env_id":
            return np.tile(np.arange(num_envs, dtype=np.int64), time_steps)
        if key == "timestep":
            return np.repeat(np.arange(time_steps, dtype=np.int64), num_envs)
    raise T15DistillError(f"{key} is required for flattened T14 datasets.")


def load_one_dataset(dataset_path: Path, *, source_index: int) -> tuple[dict[str, np.ndarray], dict[str, Any], list[str]]:
    with np.load(dataset_path, allow_pickle=False) as data:
        files = set(data.files)
        required = {"student_obs", "teacher_action", "done", "episode_id"}
        missing = sorted(required - files)
        if missing:
            raise T15DistillError(f"{repo_relative(dataset_path)} missing required arrays: {missing}")

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
            raise T15DistillError(
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
                raise T15DistillError(
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
        teacher_obs_final_dim = int(data["teacher_obs"].shape[-1]) if "teacher_obs" in files and data["teacher_obs"].ndim >= 2 else None
        if teacher_obs_final_dim is not None and teacher_obs_final_dim != EXPECTED_TEACHER_OBS_DIM:
            raise T15DistillError(
                f"teacher_obs final dim expected {EXPECTED_TEACHER_OBS_DIM} in {repo_relative(dataset_path)}, "
                f"got {teacher_obs_final_dim}."
            )

    check_finite_array("student_obs", student_obs)
    check_finite_array("teacher_action", teacher_action)
    check_finite_array("episode_id", episode_id) if np.issubdtype(episode_id.dtype, np.number) else None
    check_finite_array("env_id", env_id) if np.issubdtype(env_id.dtype, np.number) else None
    check_finite_array("timestep", timestep) if np.issubdtype(timestep.dtype, np.number) else None

    source_index_array = np.full(sample_count, source_index, dtype=np.int64)
    arrays = {
        "student_obs": student_obs,
        "teacher_action": teacher_action,
        "done": done,
        "episode_id": episode_id,
        "env_id": env_id,
        "timestep": timestep,
        "source_dataset_index": source_index_array,
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

    concat = {
        key: np.concatenate([block[key] for block in blocks], axis=0)
        for key in blocks[0].keys()
    }
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
        raise T15DistillError("at least two samples are required for train/val split.")
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
                "temporal_leakage_note": "All samples/windows from a source/env/episode group are assigned to one split.",
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
            prior_done = arrays.done[window[:-1]]
            if bool(np.any(prior_done)):
                rejected_done_crossing_windows += 1
                continue
            window_indices.append(window)
            target_indices.append(sorted_indices[local_target])
            window_labels.append(label)

    if not window_indices:
        raise T15DistillError("history mode produced zero valid windows.")
    windows = np.asarray(window_indices, dtype=np.int64)
    targets = np.asarray(target_indices, dtype=np.int64)
    labels_array = np.asarray(window_labels, dtype="U96")
    metadata = {
        "history_len": history_len,
        "total_groups": len(groups),
        "rejected_short_groups": rejected_short_groups,
        "total_candidate_windows": int(candidate_windows),
        "valid_windows": int(windows.shape[0]),
        "rejected_done_crossing_windows": int(rejected_done_crossing_windows),
        "rejected_cross_episode_windows": 0,
        "history_input_shape": [int(windows.shape[0]), history_len, EXPECTED_STUDENT_OBS_DIM],
        "target_shape": [int(targets.shape[0]), EXPECTED_ACTION_DIM],
        "window_guardrail": "Windows are built within source/env/episode groups and reject prior done crossings.",
    }
    return windows, targets, labels_array, metadata


def build_single_step_model(input_dim: int, action_dim: int) -> Any:
    _, nn_module, _, _, _ = require_torch()

    class SingleStepStudentMLP(nn_module.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn_module.Sequential(
                nn_module.Linear(input_dim, 512),
                nn_module.ELU(),
                nn_module.Linear(512, 256),
                nn_module.ELU(),
                nn_module.Linear(256, 128),
                nn_module.ELU(),
                nn_module.Linear(128, action_dim),
            )

        def forward(self, obs: Any) -> Any:
            return self.net(obs)

    return SingleStepStudentMLP()


def build_history_model(history_len: int, input_dim: int, action_dim: int) -> Any:
    _, nn_module, _, _, _ = require_torch()

    class HistoryStudentMLP(nn_module.Module):
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
            z_t = self.encode(obs_history)
            return self.action_head(z_t)

    return HistoryStudentMLP()


def make_history_dataset(obs_tensor: Any, action_tensor: Any, window_indices: Any, target_indices: Any) -> Any:
    _, _, _, _, dataset_base = require_torch()

    class HistoryWindowDataset(dataset_base):
        def __init__(self) -> None:
            self.obs = obs_tensor
            self.actions = action_tensor
            self.window_indices = window_indices
            self.target_indices = target_indices

        def __len__(self) -> int:
            return int(self.target_indices.shape[0])

        def __getitem__(self, index: int) -> tuple[Any, Any]:
            return self.obs[self.window_indices[index]], self.actions[self.target_indices[index]]

    return HistoryWindowDataset()


def prepare_data(args: argparse.Namespace, arrays: LoadedArrays, *, for_training: bool) -> PreparedData | dict[str, Any]:
    labels = group_labels(arrays)
    if args.mode == "single_step":
        train_indices, val_indices, split_metadata = split_indices_by_group(
            labels,
            seed=args.seed,
            val_fraction=args.val_fraction,
        )
        prep_metadata = {
            "mode": "single_step",
            "input_description": "student_obs [batch, 61]",
            "target_description": "teacher_action [batch, 8]",
            "input_dim": EXPECTED_STUDENT_OBS_DIM,
            "target_dim": EXPECTED_ACTION_DIM,
            "num_samples": int(arrays.student_obs.shape[0]),
        }
        if not for_training:
            return {
                "train_samples": int(train_indices.shape[0]),
                "val_samples": int(val_indices.shape[0]),
                "num_samples": int(arrays.student_obs.shape[0]),
                "split_metadata": split_metadata,
                "prep_metadata": prep_metadata,
            }
        torch_module, _, _, tensor_dataset_class, _ = require_torch()
        obs_tensor = torch_module.from_numpy(arrays.student_obs.astype(np.float32, copy=False))
        action_tensor = torch_module.from_numpy(arrays.teacher_action.astype(np.float32, copy=False))
        train_dataset = tensor_dataset_class(obs_tensor[torch_module.from_numpy(train_indices)], action_tensor[torch_module.from_numpy(train_indices)])
        val_dataset = tensor_dataset_class(obs_tensor[torch_module.from_numpy(val_indices)], action_tensor[torch_module.from_numpy(val_indices)])
        return PreparedData(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            train_samples=int(train_indices.shape[0]),
            val_samples=int(val_indices.shape[0]),
            num_samples=int(arrays.student_obs.shape[0]),
            split_metadata=split_metadata,
            prep_metadata=prep_metadata,
        )

    windows, targets, window_labels, history_metadata = build_history_window_indices(arrays, history_len=args.history_len)
    train_window_indices, val_window_indices, split_metadata = split_indices_by_group(
        window_labels,
        seed=args.seed,
        val_fraction=args.val_fraction,
    )
    prep_metadata = {
        "mode": "history",
        "input_description": f"student_obs history [batch, {args.history_len}, 61]",
        "target_description": "teacher_action [batch, 8] at current timestep",
        "input_dim": EXPECTED_STUDENT_OBS_DIM,
        "target_dim": EXPECTED_ACTION_DIM,
        "history_len": args.history_len,
        "latent_note": "H16 is the input history; encoder output z_t is a later analysis object for T18.",
        **history_metadata,
    }
    if not for_training:
        return {
            "train_samples": int(train_window_indices.shape[0]),
            "val_samples": int(val_window_indices.shape[0]),
            "num_samples": int(windows.shape[0]),
            "split_metadata": split_metadata,
            "prep_metadata": prep_metadata,
        }

    torch_module, _, _, _, _ = require_torch()
    obs_tensor = torch_module.from_numpy(arrays.student_obs.astype(np.float32, copy=False))
    action_tensor = torch_module.from_numpy(arrays.teacher_action.astype(np.float32, copy=False))
    window_tensor = torch_module.from_numpy(windows)
    target_tensor = torch_module.from_numpy(targets)
    train_dataset = make_history_dataset(
        obs_tensor,
        action_tensor,
        window_tensor[torch_module.from_numpy(train_window_indices)],
        target_tensor[torch_module.from_numpy(train_window_indices)],
    )
    val_dataset = make_history_dataset(
        obs_tensor,
        action_tensor,
        window_tensor[torch_module.from_numpy(val_window_indices)],
        target_tensor[torch_module.from_numpy(val_window_indices)],
    )
    return PreparedData(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        train_samples=int(train_window_indices.shape[0]),
        val_samples=int(val_window_indices.shape[0]),
        num_samples=int(windows.shape[0]),
        split_metadata=split_metadata,
        prep_metadata=prep_metadata,
    )


def assert_finite_tensor(name: str, tensor: Any) -> None:
    torch_module, _, _, _, _ = require_torch()
    if not torch_module.isfinite(tensor).all():
        raise T15DistillError(f"{name} contains NaN or Inf.")


def evaluate(model: Any, loader: Any, *, device: Any) -> tuple[float, float]:
    torch_module, _, _, _, _ = require_torch()
    model.eval()
    total_mse = 0.0
    total_mae = 0.0
    total_samples = 0
    with torch_module.inference_mode():
        for obs, target in loader:
            obs = obs.to(device)
            target = target.to(device)
            pred = model(obs)
            assert_finite_tensor("eval_predictions", pred)
            diff = pred - target
            mse = diff.square().mean(dim=1)
            mae = diff.abs().mean(dim=1)
            batch_size = int(obs.shape[0])
            total_mse += float(mse.sum().detach().cpu().item())
            total_mae += float(mae.sum().detach().cpu().item())
            total_samples += batch_size
    if total_samples <= 0:
        raise T15DistillError("evaluation loader produced zero samples.")
    return total_mse / total_samples, total_mae / total_samples


def finite_metric_value(path: str, value: Any) -> None:
    if value is None or isinstance(value, (bool, str, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise T15DistillError(f"metric {path} is not finite: {value}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            finite_metric_value(f"{path}[{index}]", item)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            finite_metric_value(f"{path}.{key}", item)
        return
    raise T15DistillError(f"metric {path} has unsupported type {type(value).__name__}.")


def finite_metric_dict(metrics: dict[str, Any]) -> None:
    finite_metric_value("metrics", metrics)


def write_json(path: Path, values: dict[str, Any]) -> None:
    finite_metric_dict(values)
    path.write_text(json.dumps(values, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_training_log_header(path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["epoch", "train_loss_mse", "val_mse", "val_mae"],
        )
        writer.writeheader()


def append_training_log(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["epoch", "train_loss_mse", "val_mse", "val_mae"])
        writer.writerow(row)


def checkpoint_name(args: argparse.Namespace) -> str:
    if args.mode == "history":
        return f"a2_history_h{args.history_len}_multijoint.pt"
    return "a2_single_step_multijoint.pt"


def save_checkpoint(path: Path, model: Any, args: argparse.Namespace, metrics: dict[str, Any]) -> None:
    torch_module, _, _, _, _ = require_torch()
    torch_module.save(
        {
            "model_state_dict": model.state_dict(),
            "metrics": metrics,
            "mode": args.mode,
            "history_len": args.history_len if args.mode == "history" else None,
            "input_dim": EXPECTED_STUDENT_OBS_DIM,
            "target_dim": EXPECTED_ACTION_DIM,
            "guardrails": {
                "input": "student_obs" if args.mode == "single_step" else "student_obs_history",
                "target": "teacher_action",
                "teacher_obs_used_as_input": False,
                "selected_fault_joint_index_used_as_input": False,
                "selected_fault_joint_one_hot_used_as_input": False,
                "q_lock_vector_used_as_input": False,
                "p2_fault_active_used_as_input": False,
                "health_token_used": False,
                "latent_tsne_umap_done": False,
            },
        },
        path,
    )


def write_readme(path: Path, metrics: dict[str, Any]) -> None:
    lines = [
        "# T15 Multi-Joint A2 Student Distillation",
        "",
        "Offline supervised student distillation for the RLM1 stripped / conference multi-joint P2 pipeline.",
        "",
        "## Mode",
        "",
        f"- mode: `{metrics['mode']}`",
        f"- history_len: `{metrics.get('history_len')}`",
        f"- input: `{metrics['input_description']}`",
        f"- target: `{metrics['target_description']}`",
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
            f"- train_mse: `{metrics['train_mse']}`",
            f"- train_mae: `{metrics['train_mae']}`",
            f"- val_mse: `{metrics['val_mse']}`",
            f"- val_mae: `{metrics['val_mae']}`",
            f"- train samples: `{metrics['train_samples']}`",
            f"- validation samples: `{metrics['val_samples']}`",
            "",
            "## Guardrails",
            "",
            "- Student input uses only deployment-facing `student_obs`.",
            "- `teacher_obs` is not used as student input.",
            "- selected fault joint index/one-hot, q-lock vector, and P2-active flag are not used as student input.",
            "- health token is OFF.",
            "- teacher action is the supervised target.",
            "- H16 is an input history window, not the latent itself.",
            "- latent `z_t` analysis belongs to later T18; no t-SNE/UMAP is implemented here.",
            "- This is offline supervised training only; no Isaac, RL, checkpoint pointer update, or dataset mutation.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def dry_run(args: argparse.Namespace) -> int:
    validate_args(args)
    arrays = load_datasets(args.dataset_paths)
    plan = prepare_data(args, arrays, for_training=False)
    output_dir = resolve_repo_path(args.output_dir or default_output_dir(args.mode))
    print("[T15-A2 DRY RUN]", flush=True)
    print("  no_training: true", flush=True)
    print("  no_checkpoint_write: true", flush=True)
    print(f"  mode: {args.mode}", flush=True)
    print(f"  output_dir: {repo_relative(output_dir)}", flush=True)
    print(f"  dataset_paths: {[repo_relative(path) for path in args.dataset_paths]}", flush=True)
    print(f"  dataset_sample_counts: {arrays.source_counts}", flush=True)
    print(f"  input_description: {plan['prep_metadata']['input_description']}", flush=True)
    print(f"  target_description: {plan['prep_metadata']['target_description']}", flush=True)
    print(f"  train_samples: {plan['train_samples']}", flush=True)
    print(f"  val_samples: {plan['val_samples']}", flush=True)
    print(f"  split_plan: {plan['split_metadata']}", flush=True)
    print(f"  forbidden_fields_present_but_not_used: {arrays.forbidden_fields_present}", flush=True)
    print(f"  forbidden_fields_used_as_input: false", flush=True)
    print("  teacher_obs_used_as_input: false", flush=True)
    print("  selected_fault_joint_index_used_as_input: false", flush=True)
    print("  selected_fault_joint_one_hot_used_as_input: false", flush=True)
    print("  q_lock_vector_used_as_input: false", flush=True)
    print("  p2_fault_active_used_as_input: false", flush=True)
    print("  health_token_used: false", flush=True)
    print("  latent_tsne_umap_done: false", flush=True)
    return 0


def train(args: argparse.Namespace) -> int:
    validate_args(args)
    set_seed(args.seed)
    device = select_device(args.device)
    torch_module, nn_module, data_loader_class, _, _ = require_torch()
    arrays = load_datasets(args.dataset_paths)
    prepared = prepare_data(args, arrays, for_training=True)
    if not isinstance(prepared, PreparedData):
        raise T15DistillError("internal error: prepared training data missing.")

    output_dir = resolve_repo_path(args.output_dir or default_output_dir(args.mode))
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

    model = (
        build_history_model(args.history_len, EXPECTED_STUDENT_OBS_DIM, EXPECTED_ACTION_DIM)
        if args.mode == "history"
        else build_single_step_model(EXPECTED_STUDENT_OBS_DIM, EXPECTED_ACTION_DIM)
    ).to(device)
    optimizer = torch_module.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1.0e-5)
    loss_fn = nn_module.MSELoss()
    epoch_metrics: list[dict[str, Any]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        sample_count = 0
        for batch_obs, batch_target in train_loader:
            batch_obs = batch_obs.to(device)
            batch_target = batch_target.to(device)
            pred = model(batch_obs)
            assert_finite_tensor("train_predictions", pred)
            loss = loss_fn(pred, batch_target)
            assert_finite_tensor("train_loss", loss)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            batch_size = int(batch_obs.shape[0])
            loss_sum += float(loss.detach().cpu().item()) * batch_size
            sample_count += batch_size
        if sample_count <= 0:
            raise T15DistillError("training loader produced zero samples.")
        train_loss_mse = loss_sum / sample_count
        val_mse, val_mae = evaluate(model, val_loader, device=device)
        row = {
            "epoch": epoch,
            "train_loss_mse": train_loss_mse,
            "val_mse": val_mse,
            "val_mae": val_mae,
        }
        append_training_log(training_log_path, row)
        epoch_metrics.append(row)
        print(
            f"[T15-A2] epoch={epoch}/{args.epochs} "
            f"train_loss_mse={train_loss_mse:.8f} val_mse={val_mse:.8f} val_mae={val_mae:.8f}",
            flush=True,
        )
        if args.save_every > 0 and epoch % args.save_every == 0:
            interim_metrics = {
                "mode": args.mode,
                "epoch": epoch,
                "train_loss_mse": train_loss_mse,
                "val_mse": val_mse,
                "val_mae": val_mae,
            }
            save_checkpoint(output_dir / f"checkpoint_epoch_{epoch:04d}.pt", model, args, interim_metrics)

    train_mse, train_mae = evaluate(model, eval_train_loader, device=device)
    val_mse, val_mae = evaluate(model, val_loader, device=device)
    metrics: dict[str, Any] = {
        "stage": "t15_multijoint_a2_student_distillation",
        "rlm_phase": "RLM1 stripped / conference",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "mode": args.mode,
        "history_len": args.history_len if args.mode == "history" else None,
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
        "split_metadata": prepared.split_metadata,
        **prepared.prep_metadata,
        "action_dim": EXPECTED_ACTION_DIM,
        "target_dim": EXPECTED_ACTION_DIM,
        "train_mse": train_mse,
        "train_mae": train_mae,
        "val_mse": val_mse,
        "val_mae": val_mae,
        "input_key": "student_obs" if args.mode == "single_step" else "student_obs_history",
        "target_key": "teacher_action",
        "teacher_obs_used_as_input": False,
        "selected_fault_joint_index_used_as_input": False,
        "selected_fault_joint_one_hot_used_as_input": False,
        "q_lock_vector_used_as_input": False,
        "p2_fault_active_used_as_input": False,
        "health_token_used": False,
        "forbidden_input_fields": FORBIDDEN_INPUT_FIELDS,
        "forbidden_fields_present_but_not_used": arrays.forbidden_fields_present,
        "residual_head_used": False,
        "latent_tsne_umap_done": False,
        "latent_note": "H16 is not the latent itself; encoder output z_t analysis is deferred to T18.",
        "no_nan_inf": True,
        "not_paper_grade_final": True,
        "epoch_metrics": epoch_metrics,
    }
    finite_metric_dict(metrics)
    checkpoint_path = output_dir / checkpoint_name(args)
    save_checkpoint(checkpoint_path, model, args, metrics)
    metrics["checkpoint_path"] = repo_relative(checkpoint_path)
    metrics["training_log_csv"] = repo_relative(training_log_path)
    metrics_path = output_dir / "metrics_summary.json"
    readme_path = output_dir / "README.md"
    write_json(metrics_path, metrics)
    write_readme(readme_path, metrics)
    for required_path in (checkpoint_path, training_log_path, metrics_path, readme_path, output_dir / "command.txt"):
        if not required_path.is_file():
            raise T15DistillError(f"required output missing: {repo_relative(required_path)}")
    print("[T15-A2] final metrics", flush=True)
    print(f"  train_mse: {train_mse}", flush=True)
    print(f"  train_mae: {train_mae}", flush=True)
    print(f"  val_mse: {val_mse}", flush=True)
    print(f"  val_mae: {val_mae}", flush=True)
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
        raise T15DistillError("select --dry_run or --execute_train.")
    except T15DistillError as exc:
        print(f"[T15-A2 ERROR] {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
