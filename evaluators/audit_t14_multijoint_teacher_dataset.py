#!/usr/bin/env python3
"""Offline audit utility for T14 multi-joint teacher rollout datasets.

This script performs read-only NumPy audits of the separately collected T14
datasets. It does not import Isaac, modify datasets, modify checkpoints, or run
training/evaluation. The outputs are intended for downstream A2/A2-history/A5
and optional A7 planning.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_REALISTIC_DATASET = "papers/conference/datasets/t14_multijoint_teacher_v2b_realistic_seed0/dataset.npz"
DEFAULT_LATE_DATASET = "papers/conference/datasets/t14_multijoint_teacher_v2b_late_seed0/dataset.npz"
DEFAULT_OUTPUT_ROOT = "papers/conference/results/t14_multijoint_dataset_audit"

RLM_PHASE = "RLM1 stripped / conference"
SELECTED_TEACHER_LABEL = "A1-F multi-joint teacher v2b, valid but foot-limited"
SELECTED_TEACHER_CHECKPOINT = (
    "logs/rsl_rl/teacher_p2_multijoint_velocity_curriculum_v2b__rlm1_stripped__p2_multi_joint_random/"
    "2026-06-15_18-16-03_a1f_multijoint_velocity_p2_v2b_curriculum_curriculum_s2_realistic_transition_random_p2__seed0/"
    "model_9997.pt"
)

EXPECTED_DIMS = {
    "student_obs": 61,
    "teacher_obs": 77,
    "teacher_action": 8,
    "selected_fault_joint_one_hot": 8,
    "q_lock_vector": 8,
}

REQUIRED_KEYS = [
    "student_obs",
    "teacher_obs",
    "teacher_action",
    "selected_fault_joint_index",
    "selected_fault_joint_one_hot",
    "q_lock_vector",
    "p2_fault_active",
    "fault_onset_step",
    "vx_cmd",
    "velocity_x",
    "vx_error",
    "done",
    "episode_id",
    "timestep",
    "protocol_label",
    "velocity_mode_label",
]

OPTIONAL_KEYS = ["yaw_rate", "yaw_error", "a0_action", "a7_residual_target"]

PROTOCOL_ONSET_RANGES = {
    "realistic_random": (120, 700),
    "late_random": (250, 700),
}

DEFAULT_JOINT_NAMES = [
    "front_left_leg",
    "front_right_leg",
    "left_back_leg",
    "right_back_leg",
    "front_left_foot",
    "front_right_foot",
    "left_back_foot",
    "right_back_foot",
]

PER_DATASET_FIELDS = [
    "dataset_label",
    "dataset_path",
    "exists",
    "protocol",
    "velocity_mode",
    "sample_count",
    "inferred_num_envs",
    "inferred_num_timesteps",
    "student_obs_dim",
    "teacher_obs_dim",
    "teacher_action_dim",
    "selected_fault_joint_index_min",
    "selected_fault_joint_index_max",
    "represented_joint_count",
    "all_8_joints_represented",
    "selected_fault_joint_one_hot_dim",
    "q_lock_vector_dim",
    "p2_fault_active_has_inactive",
    "p2_fault_active_has_active",
    "fault_onset_step_min",
    "fault_onset_step_max",
    "fault_onset_range_expected",
    "fault_onset_range_ok",
    "vx_cmd_min",
    "vx_cmd_max",
    "vx_cmd_mean",
    "velocity_x_mean",
    "mean_vx_error",
    "median_vx_error",
    "std_vx_error",
    "p90_vx_error",
    "mean_abs_vx_error",
    "median_abs_vx_error",
    "p90_abs_vx_error",
    "mean_yaw_error",
    "mean_abs_yaw_error",
    "done_count",
    "done_rate",
    "survival_rate",
    "no_nan_inf",
    "missing_keys",
    "failed_checks",
]

PER_JOINT_FIELDS = [
    "dataset_label",
    "protocol",
    "joint_id",
    "joint_name",
    "sample_count",
    "active_sample_count",
    "mean_vx_cmd",
    "mean_velocity_x",
    "mean_vx_error",
    "median_vx_error",
    "p90_vx_error",
    "mean_abs_vx_error",
    "median_abs_vx_error",
    "p90_abs_vx_error",
    "done_rate",
    "survival_rate",
    "mean_yaw_error",
    "mean_abs_yaw_error",
]


class AuditError(ValueError):
    """Raised for invalid audit arguments or unrecoverable audit state."""


def repo_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def repo_relative(path: str | Path) -> str:
    path = Path(path)
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline audit for T14 multi-joint teacher rollout datasets.")
    parser.add_argument("--realistic_dataset", default=DEFAULT_REALISTIC_DATASET)
    parser.add_argument("--late_dataset", default=DEFAULT_LATE_DATASET)
    parser.add_argument("--output_root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--write_outputs", action="store_true", help="Write audit outputs to output_root.")
    return parser


def safe_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(result) or math.isinf(result):
        return None
    return result


def scalar_stat(values: Any, fn: str) -> float | None:
    import numpy as np

    if values is None:
        return None
    array = np.asarray(values)
    if array.size == 0:
        return None
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return None
    if fn == "mean":
        return float(np.mean(finite))
    if fn == "median":
        return float(np.median(finite))
    if fn == "std":
        return float(np.std(finite))
    if fn == "p90":
        return float(np.percentile(finite, 90))
    if fn == "min":
        return float(np.min(finite))
    if fn == "max":
        return float(np.max(finite))
    raise AuditError(f"unsupported stat function: {fn}")


def bool_str(value: bool) -> str:
    return "true" if value else "false"


def format_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return bool_str(value)
    if isinstance(value, float):
        return f"{value:.9g}"
    if isinstance(value, (list, tuple)):
        return ";".join(str(item) for item in value)
    return str(value)


def write_csv(path: Path, rows: list[dict[str, Any]], *, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_metadata(dataset_path: Path) -> dict[str, Any]:
    metadata_path = dataset_path.parent / "metadata.json"
    if not metadata_path.is_file():
        return {}
    try:
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def first_label(values: Any) -> str:
    import numpy as np

    array = np.asarray(values)
    if array.size == 0:
        return ""
    first = array.reshape(-1)[0]
    if isinstance(first, bytes):
        return first.decode("utf-8")
    return str(first)


def numeric_no_nan_inf(npz: Any) -> bool:
    import numpy as np

    for key in npz.files:
        array = np.asarray(npz[key])
        if array.dtype.kind in {"f", "i", "u"} and not np.isfinite(array).all():
            return False
    return True


def has_array(npz: Any, key: str) -> bool:
    return key in getattr(npz, "files", [])


def final_dim(npz: Any, key: str) -> int | None:
    if not has_array(npz, key):
        return None
    shape = npz[key].shape
    if len(shape) < 2:
        return None
    return int(shape[-1])


def infer_envs_and_timesteps(npz: Any) -> tuple[int | None, int | None]:
    import numpy as np

    num_envs = None
    num_steps = None
    if has_array(npz, "env_id"):
        env_ids = np.asarray(npz["env_id"]).reshape(-1)
        if env_ids.size:
            num_envs = int(np.unique(env_ids).size)
    if has_array(npz, "timestep"):
        timesteps = np.asarray(npz["timestep"]).reshape(-1)
        if timesteps.size:
            num_steps = int(np.unique(timesteps).size)
    return num_envs, num_steps


def represented_joint_ids(npz: Any) -> list[int]:
    import numpy as np

    if not has_array(npz, "selected_fault_joint_index"):
        return []
    values = np.asarray(npz["selected_fault_joint_index"]).reshape(-1)
    if values.size == 0:
        return []
    return [int(item) for item in sorted(np.unique(values.astype(int)).tolist())]


def joint_names_from_metadata(metadata: dict[str, Any]) -> list[str]:
    names = metadata.get("supported_joint_names")
    if isinstance(names, list) and len(names) >= 8:
        return [str(name) for name in names[:8]]
    summaries = metadata.get("run_summaries")
    if isinstance(summaries, list):
        for summary in summaries:
            if not isinstance(summary, dict):
                continue
            names = summary.get("supported_joint_names")
            if isinstance(names, list) and len(names) >= 8:
                return [str(name) for name in names[:8]]
    return DEFAULT_JOINT_NAMES


def check_onset_range(protocol: str, onset_min: int | None, onset_max: int | None) -> tuple[str, bool]:
    expected = PROTOCOL_ONSET_RANGES.get(protocol)
    if expected is None:
        return "", False
    expected_text = f"{expected[0]}..{expected[1]}"
    if onset_min is None or onset_max is None:
        return expected_text, False
    return expected_text, expected[0] <= onset_min <= onset_max <= expected[1]


def audit_dataset(dataset_label: str, dataset_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    import numpy as np

    base_row = {field: "" for field in PER_DATASET_FIELDS}
    base_row.update(
        {
            "dataset_label": dataset_label,
            "dataset_path": repo_relative(dataset_path),
            "exists": dataset_path.is_file(),
        }
    )
    if not dataset_path.is_file():
        base_row["failed_checks"] = "dataset_missing"
        return base_row, [], {"dataset_label": dataset_label, "dataset_path": repo_relative(dataset_path), "exists": False}

    metadata = load_metadata(dataset_path)
    with np.load(dataset_path, allow_pickle=False) as npz:
        keys = list(npz.files)
        missing_keys = [key for key in REQUIRED_KEYS if not has_array(npz, key)]
        failed_checks: list[str] = []
        if missing_keys:
            failed_checks.append("missing_required_keys")

        sample_count = int(npz["student_obs"].shape[0]) if has_array(npz, "student_obs") else 0
        protocol = first_label(npz["protocol_label"]) if has_array(npz, "protocol_label") else dataset_label
        velocity_mode = first_label(npz["velocity_mode_label"]) if has_array(npz, "velocity_mode_label") else ""
        joint_ids = represented_joint_ids(npz)
        joint_names = joint_names_from_metadata(metadata)
        num_envs, num_steps = infer_envs_and_timesteps(npz)

        selected_indices = (
            np.asarray(npz["selected_fault_joint_index"]).reshape(-1)
            if has_array(npz, "selected_fault_joint_index")
            else np.array([])
        )
        selected_min = int(np.min(selected_indices)) if selected_indices.size else None
        selected_max = int(np.max(selected_indices)) if selected_indices.size else None
        if selected_min is None or selected_max is None or selected_min < 0 or selected_max > 7:
            failed_checks.append("selected_fault_joint_index_out_of_range")
        if len(joint_ids) != 8 or joint_ids != list(range(8)):
            failed_checks.append("not_all_8_joints_represented")

        active = (
            np.asarray(npz["p2_fault_active"]).astype(bool).reshape(-1)
            if has_array(npz, "p2_fault_active")
            else np.array([])
        )
        has_active = bool(active.any()) if active.size else False
        has_inactive = bool((~active).any()) if active.size else False
        if not has_active or not has_inactive:
            failed_checks.append("p2_fault_active_missing_active_or_inactive_samples")

        onset = np.asarray(npz["fault_onset_step"]).reshape(-1) if has_array(npz, "fault_onset_step") else np.array([])
        onset_min = int(np.min(onset)) if onset.size else None
        onset_max = int(np.max(onset)) if onset.size else None
        expected_onset, onset_ok = check_onset_range(protocol, onset_min, onset_max)
        if not onset_ok:
            failed_checks.append("fault_onset_step_range_mismatch")

        for key, expected_dim in EXPECTED_DIMS.items():
            observed = final_dim(npz, key)
            if observed != expected_dim:
                failed_checks.append(f"{key}_dim_expected_{expected_dim}_got_{observed}")

        no_nan_inf = numeric_no_nan_inf(npz)
        if not no_nan_inf:
            failed_checks.append("nan_or_inf_detected")

        vx_cmd = np.asarray(npz["vx_cmd"]).reshape(-1) if has_array(npz, "vx_cmd") else np.array([])
        velocity_x = np.asarray(npz["velocity_x"]).reshape(-1) if has_array(npz, "velocity_x") else np.array([])
        vx_error = np.asarray(npz["vx_error"]).reshape(-1) if has_array(npz, "vx_error") else np.array([])
        abs_vx_error = np.abs(vx_error)
        yaw_error = np.asarray(npz["yaw_error"]).reshape(-1) if has_array(npz, "yaw_error") else None
        done = np.asarray(npz["done"]).astype(bool).reshape(-1) if has_array(npz, "done") else np.array([])
        done_count = int(done.sum()) if done.size else 0
        done_rate = float(done_count / done.size) if done.size else None

        row = dict(base_row)
        row.update(
            {
                "protocol": protocol,
                "velocity_mode": velocity_mode,
                "sample_count": sample_count,
                "inferred_num_envs": num_envs,
                "inferred_num_timesteps": num_steps,
                "student_obs_dim": final_dim(npz, "student_obs"),
                "teacher_obs_dim": final_dim(npz, "teacher_obs"),
                "teacher_action_dim": final_dim(npz, "teacher_action"),
                "selected_fault_joint_index_min": selected_min,
                "selected_fault_joint_index_max": selected_max,
                "represented_joint_count": len(joint_ids),
                "all_8_joints_represented": joint_ids == list(range(8)),
                "selected_fault_joint_one_hot_dim": final_dim(npz, "selected_fault_joint_one_hot"),
                "q_lock_vector_dim": final_dim(npz, "q_lock_vector"),
                "p2_fault_active_has_inactive": has_inactive,
                "p2_fault_active_has_active": has_active,
                "fault_onset_step_min": onset_min,
                "fault_onset_step_max": onset_max,
                "fault_onset_range_expected": expected_onset,
                "fault_onset_range_ok": onset_ok,
                "vx_cmd_min": scalar_stat(vx_cmd, "min"),
                "vx_cmd_max": scalar_stat(vx_cmd, "max"),
                "vx_cmd_mean": scalar_stat(vx_cmd, "mean"),
                "velocity_x_mean": scalar_stat(velocity_x, "mean"),
                "mean_vx_error": scalar_stat(vx_error, "mean"),
                "median_vx_error": scalar_stat(vx_error, "median"),
                "std_vx_error": scalar_stat(vx_error, "std"),
                "p90_vx_error": scalar_stat(vx_error, "p90"),
                "mean_abs_vx_error": scalar_stat(abs_vx_error, "mean"),
                "median_abs_vx_error": scalar_stat(abs_vx_error, "median"),
                "p90_abs_vx_error": scalar_stat(abs_vx_error, "p90"),
                "mean_yaw_error": scalar_stat(yaw_error, "mean"),
                "mean_abs_yaw_error": scalar_stat(np.abs(yaw_error), "mean") if yaw_error is not None else None,
                "done_count": done_count,
                "done_rate": done_rate,
                "survival_rate": 1.0 - done_rate if done_rate is not None else None,
                "no_nan_inf": no_nan_inf,
                "missing_keys": missing_keys,
                "failed_checks": failed_checks,
            }
        )

        per_joint_rows: list[dict[str, Any]] = []
        for joint_id in range(8):
            joint_mask = selected_indices.astype(int) == joint_id if selected_indices.size else np.zeros(sample_count, dtype=bool)
            active_mask = np.logical_and(joint_mask, active) if active.size else joint_mask
            joint_done = done[joint_mask] if done.size else np.array([])
            joint_done_rate = float(joint_done.mean()) if joint_done.size else None
            joint_yaw = yaw_error[joint_mask] if yaw_error is not None and yaw_error.size == sample_count else None
            per_joint_rows.append(
                {
                    "dataset_label": dataset_label,
                    "protocol": protocol,
                    "joint_id": joint_id,
                    "joint_name": joint_names[joint_id] if joint_id < len(joint_names) else f"joint_{joint_id}",
                    "sample_count": int(joint_mask.sum()),
                    "active_sample_count": int(active_mask.sum()) if active.size else "",
                    "mean_vx_cmd": scalar_stat(vx_cmd[joint_mask], "mean") if vx_cmd.size == sample_count else None,
                    "mean_velocity_x": scalar_stat(velocity_x[joint_mask], "mean") if velocity_x.size == sample_count else None,
                    "mean_vx_error": scalar_stat(vx_error[joint_mask], "mean") if vx_error.size == sample_count else None,
                    "median_vx_error": scalar_stat(vx_error[joint_mask], "median") if vx_error.size == sample_count else None,
                    "p90_vx_error": scalar_stat(vx_error[joint_mask], "p90") if vx_error.size == sample_count else None,
                    "mean_abs_vx_error": scalar_stat(abs_vx_error[joint_mask], "mean") if vx_error.size == sample_count else None,
                    "median_abs_vx_error": scalar_stat(abs_vx_error[joint_mask], "median") if vx_error.size == sample_count else None,
                    "p90_abs_vx_error": scalar_stat(abs_vx_error[joint_mask], "p90") if vx_error.size == sample_count else None,
                    "done_rate": joint_done_rate,
                    "survival_rate": 1.0 - joint_done_rate if joint_done_rate is not None else None,
                    "mean_yaw_error": scalar_stat(joint_yaw, "mean") if joint_yaw is not None else None,
                    "mean_abs_yaw_error": scalar_stat(np.abs(joint_yaw), "mean") if joint_yaw is not None else None,
                }
            )

        detail = {
            "dataset_label": dataset_label,
            "dataset_path": repo_relative(dataset_path),
            "exists": True,
            "array_keys": keys,
            "array_shapes": {key: list(npz[key].shape) for key in keys},
            "represented_joint_ids": joint_ids,
            "joint_names": joint_names,
            "metadata_path": repo_relative(dataset_path.parent / "metadata.json")
            if (dataset_path.parent / "metadata.json").is_file()
            else None,
        }
        return row, per_joint_rows, detail


def make_summary_json(
    *,
    per_dataset_rows: list[dict[str, Any]],
    details: list[dict[str, Any]],
    output_root: Path,
) -> dict[str, Any]:
    failed = [row for row in per_dataset_rows if row.get("failed_checks")]
    return {
        "audit_scope": "t14_multijoint_teacher_dataset_audit",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "rlm_phase": RLM_PHASE,
        "selected_teacher_label": SELECTED_TEACHER_LABEL,
        "selected_teacher_checkpoint": SELECTED_TEACHER_CHECKPOINT,
        "not_paper_grade_final": True,
        "dataset_count": len(per_dataset_rows),
        "datasets_passed_required_checks": len(failed) == 0,
        "failed_dataset_labels": [row["dataset_label"] for row in failed],
        "interpretation_notes": [
            "realistic_random and late_random are separate datasets because collecting both in one process caused the second environment creation to hang.",
            "Do not treat separate-process collection as a method failure.",
            "Survival rate is computed as 1 - done_rate; timeout should not be counted as failure if it can be separated later.",
            "Candidate-level evidence for advisor sharing; not final paper-grade reporting.",
            "Latent z_t t-SNE/UMAP belongs to later T18 after A2-history/A5 training and should use encoder output z_t, not raw H16 history.",
        ],
        "per_dataset": per_dataset_rows,
        "details": details,
        "outputs": {
            "dataset_audit_summary_md": repo_relative(output_root / "dataset_audit_summary.md"),
            "dataset_audit_summary_json": repo_relative(output_root / "dataset_audit_summary.json"),
            "per_dataset_stats_csv": repo_relative(output_root / "per_dataset_stats.csv"),
            "per_joint_stats_csv": repo_relative(output_root / "per_joint_stats.csv"),
            "advisor_update_txt": repo_relative(output_root / "advisor_update_snippet.txt"),
        },
    }


def write_summary_md(path: Path, summary: dict[str, Any], per_dataset_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# T14 Multi-Joint Dataset Audit",
        "",
        "Offline audit for the T14 multi-joint A1-F teacher rollout datasets.",
        "",
        "## Selection Context",
        "",
        f"- RLM phase: `{RLM_PHASE}`",
        f"- selected teacher: `{SELECTED_TEACHER_LABEL}`",
        f"- checkpoint: `{SELECTED_TEACHER_CHECKPOINT}`",
        "- health token: `OFF`",
        "- teacher obs dim: `77`",
        "- student obs dim: `61`",
        "",
        "## Interpretation",
        "",
        "- `realistic_random` and `late_random` are separate datasets because collecting both in one process caused the second environment creation to hang.",
        "- Do not treat this as a method failure.",
        "- This is candidate-level evidence for advisor sharing, not final paper-grade reporting.",
        "- The dataset is intended for A2 single-step, A2-history H16, A5, and optional A7 A0-anchored residual training.",
        "- Survival rate is `1 - done_rate`; timeout should not be counted as failure if timeout can be separated later.",
        "",
        "## Dataset Checks",
        "",
        "| dataset | protocol | samples | envs | steps | joints | vx_cmd_mean | mean_abs_vx_error | p90_abs_vx_error | done_rate | survival_rate | no_nan_inf | failed_checks |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in per_dataset_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    format_value(row["dataset_label"]),
                    format_value(row["protocol"]),
                    format_value(row["sample_count"]),
                    format_value(row["inferred_num_envs"]),
                    format_value(row["inferred_num_timesteps"]),
                    format_value(row["represented_joint_count"]),
                    format_value(row["vx_cmd_mean"]),
                    format_value(row["mean_abs_vx_error"]),
                    format_value(row["p90_abs_vx_error"]),
                    format_value(row["done_rate"]),
                    format_value(row["survival_rate"]),
                    format_value(row["no_nan_inf"]),
                    format_value(row["failed_checks"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Required Outputs",
            "",
            f"- JSON summary: `{summary['outputs']['dataset_audit_summary_json']}`",
            f"- per-dataset CSV: `{summary['outputs']['per_dataset_stats_csv']}`",
            f"- per-joint CSV: `{summary['outputs']['per_joint_stats_csv']}`",
            f"- advisor snippet: `{summary['outputs']['advisor_update_txt']}`",
            "",
            "## Later Graph Plan",
            "",
            "- vx_cmd vs velocity_x tracking",
            "- vx_error distribution",
            "- per-joint vx_error bar plot",
            "- survival rate per joint",
            "- fault active vs inactive tracking",
            "- latent `z_t` t-SNE/UMAP after A2-history/A5 training, not now; use encoder output `z_t`, not raw H16 history.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_advisor_snippet(path: Path, per_dataset_rows: list[dict[str, Any]]) -> None:
    lines = [
        "T14 multi-joint teacher rollout dataset audit is ready.",
        "Realistic and late random protocols were collected as separate datasets because combined collection hit a second-environment creation hang; this is treated as infrastructure behavior, not method failure.",
        "Both datasets are candidate-level evidence for downstream A2, A2-history H16, A5, and optional A7 A0-anchored residual training.",
        "",
    ]
    for row in per_dataset_rows:
        lines.append(
            f"- {row['dataset_label']}: samples={row['sample_count']}, "
            f"joints={row['represented_joint_count']}/8, "
            f"vx_cmd_mean={format_value(row['vx_cmd_mean'])}, "
            f"mean_abs_vx_error={format_value(row['mean_abs_vx_error'])}, "
            f"p90_abs_vx_error={format_value(row['p90_abs_vx_error'])}, "
            f"survival_rate={format_value(row['survival_rate'])}, "
            f"no_nan_inf={format_value(row['no_nan_inf'])}."
        )
    lines.append("")
    lines.append("Latent z_t visualization is deferred to T18 after A2-history/A5 encoder training.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_outputs(
    *,
    output_root: Path,
    summary: dict[str, Any],
    per_dataset_rows: list[dict[str, Any]],
    per_joint_rows: list[dict[str, Any]],
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    write_json(output_root / "dataset_audit_summary.json", summary)
    write_csv(output_root / "per_dataset_stats.csv", per_dataset_rows, fieldnames=PER_DATASET_FIELDS)
    write_csv(output_root / "per_joint_stats.csv", per_joint_rows, fieldnames=PER_JOINT_FIELDS)
    write_summary_md(output_root / "dataset_audit_summary.md", summary, per_dataset_rows)
    write_advisor_snippet(output_root / "advisor_update_snippet.txt", per_dataset_rows)


def main() -> int:
    args = build_parser().parse_args()
    output_root = repo_path(args.output_root)
    dataset_specs = [
        ("realistic_random", repo_path(args.realistic_dataset)),
        ("late_random", repo_path(args.late_dataset)),
    ]

    per_dataset_rows: list[dict[str, Any]] = []
    per_joint_rows: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    for label, dataset_path in dataset_specs:
        row, joint_rows, detail = audit_dataset(label, dataset_path)
        per_dataset_rows.append(row)
        per_joint_rows.extend(joint_rows)
        details.append(detail)

    summary = make_summary_json(per_dataset_rows=per_dataset_rows, details=details, output_root=output_root)
    if args.write_outputs:
        write_outputs(
            output_root=output_root,
            summary=summary,
            per_dataset_rows=per_dataset_rows,
            per_joint_rows=per_joint_rows,
        )
        print(f"[T14-AUDIT] wrote {repo_relative(output_root / 'dataset_audit_summary.md')}", flush=True)
        print(f"[T14-AUDIT] wrote {repo_relative(output_root / 'dataset_audit_summary.json')}", flush=True)
        print(f"[T14-AUDIT] wrote {repo_relative(output_root / 'per_dataset_stats.csv')}", flush=True)
        print(f"[T14-AUDIT] wrote {repo_relative(output_root / 'per_joint_stats.csv')}", flush=True)
        print(f"[T14-AUDIT] wrote {repo_relative(output_root / 'advisor_update_snippet.txt')}", flush=True)
    else:
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
