#!/usr/bin/env python3
"""Create the T09-B ablation manifest scaffold without running experiments.

The collector consumes the T09-A matrix and checkpoint pointers, then writes a
Markdown manifest template with explicit pending metric fields. It does not
launch Isaac Sim, training, evaluation, or checkpoint writes.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from run_t09_ablation_matrix import (
    MatrixParseError,
    command_preview,
    load_matrix,
    read_pointer_checkpoint_path,
    resolve_repo_path,
    row_status,
    validate_schema,
)


DEFAULT_MATRIX = "configs/ablation/t09_conference_matrix.yaml"
DEFAULT_OUTPUT = "papers/conference/results/t09_ablation_manifest_template.md"
RESIDUAL_IDS = {"A3", "A4", "A5", "A6"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="T09-B ablation manifest scaffold collector")
    parser.add_argument("--matrix", default=DEFAULT_MATRIX, help="Path to the T09-A ablation matrix")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Path to write the Markdown manifest")
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Safe manifest scaffold mode. T09-B supports scaffold generation only.",
    )
    return parser.parse_args()


def markdown_escape(value: Any) -> str:
    text = "pending" if value is None else str(value)
    return text.replace("|", "\\|").replace("\n", " ")


def pointer_info(pointer_value: Any) -> dict[str, Any]:
    if pointer_value is None or pointer_value == "":
        return {
            "pointer": "",
            "pointer_exists": False,
            "checkpoint_path": "",
            "checkpoint_exists": False,
        }

    pointer_text = str(pointer_value)
    pointer_path = resolve_repo_path(pointer_text)
    pointer_exists = pointer_path.exists()
    checkpoint_path = ""
    checkpoint_exists = False

    if pointer_exists:
        raw_checkpoint, _resolved_checkpoint, resolved_exists = read_pointer_checkpoint_path(pointer_path)
        if raw_checkpoint is not None:
            checkpoint_path = raw_checkpoint
            checkpoint_exists = bool(resolved_exists)

    return {
        "pointer": pointer_text,
        "pointer_exists": pointer_exists,
        "checkpoint_path": checkpoint_path,
        "checkpoint_exists": checkpoint_exists,
    }


def manifest_row(row: dict[str, Any], status: str) -> dict[str, Any]:
    checkpoint = pointer_info(row.get("checkpoint_pointer"))
    student_checkpoint = pointer_info(row.get("student_checkpoint_pointer"))
    row_id = str(row["ablation_id"])
    is_residual = row_id in RESIDUAL_IDS

    return {
        "ablation_id": row["ablation_id"],
        "name": row["name"],
        "stage": row["stage"],
        "method": row["method"],
        "fault": row["fault"],
        "task": row["task"],
        "seed": row["seed"],
        "first_mode": row["first_mode"],
        "launcher_family": row["launcher_family"],
        "command_preview": command_preview(row),
        "checkpoint_pointer": checkpoint["pointer"],
        "checkpoint_path": checkpoint["checkpoint_path"],
        "checkpoint_exists": checkpoint["checkpoint_exists"],
        "student_checkpoint_pointer": student_checkpoint["pointer"],
        "student_checkpoint_path": student_checkpoint["checkpoint_path"],
        "residual_scale": row.get("residual_scale", "n/a"),
        "final_action_clip": "None" if is_residual else "n/a",
        "reset_hidden_on_done": "False" if is_residual else "n/a",
        "uses_teacher_policy": row["uses_teacher_policy"],
        "uses_true_fault_state": row["uses_true_fault_state"],
        "status": status,
        "observed_metrics_status": "pending",
        "metrics_placeholder": "pending: no observed metrics recorded in T09-B",
        "validation_sources": "matrix; checkpoint pointer; code-free scaffold",
        "notes": row["notes"],
    }


def render_manifest(metadata: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    columns = [
        "ablation_id",
        "name",
        "stage",
        "method",
        "fault",
        "task",
        "seed",
        "first_mode",
        "launcher_family",
        "command_preview",
        "checkpoint_pointer",
        "checkpoint_path",
        "checkpoint_exists",
        "student_checkpoint_pointer",
        "student_checkpoint_path",
        "residual_scale",
        "final_action_clip",
        "reset_hidden_on_done",
        "uses_teacher_policy",
        "uses_true_fault_state",
        "status",
        "observed_metrics_status",
        "metrics_placeholder",
        "validation_sources",
        "notes",
    ]

    lines = [
        "# T09 Conference Ablation Manifest Template",
        "",
        "Scope: Conference-stage RLM1 stripped; manifest scaffold only; no training or evaluation executed.",
        "",
        "## Purpose",
        "",
        "This T09-B manifest scaffold consumes the T09-A ablation matrix and stable checkpoint pointers.",
        "It records row identity, checkpoint dependencies, command previews, readiness state, and pending metric fields.",
        "No training is run in T09-B. No evaluation is run in T09-B. Metrics are not fabricated.",
        "",
        "## Matrix Metadata",
        "",
        f"- stage: `{metadata.get('stage')}`",
        f"- method: `{metadata.get('method')}`",
        f"- scope: `{metadata.get('scope')}`",
        f"- health_token: `{metadata.get('health_token')}`",
        f"- uncertainty_channel: `{metadata.get('uncertainty_channel')}`",
        f"- safety_shield: `{metadata.get('safety_shield')}`",
        f"- default_seed: `{metadata.get('default_seed')}`",
        "",
        "## A0-A6 Manifest",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]

    for row in rows:
        lines.append("| " + " | ".join(markdown_escape(row.get(column, "")) for column in columns) + " |")

    lines.extend(
        [
            "",
            "## Interpretation Notes",
            "",
            "- Checkpoint fields are resolved from stable pointer YAML files.",
            "- Metric fields are marked `pending` until T09-C/T09-D records observed results.",
            "- T09-B does not execute experiments, training, or evaluation.",
            "- Health token, uncertainty channel, safety shield / CBF, and P1/P2/P3 remain inactive.",
            "",
            "## Deferred",
            "",
            "- T09-C smoke runner.",
            "- T09-D result manifest collection with observed metrics.",
            "- T09.5 ablation documentation snapshot.",
            "- Full training sweeps.",
            "- Full evaluation runner.",
            "- Fault curriculum.",
            "- Health token.",
            "- Uncertainty channel.",
            "- Safety shield / CBF.",
            "- P1/P2/P3 expansion.",
            "- New methods beyond A0-A6.",
            "- Privileged critic.",
            "- Recurrent residual PPO.",
            "",
        ]
    )

    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    matrix_path = resolve_repo_path(args.matrix)
    output_path = resolve_repo_path(args.output)

    print("[INFO] T09-B manifest scaffold collector")
    if not args.dry_run:
        print("[INFO] --dry_run omitted; T09-B still performs scaffold generation only.")
    print("[INFO] No Isaac Sim, training, evaluation, metric fabrication, or checkpoint writes are performed.")
    print(f"[INFO] Matrix path: {matrix_path}")
    print(f"[INFO] Manifest path: {output_path}")

    try:
        matrix = load_matrix(matrix_path)
    except MatrixParseError as exc:
        print(f"[ERROR] unreadable matrix syntax: {exc}", file=sys.stderr)
        return 2

    metadata = matrix["metadata"]
    source_rows = matrix["rows"]
    schema_errors = validate_schema(metadata, source_rows)
    if schema_errors:
        print("[ERROR] Matrix schema is invalid:")
        for error in schema_errors:
            print(f"  - {error}")
        return 1

    manifest_rows: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    for row in source_rows:
        status, _details = row_status(row)
        status_counts[status] += 1
        manifest_rows.append(manifest_row(row, status))
        print(
            f"[ROW {row['ablation_id']}] {row['name']}: status={status} "
            f"command_preview={command_preview(row)}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_manifest(metadata, manifest_rows), encoding="utf-8")

    print("[SUMMARY] T09-B manifest scaffold:")
    print(f"  rows_processed: {len(manifest_rows)}")
    print(f"  ready_rows: {status_counts.get('ready', 0)}")
    print(f"  rows_with_missing_pointer: {status_counts.get('missing_pointer', 0)}")
    print(f"  rows_with_missing_checkpoint: {status_counts.get('missing_checkpoint', 0)}")
    print(f"  manifest_path: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
