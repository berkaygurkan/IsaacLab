#!/usr/bin/env python3
"""Guarded one-row smoke runner for T09-C.

Default behavior is preview-only. The only executable row in this initial
T09-C scaffold is A5, which maps to the already validated T08 residual smoke
path. This script does not import Isaac Sim directly and never runs more than
one selected row.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from run_t09_ablation_matrix import (
    MatrixParseError,
    load_matrix,
    resolve_repo_path,
    row_status,
    validate_schema,
)


DEFAULT_MATRIX = "configs/ablation/t09_conference_matrix.yaml"
EXECUTABLE_ABLATION_ID = "A5"
REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="T09-C one-row ablation smoke runner")
    parser.add_argument("--matrix", default=DEFAULT_MATRIX, help="Path to the T09-A ablation matrix")
    parser.add_argument("--ablation_id", help="Ablation row to preview or execute, for example A5")
    parser.add_argument("--dry_run", action="store_true", help="Preview the selected row without executing it")
    parser.add_argument(
        "--execute_smoke",
        action="store_true",
        help="Execute the selected smoke row. Initial T09-C supports A5 only.",
    )
    parser.add_argument("--num_envs", type=int, default=8, help="Number of envs for the A5 smoke command")
    parser.add_argument("--max_iterations", type=int, default=1, help="Max PPO iterations for the A5 smoke command")
    return parser.parse_args()


def load_rows(matrix_path: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    try:
        matrix = load_matrix(matrix_path)
    except MatrixParseError as exc:
        raise RuntimeError(f"unreadable matrix syntax: {exc}") from exc

    metadata = matrix["metadata"]
    rows = matrix["rows"]
    schema_errors = validate_schema(metadata, rows)
    if schema_errors:
        joined = "\n  - ".join(schema_errors)
        raise RuntimeError(f"matrix schema is invalid:\n  - {joined}")

    return metadata, {str(row["ablation_id"]): row for row in rows}


def build_preview_command(row: dict[str, Any], num_envs: int, max_iterations: int) -> str:
    if row["ablation_id"] == EXECUTABLE_ABLATION_ID:
        return (
            "TERM=xterm ./trainers/run_t08_residual_train.sh "
            f"--headless --num_envs {num_envs} --max_iterations {max_iterations} "
            f"--residual_scale {row['residual_scale']}"
        )

    parts = [str(row["launcher_family"])]
    if "residual_scale" in row:
        parts.extend(["--residual_scale", str(row["residual_scale"])])
    return " ".join(parts)


def build_a5_subprocess(row: dict[str, Any], num_envs: int, max_iterations: int) -> tuple[list[str], dict[str, str]]:
    command = [
        "./trainers/run_t08_residual_train.sh",
        "--headless",
        "--num_envs",
        str(num_envs),
        "--max_iterations",
        str(max_iterations),
        "--residual_scale",
        str(row["residual_scale"]),
    ]
    env = os.environ.copy()
    env["TERM"] = "xterm"
    return command, env


def print_selected_row(row: dict[str, Any], status: str, details: list[str], command: str) -> None:
    print(f"[ROW {row['ablation_id']}] {row['name']}")
    print(f"  status: {status}")
    print(f"  purpose: {row['purpose']}")
    print(
        "  identity: "
        f"stage={row['stage']} method={row['method']} task={row['task']} "
        f"fault={row['fault']} seed={row['seed']}"
    )
    print(f"  first_mode: {row['first_mode']}")
    print(f"  launcher_family: {row['launcher_family']}")
    if "residual_scale" in row:
        print(f"  residual_scale: {row['residual_scale']}")
    print(f"  uses_teacher_policy: {row['uses_teacher_policy']}")
    print(f"  uses_true_fault_state: {row['uses_true_fault_state']}")
    print(f"  command_preview: {command}")
    for detail in details:
        print(detail)


def main() -> int:
    args = parse_args()
    matrix_path = resolve_repo_path(args.matrix)

    print("[INFO] T09-C one-row ablation smoke runner")
    print("[INFO] Default mode is dry-run preview. Full sweeps are not supported.")
    print("[INFO] Initial executable row: A5 only.")

    if args.execute_smoke and not args.ablation_id:
        print("[ERROR] --execute_smoke requires --ablation_id.", file=sys.stderr)
        return 2
    if not args.ablation_id:
        print("[ERROR] Select exactly one row with --ablation_id.", file=sys.stderr)
        return 2

    dry_run_mode = args.dry_run or not args.execute_smoke
    if dry_run_mode and not args.dry_run:
        print("[INFO] Neither --dry_run nor --execute_smoke was provided; using dry-run preview.")

    try:
        _metadata, rows = load_rows(matrix_path)
    except RuntimeError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    row = rows.get(args.ablation_id)
    if row is None:
        print(f"[ERROR] Unknown ablation_id: {args.ablation_id}", file=sys.stderr)
        print(f"[INFO] Available rows: {', '.join(rows)}")
        return 2

    status, details = row_status(row)
    command = build_preview_command(row, args.num_envs, args.max_iterations)

    mode = "execute_smoke" if args.execute_smoke else "dry_run"
    print(f"[INFO] Mode: {mode}")
    print_selected_row(row, status, details, command)

    if dry_run_mode:
        print("[INFO] Dry-run complete. No subprocess was executed.")
        return 0

    if row["ablation_id"] != EXECUTABLE_ABLATION_ID:
        print(
            f"[ERROR] Execution for {row['ablation_id']} is deferred. "
            "Initial T09-C only executes A5 through the validated T08 residual smoke path.",
            file=sys.stderr,
        )
        return 2

    if status != "ready":
        print(f"[ERROR] Refusing A5 execution because row status is {status}.", file=sys.stderr)
        return 1

    subprocess_command, env = build_a5_subprocess(row, args.num_envs, args.max_iterations)
    print("[INFO] Executing A5 smoke command:")
    print(f"  {command}")
    completed = subprocess.run(subprocess_command, cwd=REPO_ROOT, env=env, check=False)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
