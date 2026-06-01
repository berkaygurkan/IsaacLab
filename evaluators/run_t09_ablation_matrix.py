#!/usr/bin/env python3
"""Dry-run validator for the T09-A conference ablation matrix.

This script intentionally uses only the Python standard library and never
launches Isaac Sim, training, evaluation, or checkpoint writes.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_IDS = [f"A{i}" for i in range(7)]

REQUIRED_METADATA_FIELDS = {
    "stage",
    "method",
    "scope",
    "health_token",
    "uncertainty_channel",
    "safety_shield",
    "default_seed",
}

REQUIRED_ROW_FIELDS = {
    "ablation_id",
    "name",
    "purpose",
    "stage",
    "method",
    "task",
    "fault",
    "seed",
    "checkpoint_pointer",
    "launcher_family",
    "first_mode",
    "paper_table_role",
    "uses_teacher_policy",
    "uses_true_fault_state",
    "notes",
}

RESIDUAL_IDS = {"A3", "A4", "A5", "A6"}


class MatrixParseError(ValueError):
    """Raised when the repo-owned YAML subset cannot be parsed."""


def _strip_inline_comment(line: str) -> str:
    """Strip comments outside the simple unquoted value subset used here."""
    if "#" not in line:
        return line
    return line.split("#", 1)[0].rstrip()


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if value:
        try:
            return int(value)
        except ValueError:
            pass
        try:
            return float(value)
        except ValueError:
            pass
    return value


def _split_key_value(text: str, line_no: int) -> tuple[str, Any]:
    if ":" not in text:
        raise MatrixParseError(f"line {line_no}: expected 'key: value'")
    key, raw_value = text.split(":", 1)
    key = key.strip()
    if not key:
        raise MatrixParseError(f"line {line_no}: empty key")
    return key, _parse_scalar(raw_value)


def load_matrix(path: Path) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    in_rows = False
    current_row: dict[str, Any] | None = None

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise MatrixParseError(f"could not read matrix: {exc}") from exc

    for line_no, raw_line in enumerate(lines, start=1):
        line = _strip_inline_comment(raw_line).rstrip()
        if not line.strip():
            continue

        if not raw_line.startswith(" "):
            key, value = _split_key_value(line, line_no)
            if key == "rows":
                if value != "":
                    raise MatrixParseError(f"line {line_no}: rows must be a block")
                in_rows = True
                continue
            if in_rows:
                raise MatrixParseError(f"line {line_no}: top-level key after rows is not supported")
            metadata[key] = value
            continue

        if not in_rows:
            raise MatrixParseError(f"line {line_no}: indented content before rows")

        stripped = line.strip()
        if stripped.startswith("- "):
            if not line.startswith("  - "):
                raise MatrixParseError(f"line {line_no}: row item must use two-space indentation")
            current_row = {}
            rows.append(current_row)
            first_field = stripped[2:].strip()
            if first_field:
                key, value = _split_key_value(first_field, line_no)
                current_row[key] = value
            continue

        if not line.startswith("    "):
            raise MatrixParseError(f"line {line_no}: row field must use four-space indentation")
        if current_row is None:
            raise MatrixParseError(f"line {line_no}: row field before row item")
        key, value = _split_key_value(stripped, line_no)
        current_row[key] = value

    return {"metadata": metadata, "rows": rows}


def resolve_repo_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def read_pointer_checkpoint_path(pointer_path: Path) -> tuple[str | None, Path | None, bool | None]:
    """Return raw checkpoint_path, resolved path, and existence if present."""
    try:
        lines = pointer_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None, None, None

    for line_no, raw_line in enumerate(lines, start=1):
        line = _strip_inline_comment(raw_line).strip()
        if not line:
            continue
        if ":" not in line:
            continue
        key, value = _split_key_value(line, line_no)
        if key == "checkpoint_path":
            raw_path = str(value)
            resolved_path = resolve_repo_path(raw_path)
            return raw_path, resolved_path, resolved_path.exists()
    return None, None, None


def validate_schema(metadata: dict[str, Any], rows: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []

    missing_metadata = sorted(REQUIRED_METADATA_FIELDS - metadata.keys())
    if missing_metadata:
        errors.append(f"missing metadata fields: {', '.join(missing_metadata)}")

    ids = [str(row.get("ablation_id", "")) for row in rows]
    if ids != EXPECTED_IDS:
        errors.append(f"expected ablation ids {EXPECTED_IDS}, got {ids}")

    for index, row in enumerate(rows):
        row_id = str(row.get("ablation_id", f"row_{index}"))
        missing = sorted(REQUIRED_ROW_FIELDS - row.keys())
        if missing:
            errors.append(f"{row_id}: missing required fields: {', '.join(missing)}")

        if row_id in RESIDUAL_IDS:
            for field in ("student_checkpoint_pointer", "residual_scale"):
                if field not in row:
                    errors.append(f"{row_id}: missing residual field: {field}")
        else:
            if "student_checkpoint_pointer" in row:
                errors.append(f"{row_id}: non-residual row must not set student_checkpoint_pointer")
            if "residual_scale" in row:
                errors.append(f"{row_id}: non-residual row must not set residual_scale")

    return errors


def command_preview(row: dict[str, Any]) -> str:
    parts = [str(row["launcher_family"])]
    if "residual_scale" in row:
        parts.extend(["--residual_scale", str(row["residual_scale"])])
    return " ".join(parts)


def pointer_status(label: str, pointer_value: str) -> tuple[str | None, list[str]]:
    lines: list[str] = []
    pointer_path = resolve_repo_path(pointer_value)
    pointer_exists = pointer_path.exists()
    lines.append(f"  {label}: {pointer_value} (exists: {pointer_exists})")

    if not pointer_exists:
        return "missing_pointer", lines

    raw_checkpoint, checkpoint_path, checkpoint_exists = read_pointer_checkpoint_path(pointer_path)
    if raw_checkpoint is None:
        lines.append(f"  {label}_checkpoint_path: not declared")
        return None, lines

    lines.append(f"  {label}_checkpoint_path: {raw_checkpoint} (exists: {checkpoint_exists})")
    if checkpoint_exists is False:
        return "missing_checkpoint", lines
    return None, lines


def row_status(row: dict[str, Any]) -> tuple[str, list[str]]:
    details: list[str] = []

    if row.get("first_mode") != "dry_run":
        return "unsupported_mode", details

    status, pointer_lines = pointer_status("checkpoint_pointer", str(row["checkpoint_pointer"]))
    details.extend(pointer_lines)
    if status is not None:
        return status, details

    if "student_checkpoint_pointer" in row:
        status, student_lines = pointer_status(
            "student_checkpoint_pointer", str(row["student_checkpoint_pointer"])
        )
        details.extend(student_lines)
        if status is not None:
            return status, details

    return "ready", details


def print_row(row: dict[str, Any], status: str, details: list[str]) -> None:
    row_id = row["ablation_id"]
    print(f"[ROW {row_id}] {row['name']}")
    print(f"  status: {status}")
    print(f"  purpose: {row['purpose']}")
    print(
        "  identity: "
        f"stage={row['stage']} method={row['method']} task={row['task']} "
        f"fault={row['fault']} seed={row['seed']}"
    )
    print(f"  uses_teacher_policy: {row['uses_teacher_policy']}")
    print(f"  uses_true_fault_state: {row['uses_true_fault_state']}")
    if "residual_scale" in row:
        print(f"  residual_scale: {row['residual_scale']}")
    print(f"  command_preview: {command_preview(row)}")
    print(f"  paper_table_role: {row['paper_table_role']}")
    for detail in details:
        print(detail)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="T09-A conference ablation matrix dry-run validator")
    parser.add_argument(
        "--matrix",
        default="configs/ablation/t09_conference_matrix.yaml",
        help="Path to the T09-A ablation matrix",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Validate and preview only. T09-A supports dry-run only.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    matrix_path = resolve_repo_path(args.matrix)

    print("[INFO] T09-A dry-run validator")
    if not args.dry_run:
        print("[INFO] --dry_run omitted; T09-A supports dry-run only, so no execution will start.")
    print("[INFO] No Isaac Sim, training, evaluation, or checkpoint writes are performed.")
    print(f"[INFO] Matrix path: {matrix_path}")

    try:
        matrix = load_matrix(matrix_path)
    except MatrixParseError as exc:
        print(f"[ERROR] unreadable matrix syntax: {exc}", file=sys.stderr)
        return 2

    metadata = matrix["metadata"]
    rows = matrix["rows"]
    schema_errors = validate_schema(metadata, rows)

    print(
        "[INFO] Metadata: "
        f"stage={metadata.get('stage')} method={metadata.get('method')} "
        f"scope={metadata.get('scope')} health_token={metadata.get('health_token')} "
        f"uncertainty_channel={metadata.get('uncertainty_channel')} "
        f"safety_shield={metadata.get('safety_shield')} default_seed={metadata.get('default_seed')}"
    )

    statuses: Counter[str] = Counter()
    if schema_errors:
        statuses["invalid_schema"] = len(rows) or 1
        print("[ERROR] Matrix schema is invalid:")
        for error in schema_errors:
            print(f"  - {error}")
    else:
        for row in rows:
            status, details = row_status(row)
            statuses[status] += 1
            print_row(row, status, details)

    print("[SUMMARY] row status counts:")
    for status in ("ready", "missing_pointer", "missing_checkpoint", "invalid_schema", "unsupported_mode"):
        print(f"  {status}: {statuses.get(status, 0)}")

    return 1 if schema_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
