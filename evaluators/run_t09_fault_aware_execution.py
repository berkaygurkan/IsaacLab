#!/usr/bin/env python3
"""Dry-run preflight for T09-E0 fault-aware controlled evaluation.

This script intentionally uses only the Python standard library. It never
imports Isaac Sim, task modules, RSL-RL, torch, or gym, and it never launches
training, evaluation, runtime fault injection, checkpoint writes, metric
collection, or observed-result manifest updates.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MATRIX = "configs/ablation/t09_conference_matrix.yaml"
DEFAULT_PLAN = "configs/ablation/t09_fault_aware_execution_plan.yaml"
EXPECTED_PROFILE_IDS = ["F0_none", "P4_torque_degradation", "P2_locked_joint"]
P4_PROFILE_ID = "P4_torque_degradation"
P2_PROFILE_ID = "P2_locked_joint"
P2_GATE = "P4_torque_degradation_pilot_passed"
PILOT_PREVIEW_IDS = ["A0", "A2", "A5"]
RESIDUAL_IDS = ["A3", "A4", "A5", "A6"]
EXPECTED_ROW_IDS = [f"A{i}" for i in range(7)]


class PreflightError(ValueError):
    """Raised when the dry-run preflight scaffold is inconsistent."""


def _strip_inline_comment(line: str) -> str:
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
    if lowered in {"none", "null"}:
        return lowered
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
        raise PreflightError(f"line {line_no}: expected 'key: value'")
    key, raw_value = text.split(":", 1)
    key = key.strip()
    if not key:
        raise PreflightError(f"line {line_no}: empty key")
    return key, _parse_scalar(raw_value)


def load_flat_yaml(path: Path) -> dict[str, Any]:
    """Parse the repo-owned flat YAML subset used by T09-E0 scaffolds."""
    metadata: dict[str, Any] = {}
    lists: dict[str, list[dict[str, Any]]] = {}
    current_list: str | None = None
    current_item: dict[str, Any] | None = None

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise PreflightError(f"could not read {repo_relative(path)}: {exc}") from exc

    for line_no, raw_line in enumerate(lines, start=1):
        line = _strip_inline_comment(raw_line).rstrip()
        if not line.strip():
            continue

        if not raw_line.startswith(" "):
            key, value = _split_key_value(line, line_no)
            if value == "":
                current_list = key
                current_item = None
                lists.setdefault(key, [])
            else:
                current_list = None
                current_item = None
                metadata[key] = value
            continue

        if current_list is None:
            raise PreflightError(f"line {line_no}: indented content without a list block")

        stripped = line.strip()
        if stripped.startswith("- "):
            if not line.startswith("  - "):
                raise PreflightError(f"line {line_no}: list item must use two-space indentation")
            current_item = {}
            lists[current_list].append(current_item)
            first_field = stripped[2:].strip()
            if first_field:
                key, value = _split_key_value(first_field, line_no)
                current_item[key] = value
            continue

        if not line.startswith("    "):
            raise PreflightError(f"line {line_no}: list field must use four-space indentation")
        if current_item is None:
            raise PreflightError(f"line {line_no}: list field before list item")
        key, value = _split_key_value(stripped, line_no)
        current_item[key] = value

    return {"metadata": metadata, "lists": lists}


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def repo_relative(path: str | Path) -> str:
    resolved = resolve_repo_path(path).resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(resolved)


def read_pointer_checkpoint_path(pointer_value: Any) -> tuple[str, Path, bool]:
    pointer_path = resolve_repo_path(str(pointer_value))
    if not pointer_path.is_file():
        raise PreflightError(f"checkpoint pointer missing: {repo_relative(pointer_path)}")

    pointer = load_flat_yaml(pointer_path)
    raw_checkpoint = pointer["metadata"].get("checkpoint_path")
    if not raw_checkpoint:
        raise PreflightError(f"checkpoint pointer missing checkpoint_path: {repo_relative(pointer_path)}")

    checkpoint_path = resolve_repo_path(str(raw_checkpoint))
    return str(raw_checkpoint), checkpoint_path, checkpoint_path.is_file()


def row_map(matrix: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = matrix["lists"].get("rows", [])
    rows_by_id = {str(row.get("ablation_id")): row for row in rows}
    if list(sorted(rows_by_id)) != EXPECTED_ROW_IDS:
        raise PreflightError(f"matrix rows must be exactly {EXPECTED_ROW_IDS}, got {sorted(rows_by_id)}")
    return rows_by_id


def profile_map(plan: dict[str, Any]) -> dict[str, dict[str, Any]]:
    profiles = plan["lists"].get("allowed_fault_profiles", [])
    profiles_by_id = {str(profile.get("profile_id")): profile for profile in profiles}
    if list(profiles_by_id) != EXPECTED_PROFILE_IDS:
        raise PreflightError(
            f"allowed profiles must be exactly {EXPECTED_PROFILE_IDS} in order, got {list(profiles_by_id)}"
        )
    return profiles_by_id


def validate_plan_metadata(plan: dict[str, Any]) -> None:
    metadata = plan["metadata"]
    expected = {
        "stage": "T09-E0",
        "method": "rlm1_stripped",
        "scope": "conference_stage",
        "health_token": False,
        "uncertainty_channel": False,
        "safety_shield": False,
        "p1_p3_px_expansion": False,
        "dry_run_only": True,
        "execution_enabled": False,
        "p2_gated_until": P2_GATE,
        "residual_scale_policy": "reuse_same_residual_checkpoint_eval_time_scale",
    }
    for key, expected_value in expected.items():
        actual = metadata.get(key)
        if actual != expected_value:
            raise PreflightError(f"plan metadata {key} must be {expected_value!r}, got {actual!r}")


def validate_profile_file(profile: dict[str, Any]) -> None:
    profile_id = str(profile["profile_id"])
    if profile_id == "F0_none":
        return

    config_path_value = str(profile.get("config_path", ""))
    config_path = resolve_repo_path(config_path_value)
    if not config_path.is_file():
        raise PreflightError(f"profile config missing for {profile_id}: {repo_relative(config_path)}")

    loaded = load_flat_yaml(config_path)["metadata"]
    required_pairs = {
        "profile_id": profile_id,
        "family": profile["family"],
        "stage": "T09-E0",
        "execution_enabled": False,
        "runtime_fault_injection_implemented": False,
    }
    for key, expected_value in required_pairs.items():
        actual = loaded.get(key)
        if actual != expected_value:
            raise PreflightError(
                f"profile {profile_id} field {key} must be {expected_value!r}, got {actual!r}"
            )

    if profile_id == P2_PROFILE_ID:
        if loaded.get("status") != "gated":
            raise PreflightError("P2 profile must be marked gated")
        if loaded.get("gated_by") != P2_GATE:
            raise PreflightError(f"P2 profile must be gated_by {P2_GATE}")


def validate_profiles(plan: dict[str, Any]) -> dict[str, dict[str, Any]]:
    profiles_by_id = profile_map(plan)
    p4 = profiles_by_id[P4_PROFILE_ID]
    p2 = profiles_by_id[P2_PROFILE_ID]

    if p4.get("status") != "active_for_preview":
        raise PreflightError("P4_torque_degradation must be active_for_preview")
    if p4.get("family") != "torque_scale":
        raise PreflightError("P4_torque_degradation must use torque_scale family")
    if p4.get("execution_enabled") is not False:
        raise PreflightError("P4_torque_degradation execution_enabled must be false")

    if p2.get("status") != "gated":
        raise PreflightError("P2_locked_joint must be gated")
    if p2.get("family") != "joint_lock":
        raise PreflightError("P2_locked_joint must use joint_lock family")
    if p2.get("gated_by") != P2_GATE:
        raise PreflightError(f"P2_locked_joint must be gated_by {P2_GATE}")
    if p2.get("execution_enabled") is not False:
        raise PreflightError("P2_locked_joint execution_enabled must be false")

    for profile in profiles_by_id.values():
        validate_profile_file(profile)
    return profiles_by_id


def validate_row_roles(plan: dict[str, Any], rows_by_id: dict[str, dict[str, Any]]) -> None:
    pilot_rows = plan["lists"].get("pilot_preview_rows", [])
    pilot_ids = [str(row.get("ablation_id")) for row in pilot_rows]
    if pilot_ids != PILOT_PREVIEW_IDS:
        raise PreflightError(f"pilot preview rows must be {PILOT_PREVIEW_IDS}, got {pilot_ids}")

    roles = {str(row["ablation_id"]): row for row in pilot_rows}
    expected_roles = {
        "A0": "zero_shot_healthy_ppo_fault_baseline",
        "A2": "frozen_student_fault_evaluation",
        "A5": "frozen_residual_policy_adaptation",
    }
    for row_id, expected_role in expected_roles.items():
        if roles[row_id].get("role") != expected_role:
            raise PreflightError(f"{row_id} role must be {expected_role}")
        if roles[row_id].get("fault_profile") != P4_PROFILE_ID:
            raise PreflightError(f"{row_id} pilot fault profile must be {P4_PROFILE_ID}")
        if roles[row_id].get("online_learning") is not False:
            raise PreflightError(f"{row_id} online_learning must be false")

    reference_rows = plan["lists"].get("reference_only_rows", [])
    if len(reference_rows) != 1 or reference_rows[0].get("ablation_id") != "A1":
        raise PreflightError("A1 must be the only reference-only row")
    if reference_rows[0].get("deployment_facing") is not False:
        raise PreflightError("A1 must not be deployment-facing")
    if rows_by_id["A1"].get("uses_true_fault_state") is not True:
        raise PreflightError("A1 matrix row must remain privileged reference with true_fault_state")

    residual_rows = plan["lists"].get("residual_scale_eval_rows", [])
    residual_ids = [str(row.get("ablation_id")) for row in residual_rows]
    if residual_ids != RESIDUAL_IDS:
        raise PreflightError(f"residual scale rows must be {RESIDUAL_IDS}, got {residual_ids}")
    for row in residual_rows:
        row_id = str(row["ablation_id"])
        if row.get("checkpoint_reuse") != "same_residual_checkpoint":
            raise PreflightError(f"{row_id} must reuse the same residual checkpoint")
        if row.get("residual_scale_source") != "matrix":
            raise PreflightError(f"{row_id} residual scale source must be matrix")


def checkpoint_info(row: dict[str, Any], field: str = "checkpoint_pointer") -> dict[str, Any]:
    pointer = str(row[field])
    raw_checkpoint, checkpoint_path, checkpoint_exists = read_pointer_checkpoint_path(pointer)
    if not checkpoint_exists:
        raise PreflightError(
            f"resolved checkpoint missing for {row['ablation_id']} {field}: {repo_relative(checkpoint_path)}"
        )
    return {
        "pointer": pointer,
        "checkpoint_path": raw_checkpoint,
        "checkpoint_exists": checkpoint_exists,
    }


def validate_checkpoint_dependencies(rows_by_id: dict[str, dict[str, Any]]) -> None:
    for row_id in PILOT_PREVIEW_IDS:
        checkpoint_info(rows_by_id[row_id])

    residual_pointer: str | None = None
    student_pointer: str | None = None
    for row_id in RESIDUAL_IDS:
        row = rows_by_id[row_id]
        residual = checkpoint_info(row, "checkpoint_pointer")
        student = checkpoint_info(row, "student_checkpoint_pointer")
        if residual_pointer is None:
            residual_pointer = residual["pointer"]
            student_pointer = student["pointer"]
        if residual["pointer"] != residual_pointer:
            raise PreflightError("A3/A4/A5/A6 must share the same residual checkpoint pointer")
        if student["pointer"] != student_pointer:
            raise PreflightError("A3/A4/A5/A6 must share the same student checkpoint pointer")


def future_preview_command(
    row_id: str,
    fault_profile: str,
    seed: int,
    episodes: int,
    num_envs: int,
    policy_mode: str,
) -> str:
    parts = [
        "python",
        "evaluators/run_t09_fault_aware_execution.py",
        "--dry_run",
        "--preview",
        "--ablation_id",
        row_id,
        "--fault_profile",
        fault_profile,
        "--seed",
        str(seed),
        "--episodes",
        str(episodes),
        "--num_envs",
        str(num_envs),
        "--policy_mode",
        policy_mode,
    ]
    return " ".join(parts)


def print_no_execution_summary() -> None:
    print("[T09-E0 SUMMARY] dry-run-only preflight complete")
    print("  no_training: True")
    print("  no_evaluation: True")
    print("  no_isaac_sim: True")
    print("  no_runtime_fault_injection: True")
    print("  no_checkpoint_writes: True")
    print("  no_metrics: True")
    print("  no_observed_rows: True")


def print_profile_summary(profiles_by_id: dict[str, dict[str, Any]]) -> None:
    print("[T09-E0 PROFILE PREFLIGHT]")
    for profile_id in EXPECTED_PROFILE_IDS:
        profile = profiles_by_id[profile_id]
        print(
            f"  {profile_id}: family={profile.get('family')} status={profile.get('status')} "
            f"execution_enabled={profile.get('execution_enabled')}"
        )
        if profile_id == P2_PROFILE_ID:
            print(f"    gated_by: {profile.get('gated_by')}")


def print_preview(
    row_id: str,
    fault_profile: str,
    seed: int,
    episodes: int,
    num_envs: int,
    policy_mode: str,
    rows_by_id: dict[str, dict[str, Any]],
) -> None:
    row = rows_by_id[row_id]
    primary = checkpoint_info(row, "checkpoint_pointer")
    print("[T09-E0 COMMAND PREVIEW]")
    print(f"  ablation_id: {row_id}")
    print(f"  row_name: {row['name']}")
    print(f"  row_role: {pilot_role(row_id)}")
    print(f"  fault_profile: {fault_profile}")
    print(f"  seed: {seed}")
    print(f"  episodes: {episodes}")
    print(f"  num_envs: {num_envs}")
    print(f"  policy_mode: {policy_mode}")
    print(f"  checkpoint_pointer: {primary['pointer']}")
    print(f"  resolved_checkpoint_path: {primary['checkpoint_path']}")
    if row_id in RESIDUAL_IDS:
        student = checkpoint_info(row, "student_checkpoint_pointer")
        print(f"  student_checkpoint_pointer: {student['pointer']}")
        print(f"  resolved_student_checkpoint_path: {student['checkpoint_path']}")
        print(f"  residual_scale: {row['residual_scale']} (matrix)")
        print("  checkpoint_reuse: same_residual_checkpoint")
    print("  execution_disabled_reason: T09-E0 is dry-run-only; no execution flag exists.")
    print(
        "  future_preview_command: "
        f"{future_preview_command(row_id, fault_profile, seed, episodes, num_envs, policy_mode)}"
    )


def pilot_role(row_id: str) -> str:
    if row_id == "A0":
        return "zero-shot healthy PPO fault baseline"
    if row_id == "A2":
        return "frozen student fault evaluation"
    if row_id == "A5":
        return "frozen residual-policy adaptation"
    if row_id == "A1":
        return "privileged reference only"
    if row_id in {"A3", "A4", "A6"}:
        return "residual-scale sensitivity readiness"
    return "matrix row"


def print_gated_preview(fault_profile: str, seed: int, profiles_by_id: dict[str, dict[str, Any]]) -> None:
    profile = profiles_by_id[fault_profile]
    print("[T09-E0 GATED PREVIEW]")
    print(f"  fault_profile: {fault_profile}")
    print(f"  family: {profile.get('family')}")
    print(f"  status: {profile.get('status')}")
    print(f"  seed: {seed}")
    print(f"  gated_by: {profile.get('gated_by')}")
    print("  preview_executable: False")
    print("  execution_disabled_reason: P2 remains gated until P4 pilot passes with a real pass record.")
    print(
        "  future_preview_command: python evaluators/run_t09_fault_aware_execution.py "
        f"--dry_run --preview_gated --fault_profile {fault_profile} --seed {seed}"
    )


def load_inputs(matrix_path: str, plan_path: str) -> tuple[dict[str, Any], dict[str, Any]]:
    matrix = load_flat_yaml(resolve_repo_path(matrix_path))
    plan = load_flat_yaml(resolve_repo_path(plan_path))
    return matrix, plan


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="T09-E0 fault-aware dry-run preflight")
    parser.add_argument("--matrix", default=DEFAULT_MATRIX, help="Path to the T09-A matrix")
    parser.add_argument("--plan", default=DEFAULT_PLAN, help="Path to the T09-E0 execution plan")
    parser.add_argument("--dry_run", action="store_true", help="Dry-run mode. T09-E0 supports dry-run only.")
    parser.add_argument("--verify_profiles", action="store_true", help="Validate fault profile metadata and gates")
    parser.add_argument("--preview", action="store_true", help="Print a command preview for one allowed row/profile")
    parser.add_argument("--preview_gated", action="store_true", help="Print a gated preview for P2")
    parser.add_argument("--ablation_id", choices=EXPECTED_ROW_IDS, help="Ablation row to preview")
    parser.add_argument("--fault_profile", choices=EXPECTED_PROFILE_IDS, help="Fault profile to preview")
    parser.add_argument("--seed", type=int, default=0, help="Seed label for preview metadata")
    parser.add_argument("--episodes", type=int, default=100, help="Episode budget label for preview metadata")
    parser.add_argument("--num_envs", type=int, default=32, help="Environment count label for preview metadata")
    parser.add_argument("--policy_mode", default="deterministic", help="Policy mode label for preview metadata")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.dry_run:
        print("[INFO] --dry_run omitted; T09-E0 still performs dry-run preflight only.")

    try:
        matrix, plan = load_inputs(args.matrix, args.plan)
        validate_plan_metadata(plan)
        rows_by_id = row_map(matrix)
        profiles_by_id = validate_profiles(plan)
        validate_row_roles(plan, rows_by_id)
        validate_checkpoint_dependencies(rows_by_id)

        if args.verify_profiles or (not args.preview and not args.preview_gated):
            print_profile_summary(profiles_by_id)

        if args.preview:
            if not args.ablation_id:
                raise PreflightError("--preview requires --ablation_id")
            if not args.fault_profile:
                raise PreflightError("--preview requires --fault_profile")
            if args.fault_profile == P2_PROFILE_ID:
                raise PreflightError("P2_locked_joint is gated; use --preview_gated")
            if args.fault_profile not in {"F0_none", P4_PROFILE_ID}:
                raise PreflightError(f"unsupported preview fault profile: {args.fault_profile}")
            if args.fault_profile == P4_PROFILE_ID and args.ablation_id not in PILOT_PREVIEW_IDS:
                raise PreflightError(f"P4 pilot previews are restricted to {PILOT_PREVIEW_IDS}")
            if args.ablation_id == "A1":
                raise PreflightError("A1 is privileged reference only and not deployment-facing")
            print_preview(
                args.ablation_id,
                args.fault_profile,
                args.seed,
                args.episodes,
                args.num_envs,
                args.policy_mode,
                rows_by_id,
            )

        if args.preview_gated:
            if args.fault_profile != P2_PROFILE_ID:
                raise PreflightError("--preview_gated is reserved for P2_locked_joint")
            print_gated_preview(args.fault_profile, args.seed, profiles_by_id)

        print_no_execution_summary()
    except PreflightError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        print_no_execution_summary()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
