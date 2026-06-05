#!/usr/bin/env python3
"""T09 checkpoint readiness audit.

Stdlib-only. This script reads checkpoint pointer YAML files, inspects resolved
paths, classifies readiness conservatively, and writes a Markdown audit.
"""

from __future__ import annotations

import argparse
import re
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = "papers/conference/results/t09_checkpoint_readiness_audit.md"
A0_DEMO_CHECKPOINT = (
    "logs/rsl_rl/healthy_baseline__rlm1_stripped__demo/"
    "2026-06-03_04-08-57_healthy_demo__seed0/model_1999.pt"
)

ROWS = [
    {
        "ablation_id": "A0",
        "row_name": "healthy PPO baseline",
        "stage": "healthy_baseline",
        "fault": "none",
        "seed": 0,
        "candidate_path": A0_DEMO_CHECKPOINT,
        "role": "deployment-facing healthy baseline; trained F0 and evaluated F0/P2",
    },
    {
        "ablation_id": "A1",
        "row_name": "A1-F P2 privileged teacher",
        "stage": "teacher_p2",
        "fault": "none",
        "seed": 0,
        "candidate_path": None,
        "role": "future P2 privileged teacher required before A2 distillation; not deployment-facing",
        "legacy_pointer_path": "checkpoints/rlm1_stripped/teacher/none/seed0/latest_checkpoint.yaml",
    },
    {
        "ablation_id": "A2",
        "row_name": "student no residual",
        "stage": "student",
        "fault": "none",
        "seed": 0,
        "candidate_path": None,
        "role": "future P2-distilled student candidate; no online learning and no privileged runtime inputs",
    },
    {
        "ablation_id": "A5",
        "row_name": "student + residual baseline",
        "stage": "residual",
        "fault": "none",
        "seed": 0,
        "candidate_path": None,
        "role": "future main P2 residual-policy candidate; no online learning and no privileged runtime inputs",
        "dependency_pointer_path": "checkpoints/rlm1_stripped/student/none/seed0/latest_checkpoint.yaml",
    },
    {
        "ablation_id": "A7",
        "row_name": "healthy PPO + teacher-distilled residual",
        "stage": "teacher_distilled_residual",
        "fault": "none",
        "seed": 0,
        "candidate_path": None,
        "role": "optional deferred P2 teacher-distilled residual ablation; not paper-main",
        "optional_deferred": True,
        "base_dependency_pointer_path": "checkpoints/rlm1_stripped/healthy_baseline/none/seed0/canonical_checkpoint.yaml",
        "teacher_dependency_pointer_path": "checkpoints/rlm1_stripped/teacher_p2/none/seed0/canonical_checkpoint.yaml",
    },
]


def resolve_repo_path(path_value: str | Path | None) -> Path | None:
    if path_value is None:
        return None
    path = Path(path_value)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def repo_relative(path_value: str | Path | None) -> str:
    if path_value is None:
        return "NA"
    path = resolve_repo_path(path_value)
    if path is None:
        return "NA"
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(resolved)


def parse_pointer(path_value: str | None) -> dict[str, Any]:
    if path_value is None:
        return {}
    path = resolve_repo_path(path_value)
    if path is None or not path.is_file():
        return {"pointer_exists": False}
    metadata: dict[str, Any] = {"pointer_exists": True}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        metadata[key.strip()] = value.strip()
    return metadata


def model_iteration(path_value: str | None) -> int | None:
    if not path_value:
        return None
    match = re.search(r"model_(\d+)\.pt$", Path(path_value).name)
    return int(match.group(1)) if match else None


def checkpoint_exists(path_value: str | None) -> bool:
    path = resolve_repo_path(path_value)
    return bool(path and path.is_file())


def pointer_path(*, stage: str, fault: str, seed: int, pointer_name: str) -> str:
    return f"checkpoints/rlm1_stripped/{stage}/{fault}/seed{seed}/{pointer_name}"


def select_primary_pointer(row: dict[str, Any]) -> tuple[str, str, dict[str, Any], str, str]:
    canonical_path = pointer_path(
        stage=row["stage"],
        fault=row["fault"],
        seed=int(row["seed"]),
        pointer_name="canonical_checkpoint.yaml",
    )
    latest_path = pointer_path(
        stage=row["stage"],
        fault=row["fault"],
        seed=int(row["seed"]),
        pointer_name="latest_checkpoint.yaml",
    )
    canonical_pointer = parse_pointer(canonical_path)
    if canonical_pointer.get("pointer_exists"):
        return "canonical", canonical_path, canonical_pointer, canonical_path, latest_path
    latest_pointer = parse_pointer(latest_path)
    return "latest", latest_path, latest_pointer, canonical_path, latest_path


def classify_checkpoint(
    *,
    ablation_id: str,
    pointer_type: str,
    pointer: dict[str, Any],
    pointer_resolved_path: str | None,
    optional_deferred: bool = False,
) -> tuple[str, str, str]:
    if optional_deferred:
        return (
            "optional_deferred",
            "False",
            "Optional A7 registration only; no checkpoint is expected until explicitly implemented and promoted.",
        )
    if not pointer_resolved_path or not checkpoint_exists(pointer_resolved_path):
        return "missing", "False", "Checkpoint file is missing; cannot launch controlled evaluation."

    iteration = model_iteration(pointer_resolved_path)
    stored_classification = pointer.get("classification")
    if pointer_type == "canonical":
        if stored_classification == "paper_grade_candidate" and iteration != 0:
            return (
                "paper_grade_candidate",
                "True",
                "Canonical pointer explicitly stores paper_grade_candidate and resolves to a non-model_0 checkpoint.",
            )
        if stored_classification in {"demo_grade_candidate", "smoke_or_dev_only"}:
            return (
                stored_classification,
                "False",
                f"Canonical pointer explicitly stores {stored_classification}; not controlled-eval-ready.",
            )
        return (
            "unknown_needs_manual_review",
            "False",
            "Canonical pointer exists but does not explicitly store a controlled-eval-ready classification.",
        )

    if ablation_id == "A5":
        return (
            "smoke_or_dev_only",
            "False",
            "Residual pointer remains smoke/development unless explicitly promoted by a future freeze step.",
        )
    if iteration == 0:
        return (
            "smoke_or_dev_only",
            "False",
            "Resolved checkpoint is model_0.pt; treat as smoke/development unless evidence proves otherwise.",
        )
    return (
        "unknown_needs_manual_review",
        "False",
        "Latest pointer resolves to a non-model_0 checkpoint, but no canonical freeze classification exists.",
    )


def recommended_next_action(ablation_id: str, classification: str) -> str:
    if ablation_id == "A0":
        if classification == "paper_grade_candidate":
            return "A0 canonical checkpoint is ready for future F0/P2 controlled evaluation; keep evaluation blocked until A1-F/A2/A5 P2 readiness is addressed."
        return (
            "Keep demo model_1999 for advisor playback only; train or explicitly freeze a canonical A0 checkpoint "
            "before controlled evaluation."
        )
    if ablation_id == "A1":
        if classification == "paper_grade_candidate":
            return "Confirm this is the P2-trained A1-F privileged teacher before using it as the source for A2 student distillation."
        return "Train/select/freeze A1-F under P2, then use checkpoints/rlm1_stripped/teacher_p2/none/seed0/canonical_checkpoint.yaml before A2 distillation."
    if ablation_id == "A2":
        return "Distill a non-smoke A2 student from the P2-trained A1-F teacher, then freeze the exact path."
    if ablation_id == "A5":
        return "Train/select canonical P2 student and residual checkpoints; do not use the current residual pointer as paper evidence."
    if ablation_id == "A7":
        return "Keep A7 deferred; do not train or evaluate unless explicitly promoted after A0 and P2 A1-F canonical dependencies exist."
    return f"Manual review required for classification {classification}."


def audit_rows() -> list[dict[str, Any]]:
    audited = []
    for row in ROWS:
        pointer_type, primary_pointer_path, pointer, canonical_pointer_path, latest_pointer_path = select_primary_pointer(row)
        pointer_resolved_path = pointer.get("checkpoint_path")
        candidate_path = row.get("candidate_path")
        dependency_pointer = parse_pointer(row.get("dependency_pointer_path"))
        classification, controlled_eval_use, rationale = classify_checkpoint(
            ablation_id=row["ablation_id"],
            pointer_type=pointer_type,
            pointer=pointer,
            pointer_resolved_path=pointer_resolved_path,
            optional_deferred=bool(row.get("optional_deferred")),
        )
        audited.append(
            {
                "ablation_id": row["ablation_id"],
                "row_name": row["row_name"],
                "role": row["role"],
                "stage": row["stage"],
                "fault": row["fault"],
                "seed": row["seed"],
                "pointer_type": pointer_type,
                "pointer_path": primary_pointer_path,
                "canonical_pointer_path": canonical_pointer_path,
                "latest_pointer_path": latest_pointer_path,
                "pointer_exists": pointer.get("pointer_exists", False),
                "resolved_checkpoint_path": pointer_resolved_path,
                "checkpoint_exists": checkpoint_exists(pointer_resolved_path),
                "checkpoint_filename": Path(pointer_resolved_path).name if pointer_resolved_path else "NA",
                "model_iteration": model_iteration(pointer_resolved_path),
                "log_dir": pointer.get("log_dir"),
                "pointer_stored_classification": pointer.get("classification"),
                "candidate_checkpoint_path": candidate_path,
                "candidate_checkpoint_exists": checkpoint_exists(candidate_path) if candidate_path else "NA",
                "dependency_pointer_path": row.get("dependency_pointer_path"),
                "dependency_checkpoint_path": dependency_pointer.get("checkpoint_path"),
                "base_dependency_pointer_path": row.get("base_dependency_pointer_path"),
                "teacher_dependency_pointer_path": row.get("teacher_dependency_pointer_path"),
                "legacy_pointer_path": row.get("legacy_pointer_path"),
                "classification": classification,
                "controlled_eval_use": controlled_eval_use,
                "rationale": rationale,
                "recommended_next_action": recommended_next_action(row["ablation_id"], classification),
                "selected_path_for_classification": pointer_resolved_path,
            }
        )
    return audited


def markdown_table(rows: list[dict[str, Any]]) -> str:
    headers = [
        "ablation_id",
        "row_name",
        "pointer_type",
        "canonical_pointer_path",
        "latest_pointer_path",
        "pointer_path",
        "resolved_checkpoint_path",
        "checkpoint_exists",
        "checkpoint_filename",
        "log_dir",
        "candidate_checkpoint_path",
        "classification",
        "controlled_eval_use",
        "recommended_next_action",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        values = []
        for header in headers:
            value = row.get(header)
            if header.endswith("path") or header == "log_dir":
                value = f"`{repo_relative(value)}`" if value not in (None, "NA") else "NA"
            values.append(str(value).replace("\n", " "))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def render_markdown(rows: list[dict[str, Any]]) -> str:
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "# T09 Checkpoint Readiness Audit",
        "",
        "Scope: RLM1 stripped conference-stage checkpoint readiness audit; no training, evaluation, play, or Isaac Sim executed.",
        "",
        f"Generated: {generated_at}",
        "",
        "## Method State",
        "",
        "- Health token OFF.",
        "- UQ inactive.",
        "- CBF inactive.",
        "- `P2_locked_joint` is the future conference fault scope.",
        "- No P2 runtime execution or P3 expansion.",
        "- P4 advisor-demo artifacts remain separate from controlled-evaluation artifacts.",
        "",
        "## Readiness Table",
        "",
        markdown_table(rows),
        "",
        "## Detailed Notes",
        "",
    ]
    for row in rows:
        lines.extend(
            [
                f"### {row['ablation_id']} - {row['row_name']}",
                "",
                f"- Role: {row['role']}",
                f"- Pointer type used: `{row['pointer_type']}`",
                f"- Canonical pointer path: `{repo_relative(row['canonical_pointer_path'])}`",
                f"- Latest pointer path: `{repo_relative(row['latest_pointer_path'])}`",
                f"- Classification: `{row['classification']}`",
                f"- Rationale: {row['rationale']}",
                f"- Primary pointer resolved path: `{repo_relative(row['resolved_checkpoint_path'])}`",
                f"- Pointer-stored classification: `{row['pointer_stored_classification']}`",
                f"- Candidate checkpoint path: `{repo_relative(row['candidate_checkpoint_path'])}`",
                f"- Dependency checkpoint path: `{repo_relative(row['dependency_checkpoint_path'])}`",
                f"- Base dependency pointer path: `{repo_relative(row['base_dependency_pointer_path'])}`",
                f"- Teacher dependency pointer path: `{repo_relative(row['teacher_dependency_pointer_path'])}`",
                f"- Legacy pointer path: `{repo_relative(row['legacy_pointer_path'])}`",
                f"- Recommended next action: {row['recommended_next_action']}",
                "",
            ]
        )
    lines.extend(
        [
            "## Classification Rules",
            "",
            "- If `canonical_checkpoint.yaml` exists, it is used as the primary pointer.",
            "- If no canonical pointer exists, `latest_checkpoint.yaml` is used as the fallback pointer.",
            "- `pointer_type` records whether `canonical` or `latest` was used.",
            "- `paper_grade_candidate` is reported only when the primary canonical pointer explicitly stores `classification: paper_grade_candidate`, the checkpoint exists, and the checkpoint is not `model_0.pt`.",
            "- `model_0.pt` is treated as `smoke_or_dev_only` unless explicit evidence says otherwise.",
            "- The A0 demo checkpoint `model_1999.pt` is `demo_grade_candidate`, not `paper_grade_candidate`.",
            "- The current residual pointer remains `smoke_or_dev_only` unless a future T09 freeze step promotes it.",
            "- A1-H healthy teacher pretraining is not enough for A1-F; the future A1-F teacher must be P2-trained and explicitly frozen.",
            "- Future A2 and A5 controlled evaluation requires P2-aligned canonical checkpoints.",
            "- A7 is `optional_deferred`, with no checkpoint expected until explicitly implemented and promoted under P2.",
            "- No checkpoint is promoted automatically by this audit.",
            "- Controlled evaluation must record exact checkpoint paths, not only pointer paths.",
            "",
            "## Guardrails",
            "",
            "- No training was run.",
            "- No Isaac Sim or play execution was launched.",
            "- No checkpoint pointer was modified.",
            "- No observed-result manifest was updated.",
            "- No paper-grade claim is made.",
            "- No new method family, health token, UQ, CBF, P2 runtime execution, or P3 expansion was added.",
            "",
        ]
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit T09 checkpoint readiness without runtime execution.")
    parser.add_argument("--output", default=OUTPUT_PATH)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    rows = audit_rows()
    markdown = render_markdown(rows)
    output_path = resolve_repo_path(args.output)
    if output_path is None:
        raise RuntimeError("invalid output path")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown, encoding="utf-8")
    print(markdown_table(rows))
    print(f"\nAudit written: {repo_relative(output_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
