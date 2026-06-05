#!/usr/bin/env python3
"""Freeze a selected T09 checkpoint into an explicit pointer and manifest.

Stdlib-only. This script does not train, evaluate, import Isaac Sim, or modify
latest_checkpoint.yaml. It writes only the requested output pointer and freeze
manifest when called with all required arguments.
"""

from __future__ import annotations

import argparse
import re
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
ALLOWED_CLASSIFICATIONS = ("paper_grade_candidate", "demo_grade_candidate", "smoke_or_dev_only")


def resolve_repo_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def repo_relative(path_value: str | Path) -> str:
    resolved = resolve_repo_path(path_value).resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(resolved)


def model_iteration(checkpoint_path: Path) -> int | None:
    match = re.search(r"model_(\d+)\.pt$", checkpoint_path.name)
    return int(match.group(1)) if match else None


def is_demo_or_advisor_path(path: Path) -> bool:
    path_text = repo_relative(path).lower()
    return "demo" in path_text or "advisor" in path_text


def validate_freeze_request(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    checkpoint_path = resolve_repo_path(args.checkpoint_path)
    output_pointer = resolve_repo_path(args.output_pointer)
    manifest_path = resolve_repo_path(args.manifest_path)

    if not checkpoint_path.is_file():
        raise ValueError(f"checkpoint file missing: {repo_relative(checkpoint_path)}")
    if output_pointer.name == "latest_checkpoint.yaml":
        raise ValueError("Refusing to overwrite latest_checkpoint.yaml; use canonical_checkpoint.yaml or another explicit pointer.")
    if args.classification == "paper_grade_candidate":
        if model_iteration(checkpoint_path) == 0:
            raise ValueError("Refusing paper_grade_candidate for model_0.pt.")
        if is_demo_or_advisor_path(checkpoint_path):
            raise ValueError("Refusing paper_grade_candidate for demo/advisor checkpoint path.")
    return checkpoint_path, output_pointer, manifest_path


def pointer_text(args: argparse.Namespace, checkpoint_path: Path) -> str:
    log_dir = checkpoint_path.parent
    return "\n".join(
        [
            f"stage: {args.stage}",
            f"method: {args.method}",
            f"fault: {args.fault}",
            f"seed: {args.seed}",
            f"task: {args.task}",
            f"experiment_name: {args.experiment_name}",
            f"run_name: {args.run_name}",
            f"classification: {args.classification}",
            f"log_dir: {repo_relative(log_dir)}",
            f"checkpoint_path: {repo_relative(checkpoint_path)}",
            f"frozen_at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "note: explicit T09 checkpoint freeze; latest_checkpoint.yaml was not modified",
            "",
        ]
    )


def manifest_text(args: argparse.Namespace, checkpoint_path: Path, output_pointer: Path) -> str:
    return "\n".join(
        [
            "# T09 Checkpoint Freeze Manifest",
            "",
            "Scope: explicit checkpoint freeze metadata; no training, evaluation, play, or Isaac Sim executed.",
            "",
            f"- stage: `{args.stage}`",
            f"- method: `{args.method}`",
            f"- fault: `{args.fault}`",
            f"- seed: `{args.seed}`",
            f"- task: `{args.task}`",
            f"- experiment_name: `{args.experiment_name}`",
            f"- run_name: `{args.run_name}`",
            f"- classification: `{args.classification}`",
            f"- checkpoint_path: `{repo_relative(checkpoint_path)}`",
            f"- checkpoint_filename: `{checkpoint_path.name}`",
            f"- model_iteration: `{model_iteration(checkpoint_path)}`",
            f"- output_pointer: `{repo_relative(output_pointer)}`",
            "",
            "## Policy",
            "",
            "- This freeze helper never infers paper-grade status automatically.",
            "- `paper_grade_candidate` requires an explicit classification argument.",
            "- `paper_grade_candidate` is refused for `model_0.pt` and demo/advisor paths.",
            "- `latest_checkpoint.yaml` is never overwritten by this helper.",
            "- Controlled evaluation must still record exact resolved checkpoint paths.",
            "",
            "## Guardrails",
            "",
            "- No training was run.",
            "- No Isaac Sim or play execution was launched.",
            "- No observed-result manifest was updated.",
            "- No health token, UQ, CBF, P2, or P3 expansion was added.",
            "",
        ]
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Freeze an explicit T09 checkpoint pointer and manifest.")
    parser.add_argument("--stage", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--fault", required=True)
    parser.add_argument("--seed", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--experiment_name", required=True)
    parser.add_argument("--run_name", required=True)
    parser.add_argument("--classification", required=True, choices=ALLOWED_CLASSIFICATIONS)
    parser.add_argument("--output_pointer", required=True)
    parser.add_argument("--manifest_path", required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    checkpoint_path, output_pointer, manifest_path = validate_freeze_request(args)
    output_pointer.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    output_pointer.write_text(pointer_text(args, checkpoint_path), encoding="utf-8")
    manifest_path.write_text(manifest_text(args, checkpoint_path, output_pointer), encoding="utf-8")
    print("[T09-R1] checkpoint freeze written")
    print(f"  output_pointer: {repo_relative(output_pointer)}")
    print(f"  manifest_path: {repo_relative(manifest_path)}")
    print(f"  checkpoint_path: {repo_relative(checkpoint_path)}")
    print("  latest_checkpoint_yaml_modified: False")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
