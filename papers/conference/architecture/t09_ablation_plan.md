# T09-A Conference Ablation Matrix Plan

Scope: Conference-stage RLM1 stripped; health token OFF; uncertainty channel inactive; safety shield / CBF inactive.

## Purpose

T09-A defines the static paper-facing ablation matrix for the current T05 through T08 pipeline. It is a dry-run artifact only: no training is run, no evaluation is run, and no checkpoint is written. The matrix records the initial A0-A6 rows that compare the healthy PPO baseline, privileged teacher reference, distilled student, and residual scale sensitivity settings. The companion validator checks schema, launcher families, checkpoint pointer files, and resolved checkpoint paths without importing Isaac Lab or starting Isaac Sim. This keeps the conference ablation plan reviewable before T09-B/T09-C add result collection or smoke execution.

## Matrix Rows

| ID | Name | Stage | Task | Checkpoint Pointer | Launcher Preview | Residual Scale | Paper Table Role |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A0 | `healthy_ppo` | `healthy_baseline` | `Isaac-Ant-v0` | `checkpoints/rlm1_stripped/healthy_baseline/none/seed0/latest_checkpoint.yaml` | `trainers/run_t05_healthy.sh` | n/a | Healthy PPO baseline |
| A1 | `privileged_teacher_reference` | `teacher` | `Isaac-Ant-Teacher-v0` | `checkpoints/rlm1_stripped/teacher/none/seed0/latest_checkpoint.yaml` | `trainers/run_t06_teacher.sh` | n/a | Privileged teacher reference |
| A2 | `student_distilled_no_residual` | `student` | `Isaac-Ant-Student-v0` | `checkpoints/rlm1_stripped/student/none/seed0/latest_checkpoint.yaml` | `trainers/run_t07_distill.sh` | n/a | Student without residual |
| A3 | `student_residual_scale_0_0` | `residual` | `Isaac-Ant-Student-v0` | residual pointer plus student pointer | `trainers/run_t08_residual_train.sh --residual_scale 0.0` | `0.0` | Residual runtime control |
| A4 | `student_residual_scale_0_05` | `residual` | `Isaac-Ant-Student-v0` | residual pointer plus student pointer | `trainers/run_t08_residual_train.sh --residual_scale 0.05` | `0.05` | Residual scale sensitivity |
| A5 | `student_residual_scale_0_1` | `residual` | `Isaac-Ant-Student-v0` | residual pointer plus student pointer | `trainers/run_t08_residual_train.sh --residual_scale 0.1` | `0.1` | Default residual baseline |
| A6 | `student_residual_scale_0_2` | `residual` | `Isaac-Ant-Student-v0` | residual pointer plus student pointer | `trainers/run_t08_residual_train.sh --residual_scale 0.2` | `0.2` | Residual scale sensitivity |

## Dry-Run Behavior

The validator is `evaluators/run_t09_ablation_matrix.py`. Its default behavior is dry-run validation only, even when `--dry_run` is omitted. It parses `configs/ablation/t09_conference_matrix.yaml` using a small stdlib-only parser for this repo-owned flat YAML shape. For each row, it validates required fields, confirms the row set is exactly A0-A6, resolves checkpoint pointer files relative to the repository root, and checks a declared `checkpoint_path` when present inside the pointer file. Missing pointers or missing checkpoints are reported in row status output, but they do not fail the command by default; invalid schema or unreadable matrix syntax fails the command.

Dry-run command:

```bash
python evaluators/run_t09_ablation_matrix.py \
  --matrix configs/ablation/t09_conference_matrix.yaml \
  --dry_run
```

## T09-B Manifest Scaffold

T09-B adds `evaluators/collect_t09_results.py` and `papers/conference/results/t09_ablation_manifest_template.md`. The collector consumes the same A0-A6 matrix and stable checkpoint pointers, then writes a Markdown manifest with row identity, command previews, readiness status, checkpoint paths, and metric placeholders marked `pending`. It does not run training, does not run evaluation, does not call Isaac Sim, and does not fabricate metrics.

## T09.5 Infrastructure Snapshot

T09.5 freezes the ablation infrastructure state in `papers/conference/architecture/t09_ablation_infrastructure_snapshot.md`. It records that A5 has one observed smoke result while A0/A1/A2/A3/A4/A6 remain pending and no full sweep or evaluation runner has been run.

## Checkpoint Pointer Dependencies

- A0 depends on `checkpoints/rlm1_stripped/healthy_baseline/none/seed0/latest_checkpoint.yaml`.
- A1 depends on `checkpoints/rlm1_stripped/teacher/none/seed0/latest_checkpoint.yaml`.
- A2 depends on `checkpoints/rlm1_stripped/student/none/seed0/latest_checkpoint.yaml`.
- A3-A6 depend on `checkpoints/rlm1_stripped/residual/none/seed0/latest_checkpoint.yaml` and `checkpoints/rlm1_stripped/student/none/seed0/latest_checkpoint.yaml`.

## Guardrails

- T09-A does not run training.
- T09-A does not run evaluation.
- T09-A does not call Isaac Sim.
- T09-A does not write checkpoints.
- T09-A does not modify RSL-RL, task registration, or T05/T06/T07/T08 behavior.
- A1 is the only row marked as using `true_fault_state`, and it is explicitly a privileged teacher reference rather than a deployment policy.
- A0/A2/A3/A4/A5/A6 do not use `teacher_policy` or `true_fault_state`.
- Residual rows use CLI `residual_scale` overrides; the T08 residual config is not edited per row.

## Deferred

- T09-B result collector.
- T09-C smoke runner.
- T09-D result manifest collection.
- T09.5 ablation documentation snapshot.
- Full training sweeps.
- Full evaluation runner.
- Fault curriculum.
- Health token.
- Uncertainty channel.
- Safety shield / CBF.
- P1/P2/P3 expansion.
- New methods beyond A0-A6.
- Privileged critic.
- Recurrent residual PPO.
