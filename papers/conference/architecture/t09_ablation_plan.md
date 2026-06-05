# T09-A Conference Ablation Matrix Plan

Scope: Conference-stage RLM1 stripped; P2 single joint lock main fault; health token OFF; uncertainty channel inactive; safety shield / CBF inactive.

## Purpose

T09-A defines the static paper-facing ablation matrix for the current T05 through T08 pipeline. It is a dry-run artifact only: no training is run, no evaluation is run, and no checkpoint is written. T09-R2d pivots the future conference fault scope to `P2_locked_joint`, a single locked-joint condition with default target joint `front_left_foot` and fault onset step `50`. P4 torque degradation is no longer the main conference fault and remains only as an archived advisor-demo / thesis-extension artifact. The matrix records the main A0-A6 rows that compare the healthy PPO baseline, future P2 privileged teacher reference, P2-distilled student, and residual scale sensitivity settings. T09-R2c adds A7 as an optional/deferred teacher-distilled residual ablation if time allows; A7 is not part of the main method and must not replace A5 without an explicit later decision. The companion validator checks schema, launcher families, checkpoint pointer files, and resolved checkpoint paths without importing Isaac Lab or starting Isaac Sim. This keeps the conference ablation plan reviewable before result collection or smoke execution.

## Matrix Rows

| ID | Name | Stage | Task | Checkpoint Pointer | Launcher Preview | Residual Scale | Paper Table Role |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A0 | `healthy_ppo` | `healthy_baseline` | `Isaac-Ant-v0` | `checkpoints/rlm1_stripped/healthy_baseline/none/seed0/latest_checkpoint.yaml` | `trainers/run_t05_healthy.sh` | n/a | Healthy PPO baseline; train F0, evaluate F0/P2 |
| A1 | `p2_fault_aware_privileged_teacher_reference` | `teacher_p2` | `Isaac-Ant-Teacher-v0` | `checkpoints/rlm1_stripped/teacher_p2/none/seed0/canonical_checkpoint.yaml` | `trainers/run_t06_teacher.sh` | n/a | Future P2 privileged teacher reference |
| A2 | `student_distilled_no_residual` | `student` | `Isaac-Ant-Student-v0` | `checkpoints/rlm1_stripped/student/none/seed0/latest_checkpoint.yaml` | `trainers/run_t07_distill.sh` | n/a | P2-distilled student without residual |
| A3 | `student_residual_scale_0_0` | `residual` | `Isaac-Ant-Student-v0` | residual pointer plus student pointer | `trainers/run_t08_residual_train.sh --residual_scale 0.0` | `0.0` | Residual runtime control |
| A4 | `student_residual_scale_0_05` | `residual` | `Isaac-Ant-Student-v0` | residual pointer plus student pointer | `trainers/run_t08_residual_train.sh --residual_scale 0.05` | `0.05` | Residual scale sensitivity |
| A5 | `student_residual_scale_0_1` | `residual` | `Isaac-Ant-Student-v0` | residual pointer plus student pointer | `trainers/run_t08_residual_train.sh --residual_scale 0.1` | `0.1` | Default residual baseline |
| A6 | `student_residual_scale_0_2` | `residual` | `Isaac-Ant-Student-v0` | residual pointer plus student pointer | `trainers/run_t08_residual_train.sh --residual_scale 0.2` | `0.2` | Residual scale sensitivity |
| A7 | `healthy_ppo_teacher_distilled_residual` | `optional_teacher_distilled_residual` | `Isaac-Ant-v0` or future explicit residual-distill task | A0 canonical plus future A1-F canonical teacher plus own residual pointer | deferred | `TBD`, default candidate `0.1` | Optional P2 residual-distillation ablation |

## A7 Optional Variant

A7 is an optional ablation variant, not a replacement for A5. Its base action
comes from frozen A0 healthy PPO. A future A1-F P2 fault-aware privileged teacher
provides target actions during distillation data generation. The P2 residual
target is `teacher_action_under_P2_minus_A0_action_under_P2`, and the final
runtime action is healthy PPO action plus the learned residual correction.

A7 may be considered deployment-facing only if the final runtime does not
require privileged teacher observations or `true_fault_state`. The teacher is
used during training only. A7 is currently `optional_deferred`, has no expected
checkpoint, and must not be treated as paper-main unless explicitly promoted.

## Teacher / Student / Residual Distinction

- A1-H is a healthy teacher pretrain/reference and is not the final P2 fault-aware teacher.
- A1-F is the future privileged teacher trained under a P2 single-joint-lock curriculum.
- A2 is the student distilled from A1-F under P2, with no residual correction.
- A5 is the main student plus residual correction row in the P2 conference method.
- A7 is the optional healthy PPO plus teacher-distilled residual ablation under P2.

## Dry-Run Behavior

The validator is `evaluators/run_t09_ablation_matrix.py`. Its default behavior is dry-run validation only, even when `--dry_run` is omitted. It parses `configs/ablation/t09_conference_matrix.yaml` using a small stdlib-only parser for this repo-owned flat YAML shape. For the main rows, it validates required fields, confirms the required row set A0-A6, resolves checkpoint pointer files relative to the repository root, and checks a declared `checkpoint_path` when present inside the pointer file. A7 is registered as `optional_deferred` and remains outside mandatory validator expectations until A7 tooling is explicitly added. Missing pointers or missing checkpoints are reported in row status output, but they do not fail the command by default; invalid schema or unreadable matrix syntax fails the command.

Dry-run command:

```bash
python evaluators/run_t09_ablation_matrix.py \
  --matrix configs/ablation/t09_conference_matrix.yaml \
  --dry_run
```

## P2 Controlled-Evaluation Matrix

T09-R2d adds `configs/ablation/t09_p2_controlled_eval_matrix.yaml` as the narrow future controlled-evaluation scope. It contains `A0_F0`, `A0_P2`, `A2_F0`, `A2_P2`, `A5_F0`, and `A5_P2`, plus optional/deferred `A7_F0` and `A7_P2`. A0 canonical is ready, but controlled evaluation remains blocked until A1-F, A2, and A5 are canonical/frozen and a guarded P2 runtime evaluator exists.

## T09-B Manifest Scaffold

T09-B adds `evaluators/collect_t09_results.py` and `papers/conference/results/t09_ablation_manifest_template.md`. The collector consumes the main A0-A6 rows and stable checkpoint pointers, then writes a Markdown manifest with row identity, command previews, readiness status, checkpoint paths, and metric placeholders marked `pending`. A7 remains optional/deferred and is excluded from required manifests until it is explicitly promoted. It does not run training, does not run evaluation, does not call Isaac Sim, and does not fabricate metrics.

## T09.5 Infrastructure Snapshot

T09.5 freezes the ablation infrastructure state in `papers/conference/architecture/t09_ablation_infrastructure_snapshot.md`. It records that A5 has one observed smoke result while A0/A1/A2/A3/A4/A6 remain pending and no full sweep or evaluation runner has been run.

## Checkpoint Pointer Dependencies

- A0 depends on `checkpoints/rlm1_stripped/healthy_baseline/none/seed0/latest_checkpoint.yaml`.
- A1-F depends on future canonical `checkpoints/rlm1_stripped/teacher_p2/none/seed0/canonical_checkpoint.yaml`; the existing `checkpoints/rlm1_stripped/teacher/none/seed0/latest_checkpoint.yaml` is A1-H smoke/dev context only.
- A2 depends on a future P2-distilled student canonical checkpoint; current `checkpoints/rlm1_stripped/student/none/seed0/latest_checkpoint.yaml` is smoke/dev context only.
- A3-A6 depend on future P2-aligned residual and student canonical checkpoints; current residual/student latest pointers are smoke/dev context only.
- A7 depends on A0 canonical, future A1-F P2 fault-aware teacher canonical, and its own future teacher-distilled residual canonical pointer.

## Guardrails

- T09-A does not run training.
- T09-A does not run evaluation.
- T09-A does not call Isaac Sim.
- T09-A does not write checkpoints.
- T09-A does not modify RSL-RL, task registration, or T05/T06/T07/T08 behavior.
- A1 is the only row marked as using `true_fault_state`, and it is explicitly a privileged P2 teacher reference rather than a deployment policy.
- A0/A2/A3/A4/A5/A6 do not use `teacher_policy` or `true_fault_state`.
- Residual rows use CLI `residual_scale` overrides; the T08 residual config is not edited per row.
- P4 torque degradation artifacts are advisor-demo / thesis-extension artifacts only, not conference main results.

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
- P1/P3 expansion.
- New methods beyond A0-A6, except the explicitly deferred optional A7 registration.
- Privileged critic.
- Recurrent residual PPO.
