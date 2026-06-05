# T09 Checkpoint Readiness Audit

Scope: RLM1 stripped conference-stage checkpoint readiness audit; no training, evaluation, play, or Isaac Sim executed.

Generated: 2026-06-05 23:44:21

## Method State

- Health token OFF.
- UQ inactive.
- CBF inactive.
- `P2_locked_joint` is the future conference fault scope.
- No P2 runtime execution or P3 expansion.
- P4 advisor-demo artifacts remain separate from controlled-evaluation artifacts.

## Readiness Table

| ablation_id | row_name | pointer_type | canonical_pointer_path | latest_pointer_path | pointer_path | resolved_checkpoint_path | checkpoint_exists | checkpoint_filename | log_dir | candidate_checkpoint_path | classification | controlled_eval_use | recommended_next_action |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A0 | healthy PPO baseline | canonical | `checkpoints/rlm1_stripped/healthy_baseline/none/seed0/canonical_checkpoint.yaml` | `checkpoints/rlm1_stripped/healthy_baseline/none/seed0/latest_checkpoint.yaml` | `checkpoints/rlm1_stripped/healthy_baseline/none/seed0/canonical_checkpoint.yaml` | `logs/rsl_rl/healthy_baseline__rlm1_stripped__canonical/2026-06-05_21-42-13_a0_canonical__seed0/model_1999.pt` | True | model_1999.pt | `logs/rsl_rl/healthy_baseline__rlm1_stripped__canonical/2026-06-05_21-42-13_a0_canonical__seed0` | `logs/rsl_rl/healthy_baseline__rlm1_stripped__demo/2026-06-03_04-08-57_healthy_demo__seed0/model_1999.pt` | paper_grade_candidate | True | A0 canonical checkpoint is ready for future F0/P2 controlled evaluation; keep evaluation blocked until A1-F/A2/A5 P2 readiness is addressed. |
| A1 | A1-F P2 privileged teacher | latest | `checkpoints/rlm1_stripped/teacher_p2/none/seed0/canonical_checkpoint.yaml` | `checkpoints/rlm1_stripped/teacher_p2/none/seed0/latest_checkpoint.yaml` | `checkpoints/rlm1_stripped/teacher_p2/none/seed0/latest_checkpoint.yaml` | NA | False | NA | NA | NA | missing | False | Train/select/freeze A1-F under P2, then use checkpoints/rlm1_stripped/teacher_p2/none/seed0/canonical_checkpoint.yaml before A2 distillation. |
| A2 | student no residual | latest | `checkpoints/rlm1_stripped/student/none/seed0/canonical_checkpoint.yaml` | `checkpoints/rlm1_stripped/student/none/seed0/latest_checkpoint.yaml` | `checkpoints/rlm1_stripped/student/none/seed0/latest_checkpoint.yaml` | `logs/rsl_rl/student__rlm1_stripped__none/2026-05-28_00-18-04_student__rlm1_stripped__none__seed0/model_0.pt` | True | model_0.pt | `logs/rsl_rl/student__rlm1_stripped__none/2026-05-28_00-18-04_student__rlm1_stripped__none__seed0` | NA | smoke_or_dev_only | False | Distill a non-smoke A2 student from the P2-trained A1-F teacher, then freeze the exact path. |
| A5 | student + residual baseline | latest | `checkpoints/rlm1_stripped/residual/none/seed0/canonical_checkpoint.yaml` | `checkpoints/rlm1_stripped/residual/none/seed0/latest_checkpoint.yaml` | `checkpoints/rlm1_stripped/residual/none/seed0/latest_checkpoint.yaml` | `logs/rsl_rl/residual__rlm1_stripped__none/2026-06-01_15-46-39_residual__rlm1_stripped__none__seed0/model_0.pt` | True | model_0.pt | `logs/rsl_rl/residual__rlm1_stripped__none/2026-06-01_15-46-39_residual__rlm1_stripped__none__seed0` | NA | smoke_or_dev_only | False | Train/select canonical P2 student and residual checkpoints; do not use the current residual pointer as paper evidence. |
| A7 | healthy PPO + teacher-distilled residual | latest | `checkpoints/rlm1_stripped/teacher_distilled_residual/none/seed0/canonical_checkpoint.yaml` | `checkpoints/rlm1_stripped/teacher_distilled_residual/none/seed0/latest_checkpoint.yaml` | `checkpoints/rlm1_stripped/teacher_distilled_residual/none/seed0/latest_checkpoint.yaml` | NA | False | NA | NA | NA | optional_deferred | False | Keep A7 deferred; do not train or evaluate unless explicitly promoted after A0 and P2 A1-F canonical dependencies exist. |

## Detailed Notes

### A0 - healthy PPO baseline

- Role: deployment-facing healthy baseline; trained F0 and evaluated F0/P2
- Pointer type used: `canonical`
- Canonical pointer path: `checkpoints/rlm1_stripped/healthy_baseline/none/seed0/canonical_checkpoint.yaml`
- Latest pointer path: `checkpoints/rlm1_stripped/healthy_baseline/none/seed0/latest_checkpoint.yaml`
- Classification: `paper_grade_candidate`
- Rationale: Canonical pointer explicitly stores paper_grade_candidate and resolves to a non-model_0 checkpoint.
- Primary pointer resolved path: `logs/rsl_rl/healthy_baseline__rlm1_stripped__canonical/2026-06-05_21-42-13_a0_canonical__seed0/model_1999.pt`
- Pointer-stored classification: `paper_grade_candidate`
- Candidate checkpoint path: `logs/rsl_rl/healthy_baseline__rlm1_stripped__demo/2026-06-03_04-08-57_healthy_demo__seed0/model_1999.pt`
- Dependency checkpoint path: `NA`
- Base dependency pointer path: `NA`
- Teacher dependency pointer path: `NA`
- Legacy pointer path: `NA`
- Recommended next action: A0 canonical checkpoint is ready for future F0/P2 controlled evaluation; keep evaluation blocked until A1-F/A2/A5 P2 readiness is addressed.

### A1 - A1-F P2 privileged teacher

- Role: future P2 privileged teacher required before A2 distillation; not deployment-facing
- Pointer type used: `latest`
- Canonical pointer path: `checkpoints/rlm1_stripped/teacher_p2/none/seed0/canonical_checkpoint.yaml`
- Latest pointer path: `checkpoints/rlm1_stripped/teacher_p2/none/seed0/latest_checkpoint.yaml`
- Classification: `missing`
- Rationale: Checkpoint file is missing; cannot launch controlled evaluation.
- Primary pointer resolved path: `NA`
- Pointer-stored classification: `None`
- Candidate checkpoint path: `NA`
- Dependency checkpoint path: `NA`
- Base dependency pointer path: `NA`
- Teacher dependency pointer path: `NA`
- Legacy pointer path: `checkpoints/rlm1_stripped/teacher/none/seed0/latest_checkpoint.yaml`
- Recommended next action: Train/select/freeze A1-F under P2, then use checkpoints/rlm1_stripped/teacher_p2/none/seed0/canonical_checkpoint.yaml before A2 distillation.

### A2 - student no residual

- Role: future P2-distilled student candidate; no online learning and no privileged runtime inputs
- Pointer type used: `latest`
- Canonical pointer path: `checkpoints/rlm1_stripped/student/none/seed0/canonical_checkpoint.yaml`
- Latest pointer path: `checkpoints/rlm1_stripped/student/none/seed0/latest_checkpoint.yaml`
- Classification: `smoke_or_dev_only`
- Rationale: Resolved checkpoint is model_0.pt; treat as smoke/development unless evidence proves otherwise.
- Primary pointer resolved path: `logs/rsl_rl/student__rlm1_stripped__none/2026-05-28_00-18-04_student__rlm1_stripped__none__seed0/model_0.pt`
- Pointer-stored classification: `None`
- Candidate checkpoint path: `NA`
- Dependency checkpoint path: `NA`
- Base dependency pointer path: `NA`
- Teacher dependency pointer path: `NA`
- Legacy pointer path: `NA`
- Recommended next action: Distill a non-smoke A2 student from the P2-trained A1-F teacher, then freeze the exact path.

### A5 - student + residual baseline

- Role: future main P2 residual-policy candidate; no online learning and no privileged runtime inputs
- Pointer type used: `latest`
- Canonical pointer path: `checkpoints/rlm1_stripped/residual/none/seed0/canonical_checkpoint.yaml`
- Latest pointer path: `checkpoints/rlm1_stripped/residual/none/seed0/latest_checkpoint.yaml`
- Classification: `smoke_or_dev_only`
- Rationale: Residual pointer remains smoke/development unless explicitly promoted by a future freeze step.
- Primary pointer resolved path: `logs/rsl_rl/residual__rlm1_stripped__none/2026-06-01_15-46-39_residual__rlm1_stripped__none__seed0/model_0.pt`
- Pointer-stored classification: `None`
- Candidate checkpoint path: `NA`
- Dependency checkpoint path: `logs/rsl_rl/student__rlm1_stripped__none/2026-05-28_00-18-04_student__rlm1_stripped__none__seed0/model_0.pt`
- Base dependency pointer path: `NA`
- Teacher dependency pointer path: `NA`
- Legacy pointer path: `NA`
- Recommended next action: Train/select canonical P2 student and residual checkpoints; do not use the current residual pointer as paper evidence.

### A7 - healthy PPO + teacher-distilled residual

- Role: optional deferred P2 teacher-distilled residual ablation; not paper-main
- Pointer type used: `latest`
- Canonical pointer path: `checkpoints/rlm1_stripped/teacher_distilled_residual/none/seed0/canonical_checkpoint.yaml`
- Latest pointer path: `checkpoints/rlm1_stripped/teacher_distilled_residual/none/seed0/latest_checkpoint.yaml`
- Classification: `optional_deferred`
- Rationale: Optional A7 registration only; no checkpoint is expected until explicitly implemented and promoted.
- Primary pointer resolved path: `NA`
- Pointer-stored classification: `None`
- Candidate checkpoint path: `NA`
- Dependency checkpoint path: `NA`
- Base dependency pointer path: `checkpoints/rlm1_stripped/healthy_baseline/none/seed0/canonical_checkpoint.yaml`
- Teacher dependency pointer path: `checkpoints/rlm1_stripped/teacher_p2/none/seed0/canonical_checkpoint.yaml`
- Legacy pointer path: `NA`
- Recommended next action: Keep A7 deferred; do not train or evaluate unless explicitly promoted after A0 and P2 A1-F canonical dependencies exist.

## Classification Rules

- If `canonical_checkpoint.yaml` exists, it is used as the primary pointer.
- If no canonical pointer exists, `latest_checkpoint.yaml` is used as the fallback pointer.
- `pointer_type` records whether `canonical` or `latest` was used.
- `paper_grade_candidate` is reported only when the primary canonical pointer explicitly stores `classification: paper_grade_candidate`, the checkpoint exists, and the checkpoint is not `model_0.pt`.
- `model_0.pt` is treated as `smoke_or_dev_only` unless explicit evidence says otherwise.
- The A0 demo checkpoint `model_1999.pt` is `demo_grade_candidate`, not `paper_grade_candidate`.
- The current residual pointer remains `smoke_or_dev_only` unless a future T09 freeze step promotes it.
- A1-H healthy teacher pretraining is not enough for A1-F; the future A1-F teacher must be P2-trained and explicitly frozen.
- Future A2 and A5 controlled evaluation requires P2-aligned canonical checkpoints.
- A7 is `optional_deferred`, with no checkpoint expected until explicitly implemented and promoted under P2.
- No checkpoint is promoted automatically by this audit.
- Controlled evaluation must record exact checkpoint paths, not only pointer paths.

## Guardrails

- No training was run.
- No Isaac Sim or play execution was launched.
- No checkpoint pointer was modified.
- No observed-result manifest was updated.
- No paper-grade claim is made.
- No new method family, health token, UQ, CBF, P2 runtime execution, or P3 expansion was added.
