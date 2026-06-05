# T09 Evaluation Checkpoint Freeze Policy

Scope: RLM1 stripped conference-stage; checkpoint pointer audit and evaluation policy; no training/evaluation executed.

## Purpose

This file freezes the interpretation of checkpoint pointers before fault-aware evaluation begins. It distinguishes stable checkpoint pointer files from actual model files, smoke checkpoints, and future paper-grade evaluation checkpoints. T09-R2d sets `P2_locked_joint` as the main conference fault scope; P4 torque degradation remains an advisor-demo / thesis-extension artifact only. The current pointer paths remain useful for preflight and development, but they are not automatically canonical paper-grade evidence. Future P2 fault-aware evaluation must record the exact resolved checkpoint paths used for each row, not only the pointer paths. This policy is documentation and pointer audit only; no training, evaluation, Isaac Sim launch, runtime fault injection, checkpoint write, or observed-result update was performed.

## Current Pointer Snapshot

| stage | pointer_path | resolved_checkpoint_path | checkpoint_exists | current_role | evaluation_policy |
| --- | --- | --- | --- | --- | --- |
| healthy_baseline | `checkpoints/rlm1_stripped/healthy_baseline/none/seed0/latest_checkpoint.yaml` | `logs/rsl_rl/healthy_baseline__rlm1_stripped__none/2026-05-11_23-53-43_healthy_baseline__rlm1_stripped__none__seed0/model_0.pt` | True | Baseline checkpoint pointer. | May be used as A0 zero-shot healthy PPO baseline only after the exact resolved checkpoint path is recorded in the evaluation manifest. |
| teacher | `checkpoints/rlm1_stripped/teacher/none/seed0/latest_checkpoint.yaml` | `logs/rsl_rl/teacher__rlm1_stripped__none/2026-05-13_00-03-08_teacher__rlm1_stripped__none__seed0/model_0.pt` | True | A1-H healthy teacher smoke/dev pointer. | Reference-only; not deployment-facing and not sufficient for future A1-F P2 teacher. |
| student | `checkpoints/rlm1_stripped/student/none/seed0/latest_checkpoint.yaml` | `logs/rsl_rl/student__rlm1_stripped__none/2026-05-28_00-18-04_student__rlm1_stripped__none__seed0/model_0.pt` | True | Frozen student checkpoint pointer. | Candidate frozen student checkpoint for A2 and the residual wrapper base policy, subject to exact-path recording before evaluation. |
| residual | `checkpoints/rlm1_stripped/residual/none/seed0/latest_checkpoint.yaml` | `logs/rsl_rl/residual__rlm1_stripped__none/2026-06-01_15-46-39_residual__rlm1_stripped__none__seed0/model_0.pt` | True | Latest runtime smoke / development pointer. | Must not be treated as canonical paper-grade fault-evaluation checkpoint unless explicitly promoted by a future checkpoint freeze step. |

## Required Interpretation

- Healthy baseline pointer is a baseline checkpoint pointer.
- Teacher pointer is A1-H smoke/dev context only; the future A1-F P2 privileged teacher must be separately trained and frozen.
- Student pointer is the frozen student checkpoint pointer for A2 and for the residual wrapper base policy.
- Residual pointer currently may point to a smoke-produced residual checkpoint.
- The current residual pointer must be treated as latest runtime smoke / development pointer unless explicitly promoted later.
- Future paper-grade fault evaluation must use a deliberately selected canonical residual checkpoint, not accidentally whatever the latest pointer contains.

## Canonical Evaluation Checkpoint Policy

- Before any P2 pilot or controlled evaluation, record a pointer snapshot.
- Before P2 single-seed controlled evaluation, explicitly select canonical eval checkpoints.
- A0/A2/A5 P2 fault-eval outputs must record exact checkpoint paths, not only checkpoint pointer paths.
- If a pointer changes after a smoke run, the manifest must preserve the old and new resolved checkpoint paths.
- Smoke checkpoints are infrastructure validation artifacts, not final performance checkpoints.
- A residual checkpoint can be promoted to canonical evaluation only by an explicit future T09-E checkpoint freeze step.
- A0 canonical training/freezing is scaffolded in `papers/conference/results/t09_a0_canonical_training_plan.md` and `papers/conference/results/t09_canonical_checkpoint_freeze_policy.md`.
- A1-H healthy teacher pretraining is not enough for the final teacher row. Future A1-F must be trained under P2 curriculum, remains privileged/reference-only, and should be frozen before A2 student distillation at `checkpoints/rlm1_stripped/teacher_p2/none/seed0/canonical_checkpoint.yaml`.
- Future A2 student distillation should use the frozen A1-F P2 teacher and must not require teacher observations or `true_fault_state` at runtime.
- Future A5 residual training/evaluation should follow the P2 scope. A5 remains the main residual method row.
- Optional A7 teacher-distilled residual training is deferred. A7 requires A0 canonical and the future A1-F P2 teacher canonical before any residual distillation. If trained later, A7 must use its own pointer at `checkpoints/rlm1_stripped/teacher_distilled_residual/none/seed0/canonical_checkpoint.yaml` and must not reuse the A5 residual pointer.
- Existing P4 demo artifacts and runtime smoke results must not be used as paper-grade conference evidence.

## A7 Optional Dependency Policy

- A7 is optional/deferred and not paper-main unless explicitly promoted later.
- A7 base policy dependency: `checkpoints/rlm1_stripped/healthy_baseline/none/seed0/canonical_checkpoint.yaml`.
- A7 teacher dependency: future A1-F P2 fault-aware teacher canonical pointer, `checkpoints/rlm1_stripped/teacher_p2/none/seed0/canonical_checkpoint.yaml`.
- A7 residual pointer, if trained later: `checkpoints/rlm1_stripped/teacher_distilled_residual/none/seed0/canonical_checkpoint.yaml`.
- A7 must not reuse `checkpoints/rlm1_stripped/residual/none/seed0/latest_checkpoint.yaml` or an A5 residual canonical pointer.
- A7 runtime must not require privileged teacher observations or `true_fault_state` if it is considered deployment-facing.

## Residual-Scale Rule

- A3/A4/A5/A6 reuse the same selected residual checkpoint per seed.
- `residual_scale` changes at evaluation time only.
- Do not train separate residual policies per scale in the current conference-stage plan.

## Guardrails

- No full sweep is implied by this policy; no full sweep is claimed.
- No fault-tolerance result is claimed.
- No health token, UQ, or CBF is active.
- No P1/P3/Px expansion.
- Do not interpret A5 smoke metrics as paper-grade performance.
- Do not overwrite checkpoint pointers during evaluation without snapshotting.

## Deferred Items

- Canonical checkpoint promotion step.
- T09-R0 checkpoint readiness audit for deciding which existing checkpoints are demo-grade, smoke/dev, missing, or candidates for later freeze.
- P2 runtime evaluator implementation and pilot.
- P2 single-seed controlled eval.
- T09-E3 multi-seed expansion.
- Full A0-A6 observed results.
- Faulted scenario metrics.
- Paper-grade figures/tables.
