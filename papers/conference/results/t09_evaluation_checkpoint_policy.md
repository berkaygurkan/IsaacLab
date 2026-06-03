# T09 Evaluation Checkpoint Freeze Policy

Scope: RLM1 stripped conference-stage; checkpoint pointer audit and evaluation policy; no training/evaluation executed.

## Purpose

This file freezes the interpretation of checkpoint pointers before fault-aware evaluation begins. It distinguishes stable checkpoint pointer files from actual model files, smoke checkpoints, and future paper-grade evaluation checkpoints. The current pointer paths remain useful for preflight and development, but they are not automatically canonical paper-grade evidence. Future fault-aware evaluation must record the exact resolved checkpoint paths used for each row, not only the pointer paths. This policy is documentation and pointer audit only; no training, evaluation, Isaac Sim launch, runtime fault injection, checkpoint write, or observed-result update was performed.

## Current Pointer Snapshot

| stage | pointer_path | resolved_checkpoint_path | checkpoint_exists | current_role | evaluation_policy |
| --- | --- | --- | --- | --- | --- |
| healthy_baseline | `checkpoints/rlm1_stripped/healthy_baseline/none/seed0/latest_checkpoint.yaml` | `logs/rsl_rl/healthy_baseline__rlm1_stripped__none/2026-05-11_23-53-43_healthy_baseline__rlm1_stripped__none__seed0/model_0.pt` | True | Baseline checkpoint pointer. | May be used as A0 zero-shot healthy PPO baseline only after the exact resolved checkpoint path is recorded in the evaluation manifest. |
| teacher | `checkpoints/rlm1_stripped/teacher/none/seed0/latest_checkpoint.yaml` | `logs/rsl_rl/teacher__rlm1_stripped__none/2026-05-13_00-03-08_teacher__rlm1_stripped__none__seed0/model_0.pt` | True | Privileged reference checkpoint pointer. | Reference-only; not deployment-facing and not a frozen-policy adaptation row. |
| student | `checkpoints/rlm1_stripped/student/none/seed0/latest_checkpoint.yaml` | `logs/rsl_rl/student__rlm1_stripped__none/2026-05-28_00-18-04_student__rlm1_stripped__none__seed0/model_0.pt` | True | Frozen student checkpoint pointer. | Candidate frozen student checkpoint for A2 and the residual wrapper base policy, subject to exact-path recording before evaluation. |
| residual | `checkpoints/rlm1_stripped/residual/none/seed0/latest_checkpoint.yaml` | `logs/rsl_rl/residual__rlm1_stripped__none/2026-06-01_15-46-39_residual__rlm1_stripped__none__seed0/model_0.pt` | True | Latest runtime smoke / development pointer. | Must not be treated as canonical paper-grade fault-evaluation checkpoint unless explicitly promoted by a future checkpoint freeze step. |

## Required Interpretation

- Healthy baseline pointer is a baseline checkpoint pointer.
- Teacher pointer is privileged reference only, not deployment-facing.
- Student pointer is the frozen student checkpoint pointer for A2 and for the residual wrapper base policy.
- Residual pointer currently may point to a smoke-produced residual checkpoint.
- The current residual pointer must be treated as latest runtime smoke / development pointer unless explicitly promoted later.
- Future paper-grade fault evaluation must use a deliberately selected canonical residual checkpoint, not accidentally whatever the latest pointer contains.

## Canonical Evaluation Checkpoint Policy

- Before T09-E1/P4 pilot execution, record a pointer snapshot.
- Before T09-E2 single-seed controlled evaluation, explicitly select canonical eval checkpoints.
- A0/A2/A5 fault-eval outputs must record exact checkpoint paths, not only checkpoint pointer paths.
- If a pointer changes after a smoke run, the manifest must preserve the old and new resolved checkpoint paths.
- Smoke checkpoints are infrastructure validation artifacts, not final performance checkpoints.
- A residual checkpoint can be promoted to canonical evaluation only by an explicit future T09-E checkpoint freeze step.

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
- T09-E1 P4 pilot.
- T09-E2 single-seed controlled eval.
- T09-E3 multi-seed expansion.
- Full A0-A6 observed results.
- Faulted scenario metrics.
- Paper-grade figures/tables.
