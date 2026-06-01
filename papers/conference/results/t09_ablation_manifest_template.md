# T09 Conference Ablation Manifest Template

Scope: Conference-stage RLM1 stripped; manifest scaffold only; no training or evaluation executed.

## Purpose

This T09-B manifest scaffold consumes the T09-A ablation matrix and stable checkpoint pointers.
It records row identity, checkpoint dependencies, command previews, readiness state, and pending metric fields.
No training is run in T09-B. No evaluation is run in T09-B. Metrics are not fabricated.

## Matrix Metadata

- stage: `T09-A`
- method: `rlm1_stripped`
- scope: `conference_stage`
- health_token: `False`
- uncertainty_channel: `False`
- safety_shield: `False`
- default_seed: `0`

## A0-A6 Manifest

| ablation_id | name | stage | method | fault | task | seed | first_mode | launcher_family | command_preview | checkpoint_pointer | checkpoint_path | checkpoint_exists | student_checkpoint_pointer | student_checkpoint_path | residual_scale | final_action_clip | reset_hidden_on_done | uses_teacher_policy | uses_true_fault_state | status | observed_metrics_status | metrics_placeholder | validation_sources | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A0 | healthy_ppo | healthy_baseline | rlm1_stripped | none | Isaac-Ant-v0 | 0 | dry_run | trainers/run_t05_healthy.sh | trainers/run_t05_healthy.sh | checkpoints/rlm1_stripped/healthy_baseline/none/seed0/latest_checkpoint.yaml | logs/rsl_rl/healthy_baseline__rlm1_stripped__none/2026-05-11_23-53-43_healthy_baseline__rlm1_stripped__none__seed0/model_0.pt | True |  |  | n/a | n/a | n/a | False | False | ready | pending | pending: no observed metrics recorded in T09-B | matrix; checkpoint pointer; code-free scaffold | Healthy PPO checkpoint reference for paper baseline. |
| A1 | privileged_teacher_reference | teacher | rlm1_stripped | none | Isaac-Ant-Teacher-v0 | 0 | dry_run | trainers/run_t06_teacher.sh | trainers/run_t06_teacher.sh | checkpoints/rlm1_stripped/teacher/none/seed0/latest_checkpoint.yaml | logs/rsl_rl/teacher__rlm1_stripped__none/2026-05-13_00-03-08_teacher__rlm1_stripped__none__seed0/model_0.pt | True |  |  | n/a | n/a | n/a | False | True | ready | pending | pending: no observed metrics recorded in T09-B | matrix; checkpoint pointer; code-free scaffold | Privileged teacher reference row only, not a deployment policy. |
| A2 | student_distilled_no_residual | student | rlm1_stripped | none | Isaac-Ant-Student-v0 | 0 | dry_run | trainers/run_t07_distill.sh | trainers/run_t07_distill.sh | checkpoints/rlm1_stripped/student/none/seed0/latest_checkpoint.yaml | logs/rsl_rl/student__rlm1_stripped__none/2026-05-28_00-18-04_student__rlm1_stripped__none__seed0/model_0.pt | True |  |  | n/a | n/a | n/a | False | False | ready | pending | pending: no observed metrics recorded in T09-B | matrix; checkpoint pointer; code-free scaffold | Distilled student checkpoint evaluated without residual correction. |
| A3 | student_residual_scale_0_0 | residual | rlm1_stripped | none | Isaac-Ant-Student-v0 | 0 | dry_run | trainers/run_t08_residual_train.sh | trainers/run_t08_residual_train.sh --residual_scale 0.0 | checkpoints/rlm1_stripped/residual/none/seed0/latest_checkpoint.yaml | logs/rsl_rl/residual__rlm1_stripped__none/2026-05-28_17-48-24_residual__rlm1_stripped__none__seed0/model_4.pt | True | checkpoints/rlm1_stripped/student/none/seed0/latest_checkpoint.yaml | logs/rsl_rl/student__rlm1_stripped__none/2026-05-28_00-18-04_student__rlm1_stripped__none__seed0/model_0.pt | 0.0 | None | False | False | False | ready | pending | pending: no observed metrics recorded in T09-B | matrix; checkpoint pointer; code-free scaffold | Residual wrapper path with zero residual contribution. |
| A4 | student_residual_scale_0_05 | residual | rlm1_stripped | none | Isaac-Ant-Student-v0 | 0 | dry_run | trainers/run_t08_residual_train.sh | trainers/run_t08_residual_train.sh --residual_scale 0.05 | checkpoints/rlm1_stripped/residual/none/seed0/latest_checkpoint.yaml | logs/rsl_rl/residual__rlm1_stripped__none/2026-05-28_17-48-24_residual__rlm1_stripped__none__seed0/model_4.pt | True | checkpoints/rlm1_stripped/student/none/seed0/latest_checkpoint.yaml | logs/rsl_rl/student__rlm1_stripped__none/2026-05-28_00-18-04_student__rlm1_stripped__none__seed0/model_0.pt | 0.05 | None | False | False | False | ready | pending | pending: no observed metrics recorded in T09-B | matrix; checkpoint pointer; code-free scaffold | Small residual scale sensitivity row. |
| A5 | student_residual_scale_0_1 | residual | rlm1_stripped | none | Isaac-Ant-Student-v0 | 0 | dry_run | trainers/run_t08_residual_train.sh | trainers/run_t08_residual_train.sh --residual_scale 0.1 | checkpoints/rlm1_stripped/residual/none/seed0/latest_checkpoint.yaml | logs/rsl_rl/residual__rlm1_stripped__none/2026-05-28_17-48-24_residual__rlm1_stripped__none__seed0/model_4.pt | True | checkpoints/rlm1_stripped/student/none/seed0/latest_checkpoint.yaml | logs/rsl_rl/student__rlm1_stripped__none/2026-05-28_00-18-04_student__rlm1_stripped__none__seed0/model_0.pt | 0.1 | None | False | False | False | ready | pending | pending: no observed metrics recorded in T09-B | matrix; checkpoint pointer; code-free scaffold | Default T08 residual baseline with validated residual scale. |
| A6 | student_residual_scale_0_2 | residual | rlm1_stripped | none | Isaac-Ant-Student-v0 | 0 | dry_run | trainers/run_t08_residual_train.sh | trainers/run_t08_residual_train.sh --residual_scale 0.2 | checkpoints/rlm1_stripped/residual/none/seed0/latest_checkpoint.yaml | logs/rsl_rl/residual__rlm1_stripped__none/2026-05-28_17-48-24_residual__rlm1_stripped__none__seed0/model_4.pt | True | checkpoints/rlm1_stripped/student/none/seed0/latest_checkpoint.yaml | logs/rsl_rl/student__rlm1_stripped__none/2026-05-28_00-18-04_student__rlm1_stripped__none__seed0/model_0.pt | 0.2 | None | False | False | False | ready | pending | pending: no observed metrics recorded in T09-B | matrix; checkpoint pointer; code-free scaffold | Larger residual scale sensitivity row. |

## Interpretation Notes

- Checkpoint fields are resolved from stable pointer YAML files.
- Metric fields are marked `pending` until T09-C/T09-D records observed results.
- T09-B does not execute experiments, training, or evaluation.
- Health token, uncertainty channel, safety shield / CBF, and P1/P2/P3 remain inactive.
- T09-D1 records the observed A5 one-row smoke separately in `papers/conference/results/t09_ablation_observed_smoke_manifest.md`.

## Deferred

- T09-C smoke runner.
- T09-D result manifest collection with observed metrics.
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
