# T09 Observed Ablation Smoke Manifest

Scope: Conference-stage RLM1 stripped; observed smoke manifest; A5 only; no full sweep.

## Purpose

This file records only the observed T09-C A5 smoke supplied by the user. A0, A1, A2, A3, A4, and A6 remain pending. No new training or evaluation was run during T09-D1.

## A5 Observed Result

| ablation_id | name | stage | method | fault | task | seed | residual_scale | command | run_dir | checkpoint_pointer | observed_status | observed_mode | learning_iteration | mean_value_loss | mean_surrogate_loss | mean_entropy_loss | mean_reward | mean_episode_length | residual_mean_abs_delta | residual_max_abs_delta | residual_saturation_ratio | residual_clip_fraction | uses_teacher_policy | uses_true_fault_state | obs_actor | obs_critic | policy_dim | action_dim | validation_source | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A5 | student_residual_scale_0_1 | residual | rlm1_stripped | none | Isaac-Ant-Student-v0 | 0 | 0.1 | `TERM=xterm ./trainers/run_t09_ablation_smoke.sh --ablation_id A5 --execute_smoke --num_envs 8 --max_iterations 1` | `logs/rsl_rl/residual__rlm1_stripped__none/2026-06-01_15-46-39_residual__rlm1_stripped__none__seed0` | `checkpoints/rlm1_stripped/residual/none/seed0/latest_checkpoint.yaml` | observed | one-row smoke | 0/1 | 0.0077 | -0.0138 | 11.3543 | 0.07 | 5.00 | 0.0582 | 0.0982 | 0.0430 | 0.0000 | False | False | policy only | policy only | 60 | 8 | user-supplied T09-C smoke log and checkpoint pointer inspection | T08.6 deterministic seed print, runtime summary, residual diagnostics, and PPO iteration appeared; no NaN/Inf failure and no hang were observed. |

Recorded residual diagnostics:

- `Residual/mean_abs_delta`: `0.0582`
- `Residual/max_abs_delta`: `0.0982`
- `Residual/saturation_ratio`: `0.0430`
- `Residual/clip_fraction`: `0.0000`

## Pending Rows

| ablation_id | name | status | reason |
| --- | --- | --- | --- |
| A0 | healthy_ppo | pending | not executed in T09-D1 |
| A1 | privileged_teacher_reference | pending | not executed in T09-D1 |
| A2 | student_distilled_no_residual | pending | not executed in T09-D1 |
| A3 | student_residual_scale_0_0 | pending | not executed in T09-D1 |
| A4 | student_residual_scale_0_05 | pending | not executed in T09-D1 |
| A6 | student_residual_scale_0_2 | pending | not executed in T09-D1 |

## Checkpoint Pointer

Stable residual pointer path:

```text
checkpoints/rlm1_stripped/residual/none/seed0/latest_checkpoint.yaml
```

Latest pointer contents recorded during T09-D1:

```yaml
stage: residual
method: rlm1_stripped
fault: none
seed: 0
task: Isaac-Ant-Student-v0
experiment_name: residual__rlm1_stripped__none
run_name: residual__rlm1_stripped__none__seed0
log_dir: logs/rsl_rl/residual__rlm1_stripped__none/2026-06-01_15-46-39_residual__rlm1_stripped__none__seed0
checkpoint_path: logs/rsl_rl/residual__rlm1_stripped__none/2026-06-01_15-46-39_residual__rlm1_stripped__none__seed0/model_0.pt
```

Checkpoint path:

```text
logs/rsl_rl/residual__rlm1_stripped__none/2026-06-01_15-46-39_residual__rlm1_stripped__none__seed0/model_0.pt
```

Checkpoint exists at T09-D1 inspection time: `True`.

## Guardrails

- no full sweep was run or recorded in T09-D1.
- No evaluation runner was executed.
- No fault curriculum was added.
- health token OFF.
- Uncertainty channel inactive.
- Safety shield / CBF inactive.
- P1/P2/P3 expansion inactive.
- No source code, config, launcher, or checkpoint pointer was modified for this observed manifest.

## Deferred

- Full A0-A6 observed results.
- Full evaluation runner.
- Faulted scenario evaluation.
- Multi-seed aggregation.
- Statistical summaries.
- Paper table export.
- Health token.
- Uncertainty channel.
- Safety shield / CBF.
- P1/P2/P3 expansion.
