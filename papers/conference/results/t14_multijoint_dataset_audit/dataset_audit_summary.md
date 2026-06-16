# T14 Multi-Joint Dataset Audit

Offline audit for the T14 multi-joint A1-F teacher rollout datasets.

## Selection Context

- RLM phase: `RLM1 stripped / conference`
- selected teacher: `A1-F multi-joint teacher v2b, valid but foot-limited`
- checkpoint: `logs/rsl_rl/teacher_p2_multijoint_velocity_curriculum_v2b__rlm1_stripped__p2_multi_joint_random/2026-06-15_18-16-03_a1f_multijoint_velocity_p2_v2b_curriculum_curriculum_s2_realistic_transition_random_p2__seed0/model_9997.pt`
- health token: `OFF`
- teacher obs dim: `77`
- student obs dim: `61`

## Interpretation

- `realistic_random` and `late_random` are separate datasets because collecting both in one process caused the second environment creation to hang.
- Do not treat this as a method failure.
- This is candidate-level evidence for advisor sharing, not final paper-grade reporting.
- The dataset is intended for A2 single-step, A2-history H16, A5, and optional A7 A0-anchored residual training.
- Survival rate is `1 - done_rate`; timeout should not be counted as failure if timeout can be separated later.

## Dataset Checks

| dataset | protocol | samples | envs | steps | joints | vx_cmd_mean | mean_abs_vx_error | p90_abs_vx_error | done_rate | survival_rate | no_nan_inf | failed_checks |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| realistic_random | realistic_random | 128000 | 128 | 1000 | 8 | 0.877152324 | 0.194354475 | 0.467666936 | 0.001 | 0.999 | true |  |
| late_random | late_random | 128000 | 128 | 1000 | 8 | 0.877152324 | 0.180941403 | 0.418617135 | 0.001 | 0.999 | true |  |

## Required Outputs

- JSON summary: `papers/conference/results/t14_multijoint_dataset_audit/dataset_audit_summary.json`
- per-dataset CSV: `papers/conference/results/t14_multijoint_dataset_audit/per_dataset_stats.csv`
- per-joint CSV: `papers/conference/results/t14_multijoint_dataset_audit/per_joint_stats.csv`
- advisor snippet: `papers/conference/results/t14_multijoint_dataset_audit/advisor_update_snippet.txt`

## Later Graph Plan

- vx_cmd vs velocity_x tracking
- vx_error distribution
- per-joint vx_error bar plot
- survival rate per joint
- fault active vs inactive tracking
- latent `z_t` t-SNE/UMAP after A2-history/A5 training, not now; use encoder output `z_t`, not raw H16 history.
