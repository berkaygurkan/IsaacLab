# T10 A5 History Residual P2 Alpha Sweep

Candidate-level Isaac-side A5 residual alpha sweep under RLM1 stripped. These are not paper-grade final results.

- task: `Isaac-Ant-Velocity-Flat-v0`
- num_envs: `128`
- num_steps: `1000`
- seed: `0`
- history_len: `16`
- device: `cpu`
- A2 checkpoint: `papers/conference/results/t10_a2_student_history_distill_full_h16_seed0/a2_student_history.pt`
- A5 checkpoint: `papers/conference/results/t10_a5_history_residual_distill_full_h16_seed0/a5_history_residual.pt`
- action composition: `final_action = a2_action + alpha * residual_action`
- guardrails: no training, no A0 action, no teacher_obs, no true_fault_state, no health token, no A7

## Random P2 alpha sweep

- onset: `random_uniform [30, 150]`

| alpha | post-fault mean_vel_x | post-fault mean_abs_vx_error | mean_abs_yaw_error | timeout_rate | torso_height_failure_rate | residual_action_mean_norm | residual_action_max_norm | final_action_mean_norm | P2/fallback_used | P2/simulation_override_applied | p2_fault_became_active | no_nan_inf |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.96926277 | 0.1641108808 | 0.1452205228 | 0.0374765625 | 0.0796484375 | 0.06481520452 | 0.2864333987 | 1.426113319 | 0 | 1 | true | true |
| 0.25 | 0.9445703902 | 0.1716502196 | 0.1440636051 | 0.0374765625 | 0.079140625 | 0.06424497169 | 0.1903894097 | 1.424645765 | 0 | 1 | true | true |
| 0.5 | 0.926074737 | 0.1717311366 | 0.1428464344 | 0.0374765625 | 0.07959375 | 0.06384961541 | 0.2024524808 | 1.421315165 | 0 | 1 | true | true |
| 0.75 | 0.8692848038 | 0.2018070947 | 0.1432683264 | 0.0381171875 | 0.0654375 | 0.06340189067 | 0.1792579889 | 1.41279615 | 0 | 1 | true | true |
| 1 | 0.7543116319 | 0.285645149 | 0.154713628 | 0.0377578125 | 0.068078125 | 0.06295355696 | 0.2235841453 | 1.402690626 | 0 | 1 | true | true |

## Settled P2 alpha sweep

- onset: `random_uniform [300, 300]`

| alpha | post-fault mean_vel_x | post-fault mean_abs_vx_error | mean_abs_yaw_error | timeout_rate | torso_height_failure_rate | residual_action_mean_norm | residual_action_max_norm | final_action_mean_norm | P2/fallback_used | P2/simulation_override_applied | p2_fault_became_active | no_nan_inf |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.9716214925 | 0.167929108 | 0.1396084639 | 0.03715625 | 0.0866640625 | 0.06491130449 | 0.2864333987 | 1.42596487 | 0 | 1 | true | true |
| 0.25 | 0.9610768691 | 0.1685112459 | 0.1377155511 | 0.036515625 | 0.0902109375 | 0.06463954629 | 0.2927853763 | 1.421410619 | 0 | 1 | true | true |
| 0.5 | 0.9141119446 | 0.1758629555 | 0.1385588177 | 0.03715625 | 0.0864765625 | 0.06405734787 | 0.2024524808 | 1.412592442 | 0 | 1 | true | true |
| 0.75 | 0.8478936884 | 0.210394612 | 0.1402617639 | 0.037796875 | 0.0723046875 | 0.06358696924 | 0.2310471833 | 1.406327567 | 0 | 1 | true | true |
| 1 | 0.7427150813 | 0.2867828166 | 0.146895939 | 0.037796875 | 0.0677265625 | 0.06317989706 | 0.2069664598 | 1.400490702 | 0 | 1 | true | true |

## random_p2 best candidate

- best alpha by post-fault mean_abs_vx_error: `0`
- best post-fault mean_abs_vx_error: `0.1641108808`
- alpha=0.0 base post-fault mean_abs_vx_error: `0.1641108808`
- delta vs alpha=0.0: `0`

## settled_p2 best candidate

- best alpha by post-fault mean_abs_vx_error: `0`
- best post-fault mean_abs_vx_error: `0.167929108`
- alpha=0.0 base post-fault mean_abs_vx_error: `0.167929108`
- delta vs alpha=0.0: `0`
