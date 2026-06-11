# A7 A0 Residual P2 Diagnostic Smoke

Candidate-level Isaac-side smoke diagnostic only; not paper-grade final.

## Runs

| alpha | mean_vel_x_post_fault | mean_abs_vx_error_post_fault | mean_abs_yaw_error | torso_height_failure_rate | timeout_rate | residual_action_mean_norm | residual_action_max_norm | a0_action_mean_norm | final_action_mean_norm | fallback_used | simulation_override_applied | p2_fault_became_active | no_nan_inf |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|---:|:---:|:---:|
| 0.000 | 0.348679646 | 0.651344493 | 0.318662582 | 0.032500000 | 0.000000000 | 1.397599108 | 4.064560890 | 1.687667039 | 1.687667039 | false | 1.0 | true | true |
| 0.010 | 0.335564518 | 0.664435473 | 0.314993025 | 0.032187500 | 0.000000000 | 1.398996155 | 4.157457829 | 1.670960013 | 1.661638733 | false | 1.0 | true | true |
| 0.025 | 0.351486123 | 0.653265652 | 0.312742823 | 0.031875000 | 0.000000000 | 1.400595102 | 4.059706211 | 1.656896158 | 1.633514187 | false | 1.0 | true | true |
| 0.050 | 0.142599602 | 0.857770214 | 0.327942817 | 0.032187500 | 0.000000000 | 1.379222998 | 4.229027271 | 1.624324867 | 1.578398597 | false | 1.0 | true | true |
| 0.100 | 0.123481693 | 0.876755799 | 0.312741240 | 0.032500000 | 0.000000000 | 1.380454613 | 4.289039612 | 1.594147148 | 1.501195623 | false | 1.0 | true | true |

## Interpretation

- Alpha 0.0 follows the frozen A0 action path inside the A7 evaluator: final action norm equals A0 action norm.
- No tiny positive alpha improved post-fault vx error over alpha 0.0 (0.651344493); best positive alpha was 0.025 (0.653265652).
- Required output/P2 checks: passed.

## Check Details

| run | files_present | observation_dim_61 | action_dim_8 | no_nan_inf | fallback_used_false | simulation_override_1 | p2_fault_became_active |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| alpha_000 | true | true | true | true | true | true | true |
| alpha_001 | true | true | true | true | true | true | true |
| alpha_0025 | true | true | true | true | true | true | true |
| alpha_005 | true | true | true | true | true | true | true |
| alpha_010 | true | true | true | true | true | true | true |
