# T10 Settled-Regime P2 Velocity Three-Way Comparison

## Input Directories

- A0 Healthy PPO: `papers/conference/results/t10_velocity_p2_settled_eval_seed0/a0_single`
- A1-F Random Teacher: `papers/conference/results/t10_velocity_p2_settled_eval_seed0/a1f_random_teacher_single`
- A1-F Curriculum Teacher: `papers/conference/results/t10_velocity_p2_settled_eval_seed0/a1f_curriculum_teacher_single`

## Protocol Summary

- `vx_cmd`: `1.0` m/s
- `num_envs`: `128`
- `num_steps`: `1000`
- `seed`: `0`
- P2 fault: `front_left_foot` joint lock
- fault onset: step `300`
- evaluator onset mode: `random_uniform` with `min=max=300`
- semantics: `simulation_joint_state_override_lock`
- fallback: disabled

## Interpretation Notes

- `velocity_timeseries.csv` stores per-step aggregate metrics.
- `mean_vel_x` curves are averaged across 128 parallel environments per step, not single-trajectory traces.
- The rollout is 1000 steps over 128 parallel environments; it is not 1000 independent tests.
- A0 is the deployment-facing healthy PPO baseline.
- A1-F teachers are privileged/reference policies and are not deployment-facing.

## CSV Policy Labels

- A0 Healthy PPO: `A0_Vel_healthy_train_eval_P2_random`
- A1-F Random Teacher: `A1F_Vel_P2_random_train_eval_P2_random`
- A1-F Curriculum Teacher: `A1F_Vel_P2_random_train_eval_P2_random`

## Settled-Window Summary

Pre-fault settled window: steps `250-299`.
Post-fault window: steps `300-1000`.

| policy | mean_vel_x_pre_settled | mean_vel_x_post_fault | mean_abs_vx_error_pre_settled | mean_abs_vx_error_post_fault | velocity_retention | torso_height_failure_rate | timeout_rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A0 Healthy PPO | 0.858512 | 0.265208 | 0.141488 | 0.734792 | 0.308916 | 0.057266 | 0.038437 |
| A1-F Random Teacher | 0.865715 | 0.778958 | 0.134285 | 0.221042 | 0.899786 | 0.072422 | 0.037906 |
| A1-F Curriculum Teacher | 0.972565 | 0.916884 | 0.027435 | 0.083140 | 0.942748 | 0.086008 | 0.037164 |

## Generated Files

- `vel_x_time_three_way.png`
- `vel_x_time_three_way.pdf`
- `vx_error_time_three_way.png`
- `vx_error_time_three_way.pdf`
- `vel_x_and_vx_error_three_way.png`
- `vel_x_and_vx_error_three_way.pdf`
