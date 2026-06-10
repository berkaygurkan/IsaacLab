# T10 Velocity P2-Random Candidate Evaluation Plan

Scope: T10-IL-05 controlled candidate comparison only. This is not paper-grade final evaluation and does not train, freeze checkpoints, edit checkpoint pointers, update observed-result manifests, or modify RSL-RL/Isaac Lab core.

## Checkpoints

| policy | task | checkpoint |
| --- | --- | --- |
| A0-Vel healthy | `Isaac-Ant-Velocity-Flat-v0` | `logs/rsl_rl/healthy_baseline_velocity__rlm1_stripped__none/2026-06-09_23-58-25_a0_velocity_candidate1000__seed0/model_999.pt` |
| A1-F-Vel P2 random teacher | `Isaac-Ant-Teacher-Velocity-Flat-v0` | `logs/rsl_rl/teacher_p2_velocity__rlm1_stripped__p2_locked_joint/2026-06-10_00-32-52_a1f_velocity_p2_random_candidate1000__seed0/model_999.pt` |

A0 and A1-F intentionally use different task IDs because their observation dimensions differ. A1-F is privileged/reference only and is not deployment-facing.

## Evaluation Protocol

- same `num_envs`
- same `num_steps`
- same seed where feasible
- P2 target joint: `front_left_foot`
- P2 semantics: `simulation_joint_state_override_lock`
- fallback: disabled
- P2 onset mode: `random_uniform`
- P2 onset range: `30` to `150`
- fixed velocity command: `vx_cmd=1.0`, `vy_cmd=0.0`, `yaw_rate_cmd=0.0`
- inference only, no learning

## Runner

```text
evaluators/run_t10_velocity_p2_random_eval_compare.py
```

T10-IL-05c runs each policy in a separate subprocess and therefore a separate Isaac app lifecycle:

- parent output: `<output_dir>/`
- A0 child output: `<output_dir>/a0_single/`
- A1-F child output: `<output_dir>/a1f_single/`

This avoids Isaac/USD lifecycle hangs observed when constructing `Isaac-Ant-Velocity-Flat-v0` and then `Isaac-Ant-Teacher-Velocity-Flat-v0` sequentially inside the same Python process.

Output files:

- `summary.json`
- `rollout_metrics.csv`
- `velocity_timeseries.csv`
- `command.txt`
- `subprocess_commands.txt`

Single-policy subprocess mode is internal to the compare runner and uses:

```text
--single_policy_label
--single_policy_task
--single_policy_checkpoint
--single_policy_output_dir
```

The public compare command remains the normal entry point.

## Metrics

The runner records per-policy metrics including:

- `total_reward_mean`
- `episode_length_mean` when available
- `timeout_rate` when available
- `torso_height_failure_rate` when available
- `mean_vel_x_pre_fault`
- `mean_vel_x_post_fault`
- `mean_abs_vx_error_pre_fault`
- `mean_abs_vx_error_post_fault`
- `mean_abs_yaw_error` when available
- `P2/fault_applied`
- `P2/simulation_override_applied`
- `P2/fallback_used`
- `P2/onset_step_mean`
- `P2/onset_step_min`
- `P2/onset_step_max`
- `P2/per_env_onset_randomization`
- `survived_to_fault_onset_rate`
- `mean_post_fault_survival_steps`
- `done_count`
- metric notes for unavailable values

Pre-fault velocity metrics use steps before the minimum random onset. Post-fault metrics use steps at or after the maximum random onset so the aggregate windows avoid mixed pre/post fault state as much as possible.

## Expected Interpretation

This run compares an A0 velocity-tracking policy trained healthy/no-fault against a privileged A1-F velocity-tracking teacher trained with P2 random onset. It is useful as a controlled candidate sanity check under a matched P2 random protocol. It is not a fair deployment comparison because A1-F has privileged observations.

Future deployment-facing comparison should use A0 vs A2 vs A5/A6/A7 after student/residual readiness is established.

## Command

```bash
python evaluators/run_t10_velocity_p2_random_eval_compare.py \
  --execute_eval \
  --a0_checkpoint logs/rsl_rl/healthy_baseline_velocity__rlm1_stripped__none/2026-06-09_23-58-25_a0_velocity_candidate1000__seed0/model_999.pt \
  --a1f_checkpoint logs/rsl_rl/teacher_p2_velocity__rlm1_stripped__p2_locked_joint/2026-06-10_00-32-52_a1f_velocity_p2_random_candidate1000__seed0/model_999.pt \
  --output_dir runs/t10_velocity_p2_random_eval/a0_vs_a1f_candidate1000_seed0 \
  --num_envs 128 \
  --num_steps 1000 \
  --seed 0 \
  --fault_onset_mode random_uniform \
  --fault_onset_step_min 30 \
  --fault_onset_step_max 150 \
  --headless
```

## Guardrails

- no training
- no checkpoint freeze
- no checkpoint pointer edits
- no paper-grade observed manifest update
- no student/residual/health-token/UQ/CBF/P3/P4 additions
- A1-F remains privileged/reference only
- exact checkpoint paths are recorded in `summary.json`
