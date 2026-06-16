# T17 Multi-Joint Closed-Loop Results Summary

Preliminary advisor-facing candidate evidence only; not paper-grade final.

- aggregated rows: `1`
- completed runs: `1`
- failed runs: `0`

| policy | run_status | protocol | velocity_mode | alpha | checkpoint | base_checkpoint | residual_checkpoint | num_envs | num_steps | seed | mean_vel_x_pre_fault | mean_vel_x_post_fault | mean_abs_vx_error_pre_fault | mean_abs_vx_error_post_fault | median_abs_vx_error_post_fault | p90_abs_vx_error_post_fault | survival_rate | done_rate | torso_height_failure_rate | timeout_rate | mean_abs_yaw_error | vx_cmd_mean | vx_cmd_min | vx_cmd_max | command_mode_valid | command_mode_validation_error | fault_active_any_ever | fault_active_sample_count | fault_active_env_count | post_fault_sample_count | fault_active_fraction_last_step | insufficient_post_fault_coverage | smoke_partial_fault_coverage_ok | selected_fault_joint_index_min | selected_fault_joint_index_max | selected_fault_joint_all_8_covered | supported_fault_joint_count | fallback_used | pd_surrogate_used | simulation_override_applied | p2_fault_became_active | no_nan_inf | error_type | error_message | error_summary_json | output_dir |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| a2_history | completed | realistic_random | command_random |  | papers/conference/results/t15_a2_history_h16_multijoint_seed0/a2_history_h16_multijoint.pt |  |  | 16 | 200 | 0 | 0.489426893 | 0.708524515 | 0.298546516 | 0.117350319 | 0.2143379 | 0.228219181 | 0.875 | 0.125 | 0.079375 | 0 | 0.0902623069 | 0.758195105 | 0.215259865 | 1.44405258 | true |  | true | 23 | 1 | 23 | 0.0625 | true | true | 0 | 7 | false | 8 | false | false | true | true | true |  |  |  | papers/conference/results/t17_multijoint_closed_loop_eval_smoke_a2_history_realistic/realistic_random/command_random/a2_history |
