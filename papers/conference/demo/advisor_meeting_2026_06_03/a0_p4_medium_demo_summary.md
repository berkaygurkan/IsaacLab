# T09 Demo Summary

demo_name: A0_P4_scale_0_5_long_model_logged
ablation_id: A0
fault_profile: P4_torque_degradation
target_joint: front_left_foot
torque_scale: 0.5
fault_onset_step: 50
rollout_steps_executed: 1200
fault_window_reached: True
fault_applied: True
no_nan_inf: True
runtime_smoke_status: pass
visual_stress_demo: False

## Quantitative Metrics

pre_fault_reward_mean: 0.03070203388109803
post_fault_reward_mean: 0.148690079583422
pre_fault_base_lin_vel_x_mean: 1.4639662411808967
post_fault_base_lin_vel_x_mean: 8.278691963061043
pre_fault_action_l2_mean: 1.527320556640625
post_fault_action_l2_mean: 1.2692796630444734
total_done_count: 9

## Checkpoints

checkpoint_path: logs/rsl_rl/healthy_baseline__rlm1_stripped__demo/2026-06-03_04-08-57_healthy_demo__seed0/model_1999.pt
checkpoint_pointer: None
resolved_checkpoint_path: logs/rsl_rl/healthy_baseline__rlm1_stripped__demo/2026-06-03_04-08-57_healthy_demo__seed0/model_1999.pt

note: "advisor demo only, not paper-grade result"
