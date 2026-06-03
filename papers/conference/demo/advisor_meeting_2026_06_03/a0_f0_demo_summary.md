# T09 Demo Summary

demo_name: A0_F0_long_model_logged
ablation_id: A0
fault_profile: F0_none
target_joint: None
torque_scale: None
fault_onset_step: 50
rollout_steps_executed: 1200
fault_window_reached: False
fault_applied: False
no_nan_inf: True
runtime_smoke_status: pass
visual_stress_demo: False

## Quantitative Metrics

pre_fault_reward_mean: 0.14514405900457253
post_fault_reward_mean: None
pre_fault_base_lin_vel_x_mean: 8.035371292792261
post_fault_base_lin_vel_x_mean: None
pre_fault_action_l2_mean: 1.2637666075925031
post_fault_action_l2_mean: None
total_done_count: 9

## Checkpoints

checkpoint_path: logs/rsl_rl/healthy_baseline__rlm1_stripped__demo/2026-06-03_04-08-57_healthy_demo__seed0/model_1999.pt
checkpoint_pointer: None
resolved_checkpoint_path: logs/rsl_rl/healthy_baseline__rlm1_stripped__demo/2026-06-03_04-08-57_healthy_demo__seed0/model_1999.pt

note: "advisor demo only, not paper-grade result"
