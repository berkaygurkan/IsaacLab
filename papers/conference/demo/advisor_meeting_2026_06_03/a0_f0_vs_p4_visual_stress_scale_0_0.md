# T09 A0 Demo Comparison

Scope: advisor-demo interpretation only; not paper-grade evaluation.

F0 run: `runs/t09_demo_play/2026-06-03_04-21-40_A0_F0_long_model_logged`
P4 run: `runs/t09_demo_play/2026-06-03_04-34-11_A0_P4_visual_stress_scale_0_0_model_1999_logged_retry`

| metric | F0_none | P4_torque_degradation | delta P4-F0 |
| --- | ---: | ---: | ---: |
| rollout_steps | 1200.0000 | 1200.0000 | 0.0000 |
| reward_mean | 0.1451 | 0.0126 | -0.1325 |
| base_lin_vel_x_mean | 8.0354 | 0.3091 | -7.7263 |
| action_l2_mean | 1.2638 | 1.5948 | 0.3311 |
| done_count | 9.0000 | 10.0000 | 1.0000 |
| fault_window_reached | 0.0000 | 1.0000 | 1.0000 |
| runtime_smoke_status | pass | pass | NA |

Note: advisor demo only, not paper-grade result.
