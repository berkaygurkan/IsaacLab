# T09 A0 Demo Comparison

Scope: advisor-demo interpretation only; not paper-grade evaluation.

F0 run: `runs/t09_demo_play/2026-06-03_04-21-40_A0_F0_long_model_logged`
P4 run: `runs/t09_demo_play/2026-06-03_04-22-22_A0_P4_scale_0_5_long_model_logged`

| metric | F0_none | P4_torque_degradation | delta P4-F0 |
| --- | ---: | ---: | ---: |
| rollout_steps | 1200.0000 | 1200.0000 | 0.0000 |
| reward_mean | 0.1451 | 0.1438 | -0.0014 |
| base_lin_vel_x_mean | 8.0354 | 7.9947 | -0.0406 |
| action_l2_mean | 1.2638 | 1.2800 | 0.0163 |
| done_count | 9.0000 | 9.0000 | 0.0000 |
| fault_window_reached | 0.0000 | 1.0000 | 1.0000 |
| runtime_smoke_status | pass | pass | NA |

Note: advisor demo only, not paper-grade result.
