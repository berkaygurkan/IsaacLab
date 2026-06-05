# A0 Demo Plot Report

Title: A0 advisor demo: no-fault vs P4

Note: advisor demo only, not paper-grade result

## Run Folders

- F0_none: `runs/t09_demo_play/2026-06-03_04-21-40_A0_F0_long_model_logged`
  - checkpoint_path: `logs/rsl_rl/healthy_baseline__rlm1_stripped__demo/2026-06-03_04-08-57_healthy_demo__seed0/model_1999.pt`
  - resolved_checkpoint_path: `logs/rsl_rl/healthy_baseline__rlm1_stripped__demo/2026-06-03_04-08-57_healthy_demo__seed0/model_1999.pt`
- P4 torque_scale=0.5: `runs/t09_demo_play/2026-06-03_04-22-22_A0_P4_scale_0_5_long_model_logged`
  - checkpoint_path: `logs/rsl_rl/healthy_baseline__rlm1_stripped__demo/2026-06-03_04-08-57_healthy_demo__seed0/model_1999.pt`
  - resolved_checkpoint_path: `logs/rsl_rl/healthy_baseline__rlm1_stripped__demo/2026-06-03_04-08-57_healthy_demo__seed0/model_1999.pt`
- P4 torque_scale=0.0: `runs/t09_demo_play/2026-06-03_04-34-11_A0_P4_visual_stress_scale_0_0_model_1999_logged_retry`
  - checkpoint_path: `logs/rsl_rl/healthy_baseline__rlm1_stripped__demo/2026-06-03_04-08-57_healthy_demo__seed0/model_1999.pt`
  - resolved_checkpoint_path: `logs/rsl_rl/healthy_baseline__rlm1_stripped__demo/2026-06-03_04-08-57_healthy_demo__seed0/model_1999.pt`

## Generated Figures

- `papers/conference/demo/advisor_meeting_2026_06_03/figures/reward_mean_vs_step.png`
- `papers/conference/demo/advisor_meeting_2026_06_03/figures/base_lin_vel_x_mean_vs_step.png`
- `papers/conference/demo/advisor_meeting_2026_06_03/figures/action_l2_mean_vs_step.png`
- `papers/conference/demo/advisor_meeting_2026_06_03/figures/done_count_or_dones_vs_step.png`
- `papers/conference/demo/advisor_meeting_2026_06_03/figures/summary_bar_metrics.png`

## Interpretation Placeholders

- Reward trend: TBD after advisor-demo inspection.
- Forward velocity trend: TBD after advisor-demo inspection.
- Action magnitude trend: TBD after advisor-demo inspection.
- Done/fall behavior: TBD after advisor-demo inspection.

These plots are advisor-demo artifacts only, not paper-grade figures.
