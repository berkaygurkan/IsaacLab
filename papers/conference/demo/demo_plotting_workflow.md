# T09 Demo Plotting Workflow

Scope: offline advisor-demo plotting only; not paper-grade figure generation.

## Purpose

T09-DEMO-H generates simple PNG plots for A0 no-fault versus A0 P4 demo runs.
It reads existing `step_metrics.csv` and `summary_metrics.json` files from demo
run folders. It does not launch Isaac Sim, run play, train, modify checkpoints,
or update observed-result manifests.

## Known Run Folders

```text
runs/t09_demo_play/2026-06-03_04-21-40_A0_F0_long_model_logged
runs/t09_demo_play/2026-06-03_04-22-22_A0_P4_scale_0_5_long_model_logged
runs/t09_demo_play/2026-06-03_04-34-11_A0_P4_visual_stress_scale_0_0_model_1999_logged_retry
```

## Command

```bash
python evaluators/plot_t09_demo_comparison.py \
  --f0_run_dir runs/t09_demo_play/2026-06-03_04-21-40_A0_F0_long_model_logged \
  --p4_run_dir runs/t09_demo_play/2026-06-03_04-22-22_A0_P4_scale_0_5_long_model_logged \
  --stress_run_dir runs/t09_demo_play/2026-06-03_04-34-11_A0_P4_visual_stress_scale_0_0_model_1999_logged_retry \
  --title "A0 advisor demo comparison"
```

Default output directory:

```text
papers/conference/demo/advisor_meeting_2026_06_03/figures
```

## Outputs

- `reward_mean_vs_step.png`
- `base_lin_vel_x_mean_vs_step.png`
- `action_l2_mean_vs_step.png`
- `done_count_or_dones_vs_step.png`
- `summary_bar_metrics.png`
- `a0_demo_plot_report.md`

Missing metrics are skipped with warnings. Every plot title includes:

```text
advisor demo only, not paper-grade
```

## Guardrails

- No training.
- No Isaac Sim.
- No play execution.
- No checkpoint writes or pointer updates.
- No observed-result manifest updates.
- No A2/A5/P2 expansion in this plotting patch.
- No paper-grade claims.
