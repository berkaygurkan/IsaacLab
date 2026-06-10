# T10 Velocity P2-Random Eval Plots

Scope: offline plotting only. No training, evaluation, checkpoint edits, or paper-grade manifest updates were run.

## Inputs

- input_dir: `papers/conference/results/t10_velocity_p2_random_eval_seed0_v3`
- summary: `papers/conference/results/t10_velocity_p2_random_eval_seed0_v3/summary.json`
- velocity_timeseries: `papers/conference/results/t10_velocity_p2_random_eval_seed0_v3/velocity_timeseries.csv`
- velocity column: `mean_vel_x`
- error column: `mean_abs_vx_error`

## Generated Files

- `papers/conference/results/t10_velocity_p2_random_eval_seed0_v3/plots/vel_x_time_comparison.png`
- `papers/conference/results/t10_velocity_p2_random_eval_seed0_v3/plots/vel_x_time_comparison.pdf`
- `papers/conference/results/t10_velocity_p2_random_eval_seed0_v3/plots/vx_error_time_comparison.png`
- `papers/conference/results/t10_velocity_p2_random_eval_seed0_v3/plots/vx_error_time_comparison.pdf`
- `papers/conference/results/t10_velocity_p2_random_eval_seed0_v3/plots/vel_x_and_error_time_comparison.png`
- `papers/conference/results/t10_velocity_p2_random_eval_seed0_v3/plots/vel_x_and_error_time_comparison.pdf`

## Interpretation

A1-F maintains substantially higher post-fault forward velocity than A0 in this candidate evaluation.
- A0 post-fault mean vel_x: `0.208`
- A1-F post-fault mean vel_x: `0.794`

## Caveats

- A1-F is privileged/reference only and is not deployment-facing.
- This is candidate evaluation, not paper-grade final evaluation.
