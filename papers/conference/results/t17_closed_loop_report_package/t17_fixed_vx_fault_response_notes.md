# T17 Fixed-vx Fault Response Figure Notes

Figure scope: preliminary advisor-facing multi-joint P2 fixed-vx velocity tracking; not final paper-grade statistics.

This notes file has been updated after the A2 single-step fixed-vx run became available. Regenerate the figure and plotted-runs table with `evaluators/plot_t17_fixed_vx_fault_response.py` before using the PNG/PDF as current evidence.

## Included Methods

- teacher_reference/realistic_random
- a2_single_step/realistic_random
- a2_history/realistic_random
- a5_alpha_0p25/realistic_random
- a5_alpha_0p5/realistic_random
- a5_alpha_1/realistic_random
- teacher_reference/late_random
- a2_single_step/late_random
- a2_history/late_random
- a5_alpha_0p25/late_random
- a5_alpha_0p5/late_random
- a5_alpha_1/late_random

## Missing Methods

- none expected after using `--a2_single_step_root papers/conference/results/t17_multijoint_closed_loop_eval_a2_single_step_fixed_vx1_seed0`

## Alignment Audit

- plot_mode: `absolute_time`
- fault_onset_varies: `true`
- per_env_onset_available: `true`
- per_env_velocity_available: `false`
- fault_aligned_supported: `false`
- limitation: Fault onset varies across environments, and per-env onset values are present in summary.json, but velocity_timeseries.csv/rollout_metrics.csv contain aggregate mean traces rather than per-env velocity trajectories. Fault-aligned recovery curves cannot be reconstructed from current artifacts.

## A5 Readout

- visually strongest A5 curve: `A5 alpha=1.0`
- best A5 alpha by fixed-vx mean post-fault error, realistic_random: `1.0`
- best A5 alpha by fixed-vx mean post-fault error, late_random: `1.0`

## Interpretation / Caption Guidance

- A5 improves the history-student baseline.
- A2 single-step is unexpectedly strong under fixed-vx=1.0.
- The all-method fixed-vx figure should be interpreted as an ablation comparison, not as evidence that A5 dominates every deployment-facing baseline.
- The A1-F teacher is privileged and non-deployment-facing.
- Suggested caption wording: A5 improves the history-student baseline, while A2 single-step remains a strong fixed-command baseline; further command-random comparison is required before choosing the final deployment-facing variant.
- best deployment-facing method by fixed-vx mean post-fault error, realistic_random: `a2_single_step`
- best deployment-facing method by fixed-vx mean post-fault error, late_random: `a2_single_step`
