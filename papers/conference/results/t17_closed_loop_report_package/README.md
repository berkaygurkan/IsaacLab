# T17 Closed-Loop Report Package

Post-processing package for preliminary T17 multi-joint P2 closed-loop results.

This package is generated from existing evaluator outputs only. It does not launch Isaac, train, or mutate checkpoints.

## Inputs

- `papers/conference/results/t17_multijoint_closed_loop_eval_a2h_a5_alpha_sweep_seed0`
- `papers/conference/results/t17_multijoint_closed_loop_eval_a2h_a5_alpha_sweep_fixed_vx1_seed0`
- `papers/conference/results/t17_multijoint_closed_loop_eval_a2_single_step_fixed_vx1_seed0`

## Outputs

- `t17_ablation_summary.csv/md`
- `t17_improvement_table.csv/md`
- `t17_per_joint_summary.csv/md`
- `advisor_result_snippet.md`
- `report_result_paragraph.md`
- `survival_metric_audit.md`
- `t17_fixed_vx_fault_response.png/pdf`
- `t17_fixed_vx_fault_response_notes.md`
- `t17_fixed_vx_fault_response_summary.txt`
- `t17_fixed_vx_plotted_runs.csv/md`
- `command.txt`

## Survival Metric Note

Raw `survival_rate`/`done_rate` from the original T17 runs are first-done-ever rollout metrics. Use `post_fault_alive_sample_fraction` for the current no-rerun post-fault survival proxy, and rerun with future termination-cause logging before making cause-separated survival claims.

## Fixed-vx Figure Interpretation

The fixed-vx fault-response figure should include A2 single-step from `papers/conference/results/t17_multijoint_closed_loop_eval_a2_single_step_fixed_vx1_seed0` when regenerated. A5 improves the A2-history baseline, but A2 single-step is unexpectedly strong under fixed `vx=1.0`; therefore the all-method fixed-vx figure is an ablation comparison, not evidence that A5 dominates every deployment-facing baseline. The A1-F teacher remains privileged and non-deployment-facing.
