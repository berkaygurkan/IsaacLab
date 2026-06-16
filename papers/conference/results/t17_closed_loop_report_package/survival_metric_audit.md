# T17 Survival Metric Audit

Preliminary advisor-facing audit only; not final paper-grade statistics.

## Finding

The existing `survival_rate`/`done_rate` columns are raw rollout-level first-done-ever metrics. In 1000-step Isaac rollouts, `done_rate=1` means every env eventually produced at least one done, not that every policy immediately failed after the P2 fault.

The packager therefore renames these as:

- `current_raw_done_rate` / future `raw_any_done_rate`
- `current_raw_survival_rate` / future `raw_never_done_survival_rate`

## Post-Fault Metrics Recoverable Without Rerun

The existing root `per_joint_metrics.csv` files contain fault-active sample counts and active-sample done/survival rates by joint. The packager uses these to compute `post_fault_alive_sample_fraction` and `post_fault_done_sample_rate` for each run.

- raw done-rate range in loaded rows: `1` to `1`
- post-fault alive-sample fraction range: `0.998145919` to `0.998412212`

## Metrics Requiring Future Rerun

The current artifacts do not preserve per-env post-fault termination cause. Therefore `post_fault_torso_failure_rate`, `post_fault_timeout_success_rate`, and `post_fault_non_timeout_failure_rate` cannot be reconstructed faithfully from the existing CSVs.

The evaluator has been patched for future runs to write active-sample counters `fault_active_sample_count`, `fault_active_done_count`, `fault_active_alive_count`, plus summary aliases `raw_any_done_rate`, `raw_never_done_survival_rate`, `post_fault_alive_sample_fraction`, and `post_fault_done_sample_rate`. Cause-separated timeout/collapse rates still require explicit termination-cause logging from Isaac/env extras.

## Interpretation

- Timeout should be reported separately from collapse/failure.
- A rollout-level done event should not automatically mean post-fault failure if the env later resets.
- Survival should be reported relative to post-fault behavior, not only as no-done-ever over 1000 steps.
