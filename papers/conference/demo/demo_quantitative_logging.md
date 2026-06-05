# T09 Demo Quantitative Logging

Scope: advisor-demo interpretation only; not paper-grade evaluation.

T09-R2d makes `P2_locked_joint` the main conference fault scope. The P4 logging
workflow documented here remains useful for advisor-demo interpretation and
thesis-extension exploration only; it is not the conference main evaluation
pipeline.

## Purpose

The demo videos can look visually subtle, especially when comparing `F0_none`
against `P4_torque_degradation` or A0 against A2 under visual-stress P4.
T09-DEMO-E/F adds lightweight quantitative logging so videos can be discussed
with per-step reward, done, fault, and action statistics from selected frozen
checkpoints.

## Outputs

Each A0/A2 demo play run writes:

- `demo_summary.md`
- `summary_metrics.json`

When `--log_step_csv` is set, it also writes:

- `step_metrics.csv`

The files are written under:

```text
runs/t09_demo_play/<timestamp>_<demo_name>/
```

## Logged Metrics

Per-step CSV fields include fault state, reward mean, done count, NaN/Inf
status, action norms, and target-action magnitudes before and after P4 scaling
when the target action is available. Policy-observation-derived fields such as
base linear velocity, yaw angular velocity, and base height are logged only when
the observation manager exposes a clear policy observation term mapping. Missing
fields are recorded as `NA` and do not stop the demo.

Summary metrics include pre/post-fault reward mean, base forward velocity mean
when available, action L2 mean, total done count, rollout length, fault-window
status, runtime smoke status, and the note:

```text
advisor demo only, not paper-grade result
```

## Comparison Helper

After recording the no-fault and P4 videos, compare their run folders with:

```bash
python evaluators/compare_t09_demo_runs.py --f0_run_dir runs/t09_demo_play/<timestamp>_D0_A0_no_fault --p4_run_dir runs/t09_demo_play/<timestamp>_D1_A0_P4
```

For the A0-vs-A2 visual-stress comparison, use labels:

```bash
python evaluators/compare_t09_demo_runs.py --left_run_dir runs/t09_demo_play/<timestamp>_D1_A0_P4_torque0_2 --right_run_dir runs/t09_demo_play/<timestamp>_D3_A2_P4_torque0_2 --left_label A0_P4_stress --right_label A2_P4_stress
```

The helper reads `summary_metrics.json` when available, falls back to
`step_metrics.csv`, prints a compact Markdown table, and writes a comparison
under:

```text
runs/t09_demo_play/comparisons/
```

## Visual-Stress Runs

The default P4 demo remains `torque_scale=0.5`. For advisor discussion only,
visual-stress demos may use `--torque_scale 0.2` or `--torque_scale 0.0`. These
runs are labelled `visual_stress_demo: True` in the summaries and must not be
used as paper-grade evidence.

## Guardrails

- No training is run by demo playback.
- P4 is applied during play/evaluation only, not during A0 training.
- P4 remains advisor-demo / thesis-extension material after the P2 conference pivot.
- A2 playback uses the frozen student checkpoint with no residual wrapper.
- No checkpoint pointer is modified.
- No observed-result manifest is updated.
- No P2 runtime evaluation, A5 residual logic, full sweep, or paper-grade claim is added here.
