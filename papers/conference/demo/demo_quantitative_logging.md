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

## P2 Quick Sanity Logging

T09-R2j adds a separate P2 quick sanity evaluator:

```bash
python evaluators/run_t09_p2_quick_demo_compare.py --dry_run --policy_label dry_run_A0 --task Isaac-Ant-v0 --checkpoint_path dummy.pt --fault_profile P2_locked_joint --target_joint front_left_foot --fault_onset_step 50 --num_envs 64 --num_steps 1000 --output_dir runs/t09_quick_p2_compare/dry_run
```

When executed explicitly with `--execute_demo`, it writes:

- `summary.json`
- `rollout_metrics.csv`
- `command.txt`

The summary is labelled:

```text
demo_scope: quick_p2_checkpoint_sanity
not_paper_grade: true
```

For `P2_locked_joint`, the evaluator requires
`actual_semantics=simulation_joint_state_override_lock` and fallback disabled.
A0 versus A1-F is sanity/upper-bound only because A1-F is privileged and not
deployment-facing. The fair deployment comparison remains future A0 versus A2
versus A5.

T09-R2k adds P2-specific survival diagnostics to this quick sanity evaluator.
These metrics track the first episode after the initial reset: whether each env
failed before the configured fault onset, whether it survived to the fault
window, and how many steps it survived after the fault window was reached. This
matters because a policy that terminates before onset can make raw
`P2/fault_applied` and mean reward hard to compare. For `F0_none`, the survival
fields are `null` or blank because they are P2-onset diagnostics, not generic
healthy-play metrics.

T09-R2l adds forward-velocity logging for the quick P2 sanity runs. The evaluator
records `mean_vel_x`, `mean_base_lin_vel_x`, `std_base_lin_vel_x`,
`mean_abs_base_lin_vel_x`, and `velocity_metric_source` when root/base velocity
can be resolved from the runtime env state or explicit info fields. If velocity
cannot be resolved, the evaluator warns and leaves velocity values empty rather
than fabricating a curve.

The offline plot helper:

```bash
python evaluators/plot_t09_p2_velocity_demo.py --help
```

reads A0 and A1-F `rollout_metrics.csv` files and saves an advisor-facing
velocity-vs-step plot with a vertical dashed line at `fault_onset_step=50`.
This plot is a mini demo/sanity visualization only. A1-F is a privileged
teacher/upper-bound reference, not a deployment-facing comparison. The fair
future comparison remains A0 versus A2 versus A5.

T09-R2m adds a more controlled mini-demo mode because the aggregate
`quick_mean` plot can be dominated by different nominal speed regimes between
A0 and A1-F. The new evaluator mode is:

```text
--demo_mode controlled_single_rollout
```

It is intended for same-seed, same-fault-onset traces with a later onset such as
`fault_onset_step=250`, after nominal walking has started. It logs both
`mean_vel_x` and `representative_env_vel_x`, plus pre/post windows around the
fault. If a real velocity command manager is exposed, `--target_vx` may be used
and the command source is recorded. If no real target command exists, the run
records `target_vx_available: false` and no target velocity is fabricated.

The controlled plot helper:

```bash
python evaluators/plot_t09_p2_controlled_velocity_demo.py --help
```

plots A0 and A1-F forward-velocity traces around the joint lock with a vertical
fault-onset line. This is advisor-facing sanity visualization only, not
paper-grade evidence.

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
- P2 quick sanity logging is not paper-grade evaluation and does not update observed-result manifests.
- A1-F quick sanity runs are privileged reference checks, not deployment-facing comparisons.
