# T11 P2 Target-Joint Stress

This directory is reserved for candidate-level P2 target-joint stress-test outputs. The scaffold does not train policies, modify checkpoints, modify task configs, or change P2 wrapper semantics.

## Runner

```text
evaluators/run_t11_p2_target_joint_stress_eval.py
```

The runner calls existing T10 evaluators as subprocesses:

- `evaluators/run_t10_a2_student_p2_eval.py`
- `evaluators/run_t10_a5_history_residual_p2_eval.py`
- `evaluators/run_t10_a7_a0_residual_p2_eval.py`

## Dry Run

```text
python evaluators/run_t11_p2_target_joint_stress_eval.py --dry_run --protocol random --joint front_left_foot --num_envs 16 --num_steps 200
```

Dry run prints commands and writes:

```text
papers/conference/results/t11_p2_target_joint_stress/dry_run_commands.txt
```

It does not launch Isaac and does not write fake summary metrics.

## Execution

```text
python evaluators/run_t11_p2_target_joint_stress_eval.py --execute --protocol random --joint front_left_foot --num_envs 16 --num_steps 200 --device cpu
```

Execution runs commands sequentially, stops on first failure, requires `summary.json` and `velocity_timeseries.csv` after each run, saves `terminal.log` per run, and aggregates successful summaries into:

- `t11_p2_target_joint_stress_summary.csv`
- `t11_p2_target_joint_stress_summary.md`

The default full pass covers both protocols, all eight target joints, and eight policy/alpha variants per joint for `128` planned commands.
