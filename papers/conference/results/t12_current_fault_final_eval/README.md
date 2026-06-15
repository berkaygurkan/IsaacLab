# T12 Current-Fault Final Eval

T12 Step 1 is a current-fault final-evaluation scaffold for the conference-stage RLM1 stripped implementation. It targets the teacher-seen P2 fault:

- fault: `P2_locked_joint`
- target joint: `front_left_foot`
- semantics: `simulation_joint_state_override_lock`
- fallback: disabled
- random onset: `random_uniform [30, 150]`
- settled onset: `random_uniform [300, 300]`
- target velocity: `vx_cmd = 1.0`

This is candidate-level evaluation infrastructure only. It is not paper-grade final evidence until the manual runs complete under a unified protocol.

## Files

- runner: `evaluators/run_t12_current_fault_final_eval.py`
- plotter: `evaluators/plot_t12_current_fault_final_eval.py`
- command record: `papers/conference/results/t12_current_fault_final_eval/t12_run_commands.md`
- metrics tables: `papers/conference/results/t12_current_fault_final_eval/tables/`
- plots: `papers/conference/results/t12_current_fault_final_eval/plots/`

## Policies

- A0 healthy PPO baseline
- A1-F privileged teacher reference
- A2 single-step student
- A2 history H16 student
- A5 residual over A2 history, alpha in `0.0, 0.25, 0.5, 0.75, 1.0`
- A7 residual over A0, alpha in `0.0, 0.01, 0.025, 0.05, 0.10, 0.25`

A1-F is privileged/reference only. Deployment-facing policies must not use `teacher_obs`, `true_fault_state`, health token, UQ, or CBF.

## Dry Run

```bash
python evaluators/run_t12_current_fault_final_eval.py --dry_run --protocol random --policy a2_history_h16 --num_envs 16 --num_steps 200 --device cuda
```

Dry-run mode prints commands and writes `t12_run_commands.md`. It does not launch Isaac and does not write fake metrics.

## Execution

```bash
python evaluators/run_t12_current_fault_final_eval.py --execute --protocol random --policy a2_history_h16 --num_envs 16 --num_steps 200 --device cuda
```

Execution mode runs commands sequentially, writes `terminal.log` per run, requires `summary.json` and `velocity_timeseries.csv`, and aggregates completed runs into `tables/metrics_summary.csv` and `tables/metrics_summary.md`.

## Plotting

```bash
python evaluators/plot_t12_current_fault_final_eval.py --input_root papers/conference/results/t12_current_fault_final_eval --protocol both
```

The plotter reads completed `velocity_timeseries.csv` files and saves PNG/PDF plots. It fails clearly if matplotlib is unavailable.

## Guardrails

- Do not train.
- Do not modify checkpoints.
- Do not modify task configs.
- Do not modify P2 wrapper semantics.
- Do not use health token, UQ, or CBF.
- Do not make paper-grade claims from incomplete candidate-level outputs.
