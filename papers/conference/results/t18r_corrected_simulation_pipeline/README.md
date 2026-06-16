# T18-R Corrected Simulation Pipeline

This note defines the corrected 50 Hz / H50 result phase for the conference-stage stripped RLM1 pipeline.

## Scope

- Phase: `T18-R corrected simulation pipeline`
- Method: RLM1 stripped / conference
- Teacher-student: ON
- Residual: ON
- Health token: OFF
- UQ: OFF
- CBF: OFF
- Deployment-facing policy inputs: 61-D student observation only
- Excluded from deployment-facing inputs: teacher observation, selected joint id, selected joint one-hot, q-lock vector, P2-active flag, health token, uncertainty, safety-filter outputs
- Previous T14/T15/T16/T17 artifacts remain preliminary 60 Hz / H16 artifacts and must not be overwritten.

## Control Timing Audit

The current Ant task default is defined in `source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/ant/ant_env_cfg.py`:

- physics timestep: `sim.dt = 1 / 120.0`
- action/control decimation: `decimation = 2`
- effective control timestep: `2 * (1 / 120) = 1 / 60 s`
- effective control frequency: approximately `60 Hz`

T18-R uses an explicit opt-in timing override:

- physics timestep: `sim.dt = 0.01 s`
- action/control decimation: `2`
- effective control timestep: `0.01 * 2 = 0.02 s`
- effective control frequency: `50 Hz`
- history length: `H50`
- history duration: `50 * 0.02 = 1.0 s`

The base task defaults are not changed. T18-R commands must pass:

```bash
--control_frequency_hz 50 --sim_dt 0.01 --decimation 2 --require_control_frequency_hz 50
```

The dataset collector and closed-loop evaluators record `control_frequency_hz`, `control_dt_s`, `sim_dt_s`, and `decimation` in new summaries.

## Fault Semantics

- Fault type: randomized single locked-joint P2 fault
- Selected failed joint: one of the 8 actuated Ant joints, sampled per env/episode
- q_lock: captured from the selected joint's current simulated joint position at fault onset
- Direct simulator override after onset: `q[selected_joint] = q_lock`, `qd[selected_joint] = 0`
- Fallback: disabled
- PD surrogate: disabled

## Corrected Comparison

The corrected fair residual comparison includes:

- A2-history H50
- A5-H50 residual with alpha sweep: `0.25`, `0.5`, `1.0`
- Privileged A1-F teacher only as a non-deployment reference

A2 single-step is intentionally excluded from the corrected residual-ablation figure/table for this phase. It remains a separate baseline analysis.

## Expected Roots

- `papers/conference/datasets/t18r_50hz_h50_teacher_dataset_realistic_seed0`
- `papers/conference/datasets/t18r_50hz_h50_teacher_dataset_late_seed0`
- `papers/conference/results/t18r_a2_history_h50_distill_seed0`
- `papers/conference/results/t18r_a5_h50_residual_seed0`
- `papers/conference/results/t18r_closed_loop_eval_50hz_h50_a2h_a5_alpha_sweep_seed0`
- `papers/conference/results/t18r_closed_loop_report_package_50hz_h50`

## Expected Figure and Table

- `fig_fixed_vx_50hz_h50_history_residual_ablation.png`
- `fig_fixed_vx_50hz_h50_history_residual_ablation.pdf`
- `table_t18r_50hz_h50_tracking_error.md`
- `table_t18r_50hz_h50_tracking_error.csv`

## Manual Commands

See `t18r_run_commands.md` in this directory.
