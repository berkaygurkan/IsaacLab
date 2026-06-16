# T17 Multi-Joint Closed-Loop Evaluation

Preliminary advisor-facing closed-loop scaffold for RLM1 stripped / conference A2 and A5 policies.

## Scope

- This is evaluation only, not offline training and not RL training.
- This is candidate-level evidence, not paper-grade final reporting.
- No checkpoint, dataset, task config, or P2 wrapper mutation is performed.
- Latent z_t t-SNE/UMAP is deferred to T18.

## Policies

- A2 single-step: `papers/conference/results/t15_a2_single_step_multijoint_seed0/a2_single_step_multijoint.pt`
- A2-history H16: `papers/conference/results/t15_a2_history_h16_multijoint_seed0/a2_history_h16_multijoint.pt`
- A5 residual: `papers/conference/results/t16_a5_history_residual_multijoint_seed0/a5_history_residual_multijoint.pt`
- A5 action rule: `final_action = A2_history(history) + alpha * A5_residual(history)`
- Optional teacher reference delegates to T13C if `teacher_reference` is requested.

## Fault Semantics

- one selected locked joint per env/episode
- selected joint random over all 8 Ant actuated joints
- q_lock captured from current selected joint position at onset
- direct `simulation_joint_state_override_lock`
- fallback disabled; PD surrogate disabled
- health token OFF; no explicit fault token into deployment-facing student policies

## Protocols

- `realistic_random`: onset U(120,700)
- `late_random`: onset U(250,700)
- `stress_random`: onset U(30,700), opt-in via `--protocol stress_random` or `--protocol all`

## Velocity Modes

- `command_random`: primary mode, vx_cmd in [0.2, 1.5]
- `fixed_vx_1p0`: optional comparability mode with guarded vx=1.0 forcing

## Outputs

- `results_summary.md`
- `results_summary.json`
- `per_run_metrics.csv`
- `per_joint_metrics.csv`
- `advisor_update_snippet.txt`
- per-run `summary.json`, `rollout_metrics.csv`, `velocity_timeseries.csv`, `per_joint_metrics.csv`, `terminal.log`
