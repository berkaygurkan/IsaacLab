# T09-E1 P4 Pilot Plan

Scope: RLM1 stripped conference-stage; guarded P4 torque-degradation pilot scaffold; no paper-grade claims.

## Objective

T09-E1 prepares the first tiny runtime pilot for `P4_torque_degradation`. The pilot is restricted to A0, A2, and A5. It is designed to validate fault application, action/joint mapping, checkpoint loading, and minimal pilot logging without training, checkpoint writes, observed-result manifest updates, full sweeps, or paper-grade result claims.

## Implementation Approach

P4 is implemented as an evaluator-local action wrapper in `evaluators/p4_action_degradation_wrapper.py`. The wrapper scales one selected Ant action dimension before `env.step`. It does not modify Isaac Lab event managers, task registration, source tasks, trainers, checkpoints, or checkpoint pointer YAMLs.

For A5, the intended wrapper order is:

```text
base Ant env -> P4 action degradation wrapper -> residual action wrapper -> RSL-RL vec wrapper
```

This makes P4 act on the final composed action after the frozen student and residual policy have produced it.

## Action Mapping

The runtime pilot must fail before any rollout step if the target joint cannot be mapped safely.

Required mapping checks:

- exactly one action term is active;
- `action_dim` is `8`;
- resolved joint names are available from the joint action term;
- target joint is `front_left_foot`;
- `front_left_foot` maps to exactly one action index.

The dry-run path previews this check but does not construct an Isaac Sim environment. T09-E1-B adds `--execute_mapping_preflight` for this mapping check only: it may construct the Ant env, resolve the action mapping, print the mapping, then close the env/app without loading policy checkpoints or stepping the environment.

## Pilot Rows

| ablation_id | interpretation | status |
| --- | --- | --- |
| A0 | Zero-shot healthy PPO under P4. | allowed for T09-E1 pilot |
| A2 | Frozen student under P4; no online learning. | allowed for T09-E1 pilot |
| A5 | Frozen residual-policy adaptation under P4; no online learning. | allowed for T09-E1 pilot |

A1 is privileged reference only and excluded. A3/A4/A6 are deferred until A5 proves the residual path works under P4.

## Pilot Budget

| field | value |
| --- | --- |
| fault_profile | `P4_torque_degradation` |
| target_joint | `front_left_foot` |
| torque_scale | `0.5` |
| fault_onset_step | `50` |
| num_envs | `8` |
| first episodes | `2` |
| later pilot episodes | `5` |
| policy_mode | `deterministic` |

## Policy Loading

- A0 loads the healthy baseline checkpoint pointer and records the exact resolved checkpoint path.
- A2 loads the frozen student checkpoint pointer and records the exact resolved checkpoint path.
- A5 loads the student checkpoint pointer and residual checkpoint pointer, recording both exact resolved checkpoint paths.
- The current residual checkpoint is labelled pilot/development unless explicitly promoted by a later checkpoint freeze step.

## Metrics

Minimum pilot logs:

- row id and row role;
- checkpoint pointer paths and resolved checkpoint paths;
- load success;
- `fault_applied`;
- target joint and action index;
- `action_dim`;
- `policy_dim` where available;
- episode return;
- episode length;
- termination reason if available;
- post-fault survival;
- no NaN/Inf;
- for A5: residual mean/max delta, saturation ratio, clip fraction, and residual scale.

Pilot logs are non-paper-grade and go under `runs/t09_p4_pilot/`.

## Dry-Run Commands

```bash
python evaluators/run_t09_p4_pilot_eval.py --dry_run --verify_profile --preview_mapping
python evaluators/run_t09_p4_pilot_eval.py --dry_run --ablation_id A0
python evaluators/run_t09_p4_pilot_eval.py --dry_run --ablation_id A2
python evaluators/run_t09_p4_pilot_eval.py --dry_run --ablation_id A5
```

Runtime execution requires `--execute_pilot` and should be requested separately.
The shell helper adds Isaac's `--headless` flag by default for runtime mapping preflight
and pilot execution unless `--headless` is already supplied.
Runtime debug breadcrumbs are printed with `flush=True`; `--debug_timeout_sec 30`
enables repeated Python traceback dumps during stalls, and
`--max_debug_steps_per_episode 200` bounds the pilot debug rollout.
`--policy_debug_only` constructs the env, validates P4 mapping, resets as needed,
attempts policy construction/checkpoint load, and exits before any env step.
`--debug_skip_app_close_on_error` can be paired with that debug path to surface a
pre-evaluation exception without entering a hanging `SimulationApp.close()`.
A0 policy construction converts the deprecated Ant PPO `policy` schema locally
through Isaac Lab's RSL-RL compatibility helper, then validates that
`actor.class_name`, `critic.class_name`, and policy-only actor/critic obs groups
are present before constructing `OnPolicyRunner`.
Eval-loop failures are surfaced before simulator teardown with traceback,
exception type/message, last rollout counters, last dones diagnostics, and a
status block using `pilot_status: eval_loop_exception`.
A2 done-triggered recurrent hidden-state reset uses cloned masked hidden states
and replaces the model hidden state through the local RSL-RL reset API, avoiding
unsafe in-place mutation of inference tensors.
A5 residual PPO construction uses the local RSL-RL compatibility helper, policy-only
actor/critic obs groups, and evaluator-local removal of deprecated MLP kwargs such
as `stochastic`, while preserving the actor `distribution_cfg`.

Mapping-only runtime preflight requires `--execute_mapping_preflight` and must not load checkpoints or step the environment:

```bash
bash evaluators/run_t09_p4_pilot.sh --execute_mapping_preflight --ablation_id A5 --fault_profile P4_torque_degradation
```

## Guardrails

- Default mode is dry-run.
- `--execute_pilot` is required for runtime execution.
- `--execute_mapping_preflight` is mapping-only and must not load checkpoints or step episodes.
- Runtime modes default to headless through the shell helper.
- Runtime debug tracing is enabled by default and can be disabled with `--debug_timeout_sec 0`.
- Pilot debug rollouts are bounded by `--max_debug_steps_per_episode`.
- `--policy_debug_only` must not step the environment or write observed results.
- `--debug_skip_app_close_on_error` is for debugging pre-evaluation failures only.
- Only A0/A2/A5 are allowed.
- Only `P4_torque_degradation` is allowed.
- P2 remains gated.
- No training.
- No checkpoint writes.
- No checkpoint pointer updates.
- No observed-result manifest updates.
- No paper-grade result manifests.
- No multi-fault combinations.
- No health token, UQ, CBF, or P1/P3/Px expansion.

## Deferred

- P2 locked-joint execution.
- A3/A4/A6 residual-scale sweep under P4.
- Full A0-A6 evaluation.
- Multi-seed runs.
- Paper-grade residual checkpoint promotion or retraining.
- Paper-grade figures/tables.
- Real paper claims.
