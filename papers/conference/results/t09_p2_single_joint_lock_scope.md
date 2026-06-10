# T09 P2 Single Joint Lock Conference Scope

Scope: RLM1 stripped conference-stage fault-scope decision; no training, evaluation, Isaac Sim launch, checkpoint freeze, pointer edit, observed-result update, or paper-grade claim.

## Scope Decision

T09-R2d pivots the conference fault scope to `P2_locked_joint`. The main conference controlled-evaluation fault is a single locked joint with default target joint `front_left_foot` and default `fault_onset_step: 50`. The local P2 scaffold remains the source of locked-joint profile semantics. T09-R2h upgrades the repo-owned A1-F training hook from PD effort hold toward a simulation-level joint-state override that captures the target joint position at onset and writes that joint state back to simulation each step. Runtime evaluation execution is still deferred until a guarded P2 evaluator is explicitly implemented.

T09-R2i keeps the target joint fixed but changes the intended long A1-F teacher
training distribution from fixed onset to per-env random onset:

```text
fault_onset_mode: random_uniform
fault_onset_step_min: 30
fault_onset_step_max: 150
```

The fixed-onset step-50 teacher run is runtime-hook validation, not the final
random-onset canonical teacher. The random-onset teacher should be one long run,
not separate teachers per onset. Later controlled evaluation can still use fixed
onset points or a fixed onset grid.

This pivot is intended to simplify the paper story. A single joint lock is easier to describe, easier to align with teacher/student/residual training, and more clearly connected to recovery under a discrete actuator/joint failure than the earlier torque-scale demo path. The scope change does not claim that any policy is fault-tolerant yet.

## P4 Deferred Status

`P4_torque_degradation` is no longer the main conference fault. Existing P4 files, runtime smoke notes, advisor-demo videos, quantitative logs, and plots remain useful as infrastructure and meeting-demo artifacts. They must not be used as paper-grade evidence for the conference main pipeline unless a later scope decision explicitly restores P4 as an evaluated comparison. P4 is deferred to a thesis extension or optional future comparison.

No P4 config or demo artifact is deleted by this scope decision.

## Training And Test Split

| row | training role | evaluation role | runtime privilege |
| --- | --- | --- | --- |
| A0 | Healthy PPO trained on `F0_none` only. | Evaluated on `F0_none` and zero-shot on `P2_locked_joint`. | Deployment-facing; no teacher or `true_fault_state`. |
| A1-F | Future privileged teacher trained under P2 curriculum. | Reference teacher only; not deployment-facing. | May use privileged fault information. |
| A2 | Future student distilled from A1-F under P2 curriculum. | Evaluated on `F0_none` and `P2_locked_joint`. | No teacher or `true_fault_state` at runtime. |
| A5 | Future main residual method trained for P2 residual correction. | Evaluated on `F0_none` and `P2_locked_joint`; A5 remains the main method row. | No teacher or `true_fault_state` at runtime. |
| A7 | Optional healthy PPO plus teacher-distilled residual under P2. | Optional/deferred comparison only, not a replacement for A5. | Runtime must not require teacher or `true_fault_state`. |

## Controlled-Evaluation Rows

The controlled-evaluation scope matrix is:

```text
configs/ablation/t09_p2_controlled_eval_matrix.yaml
```

It includes the main comparison rows `A0_F0`, `A0_P2`, `A2_F0`, `A2_P2`, `A5_F0`, and `A5_P2`. It also registers optional/deferred `A7_F0` and `A7_P2` rows so the teacher-distilled residual idea is preserved without becoming paper-main.

## Checkpoint Readiness

A0 canonical is ready:

```text
checkpoints/rlm1_stripped/healthy_baseline/none/seed0/canonical_checkpoint.yaml
```

Controlled evaluation remains blocked until A1-F, A2, and A5 are trained or selected and frozen as canonical checkpoints. The existing A1-H healthy teacher pretrain is not enough for A1-F. Existing A2 and A5 pointers remain smoke/development grade unless explicitly promoted by a later freeze step.

## A7 P2 Residual Target

If A7 is implemented later, its P2 residual target is:

```text
teacher_action_under_P2 - A0_action_under_P2
```

A7 may use privileged teacher/fault information only during teacher-side training or data generation. Its runtime policy must not require `teacher_policy` or `true_fault_state` if it is treated as deployment-facing.

## Current P2 Runtime Hook Status

The current A1-F training hook path is:

```text
trainers/p2_joint_lock_training_wrapper.py
```

Desired conference semantics:

```text
position_hold_joint_lock
```

Requested conference training-hook semantics:

```text
simulation_joint_state_override_lock
```

Local API path found:

```text
robot.write_joint_state_to_sim(position, velocity, joint_ids=[target_joint_id], env_ids=active_env_ids)
```

At onset, the wrapper captures `q_lock` from the target joint for each env. After
each environment step, it writes the selected joint position back to `q_lock` and
sets selected joint velocity to `0.0`. This is a runtime simulation-state
override fault model, not a permanent URDF/DOF asset modification.

For long A1-F training, onset is sampled independently per env and resampled
after env reset. This changes the gait phase and locked angle while preserving
the same `front_left_foot` target joint.

Fallback semantics, disabled by default for A1-F:

```text
pd_position_hold_surrogate
```

Zero-effort action masking is not acceptable as the main conference P2 fault.
The preflight command is:

```bash
python evaluators/preflight_t09_p2_joint_lock.py --execute_preflight --headless
```

Full A1-F training remains blocked until this hook is verified with
`actual_semantics=simulation_joint_state_override_lock` and explicitly
acknowledged.

## Quick Demo Sanity

T09-R2j adds:

```text
evaluators/run_t09_p2_quick_demo_compare.py
```

This evaluator is a quick checkpoint sanity/demo runner only. It supports A0 on
`Isaac-Ant-v0` and A1-F on `Isaac-Ant-Teacher-v0`, accepts only `F0_none` and
`P2_locked_joint`, and writes `summary.json`, `rollout_metrics.csv`, and
`command.txt` under `runs/t09_quick_p2_compare/`.

A0 versus A1-F is not a fair deployment comparison because A1-F is a privileged
teacher/reference row. It is useful only as a sanity/upper-bound check. The
future fair deployment-facing comparison remains A0 versus A2 versus A5.

T09-R2k extends this quick sanity runner with first-episode survival diagnostics:
`survival_to_fault_onset_rate`, `failed_before_fault_count`,
`reached_fault_count`, first-done-step summaries, and post-fault survival-step
summaries. These fields are included to avoid over-reading mean reward or raw
`P2/fault_applied` when a policy terminates before the configured onset. They
remain demo/sanity diagnostics only and are not paper-grade faulted scenario
metrics.

T09-R2l adds forward-velocity logging and an offline A0-vs-A1-F velocity plotter
for the first advisor-facing mini demo. The intended plot is mean forward
velocity `vx` versus simulation step, with a vertical dashed line at
`fault_onset_step=50`. This visualization is still quick sanity only; A1-F is
privileged and not deployment-facing. The fair controlled comparison remains
future A0 versus A2 versus A5.

T09-R2m adds `controlled_single_rollout` for a cleaner advisor-facing trace
around the fault onset. This mode uses a shared seed, a later onset such as
`fault_onset_step=250`, and a representative-env or small-env-average velocity
curve so the figure is less dominated by different nominal acceleration regimes.
If a real target velocity command exists, it can be recorded and plotted; if the
Ant task exposes no writable velocity command, `target_vx_available=false` is
recorded and no command is fabricated. This remains a mini demo/sanity view, not
a paper-grade controlled evaluation.

## Guardrails

- This is a scope decision, not a result claim.
- No training or evaluation is launched.
- No checkpoint pointer is modified.
- No observed-result manifest is updated.
- No paper-grade claim is made from P4 demo artifacts.
- Health token OFF, UQ inactive, and CBF inactive remain fixed.
- No P3 expansion or new method family is introduced.
- A5 remains the main residual method; A7 remains optional/deferred.
