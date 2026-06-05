# T09 P2 Single Joint Lock Conference Scope

Scope: RLM1 stripped conference-stage fault-scope decision; no training, evaluation, Isaac Sim launch, checkpoint freeze, pointer edit, observed-result update, or paper-grade claim.

## Scope Decision

T09-R2d pivots the conference fault scope to `P2_locked_joint`. The main conference controlled-evaluation fault is a single locked joint with default target joint `front_left_foot` and default `fault_onset_step: 50`. The local P2 scaffold remains the source of locked-joint profile semantics. T09-R2f wires the first repo-owned runtime hook for A1-F teacher training as an action-override surrogate. Runtime evaluation execution is still deferred until a guarded P2 evaluator is explicitly implemented.

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

The semantics are:

```text
action_override_zero_effort_surrogate
```

After onset, the selected action dimension is set to zero. This is not a true
mechanical position-hold joint lock. The preflight command is:

```bash
python evaluators/preflight_t09_p2_joint_lock.py --execute_preflight --headless
```

Full A1-F training remains blocked until this hook is verified and explicitly
acknowledged.

## Guardrails

- This is a scope decision, not a result claim.
- No training or evaluation is launched.
- No checkpoint pointer is modified.
- No observed-result manifest is updated.
- No paper-grade claim is made from P4 demo artifacts.
- Health token OFF, UQ inactive, and CBF inactive remain fixed.
- No P3 expansion or new method family is introduced.
- A5 remains the main residual method; A7 remains optional/deferred.
