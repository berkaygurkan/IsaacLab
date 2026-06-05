# T09 Optional A7 Teacher-Distilled Residual

Scope: optional ablation registration only; no training, evaluation, checkpoint
freeze, pointer edit, Isaac Sim launch, or paper-grade claim.

## Definition

A7 is an optional teacher-distilled residual ablation under the T09-R2d P2
single-joint-lock conference scope:

```text
A7 = healthy PPO + teacher-distilled residual
```

The base action comes from frozen A0 healthy PPO. A future A1-F P2 fault-aware
privileged teacher provides target actions during residual distillation data
generation. The residual target is:

```text
teacher_action_under_P2 - A0_action_under_P2
```

The intended final action is:

```text
healthy_ppo_action + residual_correction
```

Default residual-scale candidate is `0.1`, but this is still `TBD` and must be
chosen in a later implementation step.

## Method Role

A7 is optional and deferred. It is not the main RLM1 stripped method and must not
replace A5 without an explicit later decision. A5 remains the main
student-plus-residual correction row.

## Teacher / Student / Residual Distinction

- A1-H is a healthy teacher pretrain/reference and is not the final P2 fault-aware teacher.
- A1-F is the future fault-aware privileged teacher trained under a P2 single-joint-lock curriculum.
- A2 is the student distilled from A1-F under P2, with no residual correction.
- A5 is the main P2 student plus residual correction method row.
- A7 is the optional healthy PPO plus teacher-distilled residual ablation under P2.

## Runtime Privilege Rule

A7 may be considered deployment-facing only if the final runtime policy does not
require the privileged teacher, `teacher_policy`, or `true_fault_state`. The
teacher and privileged fault state may be used on the teacher side during
training/data generation only.

## Checkpoint Dependencies

Before A7 training can be considered, two dependencies must be canonical and
frozen:

```text
checkpoints/rlm1_stripped/healthy_baseline/none/seed0/canonical_checkpoint.yaml
checkpoints/rlm1_stripped/teacher_p2/none/seed0/canonical_checkpoint.yaml
```

If trained later, A7 must get its own canonical pointer:

```text
checkpoints/rlm1_stripped/teacher_distilled_residual/none/seed0/canonical_checkpoint.yaml
```

A7 must not reuse the A5 residual pointer.

## P2 Scope

The default A7 conference fault scope is:

```text
fault_profile: P2_locked_joint
target_joint: front_left_foot
fault_onset_step: 50
```

P4 torque degradation is not the main conference target for A7. P4 remains a
deferred thesis-extension or advisor-demo artifact unless a later scope decision
explicitly restores it.

## Current Classification

```text
classification: optional_deferred
controlled_eval_use: false
checkpoint_expected_now: false
```

## Guardrails

- No health token, UQ, or CBF is active.
- No P2 execution or P3 expansion is introduced by this optional registration.
- No training or evaluation is launched by this registration.
- No observed-result manifest is updated.
- No paper-grade claim is made.
