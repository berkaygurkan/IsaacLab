# Selected P2 Velocity Teacher

Scope: documentation-only selected teacher record for downstream A2/A5/A7 work. This file does not update checkpoint pointers, task configs, wrappers, checkpoints, or training code.

## Selected Checkpoint

```text
logs/rsl_rl/teacher_p2_velocity_curriculum__rlm1_stripped__p2_locked_joint/2026-06-10_19-29-53_a1f_velocity_p2_onset_curriculum_seed0_curriculum_s3_target__seed0/model_2996.pt
```

## Selected Task

```text
Isaac-Ant-Teacher-Velocity-Flat-v0
```

## Selected Evaluation Protocol

- protocol: P2 random onset `30` to `150`
- onset mode: `random_uniform`
- fault: `front_left_foot` P2 joint lock
- semantics: `simulation_joint_state_override_lock`
- fallback: disabled
- evaluation role: candidate-level seed0 teacher selection

## Downstream Use

Use this checkpoint as the teacher source for:

- A2 student distillation
- A5 teacher-gap residual with distilled student
- A7 healthy PPO plus teacher-gap residual

## Guardrails

- A1-F is privileged/reference only and is not deployment-facing.
- Deployment-facing A2/A5/A7 policies must not use `true_fault_state`.
- This is not paper-grade final evaluation evidence.
- This file is a selection note, not an executable config.
