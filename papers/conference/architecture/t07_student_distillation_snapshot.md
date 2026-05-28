# T07 Student Distillation Architecture Snapshot

Scope: Conference-stage RLM1 stripped; health token OFF.

## Stage Identity

- Stage: `student`
- Task id: `Isaac-Ant-Student-v0`
- Method variant: `rlm1_stripped`
- Fault: `none`

## Verified Interface

- Student observation group: `policy`
- Student observation dimension: `(60,)`
- Teacher observation group: `teacher_policy`
- Teacher observation dimension: `(61,)`
- Action dimension: `8`
- Action type: joint effort action

## Student-Safe Policy Terms

- `base_height`: shape `(1,)`
- `base_lin_vel`: shape `(3,)`
- `base_ang_vel`: shape `(3,)`
- `base_yaw_roll`: shape `(2,)`
- `base_angle_to_target`: shape `(1,)`
- `base_up_proj`: shape `(1,)`
- `base_heading_proj`: shape `(1,)`
- `joint_pos_norm`: shape `(8,)`
- `joint_vel_rel`: shape `(8,)`
- `feet_body_forces`: shape `(24,)`
- `actions`: shape `(8,)`

The student-safe `policy` group does not include `true_fault_state`.

## Privileged Teacher Policy Terms

- `base_height`: shape `(1,)`
- `true_fault_state`: shape `(1,)`
- `base_velocity`: shape `(3,)`
- `base_ang_vel`: shape `(3,)`
- `base_yaw_roll`: shape `(2,)`
- `base_angle_to_target`: shape `(1,)`
- `base_up_proj`: shape `(1,)`
- `base_heading_proj`: shape `(1,)`
- `joint_pos_norm`: shape `(8,)`
- `joint_vel_rel`: shape `(8,)`
- `contacts`: shape `(24,)`
- `actions`: shape `(8,)`

The `teacher_policy` group is privileged and is used only by the frozen teacher during distillation.

## Observation Mapping

```python
obs_groups = {"student": ["policy"], "teacher": ["teacher_policy"]}
```

## Network Architecture

- Student model: `RNNModel`
- Student recurrent core: `LSTM(60, 128)`
- Student MLP head: `128 -> 400 -> 200 -> 100 -> 8`
- Teacher model: `MLPModel`
- Teacher MLP: `61 -> 400 -> 200 -> 100 -> 8`
- Distillation backend: RSL-RL `DistillationRunner`

## Teacher Checkpoint Handoff

- Teacher checkpoint pointer: `checkpoints/rlm1_stripped/teacher/none/seed0/latest_checkpoint.yaml`
- Resolved teacher checkpoint used in smoke: `logs/rsl_rl/teacher__rlm1_stripped__none/2026-05-13_00-03-08_teacher__rlm1_stripped__none__seed0/model_0.pt`
- Loading behavior: the frozen T06 teacher actor is loaded into the distillation teacher with strict checkpoint loading.

## Student Artifact Convention

- TensorBoard/log path convention: `logs/rsl_rl/student__rlm1_stripped__none/`
- Student checkpoint pointer: `checkpoints/rlm1_stripped/student/none/seed0/latest_checkpoint.yaml`

## Smoke Validation

- Smoke-test command: `TERM=xterm ./trainers/run_t07_distill.sh --headless --num_envs 8 --max_iterations 1`
- Smoke-test result: one distillation iteration completed
- Mean behavior loss: `0.0042`
- Student checkpoint pointer was written

## Validation Sources

- Runtime log: observation dimensions, smoke-test completion, behavior loss, and checkpoint pointer write
- Config file: stage identity, method, fault, task, and teacher checkpoint pointer
- Code inspection: observation mapping, model architecture, action type, and distillation runner path
- Checkpoint pointer: teacher checkpoint handoff and student checkpoint convention

## Interpretation

T07 distills the privileged T06 teacher into a student policy that acts from the student-safe `policy` observation group. The student never receives `true_fault_state`; that term remains confined to the frozen teacher's `teacher_policy` input during distillation. The smoke result verifies the minimum conference-stage student handoff: teacher checkpoint resolution, strict teacher loading, one distillation update, and a stable student checkpoint pointer.

## Flow Diagram

```mermaid
flowchart LR
    P[Student-safe policy obs<br/>60-dim<br/>no true_fault_state] --> S[LSTM Student<br/>LSTM(60,128)<br/>MLP 128-400-200-100-8]
    TP[Teacher privileged obs<br/>61-dim<br/>includes true_fault_state] --> T[Frozen T06 Teacher<br/>MLP 61-400-200-100-8]
    TC[T06 teacher checkpoint pointer] --> T
    T --> L[Behavior / Distillation Loss]
    S --> L
    S --> A[Student joint effort action<br/>8-dim]
```

## Deferred Items

- T08 residual
- Health token
- Uncertainty channel
- Safety shield / CBF
- Ablation runners
- P1/P2/P3 expansion
