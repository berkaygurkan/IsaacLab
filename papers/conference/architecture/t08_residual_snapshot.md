# T08 Residual Architecture Snapshot

Scope: Conference-stage RLM1 stripped; health token OFF.

## Stage Identity

- Stage: `residual`
- Task id: `Isaac-Ant-Student-v0`
- Method variant: `rlm1_stripped`
- Fault: `none`

## Verified Interface

- Residual PPO observation group: `policy`
- Observation dimension: `(60,)`
- Action dimension: `8`
- Action type: joint effort action
- Residual PPO observation mapping: `obs_groups = {"actor": ["policy"], "critic": ["policy"]}`

The residual PPO actor and critic use only the student-safe `policy` group. The `teacher_policy` group exists in the environment with shape `(61,)` and includes `true_fault_state`, but it is not used by residual PPO actor or critic. The residual PPO path never receives `true_fault_state`.

## Residual PPO Policy Terms

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

## Network Architecture

- Frozen student model: `RNNModel`
- Frozen student recurrent core: `LSTM(60, 128)`
- Frozen student MLP head: `128 -> 400 -> 200 -> 100 -> 8`
- Frozen student output: `base_action`
- Residual PPO actor: `MLPModel`, `60 -> 400 -> 200 -> 100 -> 8`
- Residual PPO actor output: `delta_action`
- Residual PPO critic: `MLPModel`, `60 -> 400 -> 200 -> 100 -> 1`

## Action Composition

```python
final_action = base_action + residual_scale * tanh(delta_action)
```

T08-B smoke used `residual_scale = 0.1`, `final_action_clip = None`, and `reset_hidden_on_done = False`.

## Checkpoint Handoff

- Frozen T07 student checkpoint pointer: `checkpoints/rlm1_stripped/student/none/seed0/latest_checkpoint.yaml`
- Resolved frozen student checkpoint used in smoke: `logs/rsl_rl/student__rlm1_stripped__none/2026-05-28_00-18-04_student__rlm1_stripped__none__seed0/model_0.pt`
- Residual checkpoint pointer: `checkpoints/rlm1_stripped/residual/none/seed0/latest_checkpoint.yaml`
- TensorBoard/log path convention: `logs/rsl_rl/residual__rlm1_stripped__none/`

## Smoke Validation

- Smoke-test command: `TERM=xterm ./trainers/run_t08_residual_train.sh --headless --num_envs 8 --max_iterations 1`
- Smoke-test result: one residual PPO iteration completed
- Learning iteration: `0/1`
- Mean value loss: `0.0077`
- Mean surrogate loss: `-0.0138`
- Mean reward: `0.07`
- Residual checkpoint pointer was written

## Runtime Validation

- T08.6 short, long, and resume smokes passed.
- Residual diagnostics are available under `Residual/...`.
- Stable residual checkpoint pointer verified.
- See: `papers/conference/architecture/t08_residual_runtime_validation.md`

## Validation Sources

- Runtime log: residual PPO iteration, smoke metrics, observation dimensions, and checkpoint pointer write
- Config file: stage identity, method, fault, task, residual scale, and hidden reset setting
- Code inspection: residual wrapper, PPO observation mapping, actor/critic architectures, and action composition
- Checkpoint pointer: frozen student handoff and residual checkpoint convention

## Interpretation

T08 adds a minimal residual PPO layer on top of the frozen T07 student. The frozen student produces the base Ant joint effort action from the student-safe 60-dimensional `policy` observation, while the residual actor learns a bounded delta action from the same `policy` input. This preserves the conference-stage boundary: no direct `true_fault_state`, no health token, no uncertainty channel, and no safety shield.

## Flow Diagram

```mermaid
flowchart LR
    P[Student-safe policy obs<br/>60-dim<br/>no true_fault_state] --> FS[Frozen T07 Student<br/>LSTM(60,128)<br/>MLP 128-400-200-100-8]
    P --> RA[Residual PPO Actor<br/>MLP 60-400-200-100-8]
    FS --> BA[base_action<br/>8-dim]
    RA --> DA[delta_action<br/>8-dim]
    DA --> TD[tanh(delta_action)]
    BA --> C[Action composition<br/>base_action + 0.1 * tanh(delta_action)]
    TD --> C
    C --> FA[final joint effort action<br/>8-dim]
    P --> RC[Residual PPO Critic<br/>MLP 60-400-200-100-1]
```

## Deferred Items

- T09 ablation runners
- Health token
- Uncertainty channel
- Safety shield / CBF
- P1/P2/P3 expansion
- Residual tuning beyond smoke
- Faulted residual curriculum
