# Fault Config Root (Conference Stage)

This directory mirrors the active conference-stage fault abstraction.

Current active fault families:
- joint_lock
- torque_scale
- free_swing_surrogate

Purpose:
- provide stable config naming for current-stage experiments
- support T04 config-first training organization
- avoid mixing present conference-stage faults with later-phase fault expansion

Deferred by default:
- multi-fault combinations
- diagnosis-aware fault configs
- uncertainty-conditioned fault configs
- safety-integrated fault configs
- later-phase deployment-first fault configuration

## T13 Multi-Joint P2 Infrastructure

The current conference fault remains the single-joint P2 lock:

- config: `configs/fault/joint_lock/p2_locked_joint.yaml`
- mode: `target_joint_mode=single`
- target joint: `front_left_foot`
- semantics: `simulation_joint_state_override_lock`
- fallback: disabled

T13 adds an opt-in multi-joint P2 mode:

- config: `configs/fault/joint_lock/p2_multi_joint_random.yaml`
- mode: `target_joint_mode=random_per_env`
- one active locked joint per env/episode
- supported joints are resolved from the Ant joint-effort action term rather
  than hardcoded in the fault config
- onset behavior remains `random_uniform` unless overridden by the caller
- semantics are direct simulation-state override: capture `q_lock` at onset,
  then enforce the selected joint's `q = q_lock` and `qd = 0`
- fallback is disabled and the run should fail fast if the simulation joint
  state write API is unavailable
- legacy PD parameters such as `p2_kp` and `p2_kd` may still appear in CLIs for
  single-joint compatibility, but are unused when actual semantics are
  `simulation_joint_state_override_lock`

The multi-joint teacher task is:

```text
Isaac-Ant-Teacher-Velocity-MultiJointP2-Flat-v0
```

It receives a 77-D privileged policy observation:

- 61-D base student-safe velocity observation
- 8-D `p2_fault_joint_one_hot` vector over the resolved actuated joints
- 8-D `p2_fault_q_lock_vector` with zeros before capture and captured `q_lock`
  at the selected joint index after onset

Deployment-facing student observations remain 61-D and do not include the
one-hot vector, the q-lock vector, `true_fault_state`, or a health token. No UQ
or CBF channel is added by this infrastructure patch.

A7 should be described as the A0-anchored residual variant, or healthy-base
residual variant, in conference-stage docs.
