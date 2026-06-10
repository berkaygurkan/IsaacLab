# T09 A1-F P2 Runtime Hook Validation

Scope: P2 runtime hook wiring and preflight plan for A1-F teacher training; no
full training, checkpoint freeze, pointer edit, evaluation, observed manifest
update, or paper-grade claim.

## Implementation Choice

T09-R2h kept the upstream import-path fix and upgraded the P2 runtime hook
toward a simulation-level joint-state override through repo-owned trainer code.
T09-R2i adds per-env random onset for the final long A1-F teacher:

```text
trainers/p2_joint_lock_training_wrapper.py
```

`trainers/rsl_rl_train.py` supports `--enable_p2_joint_lock`. When that flag is
set, it routes the upstream Isaac Lab RSL-RL training script through the P2
wrapper shim instead of calling the upstream training script directly. The shim
adds `scripts/reinforcement_learning/rsl_rl` to `sys.path` before
`runpy.run_path(...)`, then verifies that sibling module `cli_args.py` is
resolvable.

The A1-F helper passes the hook flags:

```text
--enable_p2_joint_lock
--p2_fault_config configs/fault/joint_lock/p2_locked_joint.yaml
--p2_target_joint front_left_foot
--p2_fault_onset_mode random_uniform
--p2_fault_onset_step_min 30
--p2_fault_onset_step_max 150
--p2_fault_onset_step 50
--p2_requested_semantics simulation_joint_state_override_lock
--p2_velocity_override 0.0
--p2_kp 4.0
--p2_kd 0.4
--p2_action_clip 1.0
```

## True Joint Lock Versus Surrogate

The previous T09-R2f implementation was:

```text
action_override_zero_effort_surrogate
```

The desired conference semantics are:

```text
position_hold_joint_lock
```

The current Ant teacher task exposes an effort-action interface, but the robot
articulation is reachable from the action term's `_asset`. Local Isaac Lab
articulations expose:

```text
robot.write_joint_state_to_sim(position, velocity, joint_ids=[...], env_ids=...)
```

T09-R2h uses that API path to request:

```text
simulation_joint_state_override_lock
```

At onset, the wrapper captures the selected joint's current position per env as
`q_lock`. After each `env.step`, while the lock is active, it writes the target
joint position back to `q_lock` and writes target joint velocity to
`velocity_override`, default `0.0`.

## Fixed-Onset Validation Versus Random-Onset Teacher

The completed fixed-onset step-50 A1-F run validates the runtime hook path:

```text
target_joint: front_left_foot
fault_onset_step: 50
actual_semantics: simulation_joint_state_override_lock
P2/fault_applied: 1.0000
P2/simulation_override_applied: 1.0000
P2/fallback_used: 0.0000
```

Treat that run as a validation run, not the final random-onset canonical
teacher. The intended long A1-F teacher uses one training run with per-env
random onset:

```text
fault_onset_mode: random_uniform
fault_onset_step_min: 30
fault_onset_step_max: 150
```

The target joint remains fixed as `front_left_foot`. Randomizing onset varies
gait phase and captured locked angle, avoiding separate teacher training for
each onset. Later controlled evaluation can use fixed onset points or a fixed
onset grid.

The PD effort mode is retained only as explicit fallback:

```text
pd_position_hold_surrogate
```

Fallback is disabled by default for conference A1-F. If simulation override is
unavailable and fallback is not explicitly allowed, the wrapper fails fast. The
old zero-effort action mask is not acceptable as the main conference P2 fault.

## Target Mapping

Default runtime target:

```text
fault_profile: P2_locked_joint
target_joint: front_left_foot
fault_onset_mode: random_uniform
fault_onset_step_min: 30
fault_onset_step_max: 150
fixed_onset_validation_step: 50
expected_action_dim: 8
```

The wrapper fails fast unless `front_left_foot` resolves to exactly one action
index in the Ant action manager. Preflight also verifies that the robot
articulation object is found, joint state is readable, `write_joint_state_to_sim`
exists, per-env random onset steps are sampled, `q_lock` is captured when each
env reaches its own onset, post-step override runs, and the target joint
velocity is forced near the requested override value.

## Dry-Run Command

Use dry-run first:

```bash
bash trainers/run_t09_train_a1f_p2_teacher_canonical.sh --dry_run
```

Dry-run prints the planned P2 hook settings and delegated command without
launching Isaac Sim or training.

## Preflight Command

Runtime preflight, still with no training:

```bash
python evaluators/preflight_t09_p2_joint_lock.py --execute_preflight --headless
```

The preflight constructs the teacher env, resolves the target joint/action
mapping, attaches the P2 wrapper, resets the env, runs a tiny step smoke through
the onset window, reports whether lock logic activated, prints `q_lock`, reports
actual semantics and fallback status, and closes the env/app.

## Training Guardrail

Full A1-F training remains blocked unless the user explicitly acknowledges a
passed P2 preflight:

```bash
bash trainers/run_t09_train_a1f_p2_teacher_canonical.sh --allow_full_training --p2_preflight_passed
```

A tiny runtime training smoke is available only with explicit flags:

```bash
bash trainers/run_t09_train_a1f_p2_teacher_canonical.sh --execute_one_step_smoke --p2_preflight_passed
```

Both paths pass `--skip_checkpoint_pointer`, so no stable pointer is overwritten.

## Guardrails

- No full training was run by this wiring step.
- No checkpoint was frozen.
- No checkpoint pointer was modified.
- No A2/A5/A7 training was added.
- P4 remains deferred to advisor-demo / thesis-extension artifacts.
- Health token OFF, UQ inactive, and CBF inactive remain fixed.
- No Isaac Lab core, RSL-RL core, or task registration edit was made.
