# T09 A1-F P2 Runtime Hook Validation

Scope: P2 runtime hook wiring and preflight plan for A1-F teacher training; no
full training, checkpoint freeze, pointer edit, evaluation, observed manifest
update, or paper-grade claim.

## Implementation Choice

T09-R2f wires the P2 runtime hook through repo-owned trainer code:

```text
trainers/p2_joint_lock_training_wrapper.py
```

`trainers/rsl_rl_train.py` now supports `--enable_p2_joint_lock`. When that flag
is set, it routes the upstream Isaac Lab RSL-RL training script through the P2
wrapper shim instead of calling the upstream training script directly.

The A1-F helper passes the hook flags:

```text
--enable_p2_joint_lock
--p2_fault_config configs/fault/joint_lock/p2_locked_joint.yaml
--p2_target_joint front_left_foot
--p2_fault_onset_step 50
```

## True Joint Lock Versus Surrogate

The current implementation is a surrogate:

```text
semantics: action_override_zero_effort_surrogate
```

After `fault_onset_step`, the wrapper overrides the selected action dimension
to `0.0`. For the current Ant effort-action interface, this masks the target
joint effort command. It does not physically hold the joint at its current
position and should not be described as a true mechanical position lock.

This surrogate is acceptable for T09-R2f hook validation, but future paper text
must describe it accurately unless a true position-hold joint-lock mechanism is
implemented later.

## Target Mapping

Default runtime target:

```text
fault_profile: P2_locked_joint
target_joint: front_left_foot
fault_onset_step: 50
expected_action_dim: 8
```

The wrapper fails fast unless `front_left_foot` resolves to exactly one action
index in the Ant action manager.

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
mapping, attaches the P2 wrapper, resets the env, runs a tiny zero-action step
smoke through the onset window, reports whether lock logic activated, and closes
the env/app.

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
