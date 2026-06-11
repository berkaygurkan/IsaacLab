# T10 A2 Student P2 Eval Smoke

## Purpose

Small Isaac-side smoke evaluation for an offline-distilled A2 student under the P2 locked-joint fault.

## Policy

- policy_kind: `single_step`
- checkpoint: `papers/conference/results/t10_a2_student_distill_full_seed0/a2_student.pt`
- task: `Isaac-Ant-Velocity-Flat-v0`
- observation_dim: `61`
- action_dim: `8`

## Fault

- fault_profile: `P2_locked_joint`
- target_joint: `front_left_foot`
- semantics: `simulation_joint_state_override_lock`
- fallback_used: `False`
- p2_fault_became_active: `True`
- simulation_override_applied: `True`

## Scope

- This is not paper-grade final evidence.
- No training, residual correction, teacher observations, true fault state, health token, UQ, CBF, P3, or P4 are used.
