# T10 A5 History Residual P2 Eval

## Purpose

Isaac-side smoke evaluation for the offline-distilled A5 history residual policy under the P2 locked-joint fault.

## Policy

- policy_kind: `a5_history_residual`
- A2 checkpoint: `papers/conference/results/t10_a2_student_history_distill_full_h16_seed0/a2_student_history.pt`
- A5 checkpoint: `papers/conference/results/t10_a5_history_residual_distill_full_h16_seed0/a5_history_residual.pt`
- task: `Isaac-Ant-Velocity-Flat-v0`
- observation_dim: `61`
- action_dim: `8`
- history_len: `16`
- alpha: `1.0`
- residual_input_mode: `history_plus_base_action`
- composition: `final_action = a2_action + alpha * residual_action`
- history reset initialization: `repeat_first_observation`

## Fault

- fault_profile: `P2_locked_joint`
- target_joint: `front_left_foot`
- semantics: `simulation_joint_state_override_lock`
- fallback_used: `False`
- p2_fault_became_active: `True`
- simulation_override_applied: `True`

## Scope

- This is not paper-grade final evidence.
- No training, A0 action, teacher observations, true fault state, health token, UQ, CBF, P3, P4, or A7 are used.
