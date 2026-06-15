# T10 A7 A0 Residual P2 Eval

## Purpose

Isaac-side smoke evaluation for the offline-distilled A7 A0-residual policy under the P2 locked-joint fault.

## Policy

- policy_kind: `a7_a0_residual`
- A0 checkpoint: `logs/rsl_rl/healthy_baseline_velocity__rlm1_stripped__none/2026-06-09_23-58-25_a0_velocity_candidate1000__seed0/model_999.pt`
- A7 checkpoint: `papers/conference/results/t10_a7_a0_residual_distill_full_h16_seed0/a7_a0_residual.pt`
- task: `Isaac-Ant-Velocity-Flat-v0`
- observation_dim: `61`
- action_dim: `8`
- history_len: `16`
- alpha: `0.05`
- residual_input_mode: `history_plus_a0_action`
- composition: `final_action = a0_action + alpha * residual_action`
- history reset initialization: `repeat_first_observation`

## Fault

- fault_profile: `P2_locked_joint`
- target_joint: `front_right_foot`
- semantics: `simulation_joint_state_override_lock`
- fallback_used: `False`
- p2_fault_became_active: `True`
- simulation_override_applied: `True`

## Scope

- This is not paper-grade final evidence.
- No training, teacher observations, true fault state, health token, A2 action, A5 action, UQ, CBF, P3, or P4 are used.
