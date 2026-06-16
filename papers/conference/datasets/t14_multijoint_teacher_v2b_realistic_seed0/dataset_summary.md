# T14 Multi-Joint Teacher Dataset

Dataset scaffold for selected T13-B v2b A1-F multi-joint teacher rollouts.

## Selected Teacher

- label: `A1-F multi-joint teacher v2b, valid but foot-limited`
- checkpoint: `logs/rsl_rl/teacher_p2_multijoint_velocity_curriculum_v2b__rlm1_stripped__p2_multi_joint_random/2026-06-15_18-16-03_a1f_multijoint_velocity_p2_v2b_curriculum_curriculum_s2_realistic_transition_random_p2__seed0/model_9997.pt`
- selection note: `papers/conference/results/t13_multi_joint_teacher_selection/README.md`

## Schema

- `student_obs`: 61-D deployment-facing observation
- `teacher_obs`: 77-D privileged teacher observation
- `teacher_action`: 8-D teacher action
- `selected_fault_joint_index`: selected locked-joint action index
- `selected_fault_joint_one_hot`: 8-D selected-joint vector
- `q_lock_vector`: 8-D q-lock vector
- `p2_fault_active`, `fault_onset_step`, `vx_cmd`, `velocity_x`, `vx_error`
- `yaw_rate`, `yaw_error`, `done`, `episode_id`, `timestep`
- `protocol_label`, `velocity_mode_label`

Optional fields `a0_action` and `a7_residual_target` are written only when
`--a0_checkpoint` is explicitly provided.

## Semantics

- one selected locked joint per env/episode
- selected joint random over all 8 Ant actuated joints
- q_lock captured from the selected joint's current position at onset
- direct `simulation_joint_state_override_lock`
- enforce selected joint `q = q_lock` and `qd = 0` after onset
- fallback disabled
- PD surrogate disabled
- health token OFF

This dataset is candidate-level evidence for downstream A2/A2-history/A5/A7
scaffolding and is not paper-grade final.
