# T10 A5 History Residual Distillation Full

## Purpose

Offline supervised full run for the A5 history-student teacher-gap residual distillation scaffold.

## Dataset

- dataset: `papers/conference/datasets/t10_teacher_gap_p2_velocity_seed0/dataset.npz`
- samples: `124145`
- train samples: `111731`
- validation samples: `12414`

## Frozen Base

- A2 checkpoint: `papers/conference/results/t10_a2_student_history_distill_full_h16_seed0/a2_student_history.pt`
- A2 checkpoint guardrails ok: `True`

## Mapping

- input: `student_obs` history window with shape `[batch, 16, 61]`
- residual_input_mode: `history_plus_base_action`
- residual target: `teacher_action - a2_history_action`
- reconstruction: `a2_history_action + predicted_residual`

## Window Guardrails

- total candidate windows: `126080`
- valid windows: `124145`
- rejected cross-episode windows: `1935`
- rejected done-crossing windows: `1935`

## Metrics

- train_residual_mse: `6.756356274053859e-05`
- val_residual_mse: `0.0002660959560963647`
- train_reconstructed_action_mse: `6.75635619403402e-05`
- val_reconstructed_action_mse: `0.0002660959755270516`
- base_a2_action_mse_to_teacher: `0.0006113183917477727`
- no_nan_inf: `True`

## Guardrails

- This is not paper-grade final evidence.
- This is only a supervised residual distillation scaffold.
- A7 residuals are not trained here.
- A0 actions, teacher observations, true fault state, and health token are not used.
- Deployment-facing policies must not use `true_fault_state`.
