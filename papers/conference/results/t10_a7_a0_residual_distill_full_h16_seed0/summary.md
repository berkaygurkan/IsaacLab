# T10 A7 A0 Residual Distillation Full

## Purpose

Offline supervised full run for the A7 frozen-healthy-PPO teacher-gap residual scaffold.

## Dataset

- dataset: `papers/conference/datasets/t10_teacher_gap_p2_velocity_seed0/dataset.npz`
- samples: `124145`
- train samples: `111731`
- validation samples: `12414`

## Frozen Base Concept

- base policy: `A0_healthy_velocity`
- base checkpoint: `logs/rsl_rl/healthy_baseline_velocity__rlm1_stripped__none/2026-06-09_23-58-25_a0_velocity_candidate1000__seed0/model_999.pt`
- offline base action source: dataset-provided `a0_action`

## Mapping

- input: `student_obs` history window with shape `[batch, 16, 61]` plus `a0_action`
- residual_input_mode: `history_plus_a0_action`
- residual target: `teacher_action - a0_action`
- reconstruction: `a0_action + predicted_residual`

## Target Consistency

- a7_residual_target present: `True`
- a7 target matches teacher minus A0: `True`
- max absolute consistency error: `0.0`

## Window Guardrails

- total candidate windows: `126080`
- valid windows: `124145`
- rejected cross-episode windows: `1935`
- rejected done-crossing windows: `1935`

## Metrics

- train_residual_mse: `0.00022300752081944736`
- val_residual_mse: `0.00048554041462665177`
- train_reconstructed_action_mse: `0.00022300752055271457`
- val_reconstructed_action_mse: `0.0004855404143265639`
- base_a0_action_mse_to_teacher: `0.38681110739707947`
- no_nan_inf: `True`

## Guardrails

- This is not paper-grade final evidence.
- This is only a supervised residual distillation scaffold.
- A2/A5 residuals are not trained here.
- Teacher observations, true fault state, health token, A2 action, and A5 action are not used.
- Deployment-facing policies must not use `true_fault_state`.
