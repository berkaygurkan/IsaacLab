# T10 A2 History Student Distillation Full

## Purpose

Offline supervised behavior-distillation full run for the A2 deployment-facing history student scaffold.

## Dataset

- dataset: `papers/conference/datasets/t10_teacher_gap_p2_velocity_seed0/dataset.npz`
- samples: `124145`
- train samples: `111731`
- validation samples: `12414`

## Mapping

- input: `student_obs` history window with shape `[batch, 16, 61]`
- target: `teacher_action` at the current timestep
- `teacher_obs` is not used as model input.
- `true_fault_state` is not used by the student.

## Window Guardrails

- total candidate windows: `126080`
- valid windows: `124145`
- rejected cross-episode windows: `1935`
- rejected done-crossing windows: `1935`

## Metrics

- train_mse: `0.0005922338192678718`
- val_mse: `0.0007830870390149616`
- train_mae: `0.0194718215738488`
- val_mae: `0.020299796687821097`
- no_nan_inf: `True`

## Guardrails

- This is not paper-grade final evidence.
- This is only a supervised history-student distillation scaffold.
- A5/A7 residuals are not trained here.
- Health token, UQ, CBF, P3, and P4 remain inactive.
- Deployment-facing policies must not use `true_fault_state`.
