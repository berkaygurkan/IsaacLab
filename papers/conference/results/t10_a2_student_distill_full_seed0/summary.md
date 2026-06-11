# T10 A2 Student Distillation Smoke

## Purpose

Offline supervised behavior-distillation smoke for the A2 deployment-facing student scaffold.

## Dataset

- dataset: `papers/conference/datasets/t10_teacher_gap_p2_velocity_seed0/dataset.npz`
- samples: `128000`
- train samples: `115200`
- validation samples: `12800`

## Mapping

- input: `student_obs`
- target: `teacher_action`
- `teacher_obs` is not used as model input.
- `true_fault_state` is not used by the student.

## Metrics

- train_mse: `0.00017944459150183117`
- val_mse: `0.00021200419985689222`
- train_mae: `0.007946273287137349`
- val_mae: `0.008094012252986432`
- no_nan_inf: `True`

## Guardrails

- This is not paper-grade final evidence.
- This is only a supervised behavior-distillation scaffold.
- A5/A7 residuals are not trained here.
- Health token, UQ, CBF, P3, and P4 remain inactive.
- Deployment-facing policies must not use `true_fault_state`.
