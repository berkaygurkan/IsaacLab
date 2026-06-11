# T10 A2 Student Distillation Smoke

## Purpose

Small offline supervised smoke for the A2 student behavior-distillation scaffold.

## Dataset

`papers/conference/datasets/t10_teacher_gap_p2_velocity_seed0/dataset.npz`

## Input / Target

- input: `student_obs`
- target: `teacher_action`
- expected input dim: `61`
- expected action dim: `8`

## Guardrails

- This is not paper-grade final evidence.
- This is only a supervised behavior-distillation scaffold.
- A5/A7 residuals are not trained here.
- Deployment-facing policies must not use `true_fault_state`.
- `teacher_obs` is not used as student input.
- Health token, UQ, CBF, P3, and P4 remain inactive.
