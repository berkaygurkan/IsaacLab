# T10 A2 History Student Distillation Smoke

## Purpose

Offline supervised smoke for the A2 deployment-facing history-student distillation scaffold.

## Dataset

- dataset: `papers/conference/datasets/t10_teacher_gap_p2_velocity_seed0/dataset.npz`
- input: `student_obs` history window
- target: `teacher_action` at the current timestep
- history_len: `16`
- expected input dim: `61`
- expected action dim: `8`

## Episode-Boundary Guardrail

History windows are rejected when `episode_id` changes inside the window or when `done` appears before the target timestep in the same window.

## Scope

- This is not paper-grade final evidence.
- This is only a supervised history-student distillation scaffold.
- A5/A7 residuals are not trained here.
- Health token, UQ, CBF, P3, and P4 remain inactive.
- Deployment-facing policies must not use `true_fault_state`.
