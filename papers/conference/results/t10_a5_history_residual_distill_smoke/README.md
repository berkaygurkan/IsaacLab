# T10 A5 History Residual Distillation Smoke

## Purpose

Offline supervised smoke for the A5 history-student teacher-gap residual distillation scaffold.

## Dataset

- dataset: `papers/conference/datasets/t10_teacher_gap_p2_velocity_seed0/dataset.npz`
- frozen A2-history base checkpoint: `papers/conference/results/t10_a2_student_history_distill_full_h16_seed0/a2_student_history.pt`
- input: `student_obs` history window
- residual target: `teacher_action - a2_history_action`
- history_len: `16`
- expected input dim: `61`
- expected action dim: `8`
- expected residual dim: `8`

## Episode-Boundary Guardrail

History windows are rejected when `episode_id` changes inside the window or when `done` appears before the target timestep in the same window.

## Scope

- This is not paper-grade final evidence.
- This is only a supervised residual distillation scaffold.
- A7 residuals are not trained here.
- A0 actions, teacher observations, true fault state, health token, UQ, CBF, P3, and P4 remain inactive.
- Deployment-facing policies must not use `true_fault_state`.
