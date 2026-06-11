# T10 A7 A0 Residual Distillation Smoke

## Purpose

Small offline supervised smoke run for the A7 frozen-healthy-PPO teacher-gap residual distillation scaffold.

## Dataset

- dataset: `papers/conference/datasets/t10_teacher_gap_p2_velocity_seed0/dataset.npz`
- input history source: `student_obs`
- base action source: dataset-provided `a0_action`
- preferred target source: `a7_residual_target`

## Frozen A0 Base Concept

A7 composes a frozen healthy PPO base policy with a learned residual adaptor. This smoke trainer does not load, modify, or train the A0 checkpoint; it uses the dataset-provided A0 actions as the supervised base action.

## Residual Mapping

- residual target: `teacher_action - a0_action`
- later action composition: `a_final = a0_action + alpha * residual_action`
- residual_input_mode: `history_plus_a0_action`

## Expected Dimensions

- student_obs input dim: `61`
- history_len: `16`
- a0_action dim: `8`
- teacher_action dim: `8`
- residual dim: `8`

## Episode Guardrail

History windows are rejected if episode ids change inside the window or if `done` occurs before the target timestep.

## Scope

- This is not paper-grade final evidence.
- This is only a supervised residual distillation scaffold.
- A2 and A5 residuals are not trained here.
- Isaac-side A7 evaluation is not run here.
- Deployment-facing policies must not use `true_fault_state`.
