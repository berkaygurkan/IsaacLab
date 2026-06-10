# T10 Teacher-Gap P2 Velocity Dataset

## Purpose

Candidate-level dataset scaffold for aligned selected A1-F teacher actions, A0 healthy PPO actions, and teacher-gap residual targets under random P2 velocity evaluation.

## Downstream Usage

- A2 behavior distillation from `student_obs` to `teacher_action`
- A7 teacher-gap residual using `a7_residual_target = teacher_action - a0_action`
- Later A5 residual after an A2 distilled student exists

## Guardrail

Deployment policies must not use `true_fault_state`. `teacher_obs` may contain `true_fault_state` for teacher inference/audit only; deployment-facing A2/A5/A7 inputs must use `student_obs`.

## Selected Teacher Reason

The onset curriculum teacher is selected because it has the best post-fault velocity tracking among the evaluated A1-F P2 velocity teacher candidates.

## Files

- dataset: `papers/conference/datasets/t10_teacher_gap_p2_velocity_seed0_smoke/dataset.npz`
- metadata: `papers/conference/datasets/t10_teacher_gap_p2_velocity_seed0_smoke/metadata.json`

## Dataset

- tag: `t10_teacher_gap_p2_velocity_seed0_smoke`
- num_envs: `16`
- num_steps: `200`
- student_obs shape: `[200, 16, 61]`
- teacher_obs shape: `[200, 16, 62]`
- teacher_action shape: `[200, 16, 8]`
- a0_action shape: `[200, 16, 8]`
- a7_residual_target shape: `[200, 16, 8]`

## Fault / P2 Checks

- fallback used: `False`
- P2 fault became active: `True`
- P2 simulation override applied after onset: `True`
- P2 fault active final mean: `1.0`
- no NaN/Inf: `True`

This dataset is candidate-level seed0 evidence and is not paper-grade final.
