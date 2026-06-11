# Selected Student Policy

Status: candidate-level selection record for the conference-stage RLM1 stripped implementation. This is not final paper-grade evidence.

## Selected Policy

| field | value |
| --- | --- |
| selected policy | A2-history H16 |
| checkpoint | `papers/conference/results/t10_a2_student_history_distill_full_h16_seed0/a2_student_history.pt` |
| task | `Isaac-Ant-Velocity-Flat-v0` |
| observation dim | 61 |
| history_len | 16 |
| action dim | 8 |
| deployment-facing | yes |

## Input And Architecture Guardrails

- Uses deployment-safe student observations only.
- Health token is OFF.
- No `teacher_obs` input.
- No `true_fault_state` input.
- No residual correction is active in the selected deployment candidate.
- No A5 or A7 residual action is composed with the selected policy.

## Selection Rationale

A2-history H16 is selected as the main deployment-facing candidate because it is the best closed-loop candidate among the deployment-facing policies evaluated so far. In the stored A2 P2 evaluations, the history student outperformed the single-step student under both random-onset and settled-onset P2 protocols while preserving student-safe inputs.

This record does not update checkpoint pointers and does not claim final paper-grade status. A final unified rerun is still required before paper tables or plots are treated as final.
