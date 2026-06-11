# T10 Candidate Selection Summary

Documentation-only selection pass for the conference-stage RLM1 stripped implementation. No training, Isaac Sim run, checkpoint edit, task config edit, P2 wrapper edit, or checkpoint pointer update was performed.

## Candidate Roles

| candidate | role | deployment-facing | current decision | evidence note |
| --- | --- | :---: | --- | --- |
| A0 healthy PPO baseline | healthy-policy baseline under P2 | yes, as baseline only | retain for comparison | P2 random reference error was high relative to student and teacher candidates. |
| A1-F privileged P2 teacher reference | privileged upper-bound/reference and distillation source | no | reference only | Uses privileged fault information; selected curriculum teacher is documented in `papers/conference/results/t10_teacher_selection_summary.md`. |
| A2 single-step student | deployment-safe teacher-student baseline | yes | retain as student ablation | Random P2 post-fault vx error: `0.3028785612447281`; settled P2: `0.3414071450347397`. |
| A2-history H16 | deployment-safe history student | yes | selected main candidate | Random P2 post-fault vx error: `0.16029782813878513`; settled P2: `0.16393009481816082`. |
| A5 residual over A2 history | residual ablation over the strong history student | yes, ablation only | retain as negative residual ablation | Offline teacher-action reconstruction improved, but closed-loop alpha sweep did not improve over alpha=0 base. |
| A7 residual over A0 healthy PPO | residual ablation over frozen healthy PPO | yes, ablation only | retain as negative residual ablation | Offline A0-to-teacher residual reconstruction improved strongly, but closed-loop tiny-alpha diagnostic did not improve over alpha=0 base. |

## Decision

A2-history H16 is selected as the main deployment-facing candidate:

```text
papers/conference/results/t10_a2_student_history_distill_full_h16_seed0/a2_student_history.pt
```

The selected policy uses `history_len=16`, observation dim `61`, action dim `8`, and deployment-safe student observations only.

## Residual Interpretation

A5 residual over A2 history improved offline teacher-action reconstruction, but the Isaac-side alpha sweep did not improve closed-loop behavior over the alpha=0 A2-history base. In the random P2 sweep, the best alpha by post-fault vx error was `0`; in the settled P2 sweep, the best alpha was also `0`.

A7 residual over frozen A0 improved offline A0-to-teacher residual reconstruction strongly, but the tiny-alpha diagnostic did not improve over alpha=0 base. Alpha `0.0` reproduced the frozen A0 action path in the A7 evaluator, while positive tiny alphas did not reduce post-fault vx error.

The working interpretation is that residual imitation can improve offline action matching without improving closed-loop behavior. In the current single-joint P2 setting, the history-based student appears to handle the fault sufficiently well that residual action-gap imitation does not add useful closed-loop correction.

## Guardrails

- Health token remains OFF.
- No UQ is active.
- No CBF is active.
- Deployment-facing policies do not use `teacher_obs`.
- Deployment-facing policies do not use `true_fault_state`.
- A1-F remains privileged/reference only, not deployment-facing.
- A5 and A7 artifacts are retained as residual ablations.

## Status

These are candidate-level results. CPU/GPU device differences and protocol details may need a unified final rerun before paper-grade claims. This document is not final statistical evidence.

## Next Step

Run the final comparison rerun and generate plots/tables under one unified protocol.
