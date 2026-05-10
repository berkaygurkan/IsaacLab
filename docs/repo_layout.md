# Repository Layout (Conference Stage)

Current active scope:
- Conference paper target
- RLM1 stripped
- teacher-student active
- residual active
- health token OFF

This layout is intentionally limited to T01-T10.
Anything related to RLM1-full, uncertainty channel, safety shield, or later publication phases is deferred unless explicitly activated.

## Top-level project folders

- `configs/`: configuration-first project structure for env, fault, method, train, and eval settings.
- `envs/`: project-level environment/task registration notes and wrappers for conference-stage work.
- `faults/`: bounded fault abstraction for current conference-stage scenarios.
- `methods/rlm1/`: current method-family-specific structure for RLM1 stripped only.
- `trainers/`: training entry organization for baseline, teacher, student, and residual stages.
- `evaluators/`: evaluation organization for conference metrics and bounded comparisons.
- `logs/`: experiment logs and run outputs.
- `checkpoints/`: saved model checkpoints and stage outputs.
- `deployment/`: reserved for later deployment-related work; not active by default in conference-stage T01.
- `papers/conference/`: conference-facing artifacts, summaries, figures, and tables.

## Scope guardrails

Do not treat the following as active by default:
- health token
- uncertainty channel
- safety shield / CBF augmentation
- P1 / P2 / P3 expansion
- full thesis-wide repository breadth
