# Conference Architecture Snapshot Standard

This directory holds paper-facing architecture snapshots for completed conference-stage implementation tasks. Each snapshot should verify what the policy actually receives, what it outputs, and how the stage fits into the RLM1 stripped pipeline.

## Required Snapshot Fields

Every completed stage snapshot must include:

- Stage identity
- Task id
- Method and variant
- Fault setting
- Observation group and terms
- Observation dimension
- Action dimension
- Actor architecture
- Critic architecture
- Smoke-test command
- TensorBoard/log path
- Stable checkpoint pointer path
- Mermaid flow diagram
- Paper-facing interpretation
- Deferred items
- Validation source

## Validation Source

Each snapshot must explicitly state where its facts came from. Valid sources include:

- Runtime log
- Config file
- Code inspection
- Checkpoint pointer
- Manually supplied verified fact

Use multiple sources when dimensions, observation terms, or checkpoint paths come from different places.

## Current Defaults

For the current conference-stage `RLM1 stripped` method:

- Health token is OFF by default.
- Uncertainty channel is inactive by default.
- Safety shield / CBF is inactive by default.
- Later stages must remain marked as deferred until they are implemented and smoke-tested.

Do not mark student, residual, uncertainty, health-token, or safety-shield components as active before their corresponding implementation and smoke test are complete.
