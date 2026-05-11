# Reproducibility Rules (Conference Stage)

Current default scope:
- conference-stage only
- RLM1 stripped only

## Required identifiers for every run

Each run should have explicit values for:
- stage
- method variant
- fault family
- seed
- run name

## Current recommended stage names

- healthy_baseline
- teacher
- student
- residual

## Current required method naming

- rlm1_stripped

Do not use later-phase names by default.

## Current recommended fault naming

- none
- joint_lock
- torque_scale
- free_swing_surrogate

## Output expectations

Each run should map clearly to:
- log directory
- checkpoint directory
- evaluation summary
- paper-facing artifact reference when applicable

## Minimal naming discipline

Recommended run-name pattern:

<stage>__<method>__<fault>__seed<id>

Example:

teacher__rlm1_stripped__joint_lock__seed42

## Deferred by default

Do not expand this file by default into:
- full thesis-wide naming registry
- deployment experiment tracking
- uncertainty/safety naming branches
