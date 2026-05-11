# Fault Abstraction (Conference Stage)

Current active scope:
- conference-stage only
- RLM1 stripped
- teacher-student active
- residual active
- health token OFF

## Active fault families

This directory currently supports only the bounded conference-stage fault abstraction:

- `joint_lock/`:
  P2-like locked-joint scenarios.

- `torque_scale/`:
  P4-like actuator weakening / torque degradation scenarios.

- `free_swing_surrogate/`:
  limited P5-like surrogate fault structure, only if needed for current-stage clarity.

## Not active by default

The following are deferred unless explicitly requested:
- full multi-fault combinations
- diagnosis / FDI modules
- health-token-aware fault interfaces
- uncertainty-conditioned fault interfaces
- safety-shield-integrated fault handling
- later-phase deployment-first fault pipelines
