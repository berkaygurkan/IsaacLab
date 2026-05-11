# Fault Config Root (Conference Stage)

This directory mirrors the active conference-stage fault abstraction.

Current active fault families:
- joint_lock
- torque_scale
- free_swing_surrogate

Purpose:
- provide stable config naming for current-stage experiments
- support T04 config-first training organization
- avoid mixing present conference-stage faults with later-phase fault expansion

Deferred by default:
- multi-fault combinations
- diagnosis-aware fault configs
- uncertainty-conditioned fault configs
- safety-integrated fault configs
- later-phase deployment-first fault configuration
