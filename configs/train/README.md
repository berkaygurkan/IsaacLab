# Training Config Root (Conference Stage)

Purpose:
- define stable training-facing configuration structure for conference-stage runs

Current reproducibility expectations:
- fixed seed usage
- explicit stage naming
- stable run naming
- explicit log path naming
- explicit checkpoint path naming

Recommended current stage labels:
- healthy_baseline
- teacher
- student
- residual

Deferred by default:
- later-phase curriculum expansion
- uncertainty-aware training configuration
- safety-augmented training configuration
