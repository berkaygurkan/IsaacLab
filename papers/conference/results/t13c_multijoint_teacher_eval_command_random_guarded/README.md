# T13C Multi-Joint A1-F Teacher Evaluation

Guarded evaluator scaffold for the T13-B v2b privileged multi-joint P2 teacher.

This is evaluation-only infrastructure. It does not train, modify checkpoints, modify task configs,
or change P2 wrapper semantics.

## Policy

- task: `Isaac-Ant-Teacher-Velocity-MultiJointP2-Flat-v0`
- checkpoint: `logs/rsl_rl/teacher_p2_multijoint_velocity_curriculum_v2b__rlm1_stripped__p2_multi_joint_random/2026-06-15_18-16-03_a1f_multijoint_velocity_p2_v2b_curriculum_curriculum_s2_realistic_transition_random_p2__seed0/model_9997.pt`
- teacher observation: `77 = 61 base + 8 selected-joint one-hot + 8 q_lock vector`
- selected locked joint: one random actuated joint per env/episode
- semantics: direct `simulation_joint_state_override_lock`
- fallback/surrogate: disabled
- A7 terminology in this pipeline: A0-anchored residual variant / healthy-base residual variant

## Default Protocols

- `late_random`: onset U(250,700)
- `realistic_random`: onset U(120,700)
- `stress_random`: onset U(30,700), opt-in only

## Velocity Modes

- `fixed_vx_1p0`: forces command tensor to vx = 1.0, vy = 0, yaw = 0 before policy inference and after each env step
- `command_random`: leaves the command-conditioned task range active, vx in [0.2, 1.5]

## Command-Mode Note

T13C fixed-vx outputs generated before the command-enforcement patch were command-unforced and
must not be used as fixed-vx evidence. They may be kept only as preliminary command-conditioned
or random-command evidence.

## Outputs

- per-run: `summary.json`, `rollout_metrics.csv`, `velocity_timeseries.csv`, `per_joint_metrics.csv`, `command.txt`
- aggregate: `tables/metrics_summary.csv`, `tables/metrics_summary.md`, `tables/per_joint_metrics.csv`
