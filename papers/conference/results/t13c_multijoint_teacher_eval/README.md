# T13C Multi-Joint A1-F Teacher Evaluation

Guarded evaluation scaffold for the completed T13-B v2b privileged multi-joint
teacher.

This is evaluation-only infrastructure. It must not train, modify checkpoints,
modify task configs, or change P2 wrapper semantics.

## Candidate

- task: `Isaac-Ant-Teacher-Velocity-MultiJointP2-Flat-v0`
- checkpoint: `logs/rsl_rl/teacher_p2_multijoint_velocity_curriculum_v2b__rlm1_stripped__p2_multi_joint_random/2026-06-15_18-16-03_a1f_multijoint_velocity_p2_v2b_curriculum_curriculum_s2_realistic_transition_random_p2__seed0/model_9997.pt`
- teacher observation: `77 = 61 base + 8 selected-joint one-hot + 8 q_lock vector`
- P2 mode: `target_joint_mode=random_per_env`
- locked joint count: exactly one selected actuated joint per env/episode
- semantics: direct `simulation_joint_state_override_lock`
- fallback/surrogate: disabled

## Default Evaluation

The default evaluator plan runs:

- `late_random`: onset `U(250,700)`
- `realistic_random`: onset `U(120,700)`
- velocity mode `fixed_vx_1p0`
- `num_envs=512`
- `num_steps=1000`
- `seed=0`
- `device=cuda`
- headless Isaac

`stress_random`, onset `U(30,700)`, is opt-in only.

## Velocity Modes

- `fixed_vx_1p0`: forces the command tensor to `vx_cmd=1.0`, `vy_cmd=0`, `yaw_cmd=0` before policy inference and after each env step.
- `command_random`: leaves the command-conditioned task range active, with `vx_cmd` in `[0.2, 1.5]`.

## Command-Mode Note

T13C `fixed_vx_1p0` outputs generated before the command-enforcement patch were
command-unforced and should not be used as fixed-vx evidence. They may be kept
only as preliminary command-conditioned/random-command evidence.

## Outputs

Each protocol/velocity-mode run writes:

- `summary.json`
- `rollout_metrics.csv`
- `velocity_timeseries.csv`
- `per_joint_metrics.csv`
- `command.txt`
- `README.md`
- `terminal.log` when launched through the orchestration mode

Aggregates are written under:

- `tables/metrics_summary.csv`
- `tables/metrics_summary.md`
- `tables/per_joint_metrics.csv`

## Guardrails

- no training
- no checkpoint modification
- no task config modification
- no P2 semantics change
- no fallback
- no PD surrogate
- privileged A1-F teacher is reference-only
- deployment-facing students remain fault-descriptor-free
- A7 terminology remains `A0-anchored residual variant` or `healthy-base residual variant`
