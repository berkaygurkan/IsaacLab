# T10 A0-Vel Healthy PPO Smoke Plan

Scope: T10-IL-03 smoke launcher only. This is not paper-grade training and does not freeze checkpoints, edit checkpoint pointers, update observed-result manifests, add P2, add residuals, or modify RSL-RL core.

## Launcher

```text
trainers/run_t10_train_a0_velocity_smoke.sh
```

The launcher delegates to:

```text
trainers/rsl_rl_train.py
```

It always passes `--skip_checkpoint_pointer` so stable A0 canonical checkpoint pointers are not overwritten.

## Smoke Identity

| field | value |
| --- | --- |
| task | `Isaac-Ant-Velocity-Flat-v0` |
| stage | `healthy_baseline_velocity` |
| method | `rlm1_stripped` |
| fault | `none` |
| seed | `0` |
| default num_envs | `64` |
| default max_iterations | `10` |
| experiment_name | `healthy_baseline_velocity__rlm1_stripped__none` |
| run_name | `a0_velocity_smoke__seed0` |

## Expected Task Wiring

Runtime smoke for the task has already validated:

- `base_velocity` command is active.
- `velocity_commands` observation is active.
- reward terms include `track_lin_vel_xy_exp` and `track_ang_vel_z_exp`.
- P2 wrapper can attach to the task with `action_dim=8` and `fallback_used=False`.

The A0-Vel smoke itself is healthy/no-fault only:

- no P2 wrapper
- no privileged teacher observation
- no residual
- no health token
- no UQ
- no CBF
- no P3/P4 scope expansion

## Commands

Help / dry guard:

```bash
bash trainers/run_t10_train_a0_velocity_smoke.sh --help
```

Recommended smoke command:

```bash
bash trainers/run_t10_train_a0_velocity_smoke.sh --execute_smoke --max_iterations 10 --num_envs 64 --seed 0
```

## Validation

Shell syntax:

```bash
bash -n trainers/run_t10_train_a0_velocity_smoke.sh
```

Help:

```bash
bash trainers/run_t10_train_a0_velocity_smoke.sh --help
```

## Next Step After Smoke

If A0-Vel smoke runs cleanly, the next scaffold should be an A1-F-Vel P2 privileged teacher smoke under `Isaac-Ant-Teacher-Velocity-Flat-v0`, with P2 simulation-level joint lock enabled and checkpoint pointer updates disabled.
