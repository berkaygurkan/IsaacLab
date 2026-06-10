# T10 A1-F-Vel P2 Teacher PPO Smoke Plan

Scope: T10-IL-04 smoke launcher only. This is not paper-grade training and does not freeze checkpoints, edit stable checkpoint pointers, update observed-result manifests, train students or residuals, or modify RSL-RL core.

## Launcher

```text
trainers/run_t10_train_a1f_velocity_p2_smoke.sh
```

The launcher delegates to:

```text
trainers/rsl_rl_train.py
```

It always passes `--skip_checkpoint_pointer` so stable checkpoint pointers are not overwritten. It also enables the repo-owned P2 joint-lock training wrapper with `simulation_joint_state_override_lock` semantics and no fallback.

## Smoke Identity

| field | value |
| --- | --- |
| task | `Isaac-Ant-Teacher-Velocity-Flat-v0` |
| stage | `teacher_p2_velocity` |
| method | `rlm1_stripped` |
| fault | `P2_locked_joint` |
| seed | `0` |
| default num_envs | `64` |
| default max_iterations | `10` |
| experiment_name | `teacher_p2_velocity__rlm1_stripped__p2_locked_joint` |
| run_name | `a1f_velocity_p2_smoke__seed0` |
| P2 target joint | `front_left_foot` |
| default P2 onset mode | `fixed` |
| default P2 onset step | `50` |
| random-onset support | `random_uniform`, step range `30` to `150` |

## Expected Task Wiring

Runtime validation for the velocity teacher task has already confirmed:

- `base_velocity` command is active.
- `velocity_commands` observation is active.
- P2 wrapper attaches with `action_dim=8`.
- P2 fallback is not needed when simulation-state override is available.

Expected A1-F-Vel teacher properties:

- actor/critic observation dimension: `62`
- privileged/reference terms include `true_fault_state` and contacts
- expected rewards include `track_lin_vel_xy_exp`, `track_ang_vel_z_exp`, `alive`, `upright`, `action_l2`, `energy`, and `joint_pos_limits`
- expected P2 metrics include `P2/fault_applied`, `P2/simulation_override_applied`, `P2/fallback_used`, and onset-step diagnostics

This teacher row is privileged and not deployment-facing. Deployment-facing velocity comparisons remain future A0/A2/A5-style rows after student/residual readiness.

## Smoke Guardrails

- smoke only, not paper-grade
- fixed P2 onset by default for the tiny smoke
- random P2 onset is supported through launcher arguments and maps to the trainer's `--p2_fault_onset_*` options
- no checkpoint freeze
- no stable pointer update
- no student training
- no residual training
- no health token
- no UQ
- no CBF
- no P3/P4 expansion

## Commands

Help / dry guard:

```bash
bash trainers/run_t10_train_a1f_velocity_p2_smoke.sh --help
```

Recommended smoke command:

```bash
bash trainers/run_t10_train_a1f_velocity_p2_smoke.sh --execute_smoke --max_iterations 10 --num_envs 64 --seed 0
```

Random-onset candidate command:

```bash
bash trainers/run_t10_train_a1f_velocity_p2_smoke.sh \
  --execute_smoke \
  --max_iterations 1000 \
  --num_envs 1024 \
  --seed 0 \
  --fault_onset_mode random_uniform \
  --fault_onset_step_min 30 \
  --fault_onset_step_max 150 \
  --run_name a1f_velocity_p2_random_candidate1000__seed0
```

The launcher maps these user-facing arguments to `trainers/rsl_rl_train.py` as `--p2_fault_onset_mode`, `--p2_fault_onset_step_min`, and `--p2_fault_onset_step_max`, so they are consumed by the repo-owned P2 wrapper instead of leaking into Isaac Lab's `train.py`.

## Validation

Shell syntax:

```bash
bash -n trainers/run_t10_train_a1f_velocity_p2_smoke.sh
```

Help:

```bash
bash trainers/run_t10_train_a1f_velocity_p2_smoke.sh --help
```

Diff hygiene:

```bash
git diff --check -- trainers/run_t10_train_a1f_velocity_p2_smoke.sh papers/conference/results/t10_a1f_velocity_p2_smoke_plan.md
```

## Next Step After Smoke

If A1-F-Vel P2 smoke runs cleanly, the next step is either longer A0-Vel/A1-F-Vel sanity training or a guarded controlled velocity-evaluation scaffold using matched command velocity, matched P2 onset, and exact checkpoint paths.
