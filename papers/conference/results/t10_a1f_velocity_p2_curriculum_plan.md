# T10 A1-F Velocity P2 Onset Curriculum Plan

Scope: candidate-level A1-F privileged teacher training scaffold only. This does not run training by itself, freeze checkpoints, update checkpoint pointers, train student/residual methods, modify RSL-RL or Isaac Lab core, or change P2 semantics.

## Motivation

Teacher quality is critical for the later A2/A5/A7 path. The current no-curriculum A1-F velocity P2 random candidates provide useful baselines, but a staged onset curriculum may make the teacher more stable before exposing it to early random joint locks. This launcher keeps the same privileged teacher observation surface audited in `t10_teacher_privileged_obs_audit.md`: 62D teacher input with a single teacher-only privileged extra, `true_fault_state`.

## Launcher

```text
trainers/run_t10_train_a1f_velocity_p2_curriculum.sh
```

The launcher delegates to:

```text
trainers/rsl_rl_train.py
```

It passes `--skip_checkpoint_pointer`, `--enable_p2_joint_lock`, `--p2_requested_semantics simulation_joint_state_override_lock`, and never passes `--p2_allow_fallback`.

## Resume / Chaining

Resume is supported through the existing upstream RSL-RL CLI path:

```text
--resume --load_run <previous_timestamped_run_dir> --checkpoint <previous_model.pt>
```

All curriculum stages use the same experiment root:

```text
logs/rsl_rl/teacher_p2_velocity_curriculum__rlm1_stripped__p2_locked_joint/
```

After each stage completes, the launcher locates the latest matching run folder and highest `model_*.pt`, then resumes the next stage from that checkpoint. If no checkpoint is found, the launcher exits with an error rather than silently faking curriculum.

## Curriculum Schedule

| stage | run suffix | default iterations | onset mode | onset configuration | purpose |
| --- | --- | ---: | --- | --- | --- |
| S0 | `curriculum_s0_healthy` | 500 | fixed | step `2000` | Healthy/no-effective-fault warmup inside the normal episode horizon. |
| S1 | `curriculum_s1_late` | 750 | random_uniform | `[300, 600]` | Late fault exposure. |
| S2 | `curriculum_s2_medium` | 750 | random_uniform | `[100, 300]` | Medium fault exposure. |
| S3 | `curriculum_s3_target` | 1000 | random_uniform | `[30, 150]` | Target conference P2 random-onset condition. |

## Comparison Target

Compare the curriculum final stage against the no-curriculum candidate:

```text
a1f_velocity_p2_random_candidate3000__seed0
```

The comparison should use the same controlled velocity P2 random evaluation protocol used for T10-IL-05:

- task: `Isaac-Ant-Teacher-Velocity-Flat-v0`
- P2 target joint: `front_left_foot`
- P2 semantics: `simulation_joint_state_override_lock`
- fallback: disabled
- onset mode: `random_uniform`
- onset range: `[30, 150]`
- velocity command: `vx_cmd=1.0`

## Suggested Command

```bash
bash trainers/run_t10_train_a1f_velocity_p2_curriculum.sh \
  --execute_curriculum \
  --num_envs 1024 \
  --seed 0
```

Optional shorter/longer schedules can use:

```bash
--s0_iterations <N> --s1_iterations <N> --s2_iterations <N> --s3_iterations <N>
```

## Guardrails

- A1-F remains privileged/reference only.
- This is candidate-level training, not paper-grade final evidence.
- No checkpoint is promoted automatically.
- Stable checkpoint pointers are not updated.
- P2 fallback remains disabled.
- No student, residual, health token, UQ, CBF, P3, or P4 method branch is introduced.
