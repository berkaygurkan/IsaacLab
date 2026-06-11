# T10 A2 Student Settled P2 Evaluation Summary

## Scope

Candidate-level Isaac-side A2 student evaluation under settled-regime P2 onset. These results are not paper-grade final evidence.

## Protocol

- task: `Isaac-Ant-Velocity-Flat-v0`
- num_envs: `128`
- num_steps: `1000`
- seed: `0`
- target_vx: `1.0`
- fault: `P2_locked_joint`
- target_joint: `front_left_foot`
- semantics: `simulation_joint_state_override_lock`
- onset: `random_uniform [300, 300]`
- fallback: disabled
- device: `cpu`

## Results

| policy kind | checkpoint | post-fault mean_vel_x | post-fault mean_abs_vx_error | mean_abs_yaw_error | timeout_rate | torso_height_failure_rate | P2/fallback_used | P2/simulation_override_applied |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| single_step | `papers/conference/results/t10_a2_student_distill_full_seed0/a2_student.pt` | 0.6722072014284882 | 0.3414071450347397 | 0.14529768317379058 | 0.03715625 | 0.087484375 | 0.0 | 1.0 |
| history_h16 | `papers/conference/results/t10_a2_student_history_distill_full_h16_seed0/a2_student_history.pt` | 0.9774326449220259 | 0.16393009481816082 | 0.1404474851633422 | 0.03715625 | 0.0866640625 | 0.0 | 1.0 |

## Guardrails

- No training was run.
- No checkpoints, task configs, P2 wrapper semantics, or checkpoint pointers were modified.
- Runtime policy inputs used student-safe observations only.
- No `teacher_obs`, `true_fault_state`, health token, or residual correction was used.
- `fallback_used` remained false and P2 simulation override was observed.
