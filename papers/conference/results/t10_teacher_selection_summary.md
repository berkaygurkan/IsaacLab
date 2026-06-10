# T10 P2 Velocity Teacher Selection Summary

Scope: documentation-only teacher selection record for the conference A1-F privileged P2 velocity teacher. No training, Isaac Sim run, checkpoint edit, task config edit, P2 wrapper edit, student/residual code, or checkpoint pointer update was performed.

## Teacher Candidates

| candidate | training source | checkpoint | status |
| --- | --- | --- | --- |
| Random P2 teacher1000 | P2 random onset training | `logs/rsl_rl/teacher_p2_velocity__rlm1_stripped__p2_locked_joint/2026-06-10_00-32-52_a1f_velocity_p2_random_candidate1000__seed0/model_999.pt` | evaluated baseline |
| Random P2 teacher3000 | P2 random onset training | `logs/rsl_rl/teacher_p2_velocity__rlm1_stripped__p2_locked_joint/2026-06-10_18-58-22_a1f_velocity_p2_random_candidate3000__seed0/model_2999.pt` | evaluated baseline |
| Onset curriculum teacher3000 | staged onset curriculum ending at target P2 random onset | `logs/rsl_rl/teacher_p2_velocity_curriculum__rlm1_stripped__p2_locked_joint/2026-06-10_19-29-53_a1f_velocity_p2_onset_curriculum_seed0_curriculum_s3_target__seed0/model_2996.pt` | selected teacher source |

## Selected Teacher

Selected checkpoint:

```text
logs/rsl_rl/teacher_p2_velocity_curriculum__rlm1_stripped__p2_locked_joint/2026-06-10_19-29-53_a1f_velocity_p2_onset_curriculum_seed0_curriculum_s3_target__seed0/model_2996.pt
```

Selected task:

```text
Isaac-Ant-Teacher-Velocity-Flat-v0
```

Evaluation protocol: P2 random onset, `random_uniform` over steps `30` to `150`, target joint `front_left_foot`, semantics `simulation_joint_state_override_lock`, fallback disabled.

## Comparison Table

| candidate | mean vel x post fault | mean abs vx error post fault | mean abs yaw error | timeout rate | torso height failure rate | mean post-fault survival steps | survived to fault onset rate | fallback used | simulation override applied |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Random P2 teacher1000 | 0.7935030502206991 | 0.20649694977930094 | 0.15715440290607513 | 0.03778125 | 0.0726015625 | 846.0661157024794 | 0.9453125 | 0.0 | 1.0 |
| Random P2 teacher3000 | 0.7935533146215241 | 0.20644668537847585 | 0.14979593086242676 | 0.03790625 | 0.0721953125 | 839.0983606557377 | 0.953125 | 0.0 | 1.0 |
| Onset curriculum teacher3000 | 0.9298820181122678 | 0.07017888256581774 | 0.12630629476252944 | 0.037484375 | 0.0783203125 | 847.2333333333333 | 0.9375 | 0.0 | 1.0 |

## Selection Rationale

The onset curriculum teacher is selected because it has the strongest post-fault velocity tracking among the evaluated A1-F P2 candidates. Relative to random P2 teacher3000, it improves mean post-fault forward velocity from `0.7935533146215241` to `0.9298820181122678`, reduces mean absolute post-fault vx error from `0.20644668537847585` to `0.07017888256581774`, and reduces mean absolute yaw error from `0.14979593086242676` to `0.12630629476252944`.

The curriculum teacher has a slightly lower survived-to-fault-onset rate than random P2 teacher3000 (`0.9375` vs `0.953125`) and a slightly higher torso-height failure rate (`0.0783203125` vs `0.0721953125`). Those tradeoffs are accepted at this stage because the downstream teacher role prioritizes a stronger reference action under the post-fault target condition.

## Caveats

This is candidate-level, seed0 evidence only. It is not paper-grade final evaluation and should not be presented as final statistical evidence.

A1-F is privileged/reference only and is not deployment-facing. The A1-F teacher task includes `true_fault_state`, which must remain excluded from deployment policies.

Deployment-facing A2, A5, and A7 policies must not use `true_fault_state` in actor or critic runtime observations.

## Downstream Use

The selected curriculum checkpoint is the teacher source for:

- A2 student distillation
- A5 teacher-gap residual with distilled student
- A7 healthy PPO plus teacher-gap residual

This document records the selected source only. It does not update checkpoint pointers or introduce student/residual implementation.
