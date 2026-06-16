# T13 Multi-Joint Teacher Selection

This note records the current teacher selection after the T13-D hard-foot
finetune evaluation. It is documentation only: no checkpoints, task configs,
canonical pointers, or result artifacts are modified here.

## Selected Teacher

Selection label: A1-F multi-joint teacher v2b, valid but foot-limited.

Selected checkpoint:

```text
logs/rsl_rl/teacher_p2_multijoint_velocity_curriculum_v2b__rlm1_stripped__p2_multi_joint_random/2026-06-15_18-16-03_a1f_multijoint_velocity_p2_v2b_curriculum_curriculum_s2_realistic_transition_random_p2__seed0/model_9997.pt
```

Use this checkpoint as the T14 teacher for multi-joint rollout dataset
collection.

Validation status:

- fixed-vx T13C validation: passed as the current best all-8 random P2 teacher
  candidate, with overall post-fault vx error around `0.16-0.17`.
- command-random T13C validation: completed and retained as supporting
  command-conditioned evidence.
- known limitation: foot-joint recovery remains weaker than leg-joint recovery,
  so the teacher is valid but foot-limited.

Runtime semantics to preserve downstream:

- task: `Isaac-Ant-Teacher-Velocity-MultiJointP2-Flat-v0`
- target joint mode: `random_per_env`
- selected joint: random over all `8` supported actuated joints
- active locked joints per env/episode: `1`
- P2 semantics: direct `simulation_joint_state_override_lock`
- q_lock source: captured from the selected joint's current simulated position
  at fault onset
- post-onset enforcement: selected joint `q = q_lock` and `qd = 0`
- fallback: disabled
- PD surrogate: disabled
- teacher observation: `77 = 61 base + 8 selected-joint one-hot + 8 q_lock`
- deployment student observation remains fault-descriptor-free
- health token: OFF

## Rejected Teacher

Rejected checkpoint:

```text
logs/rsl_rl/teacher_p2_multijoint_velocity_curriculum_v2b__rlm1_stripped__p2_multi_joint_random/2026-06-15_21-17-39_a1f_multijoint_velocity_p2_hard_foot_finetune_120_700_i2000__seed0/model_11996.pt
```

Decision: reject as an over-specialized hard-foot finetune. Do not use this
checkpoint as the downstream teacher.

Reason:

- all-8 random P2 fixed-vx performance worsened versus v2b.
- overall post-fault vx error worsened from about `0.16-0.17` to about `0.245`.
- `front_left_foot` and `front_right_foot` became substantially worse.
- several non-foot joints regressed.

The T13-D run is useful as evidence that the hard-foot-only finetune can
over-specialize, but it should not replace the v2b teacher for dataset
collection or distillation.

## Next Phase

Proceed to T14: multi-joint teacher rollout dataset collection using the
selected v2b teacher checkpoint. Dataset collection should keep random
selection over all 8 supported P2 joints and the direct override semantics
listed above.
