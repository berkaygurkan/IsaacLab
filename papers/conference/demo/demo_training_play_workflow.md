# T09 Demo Training and Play Workflow

Scope: advisor-demo checkpoint generation and visual playback only; not paper-grade evaluation.

T09-R2d pivots the conference main fault scope to `P2_locked_joint`. The P4
commands in this document are retained only as advisor-demo / thesis-extension
artifacts and must not be treated as conference main results.

## A0 Healthy Demo Training

A0 is trained as healthy/no-fault PPO only. P4 torque degradation is applied later during advisor-demo play/evaluation, not during A0 training. The same selected A0 checkpoint should be used for both the no-fault demo video and the P4 demo video so the visual comparison changes only the runtime fault condition.

Recommended first run:

```bash
bash trainers/run_t09_demo_train_a0_healthy.sh --num_envs 512 --max_iterations 1000
```

Longer optional run if VRAM allows:

```bash
bash trainers/run_t09_demo_train_a0_healthy.sh --num_envs 1024 --max_iterations 2000
```

Expected log root:

```text
logs/rsl_rl/healthy_baseline__rlm1_stripped__demo/
```

## A0/A2 Demo Play

Use the exact selected A0 `model_*.pt` path for both A0 visual demos. A2 uses
the selected student checkpoint path or the student checkpoint pointer. The
no-fault videos run without a P4 wrapper. P4 videos apply torque degradation
during advisor-demo play/evaluation only; no P4 fault is applied during A0
training. Future conference controlled evaluation should use P2 single joint
lock instead of these P4 demo commands.

Current A0 demo checkpoint:

```text
logs/rsl_rl/healthy_baseline__rlm1_stripped__demo/2026-06-03_04-08-57_healthy_demo__seed0/model_1999.pt
```

Current A2 student pointer:

```text
checkpoints/rlm1_stripped/student/none/seed0/latest_checkpoint.yaml
```

A0 no-fault playback:

```bash
bash evaluators/run_t09_demo_play_policy.sh --ablation_id A0 --demo_gui --log_step_csv --demo_name D0_A0_no_fault --checkpoint_path logs/rsl_rl/healthy_baseline__rlm1_stripped__demo/2026-06-03_04-08-57_healthy_demo__seed0/model_1999.pt --fault_profile F0_none
```

A0 P4 torque-degradation playback:

```bash
bash evaluators/run_t09_demo_play_policy.sh --ablation_id A0 --demo_gui --log_step_csv --demo_name D1_A0_P4 --checkpoint_path logs/rsl_rl/healthy_baseline__rlm1_stripped__demo/2026-06-03_04-08-57_healthy_demo__seed0/model_1999.pt --fault_profile P4_torque_degradation
```

A2 no-fault playback:

```bash
bash evaluators/run_t09_demo_play_policy.sh --ablation_id A2 --demo_gui --log_step_csv --demo_name D2_A2_no_fault --checkpoint_pointer checkpoints/rlm1_stripped/student/none/seed0/latest_checkpoint.yaml --fault_profile F0_none
```

A2 P4 torque-degradation playback:

```bash
bash evaluators/run_t09_demo_play_policy.sh --ablation_id A2 --demo_gui --log_step_csv --demo_name D3_A2_P4 --checkpoint_pointer checkpoints/rlm1_stripped/student/none/seed0/latest_checkpoint.yaml --fault_profile P4_torque_degradation
```

Each playback writes `demo_summary.md` under `runs/t09_demo_play/` and records
the exact checkpoint path, fault profile, rollout length, fault-window status,
NaN/Inf status, and the note that this is an advisor demo only.

With `--log_step_csv`, each playback also writes `step_metrics.csv`. Every
playback writes `summary_metrics.json`, even when per-step CSV logging is not
requested. Use the comparison helper after recording both videos:

```bash
python evaluators/compare_t09_demo_runs.py --f0_run_dir runs/t09_demo_play/<timestamp>_D0_A0_no_fault --p4_run_dir runs/t09_demo_play/<timestamp>_D1_A0_P4
```

For a visual-stress demo only, `--torque_scale 0.2` or `--torque_scale 0.0` may
be used with `P4_torque_degradation`. These runs are labelled
`visual_stress_demo: True` and remain outside paper-grade evaluation.

Recommended A0-vs-A2 P4 visual-stress comparison:

```bash
bash evaluators/run_t09_demo_play_policy.sh --ablation_id A0 --demo_gui --log_step_csv --demo_name D1_A0_P4_torque0_2 --checkpoint_path logs/rsl_rl/healthy_baseline__rlm1_stripped__demo/2026-06-03_04-08-57_healthy_demo__seed0/model_1999.pt --fault_profile P4_torque_degradation --torque_scale 0.2
```

```bash
bash evaluators/run_t09_demo_play_policy.sh --ablation_id A2 --demo_gui --log_step_csv --demo_name D3_A2_P4_torque0_2 --checkpoint_pointer checkpoints/rlm1_stripped/student/none/seed0/latest_checkpoint.yaml --fault_profile P4_torque_degradation --torque_scale 0.2
```

```bash
python evaluators/compare_t09_demo_runs.py --left_run_dir runs/t09_demo_play/<timestamp>_D1_A0_P4_torque0_2 --right_run_dir runs/t09_demo_play/<timestamp>_D3_A2_P4_torque0_2 --left_label A0_P4_stress --right_label A2_P4_stress
```

## P2 Quick Checkpoint Sanity

T09-R2j adds a P2-specific quick demo evaluator:

```text
evaluators/run_t09_p2_quick_demo_compare.py
```

This runner is for checkpoint sanity only. It accepts `F0_none` and
`P2_locked_joint`; for P2 it attaches the same simulation-level joint-state
override wrapper used by the A1-F training/preflight path. It does not update
paper-grade manifests.

A0 versus A1-F is not a fair deployment comparison because A1-F is a privileged
teacher and may use teacher observations such as `true_fault_state`. Treat A1-F
as a sanity/upper-bound reference only. The future fair deployment comparison is
A0 versus A2 versus A5.

A0 P2 quick sanity:

```bash
python evaluators/run_t09_p2_quick_demo_compare.py --execute_demo --headless --policy_label A0_P2_quick --task Isaac-Ant-v0 --checkpoint_path logs/rsl_rl/healthy_baseline__rlm1_stripped__canonical/2026-06-05_21-42-13_a0_canonical__seed0/model_1999.pt --fault_profile P2_locked_joint --target_joint front_left_foot --fault_onset_step 50 --num_envs 64 --num_steps 1000 --output_dir runs/t09_quick_p2_compare/A0_P2_quick
```

A1-F random-onset teacher P2 quick sanity:

```bash
python evaluators/run_t09_p2_quick_demo_compare.py --execute_demo --headless --policy_label A1F_random_P2_quick --task Isaac-Ant-Teacher-v0 --checkpoint_path logs/rsl_rl/teacher_p2_random_onset__rlm1_stripped__canonical/2026-06-09_15-30-02_a1f_p2_teacher_random_onset_canonical__seed0/model_1999.pt --fault_profile P2_locked_joint --target_joint front_left_foot --fault_onset_step 50 --num_envs 64 --num_steps 1000 --output_dir runs/t09_quick_p2_compare/A1F_random_P2_quick
```

## Guardrails

- No P4 fault is applied during A0 training.
- P4 playback remains advisor-demo / thesis-extension material only.
- P2 quick comparison is sanity/demo only and not paper-grade evaluation.
- A1-F is privileged and not deployment-facing; A0-vs-A1-F is not a fair deployment comparison.
- No checkpoint pointer is updated by the demo-training helper.
- Demo play uses an exact selected checkpoint path and does not update stable checkpoint pointers.
- No observed-result manifest is updated.
- No paper-grade claim is implied.
- A2 demo play uses the frozen student checkpoint with no residual wrapper.
- No P3, health token, UQ, CBF, A5 residual logic, or residual training is added.
