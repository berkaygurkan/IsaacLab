# T09 Demo Training and Play Workflow

Scope: advisor-demo checkpoint generation and visual playback only; not paper-grade evaluation.

## A0 Healthy Demo Training

A0 is trained as healthy/no-fault PPO only. P4 torque degradation is applied later during play/evaluation demos, not during A0 training. The same selected A0 checkpoint should be used for both the no-fault demo video and the P4 demo video so the visual comparison changes only the runtime fault condition.

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

## A0 Demo Play

Use the exact selected A0 `model_*.pt` path for both visual demos. The no-fault
video runs without a P4 wrapper. The P4 video applies torque degradation during
play/evaluation only; the checkpoint was still trained healthy/no-fault.

Candidate demo checkpoint from the first controlled helper run:

```text
logs/rsl_rl/healthy_baseline__rlm1_stripped__demo/2026-06-03_03-40-01_healthy_demo__seed0/model_999.pt
```

A0 no-fault playback:

```bash
bash evaluators/run_t09_demo_play_a0.sh --demo_gui --log_step_csv --demo_name D0_A0_no_fault --checkpoint_path logs/rsl_rl/healthy_baseline__rlm1_stripped__demo/2026-06-03_03-40-01_healthy_demo__seed0/model_999.pt --fault_profile F0_none
```

A0 P4 torque-degradation playback:

```bash
bash evaluators/run_t09_demo_play_a0.sh --demo_gui --log_step_csv --demo_name D1_A0_P4 --checkpoint_path logs/rsl_rl/healthy_baseline__rlm1_stripped__demo/2026-06-03_03-40-01_healthy_demo__seed0/model_999.pt --fault_profile P4_torque_degradation
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

## Guardrails

- No P4 fault is applied during A0 training.
- No checkpoint pointer is updated by the demo-training helper.
- Demo play uses an exact selected checkpoint path and does not update stable checkpoint pointers.
- No observed-result manifest is updated.
- No paper-grade claim is implied.
- No P2/P3, health token, UQ, CBF, or residual training is added.
