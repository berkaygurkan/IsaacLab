# T09 Demo Checkpoint Selection

Scope: advisor-demo checkpoint selection only; separate from paper-grade checkpoint policy.

## Rules

- Do not overwrite stable checkpoint pointers.
- Select the demo checkpoint by exact `model_*.pt` path.
- Record the exact checkpoint path used for every demo video.
- Keep the demo checkpoint separate from paper-grade checkpoint freeze/promotion.
- Do not interpret demo training as paper-grade performance evidence.

## Expected Demo Checkpoint Location

```text
logs/rsl_rl/healthy_baseline__rlm1_stripped__demo/<timestamp>_healthy_demo__seed0/model_*.pt
```

Current candidate checkpoint for the advisor demo:

```text
logs/rsl_rl/healthy_baseline__rlm1_stripped__demo/2026-06-03_04-08-57_healthy_demo__seed0/model_1999.pt
```

This is a demo-grade candidate only. It is not automatically promoted to a
paper-grade or controlled-evaluation checkpoint.

## Video Pairing

Use the same selected A0 checkpoint for:

- A0 no-fault meeting demo.
- A0 P4 torque-degradation meeting demo.

Use `evaluators/run_t09_demo_play_a0.sh --checkpoint_path <exact_model_path>`
so `demo_summary.md` records the selected path directly. The helper writes under
`runs/t09_demo_play/` and does not promote or overwrite any pointer.

A5 remains a frozen residual-policy demo row and should use its exact selected student and residual checkpoint paths.
