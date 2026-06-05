# T09 Checkpoint Freeze Manifest

Scope: explicit checkpoint freeze metadata; no training, evaluation, play, or Isaac Sim executed.

- stage: `healthy_baseline`
- method: `rlm1_stripped`
- fault: `none`
- seed: `0`
- task: `Isaac-Ant-v0`
- experiment_name: `healthy_baseline__rlm1_stripped__canonical`
- run_name: `a0_canonical__seed0`
- classification: `paper_grade_candidate`
- checkpoint_path: `logs/rsl_rl/healthy_baseline__rlm1_stripped__canonical/2026-06-05_21-42-13_a0_canonical__seed0/model_1999.pt`
- checkpoint_filename: `model_1999.pt`
- model_iteration: `1999`
- output_pointer: `checkpoints/rlm1_stripped/healthy_baseline/none/seed0/canonical_checkpoint.yaml`

## Policy

- This freeze helper never infers paper-grade status automatically.
- `paper_grade_candidate` requires an explicit classification argument.
- `paper_grade_candidate` is refused for `model_0.pt` and demo/advisor paths.
- `latest_checkpoint.yaml` is never overwritten by this helper.
- Controlled evaluation must still record exact resolved checkpoint paths.

## Guardrails

- No training was run.
- No Isaac Sim or play execution was launched.
- No observed-result manifest was updated.
- No health token, UQ, CBF, P2, or P3 expansion was added.
