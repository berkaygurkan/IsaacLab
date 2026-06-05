# T09 A0 Canonical Training Plan

Scope: A0 healthy PPO canonical checkpoint generation scaffold; no training or
evaluation executed by this document.

## Objective

Train a non-demo A0 healthy PPO checkpoint for the real conference-stage
controlled evaluation pipeline. The method remains RLM1 stripped, health token
OFF, UQ inactive, and CBF inactive. A0 is trained healthy/no-fault only; P4
torque degradation is applied later during evaluation, not during training.

## Training Helper

Use:

```bash
bash trainers/run_t09_train_a0_canonical.sh
```

Default settings:

```text
num_envs: 4096
max_iterations: 2000
seed: 0
experiment_name: healthy_baseline__rlm1_stripped__canonical
run_name: a0_canonical__seed0
config: configs/train/healthy_baseline_demo.yaml
task: Isaac-Ant-v0
fault: none
```

The helper delegates to the existing T05 healthy PPO path and passes
`--skip_checkpoint_pointer`, so raw training does not overwrite stable pointer
files.

Expected log root:

```text
logs/rsl_rl/healthy_baseline__rlm1_stripped__canonical/
```

## Freeze Step

After training, select one exact `model_*.pt` path manually. Freeze it with:

```bash
python evaluators/freeze_t09_checkpoint.py \
  --stage healthy_baseline \
  --method rlm1_stripped \
  --fault none \
  --seed 0 \
  --task Isaac-Ant-v0 \
  --checkpoint_path <exact_model.pt> \
  --experiment_name healthy_baseline__rlm1_stripped__canonical \
  --run_name a0_canonical__seed0 \
  --classification paper_grade_candidate \
  --output_pointer checkpoints/rlm1_stripped/healthy_baseline/none/seed0/canonical_checkpoint.yaml \
  --manifest_path papers/conference/results/t09_a0_canonical_checkpoint_freeze.md
```

The freeze helper refuses `paper_grade_candidate` for `model_0.pt` and
demo/advisor paths. It does not overwrite `latest_checkpoint.yaml`.

## Blockers Remaining After A0

A0 alone is not enough to launch the full controlled ablation evaluation. T09-R0
classified A1, A2, and A5 as `smoke_or_dev_only`. A2 and A5 must be trained or
selected and frozen separately before A0/A2/A5 controlled comparisons can be
claimed.

## Guardrails

- Do not run A1/A2/A5 training in T09-R1.
- Do not launch evaluation in T09-R1.
- Do not modify RSL-RL core, Isaac Lab core, or task registration.
- Do not add health token, UQ, CBF, P2, or P3.
- Do not fabricate metrics or mark results observed.
