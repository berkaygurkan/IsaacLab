# T14 Multi-Joint Dataset Audit

This directory is reserved for the offline T14 dataset audit outputs.

Audit utility:

```text
evaluators/audit_t14_multijoint_teacher_dataset.py
```

The audit is read-only with respect to datasets and checkpoints. It does not
import Isaac, launch simulation, train policies, or modify previous T13/T14
artifacts.

## Inputs

Realistic random dataset:

```text
papers/conference/datasets/t14_multijoint_teacher_v2b_realistic_seed0/dataset.npz
```

Late random dataset:

```text
papers/conference/datasets/t14_multijoint_teacher_v2b_late_seed0/dataset.npz
```

Active phase: RLM1 stripped / conference.

Selected teacher: A1-F multi-joint teacher v2b, valid but foot-limited.

Selected checkpoint:

```text
logs/rsl_rl/teacher_p2_multijoint_velocity_curriculum_v2b__rlm1_stripped__p2_multi_joint_random/2026-06-15_18-16-03_a1f_multijoint_velocity_p2_v2b_curriculum_curriculum_s2_realistic_transition_random_p2__seed0/model_9997.pt
```

## Outputs

When run with `--write_outputs`, the audit writes:

- `dataset_audit_summary.md`
- `dataset_audit_summary.json`
- `per_dataset_stats.csv`
- `per_joint_stats.csv`
- `advisor_update_snippet.txt`

## Required Checks

The audit checks that each dataset exists and contains the required arrays,
including 61-D `student_obs`, 77-D `teacher_obs`, 8-D `teacher_action`, selected
joint indices in `0..7`, all 8 joints represented, 8-D selected-joint one-hot,
8-D q-lock vector, both inactive and active P2 samples, protocol-specific onset
ranges, and finite numeric arrays.

It also reports command and tracking statistics, done/survival rates, inferred
env/timestep counts, and per-joint tracking summaries.

Survival rate is reported as `1 - done_rate`. Timeout should not be counted as
failure if timeout can be separated in a later analysis.

## Interpretation

`realistic_random` and `late_random` are separate datasets because collecting
both protocols in one process caused the second environment creation to hang.
Do not treat separate-process collection as a method failure.

This audit is candidate-level evidence for advisor sharing, not final
paper-grade reporting.

The datasets are intended for:

- A2 single-step student distillation
- A2-history H16 student distillation
- A5 history residual distillation
- optional A7 A0-anchored residual training

## Later Graph Plan

Later plots should support:

- vx_cmd vs velocity_x tracking
- vx_error distribution
- per-joint vx_error bar plot
- survival rate per joint
- fault active vs inactive tracking
- latent `z_t` t-SNE/UMAP after A2-history/A5 training

Do not implement latent visualization in T14. Latent analysis belongs to later
T18 and should use encoder output `z_t`, not raw H16 history.

## Manual Command

```bash
cd ~/thesis/IsaacLab
conda activate isaaclab

python evaluators/audit_t14_multijoint_teacher_dataset.py \
  --write_outputs \
  --realistic_dataset papers/conference/datasets/t14_multijoint_teacher_v2b_realistic_seed0/dataset.npz \
  --late_dataset papers/conference/datasets/t14_multijoint_teacher_v2b_late_seed0/dataset.npz \
  --output_root papers/conference/results/t14_multijoint_dataset_audit
```
