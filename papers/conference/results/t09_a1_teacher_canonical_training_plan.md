# T09 A1 Teacher Canonical Training Plan

Scope: A1 privileged teacher canonical checkpoint generation scaffold; no
training or evaluation executed by this document.

## Objective

Train a non-smoke A1 privileged teacher checkpoint for the real RLM1 stripped
conference-stage pipeline. The method remains RLM1 stripped, health token OFF,
UQ inactive, and CBF inactive. A1 is trained under `fault: none`; P4 and other
fault profiles remain evaluation-time concerns and are not part of A1 training.

## Why A1 Comes Before A2

A2 student distillation should not be trained from a smoke/development teacher.
The current A1 latest pointer resolves to `model_0.pt` and remains
`smoke_or_dev_only`. A canonical A1 teacher must be trained, selected by exact
checkpoint path, and frozen before A2 canonical student training begins.

## Training Helper

Use:

```bash
bash trainers/run_t09_train_a1_teacher_canonical.sh
```

Default settings:

```text
num_envs: 4096
max_iterations: 2000
seed: 0
experiment_name: teacher__rlm1_stripped__canonical
run_name: a1_teacher_canonical__seed0
config: configs/train/teacher.yaml
task: Isaac-Ant-Teacher-v0
fault: none
```

The helper delegates to the existing T06 teacher path and passes
`--skip_checkpoint_pointer`, so raw training does not overwrite stable pointer
files.

Expected log root:

```text
logs/rsl_rl/teacher__rlm1_stripped__canonical/
```

## Freeze Step

After training, select one exact `model_*.pt` path manually. Freeze it with:

```bash
python evaluators/freeze_t09_checkpoint.py \
  --stage teacher \
  --method rlm1_stripped \
  --fault none \
  --seed 0 \
  --task Isaac-Ant-Teacher-v0 \
  --checkpoint_path <exact_model.pt> \
  --experiment_name teacher__rlm1_stripped__canonical \
  --run_name a1_teacher_canonical__seed0 \
  --classification paper_grade_candidate \
  --output_pointer checkpoints/rlm1_stripped/teacher/none/seed0/canonical_checkpoint.yaml \
  --manifest_path papers/conference/results/t09_a1_teacher_canonical_checkpoint_freeze.md
```

The freeze helper refuses `paper_grade_candidate` for `model_0.pt` and
demo/advisor paths. It does not overwrite `latest_checkpoint.yaml`.

## Deployment Role

A1 is privileged/reference-only and not deployment-facing. It is needed as a
teacher/reference checkpoint and as the upstream source for A2 student
distillation, but it should not be described as a deployed policy.

## Blockers Remaining After A1

Even after A1 is canonical, controlled evaluation remains blocked until A2
student and A5 residual readiness are improved. A2 should be trained/frozen from
the canonical teacher before A5 residual training is attempted.

## Guardrails

- Do not train A2/A5 in T09-R2.
- Do not launch evaluation in T09-R2.
- Do not modify RSL-RL core, Isaac Lab core, or task registration.
- Do not add health token, UQ, CBF, P2, or P3.
- Do not fabricate metrics or mark results observed.
