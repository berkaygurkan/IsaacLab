# T09 Canonical Checkpoint Freeze Policy

Scope: RLM1 stripped conference-stage checkpoint freeze policy; no training,
evaluation, play, Isaac Sim, checkpoint promotion, or pointer overwrite executed.

## Purpose

T09-R1 separates raw checkpoint generation from canonical checkpoint selection.
Training helpers may create candidate `model_*.pt` files, but those files do not
become controlled-evaluation checkpoints until an explicit freeze step records
the exact path. Advisor-demo checkpoints remain separate from conference-stage
controlled evaluation artifacts.

## A0 Canonical Rule

A0 canonical training is healthy/no-fault PPO only. Under the current P2-only
conference scope, P2 single joint lock is an evaluation-time condition and is
not applied during A0 training. P4 torque degradation is deferred to
advisor-demo / thesis-extension artifacts. The future A0 canonical checkpoint
should be selected by exact `model_*.pt` path from:

```text
logs/rsl_rl/healthy_baseline__rlm1_stripped__canonical/
```

The raw training helper must not update `latest_checkpoint.yaml`. The canonical
pointer should be written only by `evaluators/freeze_t09_checkpoint.py` to:

```text
checkpoints/rlm1_stripped/healthy_baseline/none/seed0/canonical_checkpoint.yaml
```

## A1-H Canonical Rule

A1-H canonical training is privileged-teacher/no-fault only. It is a reference
or pretraining context and is not deployment-facing. It is not the final P2
fault-aware teacher. A2 student distillation should not use the current A1-H
`model_0.pt` smoke/development checkpoint. If A1-H is frozen for reference
purposes, the checkpoint should be selected by exact `model_*.pt` path from:

```text
logs/rsl_rl/teacher__rlm1_stripped__canonical/
```

The raw teacher training helper must not update `latest_checkpoint.yaml`. The
canonical teacher pointer should be written only by
`evaluators/freeze_t09_checkpoint.py` to:

```text
checkpoints/rlm1_stripped/teacher/none/seed0/canonical_checkpoint.yaml
```

## A1-F P2 Canonical Rule

A1-F is the future privileged teacher for the P2-only conference scope. It must
be trained under the `P2_locked_joint` curriculum before A2 distillation. The
default P2 target joint is `front_left_foot`, and the default fault onset step
is `50`. A1-F may use privileged fault information during training, but it is
not deployment-facing.

The A1-F helper is:

```text
trainers/run_t09_train_a1f_p2_teacher_canonical.sh
```

Expected log root:

```text
logs/rsl_rl/teacher_p2__rlm1_stripped__canonical/
```

The selected A1-F checkpoint should be frozen only by exact `model_*.pt` path.
The chosen canonical pointer path is:

```text
checkpoints/rlm1_stripped/teacher_p2/none/seed0/canonical_checkpoint.yaml
```

## Freeze Requirements

- Freeze by exact checkpoint path, not by latest run directory.
- Do not classify `model_0.pt` as `paper_grade_candidate`.
- Do not classify demo/advisor checkpoints as `paper_grade_candidate`.
- Never infer paper-grade status automatically.
- Never overwrite `latest_checkpoint.yaml`.
- Record a Markdown freeze manifest beside the conference results documents.

Suggested manifest path:

```text
papers/conference/results/t09_a0_canonical_checkpoint_freeze.md
```

For A1:

```text
papers/conference/results/t09_a1_teacher_canonical_checkpoint_freeze.md
```

For A1-F P2:

```text
papers/conference/results/t09_a1f_p2_teacher_canonical_checkpoint_freeze.md
```

## Current Readiness Status

T09-R1b reports A0 canonical readiness through
`checkpoints/rlm1_stripped/healthy_baseline/none/seed0/canonical_checkpoint.yaml`.
A1-H, A2, and A5 remain `smoke_or_dev_only` until their own canonical pointers
are explicitly frozen. A1-F P2 is missing until
`checkpoints/rlm1_stripped/teacher_p2/none/seed0/canonical_checkpoint.yaml` is
created by an explicit freeze step. Controlled evaluation remains blocked until
readiness improves beyond A0 alone.

## Guardrails

- No health token, UQ, or CBF is active.
- No P2 runtime execution or P3 expansion is introduced by this policy.
- No A2/A5 training is part of T09-R2.
- No observed-result manifest update is implied.
- No paper-grade claim is made by creating this policy.
