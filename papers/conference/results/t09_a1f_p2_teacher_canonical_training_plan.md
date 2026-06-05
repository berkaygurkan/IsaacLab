# T09 A1-F P2 Teacher Canonical Training Plan

Scope: A1-F P2 privileged teacher checkpoint-generation scaffold; no training,
evaluation, play, Isaac Sim launch, checkpoint freeze, pointer edit, observed
manifest update, or paper-grade claim.

## Objective

T09-R2e prepares the canonical A1-F teacher workflow for the P2-only conference
scope. A1-F is the future privileged teacher trained under `P2_locked_joint`,
with default target joint `front_left_foot` and fault onset step `50`. The
method remains RLM1 stripped, health token OFF, UQ inactive, and CBF inactive.
P4 torque degradation is deferred to advisor-demo / thesis-extension material.

## A1-H Versus A1-F

A1-H is the earlier healthy/no-fault teacher pretrain/reference. It may be
useful as development context, but it is not the final fault-aware teacher for
the P2 conference pipeline. A1-F is the future P2 fault-aware privileged teacher
that A2 student distillation should use.

Do not train A2 from the A1-H smoke/development pointer. A2, A5, and optional
A7 remain blocked until A1-F is trained or selected and explicitly frozen as a
canonical checkpoint.

## Training Helper

Use:

```bash
bash trainers/run_t09_train_a1f_p2_teacher_canonical.sh
```

Default settings:

```text
config: configs/train/teacher_p2_canonical.yaml
fault_config: configs/fault/joint_lock/p2_locked_joint.yaml
task: Isaac-Ant-Teacher-v0
method: rlm1_stripped
fault_profile: P2_locked_joint
target_joint: front_left_foot
fault_onset_step: 50
num_envs: 4096
max_iterations: 2000
seed: 0
experiment_name: teacher_p2__rlm1_stripped__canonical
run_name: a1f_p2_teacher_canonical__seed0
```

The helper delegates to the existing T06 teacher path and passes
`--skip_checkpoint_pointer`, so raw training does not overwrite stable pointer
files. It accepts `--fault_config`, `--target_joint`, and
`--fault_onset_step`, then routes training through the repo-owned P2 wrapper by
passing `--enable_p2_joint_lock` to `trainers/rsl_rl_train.py`.

The T09-R2f hook is wired as an action-override surrogate, not a true mechanical
joint-position lock:

```text
trainers/p2_joint_lock_training_wrapper.py
semantics: action_override_zero_effort_surrogate
```

Before treating any run from this helper as canonical A1-F P2 evidence, run the
P2 preflight and verify that the hook attaches, maps `front_left_foot`, and
activates after onset.

Expected log root:

```text
logs/rsl_rl/teacher_p2__rlm1_stripped__canonical/
```

## Dry-Run And Preflight

Safe dry-run:

```bash
bash trainers/run_t09_train_a1f_p2_teacher_canonical.sh --dry_run
```

Runtime preflight, still no training:

```bash
python evaluators/preflight_t09_p2_joint_lock.py --execute_preflight --headless
```

Full training remains blocked unless the user passes both
`--p2_preflight_passed` and `--allow_full_training`. A tiny one-iteration smoke
is available only through `--execute_one_step_smoke --p2_preflight_passed`.

## Freeze Step

After a valid P2 teacher run exists, select one exact `model_*.pt` path
manually. Freeze it with:

```bash
python evaluators/freeze_t09_checkpoint.py \
  --stage teacher_p2 \
  --method rlm1_stripped \
  --fault none \
  --seed 0 \
  --task Isaac-Ant-Teacher-v0 \
  --checkpoint_path <exact_model.pt> \
  --experiment_name teacher_p2__rlm1_stripped__canonical \
  --run_name a1f_p2_teacher_canonical__seed0 \
  --classification paper_grade_candidate \
  --output_pointer checkpoints/rlm1_stripped/teacher_p2/none/seed0/canonical_checkpoint.yaml \
  --manifest_path papers/conference/results/t09_a1f_p2_teacher_canonical_checkpoint_freeze.md
```

The chosen A1-F pointer policy is:

```text
checkpoints/rlm1_stripped/teacher_p2/none/seed0/canonical_checkpoint.yaml
```

The freeze helper refuses `paper_grade_candidate` for `model_0.pt` and
demo/advisor paths. It does not overwrite `latest_checkpoint.yaml`.

## Privilege And Deployment Role

A1-F may use privileged fault information during training because it is a
teacher/reference policy. It is not deployment-facing. A2 and the future
deployment-facing residual methods must not require teacher observations or
`true_fault_state` at runtime.

## Blocked Work After T09-R2e

- A2 P2 student distillation remains blocked until A1-F is canonical.
- A5 P2 residual training remains blocked until A2 is canonical.
- A7 optional teacher-distilled residual remains blocked until A0 and A1-F are
  canonical and an explicit A7 promotion decision is made.
- Controlled evaluation remains blocked until A1-F, A2, and A5 readiness
  improves and a guarded P2 runtime evaluator exists.

## Guardrails

- Do not train in this scaffold step.
- Do not freeze any checkpoint in this scaffold step.
- Do not overwrite checkpoint pointers.
- Do not launch evaluation or play.
- Do not modify RSL-RL core, Isaac Lab core, or task registration.
- Do not add health token, UQ, CBF, P3, or P4 main scope.
- Do not make paper-grade claims.
