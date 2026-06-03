# T09-E0 Fault-Aware Controlled Evaluation Plan

Scope: T09-E0 dry-run-only preflight scaffold; RLM1 stripped; conference-stage only; health token OFF; UQ inactive; CBF inactive.

## Objective

T09-E0 prepares future fault-aware controlled evaluation without executing it. It defines evaluation intent, allowed fault profiles, row roles, checkpoint dependencies, and command previews. It does not run Isaac Sim, training, evaluation, runtime fault injection, checkpoint writes, metric parsing, or observed-result manifest updates.

Checkpoint pointer interpretation and residual checkpoint promotion policy are frozen separately in `papers/conference/results/t09_evaluation_checkpoint_policy.md`.

## Fault Profiles

| profile_id | profile_family | status | scaffold | notes |
| --- | --- | --- | --- | --- |
| `F0_none` | `none` | `active_for_preview` | n/a | Healthy/no-fault reference profile. |
| `P4_torque_degradation` | `torque_scale` | `active_for_preview` | `configs/fault/torque_scale/p4_torque_degradation.yaml` | First planned fault-aware pilot profile; preview-only in T09-E0. |
| `P2_locked_joint` | `joint_lock` | `gated` | `configs/fault/joint_lock/p2_locked_joint.yaml` | Gated until `P4_torque_degradation_pilot_passed`; not executable in T09-E0. |

No runtime fault injection behavior is implemented in T09-E0.

## Row Roles

| ablation_id | row_name | T09-E0 role | deployment_facing | first preview |
| --- | --- | --- | --- | --- |
| A0 | `healthy_ppo` | Zero-shot healthy PPO baseline under faults. | yes | P4 torque degradation preview. |
| A1 | `privileged_teacher_reference` | Privileged reference only. | no | Excluded from deployment-facing P4 pilot previews. |
| A2 | `student_distilled_no_residual` | Frozen student fault evaluation; no online learning. | yes | P4 torque degradation preview. |
| A3 | `student_residual_scale_0_0` | Residual-scale readiness row. | yes | Residual pointer readiness only in T09-E0. |
| A4 | `student_residual_scale_0_05` | Residual-scale readiness row. | yes | Residual pointer readiness only in T09-E0. |
| A5 | `student_residual_scale_0_1` | Frozen residual-policy adaptation; no online learning. | yes | P4 torque degradation preview. |
| A6 | `student_residual_scale_0_2` | Residual-scale readiness row. | yes | Residual pointer readiness only in T09-E0. |

A3/A4/A5/A6 reuse the same residual checkpoint and vary only the matrix-defined `residual_scale` at evaluation time.

## Planned Dry-Run Commands

```bash
python evaluators/run_t09_fault_aware_execution.py --dry_run --verify_profiles
python evaluators/run_t09_fault_aware_execution.py --dry_run --preview --ablation_id A0 --fault_profile P4_torque_degradation --seed 0
python evaluators/run_t09_fault_aware_execution.py --dry_run --preview --ablation_id A2 --fault_profile P4_torque_degradation --seed 0
python evaluators/run_t09_fault_aware_execution.py --dry_run --preview --ablation_id A5 --fault_profile P4_torque_degradation --seed 0
python evaluators/run_t09_fault_aware_execution.py --dry_run --preview_gated --fault_profile P2_locked_joint --seed 0
```

## Guardrails

- Default mode is dry-run.
- No execution flag is added in T09-E0.
- P4 is preview-only, not executable.
- P2 is gated and preview-gated only.
- A0 is labelled zero-shot under faults.
- A2 is labelled frozen student.
- A5 is labelled frozen residual-policy adaptation.
- A1 remains privileged reference only, not deployment-facing.
- No multi-fault combinations or new ablation rows.
- No training, evaluation, Isaac Sim launch, runtime fault injection, checkpoint writes, metric parser, paper-grade export, or observed-result manifest update.

## Completion Criteria

T09-E0 is complete when the dry-run validator can validate profile metadata and gate state, confirm P4/P2 scaffold file presence, resolve A0/A2/A5 pilot checkpoint dependencies, resolve residual plus student checkpoint dependencies for A3/A4/A5/A6 readiness, produce P4 previews for A0/A2/A5, show P2 as gated, and state that no training/evaluation/Isaac Sim/metrics/observed updates occurred.

## Suggested Next Step

T09-E1 should add a still-guarded P4 pilot scaffold with the minimal execution gate and logging structure needed for tiny P4 load/eval smoke. P2 remains locked until the P4 pilot has a real pass/fail record.
