# Conference Artifact Index

Scope: RLM1 stripped conference-stage artifacts; health token OFF; UQ inactive; CBF inactive.

## Method Pipeline Artifacts

| stage | artifact | role |
| --- | --- | --- |
| T05 healthy baseline | `papers/conference/architecture/t05_healthy_baseline_snapshot.md` | Healthy PPO reference for the RLM1 stripped conference-stage method. |
| T06 privileged teacher | `papers/conference/architecture/t06_teacher_snapshot.md` | Privileged teacher architecture snapshot and reference stage. |
| T07 student distillation | `papers/conference/architecture/t07_student_distillation_snapshot.md` | Student policy distillation snapshot for deployment-facing policy flow. |
| T08 residual PPO | `papers/conference/architecture/t08_residual_snapshot.md` and `papers/conference/architecture/t08_residual_runtime_validation.md` | Residual policy composition and runtime validation scaffold. |
| T09 ablation infrastructure | `papers/conference/architecture/t09_ablation_plan.md`, `papers/conference/architecture/t09_smoke_runner_note.md`, and `papers/conference/architecture/t09_ablation_infrastructure_snapshot.md` | A0-A6 ablation matrix, manifest scaffold, A5 smoke runner note, and infrastructure state. |

Flow snapshots:

- `papers/conference/architecture/rlm1_stripped_flow_up_to_t06.md`
- `papers/conference/architecture/rlm1_stripped_flow_up_to_t07.md`
- `papers/conference/architecture/rlm1_stripped_flow_up_to_t08.md`

## Checkpoint Pointers

These are checkpoint pointer/manifest files, not model files. They identify the stable model-file references used by the documented artifacts.

- `checkpoints/rlm1_stripped/healthy_baseline/none/seed0/latest_checkpoint.yaml`
- `checkpoints/rlm1_stripped/teacher/none/seed0/latest_checkpoint.yaml`
- `checkpoints/rlm1_stripped/student/none/seed0/latest_checkpoint.yaml`
- `checkpoints/rlm1_stripped/residual/none/seed0/latest_checkpoint.yaml`

## Result Artifacts

- T09 manifest template: `papers/conference/results/t09_ablation_manifest_template.md`
- T09 observed A5 smoke manifest: `papers/conference/results/t09_ablation_observed_smoke_manifest.md`

Observed result coverage is currently limited to the A5 one-row smoke result. A0, A1, A2, A3, A4, and A6 remain pending; no full sweep has been run.

## Paper-Ready Figures/Tables Planned

- Architecture flow figure from the RLM1 stripped flow snapshots.
- Residual composition figure from the T08 residual architecture and runtime validation artifacts.
- Ablation matrix table from the T09 manifest template and infrastructure snapshot.
- Observed smoke result row from the T09 observed A5 smoke manifest.

## Pending / Not Yet Claimed

- Full A0-A6 observed results.
- Full evaluation runner.
- Faulted scenario metrics.
- Multi-seed statistics.
- Real robot deployment.

## Guardrails

- Do not claim full ablation sweep.
- Do not claim fault tolerance results yet.
- Do not claim health token/UQ/CBF.
- A5 smoke is an infrastructure validation, not a final performance result.
