# Conference Figure Plan

This plan lists paper-facing figure candidates for the RLM1 stripped conference-stage artifact set. Mermaid diagrams already exist in the architecture snapshots and can be converted later; this scaffold does not generate images.

| figure | purpose | source artifact | status | notes |
| --- | --- | --- | --- | --- |
| F1 RLM1 stripped pipeline overview | Show the completed T05-T09 method progression from healthy PPO through ablation infrastructure. | `papers/conference/architecture/rlm1_stripped_flow_up_to_t06.md`, `papers/conference/architecture/rlm1_stripped_flow_up_to_t07.md`, `papers/conference/architecture/rlm1_stripped_flow_up_to_t08.md`, and `papers/conference/artifact_index.md` | draftable | Use architecture flow snapshots; health token OFF, UQ inactive, and CBF inactive. |
| F2 Teacher-student-residual action flow | Show teacher reference, student policy, residual action delta, and final action composition. | `papers/conference/architecture/t06_teacher_snapshot.md`, `papers/conference/architecture/t07_student_distillation_snapshot.md`, and `papers/conference/architecture/t08_residual_snapshot.md` | draftable | Keep the teacher marked as privileged reference only; deployment-facing rows use policy observations only. |
| F3 T08 residual composition diagram | Show the residual PPO wrapper, residual scale, action dimensions, and diagnostics available from runtime validation. | `papers/conference/architecture/t08_residual_snapshot.md` and `papers/conference/architecture/t08_residual_runtime_validation.md` | draftable | A5 smoke can annotate infrastructure validation only, not final performance. |
| F4 T09 ablation infrastructure flow | Show A0-A6 matrix, dry-run validation, manifest scaffold, A5 smoke runner, and observed smoke manifest. | `papers/conference/architecture/t09_ablation_plan.md`, `papers/conference/architecture/t09_smoke_runner_note.md`, `papers/conference/architecture/t09_ablation_infrastructure_snapshot.md`, `papers/conference/results/t09_ablation_manifest_template.md`, and `papers/conference/results/t09_ablation_observed_smoke_manifest.md` | draftable / pending data | A5 is observed_smoke; A0/A1/A2/A3/A4/A6 remain pending and no full sweep has been run. |

## Figure Guardrails

- Do not generate images in this scaffold.
- Do not present pending ablations as observed.
- Do not claim fault tolerance, health token, UQ, CBF, multi-seed statistics, or real robot deployment.
- Any table or figure using A5 should label it as one-row smoke infrastructure validation.
