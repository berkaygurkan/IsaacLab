# Conference Ablation Table Draft

Scope: Draft table from T09 infrastructure; observed metrics currently available only for A5 smoke.

This table is an artifact scaffold for paper drafting. It records row identity and status only; it does not invent performance metrics for pending rows, and no full sweep has been run.

| ablation_id | row_name | method_stage | deployment_facing | residual_scale | checkpoint_status | observed_status | observed_metric_status | paper_role | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A0 | `healthy_ppo` | `healthy_baseline` | yes | n/a | checkpoint pointer ready | pending | pending | Healthy PPO baseline | Baseline row from T09 matrix; no observed metrics recorded for this draft table. |
| A1 | `privileged_teacher_reference` | `teacher` | no | n/a | checkpoint pointer ready | pending | pending | Privileged teacher reference | Reference row only; not a deployment policy and no observed metrics recorded. |
| A2 | `student_distilled_no_residual` | `student` | yes | n/a | checkpoint pointer ready | pending | pending | Student without residual | Deployment-facing student row; no observed metrics recorded. |
| A3 | `student_residual_scale_0_0` | `residual` | yes | 0.0 | checkpoint pointer ready | pending | pending | Residual runtime control | Residual wrapper with zero residual contribution; no observed metrics recorded. |
| A4 | `student_residual_scale_0_05` | `residual` | yes | 0.05 | checkpoint pointer ready | pending | pending | Residual scale sensitivity | Small residual scale sensitivity row; no observed metrics recorded. |
| A5 | `student_residual_scale_0_1` | `residual` | yes | 0.1 | checkpoint pointer ready | observed_smoke | observed_smoke_metrics_available | Default residual baseline | One-row smoke passed with recorded infrastructure diagnostics; this is not a final performance result. |
| A6 | `student_residual_scale_0_2` | `residual` | yes | 0.2 | checkpoint pointer ready | pending | pending | Residual scale sensitivity | Larger residual scale sensitivity row; no observed metrics recorded. |

## Guardrails

- A5 is marked `observed_smoke` only.
- A0, A1, A2, A3, A4, and A6 remain `pending`.
- Pending rows must not be interpreted as failed or completed results.
- This table is a draft artifact index for conference writing, not a final result table.
