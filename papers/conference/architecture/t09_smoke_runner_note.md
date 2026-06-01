# T09-C One-Row Ablation Smoke Runner Note

Scope: Conference-stage RLM1 stripped; one-row smoke runner only; health token OFF.

## Purpose

T09-C adds a tightly guarded one-row smoke runner for the conference ablation matrix. The runner consumes `configs/ablation/t09_conference_matrix.yaml`, selects exactly one ablation row, resolves its checkpoint pointer dependencies, and either previews the command or executes the one supported smoke path. The default behavior is dry-run preview only.

## Supported Execution

Only A5, `student_residual_scale_0_1`, is executable in the initial T09-C scaffold. A5 maps to the already validated T08 residual smoke path:

```bash
TERM=xterm ./trainers/run_t08_residual_train.sh --headless --num_envs 8 --max_iterations 1 --residual_scale 0.1
```

A0, A1, A2, A3, A4, and A6 are preview-only for now. If execution is requested for those rows, the runner refuses clearly and leaves them deferred to later T09-C extensions.

## Observed Smoke Note

T09-D1 records one observed A5 smoke supplied by the user in `papers/conference/results/t09_ablation_observed_smoke_manifest.md`. A0, A1, A2, A3, A4, and A6 remain pending.

## Example Commands

Dry-run preview:

```bash
./trainers/run_t09_ablation_smoke.sh --ablation_id A5 --dry_run
```

Execute A5 smoke:

```bash
./trainers/run_t09_ablation_smoke.sh --ablation_id A5 --execute_smoke --num_envs 8 --max_iterations 1
```

## Guardrails

- No full ablation sweep is implemented in T09-C.
- No evaluation runner is implemented in T09-C.
- No result collection or metric writing is implemented in T09-C.
- No metrics are fabricated.
- The runner does not import Isaac Sim directly.
- Real execution requires `--execute_smoke` and a selected row.
- Initial execution support is restricted to A5.
- Health token, uncertainty channel, safety shield / CBF, and P1/P2/P3 remain inactive.

## Deferred

- A0/A1/A2/A3/A4/A6 execution support.
- Full ablation sweeps.
- Full evaluation runner.
- T09-D observed result collection.
- T09.5 ablation documentation snapshot.
- Fault curriculum.
- Health token.
- Uncertainty channel.
- Safety shield / CBF.
- P1/P2/P3 expansion.
- New methods beyond A0-A6.
