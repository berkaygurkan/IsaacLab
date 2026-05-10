# ANTIGRAVITY_GUIDE.md

## Purpose

This file explains how Antigravity should be used in this repository.

Antigravity is not the first-choice tool for every task.
It is most useful for orchestrating structured, multi-step workflows.

This file defines:
- when to use Antigravity,
- when not to use it,
- what the current active thesis target is,
- and how workflow design should stay aligned with the conference stage.

---

## Active Thesis Target

Current implementation target:
- **Conference paper configuration**
- **RLM1 stripped**

Interpretation:
- teacher-student = enabled
- residual = enabled
- health token = disabled

Antigravity workflows must optimize for the current active target first.

Future phases may be known, but they are not the default execution target.

---

## Best Use Cases for Antigravity

Antigravity is best used for:

- multi-step repository setup workflows
- repeated validation procedures
- batch experiment orchestration
- run-check-collect-report pipelines
- artifact packaging workflows
- multi-stage documentation generation

Antigravity is not the best first choice for:
- very small file edits
- isolated single-file changes
- tiny prompt rewrites
- one-line corrections

Use Codex or Claude Code for those instead.

---

## Current Recommended Antigravity Scope

At the current stage, Antigravity should focus on workflows related to:

- environment validation
- repo bootstrap
- conference-stage task sequencing
- fault-injection experiment orchestration
- metric collection organization
- artifact preparation for the conference paper

Do not prioritize workflows for:
- health-token full pipeline,
- uncertainty augmentation,
- safety-shield integration,
unless explicitly requested.

---

## Read Order

Before designing workflows, read:

1. `Selected_Method.md`
2. `System_Goals.md`
3. `Thesis_Main_Method.xlsx`
4. `PROJECT_CONTEXT_MAP.md`
5. `AGENTS.md`
6. `ANTIGRAVITY_GUIDE.md`

---

## Workflow Design Rules

### Rule 1 — Phase-correctness
Workflows must align with the current conference target.

### Rule 2 — Reusability
A workflow should be reusable and not tied to one accidental path layout unless that layout is canonical.

### Rule 3 — Explicit inputs/outputs
Each workflow should clearly define:
- inputs,
- steps,
- outputs,
- stop conditions,
- success conditions.

### Rule 4 — Minimal surprise
Prefer transparent, inspectable workflow stages.

### Rule 5 — Repo compatibility
Workflow files should fit the repository structure and should not assume external undocumented context.

---

## Suggested Workflow Directory

Recommended path:
```text
agent_workflows/antigravity/