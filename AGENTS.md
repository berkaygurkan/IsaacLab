# AGENTS.md

## Purpose

This file is the primary onboarding document for coding agents working inside this repository.

It defines:

- what this repository is for,
- what the active implementation target is,
- which files are authoritative,
- what the agent is allowed to change,
- what the agent must not infer or redesign,
- and how tasks should be interpreted within the thesis timeline.

This is a software execution-context document, not a method-selection document.

---

## Repository Mission

This repository supports a doctoral software project:

**Quadruped Fault Tolerant Control via Reinforcement Learning**

Core stack:
- Isaac Sim / Isaac Lab
- Python 3.11
- PyTorch
- YAML / Hydra-style config-first workflow
- Ubuntu workstation execution
- Unitree A1 deployment target

The repository is used for:
- simulation environment development,
- fault injection infrastructure,
- training/evaluation organization,
- experiment reproducibility,
- deployment preparation,
- and coding-agent-assisted implementation.

---

## Active Implementation Target

### Current active target
**Conference paper configuration**

### Current thesis method family
**RLM1**

### Current active variant
**RLM1 stripped**

Interpretation:
- teacher-student backbone = active
- residual learning = active
- health token = disabled

This means the repository should currently prioritize implementation compatible with:

- no health token,
- no full P1/P2/P3 extensions,
- no UQ channel,
- no CBF augmentation,
- no method-family switching.

Agents may know the future roadmap, but must optimize for the current active target first.

---

## What the Agent Must Assume

Unless the user explicitly says otherwise, assume:

- the current implementation target is **Conf**
- the active method is **RLM1 stripped**
- the current task band is **Phase 0 + Phase 1**
- the user wants practical repository progress, not abstract redesign
- the repository should remain compatible with future RLM1-full expansion

---

## Read Order

Read files in the following order before proposing repo-wide changes:

1. `Selected_Method.md`
2. `System_Goals.md`
3. `Thesis_Main_Method.xlsx`
4. `PROJECT_CONTEXT_MAP.md`
5. `GIT_WORKFLOW.md`
6. `AGENTS.md`

If a conflict appears, authority is resolved in exactly that order.

---

## Current Phase Focus

### Phase 0
- repository skeleton
- environment sanity
- Isaac Lab / Isaac Sim stability
- fault injection abstractions
- config-first organization

### Phase 1
- healthy PPO baseline organization
- RLM1 stripped implementation scaffolding
- residual integration support
- fault curriculum organization
- conference metric and artifact structure

Do not prematurely optimize for:
- P1 full health-token pipeline
- P2 uncertainty channel
- P3 safety augmentation
unless the user explicitly asks for those phases.

---

## Scope Boundaries

### Allowed
The agent may:
- write repo documentation
- write agent configuration files
- write prompt packs
- write folder responsibility docs
- write reproducibility conventions
- write workflow instructions
- write experiment naming rules
- write environment setup instructions
- write implementation task breakdown files

### Not allowed
The agent must not:
- change the selected thesis method
- replace RLM naming with M-code naming
- redesign the algorithm from scratch
- assume health token is currently active
- treat future publication phases as the present default
- invent new method families without explicit request

---

## Current Method Guardrails

### Do use
- `RLM1`
- `RLM1_stripped`
- wording such as `use_health_token: false`

### Do not default to
- `RLM1_full`
- `RLM2`
- `RLM9`
- combined P2/P3 pipeline assumptions

### Important
The stripped conference variant is not a separate method family.
It is a controlled variant of **RLM1**.

---

## Repository Behavior Rules

### Rule 1 — Config-first
Prefer configuration, schema, pathing, and orchestration clarity before code expansion.

### Rule 2 — Minimal disruption
Prefer small, local, reversible edits.

### Rule 3 — Preserve future extensibility
Current work should not block future transition to:
- RLM1 full,
- RLM1 + UQ,
- RLM1 + safety augmentation.

### Rule 4 — Do not over-generalize
The current target is narrower than the full thesis roadmap.

### Rule 5 — Reproducibility matters
All task definitions should be compatible with:
- fixed seeds,
- stable naming,
- deterministic run bookkeeping,
- checkpoint traceability.

---

## File Types the Agent Should Expect

Important repo-level documents:
- `PROJECT_CONTEXT_MAP.md`
- `GIT_WORKFLOW.md`
- `AGENTS.md`
- `CLAUDE.md`
- `.cursorrules`

Prompt packs:
- `agent_prompts/T01_*.md`
- `agent_prompts/T02_*.md`
- ...

Claude skill files:
- `.claude/skills/*/SKILL.md`

Antigravity workflow files:
- `agent_workflows/antigravity/*.yaml`

---

## Preferred Style of Work

When creating new repo-facing documents:

- be explicit
- be phase-aware
- prioritize current target over roadmap breadth
- keep naming stable
- use English inside agent-facing files
- avoid vague placeholders when concrete project context is known

When writing task prompts:
- keep them execution-oriented
- mention active phase
- mention active method variant
- mention what is explicitly out of scope

---

## Git Discipline

Assume the repository follows this model:
- `origin` = personal fork
- `upstream` = official IsaacLab repository

The agent must not:
- force push
- rewrite `main`
- rename remotes
- make upstream sync decisions without user instruction

Read `GIT_WORKFLOW.md` before proposing Git operations.

---

## Machine Model

### PC1
MacBook used for:
- writing,
- reading,
- orchestration,
- remote editing.

### PC2
Ubuntu workstation used for:
- Isaac Sim / Isaac Lab execution,
- training,
- experiments,
- logs,
- checkpoints,
- deployment preparation.

Agents should treat PC2 as the canonical execution machine.

---

## Success Criterion for Current Stage

A successful agent contribution in the current stage is one that improves:

- onboarding clarity,
- repo organization,
- phase-correct task execution,
- reproducibility structure,
- conference-target implementation readiness.

A contribution is not successful if it mainly optimizes future phases while ignoring the current conference target.

---

## Final Operational Reminder

The repository is currently in a **conference-first implementation stage**.

The agent must optimize for:
- **RLM1 stripped**
- **health token off**
- **Phase 0 / Phase 1**
- **practical progress**
- **future-compatible structure**

Not for the full thesis scope by default.