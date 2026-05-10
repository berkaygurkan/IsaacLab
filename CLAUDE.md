# CLAUDE.md

## Project Memory

This repository supports the doctoral software project:

**Quadruped Fault Tolerant Control via Reinforcement Learning**

Current implementation priority is not the full thesis roadmap.
The active target is the **conference configuration**:

- method family: **RLM1**
- active variant: **RLM1 stripped**
- teacher-student: enabled
- residual learning: enabled
- health token: disabled

Future phases may exist, but they are not the present default.

---

## What Claude Should Optimize For Now

Claude should optimize for:

- practical repo progress,
- phase-correct organization,
- stable structure,
- reproducibility,
- maintainable task decomposition,
- compatibility with later expansion.

Claude should not optimize primarily for:
- P1 full health-token integration,
- P2 uncertainty extensions,
- P3 safety augmentations,
- broad method-family exploration.

---

## Read Order

Before making important changes, read:

1. `Selected_Method.md`
2. `System_Goals.md`
3. `Thesis_Main_Method.xlsx`
4. `PROJECT_CONTEXT_MAP.md`
5. `GIT_WORKFLOW.md`
6. `AGENTS.md`
7. `CLAUDE.md`

---

## Environment Assumptions

Canonical execution machine:
- Ubuntu 22.04.5 workstation

Canonical repo path:
- `~/thesis/IsaacLab`

Isaac Sim binary path:
- `~/isaacsim`

Canonical Python environment:
- conda environment `isaaclab`
- Python 3.11

Smoke test status:
- environment validated
- Isaac Sim connected
- Isaac Lab smoke test has already passed

Claude should assume the environment is available unless the user says it changed.

---

## Repository Philosophy

### 1. Current target first
The repository is currently being prepared around the conference implementation target, not the entire thesis roadmap.

### 2. Structure before expansion
Prefer:
- scaffolding,
- file contracts,
- configuration hygiene,
- workflow clarity,
before wide implementation growth.

### 3. Preserve extension paths
Even when building for RLM1 stripped, do not block future extension toward:
- RLM1 full,
- RLM1 + uncertainty channel,
- RLM1 + safety augmentation.

### 4. Small precise edits
Prefer focused changes over broad uncontrolled rewrites.

---

## Allowed Contributions

Claude may:
- create and revise repo-level documentation
- create agent configuration files
- create skill definitions
- create task prompts
- create workflow specifications
- define folder responsibilities
- define naming rules
- define reproducibility and experiment logging rules
- define deployment readiness checklists

Claude may also help reason about:
- repository structure
- agent division of labor
- configuration schema planning
- phase-aware implementation sequencing

---

## Forbidden Assumptions

Claude must not assume:
- health token is currently active
- conference target already includes full thesis features
- M-code taxonomy should appear in implementation naming
- algorithm design should be reinvented without instruction
- future roadmap phases should override current phase priorities

---

## Naming Rules

Use thesis method identifiers only:
- `RLM1` through `RLM9`

Current active naming:
- `RLM1_stripped`
- `use_health_token: false`

Do not create implementation-first names that obscure the thesis method identity.

---

## Git and Branch Discipline

Repository model:
- `origin` = user fork
- `upstream` = official IsaacLab repository

Claude must:
- avoid force-push
- avoid risky branch rewrites
- avoid editing Git remotes
- avoid working directly on `main` for experimental edits

Preferred style:
- create focused changes
- keep branch purpose narrow
- respect `GIT_WORKFLOW.md`

---

## Preferred Types of Claude Tasks

Claude is best used for:
- repo-wide reasoning
- multi-file instruction design
- architectural documentation
- task pack generation
- workflow standardization
- documentation that aligns agent behavior with thesis phase

Claude is less suitable than file-scoped agents for:
- tiny local edits that do not require architectural reasoning

---

## What “Good Output” Looks Like

A good output from Claude in this repository:

- reflects the active conference target,
- preserves future extensibility,
- is consistent with Selected_Method authority,
- is specific rather than generic,
- reduces ambiguity for later coding agents.

A bad output:

- speaks mostly about future phases,
- blurs the active target,
- introduces naming drift,
- or assumes features that are not currently enabled.

---

## Active Phase Reminder

Current stage:
- **Phase 0**
- **Phase 1**
- **Conference target**
- **RLM1 stripped**
- **health token off**

This should appear explicitly in prompts whenever task ambiguity exists.

---

## Claude Skill Strategy

Claude skills should be created only when a workflow is:
- repeatable,
- error-prone,
- multi-step,
- or likely to recur later.

Initial skill priorities:
1. Isaac Lab environment setup
2. Fault injection recipe
3. Evaluation metrics runner
4. Deployment checklist

---

## Final Reminder

Claude should treat this repository as a **conference-first, architecture-aware, future-compatible thesis software workspace**.

Present default:
- not full roadmap,
- not full token pipeline,
- not alternative method families,
- not safety augmentation first.

Present default:
- **RLM1 stripped**
- **practical implementation support**
- **clear agent-readable structure**