# deployment-checklist

## Purpose

Use this skill when working on deployment-readiness, export preparation, runtime validation, or real-robot transition support tasks inside the thesis repository.

This skill helps Claude Code discuss deployment-related topics without confusing future deployment goals with the current active implementation target.

It is intended for structured preparation, checklist design, and repository-facing deployment guidance.

It is not a signal that deployment is the current default priority.

---

## When to Use This Skill

Use this skill when:

- writing deployment-readiness documentation
- preparing A1-related checklist files
- describing export-related repository structure
- organizing latency, runtime, or deployment validation notes
- framing real-robot preparation tasks
- documenting what should be checked before moving from sim to deployment
- writing prompt packs related to deployment preparation

Use this skill before:
- deployment checklists
- export pipeline notes
- latency profiling guidance
- recovery-readiness checklists
- deployment folder responsibility docs

---

## When Not to Use This Skill

Do not use this skill for:

- changing the active thesis phase
- assuming deployment is the present default target
- redesigning the policy architecture
- inventing a new deployment method
- activating later publication stages without instruction
- replacing the authority of Selected_Method.md

This skill supports deployment-aware preparation.
It does not activate deployment as the current default priority.

---

## Active Project Assumptions

Unless the user explicitly changes phase, assume:

- active method family is **RLM1**
- active variant is **RLM1 stripped**
- health token is OFF
- current implementation target is the **conference paper**
- active work band is **Phase 0 / Phase 1**
- deployment is an important future-facing constraint, but not the active default objective

Do not assume:
- real robot execution is the immediate task
- health-token full deployment is current scope
- uncertainty-aware deployment is current scope
- safety-augmented deployment is current default

---

## Current Role of Deployment in the Repository

At the current stage, deployment should be treated as:

- a downstream constraint
- a future compatibility target
- a source of practical design discipline
- a checklist-driven preparation area
- a reason to keep interfaces, outputs, and naming clean

At the current stage, deployment should not dominate:
- repository structure decisions
- conference-stage task scope
- present-phase abstraction choices

---

## Deployment-Relevant Concerns

When using this skill, Claude should be aware of deployment-oriented concerns such as:

- export path clarity
- runtime assumptions
- latency awareness
- checkpoint traceability
- model/version naming
- inference-side simplicity
- recovery-oriented observability
- compatibility with future Unitree A1 execution

These concerns should be acknowledged without turning the current stage into a full deployment project.

---

## Preferred Deployment Framing

When this skill is used, Claude should prefer the following framing:

### 1. Deployment as constraint, not current phase override
Keep deployment visible, but do not let it replace the active conference target.

### 2. Checklist-first
Prefer checklists, readiness notes, and validation structure over speculative deployment redesign.

### 3. Future-compatible structure
Encourage file organization and naming that will later help export and A1 testing.

### 4. Explicit scope labeling
Clearly distinguish:
- current conference-stage needs
- future real-robot preparation
- deferred advanced deployment features

### 5. Practical realism
Prefer concrete operational concerns over abstract “deployable system” language.

---

## Good Current-Stage Output Types

Good outputs include:

- deployment readiness checklists
- A1 preparation notes
- export pipeline skeleton docs
- latency profiling checklist structure
- artifact traceability rules
- pre-deployment validation notes
- “current vs future deployment scope” guidance

Bad outputs include:

- treating deployment as already active default
- requiring full robot-side implementation immediately
- assuming uncertainty channel is already part of deployment
- assuming safety augmentation is already active
- replacing conference-stage priorities with hardware-stage priorities

---

## Recommended Deployment Checklist Concepts

When describing deployment-related work, Claude should favor concepts such as:

- export readiness
- checkpoint identity and traceability
- inference latency checks
- runtime dependency clarity
- hardware-facing validation steps
- rollback readiness
- recovery behavior observation
- experiment-to-deployment mapping

Avoid vague labels such as:
- “real-world ready”
- “production deployment”
unless they are broken down into explicit checks.

---

## Output Style Rules

When using this skill:

- keep deployment grounded and practical
- keep present-phase and future-phase clearly separated
- use checklist language when possible
- avoid pretending that real-robot execution is already the current milestone
- preserve compatibility with the active conference-stage implementation target

---

## Scope Guardrails

### Allowed
Claude may:
- write deployment-facing documentation
- create readiness checklists
- describe export-related repository expectations
- define A1 preparation notes
- define latency or runtime validation headings
- support future deployment compatibility through documentation and structure

### Not allowed
Claude must not:
- declare deployment the current primary milestone without instruction
- assume full thesis deployment stack is active now
- assume uncertainty or safety augmentations are current defaults
- replace current conference-stage priorities with hardware-first priorities
- turn deployment docs into method redesign documents

---

## Relationship to Other Skills

This skill complements:

- `isaac-lab-env-setup`
  - for environment validation
- `fault-injection-recipe`
  - for fault scenario structure
- `eval-metrics-runner`
  - for conference-stage result organization

This skill focuses on future-facing deployment readiness without breaking current-phase focus.

---

## Phase Reminder

Current default:
- conference-first
- RLM1 stripped
- health token off
- Phase 0 / Phase 1
- deployment-aware, but not deployment-first

If deployment-related tasks are discussed, always keep the active phase visible.

---

## What Good Use of This Skill Looks Like

A good result:

- keeps deployment concerns visible but bounded
- supports future A1 transition
- does not derail current conference-stage progress
- improves readiness and traceability
- helps future coding agents understand what is current vs deferred

A bad result:

- promotes deployment to the default target
- overbuilds hardware-specific structure too early
- confuses current and future milestones
- assumes inactive method extensions are already enabled

---

## Final Rule

Default to deployment-aware preparation, not deployment-first execution.

Support the current conference-stage target first.
Keep future A1 readiness explicit but bounded.