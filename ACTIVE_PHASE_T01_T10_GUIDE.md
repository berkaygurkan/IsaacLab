# ACTIVE_PHASE_T01_T10_GUIDE.md

## Purpose

This file summarizes the currently active implementation band for the thesis repository.

It is intended for:
- GPT Project context
- coding-agent onboarding
- task prioritization
- phase-aware execution guidance

This file focuses only on the currently active tasks:
- **T01 to T10**

It does not replace:
- `Selected_Method.md`
- `System_Goals.md`
- `Thesis_Main_Method.xlsx`

It is a compact operational guide for the present implementation stage.

---

## Active Thesis Context

### Current active implementation target
- **Conference-stage implementation**

### Current active thesis method family
- **RLM1**

### Current active variant
- **RLM1 stripped**

### Interpretation
- teacher-student backbone: active
- residual learning: active
- health token: **OFF**

### Conference meaning
The conference paper is the stripped version of the main RLM1 line:
- no health token
- no uncertainty channel
- no safety shield
- no later-phase deployment-first assumptions

### Important rule
Future publication stages may be known, but they are **not** the current default.

The repository should currently optimize for:
- conference-stage progress
- Phase 0 / Phase 1 work
- practical implementation support
- future-compatible structure without premature expansion

---

## Why T01-T10 Matter

T01-T10 define the current implementation spine of the project.

They establish:
- repository structure
- execution environment stability
- fault-injection scaffolding
- config-first organization
- healthy baseline reference
- teacher policy stage
- student distillation stage
- residual adaptation stage
- conference curriculum and ablation logic
- conference metric and artifact pipeline

Together, they form the minimum coherent implementation package for the conference-stage target.

---

## Task Summary

### T01 — Repository Bootstrap
Goal:
Create or refine the repository skeleton so the project is easy to navigate, stable for future agents, and aligned with the current conference-stage target.

Focus:
- folder structure
- folder responsibilities
- clean repository organization
- minimal current-stage scaffolding

Not the goal:
- full future-proof expansion
- method redesign

---

### T02 — Isaac Lab Environment Setup and Sanity Validation
Goal:
Validate that the canonical Ubuntu execution environment is healthy and ready for active work.

Focus:
- conda env correctness
- Python version correctness
- Isaac Sim linkage
- path validation
- smoke-test capability

Not the goal:
- unnecessary reinstall
- deployment work

---

### T03 — Fault Injection Abstraction
Goal:
Define the repository-facing structure for current-stage fault injection.

Current fault priority:
- P2-like locked-joint scenarios
- P4-like torque degradation scenarios
- optionally limited simple surrogate fault cases only if needed for current-stage clarity

Focus:
- abstraction clarity
- config-aware structure
- current-vs-future scope separation

Not the goal:
- all future fault combinations
- uncertainty or safety augmentation

---

### T04 — Config-First Training Organization
Goal:
Organize the training-facing structure around reproducible configuration files.

Focus:
- environment config
- fault config
- method config
- training config
- evaluation config
- seed and reproducibility discipline

Not the goal:
- configuration sprawl
- later-phase overengineering

---

### T05 — Healthy PPO Baseline
Goal:
Define and preserve the healthy baseline run that serves as the reference point for current conference-stage comparisons.

Focus:
- baseline identity
- baseline checkpoint
- baseline logs
- baseline metadata
- reproducible naming

Not the goal:
- ablations
- token-enabled variants

---

### T06 — Teacher Policy Training on Ant v5
Goal:
Frame and support the privileged teacher-policy stage for the conference pipeline on Ant v5.

Focus:
- teacher policy as the first stage of the two-stage adaptation pipeline
- conference-compatible scope
- clean separation from later health-token extensions
- compatibility with later student-stage distillation

Not the goal:
- full RLM1-full expansion
- deployment-first work
- token-enabled implementation

---

### T07 — Student History Encoder and DAgger Distillation
Goal:
Frame and support the history-based student stage that learns from the teacher through conference-stage distillation.

Focus:
- student policy role in the stripped conference pipeline
- history-based encoding
- DAgger/distillation-oriented task framing
- conference-compatible structure and outputs

Not the goal:
- health-token conditioning
- uncertainty-aware extensions
- full thesis-stage complexity

---

### T08 — Residual Head
Goal:
Define the residual adaptation stage for the conference pipeline after teacher-student organization is in place.

Focus:
- residual learning as an explicit conference-stage component
- compatibility with the stripped RLM1 interpretation
- clean relation to teacher and student stages
- conference-ready implementation framing

Not the goal:
- token-enabled residual design
- safety or uncertainty augmentation
- later-stage architecture inflation

---

### T09 — Fault Curriculum and Conference Ablation Suite
Goal:
Define the current-stage curriculum logic for fault exposure and the conference-stage ablation structure.

Preferred curriculum direction:
- healthy
- mild fault
- severe fault

Typical conference ablations:
- PPO baseline
- residual-only
- no-teacher
- no-residual

Focus:
- phase-correct curriculum framing
- scenario progression
- interpretable experiment planning
- bounded conference ablation matrix

Not the goal:
- full-thesis curriculum generalization
- later-phase ablation expansion
- uncertainty / safety-stage comparisons

---

### T10 — Conference Metric Suite and Paper Artifact Pipeline
Goal:
Define the metric set and the output-to-paper artifact flow for the conference target.

Preferred current metrics:
- success rate
- return
- recovery time
- tracking error

Artifact focus:
- logs to summaries
- summaries to tables/figures
- artifact naming
- traceability
- paper-facing structure

Focus:
- bounded conference metrics
- artifact-ready grouping
- present-phase clarity
- direct support for conference figures and tables

Not the goal:
- uncertainty metrics by default
- safety-shield metrics by default
- full-thesis reporting breadth
- later-phase publication pipeline complexity

---

## Current Priority Order

Recommended active order:

1. T01 — repository structure
2. T02 — environment validation
3. T03 — fault abstraction
4. T04 — config-first organization
5. T05 — healthy baseline
6. T06 — teacher policy training
7. T07 — student history encoder + distillation
8. T08 — residual head
9. T09 — fault curriculum + conference ablation suite
10. T10 — conference metric suite + artifact pipeline

This order supports stable implementation progress.

---

## Current Conference Logic

The conference paper should validate the stripped core of the RLM1 family.

Current conference logic is:

1. establish a healthy baseline
2. define bounded fault scenarios
3. organize teacher policy stage
4. organize student distillation stage
5. add residual adaptation
6. run healthy-to-fault curriculum
7. compare against bounded conference baselines
8. report practical conference metrics
9. produce paper-ready artifacts

This is the current implementation spine.

---

## What Is In Scope Right Now

Currently in scope:
- conference-stage structure
- RLM1 stripped framing
- health token OFF
- teacher policy stage
- student distillation stage
- residual head stage
- bounded fault-injection scaffolding
- config-first setup
- healthy baseline
- conference ablations
- bounded evaluation and artifact planning

---

## What Is Not the Current Default

Not current default:
- RLM1 full with active health token
- uncertainty channel integration
- safety augmentation / shield pipeline
- later publication phases
- deployment-first work
- all future fault combinations
- full A1 journal-stage breadth

These may exist in the roadmap, but they are not the present target.

---

## How GPT Project Should Use This File

When answering questions about current implementation priorities, GPT Project should:

1. treat this file as the compact guide for the active task band
2. assume conference-stage work unless the user explicitly switches target
3. interpret T01-T10 as the present implementation spine
4. keep teacher-student + residual central
5. keep health token OFF by default
6. avoid drifting into later-phase assumptions

If there is a conflict, authority order remains:

1. `Selected_Method.md`
2. `System_Goals.md`
3. `Thesis_Main_Method.xlsx`
4. `ACTIVE_PHASE_T01_T10_GUIDE.md`

---

## Final Reminder

The repository is currently in a **conference-first implementation stage**.

The current default is:

- RLM1 stripped
- teacher-student active
- residual active
- health token OFF
- Phase 0 / Phase 1
- practical progress over roadmap breadth

All task interpretation should begin from that assumption.