# fault-injection-recipe

## Purpose

Use this skill when working on fault-injection-related tasks inside the thesis repository.

This skill helps Claude Code stay aligned with the current implementation stage when discussing, organizing, or documenting fault injection workflows.

It is intended for repository structure, task framing, configuration guidance, and implementation-scaffolding support.

It is not a license to redesign the thesis method or invent a new fault-tolerance framework.

---

## When to Use This Skill

Use this skill when:

- defining fault-injection-related repository structure
- writing prompts for fault-injection tasks
- creating documentation for fault abstractions
- planning fault scenario organization
- describing configuration responsibilities for fault parameters
- preparing evaluation workflows that depend on fault scenarios
- discussing Phase 0 / Phase 1 fault setup for the conference target

Use this skill before:
- proposing fault-related folder layouts
- drafting fault scenario prompt packs
- describing how agents should interpret current fault priorities
- writing reusable fault-injection documentation

---

## When Not to Use This Skill

Do not use this skill for:

- selecting the thesis method family
- redesigning the reinforcement learning algorithm
- inventing new fault taxonomy systems
- changing the official meaning of RLM1
- expanding automatically into future-phase methods
- replacing the authority of Selected_Method.md or Thesis_Main_Method.xlsx

This skill is about structured fault-injection support, not method invention.

---

## Active Project Assumptions

Unless the user explicitly switches phase or method target, assume:

- active method family is **RLM1**
- active variant is **RLM1 stripped**
- health token is OFF
- current implementation target is the **conference paper**
- active work band is **Phase 0 / Phase 1**
- current goal is practical fault-injection scaffolding, not full future-phase generalization

Fault-related work must support the current conference target first.

---

## Current Fault-Injection Role in the Repository

At the current stage, fault injection is primarily used for:

- building fault abstractions
- organizing configurable scenario definitions
- preparing training/evaluation structure
- supporting conference-stage experiments
- enabling later expansion without overbuilding too early

At this stage, fault injection should be treated as:
- explicit,
- configurable,
- bounded,
- and phase-aware.

It should not be treated as an excuse to build a full future-proof mega-framework before current needs are satisfied.

---

## Initial Fault Priority Assumptions

For the current conference-stage setup, assume the most relevant initial fault categories are:

- **P2** — locked joint
- **P4** — torque degradation / weakened actuator
- **P5** — free-swinging passive joint surrogate

These should be treated as the practical initial core unless the user explicitly changes scope.

Do not automatically assume:
- every possible combined fault case must be implemented immediately
- later-stage multi-fault complexity is required now
- future hardware-stage realism must fully define current abstraction structure

---

## Preferred Fault-Injection Framing

When this skill is used, Claude should prefer the following framing:

### 1. Abstraction-first
Define clear fault abstractions before discussing large experiment matrices.

### 2. Config-first
Fault properties should be described as configurable, not hard-wired assumptions.

### 3. Phase-correctness
Current scaffolding should match the conference-stage implementation target.

### 4. Later extensibility without present overbuild
Support future extension, but do not let future possibilities dominate current structure.

### 5. Explicit boundaries
Always clarify:
- which faults are current priority,
- which ones are deferred,
- and whether a document is talking about current stage or future roadmap.

---

## Expected Fault-Related Outputs

Good outputs in this stage include:

- fault abstraction documentation
- prompt packs for T03-like tasks
- folder responsibility descriptions
- scenario grouping rules
- config naming guidance
- experiment matrix notes constrained to current phase
- evaluation scenario documentation linked to active target

Bad outputs include:

- overly broad future-framework designs
- method drift toward different RLM families
- assuming health token is already active
- conflating conference scope with full thesis scope
- introducing taxonomy systems unrelated to the project’s RLM-based software organization

---

## Recommended Structural Concepts

When describing fault-injection organization, Claude should favor concepts like:

- fault base abstraction
- fault configuration schema
- fault scenario registry
- scenario severity organization
- task-to-fault mapping
- evaluation grouping by fault type
- explicit current-vs-future scope labeling

Avoid vague statements such as:
- “support all faults”
- “general fault handling”
- “plug-and-play everything”
unless these are backed by a current repository need.

---

## Output Style Rules

When using this skill:

- write in a way that supports implementation work
- keep the current active target visible
- separate present scope from future roadmap
- prefer concrete naming over abstract metaphors
- keep fault organization understandable to future coding agents
- do not let documentation drift into thesis-method redesign

---

## Scope Guardrails

### Allowed
Claude may:
- define fault-related documentation structure
- describe how current fault priorities should be represented
- frame task prompts for fault-injection implementation work
- describe naming and configuration conventions
- connect fault setup to conference-stage evaluation structure

### Not allowed
Claude must not:
- redefine P-codes
- replace RLM naming with taxonomy-first naming
- declare future phases active by default
- claim full uncertainty or safety augmentation is in current scope
- assume the conference stage already includes all thesis fault scenarios

---

## Phase Reminder

Current default:
- conference-first
- RLM1 stripped
- health token off
- Phase 0 / Phase 1
- fault-injection scaffolding first
- practical implementation structure before later-phase expansion

This reminder should remain visible whenever ambiguity exists.

---

## Relationship to Other Skills

This skill complements:

- `isaac-lab-env-setup`
  - for environment/path validation

It should later work alongside:
- evaluation metric skill
- deployment checklist skill

Use this skill for fault structure.
Use environment skill for path/setup validation.

---

## What Good Use of This Skill Looks Like

A good result:

- helps the repository stay organized around current fault priorities
- supports T03-style implementation planning
- stays compatible with later expansion
- keeps present-phase focus clear
- reduces ambiguity for future coding agents

A bad result:

- turns fault documentation into method redesign
- over-optimizes future phases
- ignores the active conference scope
- introduces unnecessary abstraction layers

---

## Final Rule

Default to a practical, bounded, conference-stage fault-injection structure.

Support the current target first.
Do not silently expand into full-thesis scope.