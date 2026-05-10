# eval-metrics-runner

## Purpose

Use this skill when working on evaluation, metrics, results organization, or experiment-reporting tasks inside the thesis repository.

This skill helps Claude Code stay aligned with the current implementation phase when describing what should be measured, how outputs should be grouped, and how evaluation-oriented task documents should be framed.

This is a repository-structure and evaluation-organization skill.
It is not a paper-writing skill and not an algorithm redesign skill.

---

## When to Use This Skill

Use this skill when:

- describing conference-stage evaluation priorities
- writing task prompts related to metrics or reporting
- organizing result tables, figure expectations, or artifact outputs
- describing what a run should log or summarize
- defining evaluation categories for fault scenarios
- clarifying present-phase versus future-phase metric scope
- writing reusable evaluation guidance for coding agents

Use this skill before:
- metric-oriented prompt packs
- evaluation documentation
- result artifact structure docs
- conference result organization notes
- experiment output naming guidance tied to evaluation

---

## When Not to Use This Skill

Do not use this skill for:

- selecting the thesis method
- redesigning learning objectives
- inventing a new benchmark philosophy
- rewriting the thesis roadmap
- making future phases the current default
- replacing authoritative project files

This skill supports evaluation organization within the active target.
It does not redefine the project.

---

## Active Project Assumptions

Unless the user explicitly changes target, assume:

- active method family is **RLM1**
- active variant is **RLM1 stripped**
- health token is OFF
- current implementation target is the **conference paper**
- active work band is **Phase 0 / Phase 1**
- evaluation support should match the conference-stage implementation

Do not assume:
- health-token metrics are currently central
- uncertainty metrics are active by default
- safety-filter metrics are current default priorities
- all future publication metrics must be fully surfaced now

---

## Current Role of Evaluation

At the current stage, evaluation is primarily used to:

- verify that conference-stage experiments are meaningful
- compare behavior under current target fault scenarios
- support practical run summaries
- structure tables and plots for the conference paper
- make later expansion possible without prematurely adopting full-thesis scope

Evaluation should currently remain:
- targeted,
- practical,
- phase-aware,
- and easy for coding agents to interpret.

---

## Preferred Conference-Stage Metric Focus

For the current conference-stage target, prioritize metrics that support practical locomotion and recovery evaluation.

Preferred current-stage metric categories include:

- success rate
- episode return
- recovery time
- command tracking error
- stability-oriented indicators
- cost of transport or energy-style efficiency signals when already relevant
- run grouping by fault type and severity

If discussing result structure, keep the emphasis on metrics that are reasonable for the active conference target.

Do not automatically center:
- uncertainty calibration metrics
- explicit diagnosis metrics
- safety-filter intervention metrics
unless the user explicitly activates later-phase goals.

---

## Evaluation Framing Rules

When using this skill, Claude should follow these rules:

### 1. Present phase first
Prioritize the current conference target over future publication phases.

### 2. Separate current and future metrics
If mentioning future-phase metrics, clearly label them as future or deferred.

### 3. Support implementation work
Describe metrics in a way that helps repository structure, task prompts, and output planning.

### 4. Avoid metric inflation
Do not expand the metric set just because more metrics are possible.

### 5. Keep comparisons interpretable
Group outputs in a way that future coding agents can understand without re-reading the whole thesis roadmap.

---

## Good Current-Stage Output Types

Good outputs include:

- conference metric summaries
- evaluation task prompts
- run-output grouping notes
- artifact naming guidance
- result table templates
- current-vs-future metric scoping notes
- evaluation folder responsibility descriptions

Bad outputs include:

- broad all-phase evaluation frameworks by default
- uncertainty-heavy reporting assumptions
- safety-augmentation-first metric organization
- diagnostic switching metrics treated as current conference defaults
- result structures that assume full thesis maturity from the beginning

---

## Recommended Evaluation Grouping Concepts

When describing evaluation organization, Claude should favor concepts such as:

- metric groups by experiment phase
- metric groups by fault type
- metric groups by severity
- baseline versus method comparison
- recovery-oriented summaries
- tracking-oriented summaries
- artifact-ready result outputs

Avoid vague labels such as:
- “full performance”
- “overall robustness”
unless they are paired with explicit measurable groupings.

---

## Output Style Rules

When using this skill:

- keep the active target visible
- distinguish current-phase metrics from future metrics
- prefer operational wording over abstract evaluation language
- support reproducibility and artifact generation
- help later agents understand what belongs in the conference-stage output layer

---

## Scope Guardrails

### Allowed
Claude may:
- define evaluation documentation structure
- describe current-stage metric priorities
- help create evaluation-related prompt packs
- define result grouping logic
- describe what artifacts a conference-stage run should produce

### Not allowed
Claude must not:
- silently activate future-phase metric families
- assume token-specific metrics are active
- assume uncertainty or safety metrics are current defaults
- replace the active conference target with full-roadmap evaluation breadth
- turn evaluation docs into thesis-method redesign

---

## Relationship to Other Skills

This skill complements:

- `isaac-lab-env-setup`
  - for environment validation
- `fault-injection-recipe`
  - for scenario structure and fault-scoped task framing

This skill focuses on what should be measured and how outputs should be organized in the current active phase.

---

## Phase Reminder

Current default:
- conference-first
- RLM1 stripped
- health token off
- Phase 0 / Phase 1
- practical evaluation support
- conference artifact readiness over full-thesis reporting breadth

Keep this reminder explicit whenever a metric-related task could drift into later-phase assumptions.

---

## What Good Use of This Skill Looks Like

A good result:

- helps organize metrics for the active conference target
- keeps evaluation structure practical
- aligns with current fault priorities
- supports later extension without demanding it now
- reduces ambiguity for future agents

A bad result:

- overbuilds the evaluation framework
- assumes future thesis scope is already active
- confuses current and deferred metric families
- makes artifact planning harder instead of easier

---

## Final Rule

Default to a bounded, conference-stage evaluation structure.

Support the current implementation target first.
Do not silently expand into full-thesis evaluation scope.