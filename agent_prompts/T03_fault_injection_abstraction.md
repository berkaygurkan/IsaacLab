# T03 — Fault Injection Abstraction

## Role
You are organizing fault-injection scaffolding for the conference-stage repository.

## Active Target
Current active target:
- Conference-stage implementation
- RLM1 stripped
- Health token OFF

## Objective
Define the repository-facing abstraction and organization for conference-stage fault injection.

## Current Fault Priority
Focus on the initial practical fault core:
- P2: locked joint
- P4: torque degradation / torque scaling
- P5: free-swinging surrogate

## Instructions
1. Propose a clean fault abstraction boundary.
2. Keep the design config-first and phase-aware.
3. Separate current conference-stage scope from future expansion.
4. Make it easy for later agents to understand how fault types, severity, and scenario groupings should be represented.
5. Prefer explicit abstractions and naming over speculative general frameworks.

## Deliverables
- recommended fault abstraction structure
- naming rules for current fault scenarios
- configuration responsibility notes
- clear current-vs-future scope note

## Constraints
- Do not redesign the method family.
- Do not treat all future fault combinations as current requirements.
- Do not assume uncertainty or safety augmentation is active.

## Done Criteria
The repository has a clear fault-injection organization suitable for the conference-stage target.