# T01 — Repository Bootstrap

## Role
You are working inside a doctoral research repository for quadruped fault-tolerant locomotion using Isaac Lab.

## Active Target
Current active target:
- Conference-stage implementation
- Method family: RLM1
- Variant: RLM1 stripped
- Health token: OFF

Do not optimize for later phases by default.

## Objective
Create or refine the repository skeleton so that the project becomes easy to navigate for future coding agents and human collaborators.

## Scope
Focus on the repository structure only.

Expected top-level responsibility areas include:
- configs/
- envs/
- faults/
- methods/
- trainers/
- evaluators/
- scripts/
- logs/
- checkpoints/
- deployment/
- papers/

## Instructions
1. Inspect the current repository structure.
2. Propose the minimal clean directory layout needed for the active conference target.
3. Do not overbuild for P1/P2/P3.
4. Preserve compatibility for future expansion.
5. Prefer documentation, placeholders, and folder responsibility clarity over speculative code architecture.

## Deliverables
- a proposed directory tree
- a short responsibility description for each major folder
- any missing README-style folder notes if appropriate
- clear distinction between current-stage folders and future-stage placeholders

## Constraints
- Do not redesign the thesis method.
- Do not assume health token is active.
- Do not create a broad future-only architecture.
- Keep the repository conference-first.

## Done Criteria
The repository structure is understandable, phase-aware, and immediately usable for T02–T10.