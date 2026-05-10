# T06 — RLM1 Stripped Residual Head Task Framing

## Role
You are supporting repository-level task framing for the active RLM1 stripped implementation.

## Active Target
Current active target:
- Conference-stage implementation
- RLM1 stripped
- Teacher-student backbone active
- Residual learning active
- Health token OFF

## Objective
Frame the implementation task for the conference-stage RLM1 stripped setup, focusing on the residual-head side without enabling the health token.

## Instructions
1. Keep the task explicitly tied to RLM1 stripped.
2. Make the “health token OFF” constraint visible.
3. Describe how the task should be scoped for the conference phase.
4. Preserve future compatibility with RLM1 full, but do not activate it.
5. Keep the wording implementation-oriented and repo-friendly.

## Deliverables
- implementation task framing
- explicit out-of-scope list
- current-stage assumptions
- future-compatibility note without scope expansion

## Constraints
- Do not treat RLM1 stripped as a different method family.
- Do not activate the health token.
- Do not assume P1/P2/P3 targets.

## Done Criteria
The task is clearly framed as conference-stage RLM1 stripped implementation support.