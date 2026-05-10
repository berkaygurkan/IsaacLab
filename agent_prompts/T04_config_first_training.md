# T04 — Config-First Training Organization

## Role
You are organizing training-facing configuration structure for the thesis repository.

## Active Target
Current active target:
- Conference-stage implementation
- RLM1 stripped
- Health token OFF

## Objective
Define a config-first structure for method, fault, and training parameters that supports reproducible conference-stage work.

## Instructions
1. Propose a configuration layout that separates:
   - environment settings
   - fault settings
   - method settings
   - training settings
   - evaluation settings
2. Prefer YAML/Hydra-style clarity.
3. Include seed and reproducibility expectations.
4. Keep the structure simple enough for T05–T10.
5. Preserve extensibility for later RLM1-full work without making it the default.

## Deliverables
- config file hierarchy proposal
- config naming conventions
- reproducibility notes
- phase-aware defaults for conference-stage work

## Constraints
- Do not assume health token is active.
- Do not optimize for P2/P3 by default.
- Do not create configuration sprawl.

## Done Criteria
The project has a clear conference-stage config organization with reproducibility discipline.