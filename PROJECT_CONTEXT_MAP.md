# PROJECT_CONTEXT_MAP.md

## Purpose

This file is the top-level orientation map for all coding agents and GPT-style project assistants used in this repository.

It explains:

- what this repository is for,
- which machine is responsible for what,
- which configuration files must be read first,
- where agent-specific instruction files live,
- how the workflow proceeds from system setup to thesis implementation,
- and which thesis phase is currently active.

This file is not a method-design document and not a literature-taxonomy file.
It is a navigation and execution-context file.

---

## Project Identity

**Project title:** Quadruped Fault Tolerant Control via Reinforcement Learning

**Primary stack:**
- Ubuntu 22.04 workstation
- Isaac Sim / Isaac Lab
- Python 3.11 isolated environment
- PyTorch
- Hydra / YAML-first configuration
- Unitree A1 deployment target
- VS Code + SSH workflow
- Claude Code / Codex / Antigravity as coding agents

**Current active thesis method family:**
- **RLM1**
- **Current variant:** **RLM1 stripped**
- Meaning:
  - teacher-student backbone present
  - residual learning present
  - **health token disabled**
- This corresponds to the **Conference paper configuration**

---

## Machine Topology

### PC1 — MacBook M1 Pro
Role:
- thesis writing
- paper writing
- reading
- agent orchestration
- remote access to main workstation
- VS Code / Codex / Antigravity / Claude-side interaction

Installed/available:
- VS Code
- Codex-compatible tooling
- Antigravity-compatible tooling
- Tailscale
- NoMachine (emergency only)

### PC2 — Ubuntu Main Workstation
Role:
- canonical development machine
- Isaac Sim / Isaac Lab execution
- training
- experiment running
- logs / checkpoints / artifacts
- deployment preparation

Installed/validated:
- Ubuntu 22.04.5
- NVIDIA driver compatible
- RTX 4070 Ti
- Isaac Sim binary
- Isaac Lab source repo
- Python 3.11 conda environment
- smoke test passed

**Rule:**  
All heavy simulation/training work happens on **PC2**.  
PC1 is the control/orchestration machine.

---

## Canonical Working Directories

### Main thesis workspace
```text
~/thesis/