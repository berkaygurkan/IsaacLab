# isaac-lab-env-setup

## Purpose

Use this skill when working on environment-aware tasks inside the thesis repository.

This skill helps Claude Code confirm that the repository is being operated in the correct machine, path, Python environment, and Isaac Sim / Isaac Lab setup before performing environment-sensitive work.

This is a verification-and-guidance skill, not a method-design skill.

---

## When to Use This Skill

Use this skill when:

- starting a new Claude Code session in this repository
- checking whether the environment is correctly activated
- preparing to run Isaac Lab commands
- validating that Isaac Sim is connected to the active conda environment
- confirming canonical paths before proposing task execution steps
- verifying that the repo is on the Ubuntu execution machine rather than the writing laptop
- checking whether the basic setup is healthy after changes

Use this skill before:
- setup-sensitive documentation
- task prompts that include execution steps
- validation workflows
- deployment preparation notes
- environment troubleshooting suggestions

---

## When Not to Use This Skill

Do not use this skill for:
- thesis method selection
- literature comparison
- algorithm redesign
- task prioritization by itself
- Git branching strategy
- fault-injection design details
- evaluation metric definitions unless environment state is directly relevant

---

## Active Project Assumptions

Unless the user explicitly says otherwise, assume:

- canonical execution machine is the Ubuntu workstation
- repo path is `~/thesis/IsaacLab`
- Isaac Sim binary path is `~/isaacsim`
- conda environment name is `isaaclab`
- Python version should be 3.11
- repository is running in conference-stage mode
- active target is **RLM1 stripped**
- health token is OFF

Do not assume:
- MacBook is the default execution machine
- Python 3.13 should be used
- a fresh Isaac Lab reinstall is needed unless evidence suggests breakage
- later thesis phases are the active default

---

## Canonical Environment Facts

Expected canonical values:

- repo path: `~/thesis/IsaacLab`
- Isaac Sim path: `~/isaacsim`
- conda env: `isaaclab`
- Python: `3.11.x`
- `ISAACSIM_PATH=$HOME/isaacsim`
- `ISAACSIM_PYTHON_EXE=$ISAACSIM_PATH/python.sh`

A healthy session should be compatible with:
- `import isaacsim`
- Isaac Lab smoke test execution
- repo-level workflow files already present

---

## Verification Checklist

When this skill is used, verify the following conceptually before making setup-dependent recommendations:

### 1. Machine role
Confirm the intended execution context:
- Ubuntu workstation = execution machine
- MacBook = orchestration / writing machine

### 2. Repository path
Confirm the repository is expected at:
- `~/thesis/IsaacLab`

### 3. Isaac Sim path
Confirm Isaac Sim is expected at:
- `~/isaacsim`

### 4. Conda environment
Confirm environment should be:
- `isaaclab`

### 5. Python version
Confirm target Python should be:
- `3.11`

### 6. Integration state
Confirm Isaac Sim and Isaac Lab should already be connected in the current project setup.

### 7. Smoke-test capability
Confirm that a basic Isaac Lab smoke test is expected to work in the healthy baseline state.

---

## Preferred Validation Commands

When the user asks for environment validation or troubleshooting, prefer a minimal check set.

### Basic path and Python checks
```bash
pwd
which python
python --version
echo $ISAACSIM_PATH
echo $ISAACSIM_PYTHON_EXE