# GIT_WORKFLOW.md

## Purpose

This file defines the Git workflow for this thesis repository.

It explains:

- what `origin` and `upstream` mean,
- why `git remote -v` shows both `fetch` and `push`,
- how to update the local clone,
- how to update the personal fork,
- how to stay synced with the official IsaacLab repository,
- and how to use both terminal and GitHub Desktop safely.

This file is a workflow reference.
It should be used by both humans and coding agents.

---

## Repository Model

This repository follows the standard **fork-based workflow**.

### Current meaning of remotes

- **origin** = the user's personal fork
- **upstream** = the official IsaacLab repository

Example:

```text
origin   = https://github.com/<user>/IsaacLab.git
upstream = https://github.com/isaac-sim/IsaacLab.git