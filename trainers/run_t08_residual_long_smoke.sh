#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
source "${REPO_ROOT}/_isaac_sim/setup_conda_env.sh"

cd "${REPO_ROOT}"
TERM=xterm ./trainers/run_t08_residual_train.sh --headless --num_envs 8 --max_iterations 5
