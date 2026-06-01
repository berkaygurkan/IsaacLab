#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
source "${REPO_ROOT}/_isaac_sim/setup_conda_env.sh"

cd "${REPO_ROOT}"
exec python evaluators/run_t09_one_row_smoke.py "$@"
