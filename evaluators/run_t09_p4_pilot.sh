#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

NEEDS_RUNTIME=0
HAS_HEADLESS=0
for ARG in "$@"; do
    if [[ "${ARG}" == "--execute_pilot" || "${ARG}" == "--execute_mapping_preflight" ]]; then
        NEEDS_RUNTIME=1
    fi
    if [[ "${ARG}" == "--headless" ]]; then
        HAS_HEADLESS=1
    fi
done

if [[ "${NEEDS_RUNTIME}" == "1" ]]; then
    source "${REPO_ROOT}/_isaac_sim/setup_conda_env.sh"
fi

EXTRA_ARGS=()
if [[ "${NEEDS_RUNTIME}" == "1" && "${HAS_HEADLESS}" == "0" ]]; then
    EXTRA_ARGS+=(--headless)
fi

cd "${REPO_ROOT}"
exec python evaluators/run_t09_p4_pilot_eval.py "${EXTRA_ARGS[@]}" "$@"
