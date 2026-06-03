#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

show_help() {
    cat <<'EOF'
T09-DEMO-A meeting demo helper.

This helper runs short visual demo rollouts only. It does not train, update
checkpoint pointers, update observed manifests, or create paper-grade results.

Recommended commands:

  D0_A0_no_fault:
    bash evaluators/run_t09_meeting_demo.sh --demo_gui --demo_name D0_A0_no_fault --ablation_id A0 --fault_profile F0_none

  D1_A0_P4:
    bash evaluators/run_t09_meeting_demo.sh --demo_gui --demo_name D1_A0_P4 --ablation_id A0 --fault_profile P4_torque_degradation

  D2_A5_P4:
    bash evaluators/run_t09_meeting_demo.sh --demo_gui --demo_name D2_A5_P4 --ablation_id A5 --fault_profile P4_torque_degradation

Useful options:
  --demo_gui                    Do not force --headless; intended for manual screen recording.
  --demo_duration_steps N       Default: 400.
  --telemetry_interval_steps N  Default: 50.
  --target_joint NAME           Fixed by validator to front_left_foot.
  --torque_scale VALUE          Fixed by validator to 0.5.
  --fault_onset_step N          Fixed by validator to 50.

Without --demo_gui, this helper adds --headless by default.
EOF
}

if [[ "$#" -eq 0 ]]; then
    show_help
    exit 0
fi

HAS_EXECUTE_DEMO=0
HAS_DEMO_GUI=0
HAS_HEADLESS=0
for ARG in "$@"; do
    if [[ "${ARG}" == "--help" || "${ARG}" == "-h" ]]; then
        show_help
        exit 0
    fi
    if [[ "${ARG}" == "--execute_demo" ]]; then
        HAS_EXECUTE_DEMO=1
    fi
    if [[ "${ARG}" == "--demo_gui" ]]; then
        HAS_DEMO_GUI=1
    fi
    if [[ "${ARG}" == "--headless" ]]; then
        HAS_HEADLESS=1
    fi
done

source "${REPO_ROOT}/_isaac_sim/setup_conda_env.sh"

EXTRA_ARGS=()
if [[ "${HAS_EXECUTE_DEMO}" == "0" ]]; then
    EXTRA_ARGS+=(--execute_demo)
fi
if [[ "${HAS_DEMO_GUI}" == "0" && "${HAS_HEADLESS}" == "0" ]]; then
    EXTRA_ARGS+=(--headless)
fi

cd "${REPO_ROOT}"
exec python evaluators/run_t09_p4_pilot_eval.py "${EXTRA_ARGS[@]}" "$@"
