#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

show_help() {
    cat <<'EOF'
T09-DEMO-F generic A0/A2 demo play helper.

This helper plays an explicitly selected A0 healthy PPO or A2 distilled-student
checkpoint for advisor demo videos only. It does not train, update checkpoint
pointers, update observed manifests, execute P2, run A5 residual logic, or
create paper-grade results.

Required:
  --ablation_id A0|A2
  --checkpoint_path PATH        Exact model_*.pt checkpoint to load.
  --checkpoint_pointer PATH     Pointer YAML to resolve, then load the exact checkpoint it names.

Supported demo cases:
  A0 no fault:
    bash evaluators/run_t09_demo_play_policy.sh \
      --ablation_id A0 \
      --demo_gui \
      --log_step_csv \
      --demo_name D0_A0_no_fault \
      --checkpoint_path logs/rsl_rl/healthy_baseline__rlm1_stripped__demo/2026-06-03_04-08-57_healthy_demo__seed0/model_1999.pt \
      --fault_profile F0_none

  A2 no fault:
    bash evaluators/run_t09_demo_play_policy.sh \
      --ablation_id A2 \
      --demo_gui \
      --log_step_csv \
      --demo_name D2_A2_no_fault \
      --checkpoint_pointer checkpoints/rlm1_stripped/student/none/seed0/latest_checkpoint.yaml \
      --fault_profile F0_none

  A0 P4 visual-stress:
    bash evaluators/run_t09_demo_play_policy.sh \
      --ablation_id A0 \
      --demo_gui \
      --log_step_csv \
      --demo_name D1_A0_P4_torque0_2 \
      --checkpoint_path logs/rsl_rl/healthy_baseline__rlm1_stripped__demo/2026-06-03_04-08-57_healthy_demo__seed0/model_1999.pt \
      --fault_profile P4_torque_degradation \
      --torque_scale 0.2

  A2 P4 visual-stress:
    bash evaluators/run_t09_demo_play_policy.sh \
      --ablation_id A2 \
      --demo_gui \
      --log_step_csv \
      --demo_name D3_A2_P4_torque0_2 \
      --checkpoint_pointer checkpoints/rlm1_stripped/student/none/seed0/latest_checkpoint.yaml \
      --fault_profile P4_torque_degradation \
      --torque_scale 0.2

Offline stress comparison:
  python evaluators/compare_t09_demo_runs.py \
    --left_run_dir runs/t09_demo_play/<timestamp>_D1_A0_P4_torque0_2 \
    --right_run_dir runs/t09_demo_play/<timestamp>_D3_A2_P4_torque0_2 \
    --left_label A0_P4_stress \
    --right_label A2_P4_stress

Accepted options:
  --ablation_id A0|A2
  --checkpoint_path PATH
  --checkpoint_pointer PATH
  --fault_profile F0_none|P4_torque_degradation
  --demo_gui                    Do not force --headless; intended for manual screen recording.
  --headless                    Run without GUI.
  --demo_name NAME
  --demo_duration_steps N       Default: 400.
  --num_envs N                  Default inherited from evaluator: 8.
  --target_joint NAME           Fixed by validator to front_left_foot.
  --torque_scale VALUE          Default: 0.5. Demo stress values: 0.2 or 0.0.
  --fault_onset_step N          Fixed by validator to 50.
  --policy_mode deterministic   Deterministic only.
  --telemetry_interval_steps N  Default: 50.
  --log_step_csv                Write per-step quantitative metrics CSV.

Output:
  runs/t09_demo_play/<timestamp>_<demo_name>/demo_summary.md
  runs/t09_demo_play/<timestamp>_<demo_name>/summary_metrics.json
  runs/t09_demo_play/<timestamp>_<demo_name>/step_metrics.csv when --log_step_csv is set

Without --demo_gui or --headless, this helper adds --headless by default.
EOF
}

if [[ "$#" -eq 0 ]]; then
    show_help
    exit 0
fi

ORIG_ARGS=("$@")
HAS_CHECKPOINT_SELECTOR=0
HAS_DEMO_GUI=0
HAS_HEADLESS=0
HAS_ABLATION_ID=0
HAS_EXECUTION_FLAG=0
ABLATION_ID=""

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --help|-h)
            show_help
            exit 0
            ;;
        --checkpoint_path|--checkpoint_pointer)
            HAS_CHECKPOINT_SELECTOR=1
            shift
            ;;
        --checkpoint_path=*|--checkpoint_pointer=*)
            HAS_CHECKPOINT_SELECTOR=1
            ;;
        --demo_gui)
            HAS_DEMO_GUI=1
            ;;
        --headless)
            HAS_HEADLESS=1
            ;;
        --ablation_id)
            HAS_ABLATION_ID=1
            ABLATION_ID="${2:-}"
            shift
            ;;
        --ablation_id=*)
            HAS_ABLATION_ID=1
            ABLATION_ID="${1#*=}"
            ;;
        --execute_pilot|--execute_mapping_preflight|--execute_demo)
            HAS_EXECUTION_FLAG=1
            ;;
    esac
    shift
done

if [[ "${HAS_ABLATION_ID}" == "0" ]]; then
    echo "error: provide --ablation_id A0 or --ablation_id A2." >&2
    exit 2
fi

if [[ "${ABLATION_ID}" != "A0" && "${ABLATION_ID}" != "A2" ]]; then
    echo "error: this helper supports only --ablation_id A0 or A2." >&2
    exit 2
fi

if [[ "${HAS_CHECKPOINT_SELECTOR}" == "0" ]]; then
    echo "error: provide --checkpoint_path or --checkpoint_pointer." >&2
    exit 2
fi

if [[ "${HAS_EXECUTION_FLAG}" == "1" ]]; then
    echo "error: do not pass execution flags; this helper supplies the guarded demo mode itself." >&2
    exit 2
fi

source "${REPO_ROOT}/_isaac_sim/setup_conda_env.sh"

EXTRA_ARGS=(
    --execute_demo
    --demo_run_root runs/t09_demo_play
    --demo_note "advisor demo only, not paper-grade result"
)
if [[ "${HAS_DEMO_GUI}" == "0" && "${HAS_HEADLESS}" == "0" ]]; then
    EXTRA_ARGS+=(--headless)
fi

cd "${REPO_ROOT}"
exec python evaluators/run_t09_p4_pilot_eval.py "${EXTRA_ARGS[@]}" "${ORIG_ARGS[@]}"
