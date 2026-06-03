#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

show_help() {
    cat <<'EOF'
T09-DEMO-C2 A0 demo play helper.

This helper plays an explicitly selected A0 healthy PPO checkpoint for advisor
demo videos only. It does not train, update checkpoint pointers, update observed
manifests, execute P2, or create paper-grade results.

Required checkpoint selector:
  --checkpoint_path PATH        Exact model_*.pt checkpoint to load.
  --checkpoint_pointer PATH     Pointer YAML to resolve, then load the exact checkpoint it names.

Supported demo cases:
  A0 no fault:
    bash evaluators/run_t09_demo_play_a0.sh \
      --demo_gui \
      --log_step_csv \
      --demo_name D0_A0_no_fault \
      --checkpoint_path logs/rsl_rl/healthy_baseline__rlm1_stripped__demo/2026-06-03_03-40-01_healthy_demo__seed0/model_999.pt \
      --fault_profile F0_none

  A0 under P4 torque degradation:
    bash evaluators/run_t09_demo_play_a0.sh \
      --demo_gui \
      --log_step_csv \
      --demo_name D1_A0_P4 \
      --checkpoint_path logs/rsl_rl/healthy_baseline__rlm1_stripped__demo/2026-06-03_03-40-01_healthy_demo__seed0/model_999.pt \
      --fault_profile P4_torque_degradation

  Visual-stress P4 demo, still advisor-demo only:
    bash evaluators/run_t09_demo_play_a0.sh \
      --demo_gui \
      --log_step_csv \
      --demo_name D1_A0_P4_torque0_2 \
      --checkpoint_path logs/rsl_rl/healthy_baseline__rlm1_stripped__demo/2026-06-03_03-40-01_healthy_demo__seed0/model_999.pt \
      --fault_profile P4_torque_degradation \
      --torque_scale 0.2

Offline comparison after recording F0 and P4 demos:
  python evaluators/compare_t09_demo_runs.py \
    --f0_run_dir runs/t09_demo_play/<timestamp>_D0_A0_no_fault \
    --p4_run_dir runs/t09_demo_play/<timestamp>_D1_A0_P4

Accepted options:
  --checkpoint_path PATH
  --checkpoint_pointer PATH
  --fault_profile F0_none|P4_torque_degradation
  --demo_gui                    Do not force --headless; intended for manual screen recording.
  --headless                    Run without GUI.
  --demo_name NAME
  --demo_duration_steps N       Default: 400.
  --num_envs N                  Default inherited from evaluator: 8.
  --target_joint NAME           Fixed by validator to front_left_foot.
  --torque_scale VALUE          Fixed by validator to 0.5.
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

HAS_CHECKPOINT_SELECTOR=0
HAS_DEMO_GUI=0
HAS_HEADLESS=0
HAS_ABLATION_ID=0
HAS_EXECUTION_FLAG=0

for ARG in "$@"; do
    case "${ARG}" in
        --help|-h)
            show_help
            exit 0
            ;;
        --checkpoint_path|--checkpoint_pointer|--checkpoint_path=*|--checkpoint_pointer=*)
            HAS_CHECKPOINT_SELECTOR=1
            ;;
        --demo_gui)
            HAS_DEMO_GUI=1
            ;;
        --headless)
            HAS_HEADLESS=1
            ;;
        --ablation_id|--ablation_id=*)
            HAS_ABLATION_ID=1
            ;;
        --execute_pilot|--execute_mapping_preflight|--execute_demo)
            HAS_EXECUTION_FLAG=1
            ;;
    esac
done

if [[ "${HAS_CHECKPOINT_SELECTOR}" == "0" ]]; then
    echo "error: provide --checkpoint_path or --checkpoint_pointer for A0 demo play." >&2
    exit 2
fi

if [[ "${HAS_ABLATION_ID}" == "1" ]]; then
    echo "error: this helper is A0-only and supplies --ablation_id A0 itself." >&2
    exit 2
fi

if [[ "${HAS_EXECUTION_FLAG}" == "1" ]]; then
    echo "error: do not pass execution flags; this helper supplies the guarded demo mode itself." >&2
    exit 2
fi

source "${REPO_ROOT}/_isaac_sim/setup_conda_env.sh"

EXTRA_ARGS=(
    --execute_demo
    --ablation_id A0
    --demo_run_root runs/t09_demo_play
    --demo_note "advisor demo only, not paper-grade result"
)
if [[ "${HAS_DEMO_GUI}" == "0" && "${HAS_HEADLESS}" == "0" ]]; then
    EXTRA_ARGS+=(--headless)
fi

cd "${REPO_ROOT}"
exec python evaluators/run_t09_p4_pilot_eval.py "${EXTRA_ARGS[@]}" "$@"
