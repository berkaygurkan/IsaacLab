#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

CONFIG="configs/train/healthy_baseline_demo.yaml"
NUM_ENVS="4096"
MAX_ITERATIONS="2000"
SEED="0"
EXPERIMENT_NAME="healthy_baseline__rlm1_stripped__canonical"
RUN_NAME="a0_canonical__seed0"
HEADLESS=1
EXTRA_ARGS=()

show_help() {
    cat <<'EOF'
T09-R1 A0 healthy PPO canonical-training helper.

Conference-stage checkpoint generation scaffold only. This helper trains A0
healthy/no-fault PPO through the existing T05 healthy training path and disables
stable checkpoint pointer updates by passing --skip_checkpoint_pointer.

Options:
  --config PATH              Default: configs/train/healthy_baseline_demo.yaml
  --num_envs N               Default: 4096
  --max_iterations N         Default: 2000
  --seed N                   Default: 0
  --experiment_name NAME     Default: healthy_baseline__rlm1_stripped__canonical
  --run_name NAME            Default: a0_canonical__seed0
  --headless                 Enabled by default
  --no-headless              Run with GUI
  --help, -h                 Show this help

Expected log root:
  logs/rsl_rl/healthy_baseline__rlm1_stripped__canonical/

After training, freeze the selected model_*.pt with:
  python evaluators/freeze_t09_checkpoint.py --stage healthy_baseline --method rlm1_stripped --fault none --seed 0 --task Isaac-Ant-v0 --checkpoint_path <exact_model.pt> --experiment_name healthy_baseline__rlm1_stripped__canonical --run_name a0_canonical__seed0 --classification paper_grade_candidate --output_pointer checkpoints/rlm1_stripped/healthy_baseline/none/seed0/canonical_checkpoint.yaml --manifest_path papers/conference/results/t09_a0_canonical_checkpoint_freeze.md

No training is run by --help.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --help|-h)
            show_help
            exit 0
            ;;
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --config=*)
            CONFIG="${1#*=}"
            shift
            ;;
        --num_envs)
            NUM_ENVS="$2"
            shift 2
            ;;
        --num_envs=*)
            NUM_ENVS="${1#*=}"
            shift
            ;;
        --max_iterations)
            MAX_ITERATIONS="$2"
            shift 2
            ;;
        --max_iterations=*)
            MAX_ITERATIONS="${1#*=}"
            shift
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --seed=*)
            SEED="${1#*=}"
            shift
            ;;
        --experiment_name)
            EXPERIMENT_NAME="$2"
            shift 2
            ;;
        --experiment_name=*)
            EXPERIMENT_NAME="${1#*=}"
            shift
            ;;
        --run_name)
            RUN_NAME="$2"
            shift 2
            ;;
        --run_name=*)
            RUN_NAME="${1#*=}"
            shift
            ;;
        --headless)
            HEADLESS=1
            shift
            ;;
        --no-headless)
            HEADLESS=0
            shift
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

COMMAND=(
    bash
    trainers/run_t05_healthy.sh
    --skip_checkpoint_pointer
    --config "${CONFIG}"
    --stage healthy_baseline
    --method rlm1_stripped
    --fault none
    --task Isaac-Ant-v0
    --seed "${SEED}"
    --experiment_name "${EXPERIMENT_NAME}"
    --run_name "${RUN_NAME}"
    --num_envs "${NUM_ENVS}"
    --max_iterations "${MAX_ITERATIONS}"
)

if [[ "${HEADLESS}" == "1" ]]; then
    COMMAND+=(--headless)
fi

COMMAND+=("${EXTRA_ARGS[@]}")

cd "${REPO_ROOT}"
echo "[T09-R1] A0 healthy PPO canonical training helper"
echo "  scope: canonical checkpoint generation scaffold; no fault during training"
echo "  task: Isaac-Ant-v0"
echo "  method: rlm1_stripped"
echo "  fault: none"
echo "  num_envs: ${NUM_ENVS}"
echo "  max_iterations: ${MAX_ITERATIONS}"
echo "  expected_log_root: logs/rsl_rl/${EXPERIMENT_NAME}/"
echo "  checkpoint_pointer_update: disabled"
echo "  canonical_freeze: manual via evaluators/freeze_t09_checkpoint.py"
printf '[T09-R1] command:'
printf ' %q' "${COMMAND[@]}"
printf '\n'
exec "${COMMAND[@]}"
