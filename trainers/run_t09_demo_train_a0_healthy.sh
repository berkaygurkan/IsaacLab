#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

CONFIG="configs/train/healthy_baseline_demo.yaml"
NUM_ENVS="512"
MAX_ITERATIONS="1000"
SEED="0"
EXPERIMENT_NAME="healthy_baseline__rlm1_stripped__demo"
RUN_NAME="healthy_demo__seed0"
HEADLESS=1
EXTRA_ARGS=()

show_help() {
    cat <<'EOF'
T09-DEMO-C1 A0 healthy PPO demo-training helper.

Advisor-demo checkpoint generation only. This is not paper-grade training.
The helper trains A0 healthy/no-fault PPO and never updates stable checkpoint
pointers because it passes --skip_checkpoint_pointer to trainers/rsl_rl_train.py.

Options:
  --config PATH              Default: configs/train/healthy_baseline_demo.yaml
  --num_envs N               Default: 512
  --max_iterations N         Default: 1000
  --seed N                   Default: 0
  --experiment_name NAME     Default: healthy_baseline__rlm1_stripped__demo
  --run_name NAME            Default: healthy_demo__seed0
  --headless                 Enabled by default
  --no-headless              Run with GUI
  --help, -h                 Show this help

Recommended first run:
  bash trainers/run_t09_demo_train_a0_healthy.sh --num_envs 512 --max_iterations 1000

Longer optional run if VRAM allows:
  bash trainers/run_t09_demo_train_a0_healthy.sh --num_envs 1024 --max_iterations 2000

Expected log root:
  logs/rsl_rl/healthy_baseline__rlm1_stripped__demo/

After training, select the demo checkpoint manually by exact model_*.pt path.
Do not overwrite stable checkpoint pointers for this demo workflow.
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
echo "[T09-DEMO-C1] A0 healthy PPO demo training helper"
echo "  scope: advisor demo checkpoint generation only; not paper-grade training"
echo "  task: Isaac-Ant-v0"
echo "  method: rlm1_stripped"
echo "  fault: none"
echo "  num_envs: ${NUM_ENVS}"
echo "  max_iterations: ${MAX_ITERATIONS}"
echo "  expected_log_root: logs/rsl_rl/${EXPERIMENT_NAME}/"
echo "  checkpoint_pointer_update: disabled"
echo "  checkpoint_promotion: manual only"
printf '[T09-DEMO-C1] command:'
printf ' %q' "${COMMAND[@]}"
printf '\n'
exec "${COMMAND[@]}"
