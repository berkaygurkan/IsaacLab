#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

CONFIG="configs/train/healthy_baseline_velocity.yaml"
TASK="Isaac-Ant-Velocity-Flat-v0"
STAGE="healthy_baseline_velocity"
METHOD="rlm1_stripped"
FAULT="none"
SEED="0"
NUM_ENVS="64"
MAX_ITERATIONS="10"
EXPERIMENT_NAME="healthy_baseline_velocity__rlm1_stripped__none"
RUN_NAME="a0_velocity_smoke__seed0"
HEADLESS=1
EXECUTE_SMOKE=0
EXTRA_ARGS=()

show_help() {
    cat <<'EOF'
T10-IL-03 A0-Vel healthy PPO smoke launcher.

Smoke-only scaffold for the velocity-tracking Ant task. This helper delegates
to trainers/rsl_rl_train.py, disables checkpoint pointer updates, and refuses
to launch unless --execute_smoke is provided.

Defaults:
  task: Isaac-Ant-Velocity-Flat-v0
  stage: healthy_baseline_velocity
  method: rlm1_stripped
  fault: none
  seed: 0
  num_envs: 64
  max_iterations: 10
  experiment_name: healthy_baseline_velocity__rlm1_stripped__none
  run_name: a0_velocity_smoke__seed0

Options:
  --execute_smoke           Required to launch the tiny PPO smoke
  --config PATH             Default: configs/train/healthy_baseline_velocity.yaml
  --task TASK               Default: Isaac-Ant-Velocity-Flat-v0
  --num_envs N              Default: 64
  --max_iterations N        Default: 10
  --seed N                  Default: 0
  --experiment_name NAME    Default: healthy_baseline_velocity__rlm1_stripped__none
  --run_name NAME           Default: a0_velocity_smoke__seed0
  --headless                Enabled by default
  --no-headless             Run with GUI
  --help, -h                Show this help

This is not paper-grade training. It does not freeze checkpoints and does not
modify stable checkpoint pointers.

Example:
  bash trainers/run_t10_train_a0_velocity_smoke.sh --execute_smoke --max_iterations 10 --num_envs 64 --seed 0
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --help|-h)
            show_help
            exit 0
            ;;
        --execute_smoke)
            EXECUTE_SMOKE=1
            shift
            ;;
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --config=*)
            CONFIG="${1#*=}"
            shift
            ;;
        --task)
            TASK="$2"
            shift 2
            ;;
        --task=*)
            TASK="${1#*=}"
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

if [[ ! -f "${REPO_ROOT}/${CONFIG}" ]]; then
    echo "[T10-IL-03 ERROR] config not found: ${CONFIG}" >&2
    exit 2
fi

COMMAND=(
    python
    trainers/rsl_rl_train.py
    --skip_checkpoint_pointer
    --config "${CONFIG}"
    --stage "${STAGE}"
    --method "${METHOD}"
    --fault "${FAULT}"
    --task "${TASK}"
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
echo "[T10-IL-03] A0-Vel healthy PPO smoke launcher"
echo "  scope: smoke-only PPO launch for velocity-tracking Ant"
echo "  not_paper_grade_training: true"
echo "  task: ${TASK}"
echo "  stage: ${STAGE}"
echo "  method: ${METHOD}"
echo "  fault: ${FAULT}"
echo "  seed: ${SEED}"
echo "  num_envs: ${NUM_ENVS}"
echo "  max_iterations: ${MAX_ITERATIONS}"
echo "  experiment_name: ${EXPERIMENT_NAME}"
echo "  run_name: ${RUN_NAME}"
echo "  expected_log_root: logs/rsl_rl/${EXPERIMENT_NAME}/"
echo "  checkpoint_pointer_update: disabled"
echo "  checkpoint_freeze: not_performed"
echo "  p2_wrapper: disabled"
echo "  privileged_teacher_observation: disabled"
echo "  residual: disabled"
printf '[T10-IL-03] command:'
printf ' %q' "${COMMAND[@]}"
printf '\n'

if [[ "${EXECUTE_SMOKE}" != "1" ]]; then
    echo "[T10-IL-03 ERROR] Refusing to launch smoke without --execute_smoke." >&2
    echo "[T10-IL-03 ERROR] No training was run." >&2
    exit 3
fi

exec "${COMMAND[@]}"
