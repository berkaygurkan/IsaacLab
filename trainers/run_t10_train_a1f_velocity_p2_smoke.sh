#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

CONFIG="configs/train/teacher_velocity.yaml"
FAULT_CONFIG="configs/fault/joint_lock/p2_locked_joint.yaml"
TASK="Isaac-Ant-Teacher-Velocity-Flat-v0"
STAGE="teacher_p2_velocity"
METHOD="rlm1_stripped"
FAULT="P2_locked_joint"
SEED="0"
NUM_ENVS="64"
MAX_ITERATIONS="10"
EXPERIMENT_NAME="teacher_p2_velocity__rlm1_stripped__p2_locked_joint"
RUN_NAME="a1f_velocity_p2_smoke__seed0"
TARGET_JOINT="front_left_foot"
FAULT_ONSET_MODE="fixed"
FAULT_ONSET_STEP="50"
FAULT_ONSET_STEP_MIN="30"
FAULT_ONSET_STEP_MAX="150"
P2_KP="4.0"
P2_KD="0.4"
P2_ACTION_CLIP="1.0"
P2_REQUESTED_SEMANTICS="simulation_joint_state_override_lock"
P2_VELOCITY_OVERRIDE="0.0"
HEADLESS=1
EXECUTE_SMOKE=0
EXTRA_ARGS=()

show_help() {
    cat <<'EOF'
T10-IL-04 A1-F-Vel P2 privileged teacher PPO smoke launcher.

Smoke-only scaffold for the velocity-tracking teacher Ant task. This helper
delegates to trainers/rsl_rl_train.py, enables the repo-owned P2 joint-lock
wrapper, disables checkpoint pointer updates, and refuses to launch unless
--execute_smoke is provided.

Defaults:
  task: Isaac-Ant-Teacher-Velocity-Flat-v0
  stage: teacher_p2_velocity
  method: rlm1_stripped
  fault: P2_locked_joint
  seed: 0
  num_envs: 64
  max_iterations: 10
  experiment_name: teacher_p2_velocity__rlm1_stripped__p2_locked_joint
  run_name: a1f_velocity_p2_smoke__seed0
  target_joint: front_left_foot
  fault_onset_mode: fixed
  fault_onset_step: 50
  fault_onset_step_min: 30
  fault_onset_step_max: 150
  requested_semantics: simulation_joint_state_override_lock
  p2_allow_fallback: false

Options:
  --execute_smoke           Required to launch the tiny PPO smoke
  --config PATH             Default: configs/train/teacher_velocity.yaml
  --fault_config PATH       Default: configs/fault/joint_lock/p2_locked_joint.yaml
  --task TASK               Default: Isaac-Ant-Teacher-Velocity-Flat-v0
  --num_envs N              Default: 64
  --max_iterations N        Default: 10
  --seed N                  Default: 0
  --experiment_name NAME    Default: teacher_p2_velocity__rlm1_stripped__p2_locked_joint
  --run_name NAME           Default: a1f_velocity_p2_smoke__seed0
  --target_joint NAME       Default: front_left_foot
  --fault_onset_mode MODE   Default: fixed; choices: fixed, random_uniform
  --fault_onset_step N      Default: 50
  --fault_onset_step_min N  Default: 30; used by random_uniform
  --fault_onset_step_max N  Default: 150; used by random_uniform
  --headless                Enabled by default
  --no-headless             Run with GUI
  --help, -h                Show this help

This is not paper-grade training. It does not freeze checkpoints and does not
modify stable checkpoint pointers.

Example:
  bash trainers/run_t10_train_a1f_velocity_p2_smoke.sh --execute_smoke --max_iterations 10 --num_envs 64 --seed 0

Random-onset candidate example:
  bash trainers/run_t10_train_a1f_velocity_p2_smoke.sh --execute_smoke --max_iterations 1000 --num_envs 1024 --seed 0 --fault_onset_mode random_uniform --fault_onset_step_min 30 --fault_onset_step_max 150 --run_name a1f_velocity_p2_random_candidate1000__seed0
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
        --fault_config)
            FAULT_CONFIG="$2"
            shift 2
            ;;
        --fault_config=*)
            FAULT_CONFIG="${1#*=}"
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
        --target_joint)
            TARGET_JOINT="$2"
            shift 2
            ;;
        --target_joint=*)
            TARGET_JOINT="${1#*=}"
            shift
            ;;
        --fault_onset_mode)
            FAULT_ONSET_MODE="$2"
            shift 2
            ;;
        --fault_onset_mode=*)
            FAULT_ONSET_MODE="${1#*=}"
            shift
            ;;
        --fault_onset_step)
            FAULT_ONSET_STEP="$2"
            shift 2
            ;;
        --fault_onset_step=*)
            FAULT_ONSET_STEP="${1#*=}"
            shift
            ;;
        --fault_onset_step_min)
            FAULT_ONSET_STEP_MIN="$2"
            shift 2
            ;;
        --fault_onset_step_min=*)
            FAULT_ONSET_STEP_MIN="${1#*=}"
            shift
            ;;
        --fault_onset_step_max)
            FAULT_ONSET_STEP_MAX="$2"
            shift 2
            ;;
        --fault_onset_step_max=*)
            FAULT_ONSET_STEP_MAX="${1#*=}"
            shift
            ;;
        --p2_allow_fallback)
            echo "[T10-IL-04 ERROR] P2 fallback is disabled for this smoke launcher." >&2
            exit 2
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

if [[ "${FAULT_ONSET_MODE}" != "fixed" && "${FAULT_ONSET_MODE}" != "random_uniform" ]]; then
    echo "[T10-IL-04 ERROR] fault_onset_mode must be fixed or random_uniform, got: ${FAULT_ONSET_MODE}" >&2
    exit 2
fi

if [[ ! -f "${REPO_ROOT}/${CONFIG}" ]]; then
    echo "[T10-IL-04 ERROR] config not found: ${CONFIG}" >&2
    exit 2
fi

if [[ ! -f "${REPO_ROOT}/${FAULT_CONFIG}" ]]; then
    echo "[T10-IL-04 ERROR] fault_config not found: ${FAULT_CONFIG}" >&2
    exit 2
fi

COMMAND=(
    python
    trainers/rsl_rl_train.py
    --skip_checkpoint_pointer
    --enable_p2_joint_lock
    --p2_fault_config "${FAULT_CONFIG}"
    --p2_target_joint "${TARGET_JOINT}"
    --p2_fault_onset_step "${FAULT_ONSET_STEP}"
    --p2_fault_onset_mode "${FAULT_ONSET_MODE}"
    --p2_fault_onset_step_min "${FAULT_ONSET_STEP_MIN}"
    --p2_fault_onset_step_max "${FAULT_ONSET_STEP_MAX}"
    --p2_kp "${P2_KP}"
    --p2_kd "${P2_KD}"
    --p2_action_clip "${P2_ACTION_CLIP}"
    --p2_requested_semantics "${P2_REQUESTED_SEMANTICS}"
    --p2_velocity_override "${P2_VELOCITY_OVERRIDE}"
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
echo "[T10-IL-04] A1-F-Vel P2 privileged teacher PPO smoke launcher"
echo "  scope: smoke-only PPO launch for velocity-tracking teacher Ant under P2"
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
echo "  P2_runtime_hook_enabled: true"
echo "  P2_fault_config: ${FAULT_CONFIG}"
echo "  P2_target_joint: ${TARGET_JOINT}"
echo "  P2_fault_onset_mode: ${FAULT_ONSET_MODE}"
echo "  P2_fault_onset_step_min: ${FAULT_ONSET_STEP_MIN}"
echo "  P2_fault_onset_step_max: ${FAULT_ONSET_STEP_MAX}"
if [[ "${FAULT_ONSET_MODE}" == "fixed" ]]; then
    echo "  P2_fixed_fault_onset_step: ${FAULT_ONSET_STEP}"
    echo "  per_env_onset_randomization: false"
else
    echo "  P2_fixed_fault_onset_step: ignored_for_random_uniform_${FAULT_ONSET_STEP}"
    echo "  per_env_onset_randomization: true"
fi
echo "  requested_semantics: ${P2_REQUESTED_SEMANTICS}"
echo "  allow_fallback: false"
echo "  P2_velocity_override: ${P2_VELOCITY_OVERRIDE}"
echo "  P2_kp: ${P2_KP}"
echo "  P2_kd: ${P2_KD}"
echo "  P2_action_clip: ${P2_ACTION_CLIP}"
echo "  privileged_teacher: true"
echo "  deployment_facing: false"
echo "  health_token: OFF"
echo "  UQ: inactive"
echo "  CBF: inactive"
echo "  P3: inactive"
echo "  P4_scope: deferred"
echo "  student: disabled"
echo "  residual: disabled"
printf '[T10-IL-04] command:'
printf ' %q' "${COMMAND[@]}"
printf '\n'

if [[ "${EXECUTE_SMOKE}" != "1" ]]; then
    echo "[T10-IL-04 ERROR] Refusing to launch smoke without --execute_smoke." >&2
    echo "[T10-IL-04 ERROR] No training was run." >&2
    exit 3
fi

exec "${COMMAND[@]}"
