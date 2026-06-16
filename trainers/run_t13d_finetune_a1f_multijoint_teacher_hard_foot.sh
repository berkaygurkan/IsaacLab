#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

CONFIG="configs/train/teacher_velocity.yaml"
FAULT_CONFIG="configs/fault/joint_lock/p2_hard_foot_random.yaml"
TASK="Isaac-Ant-Teacher-Velocity-MultiJointP2-Flat-v0"
STAGE="teacher_p2_multijoint_velocity_hard_foot_finetune"
METHOD="rlm1_stripped"
FAULT="p2_hard_foot_random"
SEED="0"
NUM_ENVS="4096"
MAX_ITERATIONS="2000"
DEVICE="cuda"
HEADLESS=1
EXECUTE_FINETUNE=0
JOINT_SET="hard_foot"
HARD_FOOT_JOINTS="front_left_foot,front_right_foot,left_back_foot"
BALANCED_FOOT_JOINTS="front_left_foot,front_right_foot,left_back_foot,right_back_foot"
TARGET_JOINT="front_left_foot"
P2_REQUESTED_SEMANTICS="simulation_joint_state_override_lock"
P2_VELOCITY_OVERRIDE="0.0"
P2_ONSET_MIN="120"
P2_ONSET_MAX="700"
P2_FIXED_ONSET="50"

SOURCE_EXPERIMENT_NAME="teacher_p2_multijoint_velocity_curriculum_v2b__rlm1_stripped__p2_multi_joint_random"
SOURCE_RUN="2026-06-15_18-16-03_a1f_multijoint_velocity_p2_v2b_curriculum_curriculum_s2_realistic_transition_random_p2__seed0"
SOURCE_CHECKPOINT="model_9997.pt"
SOURCE_CHECKPOINT_PATH="logs/rsl_rl/${SOURCE_EXPERIMENT_NAME}/${SOURCE_RUN}/${SOURCE_CHECKPOINT}"

# Keep the source experiment root so upstream RSL-RL resume lookup can resolve SOURCE_RUN.
EXPERIMENT_NAME="${SOURCE_EXPERIMENT_NAME}"
RUN_NAME="a1f_multijoint_velocity_p2_hard_foot_finetune_120_700_i2000__seed0"
EXTRA_ARGS=()
COMMAND=()

show_help() {
    cat <<'EOF'
T13-D hard-foot focused A1-F multi-joint teacher finetune launcher.

This helper resumes from the valid T13-B v2b teacher checkpoint and restricts
random_per_env P2 sampling to hard foot joints. It refuses to launch unless
--execute_finetune is provided.

Defaults:
  source_checkpoint: logs/rsl_rl/teacher_p2_multijoint_velocity_curriculum_v2b__rlm1_stripped__p2_multi_joint_random/2026-06-15_18-16-03_a1f_multijoint_velocity_p2_v2b_curriculum_curriculum_s2_realistic_transition_random_p2__seed0/model_9997.pt
  task: Isaac-Ant-Teacher-Velocity-MultiJointP2-Flat-v0
  joint_set: hard_foot
  hard_foot joints: front_left_foot,front_right_foot,left_back_foot
  balanced_foot joints: front_left_foot,front_right_foot,left_back_foot,right_back_foot
  onset: U(120,700)
  num_envs: 4096
  max_iterations: 2000
  device: cuda
  command range: task default vx in [0.2, 1.5], vy = 0, yaw = 0
  semantics: simulation_joint_state_override_lock
  fallback: disabled

Options:
  --execute_finetune        Required to launch training
  --joint_set NAME          hard_foot or balanced_foot; default: hard_foot
  --num_envs N              Default: 4096; use 2048 if VRAM is tight
  --max_iterations N        Default: 2000; suggested range 1500-3000
  --device DEVICE           Default: cuda
  --seed N                  Default: 0
  --run_name NAME           Override run name
  --headless                Enabled by default
  --no-headless             Run with GUI
  --help, -h                Show this help

This is candidate-level finetuning only. It does not freeze checkpoints, update
checkpoint pointers, enable fallback, train A2/A5/A7, or modify the source v2b
checkpoint.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --help|-h)
            show_help
            exit 0
            ;;
        --execute_finetune)
            EXECUTE_FINETUNE=1
            shift
            ;;
        --joint_set)
            JOINT_SET="$2"
            shift 2
            ;;
        --joint_set=*)
            JOINT_SET="${1#*=}"
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
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --device=*)
            DEVICE="${1#*=}"
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
        --p2_allow_fallback)
            echo "[T13D ERROR] P2 fallback is disabled for hard-foot finetuning." >&2
            exit 2
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

if [[ "${JOINT_SET}" == "hard_foot" ]]; then
    SUPPORTED_TARGET_JOINTS="${HARD_FOOT_JOINTS}"
elif [[ "${JOINT_SET}" == "balanced_foot" ]]; then
    SUPPORTED_TARGET_JOINTS="${BALANCED_FOOT_JOINTS}"
else
    echo "[T13D ERROR] --joint_set must be hard_foot or balanced_foot, got ${JOINT_SET}" >&2
    exit 2
fi

if [[ ! -f "${REPO_ROOT}/${CONFIG}" ]]; then
    echo "[T13D ERROR] config not found: ${CONFIG}" >&2
    exit 2
fi

if [[ ! -f "${REPO_ROOT}/${FAULT_CONFIG}" ]]; then
    echo "[T13D ERROR] fault_config not found: ${FAULT_CONFIG}" >&2
    exit 2
fi

if [[ ! -f "${REPO_ROOT}/${SOURCE_CHECKPOINT_PATH}" ]]; then
    echo "[T13D ERROR] source checkpoint not found: ${SOURCE_CHECKPOINT_PATH}" >&2
    exit 2
fi

build_command() {
    COMMAND=(
        python
        trainers/rsl_rl_train.py
        --skip_checkpoint_pointer
        --enable_p2_joint_lock
        --p2_fault_config "${FAULT_CONFIG}"
        --p2_target_joint "${TARGET_JOINT}"
        --p2_target_joint_mode random_per_env
        --p2_supported_target_joints "${SUPPORTED_TARGET_JOINTS}"
        --p2_fault_onset_step "${P2_FIXED_ONSET}"
        --p2_fault_onset_mode random_uniform
        --p2_fault_onset_step_min "${P2_ONSET_MIN}"
        --p2_fault_onset_step_max "${P2_ONSET_MAX}"
        --p2_expected_action_dim 8
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
        --device "${DEVICE}"
        --resume
        --load_run "${SOURCE_RUN}"
        --checkpoint "${SOURCE_CHECKPOINT}"
    )
    if [[ "${HEADLESS}" == "1" ]]; then
        COMMAND+=(--headless)
    fi
    COMMAND+=("${EXTRA_ARGS[@]}")
}

print_command() {
    printf '%q ' "${COMMAND[@]}"
}

cd "${REPO_ROOT}"
build_command

echo "[T13D] hard-foot focused A1-F multi-joint teacher finetune"
echo "  not_paper_grade_training: true"
echo "  source_checkpoint: ${SOURCE_CHECKPOINT_PATH}"
echo "  source_checkpoint_modified: false"
echo "  task: ${TASK}"
echo "  stage: ${STAGE}"
echo "  method: ${METHOD}"
echo "  fault: ${FAULT}"
echo "  joint_set: ${JOINT_SET}"
echo "  supported_target_joints: ${SUPPORTED_TARGET_JOINTS}"
echo "  target_joint_mode: random_per_env"
echo "  active_locked_joints_per_env: 1"
echo "  onset_range: [${P2_ONSET_MIN}, ${P2_ONSET_MAX}]"
echo "  num_envs: ${NUM_ENVS}"
echo "  max_iterations: ${MAX_ITERATIONS}"
echo "  device: ${DEVICE}"
echo "  experiment_name: ${EXPERIMENT_NAME}"
echo "  run_name: ${RUN_NAME}"
echo "  expected_output_run_folder: logs/rsl_rl/${EXPERIMENT_NAME}/<timestamp>_${RUN_NAME}"
echo "  privileged_teacher_obs_dim: 77"
echo "  student_obs_dim_future_distillation: 61"
echo "  q_lock_source: current selected joint position at onset"
echo "  requested_semantics: ${P2_REQUESTED_SEMANTICS}"
echo "  fallback: disabled"
echo "  pd_surrogate: disabled"
echo "  health_token: OFF"
echo "  checkpoint_pointer_update: disabled"
printf '  command_preview: '
print_command
printf '\n'

if [[ "${EXECUTE_FINETUNE}" != "1" ]]; then
    echo "[T13D ERROR] Refusing to launch finetune without --execute_finetune." >&2
    echo "[T13D ERROR] No training was run." >&2
    exit 3
fi

echo "[T13D] launching finetune"
"${COMMAND[@]}"
