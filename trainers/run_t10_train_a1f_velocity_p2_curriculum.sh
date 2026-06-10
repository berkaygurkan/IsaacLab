#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

CONFIG="configs/train/teacher_velocity.yaml"
FAULT_CONFIG="configs/fault/joint_lock/p2_locked_joint.yaml"
TASK="Isaac-Ant-Teacher-Velocity-Flat-v0"
STAGE="teacher_p2_velocity_curriculum"
METHOD="rlm1_stripped"
FAULT="P2_locked_joint"
SEED="0"
NUM_ENVS="1024"
EXPERIMENT_NAME="teacher_p2_velocity_curriculum__rlm1_stripped__p2_locked_joint"
RUN_PREFIX="a1f_velocity_p2"
TARGET_JOINT="front_left_foot"
P2_KP="4.0"
P2_KD="0.4"
P2_ACTION_CLIP="1.0"
P2_REQUESTED_SEMANTICS="simulation_joint_state_override_lock"
P2_VELOCITY_OVERRIDE="0.0"
S0_ITERATIONS="500"
S1_ITERATIONS="750"
S2_ITERATIONS="750"
S3_ITERATIONS="1000"
HEADLESS=1
EXECUTE_CURRICULUM=0
EXTRA_ARGS=()

show_help() {
    cat <<'EOF'
T10 A1-F-Vel P2 privileged teacher onset-curriculum launcher.

This helper trains the same privileged A1-F velocity teacher across four
progressively harder P2 onset schedules. It delegates to trainers/rsl_rl_train.py,
uses the repo-owned P2 simulation-state override wrapper, disables checkpoint
pointer updates, and refuses runtime launch unless --execute_curriculum is set.

Defaults:
  task: Isaac-Ant-Teacher-Velocity-Flat-v0
  stage: teacher_p2_velocity_curriculum
  method: rlm1_stripped
  fault: P2_locked_joint
  seed: 0
  num_envs: 1024
  experiment_name: teacher_p2_velocity_curriculum__rlm1_stripped__p2_locked_joint
  run_prefix: a1f_velocity_p2
  semantics: simulation_joint_state_override_lock
  fallback: disabled

Curriculum stages:
  s0 healthy/no-effective-fault: fixed onset 2000, 500 iterations
  s1 late fault: random_uniform [300, 600], 750 iterations
  s2 medium fault: random_uniform [100, 300], 750 iterations
  s3 target fault: random_uniform [30, 150], 1000 iterations

Options:
  --execute_curriculum      Required to launch training
  --config PATH             Default: configs/train/teacher_velocity.yaml
  --fault_config PATH       Default: configs/fault/joint_lock/p2_locked_joint.yaml
  --num_envs N              Default: 1024
  --seed N                  Default: 0
  --s0_iterations N         Default: 500
  --s1_iterations N         Default: 750
  --s2_iterations N         Default: 750
  --s3_iterations N         Default: 1000
  --experiment_name NAME    Default: teacher_p2_velocity_curriculum__rlm1_stripped__p2_locked_joint
  --run_prefix NAME         Default: a1f_velocity_p2
  --headless                Enabled by default
  --no-headless             Run with GUI
  --help, -h                Show this help

This is candidate-level training only. It does not freeze checkpoints, update
checkpoint pointers, enable fallback, or train student/residual methods.

Example:
  bash trainers/run_t10_train_a1f_velocity_p2_curriculum.sh --execute_curriculum
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --help|-h)
            show_help
            exit 0
            ;;
        --execute_curriculum)
            EXECUTE_CURRICULUM=1
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
        --num_envs)
            NUM_ENVS="$2"
            shift 2
            ;;
        --num_envs=*)
            NUM_ENVS="${1#*=}"
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
        --s0_iterations)
            S0_ITERATIONS="$2"
            shift 2
            ;;
        --s0_iterations=*)
            S0_ITERATIONS="${1#*=}"
            shift
            ;;
        --s1_iterations)
            S1_ITERATIONS="$2"
            shift 2
            ;;
        --s1_iterations=*)
            S1_ITERATIONS="${1#*=}"
            shift
            ;;
        --s2_iterations)
            S2_ITERATIONS="$2"
            shift 2
            ;;
        --s2_iterations=*)
            S2_ITERATIONS="${1#*=}"
            shift
            ;;
        --s3_iterations)
            S3_ITERATIONS="$2"
            shift 2
            ;;
        --s3_iterations=*)
            S3_ITERATIONS="${1#*=}"
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
        --run_prefix)
            RUN_PREFIX="$2"
            shift 2
            ;;
        --run_prefix=*)
            RUN_PREFIX="${1#*=}"
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
            echo "[T10-CURRICULUM ERROR] P2 fallback is disabled for this curriculum launcher." >&2
            exit 2
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

if [[ ! -f "${REPO_ROOT}/${CONFIG}" ]]; then
    echo "[T10-CURRICULUM ERROR] config not found: ${CONFIG}" >&2
    exit 2
fi

if [[ ! -f "${REPO_ROOT}/${FAULT_CONFIG}" ]]; then
    echo "[T10-CURRICULUM ERROR] fault_config not found: ${FAULT_CONFIG}" >&2
    exit 2
fi

STAGE_NAMES=(
    "s0_healthy"
    "s1_late"
    "s2_medium"
    "s3_target"
)
STAGE_SUFFIXES=(
    "curriculum_s0_healthy"
    "curriculum_s1_late"
    "curriculum_s2_medium"
    "curriculum_s3_target"
)
STAGE_PURPOSES=(
    "no effective fault inside normal episode horizon"
    "late random P2 onset"
    "medium random P2 onset"
    "target conference P2 random onset"
)
STAGE_ITERATIONS=(
    "${S0_ITERATIONS}"
    "${S1_ITERATIONS}"
    "${S2_ITERATIONS}"
    "${S3_ITERATIONS}"
)
STAGE_ONSET_MODES=(
    "fixed"
    "random_uniform"
    "random_uniform"
    "random_uniform"
)
STAGE_FIXED_ONSETS=(
    "2000"
    "50"
    "50"
    "50"
)
STAGE_ONSET_MINS=(
    "2000"
    "300"
    "100"
    "30"
)
STAGE_ONSET_MAXS=(
    "2000"
    "600"
    "300"
    "150"
)

find_latest_checkpoint() {
    local run_name="$1"
    local log_root="${REPO_ROOT}/logs/rsl_rl/${EXPERIMENT_NAME}"
    local run_dir
    local checkpoint
    if [[ ! -d "${log_root}" ]]; then
        echo "[T10-CURRICULUM ERROR] log root not found after stage: ${log_root}" >&2
        return 1
    fi
    run_dir="$(find "${log_root}" -maxdepth 1 -type d -name "*_${run_name}" | sort | tail -n 1)"
    if [[ -z "${run_dir}" ]]; then
        echo "[T10-CURRICULUM ERROR] no run folder found for run_name=${run_name} under ${log_root}" >&2
        return 1
    fi
    checkpoint="$(find "${run_dir}" -maxdepth 1 -type f -name "model_*.pt" | sort -V | tail -n 1)"
    if [[ -z "${checkpoint}" ]]; then
        echo "[T10-CURRICULUM ERROR] no model_*.pt checkpoint found in ${run_dir}" >&2
        return 1
    fi
    printf '%s|%s\n' "${run_dir}" "${checkpoint}"
}

build_stage_command() {
    local index="$1"
    local resume_run="$2"
    local resume_checkpoint="$3"
    local run_name="${RUN_PREFIX}_${STAGE_SUFFIXES[$index]}__seed${SEED}"
    local command=(
        python
        trainers/rsl_rl_train.py
        --skip_checkpoint_pointer
        --enable_p2_joint_lock
        --p2_fault_config "${FAULT_CONFIG}"
        --p2_target_joint "${TARGET_JOINT}"
        --p2_fault_onset_step "${STAGE_FIXED_ONSETS[$index]}"
        --p2_fault_onset_mode "${STAGE_ONSET_MODES[$index]}"
        --p2_fault_onset_step_min "${STAGE_ONSET_MINS[$index]}"
        --p2_fault_onset_step_max "${STAGE_ONSET_MAXS[$index]}"
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
        --run_name "${run_name}"
        --num_envs "${NUM_ENVS}"
        --max_iterations "${STAGE_ITERATIONS[$index]}"
    )
    if [[ "${HEADLESS}" == "1" ]]; then
        command+=(--headless)
    fi
    if [[ -n "${resume_run}" && -n "${resume_checkpoint}" ]]; then
        command+=(--resume --load_run "${resume_run}" --checkpoint "${resume_checkpoint}")
    fi
    command+=("${EXTRA_ARGS[@]}")
    printf '%q ' "${command[@]}"
}

cd "${REPO_ROOT}"
echo "[T10-CURRICULUM] A1-F-Vel P2 privileged teacher onset curriculum"
echo "  not_paper_grade_training: true"
echo "  task: ${TASK}"
echo "  stage: ${STAGE}"
echo "  method: ${METHOD}"
echo "  fault: ${FAULT}"
echo "  seed: ${SEED}"
echo "  num_envs: ${NUM_ENVS}"
echo "  experiment_name: ${EXPERIMENT_NAME}"
echo "  expected_log_root: logs/rsl_rl/${EXPERIMENT_NAME}/"
echo "  checkpoint_pointer_update: disabled"
echo "  checkpoint_freeze: not_performed"
echo "  P2_runtime_hook_enabled: true"
echo "  P2_target_joint: ${TARGET_JOINT}"
echo "  requested_semantics: ${P2_REQUESTED_SEMANTICS}"
echo "  allow_fallback: false"
echo "  privileged_teacher: true"
echo "  deployment_facing: false"
echo "  health_token: OFF"
echo "  UQ: inactive"
echo "  CBF: inactive"
echo "  P3: inactive"
echo "  P4_scope: deferred"
echo "  student: disabled"
echo "  residual: disabled"
echo "  resume_supported: true via --resume --load_run --checkpoint passthrough"

for index in "${!STAGE_NAMES[@]}"; do
    run_name="${RUN_PREFIX}_${STAGE_SUFFIXES[$index]}__seed${SEED}"
    echo "[T10-CURRICULUM] stage ${index}: ${STAGE_NAMES[$index]}"
    echo "  purpose: ${STAGE_PURPOSES[$index]}"
    echo "  run_name: ${run_name}"
    echo "  max_iterations: ${STAGE_ITERATIONS[$index]}"
    echo "  onset_mode: ${STAGE_ONSET_MODES[$index]}"
    echo "  fixed_onset_step: ${STAGE_FIXED_ONSETS[$index]}"
    echo "  onset_step_min: ${STAGE_ONSET_MINS[$index]}"
    echo "  onset_step_max: ${STAGE_ONSET_MAXS[$index]}"
    if [[ "${index}" == "0" ]]; then
        echo "  resume_checkpoint_input: none"
    else
        previous_run_name="${RUN_PREFIX}_${STAGE_SUFFIXES[$((index - 1))]}__seed${SEED}"
        echo "  resume_checkpoint_input: latest checkpoint from previous run_name ${previous_run_name}"
    fi
    echo "  expected_output_run_folder: logs/rsl_rl/${EXPERIMENT_NAME}/<timestamp>_${run_name}"
    echo "  checkpoint_pointer_update: disabled"
done

if [[ "${EXECUTE_CURRICULUM}" != "1" ]]; then
    echo "[T10-CURRICULUM ERROR] Refusing to launch curriculum without --execute_curriculum." >&2
    echo "[T10-CURRICULUM ERROR] No training was run." >&2
    exit 3
fi

resume_run=""
resume_checkpoint=""
for index in "${!STAGE_NAMES[@]}"; do
    run_name="${RUN_PREFIX}_${STAGE_SUFFIXES[$index]}__seed${SEED}"
    command_text="$(build_stage_command "${index}" "${resume_run}" "${resume_checkpoint}")"
    echo "[T10-CURRICULUM] launching stage ${index}: ${STAGE_NAMES[$index]}"
    echo "  resume_run: ${resume_run:-none}"
    echo "  resume_checkpoint: ${resume_checkpoint:-none}"
    echo "  command: ${command_text}"
    eval "${command_text}"
    latest="$(find_latest_checkpoint "${run_name}")"
    resume_run="$(basename "${latest%%|*}")"
    resume_checkpoint="$(basename "${latest##*|}")"
    echo "[T10-CURRICULUM] stage ${index} complete"
    echo "  latest_run_dir: ${resume_run}"
    echo "  latest_checkpoint: ${resume_checkpoint}"
done

echo "[T10-CURRICULUM] curriculum complete"
echo "  final_resume_run: ${resume_run}"
echo "  final_checkpoint: ${resume_checkpoint}"
echo "  checkpoint_pointer_update: disabled"
echo "  canonical_freeze: not_performed"
