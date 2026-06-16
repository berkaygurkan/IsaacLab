#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

CONFIG="configs/train/teacher_velocity.yaml"
FAULT_CONFIG="configs/fault/joint_lock/p2_multi_joint_random.yaml"
TASK="Isaac-Ant-Teacher-Velocity-MultiJointP2-Flat-v0"
STAGE="teacher_p2_multijoint_velocity_curriculum_v2"
METHOD="rlm1_stripped"
FAULT="p2_multi_joint_random"
SEED="0"
NUM_ENVS="4096"
DEVICE="cuda"
EXPERIMENT_NAME="teacher_p2_multijoint_velocity_curriculum_v2__rlm1_stripped__p2_multi_joint_random"
RUN_PREFIX="a1f_multijoint_velocity_p2_v2"
TARGET_JOINT="front_left_foot"
P2_REQUESTED_SEMANTICS="simulation_joint_state_override_lock"
P2_VELOCITY_OVERRIDE="0.0"
CURRICULUM_VARIANT="v2a"
S0_ITERATIONS="2000"
S1_ITERATIONS="3000"
S2_ITERATIONS="5000"
S3_ITERATIONS="1000"
ENABLE_S3_STRESS=0
HEADLESS=1
EXECUTE_CURRICULUM=0
EXTRA_ARGS=()
STAGE_COMMAND=()
EXPERIMENT_NAME_OVERRIDDEN=0
RUN_PREFIX_OVERRIDDEN=0

show_help() {
    cat <<'EOF'
T13-B v2 A1-F multi-joint privileged teacher curriculum launcher.

This helper stages command-conditioned privileged teacher training for the
multi-joint P2 distribution. It delegates to trainers/rsl_rl_train.py, disables
checkpoint pointer updates, uses direct simulation-state override only for P2
stages, and refuses runtime launch unless --execute_curriculum is set.

Defaults:
  task: Isaac-Ant-Teacher-Velocity-MultiJointP2-Flat-v0
  stage: teacher_p2_multijoint_velocity_curriculum_v2
  method: rlm1_stripped
  fault: p2_multi_joint_random
  seed: 0
  num_envs: 4096
  device: cuda
  curriculum_variant: v2a
  experiment_name: teacher_p2_multijoint_velocity_curriculum_v2__rlm1_stripped__p2_multi_joint_random
  run_prefix: a1f_multijoint_velocity_p2_v2
  command range: vx in [0.2, 1.5], vy = 0, yaw = 0
  privileged teacher obs: 77 = 61 + 8 selected-joint one-hot + 8 q_lock
  P2 semantics: simulation_joint_state_override_lock
  fallback: disabled

Curriculum stages:
  v2a S0 no fault, 77-D obs zero fault vectors: 2000 iterations
  v2a S1 late random P2 onset [250, 700]: 3000 iterations
  v2a S2 mixed/stress random P2 onset [30, 700]: 5000 iterations

  v2b S0 no fault, 77-D obs zero fault vectors: 2000 iterations
  v2b S1 late random P2 onset [250, 700]: 3000 iterations
  v2b S2 realistic transition P2 onset [120, 700]: 5000 iterations
  v2b optional S3 early-onset stress [30, 700]: off by default

Options:
  --execute_curriculum      Required to launch training
  --curriculum_variant NAME Default: v2a; choices: v2a, v2b
  --enable_s3_stress        Add optional v2b S3 early-onset stress stage
  --config PATH             Default: configs/train/teacher_velocity.yaml
  --fault_config PATH       Default: configs/fault/joint_lock/p2_multi_joint_random.yaml
  --num_envs N              Default: 4096; use 2048 if VRAM is tight
  --device DEVICE           Default: cuda
  --seed N                  Default: 0
  --s0_iterations N         Default: 2000
  --s1_iterations N         Default: 3000
  --s2_iterations N         Default: 5000
  --s3_iterations N         Default: 1000; used only with --enable_s3_stress
  --experiment_name NAME    Override experiment root
  --run_prefix NAME         Override run-name prefix
  --headless                Enabled by default
  --no-headless             Run with GUI
  --help, -h                Show this help

This is candidate-level training only. It does not freeze checkpoints, update
checkpoint pointers, enable fallback, or train A2/A5/A7.
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
        --curriculum_variant)
            CURRICULUM_VARIANT="$2"
            shift 2
            ;;
        --curriculum_variant=*)
            CURRICULUM_VARIANT="${1#*=}"
            shift
            ;;
        --enable_s3_stress)
            ENABLE_S3_STRESS=1
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
            EXPERIMENT_NAME_OVERRIDDEN=1
            shift 2
            ;;
        --experiment_name=*)
            EXPERIMENT_NAME="${1#*=}"
            EXPERIMENT_NAME_OVERRIDDEN=1
            shift
            ;;
        --run_prefix)
            RUN_PREFIX="$2"
            RUN_PREFIX_OVERRIDDEN=1
            shift 2
            ;;
        --run_prefix=*)
            RUN_PREFIX="${1#*=}"
            RUN_PREFIX_OVERRIDDEN=1
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
            echo "[T13B-V2 ERROR] P2 fallback is disabled for multi-joint teacher training." >&2
            exit 2
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

if [[ "${CURRICULUM_VARIANT}" != "v2a" && "${CURRICULUM_VARIANT}" != "v2b" ]]; then
    echo "[T13B-V2 ERROR] --curriculum_variant must be v2a or v2b, got ${CURRICULUM_VARIANT}" >&2
    exit 2
fi

if [[ "${CURRICULUM_VARIANT}" == "v2b" && "${EXPERIMENT_NAME_OVERRIDDEN}" == "0" ]]; then
    EXPERIMENT_NAME="teacher_p2_multijoint_velocity_curriculum_v2b__rlm1_stripped__p2_multi_joint_random"
fi

if [[ "${CURRICULUM_VARIANT}" == "v2b" && "${RUN_PREFIX_OVERRIDDEN}" == "0" ]]; then
    RUN_PREFIX="a1f_multijoint_velocity_p2_v2b"
fi

if [[ ! -f "${REPO_ROOT}/${CONFIG}" ]]; then
    echo "[T13B-V2 ERROR] config not found: ${CONFIG}" >&2
    exit 2
fi

if [[ ! -f "${REPO_ROOT}/${FAULT_CONFIG}" ]]; then
    echo "[T13B-V2 ERROR] fault_config not found: ${FAULT_CONFIG}" >&2
    exit 2
fi

STAGE_NAMES=("s0_no_fault_warmup" "s1_late_random_p2")
STAGE_SUFFIXES=("curriculum_s0_no_fault_warmup" "curriculum_s1_late_random_p2")
STAGE_PURPOSES=(
    "healthy command-tracking warmup with 77-D obs and zero fault vectors"
    "late random one-joint-per-env P2 recovery"
)
STAGE_ITERATIONS=("${S0_ITERATIONS}" "${S1_ITERATIONS}")
STAGE_P2_ENABLED=("0" "1")
STAGE_FIXED_ONSETS=("none" "50")
STAGE_ONSET_MINS=("none" "250")
STAGE_ONSET_MAXS=("none" "700")

if [[ "${CURRICULUM_VARIANT}" == "v2a" ]]; then
    STAGE_NAMES+=("s2_mixed_random_p2")
    STAGE_SUFFIXES+=("curriculum_s2_mixed_random_p2")
    STAGE_PURPOSES+=("mixed early-to-late random one-joint-per-env P2 recovery")
    STAGE_ITERATIONS+=("${S2_ITERATIONS}")
    STAGE_P2_ENABLED+=("1")
    STAGE_FIXED_ONSETS+=("50")
    STAGE_ONSET_MINS+=("30")
    STAGE_ONSET_MAXS+=("700")
else
    STAGE_NAMES+=("s2_realistic_transition_random_p2")
    STAGE_SUFFIXES+=("curriculum_s2_realistic_transition_random_p2")
    STAGE_PURPOSES+=("healthy-to-fault transition recovery with fewer immediate-fault samples")
    STAGE_ITERATIONS+=("${S2_ITERATIONS}")
    STAGE_P2_ENABLED+=("1")
    STAGE_FIXED_ONSETS+=("50")
    STAGE_ONSET_MINS+=("120")
    STAGE_ONSET_MAXS+=("700")
    if [[ "${ENABLE_S3_STRESS}" == "1" ]]; then
        STAGE_NAMES+=("s3_early_onset_stress")
        STAGE_SUFFIXES+=("curriculum_s3_early_onset_stress")
        STAGE_PURPOSES+=("optional robustness stress after realistic transition training")
        STAGE_ITERATIONS+=("${S3_ITERATIONS}")
        STAGE_P2_ENABLED+=("1")
        STAGE_FIXED_ONSETS+=("50")
        STAGE_ONSET_MINS+=("30")
        STAGE_ONSET_MAXS+=("700")
    fi
fi

find_latest_checkpoint() {
    local run_name="$1"
    local log_root="${REPO_ROOT}/logs/rsl_rl/${EXPERIMENT_NAME}"
    local run_dir
    local checkpoint
    if [[ ! -d "${log_root}" ]]; then
        echo "[T13B-V2 ERROR] log root not found after stage: ${log_root}" >&2
        return 1
    fi
    run_dir="$(find "${log_root}" -maxdepth 1 -type d -name "*_${run_name}" | sort | tail -n 1)"
    if [[ -z "${run_dir}" ]]; then
        echo "[T13B-V2 ERROR] no run folder found for run_name=${run_name} under ${log_root}" >&2
        return 1
    fi
    checkpoint="$(find "${run_dir}" -maxdepth 1 -type f -name "model_*.pt" | sort -V | tail -n 1)"
    if [[ -z "${checkpoint}" ]]; then
        echo "[T13B-V2 ERROR] no model_*.pt checkpoint found in ${run_dir}" >&2
        return 1
    fi
    printf '%s|%s\n' "${run_dir}" "${checkpoint}"
}

build_stage_command() {
    local index="$1"
    local resume_run="$2"
    local resume_checkpoint="$3"
    local run_name="${RUN_PREFIX}_${STAGE_SUFFIXES[$index]}__seed${SEED}"
    STAGE_COMMAND=(
        python
        trainers/rsl_rl_train.py
        --skip_checkpoint_pointer
    )
    if [[ "${STAGE_P2_ENABLED[$index]}" == "1" ]]; then
        STAGE_COMMAND+=(
            --enable_p2_joint_lock
            --p2_fault_config "${FAULT_CONFIG}"
            --p2_target_joint "${TARGET_JOINT}"
            --p2_target_joint_mode random_per_env
            --p2_fault_onset_step "${STAGE_FIXED_ONSETS[$index]}"
            --p2_fault_onset_mode random_uniform
            --p2_fault_onset_step_min "${STAGE_ONSET_MINS[$index]}"
            --p2_fault_onset_step_max "${STAGE_ONSET_MAXS[$index]}"
            --p2_expected_action_dim 8
            --p2_requested_semantics "${P2_REQUESTED_SEMANTICS}"
            --p2_velocity_override "${P2_VELOCITY_OVERRIDE}"
        )
    fi
    STAGE_COMMAND+=(
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
        --device "${DEVICE}"
    )
    if [[ "${HEADLESS}" == "1" ]]; then
        STAGE_COMMAND+=(--headless)
    fi
    if [[ -n "${resume_run}" && -n "${resume_checkpoint}" ]]; then
        STAGE_COMMAND+=(--resume --load_run "${resume_run}" --checkpoint "${resume_checkpoint}")
    fi
    STAGE_COMMAND+=("${EXTRA_ARGS[@]}")
}

print_stage_command() {
    printf '%q ' "${STAGE_COMMAND[@]}"
}

cd "${REPO_ROOT}"
echo "[T13B-V2] A1-F multi-joint privileged teacher curriculum"
echo "  not_paper_grade_training: true"
echo "  task: ${TASK}"
echo "  stage: ${STAGE}"
echo "  method: ${METHOD}"
echo "  fault: ${FAULT}"
echo "  curriculum_variant: ${CURRICULUM_VARIANT}"
echo "  real_target_scenario: healthy_locomotion_to_sudden_locked_joint_to_recovery"
echo "  seed: ${SEED}"
echo "  num_envs: ${NUM_ENVS}"
echo "  device: ${DEVICE}"
echo "  experiment_name: ${EXPERIMENT_NAME}"
echo "  expected_log_root: logs/rsl_rl/${EXPERIMENT_NAME}/"
echo "  checkpoint_pointer_update: disabled"
echo "  checkpoint_freeze: not_performed"
echo "  command_conditioned: true"
echo "  vx_cmd_range: [0.2, 1.5]"
echo "  vy_cmd_range: [0.0, 0.0]"
echo "  yaw_cmd_range: [0.0, 0.0]"
echo "  privileged_teacher_obs_dim: 77"
echo "  student_obs_dim_future_distillation: 61"
echo "  health_token: OFF"
echo "  UQ: inactive"
echo "  CBF: inactive"
echo "  student: disabled"
echo "  residual: disabled"
echo "  A7_term: A0-anchored residual variant"
echo "  resume_supported: true via --resume --load_run --checkpoint passthrough"

for index in "${!STAGE_NAMES[@]}"; do
    run_name="${RUN_PREFIX}_${STAGE_SUFFIXES[$index]}__seed${SEED}"
    build_stage_command "${index}" "" ""
    echo "[T13B-V2] stage ${index}: ${STAGE_NAMES[$index]}"
    echo "  purpose: ${STAGE_PURPOSES[$index]}"
    echo "  run_name: ${run_name}"
    echo "  max_iterations: ${STAGE_ITERATIONS[$index]}"
    echo "  p2_enabled: ${STAGE_P2_ENABLED[$index]}"
    echo "  target_joint_mode: $([[ "${STAGE_P2_ENABLED[$index]}" == "1" ]] && echo random_per_env || echo no_fault_no_wrapper)"
    echo "  onset_mode: $([[ "${STAGE_P2_ENABLED[$index]}" == "1" ]] && echo random_uniform || echo none)"
    echo "  onset_step_min: ${STAGE_ONSET_MINS[$index]}"
    echo "  onset_step_max: ${STAGE_ONSET_MAXS[$index]}"
    if [[ "${index}" == "0" ]]; then
        echo "  resume_checkpoint_input: none"
    else
        previous_run_name="${RUN_PREFIX}_${STAGE_SUFFIXES[$((index - 1))]}__seed${SEED}"
        echo "  resume_checkpoint_input: latest checkpoint from previous run_name ${previous_run_name}"
    fi
    echo "  expected_output_run_folder: logs/rsl_rl/${EXPERIMENT_NAME}/<timestamp>_${run_name}"
    printf '  command_preview: '
    print_stage_command
    printf '\n'
done

if [[ "${EXECUTE_CURRICULUM}" != "1" ]]; then
    echo "[T13B-V2 ERROR] Refusing to launch curriculum without --execute_curriculum." >&2
    echo "[T13B-V2 ERROR] No training was run." >&2
    exit 3
fi

resume_run=""
resume_checkpoint=""
for index in "${!STAGE_NAMES[@]}"; do
    run_name="${RUN_PREFIX}_${STAGE_SUFFIXES[$index]}__seed${SEED}"
    build_stage_command "${index}" "${resume_run}" "${resume_checkpoint}"
    echo "[T13B-V2] launching stage ${index}: ${STAGE_NAMES[$index]}"
    echo "  resume_run: ${resume_run:-none}"
    echo "  resume_checkpoint: ${resume_checkpoint:-none}"
    printf '  command: '
    print_stage_command
    printf '\n'
    "${STAGE_COMMAND[@]}"
    latest="$(find_latest_checkpoint "${run_name}")"
    resume_run="$(basename "${latest%%|*}")"
    resume_checkpoint="$(basename "${latest##*|}")"
    echo "[T13B-V2] stage ${index} complete"
    echo "  latest_run_dir: ${resume_run}"
    echo "  latest_checkpoint: ${resume_checkpoint}"
done

echo "[T13B-V2] curriculum complete"
echo "  final_resume_run: ${resume_run}"
echo "  final_checkpoint: ${resume_checkpoint}"
echo "  checkpoint_pointer_update: disabled"
echo "  canonical_freeze: not_performed"
