#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

CONFIG="configs/train/teacher_p2_canonical.yaml"
FAULT_CONFIG="configs/fault/joint_lock/p2_locked_joint.yaml"
TARGET_JOINT="front_left_foot"
FAULT_ONSET_STEP="50"
NUM_ENVS="4096"
MAX_ITERATIONS="2000"
SEED="0"
EXPERIMENT_NAME="teacher_p2__rlm1_stripped__canonical"
RUN_NAME="a1f_p2_teacher_canonical__seed0"
HEADLESS=1
DRY_RUN=0
P2_PREFLIGHT_PASSED=0
EXECUTE_ONE_STEP_SMOKE=0
ALLOW_FULL_TRAINING=0
EXTRA_ARGS=()

show_help() {
    cat <<'EOF'
T09-R2f A1-F P2 privileged teacher runtime-hook training helper.

Conference-stage checkpoint generation scaffold only. This helper prepares A1-F
teacher training metadata for the P2 single-joint-lock scope and delegates to
the existing T06 teacher training path with stable checkpoint pointer updates
disabled by passing --skip_checkpoint_pointer.

Important:
  The current repo-owned PPO launcher wires P2 through an action-override
  surrogate hook, not a true mechanical position-hold joint lock. Do not treat
  a raw run as canonical A1-F P2 evidence until preflight passes and the
  selected checkpoint is explicitly frozen.

Options:
  --config PATH              Default: configs/train/teacher_p2_canonical.yaml
  --fault_config PATH        Default: configs/fault/joint_lock/p2_locked_joint.yaml
  --target_joint NAME        Default: front_left_foot
  --fault_onset_step N       Default: 50
  --num_envs N               Default: 4096
  --max_iterations N         Default: 2000
  --seed N                   Default: 0
  --experiment_name NAME     Default: teacher_p2__rlm1_stripped__canonical
  --run_name NAME            Default: a1f_p2_teacher_canonical__seed0
  --headless                 Enabled by default
  --no-headless              Run with GUI
  --dry_run                  Print P2 hook command preview only; do not launch Isaac Sim
  --p2_preflight_passed      Required before one-step smoke or full training
  --execute_one_step_smoke   Explicit tiny runtime smoke: num_envs=8, max_iterations=1
  --allow_full_training      Required for full canonical training after preflight passes
  --help, -h                 Show this help

Expected log root:
  logs/rsl_rl/teacher_p2__rlm1_stripped__canonical/

Suggested future freeze command:
  python evaluators/freeze_t09_checkpoint.py --stage teacher_p2 --method rlm1_stripped --fault none --seed 0 --task Isaac-Ant-Teacher-v0 --checkpoint_path <exact_model.pt> --experiment_name teacher_p2__rlm1_stripped__canonical --run_name a1f_p2_teacher_canonical__seed0 --classification paper_grade_candidate --output_pointer checkpoints/rlm1_stripped/teacher_p2/none/seed0/canonical_checkpoint.yaml --manifest_path papers/conference/results/t09_a1f_p2_teacher_canonical_checkpoint_freeze.md

Safe checks:
  bash trainers/run_t09_train_a1f_p2_teacher_canonical.sh --dry_run
  python evaluators/preflight_t09_p2_joint_lock.py --dry_run

Runtime preflight, still no training:
  python evaluators/preflight_t09_p2_joint_lock.py --execute_preflight --headless

No training is run by --help or --dry_run.
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
        --fault_config)
            FAULT_CONFIG="$2"
            shift 2
            ;;
        --fault_config=*)
            FAULT_CONFIG="${1#*=}"
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
        --fault_onset_step)
            FAULT_ONSET_STEP="$2"
            shift 2
            ;;
        --fault_onset_step=*)
            FAULT_ONSET_STEP="${1#*=}"
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
        --dry_run)
            DRY_RUN=1
            shift
            ;;
        --p2_preflight_passed)
            P2_PREFLIGHT_PASSED=1
            shift
            ;;
        --execute_one_step_smoke)
            EXECUTE_ONE_STEP_SMOKE=1
            NUM_ENVS="8"
            MAX_ITERATIONS="1"
            shift
            ;;
        --allow_full_training)
            ALLOW_FULL_TRAINING=1
            shift
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

if [[ ! -f "${REPO_ROOT}/${CONFIG}" ]]; then
    echo "[T09-R2f ERROR] config not found: ${CONFIG}" >&2
    exit 2
fi

if [[ ! -f "${REPO_ROOT}/${FAULT_CONFIG}" ]]; then
    echo "[T09-R2f ERROR] fault_config not found: ${FAULT_CONFIG}" >&2
    exit 2
fi

if [[ "${EXECUTE_ONE_STEP_SMOKE}" == "1" && "${RUN_NAME}" == "a1f_p2_teacher_canonical__seed0" ]]; then
    RUN_NAME="a1f_p2_teacher_smoke__seed${SEED}"
fi

COMMAND=(
    bash
    trainers/run_t06_teacher.sh
    --skip_checkpoint_pointer
    --enable_p2_joint_lock
    --p2_fault_config "${FAULT_CONFIG}"
    --p2_target_joint "${TARGET_JOINT}"
    --p2_fault_onset_step "${FAULT_ONSET_STEP}"
    --config "${CONFIG}"
    --stage teacher_p2
    --method rlm1_stripped
    --fault P2_locked_joint
    --task Isaac-Ant-Teacher-v0
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
echo "[T09-R2f] A1-F P2 privileged teacher canonical training helper"
echo "  scope: A1-F P2 checkpoint generation scaffold; no A2/A5/A7 training"
echo "  task: Isaac-Ant-Teacher-v0"
echo "  method: rlm1_stripped"
echo "  fault_profile: P2_locked_joint"
echo "  fault_config: ${FAULT_CONFIG}"
echo "  target_joint: ${TARGET_JOINT}"
echo "  fault_onset_step: ${FAULT_ONSET_STEP}"
echo "  privileged_teacher: true"
echo "  deployment_facing: false"
echo "  health_token: OFF"
echo "  P4_scope: deferred"
echo "  num_envs: ${NUM_ENVS}"
echo "  max_iterations: ${MAX_ITERATIONS}"
echo "  expected_log_root: logs/rsl_rl/${EXPERIMENT_NAME}/"
echo "  checkpoint_pointer_update: disabled"
echo "  canonical_freeze_pointer: checkpoints/rlm1_stripped/teacher_p2/none/seed${SEED}/canonical_checkpoint.yaml"
echo "  canonical_freeze: manual via evaluators/freeze_t09_checkpoint.py"
echo "  P2_runtime_hook_enabled: True"
echo "  p2_runtime_curriculum_hook: repo-owned action-override surrogate"
printf '[T09-R2f] command:'
printf ' %q' "${COMMAND[@]}"
printf '\n'

if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[T09-R2f] dry_run: no Isaac Sim, no training, no checkpoint writes"
    exit 0
fi

if [[ "${P2_PREFLIGHT_PASSED}" != "1" ]]; then
    echo "[T09-R2f ERROR] Refusing runtime execution until P2 preflight is explicitly acknowledged with --p2_preflight_passed." >&2
    echo "[T09-R2f ERROR] Recommended first: python evaluators/preflight_t09_p2_joint_lock.py --execute_preflight --headless" >&2
    exit 3
fi

if [[ "${EXECUTE_ONE_STEP_SMOKE}" != "1" && "${ALLOW_FULL_TRAINING}" != "1" ]]; then
    echo "[T09-R2f ERROR] Full A1-F training remains blocked." >&2
    echo "[T09-R2f ERROR] Use --execute_one_step_smoke --p2_preflight_passed for a tiny hook smoke, or --allow_full_training --p2_preflight_passed for the real run." >&2
    exit 4
fi

if [[ "${EXECUTE_ONE_STEP_SMOKE}" == "1" ]]; then
    echo "[T09-R2f] one-step smoke enabled: num_envs=${NUM_ENVS}, max_iterations=${MAX_ITERATIONS}"
fi

exec "${COMMAND[@]}"
