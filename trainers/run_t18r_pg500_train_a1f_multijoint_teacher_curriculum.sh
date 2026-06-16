#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

BASE_LAUNCHER="${REPO_ROOT}/trainers/run_t13b_train_a1f_multijoint_teacher_curriculum.sh"

if [[ ! -f "${BASE_LAUNCHER}" ]]; then
    echo "[T18R-PG500 ERROR] Base curriculum launcher missing: ${BASE_LAUNCHER}" >&2
    exit 2
fi

cat <<'EOF'
[T18R-PG500] Paper-grade multi-joint A1-F teacher curriculum wrapper
  physics_frequency_hz: 500
  sim_dt: 0.002
  control_frequency_hz: 50
  decimation: 10
  control_dt: 0.02
  curriculum_variant: v2b
  default_iterations: S0=4000, S1=6000, S2=10000
  note: --execute_curriculum is still required to launch training.
EOF

exec bash "${BASE_LAUNCHER}" \
    --curriculum_variant v2b \
    --experiment_name t18r_pg500_teacher_p2_multijoint_velocity_50hz_500hzphys__rlm1_stripped__p2_multi_joint_random \
    --run_prefix a1f_multijoint_velocity_p2_t18r_pg500 \
    --s0_iterations 4000 \
    --s1_iterations 6000 \
    --s2_iterations 10000 \
    --t18r_pg500_timing \
    --require_t18r_pg500_timing \
    --control_frequency_hz 50 \
    --sim_dt 0.002 \
    --decimation 10 \
    --require_control_frequency_hz 50 \
    "$@"
