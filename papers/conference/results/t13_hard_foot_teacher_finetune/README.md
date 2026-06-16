# T13D Hard-Foot Teacher Finetune Plan

This is a candidate-level finetune plan for the valid T13-B v2b privileged
multi-joint A1-F teacher. It focuses additional training exposure on hard foot
joint locks without modifying or invalidating the v2b checkpoint.

## Source Candidate

- checkpoint: `logs/rsl_rl/teacher_p2_multijoint_velocity_curriculum_v2b__rlm1_stripped__p2_multi_joint_random/2026-06-15_18-16-03_a1f_multijoint_velocity_p2_v2b_curriculum_curriculum_s2_realistic_transition_random_p2__seed0/model_9997.pt`
- task: `Isaac-Ant-Teacher-Velocity-MultiJointP2-Flat-v0`
- teacher observation: `77 = 61 base + 8 selected-joint one-hot + 8 q_lock vector`
- command range: `vx_cmd in [0.2, 1.5]`, `vy_cmd = 0`, `yaw_cmd = 0`
- q_lock semantics: captured from the selected joint's current simulated joint position at onset
- P2 semantics: direct `simulation_joint_state_override_lock`
- fallback: disabled
- PD surrogate: disabled

## Target Subsets

Default hard-foot subset:

- `front_left_foot`
- `front_right_foot`
- `left_back_foot`

Optional balanced-foot subset:

- `front_left_foot`
- `front_right_foot`
- `left_back_foot`
- `right_back_foot`

The subset is opt-in through `--p2_supported_target_joints`. The existing all-8
multi-joint P2 config remains unchanged when that argument is empty.

## Finetune Plan

- resume from v2b `model_9997.pt`
- onset range: `U(120,700)`
- num envs: `4096` default, `2048` fallback
- iterations: conservative `2000` default; suggested safe range `1500-3000`
- checkpoint pointer updates: disabled
- no A2/A5/A7 training
- no checkpoint freezing during finetune

The launcher keeps the v2b experiment root for RSL-RL resume compatibility and
uses a distinct hard-foot run name.

## Manual Commands

Smoke preview:

```bash
cd ~/thesis/IsaacLab
conda activate isaaclab
source _isaac_sim/setup_conda_env.sh

bash trainers/run_t13d_finetune_a1f_multijoint_teacher_hard_foot.sh \
  --num_envs 128 \
  --max_iterations 25 \
  --device cuda
```

Smoke finetune:

```bash
PYTHONUNBUFFERED=1 TERM=xterm bash trainers/run_t13d_finetune_a1f_multijoint_teacher_hard_foot.sh \
  --execute_finetune \
  --num_envs 128 \
  --max_iterations 25 \
  --device cuda \
  --run_name a1f_multijoint_velocity_p2_hard_foot_finetune_smoke_i25__seed0
```

Full hard-foot finetune:

```bash
PYTHONUNBUFFERED=1 TERM=xterm bash trainers/run_t13d_finetune_a1f_multijoint_teacher_hard_foot.sh \
  --execute_finetune \
  --joint_set hard_foot \
  --num_envs 4096 \
  --max_iterations 2000 \
  --device cuda
```

VRAM fallback:

```bash
PYTHONUNBUFFERED=1 TERM=xterm bash trainers/run_t13d_finetune_a1f_multijoint_teacher_hard_foot.sh \
  --execute_finetune \
  --joint_set hard_foot \
  --num_envs 2048 \
  --max_iterations 2000 \
  --device cuda
```

Post-finetune fixed-vx eval, replacing `<FINETUNE_CHECKPOINT>` with the selected
new checkpoint:

```bash
PYTHONUNBUFFERED=1 TERM=xterm python evaluators/run_t13c_multijoint_teacher_eval.py \
  --execute_eval \
  --checkpoint <FINETUNE_CHECKPOINT> \
  --protocol default \
  --velocity_mode fixed_vx_1p0 \
  --num_envs 512 \
  --num_steps 1000 \
  --seed 0 \
  --device cuda \
  --output_root papers/conference/results/t13d_hard_foot_teacher_eval_fixed_vx
```

Post-finetune command-random eval:

```bash
PYTHONUNBUFFERED=1 TERM=xterm python evaluators/run_t13c_multijoint_teacher_eval.py \
  --execute_eval \
  --checkpoint <FINETUNE_CHECKPOINT> \
  --protocol default \
  --velocity_mode command_random \
  --num_envs 512 \
  --num_steps 1000 \
  --seed 0 \
  --device cuda \
  --output_root papers/conference/results/t13d_hard_foot_teacher_eval_command_random
```

## Risks And Controls

- Overfitting risk: hard-foot-only training can regress leg-joint performance.
- Control: keep the run short, evaluate both fixed-vx and command-random modes,
  and compare against the original v2b teacher before distillation.
- Exposure balance: use `--joint_set balanced_foot` if right-back-foot retention
  is desired.
- Deployment guardrail: students still receive no fault vector, no q_lock vector,
  no health token, no UQ, and no CBF.
