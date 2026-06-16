# T13-B Multi-Joint Privileged Teacher Training Plan

This note prepares the user-run commands for the multi-joint P2 privileged
teacher. It does not start training, freeze checkpoints, update checkpoint
pointers, run evaluation, or modify T12 results.

## Goal

Train a command-conditioned A1-F multi-joint privileged velocity teacher for
the conference-stage RLM1 stripped pipeline:

- task: `Isaac-Ant-Teacher-Velocity-MultiJointP2-Flat-v0`
- stage: `teacher_p2_multijoint_velocity_curriculum_v2`
- method: `rlm1_stripped`
- fault: `p2_multi_joint_random`
- seed: `0`
- P2 mode: `target_joint_mode=random_per_env`
- semantics: `simulation_joint_state_override_lock`
- fallback: disabled
- command range: `vx_cmd in [0.2, 1.5]`, `vy_cmd = 0`, `yaw_cmd = 0`

## Confirmed T13-A Preflight

- supported joint count: `8`
- teacher policy obs dim: `77`
- teacher privileged terms:
  - `p2_fault_joint_one_hot`, dim `8`
  - `p2_fault_q_lock_vector`, dim `8`
- teacher obs definition: `77 = 61 base obs + 8 one-hot + 8 q_lock`
- student/deployment obs remains fault-descriptor-free
- health token: OFF
- actual semantics: `simulation_joint_state_override_lock`
- fallback used: `false`
- PD surrogate used: `false`
- selected joint override applied: `true`

## V2 Curriculum Launcher

```text
trainers/run_t13b_train_a1f_multijoint_teacher_curriculum.sh
```

This launcher uses resume chaining:

```text
S0 -> S1 -> S2
```

Resume uses the existing RSL-RL flags:

```text
--resume --load_run <previous_timestamped_run_dir> --checkpoint <previous_model.pt>
```

## Velocity Command Conditioning

The multi-joint teacher task now uses a command-conditioned forward velocity
range:

```text
vx_cmd: 0.2 to 1.5 m/s
vy_cmd: 0.0
yaw_cmd: 0.0
```

This is defined on the multi-joint teacher task only. The existing base velocity
task remains fixed at `vx_cmd = 1.0`.

## Fault Curriculum

The real target scenario is:

```text
healthy forward locomotion -> sudden locked-joint P2 fault during locomotion -> recovery -> continued forward locomotion
```

The currently running curriculum is preserved as **v2a**. It includes very
early fault-onset samples in S2. Treat those early samples as robustness/stress
coverage, not the main real-scenario recovery condition.

| v2a stage | P2 hook | onset | iterations | purpose |
| --- | --- | --- | ---: | --- |
| S0 | off | no fault | 2000 | Healthy command-tracking warmup with 77-D teacher obs and zero fault vectors. |
| S1 | on | random `250-700` | 3000 | Late random one-joint-per-env P2 recovery. |
| S2 | on | random `30-700` | 5000 | Mixed early-to-late random one-joint-per-env P2 recovery; includes early-onset stress samples. |

If v2a looks too biased toward early-fault morphology learning, prefer **v2b**
for the next run:

| v2b stage | P2 hook | onset | iterations | purpose |
| --- | --- | --- | ---: | --- |
| S0 | off | no fault | 2000 | Healthy command-tracking warmup with 77-D teacher obs and zero fault vectors. |
| S1 | on | random `250-700` | 3000 | Late random one-joint-per-env P2 recovery. |
| S2 | on | random `120-700` | 5000 | Main healthy-to-fault transition recovery training. |
| S3 optional | on | random `30-700` | 1000 default | Early-onset robustness/stress only after S2 if needed. |

True no-fault 77-D warmup is supported because the multi-joint teacher
observation terms return zero `p2_fault_joint_one_hot` and zero
`p2_fault_q_lock_vector` when the P2 wrapper is not attached.

## V2 Smoke Command

This runs a tiny three-stage smoke through the same resume path. It is not a
usable teacher.

```bash
cd ~/thesis/IsaacLab
conda activate isaaclab
source _isaac_sim/setup_conda_env.sh

PYTHONUNBUFFERED=1 TERM=xterm bash trainers/run_t13b_train_a1f_multijoint_teacher_curriculum.sh \
  --execute_curriculum \
  --curriculum_variant v2a \
  --num_envs 8 \
  --device cuda \
  --s0_iterations 1 \
  --s1_iterations 1 \
  --s2_iterations 1 \
  --run_prefix a1f_multijoint_velocity_p2_v2_smoke
```

## V2 Full Curriculum Command

```bash
cd ~/thesis/IsaacLab
conda activate isaaclab
source _isaac_sim/setup_conda_env.sh

PYTHONUNBUFFERED=1 TERM=xterm bash trainers/run_t13b_train_a1f_multijoint_teacher_curriculum.sh \
  --execute_curriculum \
  --curriculum_variant v2a \
  --num_envs 4096 \
  --device cuda
```

If `4096` envs hits VRAM limits, use:

```bash
PYTHONUNBUFFERED=1 TERM=xterm bash trainers/run_t13b_train_a1f_multijoint_teacher_curriculum.sh \
  --execute_curriculum \
  --curriculum_variant v2a \
  --num_envs 2048 \
  --device cuda
```

## V2B Recommended Real-Scenario Curriculum

Use this only for a new run. It does not modify or invalidate the currently
running v2a run.

```bash
cd ~/thesis/IsaacLab
conda activate isaaclab
source _isaac_sim/setup_conda_env.sh

PYTHONUNBUFFERED=1 TERM=xterm bash trainers/run_t13b_train_a1f_multijoint_teacher_curriculum.sh \
  --execute_curriculum \
  --curriculum_variant v2b \
  --num_envs 4096 \
  --device cuda
```

Fallback if `4096` envs hits VRAM limits:

```bash
PYTHONUNBUFFERED=1 TERM=xterm bash trainers/run_t13b_train_a1f_multijoint_teacher_curriculum.sh \
  --execute_curriculum \
  --curriculum_variant v2b \
  --num_envs 2048 \
  --device cuda
```

Optional early-onset robustness/stress stage after S2:

```bash
PYTHONUNBUFFERED=1 TERM=xterm bash trainers/run_t13b_train_a1f_multijoint_teacher_curriculum.sh \
  --execute_curriculum \
  --curriculum_variant v2b \
  --enable_s3_stress \
  --num_envs 4096 \
  --device cuda
```

## V2 Stage Commands

The launcher builds and runs stages sequentially. S1 resumes from S0, S2
resumes from S1, and optional v2b S3 resumes from S2.

S0 command shape:

```bash
python trainers/rsl_rl_train.py \
  --skip_checkpoint_pointer \
  --config configs/train/teacher_velocity.yaml \
  --stage teacher_p2_multijoint_velocity_curriculum_v2 \
  --method rlm1_stripped \
  --fault p2_multi_joint_random \
  --task Isaac-Ant-Teacher-Velocity-MultiJointP2-Flat-v0 \
  --seed 0 \
  --experiment_name teacher_p2_multijoint_velocity_curriculum_v2__rlm1_stripped__p2_multi_joint_random \
  --run_name a1f_multijoint_velocity_p2_v2_curriculum_s0_no_fault_warmup__seed0 \
  --num_envs 4096 \
  --max_iterations 2000 \
  --device cuda \
  --headless
```

S1 command shape:

```bash
python trainers/rsl_rl_train.py \
  --skip_checkpoint_pointer \
  --enable_p2_joint_lock \
  --p2_fault_config configs/fault/joint_lock/p2_multi_joint_random.yaml \
  --p2_target_joint front_left_foot \
  --p2_target_joint_mode random_per_env \
  --p2_fault_onset_step 50 \
  --p2_fault_onset_mode random_uniform \
  --p2_fault_onset_step_min 250 \
  --p2_fault_onset_step_max 700 \
  --p2_expected_action_dim 8 \
  --p2_requested_semantics simulation_joint_state_override_lock \
  --p2_velocity_override 0.0 \
  --config configs/train/teacher_velocity.yaml \
  --stage teacher_p2_multijoint_velocity_curriculum_v2 \
  --method rlm1_stripped \
  --fault p2_multi_joint_random \
  --task Isaac-Ant-Teacher-Velocity-MultiJointP2-Flat-v0 \
  --seed 0 \
  --experiment_name teacher_p2_multijoint_velocity_curriculum_v2__rlm1_stripped__p2_multi_joint_random \
  --run_name a1f_multijoint_velocity_p2_v2_curriculum_s1_late_random_p2__seed0 \
  --num_envs 4096 \
  --max_iterations 3000 \
  --device cuda \
  --headless \
  --resume --load_run <S0_TIMESTAMPED_RUN_DIR> --checkpoint <S0_MODEL_PT>
```

S2 command shape:

```bash
python trainers/rsl_rl_train.py \
  --skip_checkpoint_pointer \
  --enable_p2_joint_lock \
  --p2_fault_config configs/fault/joint_lock/p2_multi_joint_random.yaml \
  --p2_target_joint front_left_foot \
  --p2_target_joint_mode random_per_env \
  --p2_fault_onset_step 50 \
  --p2_fault_onset_mode random_uniform \
  --p2_fault_onset_step_min 30 \
  --p2_fault_onset_step_max 700 \
  --p2_expected_action_dim 8 \
  --p2_requested_semantics simulation_joint_state_override_lock \
  --p2_velocity_override 0.0 \
  --config configs/train/teacher_velocity.yaml \
  --stage teacher_p2_multijoint_velocity_curriculum_v2 \
  --method rlm1_stripped \
  --fault p2_multi_joint_random \
  --task Isaac-Ant-Teacher-Velocity-MultiJointP2-Flat-v0 \
  --seed 0 \
  --experiment_name teacher_p2_multijoint_velocity_curriculum_v2__rlm1_stripped__p2_multi_joint_random \
  --run_name a1f_multijoint_velocity_p2_v2_curriculum_s2_mixed_random_p2__seed0 \
  --num_envs 4096 \
  --max_iterations 5000 \
  --device cuda \
  --headless \
  --resume --load_run <S1_TIMESTAMPED_RUN_DIR> --checkpoint <S1_MODEL_PT>
```

For v2b, S2 uses:

```text
--p2_fault_onset_step_min 120
--p2_fault_onset_step_max 700
--run_name a1f_multijoint_velocity_p2_v2b_curriculum_s2_realistic_transition_random_p2__seed0
```

The optional v2b S3 stress stage uses:

```text
--p2_fault_onset_step_min 30
--p2_fault_onset_step_max 700
--run_name a1f_multijoint_velocity_p2_v2b_curriculum_s3_early_onset_stress__seed0
```

## V0 Fixed-Vx Early-Onset Plan

The earlier command below is retained only as v0/historical documentation. It
used fixed `vx_cmd=1.0` and early random onset `30-150`, so v2 above is the
recommended restart path.

### V0 Smoke Command

This is a tiny training smoke only. It is not a usable teacher.

```bash
cd ~/thesis/IsaacLab
conda activate isaaclab
source _isaac_sim/setup_conda_env.sh

PYTHONUNBUFFERED=1 TERM=xterm python trainers/rsl_rl_train.py \
  --skip_checkpoint_pointer \
  --enable_p2_joint_lock \
  --p2_fault_config configs/fault/joint_lock/p2_multi_joint_random.yaml \
  --p2_target_joint front_left_foot \
  --p2_target_joint_mode random_per_env \
  --p2_fault_onset_step 50 \
  --p2_fault_onset_mode random_uniform \
  --p2_fault_onset_step_min 30 \
  --p2_fault_onset_step_max 150 \
  --p2_expected_action_dim 8 \
  --p2_requested_semantics simulation_joint_state_override_lock \
  --p2_velocity_override 0.0 \
  --config configs/train/teacher_velocity.yaml \
  --stage teacher_p2_multijoint_velocity_curriculum \
  --method rlm1_stripped \
  --fault p2_multi_joint_random \
  --task Isaac-Ant-Teacher-Velocity-MultiJointP2-Flat-v0 \
  --seed 0 \
  --experiment_name teacher_p2_multijoint_velocity_curriculum__rlm1_stripped__p2_multi_joint_random \
  --run_name a1f_multijoint_velocity_p2_random_smoke__seed0 \
  --num_envs 8 \
  --max_iterations 1 \
  --headless \
  --device cuda
```

### V0 Full Training Command

This is the first full multi-joint teacher candidate run. It is still
candidate-level until evaluated and explicitly selected/frozen.

```bash
cd ~/thesis/IsaacLab
conda activate isaaclab
source _isaac_sim/setup_conda_env.sh

PYTHONUNBUFFERED=1 TERM=xterm python trainers/rsl_rl_train.py \
  --skip_checkpoint_pointer \
  --enable_p2_joint_lock \
  --p2_fault_config configs/fault/joint_lock/p2_multi_joint_random.yaml \
  --p2_target_joint front_left_foot \
  --p2_target_joint_mode random_per_env \
  --p2_fault_onset_step 50 \
  --p2_fault_onset_mode random_uniform \
  --p2_fault_onset_step_min 30 \
  --p2_fault_onset_step_max 150 \
  --p2_expected_action_dim 8 \
  --p2_requested_semantics simulation_joint_state_override_lock \
  --p2_velocity_override 0.0 \
  --config configs/train/teacher_velocity.yaml \
  --stage teacher_p2_multijoint_velocity_curriculum \
  --method rlm1_stripped \
  --fault p2_multi_joint_random \
  --task Isaac-Ant-Teacher-Velocity-MultiJointP2-Flat-v0 \
  --seed 0 \
  --experiment_name teacher_p2_multijoint_velocity_curriculum__rlm1_stripped__p2_multi_joint_random \
  --run_name a1f_multijoint_velocity_p2_random_full__seed0 \
  --num_envs 4096 \
  --max_iterations 3000 \
  --headless \
  --device cuda
```

## Expected Outputs

Expected RSL-RL log root:

```text
logs/rsl_rl/teacher_p2_multijoint_velocity_curriculum_v2__rlm1_stripped__p2_multi_joint_random/
```

Expected run folder patterns:

```text
logs/rsl_rl/teacher_p2_multijoint_velocity_curriculum_v2__rlm1_stripped__p2_multi_joint_random/<timestamp>_a1f_multijoint_velocity_p2_v2_curriculum_s0_no_fault_warmup__seed0/
logs/rsl_rl/teacher_p2_multijoint_velocity_curriculum_v2__rlm1_stripped__p2_multi_joint_random/<timestamp>_a1f_multijoint_velocity_p2_v2_curriculum_s1_late_random_p2__seed0/
logs/rsl_rl/teacher_p2_multijoint_velocity_curriculum_v2__rlm1_stripped__p2_multi_joint_random/<timestamp>_a1f_multijoint_velocity_p2_v2_curriculum_s2_mixed_random_p2__seed0/
```

Expected checkpoint pattern:

```text
logs/rsl_rl/teacher_p2_multijoint_velocity_curriculum_v2__rlm1_stripped__p2_multi_joint_random/<timestamp>_<run_name>/model_*.pt
```

Checkpoint pointers are intentionally not updated by these commands because
`--skip_checkpoint_pointer` is required for P2 runtime-hook training.

## Next Step

After training, run a guarded multi-joint teacher evaluation scaffold and select
or reject the resulting teacher checkpoint before rebuilding the dataset,
A2/A2-history, A5, and A7 A0-anchored residual variant pipeline.
