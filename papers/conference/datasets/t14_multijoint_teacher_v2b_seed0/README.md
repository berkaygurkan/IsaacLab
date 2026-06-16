# T14 Multi-Joint Teacher Dataset Scaffold

This directory is reserved for the T14 multi-joint A1-F teacher rollout
dataset. The collector scaffold is:

```text
evaluators/collect_t14_multijoint_teacher_dataset.py
```

No dataset has to exist here before the collector is run manually.

## Selected Teacher

Use the documented selected teacher:

```text
logs/rsl_rl/teacher_p2_multijoint_velocity_curriculum_v2b__rlm1_stripped__p2_multi_joint_random/2026-06-15_18-16-03_a1f_multijoint_velocity_p2_v2b_curriculum_curriculum_s2_realistic_transition_random_p2__seed0/model_9997.pt
```

Selection label: A1-F multi-joint teacher v2b, valid but foot-limited.

Selection note:

```text
papers/conference/results/t13_multi_joint_teacher_selection/README.md
```

The rejected T13-D hard-foot finetune checkpoint must not be used downstream.

## Default Collection

- velocity mode: `command_random`
- protocols:
  - `realistic_random`, onset `U(120,700)`
  - `late_random`, onset `U(250,700)`
- optional stress protocol: `stress_random`, onset `U(30,700)`
- default envs: `128`
- default steps: `1000`
- seed: `0`
- device: `cuda`

## Dataset Schema

Required arrays:

- `student_obs`: deployment-facing 61-D observation
- `teacher_obs`: privileged 77-D observation
- `teacher_action`
- `selected_fault_joint_index`
- `selected_fault_joint_one_hot`
- `q_lock_vector`
- `p2_fault_active`
- `fault_onset_step`
- `vx_cmd`
- `velocity_x`
- `vx_error`
- `yaw_rate`
- `yaw_error`
- `done`
- `episode_id`
- `timestep`
- `protocol_label`
- `velocity_mode_label`

Optional arrays are written only when `--a0_checkpoint` is explicitly provided:

- `a0_action`
- `a7_residual_target = teacher_action - a0_action`

## Guardrails

- RLM phase: RLM1 stripped / conference
- health token: OFF
- one selected locked joint per env/episode
- selected joint random over all 8 Ant actuated joints
- q_lock captured from current selected joint position at onset
- direct `simulation_joint_state_override_lock`
- enforce `q[selected_joint] = q_lock` and `qd[selected_joint] = 0`
- fallback disabled
- no PD surrogate
- no checkpoint modification
- no task config modification
- no P2 wrapper modification

The dataset is for downstream A2, A2-history H16, A5, and optional A7
A0-anchored residual scaffolding. It is not paper-grade final by itself.

## Manual Commands

Compile check:

```bash
python -m py_compile evaluators/collect_t14_multijoint_teacher_dataset.py
```

Dry-run preview:

```bash
python evaluators/collect_t14_multijoint_teacher_dataset.py \
  --dry_run \
  --protocol default \
  --velocity_mode command_random \
  --num_envs 16 \
  --num_steps 200 \
  --device cuda
```

Diagnostic smoke collection, one protocol block only:

```bash
cd ~/thesis/IsaacLab
conda activate isaaclab
source _isaac_sim/setup_conda_env.sh

PYTHONUNBUFFERED=1 TERM=xterm python evaluators/collect_t14_multijoint_teacher_dataset.py \
  --execute_collect \
  --headless \
  --protocol default \
  --velocity_mode command_random \
  --output_dir papers/conference/datasets/t14_multijoint_teacher_v2b_seed0_smoke \
  --dataset_tag t14_multijoint_teacher_v2b_seed0_smoke \
  --num_envs 16 \
  --num_steps 200 \
  --seed 0 \
  --device cuda \
  --progress_every 50 \
  --debug_max_protocols 1
```

If collection fails, inspect:

```text
papers/conference/datasets/t14_multijoint_teacher_v2b_seed0_smoke/terminal.log
papers/conference/datasets/t14_multijoint_teacher_v2b_seed0_smoke/error_summary.json
```

Default command-random collection:

```bash
cd ~/thesis/IsaacLab
conda activate isaaclab
source _isaac_sim/setup_conda_env.sh

PYTHONUNBUFFERED=1 TERM=xterm python evaluators/collect_t14_multijoint_teacher_dataset.py \
  --execute_collect \
  --headless \
  --protocol default \
  --velocity_mode command_random \
  --num_envs 128 \
  --num_steps 1000 \
  --seed 0 \
  --device cuda \
  --progress_every 100
```

Optional fixed-vx comparability collection:

```bash
PYTHONUNBUFFERED=1 TERM=xterm python evaluators/collect_t14_multijoint_teacher_dataset.py \
  --execute_collect \
  --headless \
  --protocol default \
  --velocity_mode fixed_vx_1p0 \
  --output_dir papers/conference/datasets/t14_multijoint_teacher_v2b_fixed_vx_seed0 \
  --dataset_tag t14_multijoint_teacher_v2b_fixed_vx_seed0 \
  --num_envs 128 \
  --num_steps 1000 \
  --seed 0 \
  --device cuda \
  --progress_every 100
```
