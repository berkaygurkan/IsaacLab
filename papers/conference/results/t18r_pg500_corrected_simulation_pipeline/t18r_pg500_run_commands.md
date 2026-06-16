# T18-R-PG500 Manual Command Checklist

Run these manually. Codex did not run training, evaluation, Isaac Sim, or plotting.

## 1. Syntax Checks

```bash
cd ~/thesis/IsaacLab
python -m py_compile evaluators/t18r_control_timing.py
python -m py_compile evaluators/collect_t14_multijoint_teacher_dataset.py
python -m py_compile evaluators/run_t13c_multijoint_teacher_eval.py
python -m py_compile evaluators/run_t17_multijoint_closed_loop_eval.py
python -m py_compile evaluators/plot_t17_fixed_vx_fault_response.py
python -m py_compile evaluators/summarize_t18r_pg500_multiseed.py
python -m py_compile trainers/p2_joint_lock_training_wrapper.py
python -m py_compile trainers/rsl_rl_train.py
python -m py_compile trainers/train_t15_multijoint_a2_student_distill.py
python -m py_compile trainers/train_t16_multijoint_a5_history_residual.py
```

## 2. Git Diff Audit Commands

```bash
cd ~/thesis/IsaacLab
git status --short
git diff --name-only
git diff -- source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/ant/ant_env_cfg.py
git diff -- evaluators/t18r_control_timing.py trainers/p2_joint_lock_training_wrapper.py trainers/rsl_rl_train.py
```

Expected default Ant timing in `ant_env_cfg.py` remains:

- `self.decimation = 2`
- `self.sim.dt = 1 / 120.0`

## 3. Dry-Run Timing Checks

Dataset collector dry-run:

```bash
cd ~/thesis/IsaacLab
conda activate isaaclab

python evaluators/collect_t14_multijoint_teacher_dataset.py \
  --dry_run \
  --protocol realistic_random \
  --velocity_mode command_random \
  --num_envs 16 \
  --num_steps 200 \
  --seed 0 \
  --device cuda \
  --t18r_pg500_timing \
  --require_t18r_pg500_timing
```

Closed-loop H50 dry-run after checkpoints exist:

```bash
cd ~/thesis/IsaacLab
conda activate isaaclab

python evaluators/run_t17_multijoint_closed_loop_eval.py \
  --dry_run \
  --policies a2_history a5 teacher_reference \
  --alpha_values 0.25 0.5 1.0 \
  --protocol realistic_random \
  --velocity_mode fixed_vx_1p0 \
  --history_len 50 \
  --a2_history_checkpoint papers/conference/results/t18r_pg500_a2_history_h50_distill_seed0/a2_history_h50_multijoint.pt \
  --a5_checkpoint papers/conference/results/t18r_pg500_a5_h50_residual_seed0/a5_history_residual_multijoint.pt \
  --output_dir papers/conference/results/t18r_pg500_closed_loop_eval_50hz_h50_a2h_a5_alpha_sweep_seed0 \
  --num_envs 16 \
  --num_steps 200 \
  --seed 0 \
  --device cuda \
  --t18r_pg500_timing \
  --require_t18r_pg500_timing
```

## 4. Train 500 Hz Physics / 50 Hz Control Teacher

This uses the previous successful v2b curriculum shape with a 2x iteration budget:

- S0: 4000 iterations
- S1: 6000 iterations
- S2: 10000 iterations
- total: 20000 iterations

```bash
cd ~/thesis/IsaacLab
conda activate isaaclab
source _isaac_sim/setup_conda_env.sh

PYTHONUNBUFFERED=1 TERM=xterm bash trainers/run_t18r_pg500_train_a1f_multijoint_teacher_curriculum.sh \
  --execute_curriculum \
  --num_envs 4096 \
  --device cuda \
  --seed 0
```

Fallback if VRAM is tight:

```bash
PYTHONUNBUFFERED=1 TERM=xterm bash trainers/run_t18r_pg500_train_a1f_multijoint_teacher_curriculum.sh \
  --execute_curriculum \
  --num_envs 2048 \
  --device cuda \
  --seed 0
```

Expected log root:

```text
logs/rsl_rl/t18r_pg500_teacher_p2_multijoint_velocity_50hz_500hzphys__rlm1_stripped__p2_multi_joint_random/
```

## 5. Teacher Evaluation Command

Replace `<T18R_PG500_TEACHER_CHECKPOINT>` with the selected final S2 `model_*.pt`.

```bash
cd ~/thesis/IsaacLab
conda activate isaaclab
source _isaac_sim/setup_conda_env.sh

PYTHONUNBUFFERED=1 TERM=xterm python evaluators/run_t13c_multijoint_teacher_eval.py \
  --execute_eval \
  --headless \
  --checkpoint <T18R_PG500_TEACHER_CHECKPOINT> \
  --output_root papers/conference/results/t18r_pg500_teacher_eval_50hz_500hzphys_seed0 \
  --protocol default \
  --velocity_mode fixed_vx_1p0 \
  --num_envs 512 \
  --num_steps 1000 \
  --seed 0 \
  --device cuda \
  --t18r_pg500_timing \
  --require_t18r_pg500_timing
```

## 6. Teacher Checkpoint Selection Instruction

Select the best final S2 checkpoint from:

```text
logs/rsl_rl/t18r_pg500_teacher_p2_multijoint_velocity_50hz_500hzphys__rlm1_stripped__p2_multi_joint_random/<timestamp>_a1f_multijoint_velocity_p2_t18r_pg500_curriculum_s2_realistic_transition_random_p2__seed0/model_*.pt
```

Do not use previous v2b 60 Hz/H16-era teacher checkpoints for T18-R-PG500 dataset collection.

## 7. Collect Teacher Datasets

Realistic:

```bash
cd ~/thesis/IsaacLab
conda activate isaaclab
source _isaac_sim/setup_conda_env.sh

PYTHONUNBUFFERED=1 TERM=xterm python evaluators/collect_t14_multijoint_teacher_dataset.py \
  --execute_collect \
  --headless \
  --teacher_checkpoint <T18R_PG500_TEACHER_CHECKPOINT> \
  --allow_teacher_checkpoint_override \
  --protocol realistic_random \
  --velocity_mode command_random \
  --num_envs 128 \
  --num_steps 1000 \
  --seed 0 \
  --device cuda \
  --t18r_pg500_timing \
  --require_t18r_pg500_timing \
  --output_dir papers/conference/datasets/t18r_pg500_50hz_h50_teacher_dataset_realistic_seed0 \
  --dataset_tag t18r_pg500_50hz_h50_teacher_dataset_realistic_seed0
```

Late:

```bash
PYTHONUNBUFFERED=1 TERM=xterm python evaluators/collect_t14_multijoint_teacher_dataset.py \
  --execute_collect \
  --headless \
  --teacher_checkpoint <T18R_PG500_TEACHER_CHECKPOINT> \
  --allow_teacher_checkpoint_override \
  --protocol late_random \
  --velocity_mode command_random \
  --num_envs 128 \
  --num_steps 1000 \
  --seed 0 \
  --device cuda \
  --t18r_pg500_timing \
  --require_t18r_pg500_timing \
  --output_dir papers/conference/datasets/t18r_pg500_50hz_h50_teacher_dataset_late_seed0 \
  --dataset_tag t18r_pg500_50hz_h50_teacher_dataset_late_seed0
```

## 8. Dataset Audit

```bash
cd ~/thesis/IsaacLab
conda activate isaaclab

python evaluators/audit_t14_multijoint_teacher_dataset.py \
  --write_outputs \
  --realistic_dataset papers/conference/datasets/t18r_pg500_50hz_h50_teacher_dataset_realistic_seed0/dataset.npz \
  --late_dataset papers/conference/datasets/t18r_pg500_50hz_h50_teacher_dataset_late_seed0/dataset.npz \
  --output_root papers/conference/results/t18r_pg500_50hz_h50_dataset_audit
```

## 9. A2-History H50 Training

```bash
cd ~/thesis/IsaacLab
conda activate isaaclab

python trainers/train_t15_multijoint_a2_student_distill.py \
  --execute_train \
  --mode history \
  --history_len 50 \
  --require_history_len 50 \
  --require_dataset_control_frequency_hz 50 \
  --require_dataset_physics_frequency_hz 500 \
  --dataset_paths \
    papers/conference/datasets/t18r_pg500_50hz_h50_teacher_dataset_realistic_seed0/dataset.npz \
    papers/conference/datasets/t18r_pg500_50hz_h50_teacher_dataset_late_seed0/dataset.npz \
  --output_dir papers/conference/results/t18r_pg500_a2_history_h50_distill_seed0 \
  --epochs 100 \
  --batch_size 4096 \
  --lr 0.001 \
  --val_fraction 0.1 \
  --seed 0 \
  --device cuda
```

## 10. A5-H50 Residual Training

```bash
cd ~/thesis/IsaacLab
conda activate isaaclab

python trainers/train_t16_multijoint_a5_history_residual.py \
  --execute_train \
  --history_len 50 \
  --require_history_len 50 \
  --require_dataset_control_frequency_hz 50 \
  --require_dataset_physics_frequency_hz 500 \
  --base_checkpoint papers/conference/results/t18r_pg500_a2_history_h50_distill_seed0/a2_history_h50_multijoint.pt \
  --dataset_paths \
    papers/conference/datasets/t18r_pg500_50hz_h50_teacher_dataset_realistic_seed0/dataset.npz \
    papers/conference/datasets/t18r_pg500_50hz_h50_teacher_dataset_late_seed0/dataset.npz \
  --output_dir papers/conference/results/t18r_pg500_a5_h50_residual_seed0 \
  --epochs 100 \
  --batch_size 4096 \
  --lr 0.001 \
  --val_fraction 0.1 \
  --seed 0 \
  --device cuda
```

## 11. Closed-Loop Pilot Evaluation

```bash
cd ~/thesis/IsaacLab
conda activate isaaclab
source _isaac_sim/setup_conda_env.sh

PYTHONUNBUFFERED=1 TERM=xterm python evaluators/run_t17_multijoint_closed_loop_eval.py \
  --execute_eval \
  --headless \
  --policies a2_history a5 teacher_reference \
  --alpha_values 0.25 0.5 1.0 \
  --protocol default \
  --velocity_mode fixed_vx_1p0 \
  --history_len 50 \
  --a2_history_checkpoint papers/conference/results/t18r_pg500_a2_history_h50_distill_seed0/a2_history_h50_multijoint.pt \
  --a5_checkpoint papers/conference/results/t18r_pg500_a5_h50_residual_seed0/a5_history_residual_multijoint.pt \
  --teacher_checkpoint <T18R_PG500_TEACHER_CHECKPOINT> \
  --output_dir papers/conference/results/t18r_pg500_closed_loop_eval_50hz_h50_a2h_a5_alpha_sweep_seed0 \
  --num_envs 512 \
  --num_steps 1000 \
  --seed 0 \
  --device cuda \
  --progress_every 100 \
  --t18r_pg500_timing \
  --require_t18r_pg500_timing
```

## 12. Closed-Loop Final Multi-Seed Evaluation

Run the same command for seeds 0, 1, and 2. Use separate roots to avoid overwrites.

```bash
for SEED in 0 1 2; do
  PYTHONUNBUFFERED=1 TERM=xterm python evaluators/run_t17_multijoint_closed_loop_eval.py \
    --execute_eval \
    --headless \
    --policies a2_history a5 teacher_reference \
    --alpha_values 0.25 0.5 1.0 \
    --protocol default \
    --velocity_mode fixed_vx_1p0 \
    --history_len 50 \
    --a2_history_checkpoint papers/conference/results/t18r_pg500_a2_history_h50_distill_seed0/a2_history_h50_multijoint.pt \
    --a5_checkpoint papers/conference/results/t18r_pg500_a5_h50_residual_seed0/a5_history_residual_multijoint.pt \
    --teacher_checkpoint <T18R_PG500_TEACHER_CHECKPOINT> \
    --output_dir papers/conference/results/t18r_pg500_closed_loop_eval_50hz_h50_a2h_a5_alpha_sweep_seed${SEED} \
    --num_envs 512 \
    --num_steps 1000 \
    --seed ${SEED} \
    --device cuda \
    --progress_every 100 \
    --t18r_pg500_timing \
    --require_t18r_pg500_timing
done
```

## 13. Figure/Table Packaging

Figure support for seed 0 pilot:

```bash
python evaluators/plot_t17_fixed_vx_fault_response.py \
  --input_root papers/conference/results/t18r_pg500_closed_loop_eval_50hz_h50_a2h_a5_alpha_sweep_seed0 \
  --teacher_root papers/conference/results/t18r_pg500_closed_loop_eval_50hz_h50_a2h_a5_alpha_sweep_seed0 \
  --a2_single_step_root papers/conference/results/t18r_pg500_closed_loop_eval_50hz_h50_a2h_a5_alpha_sweep_seed0 \
  --output_dir papers/conference/results/t18r_pg500_closed_loop_report_package_50hz_h50 \
  --phase_label T18-R-PG500 \
  --history_label H50 \
  --exclude_a2_single_step \
  --figure_basename fig_fixed_vx_pg500_50hz_h50_history_residual_ablation \
  --tracking_table_basename t18r_pg500_seed0_plotted_runs \
  --notes_basename t18r_pg500_figure_notes \
  --summary_basename t18r_pg500_figure_summary \
  --max_step 900
```

Multi-seed mean/std table and protocol metadata:

```bash
python evaluators/summarize_t18r_pg500_multiseed.py \
  --input_roots \
    papers/conference/results/t18r_pg500_closed_loop_eval_50hz_h50_a2h_a5_alpha_sweep_seed0 \
    papers/conference/results/t18r_pg500_closed_loop_eval_50hz_h50_a2h_a5_alpha_sweep_seed1 \
    papers/conference/results/t18r_pg500_closed_loop_eval_50hz_h50_a2h_a5_alpha_sweep_seed2 \
  --output_dir papers/conference/results/t18r_pg500_closed_loop_report_package_50hz_h50
```

## 14. Expected Output File Checks

```bash
test -f papers/conference/results/t18r_pg500_closed_loop_report_package_50hz_h50/fig_fixed_vx_pg500_50hz_h50_history_residual_ablation.png
test -f papers/conference/results/t18r_pg500_closed_loop_report_package_50hz_h50/fig_fixed_vx_pg500_50hz_h50_history_residual_ablation.pdf
test -f papers/conference/results/t18r_pg500_closed_loop_report_package_50hz_h50/table_t18r_pg500_50hz_h50_tracking_error.md
test -f papers/conference/results/t18r_pg500_closed_loop_report_package_50hz_h50/table_t18r_pg500_50hz_h50_tracking_error.csv
test -f papers/conference/results/t18r_pg500_closed_loop_report_package_50hz_h50/timing_metadata.json
test -f papers/conference/results/t18r_pg500_closed_loop_report_package_50hz_h50/paper_grade_protocol_readme.md
```
