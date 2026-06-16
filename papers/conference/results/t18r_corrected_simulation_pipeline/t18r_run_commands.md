# T18-R Manual Command Checklist

Run these manually from the repository root. They are intentionally not executed by Codex.

## 0. Optional Syntax Checks

```bash
cd ~/thesis/IsaacLab
python -m py_compile evaluators/t18r_control_timing.py
python -m py_compile evaluators/collect_t14_multijoint_teacher_dataset.py
python -m py_compile evaluators/run_t13c_multijoint_teacher_eval.py
python -m py_compile evaluators/run_t17_multijoint_closed_loop_eval.py
python -m py_compile evaluators/plot_t17_fixed_vx_fault_response.py
python -m py_compile trainers/train_t15_multijoint_a2_student_distill.py
python -m py_compile trainers/train_t16_multijoint_a5_history_residual.py
```

## 1. Collect T18-R Teacher Datasets

Collect the two default protocols in separate processes. This avoids the earlier second-environment creation hang seen during combined T14 collection.

```bash
cd ~/thesis/IsaacLab
conda activate isaaclab
source _isaac_sim/setup_conda_env.sh

PYTHONUNBUFFERED=1 TERM=xterm python evaluators/collect_t14_multijoint_teacher_dataset.py \
  --execute_collect \
  --headless \
  --protocol realistic_random \
  --velocity_mode command_random \
  --num_envs 128 \
  --num_steps 1000 \
  --seed 0 \
  --device cuda \
  --control_frequency_hz 50 \
  --sim_dt 0.01 \
  --decimation 2 \
  --require_control_frequency_hz 50 \
  --output_dir papers/conference/datasets/t18r_50hz_h50_teacher_dataset_realistic_seed0 \
  --dataset_tag t18r_50hz_h50_teacher_dataset_realistic_seed0
```

```bash
cd ~/thesis/IsaacLab
conda activate isaaclab
source _isaac_sim/setup_conda_env.sh

PYTHONUNBUFFERED=1 TERM=xterm python evaluators/collect_t14_multijoint_teacher_dataset.py \
  --execute_collect \
  --headless \
  --protocol late_random \
  --velocity_mode command_random \
  --num_envs 128 \
  --num_steps 1000 \
  --seed 0 \
  --device cuda \
  --control_frequency_hz 50 \
  --sim_dt 0.01 \
  --decimation 2 \
  --require_control_frequency_hz 50 \
  --output_dir papers/conference/datasets/t18r_50hz_h50_teacher_dataset_late_seed0 \
  --dataset_tag t18r_50hz_h50_teacher_dataset_late_seed0
```

## 2. Audit T18-R Datasets

```bash
cd ~/thesis/IsaacLab
conda activate isaaclab

python evaluators/audit_t14_multijoint_teacher_dataset.py \
  --write_outputs \
  --realistic_dataset papers/conference/datasets/t18r_50hz_h50_teacher_dataset_realistic_seed0/dataset.npz \
  --late_dataset papers/conference/datasets/t18r_50hz_h50_teacher_dataset_late_seed0/dataset.npz \
  --output_root papers/conference/results/t18r_50hz_h50_dataset_audit
```

## 3. Train A2-History H50

```bash
cd ~/thesis/IsaacLab
conda activate isaaclab

python trainers/train_t15_multijoint_a2_student_distill.py \
  --execute_train \
  --mode history \
  --history_len 50 \
  --dataset_paths \
    papers/conference/datasets/t18r_50hz_h50_teacher_dataset_realistic_seed0/dataset.npz \
    papers/conference/datasets/t18r_50hz_h50_teacher_dataset_late_seed0/dataset.npz \
  --output_dir papers/conference/results/t18r_a2_history_h50_distill_seed0 \
  --epochs 100 \
  --batch_size 4096 \
  --lr 0.001 \
  --val_fraction 0.1 \
  --seed 0 \
  --device cuda
```

## 4. Train A5-H50 Residual

```bash
cd ~/thesis/IsaacLab
conda activate isaaclab

python trainers/train_t16_multijoint_a5_history_residual.py \
  --execute_train \
  --history_len 50 \
  --base_checkpoint papers/conference/results/t18r_a2_history_h50_distill_seed0/a2_history_h50_multijoint.pt \
  --dataset_paths \
    papers/conference/datasets/t18r_50hz_h50_teacher_dataset_realistic_seed0/dataset.npz \
    papers/conference/datasets/t18r_50hz_h50_teacher_dataset_late_seed0/dataset.npz \
  --output_dir papers/conference/results/t18r_a5_h50_residual_seed0 \
  --epochs 100 \
  --batch_size 4096 \
  --lr 0.001 \
  --val_fraction 0.1 \
  --seed 0 \
  --device cuda
```

## 5. Run Closed-Loop Fixed-vx Evaluation

This is the corrected residual-ablation run. A2 single-step is intentionally excluded.

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
  --a2_history_checkpoint papers/conference/results/t18r_a2_history_h50_distill_seed0/a2_history_h50_multijoint.pt \
  --a5_checkpoint papers/conference/results/t18r_a5_h50_residual_seed0/a5_history_residual_multijoint.pt \
  --output_dir papers/conference/results/t18r_closed_loop_eval_50hz_h50_a2h_a5_alpha_sweep_seed0 \
  --num_envs 512 \
  --num_steps 1000 \
  --seed 0 \
  --device cuda \
  --progress_every 100 \
  --control_frequency_hz 50 \
  --sim_dt 0.01 \
  --decimation 2 \
  --require_control_frequency_hz 50
```

## 6. Package Figure and Table Artifacts

```bash
cd ~/thesis/IsaacLab
conda activate isaaclab

python evaluators/plot_t17_fixed_vx_fault_response.py \
  --input_root papers/conference/results/t18r_closed_loop_eval_50hz_h50_a2h_a5_alpha_sweep_seed0 \
  --teacher_root papers/conference/results/t18r_closed_loop_eval_50hz_h50_a2h_a5_alpha_sweep_seed0 \
  --a2_single_step_root papers/conference/results/t18r_closed_loop_eval_50hz_h50_a2h_a5_alpha_sweep_seed0 \
  --output_dir papers/conference/results/t18r_closed_loop_report_package_50hz_h50 \
  --phase_label T18-R \
  --history_label H50 \
  --exclude_a2_single_step \
  --figure_basename fig_fixed_vx_50hz_h50_history_residual_ablation \
  --tracking_table_basename table_t18r_50hz_h50_tracking_error \
  --notes_basename t18r_50hz_h50_figure_notes \
  --summary_basename t18r_50hz_h50_figure_summary \
  --title "T18-R fixed-vx 50 Hz / H50 history residual ablation"
```

## Optional Dry-Run Checks

These should not launch Isaac:

```bash
python evaluators/collect_t14_multijoint_teacher_dataset.py \
  --dry_run \
  --protocol realistic_random \
  --velocity_mode command_random \
  --control_frequency_hz 50 \
  --sim_dt 0.01 \
  --decimation 2 \
  --require_control_frequency_hz 50
```

The collector does not consume `--history_len`; use the closed-loop dry-run below for H50 policy guards:

```bash
python evaluators/run_t17_multijoint_closed_loop_eval.py \
  --dry_run \
  --policies a2_history a5 teacher_reference \
  --alpha_values 0.25 0.5 1.0 \
  --protocol realistic_random \
  --velocity_mode fixed_vx_1p0 \
  --history_len 50 \
  --a2_history_checkpoint papers/conference/results/t18r_a2_history_h50_distill_seed0/a2_history_h50_multijoint.pt \
  --a5_checkpoint papers/conference/results/t18r_a5_h50_residual_seed0/a5_history_residual_multijoint.pt \
  --output_dir papers/conference/results/t18r_closed_loop_eval_50hz_h50_a2h_a5_alpha_sweep_seed0 \
  --num_envs 16 \
  --num_steps 200 \
  --seed 0 \
  --device cuda \
  --control_frequency_hz 50 \
  --sim_dt 0.01 \
  --decimation 2 \
  --require_control_frequency_hz 50
```
