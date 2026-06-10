# T10 Isaac Ant Velocity Tracking Bootstrap Prompt

Do not return another plan. Apply the next minimal implementation scaffold for the Isaac Ant velocity-tracking task.

Context:
- T10-IL-01 inspection found that `Isaac-Ant-v0` is manager-based classic Ant, registered in `source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/ant/__init__.py`.
- Current classic Ant has no command manager and no writable velocity command.
- Current rewards are forward/progress style, not commanded-velocity tracking.
- Isaac Lab already provides reusable velocity command/reward pieces in `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/velocity_env_cfg.py` and `source/isaaclab/isaaclab/envs/mdp/`.
- P2 wrapper compatibility depends on manager-based `action_manager`, one 8-D Ant joint effort action term, and `robot.write_joint_state_to_sim`.

Strict scope:
- Do not train.
- Do not freeze checkpoints.
- Do not edit checkpoint pointers.
- Do not update paper-grade result manifests.
- Do not modify Isaac Lab core.
- Do not modify RSL-RL core.
- Do not add health token, UQ, CBF, P3, or P4 main scope.
- Do not make paper-grade claims.

Goal:
Create the smallest repo-owned velocity-tracking Ant task scaffold and dry-run/preflight support.

Create or modify:
- `source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/ant/ant_velocity_env_cfg.py`
- `source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/ant/__init__.py`
- `evaluators/preflight_t10_ant_velocity_task.py`
- `configs/train/healthy_baseline_velocity.yaml`
- `configs/train/teacher_velocity.yaml` if needed
- `papers/conference/results/t10_ant_velocity_tracking_task_scaffold.md`

Task IDs:
- `Isaac-Ant-Velocity-Flat-v0`
- `Isaac-Ant-Teacher-Velocity-Flat-v0`

Design requirements:
- Use manager-based Ant, not direct Ant.
- Preserve Ant joint effort action with 8 actions.
- Add `base_velocity` command using `mdp.UniformVelocityCommandCfg`.
- Include velocity command in policy observations with `mdp.generated_commands`.
- Use velocity tracking rewards such as `mdp.track_lin_vel_xy_exp` and `mdp.track_ang_vel_z_exp`.
- Keep `vy_cmd=0.0` and `yaw_rate_cmd=0.0` for controlled eval configs.
- Support fixed command ranges for eval, e.g. `lin_vel_x=(1.0, 1.0)`.
- Teacher task may include privileged `true_fault_state`, but deployment-facing runtime rows must not.

P2 compatibility requirements:
- Preflight must verify `env.unwrapped.action_manager` exists.
- Verify `total_action_dim == 8`.
- Verify `front_left_foot` maps to exactly one action/joint index.
- Verify simulation-level joint-state override prerequisites are present.
- Do not execute training.

Validation:
- `python -m py_compile` on new/modified Python files.
- Run only dry-run/preflight commands that do not train.
- Do not freeze or promote checkpoints.

Return:
### 1. Files created or modified
### 2. Velocity task scaffold
### 3. Command/observation/reward wiring
### 4. P2 compatibility preflight
### 5. Guardrails preserved
### 6. Validation commands executed
### 7. Suggested next training-free preflight command
