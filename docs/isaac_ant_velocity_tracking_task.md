# Isaac Ant Velocity Tracking Task Design

Scope: inspection and design note for T10-IL-01. This document does not implement, register, train, evaluate, freeze, or promote any checkpoint.

## Files Inspected

Current Ant task and agents:

- `source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/ant/__init__.py`
- `source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/ant/ant_env_cfg.py`
- `source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/ant/agents/rsl_rl_ppo_cfg.py`
- `source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/ant/agents/rsl_rl_distillation_cfg.py`
- `source/isaaclab_tasks/isaaclab_tasks/direct/ant/__init__.py`
- `source/isaaclab_tasks/isaaclab_tasks/direct/ant/ant_env.py`
- `source/isaaclab_tasks/isaaclab_tasks/direct/locomotion/locomotion_env.py`

Velocity command/reward components:

- `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/velocity_env_cfg.py`
- `source/isaaclab/isaaclab/envs/mdp/commands/commands_cfg.py`
- `source/isaaclab/isaaclab/envs/mdp/commands/velocity_command.py`
- `source/isaaclab/isaaclab/envs/mdp/observations.py`
- `source/isaaclab/isaaclab/envs/mdp/rewards.py`

Repo wrappers/configs:

- `trainers/rsl_rl_train.py`
- `trainers/p2_joint_lock_training_wrapper.py`
- `configs/train/healthy_baseline.yaml`
- `configs/train/teacher.yaml`
- `configs/train/teacher_p2_canonical.yaml`
- `configs/fault/joint_lock/p2_locked_joint.yaml`

Related notes:

- `papers/conference/results/t09_p2_single_joint_lock_scope.md`
- `papers/conference/demo/demo_quantitative_logging.md`

## Current Task Classification

`Isaac-Ant-v0` is manager-based. It is registered in:

```text
source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/ant/__init__.py
```

The registration uses:

```text
entry_point="isaaclab.envs:ManagerBasedRLEnv"
env_cfg_entry_point=...ant_env_cfg:AntEnvCfg
rsl_rl_cfg_entry_point=...rsl_rl_ppo_cfg:AntPPORunnerCfg
```

The teacher and student task IDs are also manager-based:

```text
Isaac-Ant-Teacher-v0
Isaac-Ant-Student-v0
```

`Isaac-Ant-Direct-v0` also exists, but it is separate and registered under:

```text
source/isaaclab_tasks/isaaclab_tasks/direct/ant/__init__.py
```

The direct task uses `LocomotionEnv`, progress rewards, and direct env APIs. It is not the current RLM1 stripped conference task path.

## Current Observations

`AntEnvCfg` policy observation terms:

- `base_height`
- `base_lin_vel`
- `base_ang_vel`
- `base_yaw_roll`
- `base_angle_to_target`
- `base_up_proj`
- `base_heading_proj`
- `joint_pos_norm`
- `joint_vel_rel`
- `feet_body_forces`
- `actions`

`AntTeacherEnvCfg` replaces policy observations with a teacher set containing:

- `base_height`
- `true_fault_state`
- `base_velocity`
- `base_ang_vel`
- `base_yaw_roll`
- `base_angle_to_target`
- `base_up_proj`
- `base_heading_proj`
- `joint_pos_norm`
- `joint_vel_rel`
- `contacts`
- `actions`

`AntStudentEnvCfg` exposes:

- `policy`
- `teacher_policy`

The student-safe `policy` group does not include `true_fault_state`.

## Command Finding

The current classic Ant config has no `commands: CommandsCfg` field and no command manager term. It therefore does not expose `env.command_manager.get_command("base_velocity")` and does not expose a real writable `vx_cmd` for controlled evaluation.

The current task can log root/base velocity, but that is observed state, not a target command.

## Reward Finding

Current classic Ant rewards are forward/progress style:

- `progress` via `mdp.progress_reward` toward `target_pos=(1000.0, 0.0, 0.0)`
- `alive`
- `upright`
- `move_to_target`
- `action_l2`
- `energy`
- `joint_pos_limits`

This supports "go forward" behavior, not fixed or sampled command tracking.

## Existing Velocity Components

Isaac Lab already provides command and reward components that can be reused without touching core:

```text
mdp.UniformVelocityCommandCfg
mdp.generated_commands
mdp.track_lin_vel_xy_exp
mdp.track_ang_vel_z_exp
mdp.base_lin_vel
mdp.base_ang_vel
```

The reference pattern is in:

```text
source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/velocity_env_cfg.py
```

That file defines `CommandsCfg.base_velocity`, includes `velocity_commands` in observations, and tracks linear/angular velocity commands in rewards.

## Recommended Design

Create a new repo-owned manager-based Ant velocity config:

```text
source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/ant/ant_velocity_env_cfg.py
```

Register:

```text
Isaac-Ant-Velocity-Flat-v0
Isaac-Ant-Teacher-Velocity-Flat-v0
```

Recommended config approach:

1. Subclass or copy from `AntEnvCfg`, not from direct Ant.
2. Preserve Ant asset, terrain plane, termination, and 8-D joint effort action.
3. Add `commands: CommandsCfg`.
4. Add `velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})`.
5. Replace `progress` and `move_to_target` with velocity tracking rewards.
6. Keep action/energy/joint-limit penalties in a conservative form.
7. Define teacher variant with privileged terms only for A1-F.

Command design:

```text
base_velocity:
  lin_vel_x: sampled or fixed from {0.5, 1.0, 1.5}
  lin_vel_y: 0.0
  ang_vel_z: 0.0
  heading: 0.0 if heading_command is enabled
```

For evaluation, use fixed command ranges such as:

```text
lin_vel_x=(1.0, 1.0)
lin_vel_y=(0.0, 0.0)
ang_vel_z=(0.0, 0.0)
heading=(0.0, 0.0)
```

## P2 Wrapper Compatibility

The P2 wrapper uses manager-based action introspection:

```text
env.unwrapped.action_manager
```

It expects one action term, eight Ant actions, and resolved joint names. The new velocity task should preserve:

```text
joint_effort = mdp.JointEffortActionCfg(asset_name="robot", joint_names=[".*"], scale=7.5)
```

If that action term remains unchanged, the existing simulation-level P2 lock should continue to resolve:

```text
front_left_foot -> one Ant action/joint index
```

The intended semantics remain:

```text
simulation_joint_state_override_lock
q[joint] = q_lock
qd[joint] = 0.0
fallback disabled
```

## Risks And Fallbacks

| risk | mitigation |
| --- | --- |
| Full velocity locomotion examples include rough terrain and sensors. | Use only command, observation, and reward functions; do not copy the whole rough-terrain stack. |
| Direct Ant lacks the manager-based action manager expected by P2 wrapper. | Keep the new task manager-based. |
| Fixed command values may be resampled by `UniformVelocityCommandCfg`. | Use fixed min=max command ranges for controlled eval; optionally add a tiny repo-owned command-freeze wrapper later. |
| Teacher privilege leaks into deployment rows. | Keep A1-F reference-only and validate A2/A5/A6/A7 runtime obs groups exclude `teacher_policy` and `true_fault_state`. |
| Reward scale changes alter checkpoint comparability. | Treat velocity-task checkpoints as a new controlled task family; do not compare directly to old progress-task checkpoints as paper-grade evidence. |

## Minimal Next Implementation Task

The next Codex task should implement a dry-run scaffold only:

1. Add `ant_velocity_env_cfg.py`.
2. Register `Isaac-Ant-Velocity-Flat-v0` and `Isaac-Ant-Teacher-Velocity-Flat-v0`.
3. Add a preflight script that imports the registry, resolves env cfg, checks `commands.base_velocity`, checks policy obs contains `velocity_commands`, and checks P2 action mapping still resolves.
4. Do not train.
5. Do not run full evaluation.

## Guardrails

- No paper-grade claims yet.
- A1-F is privileged and reference-only.
- A5/A6/A7 are deployment-facing only if privileged runtime information is removed.
- RLM1 stripped terminology remains fixed.
- Health token OFF, UQ inactive, CBF inactive, and no P3 expansion remain fixed.
