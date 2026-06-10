# Isaac Ant Velocity Tracking Plan

Scope: T10-IL-01 inspection and design documentation only. No training, no Isaac Sim launch, no checkpoint freeze, no pointer edit, no paper-grade result update, and no task implementation were performed.

## Motivation

The current RLM1 stripped Ant experiments use the classic `Isaac-Ant-v0` task. That task rewards forward progress toward a far target rather than tracking a shared commanded forward velocity. This makes advisor-facing P2 velocity plots hard to interpret because policies can run at very different nominal speeds before the joint lock happens. For FTC-style controlled evaluation, A0, A1-F, A2, A5, A6, and A7 need a common command such as `vx_cmd=1.0`, the same seed, the same P2 onset, and the same locked joint.

## Current Ant Diagnosis

| question | finding |
| --- | --- |
| Current task ID used by repo | `Isaac-Ant-v0` |
| Classification | Manager-based, not direct |
| Registration path | `source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/ant/__init__.py` |
| Env entry point | `isaaclab.envs:ManagerBasedRLEnv` |
| Env config | `source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/ant/ant_env_cfg.py:AntEnvCfg` |
| Teacher config | `AntTeacherEnvCfg` in the same file |
| Student config | `AntStudentEnvCfg` in the same file |
| Direct Ant exists | Yes, `Isaac-Ant-Direct-v0`, but it is not the repo's current A0/A1/A2 task path |
| Command manager | Not present in the current classic Ant config |
| Writable velocity command | Not exposed by the current classic Ant config |

The current classic Ant policy observations are:

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

The current teacher observation variant adds privileged `true_fault_state` and names the base velocity term as `base_velocity`. The student variant exposes `policy` plus privileged `teacher_policy` for distillation.

## Why Current Task Is Insufficient

Current forward locomotion rewards include `progress`, `alive`, `upright`, `move_to_target`, `action_l2`, `energy`, and `joint_pos_limits`. The forward component is based on progress toward `target_pos=(1000.0, 0.0, 0.0)`, not tracking a command. A policy can improve by moving faster, and different policies can settle into different nominal velocity regimes. A velocity plot under P2 then mixes nominal speed differences with fault recovery behavior.

## Proposed Task

Create a repo-owned task variant:

```text
Isaac-Ant-Velocity-Flat-v0
```

Optional privileged teacher variant:

```text
Isaac-Ant-Teacher-Velocity-Flat-v0
```

Desired command:

```text
base_velocity command
vx_cmd in {0.5, 1.0, 1.5}
vy_cmd = 0.0
yaw_rate_cmd = 0.0
```

Desired policy observation:

- current Ant proprioception
- generated velocity command from `mdp.generated_commands(command_name="base_velocity")`
- no `true_fault_state` for deployment-facing A0/A2/A5/A6/A7 runtime policies

Desired privileged teacher observation:

- same velocity-aware policy observation
- optional privileged `true_fault_state` only for A1-F teacher/reference training
- A1-F remains reference-only, not deployment-facing

Desired reward:

- `track_lin_vel_xy_exp` for command tracking
- `track_ang_vel_z_exp` with yaw command fixed at zero
- alive/upright terms adapted from classic Ant
- action/energy penalty
- joint-limit penalty
- termination on low torso/root height

## Reuse From Isaac Lab

Existing reusable velocity components were found in:

```text
source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/velocity_env_cfg.py
source/isaaclab/isaaclab/envs/mdp/commands/commands_cfg.py
source/isaaclab/isaaclab/envs/mdp/commands/velocity_command.py
source/isaaclab/isaaclab/envs/mdp/observations.py
source/isaaclab/isaaclab/envs/mdp/rewards.py
```

Useful components:

- `mdp.UniformVelocityCommandCfg`
- `mdp.generated_commands`
- `mdp.track_lin_vel_xy_exp`
- `mdp.track_ang_vel_z_exp`
- `mdp.base_lin_vel`
- `mdp.base_ang_vel`

## Safest Implementation Path

The safest path is to add a repo-owned manager-based Ant velocity config under `source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/ant/`, then register new task IDs in the Ant task package. This avoids Isaac Lab core and RSL-RL core changes while reusing the existing Ant asset, action structure, observation terms, and local RSL-RL runner configs.

Do not switch to the direct Ant task for the conference pipeline. It exists, but the current repo training, teacher, student, residual, and P2 wrapper assumptions are already built around manager-based Ant with `action_manager`.

## P2 Compatibility

The existing P2 wrapper attaches through:

```text
trainers/p2_joint_lock_training_wrapper.py
```

It requires:

- `env.unwrapped.action_manager`
- exactly one joint effort action term
- `total_action_dim == 8`
- target joint `front_left_foot`
- readable joint names and joint IDs
- `robot.write_joint_state_to_sim`

The proposed velocity task should keep the Ant `ActionsCfg` as `joint_effort = mdp.JointEffortActionCfg(...)` with the same 8 Ant joints. If that is preserved, the P2 simulation-level q/qd override should attach the same way.

## Risks

- Isaac Lab velocity locomotion examples often use position-control quadrupeds, rough terrain, contact sensors, and height scanners. Copying that whole stack into Ant would add unnecessary complexity.
- Direct Ant has no manager-based `action_manager`, so it is a poor fit for the current P2 wrapper.
- A fixed `vx_cmd` may need a custom or frozen command range wrapper if `UniformVelocityCommandCfg` resampling behavior cannot be configured exactly.
- Teacher observations must not leak `true_fault_state` into deployment-facing A2/A5/A6/A7 runtime policies.

## Next Minimal Steps

1. Create `source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/ant/ant_velocity_env_cfg.py`.
2. Define `AntVelocityEnvCfg` by adapting `AntEnvCfg` and adding `CommandsCfg`.
3. Add velocity command observation using `mdp.generated_commands`.
4. Replace progress reward with velocity tracking rewards.
5. Add optional `AntTeacherVelocityEnvCfg`.
6. Register `Isaac-Ant-Velocity-Flat-v0` and `Isaac-Ant-Teacher-Velocity-Flat-v0`.
7. Add dry-run/preflight only before any training.

## Guardrails

- No paper-grade claim is made by this plan.
- A1-F remains privileged and reference-only.
- A5/A6/A7 are deployment-facing only if privileged runtime information is removed.
- Health token OFF, UQ inactive, CBF inactive, and no P3 expansion remain fixed.
