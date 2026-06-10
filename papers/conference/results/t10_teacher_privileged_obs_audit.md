# T10 Teacher Privileged Observation Audit

Scope: read-only audit for the current P2 velocity teacher. No training, simulation, config edits, task-code edits, checkpoint edits, or P2 wrapper changes were performed.

## Files Inspected

- `source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/ant/ant_velocity_env_cfg.py`
- `source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/ant/ant_env_cfg.py`
- `source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/ant/__init__.py`
- `evaluators/preflight_t10_ant_velocity_task.py`
- `configs/train/teacher_velocity.yaml`
- `configs/train/healthy_baseline_velocity.yaml`

## Task Registration

| task | env cfg | role |
| --- | --- | --- |
| `Isaac-Ant-Velocity-Flat-v0` | `AntVelocityFlatEnvCfg` | A0/student-safe velocity task |
| `Isaac-Ant-Teacher-Velocity-Flat-v0` | `AntTeacherVelocityFlatEnvCfg` | A1-F privileged/reference teacher task |

## A0 Velocity Observation Terms

Task: `Isaac-Ant-Velocity-Flat-v0`

Observation group: `policy`

| term | source/function | estimated dim | deployment-safe |
| --- | --- | ---: | --- |
| `base_height` | `mdp.base_pos_z` | 1 | yes |
| `base_lin_vel` | `mdp.base_lin_vel` | 3 | yes |
| `base_ang_vel` | `mdp.base_ang_vel` | 3 | yes |
| `velocity_commands` | `mdp.generated_commands(command_name="base_velocity")` | 3 | yes |
| `base_yaw_roll` | `mdp.base_yaw_roll` | 2 | yes |
| `base_up_proj` | `mdp.base_up_proj` | 1 | yes |
| `joint_pos_norm` | `mdp.joint_pos_limit_normalized` | 8 | yes |
| `joint_vel_rel` | `mdp.joint_vel_rel` | 8 | yes |
| `feet_body_forces` | `mdp.body_incoming_wrench`, 4 feet | 24 | yes in current config |
| `actions` | `mdp.last_action` | 8 | yes |

Observed/runtime total: `61`

Static accounting: `1 + 3 + 3 + 3 + 2 + 1 + 8 + 8 + 24 + 8 = 61`

## A1-F Teacher Velocity Observation Terms

Task: `Isaac-Ant-Teacher-Velocity-Flat-v0`

Observation group: `policy`

| term | source/function | estimated dim | deployment-safe | privileged |
| --- | --- | ---: | --- | --- |
| `base_height` | `mdp.base_pos_z` | 1 | yes | no |
| `true_fault_state` | `true_fault_state` | 1 | no | yes |
| `base_velocity` | `mdp.base_lin_vel` | 3 | yes | no |
| `base_ang_vel` | `mdp.base_ang_vel` | 3 | yes | no |
| `velocity_commands` | `mdp.generated_commands(command_name="base_velocity")` | 3 | yes | no |
| `base_yaw_roll` | `mdp.base_yaw_roll` | 2 | yes | no |
| `base_up_proj` | `mdp.base_up_proj` | 1 | yes | no |
| `joint_pos_norm` | `mdp.joint_pos_limit_normalized` | 8 | yes | no |
| `joint_vel_rel` | `mdp.joint_vel_rel` | 8 | yes | no |
| `contacts` | `mdp.body_incoming_wrench`, 4 feet | 24 | yes in current config | no |
| `actions` | `mdp.last_action` | 8 | yes | no |

Observed/runtime total: `62`

Static accounting: `1 + 1 + 3 + 3 + 3 + 2 + 1 + 8 + 8 + 24 + 8 = 62`

## Contact/Wrench Finding

Contacts are included in the 62-dimensional teacher observation under the term name `contacts`.

However, contacts are not the teacher-only dimensional difference. The A0/student-safe task also includes the same `mdp.body_incoming_wrench` signal over the same four feet under the term name `feet_body_forces`.

Therefore, the A0/A1-F dimension gap is not caused by contacts. It is caused by the 1D `true_fault_state` term.

## Base Velocity Naming Finding

There are two separate uses of the name `base_velocity`:

1. Command term: `commands.base_velocity`
   - Type: `UniformVelocityCommandCfg`
   - Command range: `lin_vel_x=(1.0, 1.0)`, `lin_vel_y=(0.0, 0.0)`, `ang_vel_z=(0.0, 0.0)`
   - Observed by policy through `velocity_commands`

2. Teacher observation term: `observations.policy.base_velocity`
   - Function: `mdp.base_lin_vel`
   - Meaning: actual base linear velocity, equivalent in content to A0's `base_lin_vel` term

The teacher's `base_velocity` observation is not the command itself. The command is exposed through `velocity_commands`.

## Privileged Extra Term

For the current velocity teacher, `true_fault_state` is the only teacher-only privileged extra term relative to the A0/student-safe task.

Current implementation in `ant_env_cfg.py` returns:

```text
torch.zeros(env.num_envs, 1, device=env.device)
```

Under the present code, this is a 1D teacher-only channel. It is structurally privileged because deployment-facing policies must not require true fault labels, even if the current implementation returns zeros by default.

## Deployment-Safe vs Privileged

Deployment-safe in the current config:

- base height
- base linear velocity
- base angular velocity
- velocity command observation
- base yaw/roll
- base up projection
- normalized joint positions
- joint velocities
- foot/body wrench signal already included by A0
- previous action

Privileged:

- `true_fault_state`

Reference-only:

- The full `Isaac-Ant-Teacher-Velocity-Flat-v0` policy input, because it includes `true_fault_state`

## Recommendation

For a cleaner teacher-improvement path, keep the current A1-F teacher clearly labeled as privileged/reference only.

If a future deployment-facing student or residual policy is trained from this teacher, distillation must remove `true_fault_state` from actor/critic runtime observations. If the thesis needs a stronger privileged teacher later, add explicit, documented teacher-only channels intentionally rather than relying on ambiguous term names such as `contacts` or `base_velocity`.

The next audit should verify the student/residual velocity tasks expose exactly the A0-safe 61-dimensional observation and do not inherit `true_fault_state`.
