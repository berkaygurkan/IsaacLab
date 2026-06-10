# T10 Ant Velocity Tracking Task Scaffold

Scope: T10-IL-02 minimal scaffold and dry-run validation only. No training, checkpoint freeze, checkpoint pointer edit, paper-grade manifest update, Isaac Lab core edit, or RSL-RL core edit is performed by this scaffold.

## Registered Task IDs

```text
Isaac-Ant-Velocity-Flat-v0
Isaac-Ant-Teacher-Velocity-Flat-v0
```

Both tasks are manager-based and registered in:

```text
source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/ant/__init__.py
```

The environment config path is:

```text
source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/ant/ant_velocity_env_cfg.py
```

## Command And Observation Scaffold

The new task adds a real `base_velocity` command through:

```text
mdp.UniformVelocityCommandCfg
```

Default scaffold command ranges:

```text
lin_vel_x = (1.0, 1.0)
lin_vel_y = (0.0, 0.0)
ang_vel_z = (0.0, 0.0)
```

The deployment-facing policy observation includes:

```text
velocity_commands = mdp.generated_commands(command_name="base_velocity")
```

The deployment-facing policy observation does not include `true_fault_state`.

## Reward Scaffold

The velocity scaffold removes the classic Ant `progress` and `move_to_target` reward terms from the new task config. The main objective is commanded velocity tracking:

```text
track_lin_vel_xy_exp
track_ang_vel_z_exp
```

It keeps conservative support terms:

```text
alive
upright
action_l2
energy
joint_pos_limits
```

## P2 Compatibility

The scaffold preserves the classic Ant 8-D joint-effort action interface:

```text
joint_effort = mdp.JointEffortActionCfg(asset_name="robot", joint_names=[".*"], scale=7.5)
```

The intended P2 compatibility requirements remain:

- `env.unwrapped.action_manager` exists.
- `total_action_dim == 8`.
- `front_left_foot` maps to exactly one action/joint index.
- `robot.write_joint_state_to_sim` remains available.
- `simulation_joint_state_override_lock` is required.
- fallback is disabled.

## Preflight

Dry-run preflight:

```bash
python evaluators/preflight_t10_ant_velocity_task.py --dry_run
```

Runtime smoke is intentionally explicit and separate:

```bash
python evaluators/preflight_t10_ant_velocity_task.py --execute_smoke --headless
```

The runtime smoke is not training. It is intended to create the task, check `base_velocity`, attach the P2 wrapper, reset, and step once.

## T10-IL-02b Runtime Launch Order

T10-IL-02b updates the runtime preflight path to mirror the known-working T09 P2 preflight order:

1. Parse CLI with no heavy runtime imports in dry-run mode.
2. Import and create `AppLauncher` only when `--execute_smoke` is set.
3. Import `gymnasium`, `isaaclab_tasks`, `parse_env_cfg`, and the P2 wrapper after AppLauncher exists.
4. Use `parse_env_cfg(...)` before `gym.make(...)`.
5. Attach the existing P2 simulation-level joint lock wrapper after `gym.make(...)`.

Runtime mode no longer falls back to static validation. If runtime launch fails, it prints explicit diagnostics such as `app_launcher_created`, `task_registration_loaded`, `gym_spec_found`, and the original exception.

In the current Codex shell, the allowed smoke command stops before AppLauncher because plain `python` cannot import `isaaclab`:

```text
app_launcher_created: false
app_launcher_import_failed: ModuleNotFoundError: No module named 'isaaclab'
```

This is an environment/interpreter mismatch in the local shell, not evidence that the Isaac Sim install or Warp package is broken. The user-side IsaacLab Python environment that runs the T09 preflight should be used for the next runtime smoke.

## Guardrails

- No paper-grade claim is made.
- A1-F / teacher task is privileged and reference-only.
- A5/A6/A7 are deployment-facing only if privileged runtime information is removed.
- Health token OFF, UQ inactive, CBF inactive, and no P3 expansion remain fixed.
