# T09-DEMO-A Meeting Demo Plan

Scope: short advisor-meeting visual demo only; not paper-grade evaluation.

## Cases

| demo_name | ablation_id | fault_profile | purpose |
| --- | --- | --- | --- |
| D0_A0_no_fault | A0 | F0_none | Healthy PPO visual reference without action degradation. |
| D1_A0_P4 | A0 | P4_torque_degradation | Healthy PPO zero-shot response under P4. |
| D2_A5_P4 | A5 | P4_torque_degradation | Frozen student plus residual policy under P4. |

## Fixed P4 Settings

- target_joint: `front_left_foot`
- resolved action index: `4`
- action_dim: `8`
- torque_scale: `0.5`
- fault_onset_step: `50`

## Runtime Defaults

- demo_duration_steps: `400`
- telemetry_interval_steps: `50`
- num_envs: `8`
- seed: `0`

## Guardrails

- No training.
- No checkpoint writes or pointer updates.
- No observed-result manifest updates.
- No paper-grade claims.
- No P2 or multi-fault execution.
- No multi-seed expansion.
- Demo summaries are written under `runs/t09_meeting_demo/`.
