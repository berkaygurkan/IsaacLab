# T11 P2 Target-Joint Stress Plan

Documentation-first and runner-first scaffold for conference-stage RLM1 stripped P2 target-joint expansion. This is candidate-level stress testing only and is not paper-grade final evidence.

## Motivation

The current P2 locked-joint setting has focused on `front_left_foot`. A2-history H16 is strong in that setting, leaving little useful room for residual correction. T11 expands the locked target joint across all Ant action joints to test whether harder or more diverse P2 cases reveal useful residual behavior from A5 or A7.

## Target Joints

- `front_left_foot`
- `front_right_foot`
- `left_back_foot`
- `right_back_foot`
- `front_left_leg`
- `front_right_leg`
- `left_back_leg`
- `right_back_leg`

## Protocols

| protocol | fault_onset_mode | min step | max step |
| --- | --- | ---: | ---: |
| random_p2 | `random_uniform` | 30 | 150 |
| settled_p2 | `random_uniform` | 300 | 300 |

The P2 semantics remain `simulation_joint_state_override_lock`; fallback remains disabled.

## First Stress-Pass Policies

| policy | checkpoint(s) | alpha values |
| --- | --- | --- |
| A2-history H16 | `papers/conference/results/t10_a2_student_history_distill_full_h16_seed0/a2_student_history.pt` | n/a |
| A5 residual over A2-history | `papers/conference/results/t10_a5_history_residual_distill_full_h16_seed0/a5_history_residual.pt` with A2 base | `0.0`, `0.25`, `0.5` |
| A7 residual over A0 | `papers/conference/results/t10_a7_a0_residual_distill_full_h16_seed0/a7_a0_residual.pt` with A0 base | `0.0`, `0.01`, `0.025`, `0.05` |

A1-F teacher is not included in the first stress runner. It remains a privileged reference and can be added later only if its evaluator supports the expanded target-joint protocol cleanly.

## Runner

The batch runner is:

```text
evaluators/run_t11_p2_target_joint_stress_eval.py
```

It orchestrates existing T10 evaluators as subprocesses and does not duplicate policy logic. It writes one output directory per protocol, target joint, policy, and alpha:

```text
papers/conference/results/t11_p2_target_joint_stress/
  random_p2/<joint_name>/a2_history/
  random_p2/<joint_name>/a5_alpha_000/
  random_p2/<joint_name>/a5_alpha_025/
  random_p2/<joint_name>/a5_alpha_050/
  random_p2/<joint_name>/a7_alpha_000/
  random_p2/<joint_name>/a7_alpha_001/
  random_p2/<joint_name>/a7_alpha_0025/
  random_p2/<joint_name>/a7_alpha_005/
  settled_p2/<joint_name>/...
```

Each execution run saves evaluator stdout/stderr to `terminal.log` in the run output directory. Successful runs are aggregated into:

- `papers/conference/results/t11_p2_target_joint_stress/t11_p2_target_joint_stress_summary.csv`
- `papers/conference/results/t11_p2_target_joint_stress/t11_p2_target_joint_stress_summary.md`

## Guardrails

- No training.
- No checkpoint modification.
- No task config modification.
- No P2 semantic modification.
- No `teacher_obs`.
- No `true_fault_state`.
- No health token.
- No UQ.
- No CBF.
- No paper-grade claims from this stress scaffold.

## Recommended First Execution

Start with one joint and one protocol before launching the full 128-command pass:

```text
python evaluators/run_t11_p2_target_joint_stress_eval.py --execute --protocol random --joint front_left_foot --num_envs 16 --num_steps 200 --device cpu
```

If that smoke execution succeeds, scale to all joints/protocols under the same command surface.
