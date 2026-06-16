# T18-R-PG500 Corrected Simulation Pipeline

This is the paper-grade corrected simulation phase for the conference-stage stripped RLM1 pipeline.

## Canonical Timing

- physics frequency: `500 Hz`
- sim_dt: `0.002 s`
- control frequency: `50 Hz`
- decimation: `10`
- control_dt: `0.02 s`
- history_len: `50`
- history duration: `1.0 s`

The earlier T18-R helper path `sim_dt=0.01, decimation=2` is a 100 Hz physics / 50 Hz control pilot path. It must not be used for T18-R-PG500 paper-grade results.

## Default Compatibility

The base Ant task remains unchanged by default:

- `sim.dt = 1 / 120.0`
- `decimation = 2`
- effective control frequency is approximately `60 Hz`

T18-R-PG500 is opt-in only through:

```bash
--t18r_pg500_timing --require_t18r_pg500_timing
```

These flags enforce:

- `sim_dt = 0.002`
- `decimation = 10`
- `control_dt = 0.02`
- `control_frequency_hz = 50`
- `physics_frequency_hz = 500`

## Method Scope

- RLM1 stripped / conference
- teacher-student ON
- residual ON
- health token OFF
- UQ OFF
- CBF OFF
- teacher may use privileged fault descriptors
- deployment-facing policies must not use selected joint id, q_lock vector, fault-active flag, health token, UQ, or CBF output

## Residual-Ablation Policies

- A2-history H50
- A5-H50 alpha=0.25
- A5-H50 alpha=0.5
- A5-H50 alpha=1.0
- privileged A1-F teacher reference

A2 single-step is intentionally excluded from the T18-R-PG500 residual-ablation figure/table.

## Commands

Use `t18r_pg500_run_commands.md`.
