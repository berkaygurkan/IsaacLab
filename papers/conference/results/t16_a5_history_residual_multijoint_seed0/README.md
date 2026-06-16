# T16 Multi-Joint A5 History Residual Distillation

Offline supervised residual distillation for the RLM1 stripped / conference multi-joint P2 pipeline.

## Definition

- A5 = frozen T15 A2-history H16 base + learned residual.
- residual target: `teacher_action - base_action`.
- residual input: `student_obs history [batch, 16, 61]`.
- reconstruction used for offline metrics: `base_action + 1.0 * residual_action`.
- This is not A7; A7 would use an A0 healthy PPO base.

## Frozen Base

- base checkpoint: `papers/conference/results/t15_a2_history_h16_multijoint_seed0/a2_history_h16_multijoint.pt`
- base action validation MSE: `0.00025525588769352537`
- base action validation MAE: `0.011037527943299747`

## Datasets

- `papers/conference/datasets/t14_multijoint_teacher_v2b_realistic_seed0/dataset.npz`: `128000` samples
- `papers/conference/datasets/t14_multijoint_teacher_v2b_late_seed0/dataset.npz`: `128000` samples

## Metrics

- residual train MSE: `0.00012140956207429043`
- residual train MAE: `0.007392700953560817`
- residual validation MSE: `0.00017540603258029065`
- residual validation MAE: `0.008434747885032883`
- reconstructed validation MSE: `0.00017540603258029065`
- reconstructed validation MAE: `0.008434748055172362`
- validation MSE improvement over base: `7.984985511323472e-05`
- validation MAE improvement over base: `0.0026027798881273854`

## Later Closed-Loop Evaluation

- Offline metrics use alpha=1.0 by default.
- Isaac-side deployment evaluation may later sweep alpha values such as 0.25, 0.5, and 1.0.
- No closed-loop evaluation is run here.

## Guardrails

- Residual input uses only deployment-facing `student_obs` history.
- `teacher_obs` is not used as residual input.
- selected fault joint index/one-hot, q-lock vector, and P2-active flag are not used as residual input.
- health token is OFF.
- A0 action is not used.
- This is A5, not A7.
- This is offline supervised training only; no Isaac, RL, checkpoint pointer update, or dataset mutation.
