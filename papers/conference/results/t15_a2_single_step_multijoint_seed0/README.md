# T15 Multi-Joint A2 Student Distillation

Offline supervised student distillation for the RLM1 stripped / conference multi-joint P2 pipeline.

## Mode

- mode: `single_step`
- history_len: `None`
- input: `student_obs [batch, 61]`
- target: `teacher_action [batch, 8]`

## Datasets

- `papers/conference/datasets/t14_multijoint_teacher_v2b_realistic_seed0/dataset.npz`: `128000` samples
- `papers/conference/datasets/t14_multijoint_teacher_v2b_late_seed0/dataset.npz`: `128000` samples

## Metrics

- train_mse: `0.0003945040258576868`
- train_mae: `0.014540776962527624`
- val_mse: `0.000398049967000845`
- val_mae: `0.014730211309102122`
- train samples: `232814`
- validation samples: `23186`

## Guardrails

- Student input uses only deployment-facing `student_obs`.
- `teacher_obs` is not used as student input.
- selected fault joint index/one-hot, q-lock vector, and P2-active flag are not used as student input.
- health token is OFF.
- teacher action is the supervised target.
- H16 is an input history window, not the latent itself.
- latent `z_t` analysis belongs to later T18; no t-SNE/UMAP is implemented here.
- This is offline supervised training only; no Isaac, RL, checkpoint pointer update, or dataset mutation.
