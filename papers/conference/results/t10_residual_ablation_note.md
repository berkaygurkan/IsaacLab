# T10 Residual Ablation Note

Documentation-only note for the conference-stage RLM1 stripped residual results. No training, Isaac Sim run, checkpoint edit, task config edit, P2 wrapper edit, or artifact removal was performed.

## Ablation Status

A5 and A7 are retained as residual ablations. They are not deleted, and their checkpoints/results remain part of the candidate record:

- A5 residual checkpoint: `papers/conference/results/t10_a5_history_residual_distill_full_h16_seed0/a5_history_residual.pt`
- A7 residual checkpoint: `papers/conference/results/t10_a7_a0_residual_distill_full_h16_seed0/a7_a0_residual.pt`
- A5 alpha sweep: `papers/conference/results/t10_a5_history_residual_p2_eval_alpha_sweep_seed0/a5_alpha_sweep_summary.md`
- A7 diagnostic: `papers/conference/results/t10_a7_a0_residual_p2_eval_diagnostic_smoke/a7_diagnostic_summary.md`

## A5 Interpretation

A5 learns a residual over the A2-history H16 base. The A2-history base is already strong under the current P2 evaluations, leaving little useful residual correction room. Although A5 improved offline teacher-action reconstruction, the closed-loop alpha sweep did not improve over the alpha=0 base.

## A7 Interpretation

A7 learns a residual over frozen A0 healthy PPO. The weak A0 base creates a large offline residual target, and the residual model can reduce offline teacher-action reconstruction error. In closed-loop deployment, however, injecting the learned residual caused distribution shift or over-correction: the tiny-alpha diagnostic did not improve over alpha=0, and larger smoke alphas were weak.

## Residual Conclusion

In this conference P2 single-joint setting, residual action-gap imitation did not improve closed-loop performance over the history-student policy.

This conclusion is deliberately narrow. It should not be generalized to all future P2/P3/P4 faults or to real robot deployment.

## Future Residual Work

Future P1/P2/P3 phases may revisit residuals with different fault types, online fine-tuning, residual regularization, safety constraints, or a deployment protocol designed specifically for stable residual injection.
