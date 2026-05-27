from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlDistillationAlgorithmCfg,
    RslRlDistillationRunnerCfg,
    RslRlMLPModelCfg,
    RslRlRNNModelCfg,
)


@configclass
class AntStudentDistillationRunnerCfg(RslRlDistillationRunnerCfg):
    seed = 0
    num_steps_per_env = 32
    max_iterations = 1000
    save_interval = 50
    experiment_name = "student__rlm1_stripped__none"
    run_name = "student__rlm1_stripped__none__seed0"
    obs_groups = {"student": ["policy"], "teacher": ["teacher_policy"]}
    student = RslRlRNNModelCfg(
        hidden_dims=[400, 200, 100],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.1),
        rnn_type="lstm",
        rnn_hidden_dim=128,
        rnn_num_layers=1,
    )
    teacher = RslRlMLPModelCfg(
        hidden_dims=[400, 200, 100],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.0),
    )
    algorithm = RslRlDistillationAlgorithmCfg(
        num_learning_epochs=2,
        learning_rate=1.0e-3,
        gradient_length=15,
        max_grad_norm=1.0,
        loss_type="mse",
    )
