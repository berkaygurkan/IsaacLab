# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOffPolicyRunnerCfg, RslRlSacActorCriticCfg, RslRlSacAlgorithmCfg


@configclass
class UnitreeA1RoughSACRunnerCfg(RslRlOffPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 1500
    save_interval = 50
    experiment_name = "unitree_a1_rough_sac"
    # empirical_normalization = False # SAC genellikle normalizasyondan faydalanır

    policy = RslRlSacActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        discrete=False, # Sürekli eylem uzayı için False
        deterministic_actor=False, # SAC stokastik bir aktördür
    )
    algorithm = RslRlSacAlgorithmCfg(
        discount_factor=0.99,
        learning_rate=3e-4, # Aktör, kritik ve alfa için öğrenme oranı
        polyak=0.995,
        entropy_coef="auto", # Otomatik entropy ayarlama
        target_entropy="auto",
        alpha_learning_rate=3e-4,
        grad_norm_clip=1.0,
        tau=0.005,
    )
    replay_buffer_kwargs = dict(
        capacity=100000,
        sampler="uniform",
        sample_batch_size=256,
        num_workers=0,
    )


@configclass
class UnitreeA1FlatSACRunnerCfg(UnitreeA1RoughSACRunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 300
        self.experiment_name = "unitree_a1_flat_sac"
        self.policy.actor_hidden_dims = [128, 128, 128]
        self.policy.critic_hidden_dims = [128, 128, 128]