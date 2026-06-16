# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Velocity-tracking manager-based Ant task configs for RLM experiments.

These configs keep the classic Ant asset and 8-D joint-effort action interface,
but replace the progress-to-far-target objective with commanded base-velocity
tracking. They are intentionally small and local to the Ant task package.
"""

import math

import isaaclab_tasks.manager_based.classic.humanoid.mdp as mdp
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from .ant_env_cfg import (
    ActionsCfg,
    AntEnvCfg,
    EventCfg,
    MySceneCfg,
    TerminationsCfg,
    p2_fault_joint_one_hot,
    p2_fault_q_lock_vector,
    true_fault_state,
)


@configclass
class AntVelocityCommandsCfg:
    """Command specifications for flat Ant velocity tracking."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.0,
        rel_heading_envs=0.0,
        heading_command=False,
        debug_vis=False,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(1.0, 1.0),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0),
            heading=(0.0, 0.0),
        ),
    )


@configclass
class AntForwardRangeVelocityCommandsCfg:
    """Command-conditioned forward velocity tracking for multi-joint teachers."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.0,
        rel_heading_envs=0.0,
        heading_command=False,
        debug_vis=False,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.2, 1.5),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0),
            heading=(0.0, 0.0),
        ),
    )


@configclass
class AntVelocityObservationsCfg:
    """Deployment-facing Ant observations with an explicit velocity command."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for velocity-tracking policy group."""

        base_height = ObsTerm(func=mdp.base_pos_z)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        base_yaw_roll = ObsTerm(func=mdp.base_yaw_roll)
        base_up_proj = ObsTerm(func=mdp.base_up_proj)
        joint_pos_norm = ObsTerm(func=mdp.joint_pos_limit_normalized)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.2)
        feet_body_forces = ObsTerm(
            func=mdp.body_incoming_wrench,
            scale=0.1,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", body_names=["front_left_foot", "front_right_foot", "left_back_foot", "right_back_foot"]
                )
            },
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class AntTeacherVelocityObservationsCfg:
    """Privileged teacher observations for velocity-tracking Ant."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Teacher policy observations. Reference-only; not deployment-facing."""

        base_height = ObsTerm(func=mdp.base_pos_z)
        true_fault_state = ObsTerm(func=true_fault_state)
        base_velocity = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        base_yaw_roll = ObsTerm(func=mdp.base_yaw_roll)
        base_up_proj = ObsTerm(func=mdp.base_up_proj)
        joint_pos_norm = ObsTerm(func=mdp.joint_pos_limit_normalized)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.2)
        contacts = ObsTerm(
            func=mdp.body_incoming_wrench,
            scale=0.1,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", body_names=["front_left_foot", "front_right_foot", "left_back_foot", "right_back_foot"]
                )
            },
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class AntMultiJointP2TeacherVelocityObservationsCfg:
    """Privileged multi-joint P2 teacher observations.

    This 77-D observation group is for the future multi-joint P2 teacher only:
    61-D base student-safe velocity observation plus 8-D selected-joint one-hot
    plus 8-D q-lock vector. The deployment-facing velocity policy group remains
    unchanged and does not receive these true fault-vector terms.
    """

    @configclass
    class PolicyCfg(ObsGroup):
        """Teacher policy observations with one-hot joint and q-lock vectors."""

        base_height = ObsTerm(func=mdp.base_pos_z)
        p2_fault_joint_one_hot = ObsTerm(func=p2_fault_joint_one_hot, params={"fallback_dim": 8})
        p2_fault_q_lock_vector = ObsTerm(func=p2_fault_q_lock_vector, params={"fallback_dim": 8})
        base_velocity = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        base_yaw_roll = ObsTerm(func=mdp.base_yaw_roll)
        base_up_proj = ObsTerm(func=mdp.base_up_proj)
        joint_pos_norm = ObsTerm(func=mdp.joint_pos_limit_normalized)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.2)
        contacts = ObsTerm(
            func=mdp.body_incoming_wrench,
            scale=0.1,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", body_names=["front_left_foot", "front_right_foot", "left_back_foot", "right_back_foot"]
                )
            },
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class AntVelocityRewardsCfg:
    """Velocity-tracking reward terms for flat Ant."""

    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=1.5,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=0.5,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    alive = RewTerm(func=mdp.is_alive, weight=0.5)
    upright = RewTerm(func=mdp.upright_posture_bonus, weight=0.1, params={"threshold": 0.93})
    action_l2 = RewTerm(func=mdp.action_l2, weight=-0.005)
    energy = RewTerm(func=mdp.power_consumption, weight=-0.05, params={"gear_ratio": {".*": 15.0}})
    joint_pos_limits = RewTerm(
        func=mdp.joint_pos_limits_penalty_ratio,
        weight=-0.1,
        params={"threshold": 0.99, "gear_ratio": {".*": 15.0}},
    )


@configclass
class AntVelocityFlatEnvCfg(AntEnvCfg):
    """Flat manager-based Ant task with commanded base-velocity tracking."""

    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=5.0, clone_in_fabric=True)
    observations: AntVelocityObservationsCfg = AntVelocityObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: AntVelocityCommandsCfg = AntVelocityCommandsCfg()
    rewards: AntVelocityRewardsCfg = AntVelocityRewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()


@configclass
class AntTeacherVelocityFlatEnvCfg(AntVelocityFlatEnvCfg):
    """Privileged teacher/reference variant of the Ant velocity-tracking task."""

    observations: AntTeacherVelocityObservationsCfg = AntTeacherVelocityObservationsCfg()


@configclass
class AntMultiJointP2TeacherVelocityFlatEnvCfg(AntVelocityFlatEnvCfg):
    """Privileged teacher/reference variant for random one-joint-per-env P2."""

    observations: AntMultiJointP2TeacherVelocityObservationsCfg = AntMultiJointP2TeacherVelocityObservationsCfg()
    commands: AntForwardRangeVelocityCommandsCfg = AntForwardRangeVelocityCommandsCfg()
