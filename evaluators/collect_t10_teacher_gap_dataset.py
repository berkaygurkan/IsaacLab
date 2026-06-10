#!/usr/bin/env python3
"""Collect T10 A1-F teacher-action datasets for A2/A5/A7 scaffolding.

This is an execution-gated dataset collector for the selected conference P2
velocity teacher. It performs inference only and writes aligned teacher/A0
action data for downstream supervised/distillation work. Isaac Sim is launched
only when ``--execute_collect`` is present.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from run_t09_p2_quick_demo_compare import (
    _asset_from_action_term,
    maybe_set_target_vx,
    prepare_agent_cfg,
    repo_relative,
    resolve_repo_path,
    scalar,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
TEACHER_TASK = "Isaac-Ant-Teacher-Velocity-Flat-v0"
A0_TASK = "Isaac-Ant-Velocity-Flat-v0"
SELECTED_TEACHER_CHECKPOINT = (
    "logs/rsl_rl/teacher_p2_velocity_curriculum__rlm1_stripped__p2_locked_joint/"
    "2026-06-10_19-29-53_a1f_velocity_p2_onset_curriculum_seed0_curriculum_s3_target__seed0/model_2996.pt"
)
DEFAULT_A0_CHECKPOINT = (
    "logs/rsl_rl/healthy_baseline_velocity__rlm1_stripped__none/"
    "2026-06-09_23-58-25_a0_velocity_candidate1000__seed0/model_999.pt"
)
DEFAULT_OUTPUT_DIR = "papers/conference/datasets/t10_teacher_gap_p2_velocity_seed0"
DEFAULT_DATASET_TAG = "t10_teacher_gap_p2_velocity_seed0"
FAULT_PROFILE = "P2_locked_joint"
TARGET_JOINT = "front_left_foot"
REQUESTED_SEMANTICS = "simulation_joint_state_override_lock"
DEFAULT_ONSET_MODE = "random_uniform"
DEFAULT_ONSET_STEP = 50
DEFAULT_ONSET_MIN = 30
DEFAULT_ONSET_MAX = 150
DEFAULT_NUM_ENVS = 128
DEFAULT_NUM_STEPS = 1000
DEFAULT_SEED = 0
TARGET_VX = 1.0
EXPECTED_STUDENT_OBS_DIM = 61
EXPECTED_TEACHER_OBS_DIM = 62
EXPECTED_ACTION_DIM = 8
TRUE_FAULT_STATE_INDEX = 1


class DatasetCollectError(ValueError):
    """Raised for invalid T10 teacher-gap dataset collection state."""


class ObservationViewVecEnvAdapter:
    """Minimal RSL-RL VecEnv view for loading a policy with custom observations."""

    def __init__(self, source_vec_env: Any, initial_obs: Any, *, obs_dim: int) -> None:
        import gymnasium as gym
        import numpy as np

        self.source_vec_env = source_vec_env
        self.num_envs = int(source_vec_env.num_envs)
        self.num_actions = int(source_vec_env.num_actions)
        self.device = source_vec_env.unwrapped.device
        self.max_episode_length = source_vec_env.max_episode_length
        self.cfg = source_vec_env.cfg
        self._obs = initial_obs
        self.single_observation_space = gym.spaces.Dict(
            {"policy": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)}
        )
        self.observation_space = gym.vector.utils.batch_space(self.single_observation_space, self.num_envs)
        self.single_action_space = source_vec_env.unwrapped.single_action_space
        self.action_space = source_vec_env.unwrapped.action_space

    @property
    def unwrapped(self) -> Any:
        return self.source_vec_env.unwrapped

    @property
    def episode_length_buf(self) -> Any:
        return self.source_vec_env.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value: Any) -> None:
        self.source_vec_env.episode_length_buf = value

    def set_observations(self, obs: Any) -> None:
        self._obs = obs

    def get_observations(self) -> Any:
        return self._obs

    def reset(self) -> tuple[Any, dict[str, Any]]:
        return self._obs, {}

    def close(self) -> None:
        return None


def build_parser(*, add_app_launcher_args: bool = False) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect aligned A1-F teacher/A0 action data for T10 P2 velocity distillation scaffolding."
    )
    parser.add_argument("--execute_collect", action="store_true", help="Launch Isaac Sim and collect the dataset.")
    parser.add_argument("--teacher_checkpoint", default=SELECTED_TEACHER_CHECKPOINT)
    parser.add_argument("--a0_checkpoint", default=DEFAULT_A0_CHECKPOINT)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dataset_tag", default=DEFAULT_DATASET_TAG)
    parser.add_argument("--num_envs", type=int, default=DEFAULT_NUM_ENVS)
    parser.add_argument("--num_steps", type=int, default=DEFAULT_NUM_STEPS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--fault_onset_mode", default=DEFAULT_ONSET_MODE, choices=(DEFAULT_ONSET_MODE,))
    parser.add_argument("--fault_onset_step_min", type=int, default=DEFAULT_ONSET_MIN)
    parser.add_argument("--fault_onset_step_max", type=int, default=DEFAULT_ONSET_MAX)
    if not add_app_launcher_args:
        parser.add_argument("--headless", action="store_true")
        parser.add_argument("--device", default=None)
    if add_app_launcher_args:
        from isaaclab.app import AppLauncher

        AppLauncher.add_app_launcher_args(parser)
    return parser


def parse_args() -> argparse.Namespace:
    pre_parser = build_parser(add_app_launcher_args=False)
    pre_args, _ = pre_parser.parse_known_args()
    if pre_args.execute_collect:
        parser = build_parser(add_app_launcher_args=True)
        args, _ = parser.parse_known_args()
        return args
    args, _ = pre_parser.parse_known_args()
    return args


def validate_args(args: argparse.Namespace) -> None:
    if args.num_envs <= 0:
        raise DatasetCollectError("--num_envs must be > 0.")
    if args.num_steps <= 0:
        raise DatasetCollectError("--num_steps must be > 0.")
    if args.seed < 0:
        raise DatasetCollectError("--seed must be non-negative.")
    if args.fault_onset_mode != DEFAULT_ONSET_MODE:
        raise DatasetCollectError("Only random_uniform P2 onset is supported for this scaffold.")
    if args.fault_onset_step_min < 0 or args.fault_onset_step_max < 0:
        raise DatasetCollectError("fault onset bounds must be non-negative.")
    if args.fault_onset_step_min > args.fault_onset_step_max:
        raise DatasetCollectError("--fault_onset_step_min must be <= --fault_onset_step_max.")
    if args.execute_collect:
        for label, checkpoint in (("teacher", args.teacher_checkpoint), ("A0", args.a0_checkpoint)):
            checkpoint_path = resolve_repo_path(checkpoint)
            if not checkpoint_path.is_file():
                raise DatasetCollectError(f"{label} checkpoint does not exist: {repo_relative(checkpoint_path)}")


def print_preview(args: argparse.Namespace) -> None:
    output_dir = resolve_repo_path(args.output_dir)
    print("[T10 TEACHER-GAP DATASET PREVIEW]")
    print("  execute_collect_required: True")
    print("  no_isaac_sim_launched: True")
    print("  no_training: True")
    print("  no_checkpoint_modification: True")
    print("  no_task_config_modification: True")
    print("  no_p2_wrapper_modification: True")
    print("  no_checkpoint_pointer_update: True")
    print("  not_paper_grade_final: True")
    print(f"  dataset_tag: {args.dataset_tag}")
    print(f"  output_dir: {repo_relative(output_dir)}")
    print(f"  teacher_task: {TEACHER_TASK}")
    print(f"  teacher_checkpoint: {args.teacher_checkpoint}")
    print(f"  a0_task: {A0_TASK}")
    print(f"  a0_checkpoint: {args.a0_checkpoint}")
    print(f"  fault_profile: {FAULT_PROFILE}")
    print(f"  target_joint: {TARGET_JOINT}")
    print(f"  semantics: {REQUESTED_SEMANTICS}")
    print("  fallback_allowed: False")
    print(f"  onset: {args.fault_onset_mode} [{args.fault_onset_step_min}, {args.fault_onset_step_max}]")
    print(f"  expected_student_obs_dim: {EXPECTED_STUDENT_OBS_DIM}")
    print(f"  expected_teacher_obs_dim: {EXPECTED_TEACHER_OBS_DIM}")
    print(f"  expected_action_dim: {EXPECTED_ACTION_DIM}")


def set_torch_seed(seed: int) -> None:
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_json(path: Path, values: dict[str, Any]) -> None:
    path.write_text(json.dumps(values, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def policy_tensor(obs: Any) -> Any:
    if isinstance(obs, dict):
        value = obs.get("policy")
    else:
        try:
            value = obs["policy"]
        except Exception:
            value = obs
    if value is None:
        raise DatasetCollectError("observation is missing policy group.")
    return value


def make_policy_obs(policy_obs: Any, *, num_envs: int) -> Any:
    from tensordict import TensorDict

    return TensorDict({"policy": policy_obs}, batch_size=[num_envs])


def student_obs_from_teacher_obs(teacher_obs: Any) -> Any:
    import torch

    if teacher_obs.ndim != 2 or teacher_obs.shape[1] != EXPECTED_TEACHER_OBS_DIM:
        raise DatasetCollectError(
            f"teacher_obs must have shape [num_envs, {EXPECTED_TEACHER_OBS_DIM}], got {tuple(teacher_obs.shape)}."
        )
    return torch.cat(
        (teacher_obs[:, :TRUE_FAULT_STATE_INDEX], teacher_obs[:, TRUE_FAULT_STATE_INDEX + 1 :]),
        dim=1,
    )


def assert_tensor_dim(name: str, tensor: Any, expected_dim: int) -> None:
    if tensor.ndim != 2 or tensor.shape[1] != expected_dim:
        raise DatasetCollectError(f"{name} expected last dim {expected_dim}, got shape {tuple(tensor.shape)}.")


def assert_finite_tensor(name: str, tensor: Any) -> None:
    import torch

    if not torch.isfinite(tensor).all():
        raise DatasetCollectError(f"{name} contains NaN or Inf.")


def current_onset_steps(p2_wrapper: Any, *, num_envs: int) -> Any:
    p2_wrapper._ensure_lock_buffers()
    onset = getattr(p2_wrapper, "per_env_fault_onset_step", None)
    if onset is None:
        raise DatasetCollectError("P2 onset-step buffer is unavailable.")
    if int(onset.numel()) != num_envs:
        raise DatasetCollectError(f"P2 onset-step count mismatch: expected {num_envs}, got {int(onset.numel())}.")
    return onset.detach().clone()


def resolve_forward_velocity_tensor(env: Any, extras: Any, *, action_term_name: str | None, num_envs: int) -> tuple[Any, str]:
    import torch

    candidates: list[tuple[str, Any]] = []
    asset = _asset_from_action_term(env, action_term_name)
    data = getattr(asset, "data", None)
    for attr_name in (
        "root_lin_vel_w",
        "root_link_lin_vel_w",
        "root_com_lin_vel_w",
        "root_lin_vel_b",
        "root_link_lin_vel_b",
        "root_com_lin_vel_b",
    ):
        value = getattr(data, attr_name, None)
        if value is not None:
            candidates.append((f"robot.data.{attr_name}", value))
    for state_name in ("root_state_w", "root_link_state_w"):
        state = getattr(data, state_name, None)
        if state is not None:
            try:
                candidates.append((f"robot.data.{state_name}[:, 7:10]", state[:, 7:10]))
            except Exception:
                pass
    if isinstance(extras, dict):
        for container_name in ("log", "episode", "metrics"):
            container = extras.get(container_name)
            if not isinstance(container, dict):
                continue
            for key in (
                "base_lin_vel_x",
                "base_lin_vel_x_mean",
                "mean_base_lin_vel_x",
                "mean_vel_x",
                "root_lin_vel_x",
            ):
                if key in container:
                    candidates.append((f"extras.{container_name}.{key}", container[key]))

    for source, value in candidates:
        try:
            tensor = torch.as_tensor(value, device=env.unwrapped.device).detach().float()
            if tensor.numel() == 0 or not torch.isfinite(tensor).all():
                continue
            if tensor.ndim == 0:
                vx = tensor.repeat(num_envs)
            elif tensor.ndim == 1:
                vx = tensor
            else:
                vx = tensor[:, 0]
            if vx.numel() == 1 and num_envs > 1:
                vx = vx.repeat(num_envs)
            if int(vx.numel()) != num_envs:
                continue
            return vx.reshape(num_envs).detach().clone(), source
        except Exception:
            continue
    raise DatasetCollectError("could not resolve per-env forward velocity.")


def target_vx_tensor(env: Any, *, num_envs: int) -> tuple[Any, str]:
    import torch

    command_manager = getattr(env.unwrapped, "command_manager", None)
    terms = getattr(command_manager, "_terms", None)
    if isinstance(terms, dict):
        for term_name, term in terms.items():
            for attr_name in ("command", "_command"):
                command = getattr(term, attr_name, None)
                if command is None:
                    continue
                try:
                    tensor = torch.as_tensor(command, device=env.unwrapped.device).detach().float()
                    if tensor.ndim == 2 and tensor.shape[0] == num_envs and tensor.shape[1] >= 1:
                        vx = tensor[:, 0].detach().clone()
                        if torch.isfinite(vx).all():
                            return vx, f"command_manager.{term_name}.{attr_name}[:, 0]"
                except Exception:
                    continue
    return torch.full((num_envs,), float(TARGET_VX), device=env.unwrapped.device), "constant_target_vx"


def done_mask_tensor(dones: Any, *, num_envs: int, device: Any) -> Any:
    import torch

    tensor = torch.as_tensor(dones, device=device)
    if tensor.ndim == 0:
        tensor = tensor.repeat(num_envs)
    if tensor.ndim > 1:
        tensor = tensor.reshape(tensor.shape[0], -1).any(dim=1)
    if int(tensor.shape[0]) != num_envs:
        raise DatasetCollectError(f"done tensor length mismatch: expected {num_envs}, got {int(tensor.shape[0])}.")
    return tensor.to(dtype=torch.bool)


def load_policy(
    *,
    vec_env: Any,
    task: str,
    checkpoint_path: Path,
    policy_label: str,
    log_prefix: str,
) -> tuple[Any, Any]:
    from rsl_rl.runners import DistillationRunner, OnPolicyRunner
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    agent_cfg = load_cfg_from_registry(task, "rsl_rl_cfg_entry_point")
    agent_cfg, agent_cfg_dict = prepare_agent_cfg(agent_cfg)
    print(f"{log_prefix} {policy_label}: runner construction start", flush=True)
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(vec_env, agent_cfg_dict, log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(vec_env, agent_cfg_dict, log_dir=None, device=agent_cfg.device)
    else:
        raise DatasetCollectError(f"{policy_label}: unsupported runner class {agent_cfg.class_name!r}.")
    print(f"{log_prefix} {policy_label}: checkpoint load start {repo_relative(checkpoint_path)}", flush=True)
    runner.load(str(checkpoint_path))
    print(f"{log_prefix} {policy_label}: checkpoint load done", flush=True)
    policy = runner.get_inference_policy(device=vec_env.unwrapped.device)
    return runner, policy


def array_shapes(values: dict[str, Any]) -> dict[str, list[int]]:
    return {name: [int(dim) for dim in value.shape] for name, value in values.items()}


def write_dataset_summary(path: Path, metadata: dict[str, Any]) -> None:
    lines = [
        "# T10 Teacher-Gap P2 Velocity Dataset",
        "",
        "## Purpose",
        "",
        "Candidate-level dataset scaffold for aligned selected A1-F teacher actions, A0 healthy PPO actions, and teacher-gap residual targets under random P2 velocity evaluation.",
        "",
        "## Downstream Usage",
        "",
        "- A2 behavior distillation from `student_obs` to `teacher_action`",
        "- A7 teacher-gap residual using `a7_residual_target = teacher_action - a0_action`",
        "- Later A5 residual after an A2 distilled student exists",
        "",
        "## Guardrail",
        "",
        "Deployment policies must not use `true_fault_state`. `teacher_obs` may contain `true_fault_state` for teacher inference/audit only; deployment-facing A2/A5/A7 inputs must use `student_obs`.",
        "",
        "## Selected Teacher Reason",
        "",
        "The onset curriculum teacher is selected because it has the best post-fault velocity tracking among the evaluated A1-F P2 velocity teacher candidates.",
        "",
        "## Files",
        "",
        f"- dataset: `{metadata['dataset_npz']}`",
        f"- metadata: `{metadata['metadata_json']}`",
        "",
        "## Dataset",
        "",
        f"- tag: `{metadata['dataset_tag']}`",
        f"- num_envs: `{metadata['num_envs']}`",
        f"- num_steps: `{metadata['num_steps']}`",
        f"- student_obs shape: `{metadata['array_shapes']['student_obs']}`",
        f"- teacher_obs shape: `{metadata['array_shapes']['teacher_obs']}`",
        f"- teacher_action shape: `{metadata['array_shapes']['teacher_action']}`",
        f"- a0_action shape: `{metadata['array_shapes']['a0_action']}`",
        f"- a7_residual_target shape: `{metadata['array_shapes']['a7_residual_target']}`",
        "",
        "## Fault / P2 Checks",
        "",
        f"- fallback used: `{metadata['fallback_used']}`",
        f"- P2 fault became active: `{metadata['p2_fault_became_active']}`",
        f"- P2 simulation override applied after onset: `{metadata['p2_simulation_override_applied_after_onset']}`",
        f"- P2 fault active final mean: `{metadata['p2_fault_active_final_mean']}`",
        f"- no NaN/Inf: `{metadata['no_nan_inf_check']}`",
        "",
        "This dataset is candidate-level seed0 evidence and is not paper-grade final.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def collect_dataset(args: argparse.Namespace) -> int:
    import gymnasium as gym
    import numpy as np
    import torch

    import isaaclab_tasks  # noqa: F401
    from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    trainers_dir = REPO_ROOT / "trainers"
    if str(trainers_dir) not in sys.path:
        sys.path.insert(0, str(trainers_dir))
    from p2_joint_lock_training_wrapper import P2JointLockActionMaskWrapper

    output_dir = resolve_repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")

    teacher_checkpoint = resolve_repo_path(args.teacher_checkpoint)
    a0_checkpoint = resolve_repo_path(args.a0_checkpoint)
    log_prefix = "[T10-DATASET]"
    env = None
    vec_env = None
    p2_wrapper = None

    try:
        print(f"{log_prefix} config load start", flush=True)
        env_cfg = load_cfg_from_registry(TEACHER_TASK, "env_cfg_entry_point")
        teacher_agent_cfg = load_cfg_from_registry(TEACHER_TASK, "rsl_rl_cfg_entry_point")
        env_cfg.scene.num_envs = args.num_envs
        env_cfg.seed = args.seed
        if getattr(args, "device", None) is not None:
            env_cfg.sim.device = args.device
        env_cfg.log_dir = str(output_dir / "isaac_logs")
        set_torch_seed(args.seed)
        print(f"{log_prefix} config load done", flush=True)

        print(f"{log_prefix} gym.make start", flush=True)
        env = gym.make(TEACHER_TASK, cfg=env_cfg)
        print(f"{log_prefix} gym.make done", flush=True)
        if isinstance(env.unwrapped, DirectMARLEnv):
            env = multi_agent_to_single_agent(env)

        print(f"{log_prefix} P2 wrapper attach start", flush=True)
        p2_wrapper = P2JointLockActionMaskWrapper(
            env,
            target_joint=TARGET_JOINT,
            fault_onset_step=DEFAULT_ONSET_STEP,
            fault_onset_mode=args.fault_onset_mode,
            fault_onset_step_min=args.fault_onset_step_min,
            fault_onset_step_max=args.fault_onset_step_max,
            expected_action_dim=EXPECTED_ACTION_DIM,
            requested_semantics=REQUESTED_SEMANTICS,
            allow_fallback=False,
            velocity_override=0.0,
            debug=True,
        )
        env = p2_wrapper
        if p2_wrapper.mapping.semantics != REQUESTED_SEMANTICS:
            raise DatasetCollectError(f"P2 actual semantics must be {REQUESTED_SEMANTICS}.")
        if p2_wrapper.fallback_used:
            raise DatasetCollectError("P2 fallback was used during wrapper attachment.")
        print(f"{log_prefix} P2 wrapper attach done", flush=True)

        target_vx_info = maybe_set_target_vx(env, TARGET_VX)
        teacher_agent_cfg, teacher_agent_cfg_dict = prepare_agent_cfg(teacher_agent_cfg)
        vec_env = RslRlVecEnvWrapper(env, clip_actions=teacher_agent_cfg.clip_actions)
        if int(vec_env.num_actions) != EXPECTED_ACTION_DIM:
            raise DatasetCollectError(f"env action dim expected {EXPECTED_ACTION_DIM}, got {int(vec_env.num_actions)}.")

        obs = vec_env.get_observations()
        teacher_obs = policy_tensor(obs).detach().float()
        student_obs = student_obs_from_teacher_obs(teacher_obs)
        assert_tensor_dim("teacher_obs", teacher_obs, EXPECTED_TEACHER_OBS_DIM)
        assert_tensor_dim("student_obs", student_obs, EXPECTED_STUDENT_OBS_DIM)
        assert_finite_tensor("teacher_obs", teacher_obs)
        assert_finite_tensor("student_obs", student_obs)

        set_torch_seed(args.seed)
        num_envs = int(vec_env.num_envs)
        device = vec_env.unwrapped.device
        initial_onset_steps = current_onset_steps(p2_wrapper, num_envs=num_envs)
        p2_action_term_name = p2_wrapper.mapping.action_term_name

        print(f"{log_prefix} teacher runner construction start", flush=True)
        if teacher_agent_cfg.class_name == "OnPolicyRunner":
            from rsl_rl.runners import OnPolicyRunner

            teacher_runner = OnPolicyRunner(vec_env, teacher_agent_cfg_dict, log_dir=None, device=teacher_agent_cfg.device)
        else:
            from rsl_rl.runners import DistillationRunner

            if teacher_agent_cfg.class_name != "DistillationRunner":
                raise DatasetCollectError(f"teacher: unsupported runner class {teacher_agent_cfg.class_name!r}.")
            teacher_runner = DistillationRunner(
                vec_env,
                teacher_agent_cfg_dict,
                log_dir=None,
                device=teacher_agent_cfg.device,
            )
        print(f"{log_prefix} teacher checkpoint load start {repo_relative(teacher_checkpoint)}", flush=True)
        teacher_runner.load(str(teacher_checkpoint))
        print(f"{log_prefix} teacher checkpoint load done", flush=True)
        teacher_policy = teacher_runner.get_inference_policy(device=vec_env.unwrapped.device)

        student_policy_obs = make_policy_obs(student_obs, num_envs=num_envs)
        a0_vec_env = ObservationViewVecEnvAdapter(
            vec_env,
            student_policy_obs,
            obs_dim=EXPECTED_STUDENT_OBS_DIM,
        )
        a0_runner, a0_policy = load_policy(
            vec_env=a0_vec_env,
            task=A0_TASK,
            checkpoint_path=a0_checkpoint,
            policy_label="A0 healthy PPO",
            log_prefix=log_prefix,
        )

        del a0_runner
        arrays: dict[str, Any] = {
            "student_obs": np.empty((args.num_steps, num_envs, EXPECTED_STUDENT_OBS_DIM), dtype=np.float32),
            "teacher_obs": np.empty((args.num_steps, num_envs, EXPECTED_TEACHER_OBS_DIM), dtype=np.float32),
            "teacher_action": np.empty((args.num_steps, num_envs, EXPECTED_ACTION_DIM), dtype=np.float32),
            "a0_action": np.empty((args.num_steps, num_envs, EXPECTED_ACTION_DIM), dtype=np.float32),
            "a7_residual_target": np.empty((args.num_steps, num_envs, EXPECTED_ACTION_DIM), dtype=np.float32),
            "done": np.empty((args.num_steps, num_envs), dtype=np.bool_),
            "reward": np.empty((args.num_steps, num_envs), dtype=np.float32),
            "episode_id": np.empty((args.num_steps, num_envs), dtype=np.int32),
            "env_id": np.tile(np.arange(num_envs, dtype=np.int32), (args.num_steps, 1)),
            "step_index": np.tile(np.arange(args.num_steps, dtype=np.int32).reshape(args.num_steps, 1), (1, num_envs)),
            "p2_fault_active": np.empty((args.num_steps, num_envs), dtype=np.bool_),
            "initial_onset_step_by_env": initial_onset_steps.detach().cpu().numpy().astype(np.int32),
            "velocity_x": np.empty((args.num_steps, num_envs), dtype=np.float32),
            "target_vx": np.empty((args.num_steps, num_envs), dtype=np.float32),
            "vx_error": np.empty((args.num_steps, num_envs), dtype=np.float32),
        }

        episode_id = torch.zeros(num_envs, dtype=torch.int32, device=device)
        no_nan_inf = True
        observed_fault_active = False
        observed_sim_override_after_onset = False
        velocity_source = None
        target_vx_source = target_vx_info.get("target_vx_source") or "pending"

        print(f"{log_prefix} rollout start", flush=True)
        for step_index in range(args.num_steps):
            step_number = step_index + 1
            teacher_obs = policy_tensor(obs).detach().float()
            student_obs = student_obs_from_teacher_obs(teacher_obs)
            assert_tensor_dim("teacher_obs", teacher_obs, EXPECTED_TEACHER_OBS_DIM)
            assert_tensor_dim("student_obs", student_obs, EXPECTED_STUDENT_OBS_DIM)
            assert_finite_tensor("teacher_obs", teacher_obs)
            assert_finite_tensor("student_obs", student_obs)
            student_policy_obs = make_policy_obs(student_obs, num_envs=num_envs)
            a0_vec_env.set_observations(student_policy_obs)

            with torch.inference_mode():
                teacher_action = teacher_policy(obs).detach().float()
                a0_action = a0_policy(student_policy_obs).detach().float()
                assert_tensor_dim("teacher_action", teacher_action, EXPECTED_ACTION_DIM)
                assert_tensor_dim("a0_action", a0_action, EXPECTED_ACTION_DIM)
                assert_finite_tensor("teacher_action", teacher_action)
                assert_finite_tensor("a0_action", a0_action)
                residual_target = teacher_action - a0_action
                assert_finite_tensor("a7_residual_target", residual_target)
                next_obs, rewards, dones, extras = vec_env.step(teacher_action)
                if hasattr(teacher_policy, "reset"):
                    teacher_policy.reset(dones)
                if hasattr(a0_policy, "reset"):
                    a0_policy.reset(dones)

            reward_tensor = torch.as_tensor(rewards, device=device).detach().float()
            done_mask = done_mask_tensor(dones, num_envs=num_envs, device=device)
            log_values = extras.get("log", {}) if isinstance(extras, dict) else {}
            fallback_value = scalar(log_values.get("P2/fallback_used"))
            if p2_wrapper.fallback_used or fallback_value > 0.0:
                raise DatasetCollectError("P2 fallback was used during dataset collection.")

            fault_mask = getattr(p2_wrapper, "last_fault_applied_mask", None)
            if fault_mask is None:
                fault_mask = torch.zeros(num_envs, dtype=torch.bool, device=device)
            else:
                fault_mask = torch.as_tensor(fault_mask, device=device).to(dtype=torch.bool).reshape(num_envs)
            if bool(fault_mask.any().item()):
                observed_fault_active = True
                sim_override_value = scalar(log_values.get("P2/simulation_override_applied"))
                if sim_override_value <= 0.0:
                    raise DatasetCollectError("P2 simulation override was not applied after onset.")
                observed_sim_override_after_onset = True

            velocity_x, step_velocity_source = resolve_forward_velocity_tensor(
                env,
                extras,
                action_term_name=p2_action_term_name,
                num_envs=num_envs,
            )
            target_vx, step_target_vx_source = target_vx_tensor(env, num_envs=num_envs)
            if velocity_source is None:
                velocity_source = step_velocity_source
                print(f"{log_prefix} velocity_source={velocity_source}", flush=True)
            if target_vx_source == "pending":
                target_vx_source = step_target_vx_source
            vx_error = velocity_x - target_vx
            for name, tensor in (
                ("reward", reward_tensor),
                ("velocity_x", velocity_x),
                ("target_vx", target_vx),
                ("vx_error", vx_error),
            ):
                assert_finite_tensor(name, tensor)

            arrays["student_obs"][step_index] = student_obs.detach().cpu().numpy()
            arrays["teacher_obs"][step_index] = teacher_obs.detach().cpu().numpy()
            arrays["teacher_action"][step_index] = teacher_action.detach().cpu().numpy()
            arrays["a0_action"][step_index] = a0_action.detach().cpu().numpy()
            arrays["a7_residual_target"][step_index] = residual_target.detach().cpu().numpy()
            arrays["done"][step_index] = done_mask.detach().cpu().numpy()
            arrays["reward"][step_index] = reward_tensor.detach().cpu().numpy()
            arrays["episode_id"][step_index] = episode_id.detach().cpu().numpy()
            arrays["p2_fault_active"][step_index] = fault_mask.detach().cpu().numpy()
            arrays["velocity_x"][step_index] = velocity_x.detach().cpu().numpy()
            arrays["target_vx"][step_index] = target_vx.detach().cpu().numpy()
            arrays["vx_error"][step_index] = vx_error.detach().cpu().numpy()

            no_nan_inf = no_nan_inf and all(
                bool(np.isfinite(arrays[name][step_index]).all())
                for name in (
                    "student_obs",
                    "teacher_obs",
                    "teacher_action",
                    "a0_action",
                    "a7_residual_target",
                    "reward",
                    "velocity_x",
                    "target_vx",
                    "vx_error",
                )
            )
            episode_id = episode_id + done_mask.to(dtype=torch.int32)
            obs = next_obs

            if step_index < 3 or step_number % 100 == 0 or step_number == args.num_steps:
                print(
                    f"{log_prefix} step={step_number} reward_mean={float(reward_tensor.mean().cpu().item()):.4f} "
                    f"done_count={int(done_mask.sum().cpu().item())} "
                    f"p2_fault_active={float(fault_mask.float().mean().cpu().item()):.4f}",
                    flush=True,
                )

        if not observed_fault_active:
            raise DatasetCollectError("P2 fault never became active during collection.")
        if not observed_sim_override_after_onset:
            raise DatasetCollectError("P2 simulation override was never observed after onset.")
        if not no_nan_inf:
            raise DatasetCollectError("dataset contains NaN or Inf values.")

        observation_dimension_check = {
            "student_obs_expected": EXPECTED_STUDENT_OBS_DIM,
            "student_obs_observed": int(arrays["student_obs"].shape[-1]),
            "student_obs_passed": int(arrays["student_obs"].shape[-1]) == EXPECTED_STUDENT_OBS_DIM,
            "teacher_obs_expected": EXPECTED_TEACHER_OBS_DIM,
            "teacher_obs_observed": int(arrays["teacher_obs"].shape[-1]),
            "teacher_obs_passed": int(arrays["teacher_obs"].shape[-1]) == EXPECTED_TEACHER_OBS_DIM,
            "true_fault_state_index_in_teacher_obs": TRUE_FAULT_STATE_INDEX,
            "student_obs_excludes_true_fault_state": True,
        }
        action_dimension_check = {
            "action_expected": EXPECTED_ACTION_DIM,
            "teacher_action_observed": int(arrays["teacher_action"].shape[-1]),
            "a0_action_observed": int(arrays["a0_action"].shape[-1]),
            "a7_residual_target_observed": int(arrays["a7_residual_target"].shape[-1]),
            "passed": all(
                int(arrays[name].shape[-1]) == EXPECTED_ACTION_DIM
                for name in ("teacher_action", "a0_action", "a7_residual_target")
            ),
        }
        if not observation_dimension_check["student_obs_passed"] or not observation_dimension_check["teacher_obs_passed"]:
            raise DatasetCollectError(f"observation dimension check failed: {observation_dimension_check}")
        if not action_dimension_check["passed"]:
            raise DatasetCollectError(f"action dimension check failed: {action_dimension_check}")

        dataset_path = output_dir / "dataset.npz"
        np.savez_compressed(dataset_path, **arrays)
        metadata_path = output_dir / "metadata.json"
        summary_path = output_dir / "dataset_summary.md"
        metadata = {
            "dataset_scope": "t10_teacher_gap_p2_velocity_candidate_dataset",
            "dataset_tag": args.dataset_tag,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "not_paper_grade_final": True,
            "note": "candidate-level dataset, seed0/default scaffold unless overridden; not paper-grade final",
            "selected_teacher_reason": "curriculum teacher has best post-fault velocity tracking",
            "selected_teacher_checkpoint_path": repo_relative(teacher_checkpoint),
            "a0_checkpoint_path": repo_relative(a0_checkpoint),
            "teacher_task": TEACHER_TASK,
            "a0_task": A0_TASK,
            "fault_profile": FAULT_PROFILE,
            "target_joint": TARGET_JOINT,
            "onset_mode": args.fault_onset_mode,
            "onset_step_min": args.fault_onset_step_min,
            "onset_step_max": args.fault_onset_step_max,
            "semantics": REQUESTED_SEMANTICS,
            "fallback_allowed": False,
            "fallback_used": False,
            "p2_fault_became_active": bool(observed_fault_active),
            "p2_simulation_override_applied_after_onset": bool(observed_sim_override_after_onset),
            "p2_fault_active_any": bool(arrays["p2_fault_active"].any()),
            "p2_fault_active_final_mean": float(arrays["p2_fault_active"][-1].mean()),
            "num_envs": args.num_envs,
            "num_steps": args.num_steps,
            "seed": args.seed,
            "target_vx": TARGET_VX,
            "target_vx_source": target_vx_source,
            "velocity_x_source": velocity_source,
            "array_shapes": array_shapes(arrays),
            "observation_dimension_check": observation_dimension_check,
            "action_dimension_check": action_dimension_check,
            "no_nan_inf_check": bool(no_nan_inf),
            "dataset_npz": repo_relative(dataset_path),
            "metadata_json": repo_relative(metadata_path),
            "dataset_summary_md": repo_relative(summary_path),
            "guardrails": {
                "no_training": True,
                "no_checkpoint_modification": True,
                "no_task_config_modification": True,
                "no_p2_wrapper_modification": True,
                "no_checkpoint_pointer_update": True,
                "student_obs_must_not_use_true_fault_state": True,
                "teacher_obs_may_include_true_fault_state_for_teacher_inference_audit_only": True,
                "a5_student_gap_target_deferred_until_a2_student_exists": True,
            },
            "downstream_usage": {
                "A2": "behavior distillation from student_obs to teacher_action",
                "A7": "healthy PPO base plus teacher-gap residual target teacher_action - a0_action",
                "A5": "later student-gap residual after A2 distilled student exists",
            },
        }
        write_json(metadata_path, metadata)
        write_dataset_summary(summary_path, metadata)
        print(f"{log_prefix} dataset written: {repo_relative(dataset_path)}", flush=True)
        print(f"{log_prefix} metadata written: {repo_relative(metadata_path)}", flush=True)
        print(f"{log_prefix} summary written: {repo_relative(summary_path)}", flush=True)
        return 0
    finally:
        if vec_env is not None:
            vec_env.close()
        elif env is not None:
            env.close()


def execute_collect(args: argparse.Namespace) -> int:
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    try:
        print("[T10-DATASET WARNING] Candidate-level dataset only; not paper-grade final.", flush=True)
        return collect_dataset(args)
    finally:
        simulation_app.close()


def main() -> int:
    try:
        args = parse_args()
        validate_args(args)
        if not args.execute_collect:
            print_preview(args)
            return 0
        return execute_collect(args)
    except DatasetCollectError as exc:
        print(f"[T10-DATASET ERROR] {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
