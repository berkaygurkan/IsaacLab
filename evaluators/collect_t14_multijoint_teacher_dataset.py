#!/usr/bin/env python3
"""Collect T14 multi-joint A1-F teacher rollout datasets.

This is an execution-gated scaffold for the selected T13-B v2b multi-joint
privileged teacher. Isaac Sim is launched only when ``--execute_collect`` is
present. The collector preserves the conference RLM1 stripped semantics:
teacher observations are privileged 77-D, downstream student observations are
61-D and fault-descriptor-free, P2 uses direct simulation-state override, and
fallback/surrogate behavior is forbidden.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

from collect_t10_teacher_gap_dataset import (
    ObservationViewVecEnvAdapter,
    assert_finite_tensor,
    assert_tensor_dim,
    done_mask_tensor,
    load_policy,
    make_policy_obs,
    policy_tensor,
    resolve_forward_velocity_tensor,
)
from run_t09_p2_quick_demo_compare import prepare_agent_cfg, repo_relative, resolve_repo_path, scalar
from run_t13c_multijoint_teacher_eval import (
    command_vx_tensor,
    force_fixed_velocity_command,
    validate_command_mode,
)
from t18r_control_timing import add_control_timing_args, apply_control_timing_to_env_cfg


REPO_ROOT = Path(__file__).resolve().parents[1]

TASK = "Isaac-Ant-Teacher-Velocity-MultiJointP2-Flat-v0"
A0_TASK = "Isaac-Ant-Velocity-Flat-v0"
POLICY_LABEL = "A1F_multijoint_teacher_v2b_valid_but_foot_limited"
RLM_PHASE = "RLM1 stripped / conference"
TEACHER_SELECTION_README = "papers/conference/results/t13_multi_joint_teacher_selection/README.md"
SELECTED_TEACHER_CHECKPOINT = (
    "logs/rsl_rl/teacher_p2_multijoint_velocity_curriculum_v2b__rlm1_stripped__p2_multi_joint_random/"
    "2026-06-15_18-16-03_a1f_multijoint_velocity_p2_v2b_curriculum_curriculum_s2_realistic_transition_random_p2__seed0/"
    "model_9997.pt"
)
REJECTED_HARD_FOOT_CHECKPOINT = (
    "logs/rsl_rl/teacher_p2_multijoint_velocity_curriculum_v2b__rlm1_stripped__p2_multi_joint_random/"
    "2026-06-15_21-17-39_a1f_multijoint_velocity_p2_hard_foot_finetune_120_700_i2000__seed0/"
    "model_11996.pt"
)
DEFAULT_OUTPUT_DIR = "papers/conference/datasets/t14_multijoint_teacher_v2b_seed0"
DEFAULT_DATASET_TAG = "t14_multijoint_teacher_v2b_seed0"

FAULT_PROFILE = "P2_locked_joint"
TARGET_JOINT_MODE = "random_per_env"
TARGET_JOINT_PLACEHOLDER = "front_left_foot"
REQUESTED_SEMANTICS = "simulation_joint_state_override_lock"
DEFAULT_ONSET_STEP = 50
EXPECTED_ACTION_DIM = 8
EXPECTED_TEACHER_OBS_DIM = 77
EXPECTED_STUDENT_OBS_DIM = 61
PRIVILEGED_BLOCK_START = 1
PRIVILEGED_BLOCK_END = 17
FIXED_VX = 1.0
COMMAND_RANDOM_RANGE = (0.2, 1.5)
DEFAULT_NUM_ENVS = 128
DEFAULT_NUM_STEPS = 1000
DEFAULT_SEED = 0
DEFAULT_DEVICE = "cuda"

PROTOCOLS = {
    "realistic_random": {
        "fault_onset_step_min": 120,
        "fault_onset_step_max": 700,
        "description": "realistic healthy-to-fault transition, matching selected v2b S2",
    },
    "late_random": {
        "fault_onset_step_min": 250,
        "fault_onset_step_max": 700,
        "description": "late random P2 onset, matching selected v2b S1",
    },
    "stress_random": {
        "fault_onset_step_min": 30,
        "fault_onset_step_max": 700,
        "description": "early-onset robustness stress protocol, opt-in only",
    },
}

VELOCITY_MODES = {
    "command_random": {
        "target_vx": None,
        "description": "task command-conditioned range vx_cmd in [0.2, 1.5]",
    },
    "fixed_vx_1p0": {
        "target_vx": FIXED_VX,
        "description": "fixed vx_cmd = 1.0, vy = 0, yaw = 0 for comparability",
    },
}


class T14DatasetError(ValueError):
    """Raised for invalid T14 dataset collection state."""


class ProgressLogger:
    """Tiny stdout + terminal.log progress logger for long Isaac collection runs."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.output_dir / "terminal.log"
        self.current_stage = "initializing"
        self._stream = self.path.open("a", encoding="utf-8")

    def close(self) -> None:
        self._stream.close()

    def set_stage(self, stage: str) -> None:
        self.current_stage = stage
        self.log(f"stage={stage}")

    def log(self, message: str) -> None:
        timestamp = datetime.now().isoformat(timespec="seconds")
        line = f"[T14-DATASET {timestamp}] {message}"
        print(line, flush=True)
        self._stream.write(line + "\n")
        self._stream.flush()

    def write_error(self, exc: BaseException) -> Path:
        error_path = self.output_dir / "error_summary.json"
        write_json(
            error_path,
            {
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "current_stage": self.current_stage,
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
                "traceback": traceback.format_exc(),
                "terminal_log": repo_relative(self.path),
            },
        )
        return error_path


def build_parser(*, add_app_launcher_args: bool = False) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect T14 multi-joint privileged teacher rollout dataset.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry_run", action="store_true", help="Preview dataset collection without launching Isaac.")
    mode.add_argument("--execute_collect", action="store_true", help="Launch Isaac Sim and collect the dataset.")
    parser.add_argument("--teacher_checkpoint", default=SELECTED_TEACHER_CHECKPOINT)
    parser.add_argument(
        "--allow_teacher_checkpoint_override",
        action="store_true",
        help="Allow a non-selected teacher checkpoint. The rejected T13-D checkpoint is still forbidden.",
    )
    parser.add_argument("--a0_checkpoint", default=None, help="Optional A0 checkpoint for a0_action/residual targets.")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dataset_tag", default=DEFAULT_DATASET_TAG)
    parser.add_argument(
        "--protocol",
        default="default",
        choices=("default", "realistic_random", "late_random", "stress_random", "all"),
        help="default collects realistic_random and late_random; stress_random is opt-in.",
    )
    parser.add_argument("--include_stress", action="store_true", help="Include stress_random with the default protocol set.")
    parser.add_argument("--velocity_mode", default="command_random", choices=("command_random", "fixed_vx_1p0", "both"))
    parser.add_argument("--num_envs", type=int, default=DEFAULT_NUM_ENVS)
    parser.add_argument("--num_steps", type=int, default=DEFAULT_NUM_STEPS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--progress_every", type=int, default=100)
    parser.add_argument(
        "--debug_max_protocols",
        type=int,
        default=0,
        help="Optional execute-mode limit on protocol/mode blocks; 0 means no limit.",
    )
    add_control_timing_args(parser)
    if not add_app_launcher_args:
        parser.add_argument("--headless", action="store_true", default=True)
        parser.add_argument("--device", default=DEFAULT_DEVICE)
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
        args.headless = True
        return args
    return pre_args


def norm_repo_path(path_value: str) -> str:
    return repo_relative(resolve_repo_path(path_value))


def selected_protocols(args: argparse.Namespace) -> list[str]:
    if args.protocol == "default":
        protocols = ["realistic_random", "late_random"]
        if args.include_stress:
            protocols.append("stress_random")
        return protocols
    if args.protocol == "all":
        return ["realistic_random", "late_random", "stress_random"]
    return [args.protocol]


def selected_velocity_modes(args: argparse.Namespace) -> list[str]:
    if args.velocity_mode == "both":
        return ["command_random", "fixed_vx_1p0"]
    return [args.velocity_mode]


def selected_protocol_mode_pairs(args: argparse.Namespace) -> list[tuple[str, str]]:
    pairs = [
        (protocol, velocity_mode)
        for protocol in selected_protocols(args)
        for velocity_mode in selected_velocity_modes(args)
    ]
    if args.debug_max_protocols > 0:
        return pairs[: args.debug_max_protocols]
    return pairs


def validate_args(args: argparse.Namespace, logger: ProgressLogger | None = None) -> None:
    if logger is not None:
        logger.set_stage("argument validation")
    if args.num_envs <= 0:
        raise T14DatasetError("--num_envs must be > 0.")
    if args.num_steps <= 0:
        raise T14DatasetError("--num_steps must be > 0.")
    if args.seed < 0:
        raise T14DatasetError("--seed must be non-negative.")
    if args.progress_every <= 0:
        raise T14DatasetError("--progress_every must be > 0.")
    if args.debug_max_protocols < 0:
        raise T14DatasetError("--debug_max_protocols must be >= 0.")

    if logger is not None:
        logger.set_stage("rejected checkpoint guard")
    checkpoint = norm_repo_path(args.teacher_checkpoint)
    selected = norm_repo_path(SELECTED_TEACHER_CHECKPOINT)
    rejected = norm_repo_path(REJECTED_HARD_FOOT_CHECKPOINT)
    if checkpoint == rejected:
        raise T14DatasetError("Rejected T13-D hard-foot checkpoint must not be used downstream.")
    if checkpoint != selected and not args.allow_teacher_checkpoint_override:
        raise T14DatasetError(
            "T14 defaults to the selected v2b model_9997.pt teacher. "
            "Use --allow_teacher_checkpoint_override only for an explicit non-rejected override."
        )
    if not checkpoint.endswith("model_9997.pt") and not args.allow_teacher_checkpoint_override:
        raise T14DatasetError("teacher checkpoint must point to model_9997.pt unless explicitly overridden.")

    if args.execute_collect:
        if logger is not None:
            logger.set_stage("selected checkpoint validation")
        teacher_checkpoint = resolve_repo_path(args.teacher_checkpoint)
        if not teacher_checkpoint.is_file():
            raise T14DatasetError(f"teacher checkpoint does not exist: {repo_relative(teacher_checkpoint)}")
        if args.a0_checkpoint:
            a0_checkpoint = resolve_repo_path(args.a0_checkpoint)
            if not a0_checkpoint.is_file():
                raise T14DatasetError(f"A0 checkpoint does not exist: {repo_relative(a0_checkpoint)}")


def write_json(path: Path, values: dict[str, Any]) -> None:
    path.write_text(json.dumps(values, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def set_torch_seed(seed: int) -> None:
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def teacher_obs_to_student_obs(teacher_obs: Any) -> Any:
    import torch

    if teacher_obs.ndim != 2 or teacher_obs.shape[1] != EXPECTED_TEACHER_OBS_DIM:
        raise T14DatasetError(
            f"teacher_obs must have shape [num_envs, {EXPECTED_TEACHER_OBS_DIM}], got {tuple(teacher_obs.shape)}."
        )
    return torch.cat(
        (teacher_obs[:, :PRIVILEGED_BLOCK_START], teacher_obs[:, PRIVILEGED_BLOCK_END:]),
        dim=1,
    )


def tensor_to_numpy(value: Any):
    return value.detach().cpu().numpy()


def command_vx_or_fail(env: Any, *, num_envs: int):
    import torch

    command = command_vx_tensor(env)
    if command is None:
        raise T14DatasetError("velocity command tensor is unavailable.")
    command = torch.as_tensor(command, device=env.unwrapped.device).detach().float().reshape(-1)
    if int(command.numel()) != num_envs:
        raise T14DatasetError(f"vx_cmd length mismatch: expected {num_envs}, got {int(command.numel())}.")
    assert_finite_tensor("vx_cmd", command)
    return command


def per_env_yaw_rate_tensor(env: Any, *, action_term_name: str | None, num_envs: int):
    import torch

    action_manager = getattr(env.unwrapped, "action_manager", None)
    terms = getattr(action_manager, "_terms", None)
    term = terms.get(action_term_name) if isinstance(terms, dict) and action_term_name else None
    if term is None and isinstance(terms, dict) and terms:
        term = next(iter(terms.values()))
    data = getattr(getattr(term, "_asset", None), "data", None)
    for attr_name in (
        "root_ang_vel_w",
        "root_link_ang_vel_w",
        "root_com_ang_vel_w",
        "root_ang_vel_b",
        "root_link_ang_vel_b",
        "root_com_ang_vel_b",
    ):
        value = getattr(data, attr_name, None)
        if value is None:
            continue
        tensor = torch.as_tensor(value, device=env.unwrapped.device).detach().float()
        if tensor.ndim >= 2 and tensor.shape[0] == num_envs and tensor.shape[-1] >= 3:
            yaw_rate = tensor[:, -1].reshape(num_envs)
            if torch.isfinite(yaw_rate).all():
                return yaw_rate, f"robot.data.{attr_name}[:, -1]"
    return torch.zeros(num_envs, device=env.unwrapped.device), "unavailable_zero_filled"


def alloc_protocol_arrays(*, num_steps: int, num_envs: int, include_a0: bool) -> dict[str, Any]:
    import numpy as np

    arrays: dict[str, Any] = {
        "student_obs": np.empty((num_steps, num_envs, EXPECTED_STUDENT_OBS_DIM), dtype=np.float32),
        "teacher_obs": np.empty((num_steps, num_envs, EXPECTED_TEACHER_OBS_DIM), dtype=np.float32),
        "teacher_action": np.empty((num_steps, num_envs, EXPECTED_ACTION_DIM), dtype=np.float32),
        "selected_fault_joint_index": np.empty((num_steps, num_envs), dtype=np.int32),
        "selected_fault_joint_one_hot": np.empty((num_steps, num_envs, EXPECTED_ACTION_DIM), dtype=np.float32),
        "q_lock_vector": np.empty((num_steps, num_envs, EXPECTED_ACTION_DIM), dtype=np.float32),
        "p2_fault_active": np.empty((num_steps, num_envs), dtype=np.bool_),
        "fault_onset_step": np.empty((num_steps, num_envs), dtype=np.int32),
        "vx_cmd": np.empty((num_steps, num_envs), dtype=np.float32),
        "velocity_x": np.empty((num_steps, num_envs), dtype=np.float32),
        "vx_error": np.empty((num_steps, num_envs), dtype=np.float32),
        "yaw_rate": np.empty((num_steps, num_envs), dtype=np.float32),
        "yaw_error": np.empty((num_steps, num_envs), dtype=np.float32),
        "done": np.empty((num_steps, num_envs), dtype=np.bool_),
        "reward": np.empty((num_steps, num_envs), dtype=np.float32),
        "episode_id": np.empty((num_steps, num_envs), dtype=np.int32),
        "env_id": np.tile(np.arange(num_envs, dtype=np.int32), (num_steps, 1)),
        "timestep": np.tile(np.arange(num_steps, dtype=np.int32).reshape(num_steps, 1), (1, num_envs)),
    }
    if include_a0:
        arrays["a0_action"] = np.empty((num_steps, num_envs, EXPECTED_ACTION_DIM), dtype=np.float32)
        arrays["a7_residual_target"] = np.empty((num_steps, num_envs, EXPECTED_ACTION_DIM), dtype=np.float32)
    return arrays


def flatten_protocol_arrays(protocol_arrays: dict[str, Any], *, protocol: str, velocity_mode: str) -> dict[str, Any]:
    import numpy as np

    flattened: dict[str, Any] = {}
    sample_count = None
    for name, value in protocol_arrays.items():
        if value.ndim == 3:
            flattened[name] = value.reshape((-1, value.shape[-1]))
        else:
            flattened[name] = value.reshape(-1)
        if sample_count is None:
            sample_count = int(flattened[name].shape[0])
    if sample_count is None:
        raise T14DatasetError("empty protocol arrays.")
    flattened["protocol_label"] = np.full(sample_count, protocol, dtype="U32")
    flattened["velocity_mode_label"] = np.full(sample_count, velocity_mode, dtype="U32")
    return flattened


def concat_blocks(blocks: list[dict[str, Any]]) -> dict[str, Any]:
    import numpy as np

    if not blocks:
        raise T14DatasetError("no dataset blocks were collected.")
    keys = sorted(blocks[0].keys())
    merged: dict[str, Any] = {}
    for key in keys:
        if any(key not in block for block in blocks):
            continue
        merged[key] = np.concatenate([block[key] for block in blocks], axis=0)
    merged["sample_index"] = np.arange(int(next(iter(merged.values())).shape[0]), dtype=np.int64)
    return merged


def array_shapes(values: dict[str, Any]) -> dict[str, list[int]]:
    return {name: [int(dim) for dim in value.shape] for name, value in values.items()}


def numeric_no_nan_inf(values: dict[str, Any]) -> bool:
    import numpy as np

    for value in values.values():
        if value.dtype.kind in {"f", "i", "u"} and not np.isfinite(value).all():
            return False
    return True


def command_stats_from_dataset(dataset: dict[str, Any], velocity_mode: str) -> dict[str, Any]:
    import numpy as np

    mask = dataset["velocity_mode_label"] == velocity_mode
    vx_cmd = dataset["vx_cmd"][mask]
    if vx_cmd.size == 0:
        return {"vx_cmd_mean": None, "vx_cmd_min": None, "vx_cmd_max": None}
    return {
        "vx_cmd_mean": float(np.mean(vx_cmd)),
        "vx_cmd_min": float(np.min(vx_cmd)),
        "vx_cmd_max": float(np.max(vx_cmd)),
    }


def collect_protocol_mode(
    args: argparse.Namespace,
    *,
    protocol: str,
    velocity_mode: str,
    logger: ProgressLogger,
) -> tuple[dict[str, Any], dict[str, Any]]:
    import gymnasium as gym
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
    protocol_cfg = PROTOCOLS[protocol]
    log_prefix = f"[T14-DATASET {protocol}/{velocity_mode}]"
    env = None
    vec_env = None
    p2_wrapper = None
    a0_vec_env = None

    try:
        logger.set_stage(f"{protocol}/{velocity_mode}: config load start")
        env_cfg = load_cfg_from_registry(TASK, "env_cfg_entry_point")
        agent_cfg = load_cfg_from_registry(TASK, "rsl_rl_cfg_entry_point")
        env_cfg.scene.num_envs = args.num_envs
        env_cfg.seed = args.seed
        if getattr(args, "device", None):
            env_cfg.sim.device = args.device
        env_cfg.log_dir = str(output_dir / "isaac_logs" / protocol / velocity_mode)
        control_timing = apply_control_timing_to_env_cfg(env_cfg, args)
        set_torch_seed(args.seed)
        logger.log(
            f"{log_prefix} config load end num_envs={args.num_envs} seed={args.seed} "
            f"device={getattr(args, 'device', DEFAULT_DEVICE)} "
            f"control_frequency_hz={control_timing['control_frequency_hz']:.9g} "
            f"physics_frequency_hz={control_timing['physics_frequency_hz']:.9g} "
            f"control_dt_s={control_timing['control_dt_s']:.9g} "
            f"sim_dt_s={control_timing['sim_dt_s']:.9g} decimation={control_timing['decimation']}"
        )

        logger.set_stage(f"{protocol}/{velocity_mode}: environment creation start")
        env = gym.make(TASK, cfg=env_cfg)
        if isinstance(env.unwrapped, DirectMARLEnv):
            env = multi_agent_to_single_agent(env)
        logger.log(f"{log_prefix} environment creation end task={TASK}")

        logger.set_stage(f"{protocol}/{velocity_mode}: wrapper attachment start")
        p2_wrapper = P2JointLockActionMaskWrapper(
            env,
            target_joint=TARGET_JOINT_PLACEHOLDER,
            target_joint_mode=TARGET_JOINT_MODE,
            fault_onset_step=DEFAULT_ONSET_STEP,
            fault_onset_mode="random_uniform",
            fault_onset_step_min=protocol_cfg["fault_onset_step_min"],
            fault_onset_step_max=protocol_cfg["fault_onset_step_max"],
            expected_action_dim=EXPECTED_ACTION_DIM,
            requested_semantics=REQUESTED_SEMANTICS,
            allow_fallback=False,
            velocity_override=0.0,
            debug=True,
        )
        env = p2_wrapper
        if p2_wrapper.mapping.semantics != REQUESTED_SEMANTICS:
            raise T14DatasetError(f"actual P2 semantics must be {REQUESTED_SEMANTICS}.")
        if p2_wrapper.mapping.target_joint_mode != TARGET_JOINT_MODE:
            raise T14DatasetError(f"target_joint_mode must be {TARGET_JOINT_MODE}.")
        if p2_wrapper.mapping.allow_fallback or p2_wrapper.fallback_used:
            raise T14DatasetError("fallback must remain disabled for T14 collection.")
        if len(p2_wrapper.mapping.supported_target_joints) != EXPECTED_ACTION_DIM:
            raise T14DatasetError(
                "T14 downstream collection must randomize over all 8 supported actuated joints; "
                f"got {p2_wrapper.mapping.supported_target_joints}."
            )
        logger.log(
            f"{log_prefix} wrapper attachment end semantics={p2_wrapper.mapping.semantics} "
            f"target_joint_mode={p2_wrapper.mapping.target_joint_mode} "
            f"supported_joint_count={len(p2_wrapper.mapping.supported_target_joints)} "
            f"fallback_used={p2_wrapper.fallback_used}"
        )

        logger.set_stage(f"{protocol}/{velocity_mode}: velocity command mode setup")
        if velocity_mode == "fixed_vx_1p0":
            target_info = force_fixed_velocity_command(env, FIXED_VX)
            if not target_info.get("target_vx_available"):
                raise T14DatasetError("fixed_vx_1p0 requires writable command tensor.")
        else:
            target_info = {
                "target_vx_available": False,
                "target_vx": None,
                "command_forced_each_step": False,
                "target_vx_mode_note": "command_random leaves the command-conditioned task range active",
            }
        logger.log(f"{log_prefix} velocity command mode setup end mode={velocity_mode} target_info={target_info}")

        logger.set_stage(f"{protocol}/{velocity_mode}: vec env setup start")
        agent_cfg, agent_cfg_dict = prepare_agent_cfg(agent_cfg)
        vec_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        num_envs = int(vec_env.num_envs)
        device = vec_env.unwrapped.device
        logger.log(f"{log_prefix} vec env setup end num_envs={num_envs} device={device}")

        if velocity_mode == "fixed_vx_1p0":
            force_fixed_velocity_command(env, FIXED_VX)
        obs = vec_env.get_observations()
        if velocity_mode == "fixed_vx_1p0":
            force_fixed_velocity_command(env, FIXED_VX)
            obs = vec_env.get_observations()

        logger.set_stage(f"{protocol}/{velocity_mode}: policy loading start")
        teacher_checkpoint = resolve_repo_path(args.teacher_checkpoint)
        teacher_runner, teacher_policy = load_policy(
            vec_env=vec_env,
            task=TASK,
            checkpoint_path=teacher_checkpoint,
            policy_label=POLICY_LABEL,
            log_prefix=log_prefix,
        )
        del teacher_runner
        logger.log(f"{log_prefix} teacher policy loading end checkpoint={repo_relative(teacher_checkpoint)}")

        include_a0 = bool(args.a0_checkpoint)
        if include_a0:
            logger.set_stage(f"{protocol}/{velocity_mode}: optional A0 policy loading start")
            initial_teacher_obs = policy_tensor(obs).detach().float()
            initial_student_obs = teacher_obs_to_student_obs(initial_teacher_obs)
            a0_vec_env = ObservationViewVecEnvAdapter(
                vec_env,
                make_policy_obs(initial_student_obs, num_envs=num_envs),
                obs_dim=EXPECTED_STUDENT_OBS_DIM,
            )
            a0_runner, a0_policy = load_policy(
                vec_env=a0_vec_env,
                task=A0_TASK,
                checkpoint_path=resolve_repo_path(args.a0_checkpoint),
                policy_label="A0 healthy PPO optional base",
                log_prefix=log_prefix,
            )
            del a0_runner
            logger.log(f"{log_prefix} optional A0 policy loading end checkpoint={args.a0_checkpoint}")
        else:
            a0_policy = None

        logger.set_stage(f"{protocol}/{velocity_mode}: rollout loop start")
        p2_wrapper._ensure_lock_buffers()
        p2_action_term_name = p2_wrapper.mapping.action_term_name
        arrays = alloc_protocol_arrays(num_steps=args.num_steps, num_envs=num_envs, include_a0=include_a0)
        episode_id = torch.zeros(num_envs, dtype=torch.int32, device=device)
        observed_fault_active = False
        observed_sim_override = False
        yaw_rate_source = None
        velocity_source = None
        no_nan_inf = True

        for step_index in range(args.num_steps):
            if velocity_mode == "fixed_vx_1p0":
                force_fixed_velocity_command(env, FIXED_VX)
                obs = vec_env.get_observations()

            teacher_obs = policy_tensor(obs).detach().float()
            student_obs = teacher_obs_to_student_obs(teacher_obs)
            assert_tensor_dim("teacher_obs", teacher_obs, EXPECTED_TEACHER_OBS_DIM)
            assert_tensor_dim("student_obs", student_obs, EXPECTED_STUDENT_OBS_DIM)
            assert_finite_tensor("teacher_obs", teacher_obs)
            assert_finite_tensor("student_obs", student_obs)

            vx_cmd = command_vx_or_fail(env, num_envs=num_envs)
            selected_index = p2_wrapper.per_env_target_action_index.detach().long()
            selected_one_hot = p2_wrapper.p2_fault_joint_one_hot.detach().float()
            q_lock_vector = p2_wrapper.p2_fault_q_lock_vector.detach().float()
            onset_step = p2_wrapper.per_env_fault_onset_step.detach().long()

            with torch.inference_mode():
                teacher_action = teacher_policy(obs).detach().float()
                assert_tensor_dim("teacher_action", teacher_action, EXPECTED_ACTION_DIM)
                assert_finite_tensor("teacher_action", teacher_action)
                if include_a0 and a0_policy is not None and a0_vec_env is not None:
                    student_policy_obs = make_policy_obs(student_obs, num_envs=num_envs)
                    a0_vec_env.set_observations(student_policy_obs)
                    a0_action = a0_policy(student_policy_obs).detach().float()
                    assert_tensor_dim("a0_action", a0_action, EXPECTED_ACTION_DIM)
                    assert_finite_tensor("a0_action", a0_action)
                    residual_target = teacher_action - a0_action
                    assert_finite_tensor("a7_residual_target", residual_target)
                else:
                    a0_action = None
                    residual_target = None

                next_obs, rewards, dones, extras = vec_env.step(teacher_action)
                if velocity_mode == "fixed_vx_1p0":
                    force_fixed_velocity_command(env, FIXED_VX)
                    next_obs = vec_env.get_observations()
                if hasattr(teacher_policy, "reset"):
                    teacher_policy.reset(dones)
                if include_a0 and a0_policy is not None and hasattr(a0_policy, "reset"):
                    a0_policy.reset(dones)

            reward_tensor = torch.as_tensor(rewards, device=device).detach().float()
            done_mask = done_mask_tensor(dones, num_envs=num_envs, device=device)
            log_values = extras.get("log", {}) if isinstance(extras, dict) else {}
            if p2_wrapper.fallback_used or scalar(log_values.get("P2/fallback_used")) > 0.0:
                raise T14DatasetError("P2 fallback was used during dataset collection.")

            fault_mask = getattr(p2_wrapper, "last_fault_applied_mask", None)
            if fault_mask is None:
                fault_mask = torch.zeros(num_envs, dtype=torch.bool, device=device)
            fault_mask = torch.as_tensor(fault_mask, device=device).to(dtype=torch.bool).reshape(num_envs)
            if bool(fault_mask.any().item()):
                observed_fault_active = True
                if scalar(log_values.get("P2/simulation_override_applied")) <= 0.0:
                    raise T14DatasetError("P2 simulation override was not applied after onset.")
                observed_sim_override = True

            velocity_x, step_velocity_source = resolve_forward_velocity_tensor(
                env,
                extras,
                action_term_name=p2_action_term_name,
                num_envs=num_envs,
            )
            yaw_rate, step_yaw_source = per_env_yaw_rate_tensor(
                env,
                action_term_name=p2_action_term_name,
                num_envs=num_envs,
            )
            if velocity_source is None:
                velocity_source = step_velocity_source
            if yaw_rate_source is None:
                yaw_rate_source = step_yaw_source
            vx_error = velocity_x - vx_cmd
            yaw_error = yaw_rate

            for name, tensor in (
                ("reward", reward_tensor),
                ("velocity_x", velocity_x),
                ("vx_error", vx_error),
                ("yaw_rate", yaw_rate),
                ("yaw_error", yaw_error),
            ):
                assert_finite_tensor(name, tensor)

            arrays["student_obs"][step_index] = tensor_to_numpy(student_obs)
            arrays["teacher_obs"][step_index] = tensor_to_numpy(teacher_obs)
            arrays["teacher_action"][step_index] = tensor_to_numpy(teacher_action)
            arrays["selected_fault_joint_index"][step_index] = tensor_to_numpy(selected_index)
            arrays["selected_fault_joint_one_hot"][step_index] = tensor_to_numpy(selected_one_hot)
            arrays["q_lock_vector"][step_index] = tensor_to_numpy(q_lock_vector)
            arrays["p2_fault_active"][step_index] = tensor_to_numpy(fault_mask)
            arrays["fault_onset_step"][step_index] = tensor_to_numpy(onset_step)
            arrays["vx_cmd"][step_index] = tensor_to_numpy(vx_cmd)
            arrays["velocity_x"][step_index] = tensor_to_numpy(velocity_x)
            arrays["vx_error"][step_index] = tensor_to_numpy(vx_error)
            arrays["yaw_rate"][step_index] = tensor_to_numpy(yaw_rate)
            arrays["yaw_error"][step_index] = tensor_to_numpy(yaw_error)
            arrays["done"][step_index] = tensor_to_numpy(done_mask)
            arrays["reward"][step_index] = tensor_to_numpy(reward_tensor)
            arrays["episode_id"][step_index] = tensor_to_numpy(episode_id)
            if include_a0 and a0_action is not None and residual_target is not None:
                arrays["a0_action"][step_index] = tensor_to_numpy(a0_action)
                arrays["a7_residual_target"][step_index] = tensor_to_numpy(residual_target)

            no_nan_inf = no_nan_inf and all(
                bool(__import__("numpy").isfinite(arrays[name][step_index]).all())
                for name in ("student_obs", "teacher_obs", "teacher_action", "reward", "velocity_x", "vx_cmd", "vx_error")
            )
            episode_id = episode_id + done_mask.to(dtype=torch.int32)
            obs = next_obs

            step_number = step_index + 1
            if step_number == 1 or step_number % args.progress_every == 0 or step_number == args.num_steps:
                logger.log(
                    f"{log_prefix} rollout progress step={step_number}/{args.num_steps} "
                    f"reward_mean={float(reward_tensor.mean().cpu().item()):.6g} "
                    f"done_count={int(done_mask.sum().cpu().item())} "
                    f"fault_active_mean={float(fault_mask.float().mean().cpu().item()):.6g} "
                    f"no_nan_inf={no_nan_inf}"
                )

        if not observed_fault_active:
            raise T14DatasetError(f"P2 fault never became active for {protocol}/{velocity_mode}.")
        if not observed_sim_override:
            raise T14DatasetError(f"P2 simulation override was never observed for {protocol}/{velocity_mode}.")
        if not no_nan_inf:
            raise T14DatasetError(f"NaN/Inf detected for {protocol}/{velocity_mode}.")

        logger.set_stage(f"{protocol}/{velocity_mode}: protocol summary creation")
        flat_arrays = flatten_protocol_arrays(arrays, protocol=protocol, velocity_mode=velocity_mode)
        summary = {
            "protocol": protocol,
            "velocity_mode": velocity_mode,
            "num_envs": num_envs,
            "num_steps": args.num_steps,
            "fault_onset_step_min": protocol_cfg["fault_onset_step_min"],
            "fault_onset_step_max": protocol_cfg["fault_onset_step_max"],
            "p2_fault_became_active": bool(observed_fault_active),
            "simulation_override_applied": bool(observed_sim_override),
            "fallback_used": False,
            "supported_joint_names": list(p2_wrapper.mapping.supported_target_joints),
            "supported_joint_count": len(p2_wrapper.mapping.supported_target_joints),
            "velocity_source": velocity_source,
            "yaw_rate_source": yaw_rate_source,
            "target_info": target_info,
            **control_timing,
            "no_nan_inf": bool(no_nan_inf),
        }
        logger.log(
            f"{log_prefix} protocol summary creation end samples={int(flat_arrays['teacher_action'].shape[0])} "
            f"fault_active_any={bool(flat_arrays['p2_fault_active'].any())}"
        )
        return flat_arrays, summary
    finally:
        logger.log(f"{log_prefix} cleanup start")
        if vec_env is not None:
            vec_env.close()
        elif env is not None:
            env.close()
        logger.log(f"{log_prefix} cleanup end")


def write_dataset_readme(path: Path, metadata: dict[str, Any]) -> None:
    lines = [
        "# T14 Multi-Joint Teacher Dataset",
        "",
        "Dataset scaffold for selected T13-B v2b A1-F multi-joint teacher rollouts.",
        "",
        "## Selected Teacher",
        "",
        f"- label: `{metadata['selected_teacher_label']}`",
        f"- checkpoint: `{metadata['selected_teacher_checkpoint']}`",
        f"- selection note: `{metadata['teacher_selection_readme']}`",
        "",
        "## Schema",
        "",
        "- `student_obs`: 61-D deployment-facing observation",
        "- `teacher_obs`: 77-D privileged teacher observation",
        "- `teacher_action`: 8-D teacher action",
        "- `selected_fault_joint_index`: selected locked-joint action index",
        "- `selected_fault_joint_one_hot`: 8-D selected-joint vector",
        "- `q_lock_vector`: 8-D q-lock vector",
        "- `p2_fault_active`, `fault_onset_step`, `vx_cmd`, `velocity_x`, `vx_error`",
        "- `yaw_rate`, `yaw_error`, `done`, `episode_id`, `timestep`",
        "- `protocol_label`, `velocity_mode_label`",
        "",
        "Optional fields `a0_action` and `a7_residual_target` are written only when",
        "`--a0_checkpoint` is explicitly provided.",
        "",
        "## Semantics",
        "",
        "- one selected locked joint per env/episode",
        "- selected joint random over all 8 Ant actuated joints",
        "- q_lock captured from the selected joint's current position at onset",
        "- direct `simulation_joint_state_override_lock`",
        "- enforce selected joint `q = q_lock` and `qd = 0` after onset",
        "- fallback disabled",
        "- PD surrogate disabled",
        "- health token OFF",
        f"- control frequency: `{metadata.get('control_frequency_hz', '')}` Hz",
        f"- physics frequency: `{metadata.get('physics_frequency_hz', '')}` Hz",
        f"- control timestep: `{metadata.get('control_dt_s', '')}` s",
        f"- sim dt / decimation: `{metadata.get('sim_dt_s', '')}` / `{metadata.get('decimation', '')}`",
        "",
        "This dataset is candidate-level evidence for downstream A2/A2-history/A5/A7",
        "scaffolding and is not paper-grade final.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_preview(args: argparse.Namespace) -> None:
    output_dir = resolve_repo_path(args.output_dir)
    print("[T14 MULTI-JOINT TEACHER DATASET PREVIEW]")
    print("  execute_collect_required: true")
    print("  no_isaac_sim_launched: true")
    print("  no_training: true")
    print("  no_checkpoint_modification: true")
    print("  no_task_config_modification: true")
    print("  no_p2_wrapper_modification: true")
    print(f"  dataset_tag: {args.dataset_tag}")
    print(f"  output_dir: {repo_relative(output_dir)}")
    print(f"  teacher_checkpoint: {norm_repo_path(args.teacher_checkpoint)}")
    print(f"  rejected_checkpoint_forbidden: {REJECTED_HARD_FOOT_CHECKPOINT}")
    print(f"  selected_protocols: {selected_protocols(args)}")
    print(f"  selected_velocity_modes: {selected_velocity_modes(args)}")
    print(f"  num_envs: {args.num_envs}")
    print(f"  num_steps: {args.num_steps}")
    print(f"  seed: {args.seed}")
    print(f"  device: {getattr(args, 'device', DEFAULT_DEVICE)}")
    print(f"  progress_every: {args.progress_every}")
    print(f"  debug_max_protocols: {args.debug_max_protocols}")
    print(f"  requested_control_frequency_hz: {args.control_frequency_hz}")
    print(f"  requested_sim_dt: {args.sim_dt}")
    print(f"  requested_decimation: {args.decimation}")
    print(f"  required_control_frequency_hz: {args.require_control_frequency_hz}")
    print(f"  t18r_pg500_timing: {args.t18r_pg500_timing}")
    print(f"  require_t18r_pg500_timing: {args.require_t18r_pg500_timing}")
    print(f"  teacher_obs_dim: {EXPECTED_TEACHER_OBS_DIM}")
    print(f"  student_obs_dim: {EXPECTED_STUDENT_OBS_DIM}")
    print("  p2_semantics: simulation_joint_state_override_lock")
    print("  fallback_allowed: false")
    print("  pd_surrogate_allowed: false")
    print("  selected_joint_random_over_all_8: true")
    print(f"  a0_optional_fields_enabled: {bool(args.a0_checkpoint)}")


def execute_collect(args: argparse.Namespace, logger: ProgressLogger) -> int:
    import numpy as np

    output_dir = resolve_repo_path(args.output_dir)
    logger.set_stage("output directory creation")
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.log(f"output_dir={repo_relative(output_dir)}")

    logger.set_stage("command.txt write")
    (output_dir / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    logger.log(f"command_txt={repo_relative(output_dir / 'command.txt')}")

    simulation_app = None
    try:
        logger.set_stage("app/Isaac launch start")
        logger.log("candidate-level dataset collection; not paper-grade final")
        from isaaclab.app import AppLauncher

        app_launcher = AppLauncher(args)
        simulation_app = app_launcher.app
        logger.log("app/Isaac launch end")

        logger.set_stage("protocol loop start")
        protocol_mode_pairs = selected_protocol_mode_pairs(args)
        logger.log(
            f"protocol_mode_pairs={protocol_mode_pairs} debug_max_protocols={args.debug_max_protocols} "
            f"progress_every={args.progress_every}"
        )
        blocks: list[dict[str, Any]] = []
        run_summaries: list[dict[str, Any]] = []
        for index, (protocol, velocity_mode) in enumerate(protocol_mode_pairs, start=1):
            logger.log(
                f"protocol loop item {index}/{len(protocol_mode_pairs)} start "
                f"protocol={protocol} velocity_mode={velocity_mode}"
            )
            block, summary = collect_protocol_mode(
                args,
                protocol=protocol,
                velocity_mode=velocity_mode,
                logger=logger,
            )
            blocks.append(block)
            run_summaries.append(summary)
            logger.log(
                f"protocol loop item {index}/{len(protocol_mode_pairs)} end "
                f"protocol={protocol} velocity_mode={velocity_mode}"
            )

        logger.set_stage("array stacking start")
        dataset = concat_blocks(blocks)
        logger.log(f"array stacking end arrays={sorted(dataset.keys())}")
        no_nan_inf = numeric_no_nan_inf(dataset)
        if not no_nan_inf:
            raise T14DatasetError("merged dataset contains NaN or Inf.")

        logger.set_stage("command mode validation")
        command_validation: dict[str, dict[str, Any]] = {}
        collected_velocity_modes = sorted({velocity_mode for _, velocity_mode in protocol_mode_pairs})
        for velocity_mode in collected_velocity_modes:
            stats = command_stats_from_dataset(dataset, velocity_mode)
            valid, error = validate_command_mode(velocity_mode, stats)
            command_validation[velocity_mode] = {
                **stats,
                "command_mode_valid": bool(valid),
                "command_mode_validation_error": error,
            }
            if not valid:
                raise T14DatasetError(f"{velocity_mode} command validation failed: {error}")
            logger.log(f"command mode validation passed velocity_mode={velocity_mode} stats={stats}")

        dataset_path = output_dir / "dataset.npz"
        metadata_path = output_dir / "metadata.json"
        summary_path = output_dir / "dataset_summary.md"
        logger.set_stage("dataset.npz write start")
        np.savez_compressed(dataset_path, **dataset)
        logger.log(f"dataset.npz write end path={repo_relative(dataset_path)}")
        supported_joint_names = run_summaries[0].get("supported_joint_names", []) if run_summaries else []

        logger.set_stage("metadata creation")
        control_timing_values = run_summaries[0] if run_summaries else {}
        metadata = {
            "dataset_scope": "t14_multijoint_teacher_rollout_dataset",
            "dataset_tag": args.dataset_tag,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "not_paper_grade_final": True,
            "selected_teacher_label": "A1-F multi-joint teacher v2b, valid but foot-limited",
            "selected_teacher_checkpoint": norm_repo_path(args.teacher_checkpoint),
            "selected_teacher_checkpoint_expected": SELECTED_TEACHER_CHECKPOINT,
            "rejected_hard_foot_checkpoint_forbidden": REJECTED_HARD_FOOT_CHECKPOINT,
            "teacher_selection_readme": TEACHER_SELECTION_README,
            "rlm_phase": RLM_PHASE,
            "health_token_enabled": False,
            "teacher_obs_dim": EXPECTED_TEACHER_OBS_DIM,
            "student_obs_dim": EXPECTED_STUDENT_OBS_DIM,
            "student_obs_excludes_fault_joint_one_hot": True,
            "student_obs_excludes_q_lock_vector": True,
            "task": TASK,
            "fault_profile": FAULT_PROFILE,
            "target_joint_mode": TARGET_JOINT_MODE,
            "selected_joint_random_over_all_8": True,
            "supported_joint_names": supported_joint_names,
            "supported_joint_count": len(supported_joint_names),
            "q_lock_semantics": "captured_from_current_selected_joint_position_at_fault_onset",
            "p2_semantics": REQUESTED_SEMANTICS,
            "post_onset_enforcement": "q[selected_joint] = q_lock and qd[selected_joint] = 0",
            "fallback_allowed": False,
            "pd_surrogate_allowed": False,
            "selected_protocols": sorted({protocol for protocol, _ in protocol_mode_pairs}),
            "selected_velocity_modes": collected_velocity_modes,
            "debug_max_protocols": args.debug_max_protocols,
            "progress_every": args.progress_every,
            "command_random_vx_range": list(COMMAND_RANDOM_RANGE),
            "num_envs": args.num_envs,
            "num_steps": args.num_steps,
            "seed": args.seed,
            "device": getattr(args, "device", DEFAULT_DEVICE),
            "control_frequency_hz": control_timing_values.get("control_frequency_hz"),
            "control_dt_s": control_timing_values.get("control_dt_s"),
            "physics_frequency_hz": control_timing_values.get("physics_frequency_hz"),
            "sim_dt_s": control_timing_values.get("sim_dt_s"),
            "decimation": control_timing_values.get("decimation"),
            "episode_length_s": control_timing_values.get("episode_length_s"),
            "render_interval": control_timing_values.get("render_interval"),
            "control_timing_source": control_timing_values.get("control_timing_source"),
            "control_timing_changed": control_timing_values.get("control_timing_changed"),
            "required_control_frequency_hz": control_timing_values.get("required_control_frequency_hz"),
            "t18r_canonical_50hz_h50": control_timing_values.get("t18r_canonical_50hz_h50"),
            "t18r_pg500_50hz_h50": control_timing_values.get("t18r_pg500_50hz_h50"),
            "a0_action_available": bool(args.a0_checkpoint),
            "a0_checkpoint": norm_repo_path(args.a0_checkpoint) if args.a0_checkpoint else None,
            "a7_residual_target_available": bool(args.a0_checkpoint),
            "run_summaries": run_summaries,
            "command_mode_validation": command_validation,
            "array_shapes": array_shapes(dataset),
            "no_nan_inf_check": bool(no_nan_inf),
            "dataset_npz": repo_relative(dataset_path),
            "metadata_json": repo_relative(metadata_path),
            "dataset_summary_md": repo_relative(summary_path),
            "terminal_log": repo_relative(logger.path),
            "guardrails": {
                "no_training": True,
                "no_checkpoint_modification": True,
                "no_task_config_modification": True,
                "no_p2_wrapper_modification": True,
                "rejected_hard_foot_checkpoint_not_used": norm_repo_path(args.teacher_checkpoint)
                != norm_repo_path(REJECTED_HARD_FOOT_CHECKPOINT),
                "deployment_student_obs_fault_descriptor_free": True,
            },
            "downstream_usage": {
                "A2": "single-step student distillation from student_obs to teacher_action",
                "A2_history": "H x 61 history-based student distillation",
                "A5": "A2-history plus residual distillation",
                "A7": "A0-anchored residual variant if a0_action is explicitly collected",
            },
        }
        logger.set_stage("metadata.json write start")
        write_json(metadata_path, metadata)
        logger.log(f"metadata.json write end path={repo_relative(metadata_path)}")
        logger.set_stage("README write start")
        write_dataset_readme(output_dir / "README.md", metadata)
        write_dataset_readme(summary_path, metadata)
        logger.log(f"README write end path={repo_relative(output_dir / 'README.md')}")
        logger.log(f"dataset_summary.md write end path={repo_relative(summary_path)}")
        logger.set_stage("final success")
        logger.log(f"dataset: {repo_relative(dataset_path)}")
        logger.log(f"metadata: {repo_relative(metadata_path)}")
        logger.log(f"summary: {repo_relative(summary_path)}")
        logger.log("collection completed successfully")
        return 0
    finally:
        if simulation_app is not None:
            logger.set_stage("app/Isaac shutdown")
            simulation_app.close()
            logger.log("app/Isaac shutdown end")


def main() -> int:
    logger: ProgressLogger | None = None
    print("[T14-DATASET] stage=argument parse start", flush=True)
    try:
        args = parse_args()
        print("[T14-DATASET] stage=argument parse end", flush=True)
        if args.execute_collect:
            output_dir = resolve_repo_path(args.output_dir)
            logger = ProgressLogger(output_dir)
            logger.set_stage("argument parse")
            logger.log(f"argv={' '.join(sys.argv)}")
            logger.log("argument parse end")
            validate_args(args, logger)
            return execute_collect(args, logger)
        validate_args(args)
        if args.dry_run:
            print_preview(args)
            return 0
        raise T14DatasetError("internal error: neither --dry_run nor --execute_collect was selected.")
    except T14DatasetError as exc:
        if logger is not None:
            error_path = logger.write_error(exc)
            logger.log(f"error_summary={repo_relative(error_path)}")
        print(f"[T14-DATASET ERROR] {exc}", file=sys.stderr, flush=True)
        return 2
    except Exception as exc:
        if logger is not None:
            error_path = logger.write_error(exc)
            logger.log(f"error_summary={repo_relative(error_path)}")
        print(f"[T14-DATASET ERROR] {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 1
    finally:
        if logger is not None:
            logger.close()


if __name__ == "__main__":
    raise SystemExit(main())
