"""Repo-owned P2 joint-lock training hook.

This module provides two pieces:

- ``P2JointLockActionMaskWrapper``: a Gym wrapper that applies the T09 P2
  single-joint-lock attempt by capturing the target joint position at onset and
  forcing the simulated joint state back to that captured angle each step when
  Isaac Lab exposes the required articulation write API.
- a tiny launcher shim that monkey-patches ``gymnasium.make`` before running the
  upstream Isaac Lab RSL-RL training script. This keeps the hook local to the
  thesis repo without editing Isaac Lab core, RSL-RL core, or task registration.

The preferred implementation is a runtime simulation-state override, not a
permanent URDF/DOF asset modification. The PD position-hold effort surrogate is
retained only as an explicit fallback when allowed by the caller.
"""

from __future__ import annotations

import argparse
import importlib.util
import runpy
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import gymnasium as gym


REPO_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_TRAIN = REPO_ROOT / "scripts" / "reinforcement_learning" / "rsl_rl" / "train.py"


@dataclass(frozen=True)
class P2JointLockMapping:
    """Resolved Ant action/joint mapping for the P2 hook."""

    action_term_name: str
    action_dim: int
    target_joint: str
    target_action_index: int
    target_joint_id: int
    joint_names: tuple[str, ...]
    fault_onset_step: int
    fault_onset_mode: str
    fault_onset_step_min: int
    fault_onset_step_max: int
    locked_action_value: float
    kp: float
    kd: float
    action_clip: float
    velocity_override: float
    requested_semantics: str
    semantics: str
    fallback_semantics: str
    allow_fallback: bool
    robot_articulation_found: bool
    joint_position_readable: bool
    joint_velocity_readable: bool
    joint_state_write_api_found: bool
    fallback_reason: str


def _joint_names_from_term(term: Any) -> list[str] | None:
    joint_names = getattr(term, "_joint_names", None)
    if joint_names is not None:
        return list(joint_names)

    descriptor = getattr(term, "IO_descriptor", None)
    if descriptor is None:
        return None
    descriptor_joint_names = getattr(descriptor, "joint_names", None)
    if descriptor_joint_names is None:
        return None
    return list(descriptor_joint_names)


def _target_joint_id_from_term(term: Any, target_action_index: int) -> int | None:
    joint_ids = getattr(term, "_joint_ids", None)
    if isinstance(joint_ids, slice):
        if joint_ids == slice(None):
            return int(target_action_index)
        return None
    if joint_ids is None:
        return None
    try:
        return int(joint_ids[target_action_index])
    except (TypeError, IndexError, ValueError):
        return None


def _action_term_asset(term: Any) -> Any | None:
    return getattr(term, "_asset", None)


def _simulation_override_unavailable_reason(
    *,
    robot_articulation_found: bool,
    joint_position_readable: bool,
    joint_velocity_readable: bool,
    joint_state_write_api_found: bool,
) -> str:
    missing: list[str] = []
    if not robot_articulation_found:
        missing.append("robot articulation object not found")
    if not joint_position_readable:
        missing.append("joint_pos tensor not readable")
    if not joint_velocity_readable:
        missing.append("joint_vel tensor not readable")
    if not joint_state_write_api_found:
        missing.append("write_joint_state_to_sim API not found")
    return "; ".join(missing) if missing else "none"


def resolve_p2_action_mapping(
    env: Any,
    *,
    target_joint: str = "front_left_foot",
    fault_onset_step: int = 50,
    fault_onset_mode: str = "fixed",
    fault_onset_step_min: int = 30,
    fault_onset_step_max: int = 150,
    expected_action_dim: int = 8,
    locked_action_value: float = 0.0,
    kp: float = 4.0,
    kd: float = 0.4,
    action_clip: float = 1.0,
    velocity_override: float = 0.0,
    requested_semantics: str = "simulation_joint_state_override_lock",
    allow_fallback: bool = False,
) -> P2JointLockMapping:
    """Resolve a target Ant joint to exactly one action index."""
    if requested_semantics not in {"simulation_joint_state_override_lock", "pd_position_hold_surrogate"}:
        raise ValueError(
            "P2 requested_semantics must be 'simulation_joint_state_override_lock' "
            f"or 'pd_position_hold_surrogate', got {requested_semantics!r}."
        )
    if fault_onset_mode not in {"fixed", "random_uniform"}:
        raise ValueError(f"P2 fault_onset_mode must be 'fixed' or 'random_uniform', got {fault_onset_mode!r}.")
    if fault_onset_step_min < 0 or fault_onset_step_max < 0:
        raise ValueError(
            f"P2 random onset bounds must be non-negative, got min={fault_onset_step_min}, "
            f"max={fault_onset_step_max}."
        )
    if fault_onset_step_min > fault_onset_step_max:
        raise ValueError(
            f"P2 random onset min must be <= max, got min={fault_onset_step_min}, max={fault_onset_step_max}."
        )

    action_manager = getattr(env.unwrapped, "action_manager", None)
    if action_manager is None:
        raise ValueError("P2 requires env.unwrapped.action_manager; none was found.")

    action_dim = int(getattr(action_manager, "total_action_dim", -1))
    if action_dim != expected_action_dim:
        raise ValueError(f"P2 expected action_dim={expected_action_dim}, got {action_dim}.")

    terms = getattr(action_manager, "_terms", None)
    if not isinstance(terms, dict) or len(terms) != 1:
        available = list(terms.keys()) if isinstance(terms, dict) else "n/a"
        raise ValueError(f"P2 requires exactly one action term, got {available}.")

    action_term_name, action_term = next(iter(terms.items()))
    term_action_dim = int(getattr(action_term, "action_dim", -1))
    if term_action_dim != expected_action_dim:
        raise ValueError(
            f"P2 expected joint action term dim={expected_action_dim}, got {term_action_dim} for {action_term_name}."
        )

    joint_names = _joint_names_from_term(action_term)
    if not joint_names:
        raise ValueError(f"P2 could not read resolved joint names from action term {action_term_name}.")
    if len(joint_names) != expected_action_dim:
        raise ValueError(f"P2 expected {expected_action_dim} joint names, got {len(joint_names)}: {joint_names}.")

    matches = [index for index, name in enumerate(joint_names) if name == target_joint]
    if len(matches) != 1:
        raise ValueError(
            f"P2 target_joint={target_joint!r} must map to exactly one action index; "
            f"matches={matches}, joint_names={joint_names}."
        )
    if fault_onset_step < 0:
        raise ValueError(f"P2 fault_onset_step must be non-negative, got {fault_onset_step}.")
    if kp <= 0.0:
        raise ValueError(f"P2 kp must be positive, got {kp}.")
    if kd < 0.0:
        raise ValueError(f"P2 kd must be non-negative, got {kd}.")
    if action_clip <= 0.0:
        raise ValueError(f"P2 action_clip must be positive, got {action_clip}.")

    target_joint_id = _target_joint_id_from_term(action_term, matches[0])
    if target_joint_id is None:
        raise ValueError(f"P2 could not resolve target joint id from action term {action_term_name}.")
    asset = _action_term_asset(action_term)
    data = getattr(asset, "data", None)
    robot_articulation_found = asset is not None
    joint_position_readable = bool(data is not None and getattr(data, "joint_pos", None) is not None)
    joint_velocity_readable = bool(data is not None and getattr(data, "joint_vel", None) is not None)
    joint_state_write_api_found = callable(getattr(asset, "write_joint_state_to_sim", None))
    simulation_override_available = (
        robot_articulation_found
        and joint_position_readable
        and joint_velocity_readable
        and joint_state_write_api_found
    )
    pd_available = robot_articulation_found and joint_position_readable and joint_velocity_readable
    fallback_reason = "none"

    if requested_semantics == "simulation_joint_state_override_lock":
        if simulation_override_available:
            actual_semantics = "simulation_joint_state_override_lock"
        elif allow_fallback and pd_available:
            actual_semantics = "pd_position_hold_surrogate"
            fallback_reason = _simulation_override_unavailable_reason(
                robot_articulation_found=robot_articulation_found,
                joint_position_readable=joint_position_readable,
                joint_velocity_readable=joint_velocity_readable,
                joint_state_write_api_found=joint_state_write_api_found,
            )
        else:
            reason = _simulation_override_unavailable_reason(
                robot_articulation_found=robot_articulation_found,
                joint_position_readable=joint_position_readable,
                joint_velocity_readable=joint_velocity_readable,
                joint_state_write_api_found=joint_state_write_api_found,
            )
            raise ValueError(
                "P2 requested simulation_joint_state_override_lock, but the required simulation-state "
                f"override API is unavailable ({reason}). Pass --p2_allow_fallback only to use the "
                "PD position-hold surrogate explicitly."
            )
    else:
        if not pd_available:
            reason = _simulation_override_unavailable_reason(
                robot_articulation_found=robot_articulation_found,
                joint_position_readable=joint_position_readable,
                joint_velocity_readable=joint_velocity_readable,
                joint_state_write_api_found=joint_state_write_api_found,
            )
            raise ValueError(f"P2 requested pd_position_hold_surrogate, but joint state is unavailable ({reason}).")
        actual_semantics = "pd_position_hold_surrogate"

    return P2JointLockMapping(
        action_term_name=action_term_name,
        action_dim=action_dim,
        target_joint=target_joint,
        target_action_index=matches[0],
        target_joint_id=target_joint_id,
        joint_names=tuple(joint_names),
        fault_onset_step=int(fault_onset_step),
        fault_onset_mode=fault_onset_mode,
        fault_onset_step_min=int(fault_onset_step_min),
        fault_onset_step_max=int(fault_onset_step_max),
        locked_action_value=float(locked_action_value),
        kp=float(kp),
        kd=float(kd),
        action_clip=float(action_clip),
        velocity_override=float(velocity_override),
        requested_semantics=requested_semantics,
        semantics=actual_semantics,
        fallback_semantics="pd_position_hold_surrogate",
        allow_fallback=bool(allow_fallback),
        robot_articulation_found=robot_articulation_found,
        joint_position_readable=joint_position_readable,
        joint_velocity_readable=joint_velocity_readable,
        joint_state_write_api_found=joint_state_write_api_found,
        fallback_reason=fallback_reason,
    )


class P2JointLockActionMaskWrapper(gym.Wrapper):
    """Apply the P2 locked-joint attempt after onset.

    The preferred mode captures the target joint position at onset and writes
    that position plus near-zero velocity back into the articulation state after
    every environment step. The PD effort surrogate is available only as an
    explicit fallback.
    """

    def __init__(
        self,
        env: Any,
        *,
        target_joint: str = "front_left_foot",
        fault_onset_step: int = 50,
        fault_onset_mode: str = "fixed",
        fault_onset_step_min: int = 30,
        fault_onset_step_max: int = 150,
        expected_action_dim: int = 8,
        locked_action_value: float = 0.0,
        kp: float = 4.0,
        kd: float = 0.4,
        action_clip: float = 1.0,
        velocity_override: float = 0.0,
        requested_semantics: str = "simulation_joint_state_override_lock",
        allow_fallback: bool = False,
        debug: bool = False,
    ) -> None:
        super().__init__(env)
        self.debug = debug
        self.mapping = resolve_p2_action_mapping(
            env,
            target_joint=target_joint,
            fault_onset_step=fault_onset_step,
            fault_onset_mode=fault_onset_mode,
            fault_onset_step_min=fault_onset_step_min,
            fault_onset_step_max=fault_onset_step_max,
            expected_action_dim=expected_action_dim,
            locked_action_value=locked_action_value,
            kp=kp,
            kd=kd,
            action_clip=action_clip,
            velocity_override=velocity_override,
            requested_semantics=requested_semantics,
            allow_fallback=allow_fallback,
        )
        self.step_count = 0
        self.last_fault_applied = False
        self.last_fault_applied_mask = None
        self.ever_fault_applied = False
        self.per_env_step_count = None
        self.per_env_fault_onset_step = None
        self.fallback_used = self.mapping.semantics != self.mapping.requested_semantics
        self.lock_active = None
        self.lock_capture_position_available = False
        self.locked_joint_position = None
        self.last_joint_position_before_override = None
        self.last_joint_position_after_override = None
        self.last_joint_velocity_before_override = None
        self.last_joint_velocity_after_override = None
        self.last_pd_effort = None
        self.last_target_action_before = None
        self.last_target_action_after = None
        self.last_action_before_fault = None
        self.last_action_after_fault = None
        self.post_step_override_applied = False
        self.target_joint_velocity_after_override_abs_max = None
        self.fallback_reason = self.mapping.fallback_reason
        if self.debug:
            print(f"[T09-R2i P2] action mapping: {self.mapping}", flush=True)
            print("P2_runtime_hook_enabled: True", flush=True)
            print(f"requested_semantics: {self.mapping.requested_semantics}", flush=True)
            print(f"actual_semantics: {self.mapping.semantics}", flush=True)
            print(f"target_joint: {self.mapping.target_joint}", flush=True)
            print(f"target_joint_id: {self.mapping.target_joint_id}", flush=True)
            print(f"fault_onset_mode: {self.mapping.fault_onset_mode}", flush=True)
            print(f"fault_onset_step: {self.mapping.fault_onset_step}", flush=True)
            print(f"fault_onset_step_min: {self.mapping.fault_onset_step_min}", flush=True)
            print(f"fault_onset_step_max: {self.mapping.fault_onset_step_max}", flush=True)
            print(f"per_env_onset_randomization: {self.per_env_onset_randomization}", flush=True)
            print("fault_profile: P2_locked_joint", flush=True)
            print(f"velocity_override: {self.mapping.velocity_override}", flush=True)
            print(f"joint_state_write_api_found: {self.mapping.joint_state_write_api_found}", flush=True)
            print(f"fallback_used: {self.fallback_used}", flush=True)
            print(f"fallback_reason: {self.fallback_reason}", flush=True)

    @property
    def fault_applied(self) -> bool:
        return self.ever_fault_applied

    def reset(self, **kwargs: Any):
        self.step_count = 0
        self.last_fault_applied = False
        self.last_fault_applied_mask = None
        self.ever_fault_applied = False
        self.per_env_step_count = None
        self.per_env_fault_onset_step = None
        self.fallback_used = self.mapping.semantics != self.mapping.requested_semantics
        self.lock_active = None
        self.lock_capture_position_available = False
        self.locked_joint_position = None
        self.last_joint_position_before_override = None
        self.last_joint_position_after_override = None
        self.last_joint_velocity_before_override = None
        self.last_joint_velocity_after_override = None
        self.last_pd_effort = None
        self.last_target_action_before = None
        self.last_target_action_after = None
        self.last_action_before_fault = None
        self.last_action_after_fault = None
        self.post_step_override_applied = False
        self.target_joint_velocity_after_override_abs_max = None
        return self.env.reset(**kwargs)

    @property
    def per_env_onset_randomization(self) -> bool:
        return self.mapping.fault_onset_mode == "random_uniform"

    def _num_envs(self) -> int:
        num_envs = getattr(self.unwrapped, "num_envs", None)
        if num_envs is not None:
            return int(num_envs)
        asset = self._asset()
        data = getattr(asset, "data", None)
        joint_pos = getattr(data, "joint_pos", None)
        if joint_pos is None:
            raise RuntimeError("P2 cannot infer num_envs because robot joint_pos is unavailable.")
        return int(joint_pos.shape[0])

    def _device(self):
        device = getattr(self.unwrapped, "device", None)
        if device is not None:
            return device
        asset = self._asset()
        device = getattr(asset, "device", None)
        if device is not None:
            return device
        data = getattr(asset, "data", None)
        joint_pos = getattr(data, "joint_pos", None)
        if joint_pos is not None:
            return joint_pos.device
        return "cpu"

    def _ensure_lock_buffers(self) -> None:
        import torch

        if (
            self.lock_active is not None
            and self.locked_joint_position is not None
            and self.per_env_step_count is not None
            and self.per_env_fault_onset_step is not None
        ):
            return
        num_envs = self._num_envs()
        device = self._device()
        self.lock_active = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.locked_joint_position = torch.zeros(num_envs, dtype=torch.float32, device=device)
        self.per_env_step_count = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.per_env_fault_onset_step = torch.zeros(num_envs, dtype=torch.long, device=device)
        self._sample_onset_steps(torch.arange(num_envs, device=device))

    def _sample_onset_steps(self, env_ids) -> None:
        import torch

        if self.per_env_fault_onset_step is None:
            raise RuntimeError("P2 onset-step buffer was not initialized.")
        if env_ids.numel() == 0:
            return
        if self.mapping.fault_onset_mode == "random_uniform":
            sampled = torch.randint(
                low=self.mapping.fault_onset_step_min,
                high=self.mapping.fault_onset_step_max + 1,
                size=(env_ids.numel(),),
                device=env_ids.device,
                dtype=torch.long,
            )
            self.per_env_fault_onset_step[env_ids] = sampled
        else:
            self.per_env_fault_onset_step[env_ids] = int(self.mapping.fault_onset_step)

    def _fault_due_mask(self):
        self._ensure_lock_buffers()
        if self.per_env_step_count is None or self.per_env_fault_onset_step is None:
            raise RuntimeError("P2 onset buffers were not initialized.")
        return self.per_env_step_count >= self.per_env_fault_onset_step

    def _onset_stats(self) -> tuple[float, float, float]:
        self._ensure_lock_buffers()
        if self.per_env_fault_onset_step is None:
            raise RuntimeError("P2 onset-step buffer was not initialized.")
        onset = self.per_env_fault_onset_step.to(dtype=self.per_env_fault_onset_step.dtype)
        return (
            float(onset.float().mean().detach().cpu().item()),
            float(onset.min().detach().cpu().item()),
            float(onset.max().detach().cpu().item()),
        )

    def _asset(self) -> Any | None:
        terms = getattr(self.unwrapped.action_manager, "_terms", None)
        if not isinstance(terms, dict):
            return None
        term = terms.get(self.mapping.action_term_name)
        if term is None:
            return None
        return _action_term_asset(term)

    def _target_joint_position_velocity(self):
        asset = self._asset()
        data = getattr(asset, "data", None)
        if data is None:
            return None, None
        joint_pos = getattr(data, "joint_pos", None)
        joint_vel = getattr(data, "joint_vel", None)
        if joint_pos is None or joint_vel is None:
            return None, None
        return joint_pos[:, self.mapping.target_joint_id], joint_vel[:, self.mapping.target_joint_id]

    def _raw_action_from_effort(self, effort):
        import torch

        terms = getattr(self.unwrapped.action_manager, "_terms", None)
        term = terms.get(self.mapping.action_term_name) if isinstance(terms, dict) else None
        scale = getattr(term, "_scale", 1.0)
        offset = getattr(term, "_offset", 0.0)
        action_index = self.mapping.target_action_index

        if isinstance(scale, torch.Tensor):
            if scale.ndim == 0:
                scale_value = torch.full_like(effort, float(scale.item()))
            elif scale.ndim == 1:
                scale_value = torch.full_like(effort, float(scale[action_index].item()))
            else:
                scale_value = scale[:, action_index]
        else:
            scale_value = torch.full_like(effort, float(scale))
        if isinstance(offset, torch.Tensor):
            if offset.ndim == 0:
                offset_value = torch.full_like(effort, float(offset.item()))
            elif offset.ndim == 1:
                offset_value = torch.full_like(effort, float(offset[action_index].item()))
            else:
                offset_value = offset[:, action_index]
        else:
            offset_value = torch.full_like(effort, float(offset))
        if torch.any(torch.abs(scale_value) < 1.0e-8):
            raise RuntimeError("P2 cannot convert effort to raw action because action scale is near zero.")
        raw_action = (effort - offset_value) / scale_value
        return torch.clamp(raw_action, min=-self.mapping.action_clip, max=self.mapping.action_clip)

    def _capture_locked_position_if_needed(self, env_ids=None) -> None:
        import torch

        self._ensure_lock_buffers()
        if self.lock_active is None or self.locked_joint_position is None:
            raise RuntimeError("P2 lock buffers were not initialized.")
        if env_ids is None:
            candidate_env_ids = torch.arange(self._num_envs(), device=self._device())
        else:
            candidate_env_ids = env_ids
        inactive_mask = ~self.lock_active[candidate_env_ids]
        inactive_env_ids = candidate_env_ids[torch.nonzero(inactive_mask, as_tuple=False).squeeze(-1)]
        if inactive_env_ids.numel() == 0:
            return
        joint_pos, _ = self._target_joint_position_velocity()
        if joint_pos is None:
            raise RuntimeError("P2 cannot capture locked joint position because joint_pos is unavailable.")
        self.locked_joint_position[inactive_env_ids] = joint_pos[inactive_env_ids].detach().clone()
        self.lock_active[inactive_env_ids] = True
        self.lock_capture_position_available = True

    def _apply_position_hold_or_fallback(self, locked_action, fault_env_ids):
        import torch

        self._capture_locked_position_if_needed(fault_env_ids)
        if self.lock_active is None or self.locked_joint_position is None:
            raise RuntimeError("P2 lock buffers were not initialized for PD fallback.")

        joint_pos, joint_vel = self._target_joint_position_velocity()
        if joint_pos is None or joint_vel is None:
            raise RuntimeError("P2 cannot apply PD position-hold surrogate because joint state is unavailable.")

        active_env_ids = torch.nonzero(self.lock_active, as_tuple=False).squeeze(-1)
        if active_env_ids.numel() == 0:
            return locked_action
        effort = self.mapping.kp * (self.locked_joint_position - joint_pos) - self.mapping.kd * joint_vel
        if not torch.isfinite(effort).all():
            raise RuntimeError("P2 PD position-hold surrogate produced non-finite effort.")
        raw_action = self._raw_action_from_effort(effort)
        locked_action[active_env_ids, self.mapping.target_action_index] = raw_action[active_env_ids]
        self.last_pd_effort = effort.detach().clone()
        self.fallback_used = self.mapping.semantics != self.mapping.requested_semantics
        return locked_action

    def _apply_simulation_joint_state_override(self) -> None:
        import torch

        fault_env_ids = torch.nonzero(self._fault_due_mask(), as_tuple=False).squeeze(-1)
        self._capture_locked_position_if_needed(fault_env_ids)
        if self.lock_active is None or self.locked_joint_position is None:
            raise RuntimeError("P2 lock buffers were not initialized for simulation-state override.")
        active_env_ids = torch.nonzero(self.lock_active, as_tuple=False).squeeze(-1)
        if active_env_ids.numel() == 0:
            return

        asset = self._asset()
        write_joint_state_to_sim = getattr(asset, "write_joint_state_to_sim", None)
        if not callable(write_joint_state_to_sim):
            raise RuntimeError("P2 simulation-state override requires robot.write_joint_state_to_sim.")
        joint_pos, joint_vel = self._target_joint_position_velocity()
        if joint_pos is None or joint_vel is None:
            raise RuntimeError("P2 simulation-state override requires readable joint_pos and joint_vel.")

        desired_pos = self.locked_joint_position[active_env_ids].unsqueeze(-1)
        desired_vel = torch.full_like(desired_pos, float(self.mapping.velocity_override))
        self.last_joint_position_before_override = joint_pos[active_env_ids].detach().clone()
        self.last_joint_velocity_before_override = joint_vel[active_env_ids].detach().clone()
        write_joint_state_to_sim(
            desired_pos,
            desired_vel,
            joint_ids=[self.mapping.target_joint_id],
            env_ids=active_env_ids,
        )
        joint_pos_after, joint_vel_after = self._target_joint_position_velocity()
        if joint_pos_after is None or joint_vel_after is None:
            raise RuntimeError("P2 could not read joint state after simulation-state override.")
        self.last_joint_position_after_override = joint_pos_after[active_env_ids].detach().clone()
        self.last_joint_velocity_after_override = joint_vel_after[active_env_ids].detach().clone()
        self.target_joint_velocity_after_override_abs_max = float(
            torch.max(torch.abs(self.last_joint_velocity_after_override)).detach().cpu().item()
        )
        self.post_step_override_applied = True
        self.fallback_used = False

    def _clear_done_env_locks(self, terminated: Any, truncated: Any) -> None:
        import torch

        if self.lock_active is None:
            return
        done = self._done_tensor(terminated)
        truncated_done = self._done_tensor(truncated)
        if done is None and truncated_done is None:
            return
        if done is None:
            done = truncated_done
        elif truncated_done is not None:
            done = torch.logical_or(done, truncated_done)
        if done is None:
            return
        done_env_ids = torch.nonzero(done, as_tuple=False).squeeze(-1)
        if done_env_ids.numel() == 0:
            return
        self.lock_active[done_env_ids] = False
        if self.locked_joint_position is not None:
            self.locked_joint_position[done_env_ids] = 0.0
        if self.per_env_step_count is not None:
            self.per_env_step_count[done_env_ids] = 0
        self._sample_onset_steps(done_env_ids)

    def _done_tensor(self, value: Any):
        import torch

        if value is None:
            return None
        if isinstance(value, dict):
            tensors = [self._done_tensor(item) for item in value.values()]
            tensors = [item for item in tensors if item is not None]
            if not tensors:
                return None
            result = tensors[0]
            for item in tensors[1:]:
                result = torch.logical_or(result, item)
            return result
        tensor = torch.as_tensor(value, device=self._device())
        if tensor.ndim == 0:
            tensor = tensor.repeat(self._num_envs())
        if tensor.ndim > 1:
            tensor = tensor.reshape(tensor.shape[0], -1).any(dim=1)
        return tensor.to(dtype=torch.bool)

    def step(self, action: Any):
        import torch

        action_tensor = torch.as_tensor(action, device=self.unwrapped.device, dtype=torch.float32)
        if action_tensor.ndim != 2 or action_tensor.shape[1] != self.mapping.action_dim:
            raise ValueError(
                f"P2 expected action shape (num_envs, {self.mapping.action_dim}), got {tuple(action_tensor.shape)}."
            )
        if not torch.isfinite(action_tensor).all():
            raise RuntimeError("P2 received non-finite action before joint-lock surrogate.")

        locked_action = action_tensor.clone()
        fault_due_mask = self._fault_due_mask()
        fault_env_ids = torch.nonzero(fault_due_mask, as_tuple=False).squeeze(-1)
        apply_fault = bool(fault_due_mask.any().item())
        if apply_fault and self.mapping.semantics == "pd_position_hold_surrogate":
            locked_action = self._apply_position_hold_or_fallback(locked_action, fault_env_ids)
        elif apply_fault and self.mapping.semantics == "simulation_joint_state_override_lock":
            self._capture_locked_position_if_needed(fault_env_ids)

        if not torch.isfinite(locked_action).all():
            raise RuntimeError("P2 produced non-finite action after joint-lock surrogate.")

        self.last_fault_applied = apply_fault
        self.last_fault_applied_mask = fault_due_mask.detach().clone()
        self.ever_fault_applied = self.ever_fault_applied or apply_fault
        self.last_target_action_before = action_tensor[:, self.mapping.target_action_index].detach().clone()
        self.last_target_action_after = locked_action[:, self.mapping.target_action_index].detach().clone()
        self.last_action_before_fault = action_tensor.detach().clone()
        self.last_action_after_fault = locked_action.detach().clone()
        self.step_count += 1
        if self.per_env_step_count is not None:
            self.per_env_step_count += 1

        obs, reward, terminated, truncated, extras = self.env.step(locked_action)
        if apply_fault and self.mapping.semantics == "simulation_joint_state_override_lock":
            self._clear_done_env_locks(terminated, truncated)
            self._apply_simulation_joint_state_override()
        else:
            self._clear_done_env_locks(terminated, truncated)
        if isinstance(extras, dict):
            log_extras = extras.setdefault("log", {})
            onset_mean, onset_min, onset_max = self._onset_stats()
            log_extras["P2/fault_applied"] = fault_due_mask.float().mean()
            log_extras["P2/target_action_index"] = torch.tensor(
                float(self.mapping.target_action_index), device=self.unwrapped.device
            )
            log_extras["P2/locked_action_value"] = torch.tensor(
                float(self.mapping.locked_action_value), device=self.unwrapped.device
            )
            log_extras["P2/fallback_used"] = torch.tensor(float(self.fallback_used), device=self.unwrapped.device)
            log_extras["P2/simulation_override_applied"] = torch.tensor(
                float(self.post_step_override_applied), device=self.unwrapped.device
            )
            log_extras["P2/onset_step_mean"] = torch.tensor(onset_mean, device=self.unwrapped.device)
            log_extras["P2/onset_step_min"] = torch.tensor(onset_min, device=self.unwrapped.device)
            log_extras["P2/onset_step_max"] = torch.tensor(onset_max, device=self.unwrapped.device)
            log_extras["P2/per_env_onset_randomization"] = torch.tensor(
                float(self.per_env_onset_randomization), device=self.unwrapped.device
            )
        return obs, reward, terminated, truncated, extras


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run upstream RSL-RL training with the repo-owned P2 wrapper.")
    parser.add_argument("--p2_fault_config", default="configs/fault/joint_lock/p2_locked_joint.yaml")
    parser.add_argument("--p2_target_joint", default="front_left_foot")
    parser.add_argument("--p2_fault_onset_step", type=int, default=50)
    parser.add_argument("--p2_fault_onset_mode", default="fixed", choices=("fixed", "random_uniform"))
    parser.add_argument("--p2_fault_onset_step_min", type=int, default=30)
    parser.add_argument("--p2_fault_onset_step_max", type=int, default=150)
    parser.add_argument("--p2_expected_action_dim", type=int, default=8)
    parser.add_argument("--p2_locked_action_value", type=float, default=0.0)
    parser.add_argument("--p2_kp", type=float, default=4.0)
    parser.add_argument("--p2_kd", type=float, default=0.4)
    parser.add_argument("--p2_action_clip", type=float, default=1.0)
    parser.add_argument(
        "--p2_requested_semantics",
        default="simulation_joint_state_override_lock",
        choices=("simulation_joint_state_override_lock", "pd_position_hold_surrogate"),
    )
    parser.add_argument("--p2_allow_fallback", action="store_true")
    parser.add_argument("--p2_velocity_override", type=float, default=0.0)
    parser.add_argument("--p2_task", default="Isaac-Ant-Teacher-v0")
    parser.add_argument("--p2_debug", action="store_true")
    return parser


def _split_wrapper_and_upstream_args(argv: list[str]) -> tuple[argparse.Namespace, Path, list[str]]:
    if "--" not in argv:
        raise SystemExit("P2 wrapper launcher requires '-- <upstream_train.py> <upstream args...>'.")
    split_index = argv.index("--")
    wrapper_args = argv[:split_index]
    upstream = argv[split_index + 1 :]
    if not upstream:
        upstream_script = UPSTREAM_TRAIN
        upstream_args: list[str] = []
    else:
        first = Path(upstream[0])
        if upstream[0].endswith(".py"):
            upstream_script = first if first.is_absolute() else REPO_ROOT / first
            upstream_args = upstream[1:]
        else:
            upstream_script = UPSTREAM_TRAIN
            upstream_args = upstream

    args = _build_parser().parse_args(wrapper_args)
    return args, upstream_script, upstream_args


def _install_gym_make_patch(args: argparse.Namespace) -> None:
    import gymnasium as gym

    original_make: Callable[..., Any] = gym.make

    def make_with_p2_hook(id: Any, *make_args: Any, **make_kwargs: Any) -> Any:
        env = original_make(id, *make_args, **make_kwargs)
        task_name = str(id)
        if task_name != args.p2_task:
            return env
        wrapped = P2JointLockActionMaskWrapper(
            env,
            target_joint=args.p2_target_joint,
            fault_onset_step=args.p2_fault_onset_step,
            fault_onset_mode=args.p2_fault_onset_mode,
            fault_onset_step_min=args.p2_fault_onset_step_min,
            fault_onset_step_max=args.p2_fault_onset_step_max,
            expected_action_dim=args.p2_expected_action_dim,
            locked_action_value=args.p2_locked_action_value,
            kp=args.p2_kp,
            kd=args.p2_kd,
            action_clip=args.p2_action_clip,
            velocity_override=args.p2_velocity_override,
            requested_semantics=args.p2_requested_semantics,
            allow_fallback=args.p2_allow_fallback,
            debug=True,
        )
        print("[T09-R2i P2] gym.make hook attached", flush=True)
        print(f"  task: {task_name}", flush=True)
        print(f"  fault_config: {args.p2_fault_config}", flush=True)
        print("  no_checkpoint_pointer_update_from_hook: True", flush=True)
        return wrapped

    gym.make = make_with_p2_hook


def _prepare_upstream_import_path(upstream_script: Path) -> None:
    upstream_dir = upstream_script.parent.resolve()
    expected_cli_args = upstream_dir / "cli_args.py"
    if not expected_cli_args.is_file():
        raise SystemExit(
            "Upstream RSL-RL sibling import is not resolvable: "
            f"expected cli_args.py at {expected_cli_args}"
        )
    upstream_dir_text = str(upstream_dir)
    if upstream_dir_text not in sys.path:
        sys.path.insert(0, upstream_dir_text)
    if importlib.util.find_spec("cli_args") is None:
        raise SystemExit(
            "Failed to resolve upstream cli_args module after adding script directory to sys.path: "
            f"{upstream_dir_text}"
        )
    print("[T09-R2i P2] upstream import path ready", flush=True)
    print(f"  upstream_script_dir: {upstream_dir_text}", flush=True)
    print(f"  cli_args_path: {expected_cli_args}", flush=True)


def main() -> int:
    args, upstream_script, upstream_args = _split_wrapper_and_upstream_args(sys.argv[1:])
    fault_config = Path(args.p2_fault_config)
    if not fault_config.is_absolute():
        fault_config = REPO_ROOT / fault_config
    if not fault_config.is_file():
        raise SystemExit(f"P2 fault config not found: {fault_config}")
    if not upstream_script.is_file():
        raise SystemExit(f"Upstream train script not found: {upstream_script}")

    print("[T09-R2i P2] installing runtime training hook", flush=True)
    print("P2_runtime_hook_enabled: True", flush=True)
    print(f"target_joint: {args.p2_target_joint}", flush=True)
    print(f"fault_onset_mode: {args.p2_fault_onset_mode}", flush=True)
    print(f"fault_onset_step: {args.p2_fault_onset_step}", flush=True)
    print(f"fault_onset_step_min: {args.p2_fault_onset_step_min}", flush=True)
    print(f"fault_onset_step_max: {args.p2_fault_onset_step_max}", flush=True)
    print(f"per_env_onset_randomization: {args.p2_fault_onset_mode == 'random_uniform'}", flush=True)
    print("fault_profile: P2_locked_joint", flush=True)
    print(f"requested_semantics: {args.p2_requested_semantics}", flush=True)
    print(f"allow_fallback: {args.p2_allow_fallback}", flush=True)
    print("fallback_semantics: pd_position_hold_surrogate", flush=True)
    print(f"velocity_override: {args.p2_velocity_override}", flush=True)
    print(f"P2_kp: {args.p2_kp}", flush=True)
    print(f"P2_kd: {args.p2_kd}", flush=True)
    print(f"P2_action_clip: {args.p2_action_clip}", flush=True)

    _install_gym_make_patch(args)
    _prepare_upstream_import_path(upstream_script)
    sys.argv = [str(upstream_script)] + upstream_args
    runpy.run_path(str(upstream_script), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
