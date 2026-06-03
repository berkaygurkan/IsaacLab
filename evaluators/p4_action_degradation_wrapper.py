"""P4 torque-degradation action wrapper for the T09-E1 pilot.

This wrapper is evaluator-local on purpose. It does not modify task
registration, Isaac Lab event managers, checkpoint pointers, or training code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import torch


@dataclass(frozen=True)
class P4ActionMapping:
    """Resolved Ant action/joint mapping for P4 torque degradation."""

    action_term_name: str
    action_dim: int
    target_joint: str
    target_action_index: int
    joint_names: tuple[str, ...]
    torque_scale: float
    fault_onset_step: int


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


def resolve_p4_action_mapping(
    env: gym.Env,
    *,
    target_joint: str = "front_left_foot",
    torque_scale: float = 0.5,
    fault_onset_step: int = 50,
    expected_action_dim: int = 8,
) -> P4ActionMapping:
    """Resolve the selected Ant joint to an action index, failing before rollout."""
    action_manager = getattr(env.unwrapped, "action_manager", None)
    if action_manager is None:
        raise ValueError("P4 requires env.unwrapped.action_manager; none was found.")

    action_dim = int(getattr(action_manager, "total_action_dim", -1))
    if action_dim != expected_action_dim:
        raise ValueError(f"P4 expected action_dim={expected_action_dim}, got {action_dim}.")

    terms = getattr(action_manager, "_terms", None)
    if not isinstance(terms, dict) or len(terms) != 1:
        available = list(terms.keys()) if isinstance(terms, dict) else "n/a"
        raise ValueError(f"P4 requires exactly one action term, got {available}.")

    action_term_name, action_term = next(iter(terms.items()))
    term_action_dim = int(getattr(action_term, "action_dim", -1))
    if term_action_dim != expected_action_dim:
        raise ValueError(
            f"P4 expected joint action term dim={expected_action_dim}, got {term_action_dim} for {action_term_name}."
        )

    joint_names = _joint_names_from_term(action_term)
    if not joint_names:
        raise ValueError(f"P4 could not read resolved joint names from action term {action_term_name}.")
    if len(joint_names) != expected_action_dim:
        raise ValueError(f"P4 expected {expected_action_dim} joint names, got {len(joint_names)}: {joint_names}.")

    matches = [index for index, name in enumerate(joint_names) if name == target_joint]
    if len(matches) != 1:
        raise ValueError(
            f"P4 target_joint={target_joint!r} must map to exactly one action index; "
            f"matches={matches}, joint_names={joint_names}."
        )

    if torque_scale < 0.0 or torque_scale > 1.0:
        raise ValueError(f"P4 torque_scale must be in [0, 1], got {torque_scale}.")
    if fault_onset_step < 0:
        raise ValueError(f"P4 fault_onset_step must be non-negative, got {fault_onset_step}.")

    return P4ActionMapping(
        action_term_name=action_term_name,
        action_dim=action_dim,
        target_joint=target_joint,
        target_action_index=matches[0],
        joint_names=tuple(joint_names),
        torque_scale=float(torque_scale),
        fault_onset_step=int(fault_onset_step),
    )


class P4ActionDegradationWrapper(gym.Wrapper):
    """Scale one selected Ant action dimension after the configured onset step."""

    def __init__(
        self,
        env: gym.Env,
        *,
        target_joint: str = "front_left_foot",
        torque_scale: float = 0.5,
        fault_onset_step: int = 50,
        expected_action_dim: int = 8,
        debug: bool = False,
    ) -> None:
        super().__init__(env)
        self.debug = debug
        self.mapping = resolve_p4_action_mapping(
            env,
            target_joint=target_joint,
            torque_scale=torque_scale,
            fault_onset_step=fault_onset_step,
            expected_action_dim=expected_action_dim,
        )
        self.step_count = 0
        self.last_fault_applied = False
        self.ever_fault_applied = False
        self.last_action_before_fault: torch.Tensor | None = None
        self.last_action_after_fault: torch.Tensor | None = None
        if self.debug:
            print(f"[T09-E1 P4] action mapping: {self.mapping}", flush=True)

    @property
    def fault_applied(self) -> bool:
        return self.ever_fault_applied

    def reset(self, **kwargs: Any):
        self.step_count = 0
        self.last_fault_applied = False
        self.ever_fault_applied = False
        self.last_action_before_fault = None
        self.last_action_after_fault = None
        return self.env.reset(**kwargs)

    def step(self, action):
        action_tensor = torch.as_tensor(action, device=self.env.unwrapped.device, dtype=torch.float32)
        if action_tensor.ndim != 2 or action_tensor.shape[1] != self.mapping.action_dim:
            raise ValueError(
                f"P4 expected action shape (num_envs, {self.mapping.action_dim}), got {tuple(action_tensor.shape)}."
            )
        if not torch.isfinite(action_tensor).all():
            raise RuntimeError("P4 received non-finite action before degradation.")

        degraded_action = action_tensor.clone()
        apply_fault = self.step_count >= self.mapping.fault_onset_step
        if apply_fault:
            degraded_action[:, self.mapping.target_action_index] *= self.mapping.torque_scale

        if not torch.isfinite(degraded_action).all():
            raise RuntimeError("P4 produced non-finite action after degradation.")

        self.last_fault_applied = apply_fault
        self.ever_fault_applied = self.ever_fault_applied or apply_fault
        self.last_action_before_fault = action_tensor.detach().clone()
        self.last_action_after_fault = degraded_action.detach().clone()
        self.step_count += 1

        obs, reward, terminated, truncated, extras = self.env.step(degraded_action)
        if isinstance(extras, dict):
            log_extras = extras.setdefault("log", {})
            log_extras["P4/fault_applied"] = torch.tensor(float(apply_fault), device=self.env.unwrapped.device)
            log_extras["P4/target_action_index"] = torch.tensor(
                float(self.mapping.target_action_index), device=self.env.unwrapped.device
            )
            log_extras["P4/torque_scale"] = torch.tensor(float(self.mapping.torque_scale), device=self.env.unwrapped.device)
        return obs, reward, terminated, truncated, extras
