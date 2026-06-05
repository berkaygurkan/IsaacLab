"""Repo-owned P2 joint-lock training hook.

This module provides two pieces:

- ``P2JointLockActionMaskWrapper``: a Gym wrapper that applies the T09 P2
  single-joint-lock surrogate by overriding one action dimension after onset.
- a tiny launcher shim that monkey-patches ``gymnasium.make`` before running the
  upstream Isaac Lab RSL-RL training script. This keeps the hook local to the
  thesis repo without editing Isaac Lab core, RSL-RL core, or task registration.

The current implementation is an action-override surrogate, not a true
mechanical position-hold joint lock.
"""

from __future__ import annotations

import argparse
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
    """Resolved Ant action/joint mapping for the P2 surrogate hook."""

    action_term_name: str
    action_dim: int
    target_joint: str
    target_action_index: int
    joint_names: tuple[str, ...]
    fault_onset_step: int
    locked_action_value: float
    semantics: str


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


def resolve_p2_action_mapping(
    env: Any,
    *,
    target_joint: str = "front_left_foot",
    fault_onset_step: int = 50,
    expected_action_dim: int = 8,
    locked_action_value: float = 0.0,
) -> P2JointLockMapping:
    """Resolve a target Ant joint to exactly one action index."""
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

    return P2JointLockMapping(
        action_term_name=action_term_name,
        action_dim=action_dim,
        target_joint=target_joint,
        target_action_index=matches[0],
        joint_names=tuple(joint_names),
        fault_onset_step=int(fault_onset_step),
        locked_action_value=float(locked_action_value),
        semantics="action_override_zero_effort_surrogate",
    )


class P2JointLockActionMaskWrapper(gym.Wrapper):
    """Override one selected action dimension after onset.

    This is a surrogate for a locked joint. It does not hold the physical joint
    position in the simulator. For the current Ant effort-action interface, it
    masks the selected joint command to ``locked_action_value`` after onset.
    """

    def __init__(
        self,
        env: Any,
        *,
        target_joint: str = "front_left_foot",
        fault_onset_step: int = 50,
        expected_action_dim: int = 8,
        locked_action_value: float = 0.0,
        debug: bool = False,
    ) -> None:
        super().__init__(env)
        self.debug = debug
        self.mapping = resolve_p2_action_mapping(
            env,
            target_joint=target_joint,
            fault_onset_step=fault_onset_step,
            expected_action_dim=expected_action_dim,
            locked_action_value=locked_action_value,
        )
        self.step_count = 0
        self.last_fault_applied = False
        self.ever_fault_applied = False
        self.last_action_before_fault = None
        self.last_action_after_fault = None
        if self.debug:
            print(f"[T09-R2f P2] action mapping: {self.mapping}", flush=True)
            print("P2_runtime_hook_enabled: True", flush=True)
            print(f"target_joint: {self.mapping.target_joint}", flush=True)
            print(f"fault_onset_step: {self.mapping.fault_onset_step}", flush=True)
            print("fault_profile: P2_locked_joint", flush=True)
            print(f"P2_semantics: {self.mapping.semantics}", flush=True)

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
        apply_fault = self.step_count >= self.mapping.fault_onset_step
        if apply_fault:
            locked_action[:, self.mapping.target_action_index] = self.mapping.locked_action_value

        if not torch.isfinite(locked_action).all():
            raise RuntimeError("P2 produced non-finite action after joint-lock surrogate.")

        self.last_fault_applied = apply_fault
        self.ever_fault_applied = self.ever_fault_applied or apply_fault
        self.last_action_before_fault = action_tensor.detach().clone()
        self.last_action_after_fault = locked_action.detach().clone()
        self.step_count += 1

        obs, reward, terminated, truncated, extras = self.env.step(locked_action)
        if isinstance(extras, dict):
            log_extras = extras.setdefault("log", {})
            log_extras["P2/fault_applied"] = torch.tensor(float(apply_fault), device=self.unwrapped.device)
            log_extras["P2/target_action_index"] = torch.tensor(
                float(self.mapping.target_action_index), device=self.unwrapped.device
            )
            log_extras["P2/locked_action_value"] = torch.tensor(
                float(self.mapping.locked_action_value), device=self.unwrapped.device
            )
        return obs, reward, terminated, truncated, extras


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run upstream RSL-RL training with the repo-owned P2 wrapper.")
    parser.add_argument("--p2_fault_config", default="configs/fault/joint_lock/p2_locked_joint.yaml")
    parser.add_argument("--p2_target_joint", default="front_left_foot")
    parser.add_argument("--p2_fault_onset_step", type=int, default=50)
    parser.add_argument("--p2_expected_action_dim", type=int, default=8)
    parser.add_argument("--p2_locked_action_value", type=float, default=0.0)
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
            expected_action_dim=args.p2_expected_action_dim,
            locked_action_value=args.p2_locked_action_value,
            debug=True,
        )
        print("[T09-R2f P2] gym.make hook attached", flush=True)
        print(f"  task: {task_name}", flush=True)
        print(f"  fault_config: {args.p2_fault_config}", flush=True)
        print("  no_checkpoint_pointer_update_from_hook: True", flush=True)
        return wrapped

    gym.make = make_with_p2_hook


def main() -> int:
    args, upstream_script, upstream_args = _split_wrapper_and_upstream_args(sys.argv[1:])
    fault_config = Path(args.p2_fault_config)
    if not fault_config.is_absolute():
        fault_config = REPO_ROOT / fault_config
    if not fault_config.is_file():
        raise SystemExit(f"P2 fault config not found: {fault_config}")
    if not upstream_script.is_file():
        raise SystemExit(f"Upstream train script not found: {upstream_script}")

    print("[T09-R2f P2] installing runtime training hook", flush=True)
    print("P2_runtime_hook_enabled: True", flush=True)
    print(f"target_joint: {args.p2_target_joint}", flush=True)
    print(f"fault_onset_step: {args.p2_fault_onset_step}", flush=True)
    print("fault_profile: P2_locked_joint", flush=True)
    print("P2_semantics: action_override_zero_effort_surrogate", flush=True)

    _install_gym_make_patch(args)
    sys.argv = [str(upstream_script)] + upstream_args
    runpy.run_path(str(upstream_script), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
