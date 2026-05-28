"""Smoke test for the T08-A residual action wrapper scaffold."""

from __future__ import annotations

import argparse
import faulthandler
import sys
from pathlib import Path

print("[T08-A SMOKE] smoke script start", flush=True)

from isaaclab.app import AppLauncher

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "train" / "residual.yaml"


def _strip_comment(value: str) -> str:
    return value.split("#", 1)[0].strip()


def _load_flat_yaml(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values

    for line in path.read_text(encoding="utf-8").splitlines():
        line = _strip_comment(line)
        if not line or ":" not in line or line.startswith(" "):
            continue
        key, value = line.split(":", 1)
        values[key.strip()] = value.strip().strip("\"'")
    return values


def _resolve_repo_path(path_value: str | Path) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def _repo_relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(resolved)


parser = argparse.ArgumentParser(description="Smoke-test the T08-A residual action wrapper.")
parser.add_argument("--config", default=str(DEFAULT_CONFIG.relative_to(REPO_ROOT)))
parser.add_argument("--task", default=None)
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--num_steps", type=int, default=2)
parser.add_argument("--student_checkpoint_pointer", default=None)
parser.add_argument("--residual_scale", type=float, default=None)
parser.add_argument("--final_action_clip", type=float, default=None)
parser.add_argument("--debug_timeout_sec", type=float, default=30.0)
parser.add_argument("--reset_hidden_on_done", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

faulthandler.enable(file=sys.stderr)
if args_cli.debug_timeout_sec > 0:
    faulthandler.dump_traceback_later(args_cli.debug_timeout_sec, repeat=True, file=sys.stderr)

config_path = _resolve_repo_path(args_cli.config)
if not config_path.is_file():
    raise FileNotFoundError(f"Residual config not found: {_repo_relative(config_path)}")
config = _load_flat_yaml(config_path)
print(f"[T08-A SMOKE] config loaded: {_repo_relative(config_path)}", flush=True)

print("[T08-A SMOKE] app launcher creation start", flush=True)
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
print("[T08-A SMOKE] app launcher created", flush=True)

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: E402,F401
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

from residual_action_wrapper import ResidualActionWrapper  # noqa: E402


def _optional_float(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def main() -> None:
    task = args_cli.task or config.get("task", "Isaac-Ant-Student-v0")
    num_envs = args_cli.num_envs if args_cli.num_envs is not None else 8
    residual_scale = args_cli.residual_scale
    if residual_scale is None:
        residual_scale = float(config.get("residual_scale", 0.1))
    final_action_clip = args_cli.final_action_clip
    if final_action_clip is None:
        final_action_clip = _optional_float(config.get("final_action_clip"))
    student_pointer = args_cli.student_checkpoint_pointer or config.get(
        "student_checkpoint_pointer", "checkpoints/rlm1_stripped/student/none/seed0/latest_checkpoint.yaml"
    )

    print("[T08-A SMOKE] env creation start", flush=True)
    env_cfg = parse_env_cfg(task, device=args_cli.device, num_envs=num_envs)
    env = gym.make(task, cfg=env_cfg)
    print("[T08-A SMOKE] env created", flush=True)

    print("[T08-A SMOKE] wrapper creation start", flush=True)
    wrapper = ResidualActionWrapper(
        env,
        student_checkpoint_pointer=student_pointer,
        residual_scale=residual_scale,
        final_action_clip=final_action_clip,
        device=env.unwrapped.device,
        reset_hidden_on_done=args_cli.reset_hidden_on_done,
        debug=True,
    )
    print("[T08-A SMOKE] wrapper created", flush=True)

    print("[T08-A SMOKE] wrapper reset start", flush=True)
    obs, _ = wrapper.reset()
    print("[T08-A SMOKE] wrapper reset done", flush=True)

    policy_terms = env.unwrapped.observation_manager.active_terms["policy"]
    if "true_fault_state" in policy_terms:
        raise AssertionError("policy group exposes true_fault_state")
    if obs["policy"].shape != (num_envs, 60):
        raise AssertionError(f"Expected policy obs shape {(num_envs, 60)}, got {tuple(obs['policy'].shape)}")
    if wrapper.action_dim != 8:
        raise AssertionError(f"Expected action dim 8, got {wrapper.action_dim}")

    print("[T08-A SMOKE] step loop start", flush=True)
    for step_idx in range(args_cli.num_steps):
        print(f"[T08-A SMOKE] step {step_idx + 1} start", flush=True)
        delta_action = torch.ones(num_envs, wrapper.action_dim, device=env.unwrapped.device)
        wrapper.step(delta_action)
        print(f"[T08-A SMOKE] step {step_idx + 1} done", flush=True)

        expected_final = wrapper.last_base_action + residual_scale * wrapper.last_bounded_delta_action
        if final_action_clip is not None:
            expected_final = torch.clamp(expected_final, -final_action_clip, final_action_clip)

        torch.testing.assert_close(wrapper.last_final_action, expected_final, rtol=1.0e-5, atol=1.0e-6)
        if residual_scale == 0.0 and final_action_clip is None:
            torch.testing.assert_close(wrapper.last_final_action, wrapper.last_base_action, rtol=1.0e-5, atol=1.0e-6)

        for name, tensor in {
            "base_action": wrapper.last_base_action,
            "delta_action": wrapper.last_delta_action,
            "bounded_delta_action": wrapper.last_bounded_delta_action,
            "final_action": wrapper.last_final_action,
        }.items():
            if tensor.shape != (num_envs, wrapper.action_dim):
                raise AssertionError(f"{name} has invalid shape: {tuple(tensor.shape)}")
            if not torch.isfinite(tensor).all():
                raise AssertionError(f"{name} contains non-finite values")

    print("[T08-A SMOKE] smoke success", flush=True)
    print("[INFO] T08-A residual wrapper smoke passed", flush=True)
    print(f"[INFO] task: {task}", flush=True)
    print(f"[INFO] num_envs: {num_envs}", flush=True)
    print(f"[INFO] num_steps: {args_cli.num_steps}", flush=True)
    print(f"[INFO] policy_dim: {wrapper.policy_dim}", flush=True)
    print(f"[INFO] action_dim: {wrapper.action_dim}", flush=True)
    print(f"[INFO] residual_scale: {residual_scale}", flush=True)
    print(f"[INFO] final_action_clip: {final_action_clip}", flush=True)
    print(f"[INFO] reset_hidden_on_done: {args_cli.reset_hidden_on_done}", flush=True)
    print(f"[INFO] student_pointer: {_repo_relative(wrapper.student_pointer_path)}", flush=True)
    print(f"[INFO] student_checkpoint: {_repo_relative(wrapper.student_checkpoint_path)}", flush=True)

    wrapper.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        faulthandler.cancel_dump_traceback_later()
        simulation_app.close()
