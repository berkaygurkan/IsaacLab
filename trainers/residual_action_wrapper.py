"""Residual action wrapper scaffold for T08-A."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import torch
from rsl_rl.models import RNNModel
from tensordict import TensorDict


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STUDENT_POINTER = "checkpoints/rlm1_stripped/student/none/seed0/latest_checkpoint.yaml"


def _log(message: str, *, enabled: bool = True) -> None:
    if enabled:
        print(message, flush=True)


def _strip_comment(value: str) -> str:
    return value.split("#", 1)[0].strip()


def load_flat_yaml(path: Path) -> dict[str, str]:
    """Load a tiny flat YAML file used by conference-stage launchers."""
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


def resolve_repo_path(path_value: str | Path) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def repo_relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(resolved)


def resolve_checkpoint_from_pointer(pointer_value: str | Path) -> tuple[Path, Path]:
    """Resolve a stable repo checkpoint pointer into the concrete checkpoint file."""
    pointer_path = resolve_repo_path(pointer_value)
    if not pointer_path.is_file():
        raise FileNotFoundError(f"Student checkpoint pointer not found: {repo_relative(pointer_path)}")

    pointer = load_flat_yaml(pointer_path)
    checkpoint_value = pointer.get("checkpoint_path")
    if not checkpoint_value:
        raise ValueError(f"Student checkpoint pointer missing checkpoint_path: {repo_relative(pointer_path)}")

    checkpoint_path = resolve_repo_path(checkpoint_value)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Student checkpoint not found: {repo_relative(checkpoint_path)}")
    return pointer_path, checkpoint_path


def _dim_width(group_dim: tuple[int, ...] | list[tuple[int, ...]]) -> int:
    if isinstance(group_dim, tuple) and len(group_dim) == 1:
        return group_dim[0]
    raise ValueError(f"Expected a concatenated 1D observation group, got dimension metadata: {group_dim}")


def _action_dim(env: gym.Env) -> int:
    if hasattr(env.unwrapped, "action_manager"):
        return env.unwrapped.action_manager.total_action_dim
    return gym.spaces.flatdim(env.unwrapped.single_action_space)


def _shape_of(value: Any) -> tuple[int, ...] | str:
    shape = getattr(value, "shape", None)
    if shape is None:
        return "n/a"
    return tuple(shape)


def _tensor_stats(tensor: torch.Tensor) -> str:
    detached = tensor.detach()
    finite = torch.isfinite(detached)
    finite_values = detached[finite]
    finite_stats = "no finite values"
    if finite_values.numel() > 0:
        finite_stats = (
            f"min={finite_values.min().item():.6g}, "
            f"max={finite_values.max().item():.6g}, "
            f"mean={finite_values.mean().item():.6g}"
        )
    return (
        f"shape={tuple(detached.shape)}, dtype={detached.dtype}, device={detached.device}, "
        f"nan={torch.isnan(detached).sum().item()}, "
        f"posinf={torch.isposinf(detached).sum().item()}, "
        f"neginf={torch.isneginf(detached).sum().item()}, {finite_stats}"
    )


def _validate_finite(name: str, tensor: torch.Tensor) -> None:
    if not torch.isfinite(tensor).all():
        message = f"[ERROR] T08.6 non-finite {name}: {_tensor_stats(tensor)}"
        print(message, flush=True)
        raise RuntimeError(message)


def _build_student_model(*, num_envs: int, policy_dim: int, action_dim: int, device: str) -> RNNModel:
    obs = TensorDict(
        {"policy": torch.zeros(num_envs, policy_dim, device=device)},
        batch_size=[num_envs],
        device=device,
    )
    model = RNNModel(
        obs=obs,
        obs_groups={"student": ["policy"]},
        obs_set="student",
        output_dim=action_dim,
        hidden_dims=[400, 200, 100],
        activation="elu",
        obs_normalization=False,
        distribution_cfg={"class_name": "GaussianDistribution", "init_std": 0.1, "std_type": "scalar"},
        rnn_type="lstm",
        rnn_hidden_dim=128,
        rnn_num_layers=1,
    )
    return model.to(device)


def _mask_hidden_state(hidden_state: torch.Tensor, dones: torch.Tensor, num_envs: int) -> None:
    if hidden_state.shape[-2] != num_envs:
        raise ValueError(
            "Invalid student hidden-state shape for done masking: "
            f"hidden_state={tuple(hidden_state.shape)}, num_envs={num_envs}"
        )
    keep = (~dones.to(device=hidden_state.device, dtype=torch.bool)).to(dtype=hidden_state.dtype).view(1, num_envs, 1)
    hidden_state.mul_(keep)


class ResidualActionWrapper(gym.Wrapper):
    """Compose frozen T07 student actions with externally supplied residual deltas."""

    def __init__(
        self,
        env: gym.Env,
        *,
        student_checkpoint_pointer: str | Path = DEFAULT_STUDENT_POINTER,
        residual_scale: float = 0.1,
        final_action_clip: float | None = None,
        device: str | None = None,
        reset_hidden_on_done: bool = False,
        debug: bool = False,
    ) -> None:
        super().__init__(env)
        self.debug = debug
        self.device = device or getattr(env.unwrapped, "device", "cpu")
        self.residual_scale = float(residual_scale)
        self.final_action_clip = final_action_clip
        self.reset_hidden_on_done = reset_hidden_on_done

        self.num_envs = env.unwrapped.num_envs
        self.action_dim = _action_dim(env)
        self.policy_dim = self._validate_policy_contract()

        _log("[T08-A WRAPPER] student checkpoint pointer resolve start", enabled=self.debug)
        self.student_pointer_path, self.student_checkpoint_path = resolve_checkpoint_from_pointer(
            student_checkpoint_pointer
        )
        _log(
            f"[T08-A WRAPPER] student checkpoint pointer resolved: {repo_relative(self.student_pointer_path)}",
            enabled=self.debug,
        )
        _log(
            f"[T08-A WRAPPER] student checkpoint resolved: {repo_relative(self.student_checkpoint_path)}",
            enabled=self.debug,
        )
        _log("[T08-A WRAPPER] frozen student construction start", enabled=self.debug)
        self.student_model = _build_student_model(
            num_envs=self.num_envs,
            policy_dim=self.policy_dim,
            action_dim=self.action_dim,
            device=self.device,
        )
        _log("[T08-A WRAPPER] frozen student checkpoint load start", enabled=self.debug)
        checkpoint = torch.load(self.student_checkpoint_path, weights_only=False, map_location=self.device)
        if "student_state_dict" not in checkpoint:
            raise ValueError(f"Checkpoint has no student_state_dict: {repo_relative(self.student_checkpoint_path)}")
        self.student_model.load_state_dict(checkpoint["student_state_dict"], strict=True)
        self.student_model.eval()
        _log("[T08-A WRAPPER] frozen student loaded", enabled=self.debug)

        self._current_policy_obs: torch.Tensor | None = None
        self.last_base_action: torch.Tensor | None = None
        self.last_delta_action: torch.Tensor | None = None
        self.last_bounded_delta_action: torch.Tensor | None = None
        self.last_residual_action: torch.Tensor | None = None
        self.last_final_action: torch.Tensor | None = None
        self.last_mean_abs_delta: torch.Tensor | None = None
        self.last_max_abs_delta: torch.Tensor | None = None
        self.last_saturation_ratio: torch.Tensor | None = None
        self.last_clip_fraction: torch.Tensor | None = None

    def reset(self, **kwargs: Any):
        obs, extras = self.env.reset(**kwargs)
        self.student_model.reset()
        self._current_policy_obs = self._extract_policy_obs(obs)
        return obs, extras

    def step(self, delta_action: torch.Tensor):
        if self._current_policy_obs is None:
            self._current_policy_obs = self._extract_policy_obs(self.env.unwrapped.observation_manager.compute())

        delta_action = torch.as_tensor(delta_action, device=self.device, dtype=torch.float32)
        if delta_action.shape != (self.num_envs, self.action_dim):
            raise ValueError(f"Invalid residual action shape: {tuple(delta_action.shape)}")
        _validate_finite("delta_action", delta_action)

        base_action = self._infer_base_action(self._current_policy_obs)
        _log("[T08-A WRAPPER] base action computed", enabled=self.debug)
        bounded_delta_action = torch.tanh(delta_action)
        _validate_finite("bounded_delta", bounded_delta_action)
        residual_action = self.residual_scale * bounded_delta_action
        _validate_finite("residual_action", residual_action)
        final_action = base_action + residual_action
        clip_fraction = torch.zeros((), device=self.device, dtype=torch.float32)
        if self.final_action_clip is not None:
            clip_limit = float(self.final_action_clip)
            clip_fraction = (final_action.abs() > clip_limit).to(dtype=torch.float32).mean()
            final_action = torch.clamp(final_action, -self.final_action_clip, self.final_action_clip)
        _validate_finite("final_action", final_action)
        _log("[T08-A WRAPPER] final action composed", enabled=self.debug)

        mean_abs_delta = residual_action.abs().mean()
        max_abs_delta = residual_action.abs().max()
        saturation_ratio = (bounded_delta_action.abs() > 0.95).to(dtype=torch.float32).mean()

        obs, reward, terminated, truncated, extras = self.env.step(final_action)
        self._append_residual_diagnostics(extras, mean_abs_delta, max_abs_delta, saturation_ratio, clip_fraction)
        _log("[T08-A WRAPPER] env step returned", enabled=self.debug)
        _log("[T08-A WRAPPER] env step done", enabled=self.debug)
        _log(
            "[T08-A WRAPPER] terminated/truncated: "
            f"terminated={type(terminated).__name__} shape={_shape_of(terminated)}, "
            f"truncated={type(truncated).__name__} shape={_shape_of(truncated)}",
            enabled=self.debug,
        )
        _log("[T08-A WRAPPER] dones compute start", enabled=self.debug)
        dones = (terminated | truncated).to(device=self.device, dtype=torch.bool).reshape(-1)
        if dones.shape != (self.num_envs,):
            raise ValueError(f"Invalid dones shape for student RNN reset: {tuple(dones.shape)}")
        _log(f"[T08-A WRAPPER] dones computed: shape={tuple(dones.shape)} dtype={dones.dtype}", enabled=self.debug)
        if self.reset_hidden_on_done:
            _log("[T08-A WRAPPER] student hidden manual mask start", enabled=self.debug)
            self._mask_student_hidden_state(dones)
            _log("[T08-A WRAPPER] student hidden manual mask done", enabled=self.debug)
        else:
            _log("[T08-A WRAPPER] student hidden reset skipped", enabled=self.debug)
        _log("[T08-A WRAPPER] next policy cache start", enabled=self.debug)
        if isinstance(obs, dict):
            _log(f"[T08-A WRAPPER] obs keys after step: {list(obs.keys())}", enabled=self.debug)
        self._current_policy_obs = self._extract_policy_obs(obs)
        _log("[T08-A WRAPPER] next policy cache done", enabled=self.debug)

        self.last_base_action = base_action.detach().clone()
        self.last_delta_action = delta_action.detach().clone()
        self.last_bounded_delta_action = bounded_delta_action.detach().clone()
        self.last_residual_action = residual_action.detach().clone()
        self.last_final_action = final_action.detach().clone()
        self.last_mean_abs_delta = mean_abs_delta.detach().clone()
        self.last_max_abs_delta = max_abs_delta.detach().clone()
        self.last_saturation_ratio = saturation_ratio.detach().clone()
        self.last_clip_fraction = clip_fraction.detach().clone()
        _log("[T08-A WRAPPER] step return start", enabled=self.debug)
        return obs, reward, terminated, truncated, extras

    def _validate_policy_contract(self) -> int:
        observation_manager = self.env.unwrapped.observation_manager
        active_terms = observation_manager.active_terms
        group_dims = observation_manager.group_obs_dim

        if "policy" not in active_terms:
            raise ValueError("T08-A residual wrapper requires a 'policy' observation group.")
        if "true_fault_state" in active_terms["policy"]:
            raise ValueError("T08-A residual policy must not receive true_fault_state.")

        policy_dim = _dim_width(group_dims["policy"])
        if policy_dim != 60:
            raise ValueError(f"Expected T07 student policy dim 60, got {policy_dim}.")
        return policy_dim

    def _extract_policy_obs(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        if not isinstance(obs, dict):
            raise TypeError(f"Expected observation dictionary, got {type(obs).__name__}")
        if "policy" not in obs:
            raise ValueError(f"Observation dictionary missing 'policy'. Available groups: {list(obs.keys())}")
        policy_obs = obs["policy"].to(device=self.device, dtype=torch.float32)
        if policy_obs.shape != (self.num_envs, self.policy_dim):
            raise ValueError(f"Invalid policy observation shape: {tuple(policy_obs.shape)}")
        return policy_obs

    def _infer_base_action(self, policy_obs: torch.Tensor) -> torch.Tensor:
        obs_td = TensorDict({"policy": policy_obs}, batch_size=[self.num_envs], device=self.device)
        with torch.inference_mode():
            base_action = self.student_model(obs_td, stochastic_output=False)
        if base_action.shape != (self.num_envs, self.action_dim):
            raise ValueError(f"Invalid base action shape: {tuple(base_action.shape)}")
        return base_action

    def _mask_student_hidden_state(self, dones: torch.Tensor) -> None:
        hidden_state = self.student_model.get_hidden_state()
        if hidden_state is None:
            return

        with torch.no_grad():
            if isinstance(hidden_state, tuple):
                for state in hidden_state:
                    _mask_hidden_state(state, dones, self.num_envs)
            else:
                _mask_hidden_state(hidden_state, dones, self.num_envs)

    def _append_residual_diagnostics(
        self,
        extras: dict[str, Any],
        mean_abs_delta: torch.Tensor,
        max_abs_delta: torch.Tensor,
        saturation_ratio: torch.Tensor,
        clip_fraction: torch.Tensor,
    ) -> None:
        extras_key = "episode" if "episode" in extras else "log"
        log_extras = extras.setdefault(extras_key, {})
        log_extras["Residual/mean_abs_delta"] = mean_abs_delta.detach()
        log_extras["Residual/max_abs_delta"] = max_abs_delta.detach()
        log_extras["Residual/saturation_ratio"] = saturation_ratio.detach()
        log_extras["Residual/clip_fraction"] = clip_fraction.detach()
