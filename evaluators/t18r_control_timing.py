"""Opt-in control-timing helpers for the T18-R 50 Hz / H50 phase.

The base Ant task defaults are left untouched. T18-R callers pass explicit
timing flags, and these helpers apply them to the loaded env config before
``gym.make`` while recording enough metadata to audit the effective control
rate in downstream summaries.
"""

from __future__ import annotations

import argparse
from typing import Any


T18R_CONTROL_FREQUENCY_HZ = 50.0
T18R_CONTROL_DT_S = 0.02
T18R_SIM_DT_S = 0.01
T18R_DECIMATION = 2
T18R_HISTORY_LEN = 50
T18R_PG500_PHYSICS_FREQUENCY_HZ = 500.0
T18R_PG500_SIM_DT_S = 0.002
T18R_PG500_DECIMATION = 10
CONTROL_FREQUENCY_TOL = 1.0e-6


def add_control_timing_args(parser: argparse.ArgumentParser) -> None:
    """Add optional control-timing CLI args without changing existing defaults."""

    parser.add_argument(
        "--control_frequency_hz",
        type=float,
        default=None,
        help=(
            "Optional requested control frequency. For T18-R use 50; if sim_dt and "
            "decimation are omitted with 50, this applies sim_dt=0.01 and decimation=2."
        ),
    )
    parser.add_argument(
        "--t18r_pg500_timing",
        action="store_true",
        help=(
            "Apply the paper-grade T18-R-PG500 timing profile: sim_dt=0.002, "
            "decimation=10, 500 Hz physics, 50 Hz control."
        ),
    )
    parser.add_argument(
        "--sim_dt",
        type=float,
        default=None,
        help="Optional physics timestep override. T18-R canonical value is 0.01.",
    )
    parser.add_argument(
        "--decimation",
        type=int,
        default=None,
        help="Optional action/control decimation override. T18-R canonical value is 2.",
    )
    parser.add_argument(
        "--episode_length_s",
        type=float,
        default=None,
        help="Optional episode length override in seconds; omitted keeps task default.",
    )
    parser.add_argument(
        "--require_control_frequency_hz",
        type=float,
        default=None,
        help="Fail fast unless the final env timing matches this control frequency.",
    )
    parser.add_argument(
        "--require_t18r_pg500_timing",
        action="store_true",
        help=(
            "Fail fast unless the final timing is exactly the T18-R-PG500 profile: "
            "sim_dt=0.002, decimation=10, control_dt=0.02, control_frequency=50 Hz, physics=500 Hz."
        ),
    )


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _control_dt(sim_dt: float, decimation: int) -> float:
    return float(sim_dt) * float(decimation)


def _control_frequency(sim_dt: float, decimation: int) -> float:
    return 1.0 / _control_dt(sim_dt, decimation)


def _physics_frequency(sim_dt: float) -> float:
    return 1.0 / float(sim_dt)


def current_control_timing_metadata(env_cfg: Any) -> dict[str, Any]:
    """Return effective timing metadata from an env config without mutating it."""

    sim_dt = float(getattr(env_cfg.sim, "dt"))
    decimation = int(getattr(env_cfg, "decimation"))
    control_dt = _control_dt(sim_dt, decimation)
    return {
        "control_frequency_hz": _control_frequency(sim_dt, decimation),
        "control_dt_s": control_dt,
        "physics_frequency_hz": _physics_frequency(sim_dt),
        "sim_dt_s": sim_dt,
        "decimation": decimation,
        "episode_length_s": _optional_float(getattr(env_cfg, "episode_length_s", None)),
        "render_interval": _optional_int(getattr(env_cfg.sim, "render_interval", None)),
    }


def apply_control_timing_to_env_cfg(env_cfg: Any, args: argparse.Namespace) -> dict[str, Any]:
    """Apply opt-in control timing overrides and return audit metadata.

    Default behavior is intentionally no-op. Passing ``--control_frequency_hz 50``
    with no explicit ``--sim_dt``/``--decimation`` applies the canonical T18-R
    timing: 100 Hz physics, decimation 2, 50 Hz control.
    """

    original = current_control_timing_metadata(env_cfg)
    requested_freq = getattr(args, "control_frequency_hz", None)
    requested_sim_dt = getattr(args, "sim_dt", None)
    requested_decimation = getattr(args, "decimation", None)
    requested_episode_length_s = getattr(args, "episode_length_s", None)
    required_freq = getattr(args, "require_control_frequency_hz", None)
    pg500_requested = bool(getattr(args, "t18r_pg500_timing", False))
    pg500_required = bool(getattr(args, "require_t18r_pg500_timing", False))

    final_sim_dt = original["sim_dt_s"]
    final_decimation = original["decimation"]
    source = "task_default"

    if pg500_requested:
        if requested_freq is not None and abs(float(requested_freq) - T18R_CONTROL_FREQUENCY_HZ) > CONTROL_FREQUENCY_TOL:
            raise ValueError("--t18r_pg500_timing requires --control_frequency_hz 50 when that flag is provided.")
        if requested_sim_dt is not None and abs(float(requested_sim_dt) - T18R_PG500_SIM_DT_S) > CONTROL_FREQUENCY_TOL:
            raise ValueError("--t18r_pg500_timing requires --sim_dt 0.002 when that flag is provided.")
        if requested_decimation is not None and int(requested_decimation) != T18R_PG500_DECIMATION:
            raise ValueError("--t18r_pg500_timing requires --decimation 10 when that flag is provided.")
        final_sim_dt = T18R_PG500_SIM_DT_S
        final_decimation = T18R_PG500_DECIMATION
        source = "t18r_pg500_50hz_h50"
    elif requested_freq is not None and requested_sim_dt is None and requested_decimation is None:
        if abs(float(requested_freq) - T18R_CONTROL_FREQUENCY_HZ) > CONTROL_FREQUENCY_TOL:
            raise ValueError(
                "Non-canonical --control_frequency_hz requires explicit --sim_dt and --decimation."
            )
        final_sim_dt = T18R_SIM_DT_S
        final_decimation = T18R_DECIMATION
        source = "t18r_canonical_50hz_h50"
    else:
        if requested_sim_dt is not None:
            final_sim_dt = float(requested_sim_dt)
            source = "explicit_cli"
        if requested_decimation is not None:
            final_decimation = int(requested_decimation)
            source = "explicit_cli"
        if requested_freq is not None and requested_sim_dt is None and requested_decimation is not None:
            final_sim_dt = 1.0 / (float(requested_freq) * float(final_decimation))
            source = "derived_from_control_frequency_and_decimation"
        if requested_freq is not None and requested_sim_dt is not None and requested_decimation is None:
            final_decimation_float = 1.0 / (float(requested_freq) * float(final_sim_dt))
            final_decimation = int(round(final_decimation_float))
            if abs(final_decimation_float - final_decimation) > CONTROL_FREQUENCY_TOL:
                raise ValueError(
                    "--control_frequency_hz and --sim_dt do not imply an integer decimation; "
                    "provide --decimation explicitly."
                )
            source = "derived_from_control_frequency_and_sim_dt"

    if final_sim_dt <= 0.0:
        raise ValueError("--sim_dt must be positive.")
    if final_decimation <= 0:
        raise ValueError("--decimation must be positive.")

    env_cfg.sim.dt = float(final_sim_dt)
    env_cfg.decimation = int(final_decimation)
    if hasattr(env_cfg.sim, "render_interval"):
        env_cfg.sim.render_interval = int(final_decimation)
    if requested_episode_length_s is not None:
        if float(requested_episode_length_s) <= 0.0:
            raise ValueError("--episode_length_s must be positive.")
        env_cfg.episode_length_s = float(requested_episode_length_s)

    final = current_control_timing_metadata(env_cfg)
    if required_freq is not None and abs(final["control_frequency_hz"] - float(required_freq)) > 1.0e-4:
        raise ValueError(
            f"effective control frequency {final['control_frequency_hz']:.9g} Hz does not match "
            f"required {float(required_freq):.9g} Hz."
        )
    if pg500_required:
        validate_pg500_timing_metadata(final)

    final.update(
        {
            "control_timing_source": source,
            "control_timing_changed": final != original,
            "requested_control_frequency_hz": _optional_float(requested_freq),
            "required_control_frequency_hz": _optional_float(required_freq),
            "original_control_frequency_hz": original["control_frequency_hz"],
            "original_control_dt_s": original["control_dt_s"],
            "original_physics_frequency_hz": original["physics_frequency_hz"],
            "original_sim_dt_s": original["sim_dt_s"],
            "original_decimation": original["decimation"],
            "original_episode_length_s": original["episode_length_s"],
            "t18r_canonical_50hz_h50": bool(
                abs(final["control_frequency_hz"] - T18R_CONTROL_FREQUENCY_HZ) <= 1.0e-4
            ),
            "t18r_pg500_50hz_h50": timing_matches_pg500(final),
        }
    )
    return final


def timing_matches_pg500(metadata: dict[str, Any]) -> bool:
    return (
        abs(float(metadata.get("sim_dt_s")) - T18R_PG500_SIM_DT_S) <= 1.0e-9
        and int(metadata.get("decimation")) == T18R_PG500_DECIMATION
        and abs(float(metadata.get("control_dt_s")) - T18R_CONTROL_DT_S) <= 1.0e-9
        and abs(float(metadata.get("control_frequency_hz")) - T18R_CONTROL_FREQUENCY_HZ) <= 1.0e-6
        and abs(float(metadata.get("physics_frequency_hz")) - T18R_PG500_PHYSICS_FREQUENCY_HZ) <= 1.0e-6
    )


def validate_pg500_timing_metadata(metadata: dict[str, Any]) -> None:
    if not timing_matches_pg500(metadata):
        raise ValueError(
            "T18-R-PG500 timing guard failed: expected sim_dt=0.002, decimation=10, "
            "control_dt=0.02, control_frequency_hz=50, physics_frequency_hz=500; "
            f"got sim_dt={metadata.get('sim_dt_s')}, decimation={metadata.get('decimation')}, "
            f"control_dt={metadata.get('control_dt_s')}, "
            f"control_frequency={metadata.get('control_frequency_hz')}, "
            f"physics_frequency={metadata.get('physics_frequency_hz')}."
        )


def validate_pg500_history_len(history_len: int) -> None:
    if int(history_len) != T18R_HISTORY_LEN:
        raise ValueError(f"T18-R-PG500 H50 guard failed: expected history_len=50, got {history_len}.")
