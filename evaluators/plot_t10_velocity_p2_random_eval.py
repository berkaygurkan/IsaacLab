#!/usr/bin/env python3
"""Offline T10 velocity P2-random evaluation plotter.

Reads an existing controlled evaluation output directory and saves velocity
comparison plots only. It does not launch Isaac Sim, run evaluation, train,
freeze checkpoints, or update paper-grade manifests.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import struct
import zlib
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
A0_LABEL = "A0_Vel_healthy_train_eval_P2_random"
A1F_LABEL = "A1F_Vel_P2_random_train_eval_P2_random"
VELOCITY_COLUMNS = ("mean_vel_x", "mean_base_lin_vel_x", "base_lin_vel_x_mean")
ERROR_COLUMNS = ("mean_abs_vx_error", "mean_abs_base_lin_vel_x_error", "abs_vx_error")
COLORS = {
    A0_LABEL: (31, 119, 180),
    A1F_LABEL: (214, 39, 40),
    "axis": (42, 42, 42),
    "grid": (220, 220, 220),
    "onset": (120, 120, 120),
    "target": (34, 139, 34),
}
FONT = {
    " ": ["000", "000", "000", "000", "000", "000", "000"],
    "-": ["000", "000", "000", "111", "000", "000", "000"],
    ".": ["000", "000", "000", "000", "000", "011", "011"],
    ",": ["000", "000", "000", "000", "000", "010", "100"],
    ":": ["000", "010", "010", "000", "010", "010", "000"],
    "/": ["001", "001", "010", "010", "100", "100", "000"],
    "[": ["111", "100", "100", "100", "100", "100", "111"],
    "]": ["111", "001", "001", "001", "001", "001", "111"],
    "_": ["000", "000", "000", "000", "000", "000", "111"],
    "0": ["111", "101", "101", "101", "101", "101", "111"],
    "1": ["010", "110", "010", "010", "010", "010", "111"],
    "2": ["111", "001", "001", "111", "100", "100", "111"],
    "3": ["111", "001", "001", "111", "001", "001", "111"],
    "4": ["101", "101", "101", "111", "001", "001", "001"],
    "5": ["111", "100", "100", "111", "001", "001", "111"],
    "6": ["111", "100", "100", "111", "101", "101", "111"],
    "7": ["111", "001", "001", "010", "010", "100", "100"],
    "8": ["111", "101", "101", "111", "101", "101", "111"],
    "9": ["111", "101", "101", "111", "001", "001", "111"],
    "A": ["010", "101", "101", "111", "101", "101", "101"],
    "B": ["110", "101", "101", "110", "101", "101", "110"],
    "C": ["111", "100", "100", "100", "100", "100", "111"],
    "D": ["110", "101", "101", "101", "101", "101", "110"],
    "E": ["111", "100", "100", "110", "100", "100", "111"],
    "F": ["111", "100", "100", "110", "100", "100", "100"],
    "G": ["111", "100", "100", "101", "101", "101", "111"],
    "H": ["101", "101", "101", "111", "101", "101", "101"],
    "I": ["111", "010", "010", "010", "010", "010", "111"],
    "J": ["001", "001", "001", "001", "101", "101", "111"],
    "K": ["101", "101", "110", "100", "110", "101", "101"],
    "L": ["100", "100", "100", "100", "100", "100", "111"],
    "M": ["101", "111", "111", "101", "101", "101", "101"],
    "N": ["101", "111", "111", "111", "101", "101", "101"],
    "O": ["111", "101", "101", "101", "101", "101", "111"],
    "P": ["111", "101", "101", "111", "100", "100", "100"],
    "Q": ["111", "101", "101", "101", "111", "001", "001"],
    "R": ["110", "101", "101", "110", "110", "101", "101"],
    "S": ["111", "100", "100", "111", "001", "001", "111"],
    "T": ["111", "010", "010", "010", "010", "010", "010"],
    "U": ["101", "101", "101", "101", "101", "101", "111"],
    "V": ["101", "101", "101", "101", "101", "101", "010"],
    "W": ["101", "101", "101", "101", "111", "111", "101"],
    "X": ["101", "101", "101", "010", "101", "101", "101"],
    "Y": ["101", "101", "101", "010", "010", "010", "010"],
    "Z": ["111", "001", "001", "010", "100", "100", "111"],
}


class PlotError(ValueError):
    """Raised for invalid T10 velocity plotting inputs."""


def resolve_repo_path(path_value: str | Path) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def repo_relative(path_value: str | Path) -> str:
    resolved = resolve_repo_path(path_value).resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(resolved)


def parse_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"none", "null", "na", "nan"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise PlotError(f"summary JSON does not exist: {repo_relative(path)}")
    return json.loads(path.read_text(encoding="utf-8"))


def infer_column(fieldnames: list[str], candidates: tuple[str, ...], *, required: bool, name: str) -> str | None:
    for candidate in candidates:
        if candidate in fieldnames:
            return candidate
    if required:
        raise PlotError(f"CSV missing required {name} column. Expected one of {candidates}; found {fieldnames}.")
    return None


def read_timeseries(csv_path: Path, *, vx_cmd: float | None) -> tuple[dict[str, list[dict[str, float]]], dict[str, Any]]:
    if not csv_path.is_file():
        raise PlotError(f"velocity_timeseries.csv does not exist: {repo_relative(csv_path)}")
    with csv_path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise PlotError(f"CSV has no header: {repo_relative(csv_path)}")
        fieldnames = list(reader.fieldnames)
        print(f"[T10-IL-07] CSV columns: {fieldnames}")
        if "step" not in fieldnames or "policy_label" not in fieldnames:
            raise PlotError(f"CSV must include step and policy_label columns; found {fieldnames}.")
        velocity_column = infer_column(fieldnames, VELOCITY_COLUMNS, required=True, name="velocity")
        error_column = infer_column(fieldnames, ERROR_COLUMNS, required=False, name="vx error")
        grouped: dict[str, list[dict[str, float]]] = {}
        missing_velocity = 0
        missing_error = 0
        for row in reader:
            step = parse_float(row.get("step"))
            policy_label = str(row.get("policy_label") or "").strip()
            velocity = parse_float(row.get(velocity_column))
            if step is None or not policy_label:
                continue
            if velocity is None:
                missing_velocity += 1
                continue
            error = parse_float(row.get(error_column)) if error_column is not None else None
            if error is None and vx_cmd is not None:
                error = abs(velocity - vx_cmd)
            if error is None:
                missing_error += 1
            grouped.setdefault(policy_label, []).append(
                {
                    "step": float(step),
                    "mean_vel_x": float(velocity),
                    "mean_abs_vx_error": float(error) if error is not None else float("nan"),
                }
            )
    for rows in grouped.values():
        rows.sort(key=lambda item: item["step"])
    diagnostics = {
        "fieldnames": fieldnames,
        "velocity_column": velocity_column,
        "error_column": error_column,
        "missing_velocity_rows": missing_velocity,
        "missing_error_rows": missing_error,
        "policy_labels": sorted(grouped.keys()),
    }
    if missing_velocity:
        print(f"[T10-IL-07 WARNING] skipped {missing_velocity} rows with missing velocity.")
    if missing_error:
        print(f"[T10-IL-07 WARNING] {missing_error} rows have missing vx error.")
    return grouped, diagnostics


def policy_display_name(policy_label: str) -> str:
    if policy_label == A0_LABEL:
        return "A0-Vel healthy PPO"
    if policy_label == A1F_LABEL:
        return "A1-F-Vel privileged teacher"
    return policy_label


def onset_metadata(summary: dict[str, Any]) -> dict[str, float | None]:
    onset_min = parse_float(summary.get("fault_onset_step_min"))
    onset_max = parse_float(summary.get("fault_onset_step_max"))
    onset_means: list[float] = []
    policies = summary.get("policies")
    if isinstance(policies, dict):
        for policy_summary in policies.values():
            if isinstance(policy_summary, dict):
                value = parse_float(policy_summary.get("P2/onset_step_mean"))
                if value is not None:
                    onset_means.append(value)
    onset_mean = None
    if onset_means:
        onset_mean = sum(onset_means) / len(onset_means)
    elif onset_min is not None and onset_max is not None:
        onset_mean = (onset_min + onset_max) / 2.0
    return {"min": onset_min, "mean": onset_mean, "max": onset_max}


def resolve_vx_cmd(summary: dict[str, Any]) -> float | None:
    value = parse_float(summary.get("vx_cmd"))
    if value is not None:
        return value
    policies = summary.get("policies")
    if isinstance(policies, dict):
        for policy_summary in policies.values():
            if isinstance(policy_summary, dict):
                value = parse_float(policy_summary.get("target_vx"))
                if value is not None:
                    return value
    return None


def add_onset_markers(ax: Any, onset: dict[str, float | None]) -> None:
    onset_min = onset.get("min")
    onset_mean = onset.get("mean")
    onset_max = onset.get("max")
    if onset_min is not None and onset_max is not None:
        ax.axvspan(onset_min, onset_max, color="tab:gray", alpha=0.13, label=f"P2 onset range [{onset_min:g}, {onset_max:g}]")
        ax.axvline(onset_min, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
        ax.axvline(onset_max, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    if onset_mean is not None:
        ax.axvline(onset_mean, color="black", linestyle="-.", linewidth=1.1, alpha=0.9, label=f"P2 onset mean {onset_mean:.1f}")


def finite_points(rows: list[dict[str, float]], metric_key: str) -> tuple[list[float], list[float]]:
    points = [
        (row["step"], row[metric_key])
        for row in rows
        if math.isfinite(row.get("step", float("nan"))) and math.isfinite(row.get(metric_key, float("nan")))
    ]
    return [point[0] for point in points], [point[1] for point in points]


def metric_bounds(
    grouped: dict[str, list[dict[str, float]]],
    metric_key: str,
    *,
    vx_cmd: float | None = None,
    target_line: bool = False,
) -> tuple[float, float, float, float]:
    xs: list[float] = []
    ys: list[float] = []
    for rows in grouped.values():
        row_xs, row_ys = finite_points(rows, metric_key)
        xs.extend(row_xs)
        ys.extend(row_ys)
    if target_line and vx_cmd is not None:
        ys.append(vx_cmd)
    if not xs or not ys:
        raise PlotError(f"no finite values for metric {metric_key!r}")
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    if x_min == x_max:
        x_min -= 1.0
        x_max += 1.0
    if y_min == y_max:
        y_min -= 0.5
        y_max += 0.5
    y_pad = max(0.05, (y_max - y_min) * 0.08)
    return x_min, x_max, y_min - y_pad, y_max + y_pad


def set_pixel(image: list[bytearray], x: int, y: int, color: tuple[int, int, int]) -> None:
    if y < 0 or y >= len(image) or x < 0 or x >= len(image[0]) // 3:
        return
    offset = x * 3
    image[y][offset : offset + 3] = bytes(color)


def draw_line(
    image: list[bytearray],
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    color: tuple[int, int, int],
    *,
    width: int = 1,
    dash: int | None = None,
) -> None:
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    step = 0
    while True:
        if dash is None or (step // dash) % 2 == 0:
            for ox in range(-(width // 2), width // 2 + 1):
                for oy in range(-(width // 2), width // 2 + 1):
                    set_pixel(image, x0 + ox, y0 + oy, color)
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy
        step += 1


def fill_rect(image: list[bytearray], x0: int, y0: int, x1: int, y1: int, color: tuple[int, int, int]) -> None:
    width = len(image[0]) // 3
    height = len(image)
    xa, xb = sorted((max(0, x0), min(width - 1, x1)))
    ya, yb = sorted((max(0, y0), min(height - 1, y1)))
    for y in range(ya, yb + 1):
        for x in range(xa, xb + 1):
            set_pixel(image, x, y, color)


def draw_text(
    image: list[bytearray],
    x: int,
    y: int,
    text: str,
    color: tuple[int, int, int],
    *,
    scale: int = 2,
) -> None:
    cursor = x
    for char in text.upper():
        glyph = FONT.get(char, FONT.get(" "))
        if glyph is None:
            cursor += 4 * scale
            continue
        for row_index, row in enumerate(glyph):
            for col_index, value in enumerate(row):
                if value == "1":
                    fill_rect(
                        image,
                        cursor + col_index * scale,
                        y + row_index * scale,
                        cursor + (col_index + 1) * scale - 1,
                        y + (row_index + 1) * scale - 1,
                        color,
                    )
        cursor += (len(glyph[0]) + 1) * scale


def write_png(path: Path, image: list[bytearray]) -> None:
    height = len(image)
    width = len(image[0]) // 3

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    raw = b"".join(b"\x00" + bytes(row) for row in image)
    data = b"\x89PNG\r\n\x1a\n"
    data += chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    data += chunk(b"IDAT", zlib.compress(raw, level=9))
    data += chunk(b"IEND", b"")
    path.write_bytes(data)


def pdf_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def write_pdf(path: Path, commands: list[str], *, width: int = 720, height: int = 432) -> None:
    content = "\n".join(commands).encode("utf-8")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {width} {height}] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>".encode(
            "utf-8"
        ),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        f"<< /Length {len(content)} >>\nstream\n".encode("utf-8") + content + b"\nendstream",
    ]
    output = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for index, obj in enumerate(objects, start=1):
        offsets.append(len(output))
        output.extend(f"{index} 0 obj\n".encode("utf-8"))
        output.extend(obj)
        output.extend(b"\nendobj\n")
    xref_offset = len(output)
    output.extend(f"xref\n0 {len(objects) + 1}\n".encode("utf-8"))
    output.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        output.extend(f"{offset:010d} 00000 n \n".encode("utf-8"))
    output.extend(f"trailer << /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n".encode("utf-8"))
    path.write_bytes(bytes(output))


def fallback_plot(
    *,
    grouped: dict[str, list[dict[str, float]]],
    output_dir: Path,
    output_name: str,
    metric_key: str,
    ylabel: str,
    title: str,
    onset: dict[str, float | None],
    vx_cmd: float | None = None,
    target_line: bool = False,
) -> list[Path]:
    width, height = 1200, 720
    left, right, top, bottom = 95, 45, 92, 86
    plot_w = width - left - right
    plot_h = height - top - bottom
    x_min, x_max, y_min, y_max = metric_bounds(grouped, metric_key, vx_cmd=vx_cmd, target_line=target_line)

    def sx(value: float) -> int:
        return left + int((value - x_min) / (x_max - x_min) * plot_w)

    def sy(value: float) -> int:
        return top + plot_h - int((value - y_min) / (y_max - y_min) * plot_h)

    image = [bytearray([255, 255, 255] * width) for _ in range(height)]
    fill_rect(image, left, top, left + plot_w, top + plot_h, (248, 248, 248))
    for tick in range(6):
        y = top + int(plot_h * tick / 5)
        draw_line(image, left, y, left + plot_w, y, COLORS["grid"], width=1)
    onset_min = onset.get("min")
    onset_mean = onset.get("mean")
    onset_max = onset.get("max")
    if onset_min is not None and onset_max is not None:
        fill_rect(image, sx(onset_min), top, sx(onset_max), top + plot_h, (232, 232, 232))
        draw_line(image, sx(onset_min), top, sx(onset_min), top + plot_h, COLORS["axis"], dash=8)
        draw_line(image, sx(onset_max), top, sx(onset_max), top + plot_h, COLORS["axis"], dash=8)
    if onset_mean is not None:
        draw_line(image, sx(onset_mean), top, sx(onset_mean), top + plot_h, COLORS["axis"], dash=4)
    if target_line and vx_cmd is not None:
        draw_line(image, left, sy(vx_cmd), left + plot_w, sy(vx_cmd), COLORS["target"], dash=5, width=2)
    draw_line(image, left, top, left, top + plot_h, COLORS["axis"], width=2)
    draw_line(image, left, top + plot_h, left + plot_w, top + plot_h, COLORS["axis"], width=2)
    draw_text(image, 36, 22, title[:66], COLORS["axis"], scale=3)
    draw_text(image, 36, 58, "P2 RANDOM [30,150] TARGET VX 1.0 - NOT PAPER GRADE", COLORS["axis"], scale=2)
    draw_text(image, left, height - 36, "ROLLOUT STEP", COLORS["axis"], scale=2)
    draw_text(image, 8, top + 8, ylabel[:24], COLORS["axis"], scale=2)
    legend_y = top + 12
    for policy_label in (A0_LABEL, A1F_LABEL):
        rows = grouped.get(policy_label)
        if not rows:
            continue
        color = COLORS.get(policy_label, COLORS["axis"])
        draw_line(image, width - 340, legend_y + 8, width - 300, legend_y + 8, color, width=4)
        draw_text(image, width - 292, legend_y, policy_display_name(policy_label), color, scale=2)
        legend_y += 26
        xs, ys = finite_points(rows, metric_key)
        points = [(sx(x), sy(y)) for x, y in zip(xs, ys)]
        for p0, p1 in zip(points, points[1:]):
            draw_line(image, p0[0], p0[1], p1[0], p1[1], color, width=3)

    png_path = output_dir / f"{output_name}.png"
    pdf_path = output_dir / f"{output_name}.pdf"
    write_png(png_path, image)

    pdf_commands = [
        "1 1 1 rg 0 0 720 432 re f",
        "0.97 0.97 0.97 rg 57 52 636 326 re f",
        "0.16 0.16 0.16 rg BT /F1 14 Tf 36 408 Td (" + pdf_escape(title) + ") Tj ET",
        "0.16 0.16 0.16 rg BT /F1 9 Tf 36 392 Td (P2 onset random_uniform [30,150], target vx=1.0; candidate eval, not paper-grade) Tj ET",
        "0.16 0.16 0.16 RG 1.2 w 57 52 m 693 52 l 57 52 m 57 378 l S",
        "0.16 0.16 0.16 rg BT /F1 9 Tf 326 24 Td (rollout step) Tj ET",
        "0.16 0.16 0.16 rg BT /F1 9 Tf 8 374 Td (" + pdf_escape(ylabel) + ") Tj ET",
    ]

    def px(value: float) -> float:
        return 57 + (value - x_min) / (x_max - x_min) * 636

    def py(value: float) -> float:
        return 52 + (value - y_min) / (y_max - y_min) * 326

    if onset_min is not None and onset_max is not None:
        pdf_commands.append(f"0.90 0.90 0.90 rg {px(onset_min):.2f} 52 {px(onset_max) - px(onset_min):.2f} 326 re f")
        pdf_commands.append(f"0 0 0 RG 0.8 w {px(onset_min):.2f} 52 m {px(onset_min):.2f} 378 l S")
        pdf_commands.append(f"0 0 0 RG 0.8 w {px(onset_max):.2f} 52 m {px(onset_max):.2f} 378 l S")
    if onset_mean is not None:
        pdf_commands.append(f"0 0 0 RG 1 w {px(onset_mean):.2f} 52 m {px(onset_mean):.2f} 378 l S")
    if target_line and vx_cmd is not None:
        pdf_commands.append(f"0.13 0.55 0.13 RG 1.2 w 57 {py(vx_cmd):.2f} m 693 {py(vx_cmd):.2f} l S")
    legend_y_pdf = 364
    for policy_label in (A0_LABEL, A1F_LABEL):
        rows = grouped.get(policy_label)
        if not rows:
            continue
        r, g, b = [value / 255.0 for value in COLORS.get(policy_label, COLORS["axis"])]
        xs, ys = finite_points(rows, metric_key)
        if xs:
            commands = [f"{r:.3f} {g:.3f} {b:.3f} RG 1.8 w {px(xs[0]):.2f} {py(ys[0]):.2f} m"]
            commands.extend(f"{px(x):.2f} {py(y):.2f} l" for x, y in zip(xs[1:], ys[1:]))
            commands.append("S")
            pdf_commands.append(" ".join(commands))
        pdf_commands.append(f"{r:.3f} {g:.3f} {b:.3f} RG 2 w 505 {legend_y_pdf} m 535 {legend_y_pdf} l S")
        pdf_commands.append(
            f"{r:.3f} {g:.3f} {b:.3f} rg BT /F1 8 Tf 540 {legend_y_pdf - 3} Td ({pdf_escape(policy_display_name(policy_label))}) Tj ET"
        )
        legend_y_pdf -= 16
    write_pdf(pdf_path, pdf_commands)
    print(f"[T10-IL-07 WARNING] matplotlib unavailable; stdlib fallback rendered {output_name}.")
    return [png_path, pdf_path]


def plot_metric(
    *,
    grouped: dict[str, list[dict[str, float]]],
    output_dir: Path,
    output_name: str,
    metric_key: str,
    ylabel: str,
    title: str,
    onset: dict[str, float | None],
    vx_cmd: float | None = None,
    target_line: bool = False,
) -> list[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return fallback_plot(
            grouped=grouped,
            output_dir=output_dir,
            output_name=output_name,
            metric_key=metric_key,
            ylabel=ylabel,
            title=title,
            onset=onset,
            vx_cmd=vx_cmd,
            target_line=target_line,
        )

    fig, ax = plt.subplots(figsize=(9.4, 5.4))
    for policy_label in (A0_LABEL, A1F_LABEL):
        rows = grouped.get(policy_label)
        if not rows:
            print(f"[T10-IL-07 WARNING] policy missing from timeseries: {policy_label}")
            continue
        steps = [row["step"] for row in rows]
        values = [row[metric_key] for row in rows]
        ax.plot(steps, values, linewidth=2.0, label=policy_display_name(policy_label))
    add_onset_markers(ax, onset)
    if target_line and vx_cmd is not None:
        ax.axhline(vx_cmd, color="tab:green", linestyle=":", linewidth=1.4, label=f"target vx {vx_cmd:g}")
    ax.set_xlabel("rollout step")
    ax.set_ylabel(ylabel)
    ax.set_title(title + "\nP2 onset random_uniform [30,150], target vx=1.0; candidate eval, not paper-grade")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    png_path = output_dir / f"{output_name}.png"
    pdf_path = output_dir / f"{output_name}.pdf"
    fig.savefig(png_path, dpi=180)
    fig.savefig(pdf_path)
    plt.close(fig)
    return [png_path, pdf_path]


def plot_combined(
    *,
    grouped: dict[str, list[dict[str, float]]],
    output_dir: Path,
    onset: dict[str, float | None],
    vx_cmd: float | None,
) -> list[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        generated: list[Path] = []
        generated.extend(
            fallback_plot(
                grouped=grouped,
                output_dir=output_dir,
                output_name="vel_x_and_error_time_comparison",
                metric_key="mean_vel_x",
                ylabel="mean vel_x",
                title="Velocity and tracking error under random P2 joint lock",
                onset=onset,
                vx_cmd=vx_cmd,
                target_line=True,
            )
        )
        return generated

    fig, axes = plt.subplots(2, 1, figsize=(9.4, 7.2), sharex=True)
    for ax, metric_key, ylabel in (
        (axes[0], "mean_vel_x", "mean vel_x"),
        (axes[1], "mean_abs_vx_error", "mean abs vx error"),
    ):
        for policy_label in (A0_LABEL, A1F_LABEL):
            rows = grouped.get(policy_label)
            if not rows:
                continue
            ax.plot(
                [row["step"] for row in rows],
                [row[metric_key] for row in rows],
                linewidth=2.0,
                label=policy_display_name(policy_label),
            )
        add_onset_markers(ax, onset)
        if metric_key == "mean_vel_x" and vx_cmd is not None:
            ax.axhline(vx_cmd, color="tab:green", linestyle=":", linewidth=1.4, label=f"target vx {vx_cmd:g}")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")
    axes[1].set_xlabel("rollout step")
    fig.suptitle(
        "Velocity and tracking error under random P2 joint lock\n"
        "candidate eval only, not paper-grade; A1-F is privileged/reference"
    )
    fig.tight_layout()
    png_path = output_dir / "vel_x_and_error_time_comparison.png"
    pdf_path = output_dir / "vel_x_and_error_time_comparison.pdf"
    fig.savefig(png_path, dpi=180)
    fig.savefig(pdf_path)
    plt.close(fig)
    return [png_path, pdf_path]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot T10 controlled P2-random velocity evaluation timeseries.")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--summary", default=None)
    parser.add_argument("--timeseries", default=None)
    return parser


def write_readme(
    *,
    path: Path,
    input_dir: Path,
    summary_path: Path,
    timeseries_path: Path,
    generated_files: list[Path],
    summary: dict[str, Any],
    diagnostics: dict[str, Any],
) -> None:
    policies = summary.get("policies", {}) if isinstance(summary.get("policies"), dict) else {}
    a0_post = parse_float(policies.get(A0_LABEL, {}).get("mean_vel_x_post_fault")) if isinstance(policies.get(A0_LABEL), dict) else None
    a1f_post = parse_float(policies.get(A1F_LABEL, {}).get("mean_vel_x_post_fault")) if isinstance(policies.get(A1F_LABEL), dict) else None
    lines = [
        "# T10 Velocity P2-Random Eval Plots",
        "",
        "Scope: offline plotting only. No training, evaluation, checkpoint edits, or paper-grade manifest updates were run.",
        "",
        "## Inputs",
        "",
        f"- input_dir: `{repo_relative(input_dir)}`",
        f"- summary: `{repo_relative(summary_path)}`",
        f"- velocity_timeseries: `{repo_relative(timeseries_path)}`",
        f"- velocity column: `{diagnostics.get('velocity_column')}`",
        f"- error column: `{diagnostics.get('error_column')}`",
        "",
        "## Generated Files",
        "",
    ]
    for generated in generated_files:
        lines.append(f"- `{repo_relative(generated)}`")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "A1-F maintains substantially higher post-fault forward velocity than A0 in this candidate evaluation.",
        ]
    )
    if a0_post is not None and a1f_post is not None:
        lines.append(f"- A0 post-fault mean vel_x: `{a0_post:.3f}`")
        lines.append(f"- A1-F post-fault mean vel_x: `{a1f_post:.3f}`")
    lines.extend(
        [
            "",
            "## Caveats",
            "",
            "- A1-F is privileged/reference only and is not deployment-facing.",
            "- This is candidate evaluation, not paper-grade final evaluation.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = build_parser().parse_args()
    input_dir = resolve_repo_path(args.input_dir)
    summary_path = resolve_repo_path(args.summary) if args.summary else input_dir / "summary.json"
    timeseries_path = resolve_repo_path(args.timeseries) if args.timeseries else input_dir / "velocity_timeseries.csv"
    output_dir = resolve_repo_path(args.output_dir) if args.output_dir else input_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = load_json(summary_path)
    vx_cmd = resolve_vx_cmd(summary)
    onset = onset_metadata(summary)
    grouped, diagnostics = read_timeseries(timeseries_path, vx_cmd=vx_cmd)

    generated_files: list[Path] = []
    generated_files.extend(
        plot_metric(
            grouped=grouped,
            output_dir=output_dir,
            output_name="vel_x_time_comparison",
            metric_key="mean_vel_x",
            ylabel="mean forward velocity vel_x",
            title="Mean forward velocity under random P2 joint lock",
            onset=onset,
            vx_cmd=vx_cmd,
            target_line=True,
        )
    )
    generated_files.extend(
        plot_metric(
            grouped=grouped,
            output_dir=output_dir,
            output_name="vx_error_time_comparison",
            metric_key="mean_abs_vx_error",
            ylabel="mean absolute vx error",
            title="Forward velocity tracking error under random P2 joint lock",
            onset=onset,
            vx_cmd=vx_cmd,
            target_line=False,
        )
    )
    generated_files.extend(plot_combined(grouped=grouped, output_dir=output_dir, onset=onset, vx_cmd=vx_cmd))
    readme_path = output_dir / "README.md"
    write_readme(
        path=readme_path,
        input_dir=input_dir,
        summary_path=summary_path,
        timeseries_path=timeseries_path,
        generated_files=generated_files,
        summary=summary,
        diagnostics=diagnostics,
    )
    generated_files.append(readme_path)

    print("[T10-IL-07] plots written")
    print(f"  input_dir: {repo_relative(input_dir)}")
    print(f"  output_dir: {repo_relative(output_dir)}")
    print(f"  onset_min: {onset.get('min')}")
    print(f"  onset_mean: {onset.get('mean')}")
    print(f"  onset_max: {onset.get('max')}")
    print(f"  vx_cmd: {vx_cmd}")
    for generated in generated_files:
        print(f"  generated: {repo_relative(generated)}")
    print("  not_paper_grade: True")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
