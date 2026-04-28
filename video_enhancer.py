"""Video perturbations implemented with ffmpeg filters/codecs."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any


class ProcessingError(RuntimeError):
    """Raised when ffmpeg or ffprobe fails."""


def apply_video_perturbation(
    input_path: str | Path,
    output_path: str | Path,
    params: dict[str, Any],
    ffmpeg: str = "ffmpeg",
    ffprobe: str = "ffprobe",
) -> str:
    """Apply one video perturbation and copy the existing audio stream."""

    input_file = Path(input_path)
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    operation = params.get("operation")
    if operation == "video_compression":
        cmd = _base_cmd(ffmpeg, input_file) + [
            "-c:v",
            "libx264",
            "-preset",
            str(params.get("preset", "medium")),
            "-crf",
            str(params["crf"]),
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "copy",
            "-movflags",
            "+faststart",
            str(output_file),
        ]
    elif operation == "resize_down_up":
        width, height = probe_video_dimensions(input_file, ffprobe)
        scale = float(params["scale"])
        down_w = _even_dimension(width * scale)
        down_h = _even_dimension(height * scale)
        filter_expr = (
            f"scale={down_w}:{down_h}:flags=bicubic,"
            f"scale={width}:{height}:flags=bicubic"
        )
        cmd = _filtered_video_cmd(ffmpeg, input_file, output_file, filter_expr, params)
    elif operation == "gaussian_blur":
        sigma = float(params["sigma"])
        filter_expr = f"gblur=sigma={sigma:.4f}:steps=1"
        cmd = _filtered_video_cmd(ffmpeg, input_file, output_file, filter_expr, params)
    elif operation == "video_noise":
        sigma = max(0, min(100, int(round(float(params["sigma"])))))
        filter_expr = f"noise=alls={sigma}:allf=t"
        cmd = _filtered_video_cmd(ffmpeg, input_file, output_file, filter_expr, params)
    else:
        raise ProcessingError(f"Unsupported video operation: {operation}")

    _run(cmd)
    return "success"


def probe_video_dimensions(
    input_path: str | Path, ffprobe: str = "ffprobe"
) -> tuple[int, int]:
    """Return width and height for the first video stream."""

    cmd = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height",
        "-of",
        "json",
        str(input_path),
    ]
    result = _run(cmd)
    try:
        data = json.loads(result.stdout)
        stream = data["streams"][0]
        return int(stream["width"]), int(stream["height"])
    except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ProcessingError(f"Could not read video dimensions for {input_path}") from exc


def _base_cmd(ffmpeg: str, input_file: Path) -> list[str]:
    return [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(input_file),
    ]


def _filtered_video_cmd(
    ffmpeg: str,
    input_file: Path,
    output_file: Path,
    filter_expr: str,
    params: dict[str, Any],
) -> list[str]:
    return _base_cmd(ffmpeg, input_file) + [
        "-vf",
        filter_expr,
        "-c:v",
        "libx264",
        "-preset",
        str(params.get("preset", "medium")),
        "-crf",
        str(params.get("reencode_crf", 18)),
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "copy",
        "-movflags",
        "+faststart",
        str(output_file),
    ]


def _even_dimension(value: float) -> int:
    dimension = max(2, int(round(value)))
    return dimension if dimension % 2 == 0 else dimension - 1


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    try:
        result = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise ProcessingError(f"Executable not found: {cmd[0]}") from exc

    if result.returncode != 0:
        stderr = result.stderr.strip() or result.stdout.strip()
        raise ProcessingError(f"{cmd[0]} failed: {stderr}")
    return result
