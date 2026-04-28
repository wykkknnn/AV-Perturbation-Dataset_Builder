"""Audio perturbations implemented with ffmpeg."""

from __future__ import annotations

import math
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any


class ProcessingError(RuntimeError):
    """Raised when ffmpeg or ffprobe fails."""


def apply_audio_perturbation(
    input_path: str | Path,
    output_path: str | Path,
    params: dict[str, Any],
    ffmpeg: str = "ffmpeg",
    ffprobe: str = "ffprobe",
) -> str:
    """Apply one audio perturbation and copy the existing video stream."""

    input_file = Path(input_path)
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if not has_audio_stream(input_file, ffprobe):
        shutil.copy2(input_file, output_file)
        return "skipped_no_audio"

    operation = params.get("operation")
    if operation == "audio_compression":
        cmd = _base_cmd(ffmpeg, input_file) + [
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            str(params["bitrate"]),
            "-movflags",
            "+faststart",
            str(output_file),
        ]
    elif operation == "audio_noise":
        cmd = _audio_noise_cmd(ffmpeg, input_file, output_file, params)
    else:
        raise ProcessingError(f"Unsupported audio operation: {operation}")

    _run(cmd)
    return "success"


def has_audio_stream(input_path: str | Path, ffprobe: str = "ffprobe") -> bool:
    cmd = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=index",
        "-of",
        "csv=p=0",
        str(input_path),
    ]
    result = _run(cmd)
    return bool(result.stdout.strip())


def probe_duration(input_path: str | Path, ffprobe: str = "ffprobe") -> float | None:
    cmd = [
        ffprobe,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(input_path),
    ]
    result = _run(cmd)
    try:
        duration = float(result.stdout.strip())
    except ValueError:
        return None
    return duration if duration > 0 else None


def probe_mean_volume_db(
    input_path: str | Path, ffmpeg: str = "ffmpeg"
) -> float:
    """Estimate mean audio level in dBFS with ffmpeg volumedetect."""

    cmd = [
        ffmpeg,
        "-hide_banner",
        "-nostats",
        "-i",
        str(input_path),
        "-vn",
        "-sn",
        "-dn",
        "-af",
        "volumedetect",
        "-f",
        "null",
        "-",
    ]
    result = _run(cmd)
    match = re.search(r"mean_volume:\s*(-?\d+(?:\.\d+)?)\s*dB", result.stderr)
    if not match:
        return -20.0
    return float(match.group(1))


def _audio_noise_cmd(
    ffmpeg: str,
    input_file: Path,
    output_file: Path,
    params: dict[str, Any],
) -> list[str]:
    snr_db = float(params["snr_db"])
    mean_volume_db = probe_mean_volume_db(input_file, ffmpeg)
    noise_db = mean_volume_db - snr_db
    amplitude = max(0.00001, min(0.5, math.pow(10.0, noise_db / 20.0)))
    duration = probe_duration(input_file)
    duration_part = f":d={duration + 0.25:.3f}" if duration else ""

    filter_complex = (
        "[0:a:0]aresample=48000[a];"
        f"anoisesrc=r=48000:a={amplitude:.8f}{duration_part}:c=white[n];"
        "[a][n]amix=inputs=2:duration=first:dropout_transition=0,"
        "alimiter=limit=0.98[aout]"
    )

    return _base_cmd(ffmpeg, input_file) + [
        "-filter_complex",
        filter_complex,
        "-map",
        "0:v?",
        "-map",
        "[aout]",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        str(params.get("output_bitrate", "128k")),
        "-movflags",
        "+faststart",
        str(output_file),
    ]


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
