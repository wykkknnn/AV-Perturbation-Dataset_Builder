"""YAML config loading and perturbation expansion."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from dataset_scanner import DEFAULT_VIDEO_EXTENSIONS


VALID_APPLY_TO = {"real", "fake", "both"}


class ConfigError(ValueError):
    """Raised when a config file is missing required or valid values."""


@dataclass(frozen=True)
class PerturbationSpec:
    enhancement_name: str
    strength: str
    apply_to: str
    video_params: dict[str, Any] = field(default_factory=dict)
    audio_params: dict[str, Any] = field(default_factory=dict)


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load a YAML config and apply top-level defaults."""

    path = Path(config_path).expanduser()
    if not path.is_file():
        raise ConfigError(f"Config file does not exist: {path}")

    try:
        import yaml
    except ImportError as exc:
        raise ConfigError(
            "PyYAML is required to read config.yaml. Install it with "
            "`pip install pyyaml` or `pip install -r requirements.txt`."
        ) from exc

    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    if not isinstance(raw, dict):
        raise ConfigError("Config root must be a YAML mapping.")

    config = dict(raw)
    config.setdefault("run_name", datetime.now().strftime("run_%Y%m%d_%H%M%S"))
    config.setdefault("output_dir", "outputs")
    config.setdefault("apply_to", "both")
    config.setdefault("overwrite", False)
    config.setdefault("video_extensions", list(DEFAULT_VIDEO_EXTENSIONS))
    config.setdefault("perturbations", {})

    _validate_apply_to(config["apply_to"], "apply_to")
    if not isinstance(config["perturbations"], dict):
        raise ConfigError("`perturbations` must be a mapping.")

    return config


def build_perturbation_specs(config: dict[str, Any]) -> list[PerturbationSpec]:
    """Expand configured perturbation levels into concrete jobs."""

    global_apply_to = config.get("apply_to", "both")
    specs: list[PerturbationSpec] = []
    perturbations = config.get("perturbations", {})

    specs.extend(_build_video_compression(perturbations, global_apply_to))
    specs.extend(_build_resize_down_up(perturbations, global_apply_to))
    specs.extend(_build_gaussian_blur(perturbations, global_apply_to))
    specs.extend(_build_video_noise(perturbations, global_apply_to))
    specs.extend(_build_audio_compression(perturbations, global_apply_to))
    specs.extend(_build_audio_noise(perturbations, global_apply_to))

    return specs


def _build_video_compression(
    perturbations: dict[str, Any], global_apply_to: str
) -> list[PerturbationSpec]:
    block = _enabled_block(perturbations, "video_compression")
    if block is None:
        return []

    preset = str(block.get("preset", "medium"))
    levels = _required_list(block, ("crf", "crf_levels"), "video_compression")
    apply_to = _resolve_apply_to(block, global_apply_to, "video_compression")

    specs: list[PerturbationSpec] = []
    for crf in levels:
        crf_int = int(crf)
        if crf_int < 0 or crf_int > 51:
            raise ConfigError("video_compression CRF must be between 0 and 51.")
        specs.append(
            PerturbationSpec(
                enhancement_name="video_compression",
                strength=f"crf{crf_int}",
                apply_to=apply_to,
                video_params={
                    "operation": "video_compression",
                    "crf": crf_int,
                    "preset": preset,
                },
            )
        )
    return specs


def _build_resize_down_up(
    perturbations: dict[str, Any], global_apply_to: str
) -> list[PerturbationSpec]:
    block = _enabled_block(perturbations, "resize_down_up")
    if block is None:
        return []

    scales = _required_list(block, ("scales",), "resize_down_up")
    apply_to = _resolve_apply_to(block, global_apply_to, "resize_down_up")
    crf = int(block.get("reencode_crf", 18))
    preset = str(block.get("preset", "medium"))

    specs: list[PerturbationSpec] = []
    for scale in scales:
        scale_float = float(scale)
        if scale_float <= 0 or scale_float >= 1:
            raise ConfigError("resize_down_up scales must be > 0 and < 1.")
        specs.append(
            PerturbationSpec(
                enhancement_name="resize_down_up",
                strength=f"scale{_format_token(scale_float)}",
                apply_to=apply_to,
                video_params={
                    "operation": "resize_down_up",
                    "scale": scale_float,
                    "reencode_crf": crf,
                    "preset": preset,
                },
            )
        )
    return specs


def _build_gaussian_blur(
    perturbations: dict[str, Any], global_apply_to: str
) -> list[PerturbationSpec]:
    block = _enabled_block(perturbations, "gaussian_blur")
    if block is None:
        return []

    kernel_sizes = _required_list(block, ("kernel_sizes", "kernels"), "gaussian_blur")
    apply_to = _resolve_apply_to(block, global_apply_to, "gaussian_blur")
    crf = int(block.get("reencode_crf", 18))
    preset = str(block.get("preset", "medium"))

    specs: list[PerturbationSpec] = []
    for kernel_size in kernel_sizes:
        kernel_int = int(kernel_size)
        if kernel_int <= 0 or kernel_int % 2 == 0:
            raise ConfigError("gaussian_blur kernel_sizes must be positive odd ints.")
        sigma = float(block.get("sigma", max(0.1, (kernel_int - 1) / 6.0)))
        specs.append(
            PerturbationSpec(
                enhancement_name="gaussian_blur",
                strength=f"k{kernel_int}",
                apply_to=apply_to,
                video_params={
                    "operation": "gaussian_blur",
                    "kernel_size": kernel_int,
                    "sigma": sigma,
                    "reencode_crf": crf,
                    "preset": preset,
                },
            )
        )
    return specs


def _build_video_noise(
    perturbations: dict[str, Any], global_apply_to: str
) -> list[PerturbationSpec]:
    block = _enabled_block(perturbations, "video_noise")
    if block is None:
        return []

    sigmas = _required_list(block, ("sigmas", "sigma_levels"), "video_noise")
    apply_to = _resolve_apply_to(block, global_apply_to, "video_noise")
    crf = int(block.get("reencode_crf", 18))
    preset = str(block.get("preset", "medium"))

    specs: list[PerturbationSpec] = []
    for sigma in sigmas:
        sigma_float = float(sigma)
        if sigma_float < 0 or sigma_float > 100:
            raise ConfigError("video_noise sigmas must be between 0 and 100.")
        specs.append(
            PerturbationSpec(
                enhancement_name="video_noise",
                strength=f"sigma{_format_token(sigma_float)}",
                apply_to=apply_to,
                video_params={
                    "operation": "video_noise",
                    "sigma": sigma_float,
                    "reencode_crf": crf,
                    "preset": preset,
                },
            )
        )
    return specs


def _build_audio_compression(
    perturbations: dict[str, Any], global_apply_to: str
) -> list[PerturbationSpec]:
    block = _enabled_block(perturbations, "audio_compression")
    if block is None:
        return []

    bitrates = _required_list(block, ("bitrates",), "audio_compression")
    apply_to = _resolve_apply_to(block, global_apply_to, "audio_compression")

    specs: list[PerturbationSpec] = []
    for bitrate in bitrates:
        bitrate_text = str(bitrate)
        specs.append(
            PerturbationSpec(
                enhancement_name="audio_compression",
                strength=f"br{_format_token(bitrate_text)}",
                apply_to=apply_to,
                audio_params={
                    "operation": "audio_compression",
                    "bitrate": bitrate_text,
                },
            )
        )
    return specs


def _build_audio_noise(
    perturbations: dict[str, Any], global_apply_to: str
) -> list[PerturbationSpec]:
    block = _enabled_block(perturbations, "audio_noise")
    if block is None:
        return []

    snr_levels = _required_list(block, ("snr_db", "snr_levels"), "audio_noise")
    apply_to = _resolve_apply_to(block, global_apply_to, "audio_noise")
    output_bitrate = str(block.get("output_bitrate", "128k"))

    specs: list[PerturbationSpec] = []
    for snr in snr_levels:
        snr_float = float(snr)
        specs.append(
            PerturbationSpec(
                enhancement_name="audio_noise",
                strength=f"snr{_format_token(snr_float)}db",
                apply_to=apply_to,
                audio_params={
                    "operation": "audio_noise",
                    "snr_db": snr_float,
                    "output_bitrate": output_bitrate,
                },
            )
        )
    return specs


def _enabled_block(perturbations: dict[str, Any], name: str) -> dict[str, Any] | None:
    block = perturbations.get(name)
    if block is None or block is False:
        return None
    if block is True:
        raise ConfigError(f"`{name}` must be a mapping with parameter levels.")
    if not isinstance(block, dict):
        raise ConfigError(f"`{name}` must be a mapping.")
    if block.get("enabled", True) is False:
        return None
    return block


def _required_list(
    block: dict[str, Any], keys: tuple[str, ...], perturbation_name: str
) -> list[Any]:
    for key in keys:
        if key in block:
            value = block[key]
            if not isinstance(value, list) or not value:
                raise ConfigError(f"`{perturbation_name}.{key}` must be a non-empty list.")
            return value
    joined = " or ".join(keys)
    raise ConfigError(f"`{perturbation_name}` requires {joined}.")


def _resolve_apply_to(
    block: dict[str, Any], global_apply_to: str, context: str
) -> str:
    value = str(block.get("apply_to", global_apply_to)).lower()
    _validate_apply_to(value, f"{context}.apply_to")
    return value


def _validate_apply_to(value: str, field_name: str) -> None:
    if str(value).lower() not in VALID_APPLY_TO:
        raise ConfigError(f"`{field_name}` must be one of: both, real, fake.")


def _format_token(value: object) -> str:
    token = f"{value:g}" if isinstance(value, float) else str(value)
    return (
        token.strip()
        .lower()
        .replace(".", "p")
        .replace("/", "_")
        .replace(" ", "")
    )
