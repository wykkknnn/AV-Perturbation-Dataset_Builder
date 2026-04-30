"""Output directory, metadata, and logging helpers."""

from __future__ import annotations

import csv
import json
import logging
import re
import shutil
from pathlib import Path
from typing import Any

from config_loader import PerturbationSpec
from dataset_scanner import OUTPUT_LABEL_DIRS, VideoSample


METADATA_FIELDS: tuple[str, ...] = (
    "original_path",
    "output_path",
    "label",
    "enhancement_name",
    "strength",
    "apply_to",
    "video_params",
    "audio_params",
    "status",
)


def prepare_run_directory(
    output_dir: str | Path,
    run_name: str,
    config_path: str | Path,
    overwrite: bool = False,
) -> Path:
    run_dir = Path(output_dir).expanduser() / run_name
    if run_dir.exists() and not overwrite:
        raise FileExistsError(
            f"Output run already exists: {run_dir}. Set overwrite: true or "
            "choose a new run_name."
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, run_dir / "config.yaml")
    return run_dir


def setup_logger(run_dir: str | Path) -> logging.Logger:
    logger = logging.getLogger("av_perturbation_builder")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    file_handler = logging.FileHandler(Path(run_dir) / "logs.txt", mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


def ensure_variant_structure(run_dir: str | Path, spec: PerturbationSpec) -> Path:
    variant_dir = Path(run_dir) / variant_dir_name(spec)
    for label_dir in OUTPUT_LABEL_DIRS.values():
        (variant_dir / label_dir).mkdir(parents=True, exist_ok=True)
    return variant_dir


def build_output_path(
    run_dir: str | Path, spec: PerturbationSpec, sample: VideoSample
) -> Path:
    return Path(run_dir) / variant_dir_name(spec) / sample.label_dir / sample.relative_path


def copy_unperturbed(sample: VideoSample, output_path: str | Path) -> str:
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(sample.original_path, output_file)
    return "copied_unperturbed"


def variant_dir_name(spec: PerturbationSpec) -> str:
    return sanitize_name(f"{spec.enhancement_name}_{spec.strength}")


def sanitize_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip("._")
    return cleaned or "unnamed"


class MetadataWriter:
    def __init__(self, metadata_path: str | Path) -> None:
        self.metadata_path = Path(metadata_path)
        self._handle = None
        self._writer: csv.DictWriter[str] | None = None

    def __enter__(self) -> "MetadataWriter":
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.metadata_path.open("w", encoding="utf-8", newline="")
        self._writer = csv.DictWriter(self._handle, fieldnames=METADATA_FIELDS)
        self._writer.writeheader()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        if self._handle:
            self._handle.close()

    def write_row(
        self,
        sample: VideoSample,
        output_path: str | Path,
        spec: PerturbationSpec,
        status: str,
    ) -> None:
        if self._writer is None:
            raise RuntimeError("MetadataWriter must be used as a context manager.")

        self._writer.writerow(
            {
                "original_path": str(sample.original_path),
                "output_path": str(Path(output_path)),
                "label": sample.logical_label,
                "enhancement_name": spec.enhancement_name,
                "strength": spec.strength,
                "apply_to": spec.apply_to,
                "video_params": _json_field(spec.video_params),
                "audio_params": _json_field(spec.audio_params),
                "status": status,
            }
        )


def _json_field(value: dict[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False)
