"""Dataset scanning utilities for real/fake video test sets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


DEFAULT_VIDEO_EXTENSIONS: tuple[str, ...] = (
    ".mp4",
    ".mov",
    ".mkv",
    ".avi",
    ".webm",
    ".m4v",
)

LABELS: tuple[str, str] = ("real", "fake")


@dataclass(frozen=True)
class VideoSample:
    """One input video and the class folder it came from."""

    original_path: Path
    label: str
    relative_path: Path


def scan_dataset(
    input_dir: str | Path,
    video_extensions: Sequence[str] | None = None,
    labels: Iterable[str] = LABELS,
) -> list[VideoSample]:
    """Scan input_dir/{real,fake} recursively for video files."""

    root = Path(input_dir).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {root}")

    extensions = {
        ext.lower() if ext.startswith(".") else f".{ext.lower()}"
        for ext in (video_extensions or DEFAULT_VIDEO_EXTENSIONS)
    }

    samples: list[VideoSample] = []
    for label in labels:
        class_dir = root / label
        if not class_dir.is_dir():
            raise FileNotFoundError(
                f"Expected class directory is missing: {class_dir}"
            )

        for path in sorted(class_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in extensions:
                samples.append(
                    VideoSample(
                        original_path=path.resolve(),
                        label=label,
                        relative_path=path.relative_to(class_dir),
                    )
                )

    return samples
