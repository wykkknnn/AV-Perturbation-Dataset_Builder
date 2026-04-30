"""Dataset scanning utilities for 0_real/1_fake style video test sets."""

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

LOGICAL_LABELS: tuple[str, str] = ("real", "fake")
OUTPUT_LABEL_DIRS: dict[str, str] = {
    "real": "0_real",
    "fake": "1_fake",
}
LABEL_ALIASES: dict[str, tuple[str, ...]] = {
    "real": ("0_real", "real"),
    "fake": ("1_fake", "fake"),
}


@dataclass(frozen=True)
class VideoSample:
    """One input video and the class folder it came from."""

    original_path: Path
    label_dir: str
    logical_label: str
    relative_path: Path


def scan_dataset(
    input_dir: str | Path,
    video_extensions: Sequence[str] | None = None,
    labels: Iterable[str] = LOGICAL_LABELS,
) -> list[VideoSample]:
    """Scan input_dir label folders recursively for video files."""

    root = Path(input_dir).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {root}")

    extensions = {
        ext.lower() if ext.startswith(".") else f".{ext.lower()}"
        for ext in (video_extensions or DEFAULT_VIDEO_EXTENSIONS)
    }

    samples: list[VideoSample] = []
    for logical_label in labels:
        class_dir = _resolve_class_dir(root, logical_label)
        if class_dir is None:
            raise FileNotFoundError(
                f"Expected class directory is missing for label '{logical_label}'. "
                f"Tried: {_candidate_dirs(root, logical_label)}"
            )

        for path in sorted(class_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in extensions:
                samples.append(
                    VideoSample(
                        original_path=path.resolve(),
                        label_dir=OUTPUT_LABEL_DIRS.get(logical_label, logical_label),
                        logical_label=logical_label,
                        relative_path=path.relative_to(class_dir),
                    )
                )

    return samples


def _resolve_class_dir(root: Path, label: str) -> Path | None:
    for name in LABEL_ALIASES.get(label, (label,)):
        candidate = root / name
        if candidate.is_dir():
            return candidate
    return None


def _candidate_dirs(root: Path, label: str) -> list[str]:
    return [str(root / name) for name in LABEL_ALIASES.get(label, (label,))]
