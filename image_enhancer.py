"""CLI tool for generating perturbed image datasets."""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml
from PIL import Image, ImageEnhance, ImageFilter

VALID_APPLY_TO = {"real", "fake", "both"}


class ConfigError(ValueError):
    """Raised when image config is missing required values."""


@dataclass(frozen=True)
class ImageSample:
    original_path: Path
    label_dir: str
    logical_label: str
    relative_path: Path


@dataclass(frozen=True)
class ImagePerturbationSpec:
    enhancement_name: str
    strength: str
    apply_to: str
    params: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate perturbed image test datasets from label folders."
    )
    parser.add_argument(
        "--config",
        default="image_config.yaml",
        help="Path to image YAML config. Defaults to ./image_config.yaml.",
    )
    parser.add_argument("--input-dir", help="Override input_dir from config.")
    parser.add_argument("--output-dir", help="Override output_dir from config.")
    parser.add_argument("--run-name", help="Override run_name from config.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into an existing outputs/run_name directory.",
    )
    parser.add_argument(
        "--prompt-paths",
        action="store_true",
        help="Interactively input input/output/run paths in CLI.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        config = load_config(args.config)
        apply_cli_overrides(config, args)
        validate_required_config(config)

        run_dir = prepare_run_directory(
            output_dir=config["output_dir"],
            run_name=config["run_name"],
            config_path=Path(args.config),
            overwrite=bool(config.get("overwrite", False)),
        )
        specs = build_perturbation_specs(config)
        if not specs:
            raise ConfigError("No image perturbations are enabled in config.")

        samples = scan_dataset(config)
        if not samples:
            raise ConfigError("No input images found. Check image_extensions and folders.")

        metadata_path = run_dir / "metadata.csv"
        with metadata_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "original_path",
                    "output_path",
                    "label_dir",
                    "logical_label",
                    "enhancement_name",
                    "strength",
                    "apply_to",
                    "params",
                    "status",
                ],
            )
            writer.writeheader()

            for spec in specs:
                variant_root = run_dir / f"{spec.enhancement_name}_{spec.strength}"
                for sample in samples:
                    output_path = variant_root / sample.label_dir / sample.relative_path
                    if not should_apply(sample.logical_label, spec.apply_to):
                        status = copy_unperturbed(sample.original_path, output_path)
                    else:
                        status = apply_image_perturbation(sample.original_path, output_path, spec)

                    writer.writerow(
                        {
                            "original_path": str(sample.original_path),
                            "output_path": str(output_path),
                            "label_dir": sample.label_dir,
                            "logical_label": sample.logical_label,
                            "enhancement_name": spec.enhancement_name,
                            "strength": spec.strength,
                            "apply_to": spec.apply_to,
                            "params": spec.params,
                            "status": status,
                        }
                    )

        print(f"Finished. Outputs written to: {run_dir}")
        return 0
    except (ConfigError, FileExistsError, FileNotFoundError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


def load_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path).expanduser()
    if not path.is_file():
        raise ConfigError(f"Config file does not exist: {path}")

    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    if not isinstance(raw, dict):
        raise ConfigError("Config root must be a YAML mapping.")

    config = dict(raw)
    config.setdefault("run_name", datetime.now().strftime("image_run_%Y%m%d_%H%M%S"))
    config.setdefault("output_dir", "outputs")
    config.setdefault("apply_to", "both")
    config.setdefault("overwrite", False)
    config.setdefault("image_extensions", [".png", ".jpg", ".jpeg", ".bmp", ".webp"])
    config.setdefault("label_dirs", ["real", "fake"])
    config.setdefault(
        "label_map",
        {
            "real": "real",
            "fake": "fake",
            "0_real": "real",
            "1_fake": "fake",
        },
    )
    config.setdefault("output_format", "keep")
    config.setdefault("jpeg_quality", 95)
    config.setdefault("perturbations", {})

    validate_apply_to(str(config["apply_to"]).lower(), "apply_to")
    if not isinstance(config["perturbations"], dict):
        raise ConfigError("`perturbations` must be a mapping.")
    if not isinstance(config["label_dirs"], list) or not config["label_dirs"]:
        raise ConfigError("`label_dirs` must be a non-empty list.")
    if not isinstance(config["label_map"], dict):
        raise ConfigError("`label_map` must be a mapping.")

    return config


def apply_cli_overrides(config: dict[str, Any], args: argparse.Namespace) -> None:
    if args.prompt_paths:
        prompt_path_overrides(config)
    if args.input_dir:
        config["input_dir"] = args.input_dir
    if args.output_dir:
        config["output_dir"] = args.output_dir
    if args.run_name:
        config["run_name"] = args.run_name
    if args.overwrite:
        config["overwrite"] = True


def prompt_path_overrides(config: dict[str, Any]) -> None:
    config["input_dir"] = prompt_value(
        "input_dir",
        str(config.get("input_dir", "")),
    )
    config["output_dir"] = prompt_value(
        "output_dir",
        str(config.get("output_dir", "")),
    )
    config["run_name"] = prompt_value(
        "run_name",
        str(config.get("run_name", "")),
    )


def prompt_value(field_name: str, current_value: str) -> str:
    prompt_text = f"{field_name} [{current_value}]: "
    entered = input(prompt_text).strip()
    return entered or current_value


def validate_required_config(config: dict[str, Any]) -> None:
    if not config.get("input_dir"):
        raise ConfigError("`input_dir` is required in image config or via --input-dir.")
    if not config.get("run_name"):
        raise ConfigError("`run_name` is required.")
    if not config.get("output_dir"):
        raise ConfigError("`output_dir` is required.")


def prepare_run_directory(
    output_dir: str | Path, run_name: str, config_path: Path, overwrite: bool
) -> Path:
    root = Path(output_dir).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    run_dir = root / run_name
    if run_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Run directory already exists: {run_dir}. Set overwrite=true or --overwrite."
            )
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(config_path, run_dir / "config.yaml")
    return run_dir


def scan_dataset(config: dict[str, Any]) -> list[ImageSample]:
    root = Path(str(config["input_dir"])).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {root}")

    extensions = {
        ext.lower() if str(ext).startswith(".") else f".{str(ext).lower()}"
        for ext in config.get("image_extensions", [])
    }
    if not extensions:
        raise ConfigError("`image_extensions` cannot be empty.")

    label_map = {str(k): str(v).lower() for k, v in config.get("label_map", {}).items()}

    samples: list[ImageSample] = []
    for label_dir in config.get("label_dirs", []):
        label_name = str(label_dir)
        class_dir = root / label_name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Expected class directory is missing: {class_dir}")

        logical_label = label_map.get(label_name, label_name.lower())
        validate_apply_to(logical_label, f"label_map[{label_name}]")

        for path in sorted(class_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in extensions:
                samples.append(
                    ImageSample(
                        original_path=path.resolve(),
                        label_dir=label_name,
                        logical_label=logical_label,
                        relative_path=path.relative_to(class_dir),
                    )
                )
    return samples


def build_perturbation_specs(config: dict[str, Any]) -> list[ImagePerturbationSpec]:
    global_apply_to = str(config.get("apply_to", "both")).lower()
    perturbations = config.get("perturbations", {})
    specs: list[ImagePerturbationSpec] = []
    specs.extend(build_gaussian_blur(perturbations, global_apply_to))
    specs.extend(build_jpeg_compression(perturbations, global_apply_to))
    specs.extend(build_gaussian_noise(perturbations, global_apply_to))
    specs.extend(build_resize_down_up(perturbations, global_apply_to))
    specs.extend(build_brightness(perturbations, global_apply_to))
    return specs


def build_gaussian_blur(
    perturbations: dict[str, Any], global_apply_to: str
) -> list[ImagePerturbationSpec]:
    block = enabled_block(perturbations, "gaussian_blur")
    if block is None:
        return []

    radii = required_list(block, ("radii",), "gaussian_blur")
    apply_to = resolve_apply_to(block, global_apply_to, "gaussian_blur")
    specs: list[ImagePerturbationSpec] = []
    for radius in radii:
        radius_float = float(radius)
        if radius_float <= 0:
            raise ConfigError("gaussian_blur radii must be > 0.")
        specs.append(
            ImagePerturbationSpec(
                enhancement_name="gaussian_blur",
                strength=f"r{format_token(radius_float)}",
                apply_to=apply_to,
                params={"operation": "gaussian_blur", "radius": radius_float},
            )
        )
    return specs


def build_jpeg_compression(
    perturbations: dict[str, Any], global_apply_to: str
) -> list[ImagePerturbationSpec]:
    block = enabled_block(perturbations, "jpeg_compression")
    if block is None:
        return []

    qualities = required_list(block, ("qualities",), "jpeg_compression")
    apply_to = resolve_apply_to(block, global_apply_to, "jpeg_compression")
    specs: list[ImagePerturbationSpec] = []
    for quality in qualities:
        quality_int = int(quality)
        if quality_int < 1 or quality_int > 95:
            raise ConfigError("jpeg_compression qualities must be between 1 and 95.")
        specs.append(
            ImagePerturbationSpec(
                enhancement_name="jpeg_compression",
                strength=f"q{quality_int}",
                apply_to=apply_to,
                params={"operation": "jpeg_compression", "quality": quality_int},
            )
        )
    return specs


def build_gaussian_noise(
    perturbations: dict[str, Any], global_apply_to: str
) -> list[ImagePerturbationSpec]:
    block = enabled_block(perturbations, "gaussian_noise")
    if block is None:
        return []

    sigmas = required_list(block, ("sigmas",), "gaussian_noise")
    apply_to = resolve_apply_to(block, global_apply_to, "gaussian_noise")
    specs: list[ImagePerturbationSpec] = []
    for sigma in sigmas:
        sigma_float = float(sigma)
        if sigma_float < 0:
            raise ConfigError("gaussian_noise sigmas must be >= 0.")
        specs.append(
            ImagePerturbationSpec(
                enhancement_name="gaussian_noise",
                strength=f"sigma{format_token(sigma_float)}",
                apply_to=apply_to,
                params={"operation": "gaussian_noise", "sigma": sigma_float},
            )
        )
    return specs


def build_resize_down_up(
    perturbations: dict[str, Any], global_apply_to: str
) -> list[ImagePerturbationSpec]:
    block = enabled_block(perturbations, "resize_down_up")
    if block is None:
        return []

    scales = required_list(block, ("scales",), "resize_down_up")
    apply_to = resolve_apply_to(block, global_apply_to, "resize_down_up")
    specs: list[ImagePerturbationSpec] = []
    for scale in scales:
        scale_float = float(scale)
        if scale_float <= 0 or scale_float >= 1:
            raise ConfigError("resize_down_up scales must be > 0 and < 1.")
        specs.append(
            ImagePerturbationSpec(
                enhancement_name="resize_down_up",
                strength=f"scale{format_token(scale_float)}",
                apply_to=apply_to,
                params={"operation": "resize_down_up", "scale": scale_float},
            )
        )
    return specs


def build_brightness(
    perturbations: dict[str, Any], global_apply_to: str
) -> list[ImagePerturbationSpec]:
    block = enabled_block(perturbations, "brightness")
    if block is None:
        return []

    factors = required_list(block, ("factors",), "brightness")
    apply_to = resolve_apply_to(block, global_apply_to, "brightness")
    specs: list[ImagePerturbationSpec] = []
    for factor in factors:
        factor_float = float(factor)
        if factor_float <= 0:
            raise ConfigError("brightness factors must be > 0.")
        specs.append(
            ImagePerturbationSpec(
                enhancement_name="brightness",
                strength=f"f{format_token(factor_float)}",
                apply_to=apply_to,
                params={"operation": "brightness", "factor": factor_float},
            )
        )
    return specs


def should_apply(logical_label: str, apply_to: str) -> bool:
    return apply_to == "both" or apply_to == logical_label


def copy_unperturbed(input_path: Path, output_path: Path) -> str:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(input_path, output_path)
    return "copied_unperturbed"


def apply_image_perturbation(
    input_path: Path, output_path: Path, spec: ImagePerturbationSpec
) -> str:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    operation = str(spec.params["operation"])

    with Image.open(input_path) as image:
        rgb = image.convert("RGB")
        if operation == "gaussian_blur":
            out = rgb.filter(ImageFilter.GaussianBlur(radius=float(spec.params["radius"])))
            save_image(out, output_path, prefer_jpeg=False)
            return "success"
        if operation == "jpeg_compression":
            quality = int(spec.params["quality"])
            jpeg_path = output_path.with_suffix(".jpg")
            save_image(rgb, jpeg_path, prefer_jpeg=True, quality=quality)
            return "success_jpeg"
        if operation == "gaussian_noise":
            sigma = float(spec.params["sigma"])
            arr = np.asarray(rgb).astype(np.float32)
            noise = np.random.normal(0.0, sigma, arr.shape)
            noisy = np.clip(arr + noise, 0, 255).astype(np.uint8)
            out = Image.fromarray(noisy, mode="RGB")
            save_image(out, output_path, prefer_jpeg=False)
            return "success"
        if operation == "resize_down_up":
            scale = float(spec.params["scale"])
            width, height = rgb.size
            down_w = max(1, int(round(width * scale)))
            down_h = max(1, int(round(height * scale)))
            resized = rgb.resize((down_w, down_h), Image.Resampling.BICUBIC)
            out = resized.resize((width, height), Image.Resampling.BICUBIC)
            save_image(out, output_path, prefer_jpeg=False)
            return "success"
        if operation == "brightness":
            factor = float(spec.params["factor"])
            out = ImageEnhance.Brightness(rgb).enhance(factor)
            save_image(out, output_path, prefer_jpeg=False)
            return "success"
    return "failed_unknown"


def save_image(
    image: Image.Image, output_path: Path, prefer_jpeg: bool, quality: int = 95
) -> None:
    if prefer_jpeg:
        image.save(output_path, quality=quality, optimize=True)
        return

    suffix = output_path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        image.save(output_path, quality=quality, optimize=True)
    else:
        image.save(output_path)


def enabled_block(perturbations: dict[str, Any], name: str) -> dict[str, Any] | None:
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


def required_list(
    block: dict[str, Any], keys: Iterable[str], perturbation_name: str
) -> list[Any]:
    for key in keys:
        if key in block:
            value = block[key]
            if not isinstance(value, list) or not value:
                raise ConfigError(f"`{perturbation_name}.{key}` must be a non-empty list.")
            return value
    joined = " or ".join(keys)
    raise ConfigError(f"`{perturbation_name}` requires {joined}.")


def resolve_apply_to(
    block: dict[str, Any], global_apply_to: str, context: str
) -> str:
    value = str(block.get("apply_to", global_apply_to)).lower()
    validate_apply_to(value, f"{context}.apply_to")
    return value


def validate_apply_to(value: str, field_name: str) -> None:
    if value not in VALID_APPLY_TO:
        raise ConfigError(f"`{field_name}` must be one of: both, real, fake.")


def format_token(value: object) -> str:
    token = f"{value:g}" if isinstance(value, float) else str(value)
    return (
        token.strip()
        .lower()
        .replace(".", "p")
        .replace("/", "_")
        .replace(" ", "")
    )


if __name__ == "__main__":
    raise SystemExit(main())
