"""CLI entry point for the local AV perturbation dataset builder."""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

from audio_enhancer import ProcessingError as AudioProcessingError
from audio_enhancer import apply_audio_perturbation
from config_loader import ConfigError, PerturbationSpec, build_perturbation_specs, load_config
from dataset_scanner import VideoSample, scan_dataset
from output_manager import (
    MetadataWriter,
    build_output_path,
    copy_unperturbed,
    ensure_variant_structure,
    prepare_run_directory,
    setup_logger,
)
from video_enhancer import ProcessingError as VideoProcessingError
from video_enhancer import apply_video_perturbation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate perturbed real/fake AV test datasets with ffmpeg."
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config.yaml. Defaults to ./config.yaml.",
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
        _apply_cli_overrides(config, args)
        _validate_required_config(config)
        _check_tools("ffmpeg", "ffprobe")

        specs = build_perturbation_specs(config)
        if not specs:
            raise ConfigError("No perturbations are enabled in config.yaml.")

        run_dir = prepare_run_directory(
            config["output_dir"],
            config["run_name"],
            args.config,
            overwrite=bool(config.get("overwrite", False)),
        )
        logger = setup_logger(run_dir)

        samples = scan_dataset(
            config["input_dir"],
            video_extensions=config.get("video_extensions"),
        )
        logger.info("Found %d input videos.", len(samples))
        logger.info("Expanded %d perturbation jobs.", len(specs))

        metadata_path = run_dir / "metadata.csv"
        with MetadataWriter(metadata_path) as metadata:
            for spec in specs:
                ensure_variant_structure(run_dir, spec)
                logger.info(
                    "Starting %s/%s apply_to=%s",
                    spec.enhancement_name,
                    spec.strength,
                    spec.apply_to,
                )
                for sample in samples:
                    output_path = build_output_path(run_dir, spec, sample)
                    status = _process_one(sample, output_path, spec, logger)
                    metadata.write_row(sample, output_path, spec, status)

        logger.info("Finished. Outputs written to %s", run_dir)
        return 0
    except (ConfigError, FileExistsError, FileNotFoundError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


def _apply_cli_overrides(config: dict[str, object], args: argparse.Namespace) -> None:
    if args.prompt_paths:
        _prompt_path_overrides(config)
    if args.input_dir:
        config["input_dir"] = args.input_dir
    if args.output_dir:
        config["output_dir"] = args.output_dir
    if args.run_name:
        config["run_name"] = args.run_name
    if args.overwrite:
        config["overwrite"] = True


def _prompt_path_overrides(config: dict[str, object]) -> None:
    config["input_dir"] = _prompt_value(
        "input_dir",
        str(config.get("input_dir", "")),
    )
    config["output_dir"] = _prompt_value(
        "output_dir",
        str(config.get("output_dir", "")),
    )
    config["run_name"] = _prompt_value(
        "run_name",
        str(config.get("run_name", "")),
    )


def _prompt_value(field_name: str, current_value: str) -> str:
    prompt_text = f"{field_name} [{current_value}]: "
    entered = input(prompt_text).strip()
    return entered or current_value


def _validate_required_config(config: dict[str, object]) -> None:
    if not config.get("input_dir"):
        raise ConfigError("`input_dir` is required in config.yaml or via --input-dir.")
    if not config.get("run_name"):
        raise ConfigError("`run_name` is required.")
    if not config.get("output_dir"):
        raise ConfigError("`output_dir` is required.")


def _check_tools(*tool_names: str) -> None:
    missing = [name for name in tool_names if shutil.which(name) is None]
    if missing:
        joined = ", ".join(missing)
        raise ConfigError(f"Missing required executable(s): {joined}")


def _process_one(
    sample: VideoSample,
    output_path: Path,
    spec: PerturbationSpec,
    logger,
) -> str:
    if not _should_apply(sample.label, spec.apply_to):
        status = copy_unperturbed(sample, output_path)
        logger.info("%s -> %s [%s]", sample.original_path, output_path, status)
        return status

    try:
        status = _apply_spec(sample.original_path, output_path, spec)
        logger.info("%s -> %s [%s]", sample.original_path, output_path, status)
        return status
    except (AudioProcessingError, VideoProcessingError, OSError) as exc:
        status = f"failed: {exc}"
        logger.error("%s -> %s [%s]", sample.original_path, output_path, status)
        return status


def _apply_spec(input_path: Path, output_path: Path, spec: PerturbationSpec) -> str:
    if spec.video_params and spec.audio_params:
        return _apply_combined_spec(input_path, output_path, spec)
    if spec.video_params:
        return apply_video_perturbation(input_path, output_path, spec.video_params)
    if spec.audio_params:
        return apply_audio_perturbation(input_path, output_path, spec.audio_params)

    shutil.copy2(input_path, output_path)
    return "copied_noop"


def _apply_combined_spec(input_path: Path, output_path: Path, spec: PerturbationSpec) -> str:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="av_perturb_") as temp_dir:
        intermediate = Path(temp_dir) / output_path.name
        video_status = apply_video_perturbation(input_path, intermediate, spec.video_params)
        audio_status = apply_audio_perturbation(intermediate, output_path, spec.audio_params)
    if video_status == "success" and audio_status == "success":
        return "success"
    return f"success_with_notes: video={video_status}; audio={audio_status}"


def _should_apply(label: str, apply_to: str) -> bool:
    return apply_to == "both" or apply_to == label


if __name__ == "__main__":
    raise SystemExit(main())
