# AV Perturbation Dataset Builder

Local ffmpeg-based generator for perturbed deepfake robustness test sets.
It scans an input dataset shaped like:

```text
test_original/
  real/
    *.mp4
  fake/
    *.mp4
```

Each configured perturbation strength creates a new dataset under
`outputs/run_name/enhancement_name_strength/{real,fake}`. Classes outside
`apply_to` are copied unchanged so every output folder remains a complete
real/fake test set.

## Setup

Install ffmpeg and ffprobe on your machine, then install the Python dependency:

```bash
pip install -r requirements.txt
```

## Run (Unified Entry)

```bash
python main.py --config config.yaml
```

After launching, the CLI asks two questions:

1. `Content type to enhance (video/audio/image):`
2. `Enter the folder path to process:`

Routing behavior:

- `video` -> uses `video_config.yaml`
- `audio` -> uses `audio_config.yaml`
- `image` -> uses `image_config.yaml` (delegates to `image_enhancer.py`)

If a mode-specific config is missing, it falls back to the file passed with `--config`.

Useful overrides:

```bash
python main.py --config config.yaml --input-dir video --run-name exp_video_only
```

Interactive path input in CLI (no config edits needed):

```bash
python main.py --config config.yaml --prompt-paths
```

`apply_to` can be set globally or inside any perturbation block (`real`, `fake`, `both`):

```yaml
apply_to: both
perturbations:
  video_compression:
    apply_to: fake
    crf: [32]
```

## Supported first-version perturbations

- `video_compression`: H.264 CRF levels
- `resize_down_up`: downscale then upscale back to original dimensions
- `gaussian_blur`: ffmpeg Gaussian blur using configured kernel sizes
- `video_noise`: ffmpeg video noise strengths
- `audio_compression`: AAC bitrates
- `audio_noise`: white background noise using target SNR levels

## Outputs

```text
outputs/run_name/
  config.yaml
  metadata.csv
  logs.txt
  enhancement_name_strength/
    0_real/
    1_fake/
```

`metadata.csv` records `original_path`, `output_path`, `label`,
`enhancement_name`, `strength`, `apply_to`, `video_params`, `audio_params`,
and `status`.

## Direct Image Pipeline (Optional)

You can still call `image_enhancer.py` directly for image-only datasets.

Expected input layout example:

```text
CNN_synth_testset/stargan/
  0_real/
    *.png
  1_fake/
    *.png
```

### Run image perturbation

```bash
python image_enhancer.py --config image_config.yaml
```

Useful overrides:

```bash
python image_enhancer.py --config image_config.yaml --run-name stargan_aug_exp1
python image_enhancer.py --config image_config.yaml --overwrite
```

Interactive path input in CLI (no config edits needed):

```bash
python image_enhancer.py --config image_config.yaml --prompt-paths
```

Outputs are written to:

```text
outputs/run_name/
  config.yaml
  metadata.csv
  enhancement_name_strength/
    0_real/
    1_fake/
```

### Parameter tuning guide

- `gaussian_blur.radii`: larger value means more blur.
  - Mild: `[1.0, 1.5]`
  - Medium: `[2.0, 3.0]`
  - Strong: `[4.0, 6.0]`
- `jpeg_compression.qualities`: lower value means stronger compression artifacts.
  - Mild: `[80, 70]`
  - Medium: `[60, 50]`
  - Strong: `[40, 30]`
- `gaussian_noise.sigmas`: larger value means stronger additive noise.
  - Mild: `[2, 4]`
  - Medium: `[6, 8]`
  - Strong: `[12, 16]`
- `resize_down_up.scales`: smaller value means more detail loss after down-up sampling.
  - Mild: `[0.85, 0.75]`
  - Medium: `[0.6, 0.5]`
  - Strong: `[0.4, 0.3]`
- `brightness.factors`:
  - Darker: values `< 1.0` (e.g. `0.8`)
  - Brighter: values `> 1.0` (e.g. `1.2`)

`apply_to` can be set globally or inside each perturbation block:

```yaml
apply_to: both
perturbations:
  gaussian_blur:
    enabled: true
    apply_to: fake
    radii: [2.0]
```
