<p align="center">
  <img src="assets/background.png" alt="VidaForge cover" width="100%">
</p>

<h1 align="center">VidaForge</h1>

<p align="center">
  <img src="assets/logo.png" alt="VidaForge logo" width="180">
</p>

<p align="center">
  <strong>Building a Video Foundation Model Pretraining Data Pipeline from Scratch in an Academic Lab</strong>
</p>

<p align="center">
  <a href="https://yanmaaaaaa.notion.site/vidaforge">Blog</a>
  ·
  <a href="#what-vidaforge-does">Pipeline</a>
  ·
  <a href="#quick-start">Quick Start</a>
  ·
  <a href="#citation">Citation</a>
</p>

VidaForge is a research-oriented data pipeline for video foundation model pretraining. It turns raw videos into standardized videos, scene-level clips, curated clips, annotated clips, and training-ready datasets for concrete training repositories.

The project started from a simple frustration: public video foundation model reports often spend less and less space on data processing, even as model quality keeps improving. The data work did not suddenly become trivial. More likely, the most valuable details moved into internal systems.

VidaForge is an attempt to make that part concrete in an academic lab. A data recipe should be easy to change. Intermediate assets should be easy to open and inspect. Rejected samples should stay available for analysis. Most importantly, a data decision should eventually be tested in real pretraining runs, not only in a spreadsheet.

Read the full project story here: https://yanmaaaaaa.notion.site/vidaforge

## Why VidaForge

Video data work is easy to describe vaguely and hard to study carefully. Raw videos come with mixed formats, broken files, variable frame rates, long takes, watermarks, repeated content, low-motion clips, and missing metadata. Once the raw pool gets large, every design choice becomes expensive: when to transcode, when to cut clips, how to keep rejected samples, how to compare selection recipes, and what exact format the training code will read.

VidaForge keeps the pipeline organized around data states:

```text
raw videos
  -> standardized videos
  -> video clips
  -> curated clips
  -> annotated clips
  -> training datasets
```

This is the main design bias of the project. Each stage leaves something concrete on disk, and the final output has to enter an actual video foundation model training loop.

## What VidaForge Does

VidaForge follows a five-stage pipeline:

1. **Ingestion**: probe raw videos, screen invalid inputs, and transcode videos into a standardized H.265/MP4 format.
2. **Segmentation**: detect scene boundaries and cut videos into 2-10 second clips.
3. **Selection**: extract context frames/audio, score quality signals, run hash-based and semantic deduplication, and write selection decisions.
4. **Annotation**: generate camera motion labels, multi-level captions, and structured semantic tags with VLM services.
5. **Packaging**: convert processed clips and metadata into dataset formats consumed by specific video pretraining codebases.

The pipeline stores media assets under `data/` and structured records under `meta/`. Metadata is written as Parquet shards so each stage can be resumed, inspected, and compared across data recipes.

## Repository Layout

```text
vidaforge/                  # reusable library code
recipe/                     # Hydra entrypoints for each stage
configs/                    # stage and step configs
scripts/                    # shell runners for common stage/step runs
README.md
pyproject.toml
```

The main entrypoints are:

```text
recipe/stage1_ingestion.py
recipe/stage2_segmentation.py
recipe/stage3_selection.py
recipe/stage4_annotation.py
recipe/stage5_packaging.py
```

## Installation

VidaForge targets Python 3.11 on Linux.

```bash
git clone git@github.com:GAIR-NLP/VidaForge.git
cd VidaForge

uv venv .venv --python 3.11
source .venv/bin/activate
uv sync
```

External tools and model assets are expected to be installed separately:

- FFmpeg / ffprobe for media probing, transcoding, clipping, and frame extraction.
- TransNetV2 weights if you use TransNetV2 scene detection.
- Aesthetic, OCR/text, Cosmos-Embed, and VLM models if you enable the corresponding Stage 3/4 steps.
- A Ray runtime if you run the distributed scripts with `RAY_ADDRESS=auto`.

## Data Directories

Most scripts use two paths:

```bash
RAW_DIR=/path/to/raw_videos
DATA_DIR=/path/to/vidaforge_output
```

`RAW_DIR` contains the original videos or raw video shards. `DATA_DIR` stores VidaForge outputs:

```text
DATA_DIR/
├─ data/     # video clips, frames, audio, tensor caches
└─ meta/     # Parquet metadata and summary.json files
```

## Quick Start

Stage runners are in `scripts/stage*/`. They are templates: set `RAW_DIR`, `DATA_DIR`, `RUN_ID`, and any model paths needed by the selected step.

```bash
export RAW_DIR=/path/to/raw_videos
export DATA_DIR=/path/to/vidaforge_output
export RUN_ID=example_run

bash scripts/stage1_ingestion/run_step1_probe.sh
bash scripts/stage1_ingestion/run_step2_screen.sh
bash scripts/stage1_ingestion/run_step3_transcode.sh

bash scripts/stage2_segmentation/run_step1_detect.sh
bash scripts/stage2_segmentation/run_step2_clip.sh
```

For a compact end-to-end template:

```bash
bash scripts/run_pipeline_example.sh probe,screen,transcode,detect,clip
```

The top-level example script is meant to be edited for a specific machine and model setup. It keeps the stage order in one place, while the stage-level scripts remain the safer way to debug individual steps.

## Stage Notes

### Stage 1: Ingestion

Probe scans raw inputs and writes initial video records. Screen skips obviously invalid videos using media properties such as resolution, fps, and duration. Transcode writes standardized videos and calibrated metadata for later stages.

### Stage 2: Segmentation

Detect writes candidate cut points for each standardized video. Clip then uses FFmpeg to cut scene-level video clips. Splitting detect and clip lets you inspect scene boundaries before producing a large number of clip files.

### Stage 3: Selection

Context extracts frames and optional audio for later filtering, annotation, and visual inspection. Filter writes quality signals such as motion, aesthetic, text, and low-level visual quality. Dedup supports hash-based near-duplicate matching with PDQ and semantic duplicate matching with video embeddings. Select combines quality signals and duplicate groups into a concrete data recipe.

### Stage 4: Annotation

Camera, caption, and tag are separate steps. Camera records camera motion and scene dynamics. Caption produces multi-level descriptions. Tag writes structured fields such as domain, scene, subjects, actions, style, text, and watermark for distribution analysis and sampling.

### Stage 5: Packaging

Packaging bridges processed clips and real training repositories. Current code focuses on target-specific outputs such as diffusion-model tensor caches and V-JEPA-style manifests. The exact output format depends on the downstream training code.

## Citation

If you find VidaForge useful, please cite:

```bibtex
@misc{ma2026vidaforge,
  title        = {{VidaForge}: Building a Video Foundation Model Pretraining Data Pipeline from Scratch in an Academic Lab},
  author       = {Ma, Yan and Su, Jiadi and Hu, Zhulin and Chern, Ethan and Zhang, Linhao and Mi, TianTian and Liu, Pengfei},
  year         = {2026},
  howpublished = {\url{https://github.com/GAIR-NLP/VidaForge}},
  note         = {Blog and open-source project}
}
```

## License

This project is released under the MIT License.
