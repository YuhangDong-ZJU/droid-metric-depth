# DROID Metric Depth

Standalone scripts for generating full-resolution metric depth annotations for the two external ZED cameras in the DROID 24K dataset.

The pipeline downloads selected SVO chunks from [`Sponbebob4258/droid-24k-external-svo`](https://huggingface.co/datasets/Sponbebob4258/droid-24k-external-svo), installs its own Conda environments, installs ZED SDK 5.4.1 without `sudo`, downloads a pinned FoundationStereo checkout and checkpoint, and runs one worker per selected GPU.

## Output

- Input and output resolution: `1280x720`
- Method: FoundationStereo hierarchical inference with `small_ratio=0.5`
- Default iterations: `32`
- Storage: lossless `uint16` PNG in millimetres
- Invalid value: `0`
- Retained depth range: `0.02-10.0 m`
- Cameras: `external_1` and `external_2` only
- Per-episode sidecars include the rectified intrinsic matrix, baseline, source information, inference settings, timestamps, and depth statistics

## Requirements

- Ubuntu 22.04 x86_64
- NVIDIA GPU and a working driver
- Conda available in `PATH`
- `git`, `curl`/`wget`, `tar`, `zstd`, and standard build/runtime utilities
- Outbound access to Hugging Face, GitHub, Stereolabs, PyPI, and the PyTorch wheel index
- Enough storage for the selected SVO chunks and lossless depth PNGs

No administrator privileges are required. The ZED SDK is extracted into the runtime directory supplied on the command line.

### uv-based global Python image variant (untested)

For container images that do not provide Conda, use the scripts in
[`global_python/`](global_python/). This path now bootstraps a pinned,
runtime-local `uv 0.12.5`, uses uv to create the isolated environment, and uses
`uv pip` for dependency installation. It does not install packages into the
image's global Python. Ubuntu 22.04/CUDA 12 and Ubuntu 24.04/CUDA 12 or 13 are
intended to work without `sudo`; however, the uv migration is currently marked
**untested**. The conversion and output format are otherwise unchanged.

## Quick start

```bash
git clone https://github.com/YuhangDong-ZJU/droid-metric-depth.git
cd droid-metric-depth

bash run_droid_depth_all.sh \
  "0-3" \
  /data/droid_depth_input \
  /data/droid_depth_output \
  "0,1,2,3,4,5,6,7" \
  /data/droid_depth_runtime \
  2
```

## Standard relative layout

For clusters that expose datasets and experiment resources through relative
paths, use the standard-layout wrappers. Paths are anchored to the repository
root, so the commands work regardless of the caller's current directory:

```text
droid-metric-depth/
├── DATA/
│   ├── droid_depth_input
│   └── droid_depth_output
└── Res/
    └── <exp_name>
```

If Miniforge is not already in `PATH`, provide its installation root once:

```bash
export DROID_DEPTH_CONDA_ROOT=/path/to/miniforge
```

Download with the default dataset name `droid_depth_input`:

```bash
bash download_standard_layout.sh "0-18"
```

Run conversion with an experiment-specific runtime/checkpoint directory:

```bash
bash run_standard_layout.sh \
  "2-8" \
  "h100_1" \
  "0,1,2,3,4,5,6,7" \
  12
```

This resolves the input, output, and runtime paths to
`./DATA/droid_depth_input`, `./DATA/droid_depth_output`, and
`./Res/h100_1`, respectively. The `DATA/` and `Res/` entries may be regular
directories or symlinks managed by the cluster.

The positional arguments are:

```text
<chunks> <input_dir> <output_dir> [gpu_ids] [runtime_dir] [batch_size] [repo_id]
```

Chunk selections accept ranges and comma-separated values, for example:

```text
0
0-3
0,2,5-7
```

The downloader first fetches only the selected chunk manifests, then downloads the exact SVO and calibration paths with parallel workers. Set `DROID_DEPTH_DOWNLOAD_WORKERS` to override the default of 8:

```bash
export DROID_DEPTH_DOWNLOAD_WORKERS=16
```

## Batch size

For `1280x720` hierarchical inference with 32 iterations:

- RTX 4090 24 GB: batch size `2` is the tested default (about 12.8 GiB per GPU)
- 80 GB H100: start with batch size `8`; increase to `12` after confirming stability

If a multi-frame batch runs out of memory, the converter retries those frames one at a time. Repeated fallback is safe but slower.

## Resume behaviour

The pipeline is restartable:

- valid local SVO files are reused
- valid completed depth episodes are skipped
- incomplete temporary output directories are not treated as completed episodes
- failed episodes do not stop the remaining jobs in that worker

Run the same command again to resume after an interruption.

## Upstream software

This repository does not redistribute the ZED SDK, FoundationStereo source, DINOv2 assets, or model weights. The scripts download them from their upstream locations and verify pinned versions/checksums where applicable. Users are responsible for complying with the DROID, FoundationStereo, Stereolabs ZED SDK, model-weight, and dataset licences.
