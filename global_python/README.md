# Global Python variant

This directory contains the no-Conda version of the DROID metric-depth
pipeline. It uses the container image's Python only to create an isolated
virtual environment under the selected runtime directory. All downloaded
Python packages, including PyTorch and PyZED, are installed into that virtual
environment; the image's global Python packages are never modified.

The image must provide Python 3.10-3.13, an NVIDIA driver, and the system
commands checked by the scripts. Conda and administrator privileges are not
required. If the standard-library `venv` module is unavailable, the script
bootstraps `virtualenv` inside the runtime directory without installing it into
the global Python environment.

If the image does not provide the `zstd` command required by the ZED SDK
installer, the script installs Python `zstandard` into the isolated environment
and creates a decompression-only fallback under `<runtime_dir>/tools/bin`.
This also requires neither Conda nor `sudo`.

Rootless ZED SDK installation supports these pinned combinations:

- Ubuntu 22.04 with CUDA 12
- Ubuntu 24.04 with CUDA 12 or CUDA 13

The CUDA major version is detected from the image. Override it only when
necessary with `ZED_CUDA_MAJOR=12` or `ZED_CUDA_MAJOR=13`.

Run the complete download-and-convert pipeline with:

```bash
cd global_python

bash run_droid_depth_all.sh \
  "2-8" \
  /path/to/droid_depth_input \
  /path/to/droid_depth_output \
  "0,1,2,3,4,5,6,7" \
  /path/to/droid_depth_runtime \
  12
```

The scripts use `python3` by default. If the image's base executable has a
different name, select it explicitly:

```bash
PYTHON_BIN=python bash run_droid_depth_all.sh ...
```

The resulting environment is stored at:

```text
<runtime_dir>/python-env
```

It is reused by subsequent runs. Set `DROID_DEPTH_PYTHON_ENV` before invoking
the scripts to choose a different location.

Downloading and conversion can also be run separately:

```bash
bash download_droid_svo_inputs.sh \
  "2-15" \
  /path/to/shared/droid_depth_input

bash run_droid_depth_conversion.sh \
  "2-8" \
  /path/to/shared/droid_depth_input \
  /path/to/droid_depth_output \
  "0,1,2,3,4,5,6,7" \
  /path/to/droid_depth_runtime \
  12
```
