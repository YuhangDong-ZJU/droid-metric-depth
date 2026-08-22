# Global Python variant

This directory contains the no-Conda version of the DROID metric-depth
pipeline. It uses the container image's global Python environment and installs
missing Python dependencies with `pip`.

The image must provide Python with `pip`, an NVIDIA driver, and the system
commands checked by the scripts. Python 3.10 is recommended and is required for
the pinned automatic PyZED wheel installation.

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

The scripts use `python3` by default. If the global executable has a different
name, select it explicitly:

```bash
PYTHON_BIN=python bash run_droid_depth_all.sh ...
```

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
