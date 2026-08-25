#!/usr/bin/env bash
set -Eeuo pipefail
export PYTHONNOUSERSITE=1

if [[ $# -lt 3 || $# -gt 6 ]]; then
  echo "Usage: bash $0 <chunks> <input_dir> <output_dir> [gpu_ids] [work_dir] [batch_size]"
  echo "Example: bash $0 0-3 /data/droid_depth_input /data/droid_depth_output 0,1,2,3 /data/droid_depth_runtime 8"
  exit 2
fi

CHUNKS="$1"
INPUT_DIR="$2"
OUTPUT_DIR="$3"
GPU_IDS="${4:-all}"
WORK_DIR="${5:-$HOME/droid_depth_runtime}"
BATCH_SIZE="${6:-2}"
CHECK_ONLY="${DROID_DEPTH_CHECK_ONLY:-0}"
ENV_NAME="droid_depth_convert"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONVERTER="$SCRIPT_DIR/convert_droid_depth.py"
FS_ROOT="$WORK_DIR/FoundationStereo"
MODEL_DIR="$WORK_DIR/23-51-11"
CHECKPOINT="$MODEL_DIR/model_best_bp2.pth"
CONFIG="$MODEL_DIR/cfg.yaml"
FS_COMMIT="6e8806816b533e4d13ddbb95ffa907b797060a62"
CHECKPOINT_SHA256="60e79bde9c6a00acea551625ff814fe06e5a6806e2c0c9829baee248de87c5f1"
ZED_SDK_VERSION="5.4.1"
ZED_SDK_ROOT="$WORK_DIR/zed-sdk-$ZED_SDK_VERSION"
ZED_SETTINGS_DIR="$INPUT_DIR/calibrations"
PYZED_VERSION="5.4"
PYZED_WHEEL_SHA256="7521e6d7da8e8a98603dd7a0302c18fba2455024c9e77ec186d1ee01a4a786e9"

detect_zed_cuda_major() {
  local detected
  if [[ "${ZED_CUDA_MAJOR:-auto}" =~ ^(12|13)$ ]]; then
    echo "${BASH_REMATCH[1]}"
    return 0
  fi
  if [[ "${CUDA_VERSION:-}" =~ ^(12|13)(\.|$) ]]; then
    echo "${BASH_REMATCH[1]}"
    return 0
  fi
  detected="$(nvcc --version 2>/dev/null \
    | sed -n 's/.*release \([0-9][0-9]*\)\..*/\1/p' \
    | head -n 1)"
  if [[ "$detected" =~ ^(12|13)$ ]]; then
    echo "$detected"
    return 0
  fi
  detected="$(nvidia-smi 2>/dev/null \
    | sed -n 's/.*CUDA Version: \([0-9][0-9]*\).*/\1/p' \
    | head -n 1)"
  if [[ "$detected" =~ ^(12|13)$ ]]; then
    echo "$detected"
    return 0
  fi
  echo 12
}

select_zed_installer() {
  local cuda_major="$1"
  local os_release_file="${DROID_OS_RELEASE_FILE:-/etc/os-release}"
  if [[ ! -r "$os_release_file" ]]; then
    echo "ERROR: cannot identify the operating system from $os_release_file." >&2
    return 1
  fi
  # shellcheck disable=SC1091
  source "$os_release_file"

  # Stereolabs does not publish a native Debian build. Debian 12 uses a newer
  # glibc than Ubuntu 22.04, so for offline SVO playback we can rootlessly
  # extract and strictly validate the Ubuntu 22.04/CUDA 12 binaries instead.
  # The checks below still fail fast on unresolved libraries or PyZED import
  # errors before any conversion workers are started.
  if [[ "${ID:-}" == "debian" && "${VERSION_ID:-}" == "12" ]]; then
    echo "WARNING: Debian 12 is not officially supported by the ZED SDK." >&2
    echo "Using the Ubuntu 22.04/CUDA 12 ZED build in experimental rootless compatibility mode." >&2
    cuda_major=12
  elif [[ "${ID:-}" != "ubuntu" ]]; then
    echo "ERROR: unsupported operating system for rootless ZED SDK installation." >&2
    echo "Detected: ${ID:-unknown} ${VERSION_ID:-unknown}" >&2
    return 1
  fi
  if [[ "${VERSION_ID:-}" == "22.04" && "$cuda_major" == "13" ]]; then
    echo "Ubuntu 22.04 detected; selecting the CUDA 12 ZED build." >&2
    cuda_major=12
  fi

  case "${ID:-}:${VERSION_ID:-}:$cuda_major" in
    ubuntu:22.04:12|debian:12:12)
      ZED_BUILD_ID="ubuntu22-cuda12"
      ZED_INSTALLER_NAME="ZED_SDK_Ubuntu22_cuda12_v${ZED_SDK_VERSION}.run"
      ZED_INSTALLER_URL="https://download.stereolabs.com/zedsdk/${ZED_SDK_VERSION}/cu12/ubuntu22"
      ZED_INSTALLER_SHA256="35edc822377c5b548fb80f251d8347702cfc2f064f6b9920e29feebe387aec26"
      ;;
    ubuntu:24.04:12)
      ZED_BUILD_ID="ubuntu24-cuda12"
      ZED_INSTALLER_NAME="ZED_SDK_Ubuntu24_cuda12_v${ZED_SDK_VERSION}.run"
      ZED_INSTALLER_URL="https://download.stereolabs.com/zedsdk/${ZED_SDK_VERSION}/cu12/ubuntu24"
      ZED_INSTALLER_SHA256="bbed0c5fc563cdf1b611d1ea5fdbecd8ae5d059f85d0cf56ca93d8a0d6706877"
      ;;
    ubuntu:24.04:13)
      ZED_BUILD_ID="ubuntu24-cuda13"
      ZED_INSTALLER_NAME="ZED_SDK_Ubuntu24_cuda13_v${ZED_SDK_VERSION}.run"
      ZED_INSTALLER_URL="https://download.stereolabs.com/zedsdk/${ZED_SDK_VERSION}/cu13/ubuntu24"
      ZED_INSTALLER_SHA256="b576415517e869f8346beb961b5faa554e3b2e2dde20ca2eaa72f03cbef321ed"
      ;;
    *)
      echo "ERROR: unsupported ZED SDK combination: ${ID:-unknown} ${VERSION_ID:-unknown}, CUDA $cuda_major." >&2
      echo "Supported: Ubuntu 22.04/CUDA 12, Ubuntu 24.04/CUDA 12 or 13; Debian 12 experimental/CUDA 12." >&2
      return 1
      ;;
  esac
  export ZED_BUILD_ID ZED_INSTALLER_NAME ZED_INSTALLER_URL ZED_INSTALLER_SHA256
}

get_local_zed_version() {
  local version_file="$ZED_SDK_ROOT/zed-config-version.cmake"
  if [[ ! -f "$version_file" ]]; then
    return 0
  fi
  sed -n 's/^[[:space:]]*set(PACKAGE_VERSION "\([^"]*\)").*/\1/p' \
    "$version_file" | head -n 1
}

ensure_zed_sdk() {
  local installed_version installer actual_sha256 partial_root stale_root
  local build_marker installed_build cuda_major
  cuda_major="$(detect_zed_cuda_major)"
  select_zed_installer "$cuda_major"
  build_marker="$ZED_SDK_ROOT/.droid_depth_build"
  installed_build="$(cat "$build_marker" 2>/dev/null || true)"
  installed_version="$(get_local_zed_version)"
  if [[ "$installed_version" == "$ZED_SDK_VERSION" \
      && -r "$ZED_SDK_ROOT/lib/libsl_zed.so" \
      && -r "$ZED_SDK_ROOT/resources/neural_depth_5.3.model" \
      && ( "$installed_build" == "$ZED_BUILD_ID" \
        || ( -z "$installed_build" && "$ZED_BUILD_ID" == "ubuntu22-cuda12" ) ) ]]; then
    printf '%s\n' "$ZED_BUILD_ID" > "$build_marker"
    echo "User-local ZED SDK ready: $ZED_SDK_ROOT ($installed_version, $ZED_BUILD_ID)"
    return 0
  fi

  if [[ "$(uname -m)" != "x86_64" ]]; then
    echo "ERROR: rootless ZED SDK installation supports x86_64 only."
    return 1
  fi

  installer="$WORK_DIR/$ZED_INSTALLER_NAME"
  echo "Downloading ZED SDK $ZED_SDK_VERSION ($ZED_BUILD_ID) for user-local extraction."
  wget --https-only --continue --progress=dot:giga \
    --output-document="$installer" "$ZED_INSTALLER_URL"
  if [[ ! -s "$installer" ]] || ! head -c 64 "$installer" | grep -q '^#!/bin/'; then
    echo "ERROR: downloaded ZED SDK installer is invalid: $installer"
    return 1
  fi
  actual_sha256="$(sha256sum "$installer" | awk '{print $1}')"
  if [[ "$actual_sha256" != "$ZED_INSTALLER_SHA256" ]]; then
    echo "ERROR: ZED SDK installer SHA256 mismatch."
    echo "Expected: $ZED_INSTALLER_SHA256"
    echo "Actual:   $actual_sha256"
    return 1
  fi

  chmod 700 "$installer"
  partial_root="${ZED_SDK_ROOT}.partial.$$"
  rm -rf -- "$partial_root"
  mkdir -p "$partial_root"
  (
    cd "$partial_root"
    "$installer" --tar xf \
      ./lib \
      ./resources \
      ./get_python_api.py \
      ./zed-config.cmake \
      ./zed-config-version.cmake \
      ./doc/license/LICENSE.txt
  )

  installed_version="$(sed -n 's/^[[:space:]]*set(PACKAGE_VERSION "\([^"]*\)").*/\1/p' \
    "$partial_root/zed-config-version.cmake" | head -n 1)"
  if [[ "$installed_version" != "$ZED_SDK_VERSION" \
      || ! -r "$partial_root/lib/libsl_zed.so" ]]; then
    echo "ERROR: extracted ZED SDK is incomplete or has the wrong version."
    return 1
  fi
  if [[ -e "$ZED_SDK_ROOT" ]]; then
    stale_root="${ZED_SDK_ROOT}.incomplete.$(date +%Y%m%d%H%M%S)"
    mv "$ZED_SDK_ROOT" "$stale_root"
    echo "Moved the previous incomplete SDK to: $stale_root"
  fi
  printf '%s\n' "$ZED_BUILD_ID" > "$partial_root/.droid_depth_build"
  mv "$partial_root" "$ZED_SDK_ROOT"
  rm -f "$installer"
  echo "User-local ZED SDK installed without sudo: $ZED_SDK_ROOT ($ZED_BUILD_ID)"
}

ensure_pyzed() {
  local wheel wheel_url actual_sha256 installed_version pyzed_so
  wheel="$WORK_DIR/pyzed-${PYZED_VERSION}-cp310-cp310-linux_x86_64.whl"
  wheel_url="https://download.stereolabs.com/zedsdk/${PYZED_VERSION}/whl/linux_x86_64/$(basename "$wheel")"
  installed_version="$($CONDA_PREFIX/bin/python -c \
    'import importlib.metadata as m; print(m.version("pyzed"))' 2>/dev/null || true)"
  if [[ "$installed_version" != "$PYZED_VERSION" ]]; then
    wget --https-only --continue --progress=dot:giga \
      --output-document="$wheel" "$wheel_url"
    actual_sha256="$(sha256sum "$wheel" | awk '{print $1}')"
    if [[ "$actual_sha256" != "$PYZED_WHEEL_SHA256" ]]; then
      echo "ERROR: PyZED wheel SHA256 mismatch."
      return 1
    fi
    "$CONDA_PREFIX/bin/python" -m pip install --force-reinstall --no-deps "$wheel"
  fi

  pyzed_so="$($CONDA_PREFIX/bin/python -c 'import importlib.util; print(importlib.util.find_spec("pyzed.sl").origin)')"
  "$CONDA_PREFIX/bin/patchelf" --set-rpath \
    "$ZED_SDK_ROOT/lib:/usr/local/cuda/lib64" "$pyzed_so"
  if LD_LIBRARY_PATH="$ZED_SDK_ROOT/lib:${LD_LIBRARY_PATH:-}" \
      ldd "$pyzed_so" | grep -q 'not found'; then
    echo "ERROR: the user-local ZED SDK has unresolved shared-library dependencies."
    LD_LIBRARY_PATH="$ZED_SDK_ROOT/lib:${LD_LIBRARY_PATH:-}" ldd "$pyzed_so"
    return 1
  fi
  ZED_DIR="$ZED_SDK_ROOT" LD_LIBRARY_PATH="$ZED_SDK_ROOT/lib:${LD_LIBRARY_PATH:-}" \
    "$CONDA_PREFIX/bin/python" -c 'import pyzed.sl'
  echo "PyZED $PYZED_VERSION is linked to the user-local ZED SDK."
}

if [[ ! "$BATCH_SIZE" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: batch_size must be a positive integer, got: $BATCH_SIZE"
  exit 2
fi
if [[ "$CHECK_ONLY" != "0" && "$CHECK_ONLY" != "1" ]]; then
  echo "ERROR: DROID_DEPTH_CHECK_ONLY must be 0 or 1."
  exit 2
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda is not installed or is not in PATH."
  exit 1
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi is not available."
  exit 1
fi
if [[ ! -f "$CONVERTER" ]]; then
  echo "ERROR: converter is missing: $CONVERTER"
  exit 1
fi

eval "$(conda shell.bash hook)"
mkdir -p "$WORK_DIR"
if [[ "$CHECK_ONLY" == "0" ]]; then
  mkdir -p "$OUTPUT_DIR"
fi

if ! conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
  conda create -n "$ENV_NAME" --override-channels -c conda-forge \
    python=3.10 pip ffmpeg zstd patchelf git wget -y
fi
CONDA_PREFIX="$(conda run -n "$ENV_NAME" python -c 'import sys; print(sys.prefix)')"
if [[ ! -x "$CONDA_PREFIX/bin/zstd" \
    || ! -x "$CONDA_PREFIX/bin/patchelf" \
    || ! -x "$CONDA_PREFIX/bin/git" \
    || ! -x "$CONDA_PREFIX/bin/wget" ]]; then
  conda install -n "$ENV_NAME" --override-channels -c conda-forge \
    zstd patchelf git wget -y
fi
export PATH="$CONDA_PREFIX/bin:$PATH"

ensure_zed_sdk
export ZED_DIR="$ZED_SDK_ROOT"
export LD_LIBRARY_PATH="$ZED_SDK_ROOT/lib:${LD_LIBRARY_PATH:-}"

if ! conda run -n "$ENV_NAME" python -c "import torch, torchvision, xformers, timm, omegaconf, scipy, imageio" >/dev/null 2>&1; then
  conda run -n "$ENV_NAME" python -m pip install --upgrade pip
  conda run -n "$ENV_NAME" python -m pip install \
    --index-url https://download.pytorch.org/whl/cu121 \
    torch==2.4.1 torchvision==0.19.1
  conda run -n "$ENV_NAME" python -m pip install \
    --index-url https://download.pytorch.org/whl/cu121 \
    --no-deps xformers==0.0.28.post1
  conda run -n "$ENV_NAME" python -m pip install \
    numpy==1.26.4 omegaconf==2.3.0 timm==1.0.22 scipy imageio pillow \
    opencv-python-headless einops huggingface_hub hf_xet
fi

ensure_pyzed

if [[ "$CHECK_ONLY" == "0" ]]; then
  conda run --no-capture-output -n "$ENV_NAME" \
    python - "$INPUT_DIR" "$CHUNKS" "$ZED_SETTINGS_DIR" <<'PY'
import json
import sys
from pathlib import Path


def parse_chunks(value: str) -> list[int]:
    chunks: set[int] = set()
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start_text, end_text = item.split("-", 1)
            chunks.update(range(int(start_text), int(end_text) + 1))
        else:
            chunks.add(int(item))
    return sorted(chunks)


input_dir = Path(sys.argv[1])
chunks = parse_chunks(sys.argv[2])
settings_dir = Path(sys.argv[3])
serials: set[str] = set()
for chunk in chunks:
    manifest = input_dir / "manifests/chunks" / f"chunk-{chunk:03d}.jsonl"
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        serials.update(str(camera["serial"]) for camera in row["cameras"].values())

missing = [settings_dir / f"SN{serial}.conf" for serial in sorted(serials)]
missing = [path for path in missing if not path.is_file()]
if missing:
    preview = "\n".join(str(path) for path in missing[:10])
    raise RuntimeError(f"Missing {len(missing)} ZED calibration files, including:\n{preview}")
print(f"ZED calibration preflight complete: {len(serials)} files.", flush=True)
PY
fi

if [[ ! -d "$FS_ROOT/.git" ]]; then
  git clone --recursive https://github.com/NVlabs/FoundationStereo.git "$FS_ROOT"
fi
git -C "$FS_ROOT" fetch --all --tags
git -C "$FS_ROOT" checkout --detach "$FS_COMMIT"
git -C "$FS_ROOT" submodule update --init --recursive

# FoundationStereo vendors a DINOv2 snapshot at the pinned commit. Its upstream
# code still asks Torch Hub for facebookresearch/dinov2, which lets multiple GPU
# workers race while downloading and extracting one shared cache. Rewrite that
# exact call to load the pinned local snapshot instead.
DINOV2_SOURCE="$FS_ROOT/dinov2"
DPT_SOURCE="$FS_ROOT/depth_anything/dpt.py"
if [[ ! -f "$DINOV2_SOURCE/dinov2/layers/block.py" ]]; then
  echo "ERROR: pinned FoundationStereo DINOv2 source is incomplete: $DINOV2_SOURCE"
  exit 1
fi
"$CONDA_PREFIX/bin/python" - "$DPT_SOURCE" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8")
remote_load = "        self.pretrained = torch.hub.load('facebookresearch/dinov2', 'dinov2_{:}14'.format(encoder), pretrained=pretrained_dino)"
local_load = """        self.pretrained = torch.hub.load(
            os.path.realpath(os.path.join(code_dir, '..', 'dinov2')),
            'dinov2_{:}14'.format(encoder),
            source='local',
            pretrained=pretrained_dino,
        )"""

if remote_load in text:
    path.write_text(text.replace(remote_load, local_load, 1), encoding="utf-8")
elif local_load not in text:
    raise RuntimeError(f"Unrecognized DINOv2 loader in {path}")
PY

echo "Validating pinned local DINOv2 before starting GPU workers."
CUDA_VISIBLE_DEVICES="" conda run --no-capture-output -n "$ENV_NAME" \
  python - "$FS_ROOT" <<'PY'
import sys

sys.path.insert(0, sys.argv[1])
from depth_anything.dpt import DPT_DINOv2

model = DPT_DINOv2(encoder="vitl", pretrained_dino=False)
assert model.pretrained.__class__.__name__ == "DinoVisionTransformer"
print("Pinned local DINOv2 validation complete.", flush=True)
PY

mkdir -p "$MODEL_DIR"
if [[ ! -f "$CHECKPOINT" ]]; then
  conda run --no-capture-output -n "$ENV_NAME" python - "$MODEL_DIR" <<'PY'
import sys
from huggingface_hub import hf_hub_download

hf_hub_download(
    repo_id="Felix-Zhenghao/FoundationStereo",
    filename="model_best_bp2.pth",
    local_dir=sys.argv[1],
)
PY
fi

cat >"$CONFIG" <<'YAML'
corr_implementation: reg
corr_levels: 2
corr_radius: 4
finetune_ckpt_name: model_best_bp2.pth
finetune_from: null
hidden_dims: [128, 128, 128]
img_gamma: null
inference_tile: 0
low_memory: 0
max_disp: 416
max_val_sample: null
mixed_precision: true
n_downsample: 2
n_gru_layers: 3
notes: ''
num_steps: 200000
num_worker: 8
slow_fast_gru: false
tags_more: []
tile_min_overlap: [16, 16]
tile_wtype: gaussian
time_limit: 14400
train_iters: 22
val_interval: 1
valid_iters: 32
wdecay: 0
world_size: 32
vit_size: vitl
YAML

ACTUAL_SHA256="$(sha256sum "$CHECKPOINT" | awk '{print $1}')"
if [[ "$ACTUAL_SHA256" != "$CHECKPOINT_SHA256" ]]; then
  echo "ERROR: FoundationStereo checkpoint SHA256 mismatch."
  exit 1
fi

if [[ "$GPU_IDS" == "all" ]]; then
  GPU_IDS="$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd, -)"
fi
IFS=',' read -r -a GPU_ARRAY <<<"$GPU_IDS"
if [[ ${#GPU_ARRAY[@]} -eq 0 ]]; then
  echo "ERROR: no GPU was selected."
  exit 1
fi

if [[ "$CHECK_ONLY" == "1" ]]; then
  CHECK_GPU="${GPU_ARRAY[0]}"
  echo "Running FoundationStereo smoke test on GPU $CHECK_GPU."
  conda run --no-capture-output -n "$ENV_NAME" \
    python - "$CONVERTER" "$FS_ROOT" "$CHECKPOINT" "$CONFIG" "$CHECK_GPU" <<'PY'
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

import numpy as np
import torch

converter_path, fs_root, checkpoint, config = map(Path, sys.argv[1:5])
gpu_id = int(sys.argv[5])
spec = spec_from_file_location("droid_depth_converter", converter_path)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot import converter: {converter_path}")
module = module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

if not torch.cuda.is_available():
    raise RuntimeError("torch.cuda.is_available() is False")
estimator = module.Estimator(fs_root, checkpoint, config, gpu_id, 2)
image = np.zeros((720, 1280, 3), dtype=np.uint8)
disparity = estimator.infer([image], [image])
if disparity.shape != (1, 720, 1280) or not np.isfinite(disparity).all():
    raise RuntimeError(f"Unexpected smoke-test output: {disparity.shape}")
print(
    f"Environment ready: GPU {gpu_id} ({torch.cuda.get_device_name(gpu_id)}), "
    f"PyTorch {torch.__version__}, CUDA {torch.version.cuda}",
    flush=True,
)
PY
  echo "Environment check complete: $WORK_DIR"
  exit 0
fi

echo "GPU workers: ${GPU_ARRAY[*]}"
echo "Batch size per GPU worker: $BATCH_SIZE"

export OMP_NUM_THREADS=4
export HF_XET_HIGH_PERFORMANCE=1

PIDS=()
for WORKER_INDEX in "${!GPU_ARRAY[@]}"; do
  GPU_ID="${GPU_ARRAY[$WORKER_INDEX]}"
  conda run --no-capture-output -n "$ENV_NAME" python "$CONVERTER" \
    --chunks "$CHUNKS" \
    --input-dir "$INPUT_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --zed-settings-dir "$ZED_SETTINGS_DIR" \
    --fs-root "$FS_ROOT" \
    --checkpoint "$CHECKPOINT" \
    --config "$CONFIG" \
    --checkpoint-sha256 "$CHECKPOINT_SHA256" \
    --gpu-id "$GPU_ID" \
    --worker-index "$WORKER_INDEX" \
    --num-workers "${#GPU_ARRAY[@]}" \
    --batch-size "$BATCH_SIZE" \
    --iters 32 &
  PIDS+=("$!")
done

FAILED=0
for PID in "${PIDS[@]}"; do
  if ! wait "$PID"; then
    FAILED=1
  fi
done

if [[ "$FAILED" -ne 0 ]]; then
  echo "ERROR: one or more GPU workers failed. Check $OUTPUT_DIR/logs/."
  exit 1
fi

echo "Depth conversion complete: $OUTPUT_DIR"
