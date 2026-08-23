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
BASE_PYTHON="${DROID_DEPTH_BASE_PYTHON:-${PYTHON_BIN:-python3}}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_ENV_DIR="${DROID_DEPTH_PYTHON_ENV:-$WORK_DIR/python-env}"
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

# shellcheck source=prepare_python_env.sh
source "$SCRIPT_DIR/prepare_python_env.sh"

detect_zed_cuda_major() {
  local detected
  if [[ "${ZED_CUDA_MAJOR:-auto}" =~ ^(12|13)$ ]]; then
    echo "${BASH_REMATCH[1]}"
    return 0
  fi
  detected="$($BASE_PYTHON -c \
    'import torch; print((torch.version.cuda or "").split(".")[0])' \
    2>/dev/null || true)"
  if [[ "$detected" =~ ^(12|13)$ ]]; then
    echo "$detected"
    return 0
  fi
  detected="$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: \([0-9][0-9]*\).*/\1/p' | head -n 1)"
  if [[ "$detected" =~ ^(12|13)$ ]]; then
    echo "$detected"
    return 0
  fi
  echo 12
}

select_zed_installer() {
  local cuda_major="$1"
  if [[ ! -r /etc/os-release ]]; then
    echo "ERROR: cannot identify the operating system from /etc/os-release." >&2
    return 1
  fi
  # shellcheck disable=SC1091
  source /etc/os-release
  if [[ "${ID:-}" != "ubuntu" ]]; then
    echo "ERROR: rootless ZED SDK installation requires Ubuntu." >&2
    echo "Detected: ${ID:-unknown} ${VERSION_ID:-unknown}" >&2
    return 1
  fi
  # A recent NVIDIA driver may advertise CUDA 13 even when an Ubuntu 22 image
  # intentionally uses the CUDA 12 runtime. The CUDA 12 ZED build remains
  # compatible with that newer driver and is the pinned Ubuntu 22 build here.
  if [[ "${VERSION_ID:-}" == "22.04" && "$cuda_major" == "13" ]]; then
    echo "Ubuntu 22.04 detected; selecting the pinned CUDA 12 ZED build." >&2
    cuda_major=12
  fi

  case "${VERSION_ID:-}:$cuda_major" in
    22.04:12)
      ZED_BUILD_ID="ubuntu22-cuda12"
      ZED_INSTALLER_NAME="ZED_SDK_Ubuntu22_cuda12_v${ZED_SDK_VERSION}.run"
      ZED_INSTALLER_URL="https://download.stereolabs.com/zedsdk/${ZED_SDK_VERSION}/cu12/ubuntu22"
      ZED_INSTALLER_SHA256="35edc822377c5b548fb80f251d8347702cfc2f064f6b9920e29feebe387aec26"
      ;;
    24.04:12)
      ZED_BUILD_ID="ubuntu24-cuda12"
      ZED_INSTALLER_NAME="ZED_SDK_Ubuntu24_cuda12_v${ZED_SDK_VERSION}.run"
      ZED_INSTALLER_URL="https://download.stereolabs.com/zedsdk/${ZED_SDK_VERSION}/cu12/ubuntu24"
      ZED_INSTALLER_SHA256="bbed0c5fc563cdf1b611d1ea5fdbecd8ae5d059f85d0cf56ca93d8a0d6706877"
      ;;
    24.04:13)
      ZED_BUILD_ID="ubuntu24-cuda13"
      ZED_INSTALLER_NAME="ZED_SDK_Ubuntu24_cuda13_v${ZED_SDK_VERSION}.run"
      ZED_INSTALLER_URL="https://download.stereolabs.com/zedsdk/${ZED_SDK_VERSION}/cu13/ubuntu24"
      ZED_INSTALLER_SHA256="b576415517e869f8346beb961b5faa554e3b2e2dde20ca2eaa72f03cbef321ed"
      ;;
    *)
      echo "ERROR: unsupported rootless ZED SDK combination: Ubuntu ${VERSION_ID:-unknown}, CUDA $cuda_major." >&2
      echo "Supported: Ubuntu 22.04/CUDA 12, Ubuntu 24.04/CUDA 12 or 13." >&2
      return 1
      ;;
  esac
  export ZED_BUILD_ID ZED_INSTALLER_NAME ZED_INSTALLER_URL ZED_INSTALLER_SHA256
}

ensure_zstd() {
  local shim_dir shim
  if command -v zstd >/dev/null 2>&1; then
    echo "zstd ready: $(command -v zstd)"
    return 0
  fi

  if ! "$PYTHON_BIN" -c 'import zstandard' >/dev/null 2>&1; then
    echo "Installing the Python zstandard fallback into the isolated environment."
    "$PYTHON_BIN" -m pip install --disable-pip-version-check zstandard==0.23.0
  fi

  shim_dir="$WORK_DIR/tools/bin"
  shim="$shim_dir/zstd"
  mkdir -p "$shim_dir"
  "$PYTHON_BIN" - "$shim" <<'PY'
from pathlib import Path
import os
import sys

path = Path(sys.argv[1])
program = f"""#!{sys.executable}
import sys
import zstandard

args = sys.argv[1:]
if "--version" in args or "-V" in args:
    print("zstd Python fallback (zstandard " + zstandard.__version__ + ")")
    raise SystemExit(0)

allowed = {{"-d", "-c", "-dc", "-cd", "--decompress", "--stdout", "-q", "--quiet"}}
unsupported = [arg for arg in args if arg not in allowed]
if unsupported:
    print("Unsupported zstd fallback arguments: " + " ".join(unsupported), file=sys.stderr)
    raise SystemExit(2)
if not any(arg in {{"-d", "-dc", "-cd", "--decompress"}} for arg in args):
    print("The zstd fallback only supports decompression.", file=sys.stderr)
    raise SystemExit(2)

zstandard.ZstdDecompressor().copy_stream(sys.stdin.buffer, sys.stdout.buffer)
"""
path.write_text(program, encoding="utf-8")
os.chmod(path, 0o700)
PY
  export PATH="$shim_dir:$PATH"
  zstd --version
  echo "System zstd was unavailable; using the runtime-local Python fallback: $shim"
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
  local installed_version installer actual_sha256 partial_root stale_root build_marker installed_build
  local cuda_major
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
  echo "User-local ZED SDK installed without sudo: $ZED_SDK_ROOT"
}

ensure_pyzed() {
  local wheel wheel_url actual_sha256 expected_sha256 installed_version pyzed_so python_tag
  python_tag="$($PYTHON_BIN -c 'import sys; print(f"cp{sys.version_info.major}{sys.version_info.minor}")')"
  case "$python_tag" in
    cp310) expected_sha256="7521e6d7da8e8a98603dd7a0302c18fba2455024c9e77ec186d1ee01a4a786e9" ;;
    cp311) expected_sha256="d132cd7e03e1a5749f9d258c33f9f3e4c8cee421fe7164a02a32f38b018da83f" ;;
    cp312) expected_sha256="554363fcaa76fc307c180cd1da386ce4a48bdf4200bd83827ef5a5404b8255e1" ;;
    cp313) expected_sha256="95d1272279c07af2c2c79f098c55fcc6c251ef3b4dd82c89bd6154cdb0e4e3e0" ;;
    *)
      echo "ERROR: automatic PyZED installation supports Python 3.10-3.13, got $python_tag."
      return 1
      ;;
  esac
  wheel="$WORK_DIR/pyzed-${PYZED_VERSION}-${python_tag}-${python_tag}-linux_x86_64.whl"
  wheel_url="https://download.stereolabs.com/zedsdk/${PYZED_VERSION}/whl/linux_x86_64/$(basename "$wheel")"
  installed_version="$($PYTHON_BIN -c \
    'import importlib.metadata as m; print(m.version("pyzed"))' 2>/dev/null || true)"
  if [[ "$installed_version" != "$PYZED_VERSION" ]]; then
    wget --https-only --continue --progress=dot:giga \
      --output-document="$wheel" "$wheel_url"
    actual_sha256="$(sha256sum "$wheel" | awk '{print $1}')"
    if [[ "$actual_sha256" != "$expected_sha256" ]]; then
      echo "ERROR: PyZED wheel SHA256 mismatch."
      echo "Expected: $expected_sha256"
      echo "Actual:   $actual_sha256"
      return 1
    fi
    "$PYTHON_BIN" -m pip install --force-reinstall --no-deps "$wheel"
  fi

  pyzed_so="$($PYTHON_BIN -c 'import importlib.util; print(importlib.util.find_spec("pyzed.sl").origin)')"
  if LD_LIBRARY_PATH="$ZED_SDK_ROOT/lib:${LD_LIBRARY_PATH:-}" \
      ldd "$pyzed_so" | grep -q 'not found'; then
    echo "ERROR: the user-local ZED SDK has unresolved shared-library dependencies."
    LD_LIBRARY_PATH="$ZED_SDK_ROOT/lib:${LD_LIBRARY_PATH:-}" ldd "$pyzed_so"
    return 1
  fi
  ZED_DIR="$ZED_SDK_ROOT" LD_LIBRARY_PATH="$ZED_SDK_ROOT/lib:${LD_LIBRARY_PATH:-}" \
    "$PYTHON_BIN" -c 'import pyzed.sl'
  echo "PyZED $PYZED_VERSION is linked to the user-local ZED SDK."
}

if [[ ! "$BATCH_SIZE" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: batch_size must be a positive integer, got: $BATCH_SIZE"
  exit 2
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi is not available."
  exit 1
fi
if [[ ! -f "$CONVERTER" ]]; then
  echo "ERROR: converter is missing: $CONVERTER"
  exit 1
fi
for command_name in ffmpeg git wget sha256sum ldd; do
  if ! command -v "$command_name" >/dev/null 2>&1; then
    echo "ERROR: required command is not available: $command_name"
    exit 1
  fi
done

mkdir -p "$WORK_DIR" "$OUTPUT_DIR"
prepare_python_env "$BASE_PYTHON" "$PYTHON_ENV_DIR"
PYTHON_BIN="$DROID_DEPTH_PYTHON"

if ! "$PYTHON_BIN" -c \
  "import torch, torchvision, xformers, timm, omegaconf, scipy, imageio, PIL, cv2, einops, huggingface_hub, hf_xet, numpy, zstandard; assert torch.__version__.startswith('2.4.1'); assert torchvision.__version__.startswith('0.19.1'); assert numpy.__version__ == '1.26.4'; assert cv2.__version__ == '4.11.0'" \
  >/dev/null 2>&1; then
  echo "Installing conversion dependencies into the isolated Python environment."
  "$PYTHON_BIN" -m pip install --disable-pip-version-check --upgrade pip setuptools wheel
  "$PYTHON_BIN" -m pip install \
    --index-url https://download.pytorch.org/whl/cu121 \
    torch==2.4.1 torchvision==0.19.1
  "$PYTHON_BIN" -m pip install \
    --index-url https://download.pytorch.org/whl/cu121 \
    --no-deps xformers==0.0.28.post1
  "$PYTHON_BIN" -m pip install \
    numpy==1.26.4 omegaconf==2.3.0 timm==1.0.22 scipy==1.15.3 \
    imageio==2.37.4 pillow einops==0.8.2 \
    opencv-python-headless==4.11.0.86 \
    huggingface_hub==1.28.0 hf_xet==1.6.0 zstandard==0.23.0
fi

if ! "$PYTHON_BIN" -m pip check; then
  echo "ERROR: dependency conflicts remain inside the isolated Python environment."
  exit 1
fi

"$PYTHON_BIN" - <<'PY'
import sys
import cv2
import einops
import hf_xet
import huggingface_hub
import imageio
import numpy
import omegaconf
import scipy
import timm
import torch
import torchvision
import xformers
from PIL import Image

if not torch.cuda.is_available():
    print("ERROR: torch.cuda.is_available() is False", file=sys.stderr)
    raise SystemExit(1)
print(f"Isolated Python ready: {sys.executable}", flush=True)
print(f"PyTorch: {torch.__version__}; CUDA runtime: {torch.version.cuda}", flush=True)
PY

ensure_zstd
ensure_zed_sdk
export ZED_DIR="$ZED_SDK_ROOT"
export LD_LIBRARY_PATH="$ZED_SDK_ROOT/lib:${LD_LIBRARY_PATH:-}"

ensure_pyzed

"$PYTHON_BIN" - "$INPUT_DIR" "$CHUNKS" "$ZED_SETTINGS_DIR" <<'PY'
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
"$PYTHON_BIN" - "$DPT_SOURCE" <<'PY'
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
CUDA_VISIBLE_DEVICES="" "$PYTHON_BIN" - "$FS_ROOT" <<'PY'
import sys

sys.path.insert(0, sys.argv[1])
from depth_anything.dpt import DPT_DINOv2

model = DPT_DINOv2(encoder="vitl", pretrained_dino=False)
assert model.pretrained.__class__.__name__ == "DinoVisionTransformer"
print("Pinned local DINOv2 validation complete.", flush=True)
PY

mkdir -p "$MODEL_DIR"
if [[ ! -f "$CHECKPOINT" ]]; then
  "$PYTHON_BIN" - "$MODEL_DIR" <<'PY'
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

echo "GPU workers: ${GPU_ARRAY[*]}"
echo "Batch size per GPU worker: $BATCH_SIZE"

export OMP_NUM_THREADS=4
export HF_XET_HIGH_PERFORMANCE=1

PIDS=()
for WORKER_INDEX in "${!GPU_ARRAY[@]}"; do
  GPU_ID="${GPU_ARRAY[$WORKER_INDEX]}"
  "$PYTHON_BIN" "$CONVERTER" \
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
