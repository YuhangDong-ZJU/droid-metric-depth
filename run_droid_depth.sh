#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
INPUT_DIR="$ROOT/DATA/droid_depth_input"
OUTPUT_DIR="$ROOT/DATA/droid_depth_output"

usage() {
  echo "Usage:"
  echo "  bash $0 download <chunks>"
  echo "  bash $0 check <exp_name> [gpu_id]"
  echo "  bash $0 convert <chunks> <exp_name> [gpu_ids] [batch_size]"
  echo "  bash $0 visualize <episode> <exp_name> [camera] [gpu_id]"
}

check_name() {
  [[ "$1" =~ ^[A-Za-z0-9._-]+$ ]] || {
    echo "ERROR: invalid name: $1"
    exit 2
  }
}

if [[ -n "${MINIFORGE_HOME:-}" ]]; then
  export PATH="$MINIFORGE_HOME/bin:$PATH"
fi

case "${1:-}" in
  download)
    [[ $# -eq 2 ]] || { usage; exit 2; }
    exec bash "$ROOT/download_droid_svo_inputs.sh" \
      "$2" "$INPUT_DIR"
    ;;

  check)
    [[ $# -ge 2 && $# -le 3 ]] || { usage; exit 2; }
    check_name "$2"
    export DROID_DEPTH_CHECK_ONLY=1
    exec bash "$ROOT/run_droid_depth_conversion.sh" \
      "0" \
      "$INPUT_DIR" \
      "$OUTPUT_DIR" \
      "${3:-0}" \
      "$ROOT/Res/$2" \
      "1"
    ;;

  convert)
    [[ $# -ge 3 && $# -le 5 ]] || { usage; exit 2; }
    EXP_NAME="$3"
    check_name "$EXP_NAME"
    exec bash "$ROOT/run_droid_depth_conversion.sh" \
      "$2" \
      "$INPUT_DIR" \
      "$OUTPUT_DIR" \
      "${4:-all}" \
      "$ROOT/Res/$EXP_NAME" \
      "${5:-2}"
    ;;

  visualize)
    [[ $# -ge 3 && $# -le 5 ]] || { usage; exit 2; }
    [[ "$2" =~ ^[0-9]+$ ]] || {
      echo "ERROR: episode must be a non-negative integer: $2"
      exit 2
    }
    check_name "$3"
    CAMERA="${4:-both}"
    [[ "$CAMERA" =~ ^(both|1|2|external_1|external_2)$ ]] || {
      echo "ERROR: camera must be both, 1, 2, external_1 or external_2"
      exit 2
    }
    GPU_ID="${5:-0}"
    [[ "$GPU_ID" =~ ^[0-9]+$ ]] || {
      echo "ERROR: gpu_id must be a non-negative integer: $GPU_ID"
      exit 2
    }
    if ! command -v conda >/dev/null 2>&1; then
      echo "ERROR: conda is not installed or is not in PATH."
      exit 1
    fi
    eval "$(conda shell.bash hook)"
    ENV_NAME="droid_depth_convert"
    if ! conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
      echo "ERROR: $ENV_NAME is not ready; run the check command first."
      exit 1
    fi
    ZED_SDK_ROOT="$ROOT/Res/$3/zed-sdk-5.4.1"
    if [[ ! -r "$ZED_SDK_ROOT/lib/libsl_zed.so" ]]; then
      echo "ERROR: ZED SDK is not ready: $ZED_SDK_ROOT"
      exit 1
    fi
    export PYTHONNOUSERSITE=1
    export ZED_DIR="$ZED_SDK_ROOT"
    export LD_LIBRARY_PATH="$ZED_SDK_ROOT/lib:${LD_LIBRARY_PATH:-}"
    exec conda run --no-capture-output -n "$ENV_NAME" \
      python "$ROOT/visualize_droid_depth.py" \
      --input-dir "$INPUT_DIR" \
      --output-dir "$OUTPUT_DIR" \
      --episode "$2" \
      --camera "$CAMERA" \
      --gpu-id "$GPU_ID" \
      --overwrite
    ;;

  *)
    usage
    exit 2
    ;;
esac
