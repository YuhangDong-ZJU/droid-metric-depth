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

  *)
    usage
    exit 2
    ;;
esac
