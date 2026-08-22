#!/usr/bin/env bash
set -Eeuo pipefail
export PYTHONNOUSERSITE=1

if [[ $# -lt 3 || $# -gt 7 ]]; then
  echo "Usage: bash $0 <chunks> <input_dir> <output_dir> [gpu_ids] [work_dir] [batch_size] [repo_id]"
  echo "Example: bash $0 0-3 /data/droid_depth_input /data/droid_depth_output 0,1,2,3,4,5,6,7 /data/droid_depth_runtime 8"
  exit 2
fi

CHUNKS="$1"
INPUT_DIR="$2"
OUTPUT_DIR="$3"
GPU_IDS="${4:-all}"
WORK_DIR="${5:-$HOME/droid_depth_runtime}"
BATCH_SIZE="${6:-2}"
REPO_ID="${7:-Sponbebob4258/droid-24k-external-svo}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

bash "$SCRIPT_DIR/download_droid_svo_inputs.sh" \
  "$CHUNKS" \
  "$INPUT_DIR" \
  "$REPO_ID"

bash "$SCRIPT_DIR/run_droid_depth_conversion.sh" \
  "$CHUNKS" \
  "$INPUT_DIR" \
  "$OUTPUT_DIR" \
  "$GPU_IDS" \
  "$WORK_DIR" \
  "$BATCH_SIZE"
