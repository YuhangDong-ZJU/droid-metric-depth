#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -lt 2 || $# -gt 6 ]]; then
  echo "Usage: bash $0 <chunks> <exp_name> [gpu_ids] [batch_size] [input_dataset_name] [output_dataset_name]"
  echo "Example: bash $0 2-8 h100_1 0,1,2,3,4,5,6,7 12"
  exit 2
fi

CHUNKS="$1"
EXP_NAME="$2"
GPU_IDS="${3:-all}"
BATCH_SIZE="${4:-2}"
INPUT_DATASET_NAME="${5:-droid_depth_input}"
OUTPUT_DATASET_NAME="${6:-droid_depth_output}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

for name in "$EXP_NAME" "$INPUT_DATASET_NAME" "$OUTPUT_DATASET_NAME"; do
  if [[ ! "$name" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "ERROR: names may contain only letters, numbers, '.', '_' and '-': $name"
    exit 2
  fi
done

if [[ -n "${DROID_DEPTH_CONDA_ROOT:-}" ]]; then
  if [[ ! -x "$DROID_DEPTH_CONDA_ROOT/bin/conda" ]]; then
    echo "ERROR: conda was not found at $DROID_DEPTH_CONDA_ROOT/bin/conda"
    exit 1
  fi
  export PATH="$DROID_DEPTH_CONDA_ROOT/bin:$PATH"
fi

INPUT_DIR="$SCRIPT_DIR/DATA/$INPUT_DATASET_NAME"
OUTPUT_DIR="$SCRIPT_DIR/DATA/$OUTPUT_DATASET_NAME"
WORK_DIR="$SCRIPT_DIR/Res/$EXP_NAME"

echo "Standard input path:  ./DATA/$INPUT_DATASET_NAME"
echo "Standard output path: ./DATA/$OUTPUT_DATASET_NAME"
echo "Standard runtime path: ./Res/$EXP_NAME"

exec bash "$SCRIPT_DIR/run_droid_depth_conversion.sh" \
  "$CHUNKS" \
  "$INPUT_DIR" \
  "$OUTPUT_DIR" \
  "$GPU_IDS" \
  "$WORK_DIR" \
  "$BATCH_SIZE"
