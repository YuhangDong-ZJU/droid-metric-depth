#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -lt 1 || $# -gt 3 ]]; then
  echo "Usage: bash $0 <chunks> [dataset_name] [repo_id]"
  echo "Example: bash $0 0-18 droid_depth_input"
  exit 2
fi

CHUNKS="$1"
DATASET_NAME="${2:-droid_depth_input}"
REPO_ID="${3:-Sponbebob4258/droid-24k-external-svo}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if [[ ! "$DATASET_NAME" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "ERROR: dataset_name may contain only letters, numbers, '.', '_' and '-'."
  exit 2
fi

if [[ -n "${DROID_DEPTH_CONDA_ROOT:-}" ]]; then
  if [[ ! -x "$DROID_DEPTH_CONDA_ROOT/bin/conda" ]]; then
    echo "ERROR: conda was not found at $DROID_DEPTH_CONDA_ROOT/bin/conda"
    exit 1
  fi
  export PATH="$DROID_DEPTH_CONDA_ROOT/bin:$PATH"
fi

INPUT_DIR="$SCRIPT_DIR/DATA/$DATASET_NAME"

echo "Standard dataset path: ./DATA/$DATASET_NAME"

exec bash "$SCRIPT_DIR/download_droid_svo_inputs.sh" \
  "$CHUNKS" \
  "$INPUT_DIR" \
  "$REPO_ID"
