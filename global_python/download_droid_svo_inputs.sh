#!/usr/bin/env bash
set -Eeuo pipefail
export PYTHONNOUSERSITE=1

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: bash $0 <chunks> <download_dir> [repo_id]"
  echo "Example: bash $0 0-3 /data/droid_depth_input"
  exit 2
fi

CHUNKS="$1"
DOWNLOAD_DIR="$2"
REPO_ID="${3:-Sponbebob4258/droid-24k-external-svo}"
BASE_PYTHON="${PYTHON_BIN:-python3}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_ENV_DIR="${DROID_DEPTH_PYTHON_ENV:-$DOWNLOAD_DIR/.droid_depth_runtime/python-env}"

# shellcheck source=prepare_uv_env.sh
source "$SCRIPT_DIR/prepare_uv_env.sh"
prepare_uv_env "$BASE_PYTHON" "$PYTHON_ENV_DIR"
PYTHON_BIN="$DROID_DEPTH_PYTHON"
UV_BIN="$DROID_DEPTH_UV"

if ! "$PYTHON_BIN" - <<'PY'
import huggingface_hub
import hf_xet
PY
then
  echo "Installing download dependencies into the isolated Python environment."
  "$UV_BIN" pip install --python "$PYTHON_BIN" \
    huggingface_hub==1.28.0 hf_xet==1.6.0
fi

mkdir -p "$DOWNLOAD_DIR"
export HF_XET_HIGH_PERFORMANCE=1

"$PYTHON_BIN" - "$REPO_ID" "$CHUNKS" "$DOWNLOAD_DIR" <<'PY'
from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from huggingface_hub import hf_hub_download


def parse_chunks(value: str) -> list[int]:
    chunks: set[int] = set()
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start_text, end_text = item.split("-", 1)
            start, end = int(start_text), int(end_text)
            if start > end:
                raise ValueError(f"Invalid chunk range: {item}")
            chunks.update(range(start, end + 1))
        else:
            chunks.add(int(item))
    if not chunks or min(chunks) < 0:
        raise ValueError("No valid chunks were selected")
    return sorted(chunks)


repo_id = sys.argv[1]
chunks = parse_chunks(sys.argv[2])
download_dir = Path(sys.argv[3]).resolve()
download_workers = int(os.environ.get("DROID_DEPTH_DOWNLOAD_WORKERS", "8"))
if download_workers < 1:
    raise ValueError("DROID_DEPTH_DOWNLOAD_WORKERS must be positive")

print(f"Repository: {repo_id}", flush=True)
print(f"Chunks: {chunks}", flush=True)
print(f"Destination: {download_dir}", flush=True)
print(f"Direct-download workers: {download_workers}", flush=True)

def download_file(filename: str) -> Path:
    return Path(
        hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=filename,
            local_dir=download_dir,
        )
    )


# Download only the selected chunk manifests by their exact Hub paths. Unlike
# snapshot_download(), this does not enumerate every file in the repository.
manifest_names = [f"manifests/chunks/chunk-{chunk:03d}.jsonl" for chunk in chunks]
for manifest_name in manifest_names:
    print(f"Downloading manifest: {manifest_name}", flush=True)
    download_file(manifest_name)

episodes = 0
files = 0
total_bytes = 0
required_serials: set[str] = set()
expected_svo_bytes: dict[str, int] = {}
for chunk in chunks:
    manifest = download_dir / "manifests/chunks" / f"chunk-{chunk:03d}.jsonl"
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if int(row["episode_chunk"]) != chunk:
            raise RuntimeError(f"Chunk mismatch in {manifest}")
        cameras = row["cameras"]
        if set(cameras) != {"external_1", "external_2"}:
            raise RuntimeError(f"Unexpected cameras in episode {row['episode_index']}")
        for role in ("external_1", "external_2"):
            camera = cameras[role]
            serial = str(camera["serial"])
            required_serials.add(serial)
            svo_name = str(camera["svo_path"])
            expected_bytes = int(camera["bytes"])
            previous_bytes = expected_svo_bytes.setdefault(svo_name, expected_bytes)
            if previous_bytes != expected_bytes:
                raise RuntimeError(f"Conflicting sizes for {svo_name}")
            files += 1
            total_bytes += expected_bytes
        episodes += 1

calibration_names = [f"calibrations/SN{serial}.conf" for serial in sorted(required_serials)]
pending_svos = []
reused_svos = 0
for svo_name, expected_bytes in sorted(expected_svo_bytes.items()):
    svo = download_dir / svo_name
    if svo.is_file() and svo.stat().st_size == expected_bytes:
        reused_svos += 1
    else:
        pending_svos.append(svo_name)

pending_names = calibration_names + pending_svos
print(
    f"Download plan: episodes={episodes}, SVOs={len(expected_svo_bytes)}, "
    f"reused={reused_svos}, pending={len(pending_svos)}, "
    f"calibrations={len(calibration_names)}",
    flush=True,
)
if pending_names:
    report_every = max(1, len(pending_names) // 100)
    completed_bytes = 0
    with ThreadPoolExecutor(max_workers=download_workers) as pool:
        futures = {pool.submit(download_file, name): name for name in pending_names}
        for completed, future in enumerate(as_completed(futures), start=1):
            name = futures[future]
            path = future.result()
            expected_bytes = expected_svo_bytes.get(name)
            actual_bytes = path.stat().st_size
            if expected_bytes is not None and actual_bytes != expected_bytes:
                raise RuntimeError(
                    f"Size mismatch after download: {path} "
                    f"({actual_bytes} != {expected_bytes})"
                )
            completed_bytes += actual_bytes
            if completed == 1 or completed % report_every == 0 or completed == len(futures):
                print(
                    f"Downloaded {completed}/{len(futures)} files "
                    f"({completed_bytes / 2**30:.2f} GiB materialized)",
                    flush=True,
                )

for svo_name, expected_bytes in expected_svo_bytes.items():
    svo = download_dir / svo_name
    if not svo.is_file() or svo.stat().st_size != expected_bytes:
        raise RuntimeError(f"Missing or size-mismatched SVO: {svo}")

missing_calibrations = [
    download_dir / "calibrations" / f"SN{serial}.conf"
    for serial in sorted(required_serials)
    if not (download_dir / "calibrations" / f"SN{serial}.conf").is_file()
]
if missing_calibrations:
    preview = "\n".join(str(path) for path in missing_calibrations[:10])
    raise RuntimeError(
        f"Missing {len(missing_calibrations)} ZED calibration files, including:\n{preview}"
    )

print("Download and validation complete.", flush=True)
print(f"Episodes: {episodes}", flush=True)
print(f"SVO files: {files}", flush=True)
print(f"SVO bytes: {total_bytes}", flush=True)
print(f"ZED calibration files: {len(required_serials)}", flush=True)
PY
