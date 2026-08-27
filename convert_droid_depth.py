#!/usr/bin/env python3
"""Convert downloaded DROID external SVOs to native 1280x720 FS depth."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from queue import Full, Queue
from threading import Event, Thread
from types import ModuleType
from typing import Any, Iterator, Sequence

import imageio.v3 as iio
import numpy as np


ROLES = ("external_1", "external_2")
DEPTH_KEYS = {
    "external_1": "observation.images.depth_01",
    "external_2": "observation.images.depth_02",
}
EXPECTED_SHAPE = (720, 1280)
MIN_DEPTH_MM = 20.0
MAX_DEPTH_MM = 10_000.0
MAX_TAIL_SHORTFALL = 2
INTERPOLATE_INT_MAX_ERROR = (
    "upsample_bilinear2d_nhwc only supports output tensors with less than INT_MAX elements"
)
FATAL_CUDA_ERROR_MARKERS = (
    "an illegal memory access was encountered",
    "cudaerrorillegaladdress",
    "code=700",
    "err [700]",
    "device-side assert triggered",
    "unspecified launch failure",
)
_STOP = object()


class ConversionError(RuntimeError):
    pass


def is_fatal_cuda_error(error: BaseException) -> bool:
    """Return whether a fresh process is required to obtain a valid CUDA context."""
    seen: set[int] = set()
    current: BaseException | None = error
    messages: list[str] = []
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        messages.append(f"{type(current).__name__}: {current}".lower())
        current = current.__cause__ or current.__context__
    combined = "\n".join(messages)
    return any(marker in combined for marker in FATAL_CUDA_ERROR_MARKERS)


@dataclass(frozen=True)
class StereoFrame:
    left_rgb: np.ndarray
    right_rgb: np.ndarray
    intrinsic: np.ndarray
    baseline_m: float
    timestamp_ms: float
    svo_frame_count: int


@dataclass(frozen=True)
class ConversionResult:
    status: str
    tail_missing_count: int = 0
    tail_retry_recovered_frames: int = 0


@dataclass(frozen=True)
class _PrefetchFailure:
    error: BaseException


class PrefetchedStereo:
    def __init__(self, source: Iterator[StereoFrame], capacity: int) -> None:
        self.source = source
        self.queue: Queue[StereoFrame | _PrefetchFailure | object] = Queue(capacity)
        self.stopped = Event()
        self.thread = Thread(target=self._run, daemon=True)
        self.thread.start()

    def _put(self, value: StereoFrame | _PrefetchFailure | object) -> bool:
        while not self.stopped.is_set():
            try:
                self.queue.put(value, timeout=0.1)
                return True
            except Full:
                continue
        return False

    def _run(self) -> None:
        try:
            for frame in self.source:
                if not self._put(frame):
                    return
            self._put(_STOP)
        except BaseException as error:
            self._put(_PrefetchFailure(error))
        finally:
            close = getattr(self.source, "close", None)
            if close is not None:
                close()

    def __iter__(self) -> PrefetchedStereo:
        return self

    def __next__(self) -> StereoFrame:
        value = self.queue.get()
        if value is _STOP:
            raise StopIteration
        if isinstance(value, _PrefetchFailure):
            raise value.error
        return value

    def close(self) -> None:
        self.stopped.set()
        self.thread.join()

    def __enter__(self) -> PrefetchedStereo:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


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
        raise ValueError("No valid chunks selected")
    return sorted(chunks)


def read_rows(input_dir: Path, chunks: Sequence[int]) -> list[dict[str, Any]]:
    rows = []
    for chunk in chunks:
        path = input_dir / "manifests/chunks" / f"chunk-{chunk:03d}.jsonl"
        with path.open(encoding="utf-8") as handle:
            chunk_rows = [json.loads(line) for line in handle if line.strip()]
        for row in chunk_rows:
            if int(row["episode_chunk"]) != chunk:
                raise ConversionError(f"Chunk mismatch in {path}")
            if set(row["cameras"]) != set(ROLES):
                raise ConversionError(f"Unexpected camera set in episode {row['episode_index']}")
        rows.extend(chunk_rows)
    return rows


def depth_dir(output_dir: Path, row: dict[str, Any], role: str) -> Path:
    return (
        output_dir
        / "images"
        / f"chunk-{int(row['episode_chunk']):03d}"
        / DEPTH_KEYS[role]
        / f"episode_{int(row['episode_index']):06d}"
    )


def metadata_path(output_dir: Path, row: dict[str, Any], role: str) -> Path:
    return (
        output_dir
        / "annotations/foundation_stereo_depth"
        / f"chunk-{int(row['episode_chunk']):03d}"
        / DEPTH_KEYS[role]
        / f"episode_{int(row['episode_index']):06d}.json"
    )


def source_svo(input_dir: Path, row: dict[str, Any], role: str) -> Path:
    camera = row["cameras"][role]
    path = input_dir / camera["svo_path"]
    if not path.is_file() or path.stat().st_size != int(camera["bytes"]):
        raise ConversionError(f"Missing or size-mismatched SVO: {path}")
    return path


def depth_is_valid(path: Path, frame_count: int) -> bool:
    frames = sorted(path.glob("frame_*.png")) if path.is_dir() else []
    if len(frames) != frame_count:
        return False
    if any(frame.name != f"frame_{index:06d}.png" for index, frame in enumerate(frames)):
        return False
    try:
        for frame in (frames[0], frames[-1]):
            depth = iio.imread(frame)
            if depth.shape != EXPECTED_SHAPE or depth.dtype != np.uint16:
                return False
    except (OSError, ValueError):
        return False
    return True


def metadata_is_valid(
    path: Path,
    row: dict[str, Any],
    role: str,
    checkpoint_sha256: str,
    iters: int,
) -> bool:
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
        frame_count = int(row["length"])
        decoded_frames = int(metadata["source"]["decoded_frame_count"])
        tail_missing_count = int(metadata["source"]["tail_missing_count"])
        return (
            metadata["schema_version"] == "1.1"
            and int(metadata["source"]["episode_index"]) == int(row["episode_index"])
            and metadata["source"]["camera_role"] == role
            and int(metadata["source"]["frame_count"]) == frame_count
            and decoded_frames + tail_missing_count == frame_count
            and 0 <= tail_missing_count <= MAX_TAIL_SHORTFALL
            and len(metadata["source"]["timestamps_ms"]) == frame_count
            and metadata["inference"]["checkpoint_sha256"] == checkpoint_sha256
            and metadata["inference"]["method"] == "FoundationStereo hierarchical"
            and int(metadata["inference"]["iters"]) == iters
            and float(metadata["inference"]["small_ratio"]) == 0.5
            and metadata["inference"]["input_resolution"] == [1280, 720]
            and metadata["inference"]["output_resolution"] == [1280, 720]
            and metadata["inference"]["resized"] is False
            and metadata["depth"]["storage"] == "uint16 millimeter PNG"
            and int(metadata["depth"]["invalid_value"]) == 0
            and metadata["depth"]["valid_range_m"] == [0.02, 10.0]
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


class PngWriter:
    def __init__(self, destination: Path, fps: int, gpu_id: int) -> None:
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise ConversionError("ffmpeg is not available")
        self.destination = destination
        self.temporary = destination.with_name(
            f".{destination.name}.tmp.gpu{gpu_id}.{os.getpid()}"
        )
        shutil.rmtree(self.temporary, ignore_errors=True)
        self.temporary.mkdir(parents=True, exist_ok=True)
        self.process = subprocess.Popen(
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-nostdin",
                "-y",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "gray16le",
                "-s",
                "1280x720",
                "-r",
                str(fps),
                "-i",
                "-",
                "-vcodec",
                "png",
                "-start_number",
                "0",
                "-compression_level",
                "6",
                "-pred",
                "sub",
                str(self.temporary / "frame_%06d.png"),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        self.frame_count = 0

    def write(self, depth: np.ndarray) -> None:
        if depth.shape != EXPECTED_SHAPE or depth.dtype != np.uint16:
            raise ValueError(f"Unexpected depth frame: {depth.shape}/{depth.dtype}")
        if self.process.stdin is None:
            raise ConversionError("ffmpeg stdin is unavailable")
        self.process.stdin.write(np.ascontiguousarray(depth.astype("<u2", copy=False)).tobytes())
        self.frame_count += 1

    def close(self, expected_frames: int) -> None:
        if self.process.stdin is not None:
            self.process.stdin.close()
        stderr = self.process.stderr.read().decode(errors="replace") if self.process.stderr else ""
        return_code = self.process.wait()
        if return_code or self.frame_count != expected_frames:
            self.cleanup()
            raise ConversionError(
                stderr.strip() or f"PNG frame count {self.frame_count} != {expected_frames}"
            )
        if len(list(self.temporary.glob("frame_*.png"))) != expected_frames:
            self.cleanup()
            raise ConversionError("Temporary PNG validation failed")
        if self.destination.exists():
            shutil.rmtree(self.destination)
        self.destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(self.temporary, self.destination)

    def cleanup(self) -> None:
        if self.process.poll() is None:
            self.process.kill()
            self.process.wait()
        shutil.rmtree(self.temporary, ignore_errors=True)


@dataclass(frozen=True)
class _InferredFrame:
    disparity: np.ndarray
    intrinsic: np.ndarray
    baseline_m: float
    timestamp_ms: float


@dataclass(frozen=True)
class _ReadyDepth:
    depth: np.ndarray
    timestamp_ms: float | None


class AsyncDepthWriter:
    def __init__(self, destination: Path, fps: int, gpu_id: int, capacity: int) -> None:
        self.writer = PngWriter(destination, fps, gpu_id)
        self.queue: Queue[_InferredFrame | _ReadyDepth | object] = Queue(capacity)
        self.error: BaseException | None = None
        self.valid_pixels = 0
        self.total_pixels = 0
        self.timestamps: list[float | None] = []
        self.thread = Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self) -> None:
        while True:
            value = self.queue.get()
            try:
                if value is _STOP:
                    return
                if self.error is None:
                    if isinstance(value, _InferredFrame):
                        depth = depth_from_disparity(
                            value.disparity,
                            value.intrinsic,
                            value.baseline_m,
                        )
                        timestamp_ms = value.timestamp_ms
                    elif isinstance(value, _ReadyDepth):
                        depth = value.depth
                        timestamp_ms = value.timestamp_ms
                    else:
                        raise TypeError(f"Unexpected depth queue item: {type(value).__name__}")
                    self.writer.write(depth)
                    self.valid_pixels += int(np.count_nonzero(depth))
                    self.total_pixels += depth.size
                    self.timestamps.append(timestamp_ms)
            except BaseException as error:
                self.error = error
            finally:
                self.queue.task_done()

    def _raise(self) -> None:
        if self.error is not None:
            raise ConversionError("Asynchronous PNG writer failed") from self.error

    def write_disparity(self, frame: StereoFrame, disparity: np.ndarray) -> None:
        self._raise()
        self.queue.put(
            _InferredFrame(
                disparity=disparity,
                intrinsic=frame.intrinsic,
                baseline_m=frame.baseline_m,
                timestamp_ms=frame.timestamp_ms,
            )
        )
        self._raise()

    def write_depth(self, depth: np.ndarray, timestamp_ms: float | None) -> None:
        self._raise()
        self.queue.put(_ReadyDepth(depth, timestamp_ms))
        self._raise()

    def close(self, expected_frames: int) -> None:
        self.queue.put(_STOP)
        self.queue.join()
        self.thread.join()
        self._raise()
        self.writer.close(expected_frames)

    def cleanup(self) -> None:
        if self.thread.is_alive():
            self.queue.put(_STOP)
            self.queue.join()
            self.thread.join()
        self.writer.cleanup()


def iter_stereo(
    svo_path: Path,
    frame_count: int,
    gpu_id: int,
    zed_settings_dir: Path,
    *,
    frame_start: int = 0,
) -> Iterator[StereoFrame]:
    """Read a stereo prefix, tolerating only a one/two-frame EOF at the tail.

    A nonzero ``frame_start`` still replays from SVO position zero so codecs can
    rebuild their state, but it only retrieves and yields frames from that index.
    """
    import pyzed.sl as sl

    if not 0 <= frame_start < frame_count:
        raise ValueError(f"Expected 0 <= frame_start < {frame_count}, got {frame_start}")

    init = sl.InitParameters()
    init.set_from_svo_file(str(svo_path))
    # The default (1) prints one INFO block for every SVO open. With thousands
    # of episodes this obscures the worker progress without adding diagnostics;
    # open failures are still surfaced through the returned ERROR_CODE below.
    init.sdk_verbose = 0
    # ZED expects a directory path with a trailing separator here. Without the
    # separator it silently falls back to /usr/local/zed/settings and attempts
    # a network download when the factory calibration is not installed there.
    init.optional_settings_path = str(zed_settings_dir.resolve()) + os.sep
    init.svo_real_time_mode = False
    init.depth_mode = sl.DEPTH_MODE.NONE
    init.sdk_gpu_id = gpu_id
    init.coordinate_units = sl.UNIT.METER
    camera = sl.Camera()
    error = camera.open(init)
    if error != sl.ERROR_CODE.SUCCESS:
        raise ConversionError(f"Cannot open {svo_path}: {error}")
    try:
        svo_frames = int(camera.get_svo_number_of_frames())
        if frame_count - svo_frames > MAX_TAIL_SHORTFALL:
            raise ConversionError(
                f"SVO frame count {svo_frames} is more than {MAX_TAIL_SHORTFALL} "
                f"frames short of {frame_count}: {svo_path}"
            )
        calibration = camera.get_camera_information().camera_configuration.calibration_parameters
        left_calibration = calibration.left_cam
        intrinsic = np.array(
            [
                [left_calibration.fx, 0.0, left_calibration.cx],
                [0.0, left_calibration.fy, left_calibration.cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        baseline = float(calibration.stereo_transform.get_translation().get()[0])
        if not 0.01 < baseline < 1.0:
            raise ConversionError(f"Implausible baseline {baseline} m: {svo_path}")
        camera.set_svo_position(0)
        left_mat, right_mat = sl.Mat(), sl.Mat()
        runtime = sl.RuntimeParameters()
        for frame_index in range(frame_count):
            error = camera.grab(runtime)
            if (
                error == sl.ERROR_CODE.END_OF_SVOFILE_REACHED
                and frame_count - frame_index <= MAX_TAIL_SHORTFALL
            ):
                return
            if error != sl.ERROR_CODE.SUCCESS:
                raise ConversionError(
                    f"Cannot grab frame {frame_index} from {svo_path}: {error}"
                )
            if frame_index < frame_start:
                continue
            camera.retrieve_image(left_mat, sl.VIEW.LEFT)
            camera.retrieve_image(right_mat, sl.VIEW.RIGHT)
            left = np.ascontiguousarray(left_mat.get_data().copy()[..., 2::-1])
            right = np.ascontiguousarray(right_mat.get_data().copy()[..., 2::-1])
            if left.shape != (720, 1280, 3) or right.shape != left.shape:
                raise ConversionError(f"Unexpected stereo shape {left.shape}/{right.shape}")
            timestamp = float(camera.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_milliseconds())
            yield StereoFrame(left, right, intrinsic, baseline, timestamp, svo_frames)
    finally:
        camera.close()


def install_utils_shim() -> None:
    module = ModuleType("Utils")

    def freeze_model(model: Any) -> Any:
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad = False
        for buffer in model.buffers():
            buffer.requires_grad = False
        return model

    def get_resize_keep_aspect_ratio(
        height: int,
        width: int,
        divider: int = 16,
        max_H: int = 1232,
        max_W: int = 1232,
    ) -> tuple[int, int]:
        rounded_height = int(np.ceil(height / divider) * divider)
        rounded_width = int(np.ceil(width / divider) * divider)
        if rounded_height > max_H or rounded_width > max_W:
            if rounded_height > rounded_width:
                rounded_width = int(np.ceil((rounded_width * max_H / rounded_height) / divider) * divider)
                rounded_height = max_H
            else:
                rounded_height = int(np.ceil((rounded_height * max_W / rounded_width) / divider) * divider)
                rounded_width = max_W
        return rounded_height, rounded_width

    module.freeze_model = freeze_model
    module.get_resize_keep_aspect_ratio = get_resize_keep_aspect_ratio
    sys.modules["Utils"] = module


class Estimator:
    def __init__(self, fs_root: Path, checkpoint: Path, config: Path, gpu_id: int, iters: int) -> None:
        import torch
        from omegaconf import OmegaConf

        torch.cuda.set_device(gpu_id)
        install_utils_shim()
        sys.path.insert(0, str(fs_root))
        from core.foundation_stereo import FoundationStereo
        from core.utils.utils import InputPadder

        cfg = OmegaConf.load(config)
        cfg["vit_size"] = "vitl"
        model = FoundationStereo(cfg)
        try:
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(checkpoint, map_location="cpu")
        state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
        model.load_state_dict(state)
        self.gpu_id = gpu_id
        self.device = f"cuda:{gpu_id}"
        self.model = model.to(self.device).eval()
        self.torch = torch
        self.InputPadder = InputPadder
        self.iters = iters
        self.oom_fallbacks = 0
        self.size_limit_fallbacks = 0
        self.max_inference_batch_size: int | None = None
        self.max_successful_batch_size = 0

    def _infer(self, left_images: Sequence[np.ndarray], right_images: Sequence[np.ndarray]) -> np.ndarray:
        torch = self.torch
        left = torch.as_tensor(np.stack(left_images), device=self.device).float().permute(0, 3, 1, 2).contiguous()
        right = torch.as_tensor(np.stack(right_images), device=self.device).float().permute(0, 3, 1, 2).contiguous()
        padder = self.InputPadder(left.shape, divis_by=32, force_square=False)
        left, right = padder.pad(left, right)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            disparity = self.model.run_hierachical(
                left,
                right,
                iters=self.iters,
                test_mode=True,
                small_ratio=0.5,
            )
        disparity = padder.unpad(disparity.float()).reshape(len(left_images), 720, 1280)
        self.max_successful_batch_size = max(
            self.max_successful_batch_size,
            len(left_images),
        )
        return disparity.detach().cpu().numpy()

    def _infer_split(
        self,
        left_images: Sequence[np.ndarray],
        right_images: Sequence[np.ndarray],
        midpoint: int,
    ) -> np.ndarray:
        return np.concatenate(
            (
                self.infer(left_images[:midpoint], right_images[:midpoint]),
                self.infer(left_images[midpoint:], right_images[midpoint:]),
            ),
            axis=0,
        )

    def infer(self, left_images: Sequence[np.ndarray], right_images: Sequence[np.ndarray]) -> np.ndarray:
        if (
            self.max_inference_batch_size is not None
            and len(left_images) > self.max_inference_batch_size
        ):
            return self._infer_split(
                left_images,
                right_images,
                self.max_inference_batch_size,
            )
        try:
            return self._infer(left_images, right_images)
        except RuntimeError as error:
            is_oom = isinstance(error, self.torch.cuda.OutOfMemoryError)
            is_size_limit = INTERPOLATE_INT_MAX_ERROR in str(error)
            if len(left_images) == 1 or not (is_oom or is_size_limit):
                raise
            self.oom_fallbacks += int(is_oom)
            self.size_limit_fallbacks += int(is_size_limit)
            midpoint = len(left_images) // 2
            reason = "CUDA OOM" if is_oom else "operator size limit"
        # Retry only after leaving the exception handler. Its traceback can
        # retain large CUDA tensors until the exception variable is cleared.
        self.torch.cuda.empty_cache()
        print(
            f"[GPU {self.gpu_id}] inference batch {len(left_images)} hit {reason}; "
            f"retrying with chunks of at most {midpoint}",
            flush=True,
        )
        if self.max_inference_batch_size is None:
            self.max_inference_batch_size = midpoint
        else:
            self.max_inference_batch_size = min(
                self.max_inference_batch_size,
                midpoint,
            )
        return self._infer_split(left_images, right_images, midpoint)


def depth_from_disparity(disparity: np.ndarray, intrinsic: np.ndarray, baseline: float) -> np.ndarray:
    x = np.arange(disparity.shape[1], dtype=np.float32)[None, :]
    valid = np.isfinite(disparity) & (disparity > 0) & ((x - disparity) >= 0)
    depth_mm_float = np.zeros(disparity.shape, dtype=np.float32)
    depth_mm_float[valid] = intrinsic[0, 0] * baseline * 1000.0 / disparity[valid]
    valid &= (
        np.isfinite(depth_mm_float)
        & (depth_mm_float >= MIN_DEPTH_MM)
        & (depth_mm_float <= MAX_DEPTH_MM)
    )
    depth = np.zeros(disparity.shape, dtype=np.uint16)
    depth[valid] = np.rint(depth_mm_float[valid]).astype(np.uint16)
    return depth


def convert_one(
    estimator: Estimator,
    input_dir: Path,
    output_dir: Path,
    zed_settings_dir: Path,
    row: dict[str, Any],
    role: str,
    gpu_id: int,
    batch_size: int,
    checkpoint_sha256: str,
    overwrite: bool,
    prefetch: bool,
) -> ConversionResult:
    destination = depth_dir(output_dir, row, role)
    sidecar = metadata_path(output_dir, row, role)
    frame_count = int(row["length"])
    if (
        not overwrite
        and depth_is_valid(destination, frame_count)
        and metadata_is_valid(sidecar, row, role, checkpoint_sha256, estimator.iters)
    ):
        return ConversionResult("skipped")

    estimator.max_successful_batch_size = 0
    svo = source_svo(input_dir, row, role)
    camera = row["cameras"][role]
    calibration_file = zed_settings_dir / f"SN{camera['serial']}.conf"
    if not calibration_file.is_file():
        raise ConversionError(f"Missing ZED calibration file: {calibration_file}")
    calibration_sha256 = hashlib.sha256(calibration_file.read_bytes()).hexdigest()
    prefetch_capacity = max(batch_size * 2, 4) if prefetch else 0
    writer_capacity = max(batch_size, 4)
    writer = AsyncDepthWriter(destination, 15, gpu_id, writer_capacity)
    pending: list[StereoFrame] = []
    first: StereoFrame | None = None
    inference_seconds = 0.0
    started = time.perf_counter()
    oom_fallback_start = estimator.oom_fallbacks
    size_limit_fallback_start = estimator.size_limit_fallbacks
    initial_decoded_frames = 0
    decoded_frames = 0
    retry_recovered_frames = 0
    tail_retry_attempted = False

    def flush() -> None:
        nonlocal inference_seconds
        if not pending:
            return
        inference_start = time.perf_counter()
        disparities = estimator.infer(
            [frame.left_rgb for frame in pending],
            [frame.right_rgb for frame in pending],
        )
        inference_seconds += time.perf_counter() - inference_start
        for frame, disparity in zip(pending, disparities, strict=True):
            writer.write_disparity(frame, disparity)
        pending.clear()

    try:
        initial_source = iter_stereo(svo, frame_count, gpu_id, zed_settings_dir)
        initial_frames = (
            PrefetchedStereo(initial_source, prefetch_capacity)
            if prefetch
            else closing(initial_source)
        )
        with initial_frames as frames:
            for frame in frames:
                if first is None:
                    first = frame
                pending.append(frame)
                decoded_frames += 1
                if len(pending) == batch_size:
                    flush()
        initial_decoded_frames = decoded_frames
        if decoded_frames < frame_count:
            tail_retry_attempted = True
            retry_source = iter_stereo(
                svo,
                frame_count,
                gpu_id,
                zed_settings_dir,
                frame_start=decoded_frames,
            )
            retry_frames = (
                PrefetchedStereo(retry_source, prefetch_capacity)
                if prefetch
                else closing(retry_source)
            )
            with retry_frames as frames:
                for frame in frames:
                    pending.append(frame)
                    decoded_frames += 1
                    retry_recovered_frames += 1
                    if len(pending) == batch_size:
                        flush()
        flush()
        tail_missing_count = frame_count - decoded_frames
        if decoded_frames == 0 or not 0 <= tail_missing_count <= MAX_TAIL_SHORTFALL:
            raise ConversionError(
                f"Decoded {decoded_frames}/{frame_count} frames from {svo}; "
                f"tail shortfall exceeds {MAX_TAIL_SHORTFALL}"
            )
        if tail_missing_count:
            zero_depth = np.zeros(EXPECTED_SHAPE, dtype=np.uint16)
            for _ in range(tail_missing_count):
                writer.write_depth(zero_depth, None)
        writer.close(frame_count)
    except BaseException:
        writer.cleanup()
        raise

    if (
        first is None
        or len(writer.timestamps) != frame_count
        or not depth_is_valid(destination, frame_count)
    ):
        raise ConversionError(f"Output validation failed: {destination}")
    missing_frame_indices = list(range(decoded_frames, frame_count))
    write_json_atomic(
        sidecar,
        {
            "schema_version": "1.1",
            "source": {
                "episode_index": int(row["episode_index"]),
                "episode_chunk": int(row["episode_chunk"]),
                "source_episode_id": row.get("source_episode_id"),
                "camera_role": role,
                "camera_serial": str(camera["serial"]),
                "svo_path": str(camera["svo_path"]),
                "svo_bytes": int(camera["bytes"]),
                "svo_frame_count": first.svo_frame_count,
                "frame_count": frame_count,
                "decoded_frame_count": decoded_frames,
                "initial_decoded_frame_count": initial_decoded_frames,
                "tail_retry_attempted": tail_retry_attempted,
                "tail_retry_recovered_frames": retry_recovered_frames,
                "tail_missing_count": tail_missing_count,
                "missing_frame_indices": missing_frame_indices,
                "source_decode_complete": tail_missing_count == 0,
                "timestamps_ms": writer.timestamps,
            },
            "calibration": {
                "intrinsic": first.intrinsic.tolist(),
                "baseline_m": first.baseline_m,
                "source": "native rectified SVO calibration_parameters",
                "factory_file": calibration_file.name,
                "factory_sha256": calibration_sha256,
            },
            "inference": {
                "method": "FoundationStereo hierarchical",
                "input_resolution": [1280, 720],
                "output_resolution": [1280, 720],
                "resized": False,
                "iters": estimator.iters,
                "small_ratio": 0.5,
                "batch_size_requested": batch_size,
                "batch_size_effective_max": min(
                    batch_size,
                    estimator.max_successful_batch_size,
                ),
                "batch_size_effective_limit": estimator.max_inference_batch_size,
                "prefetch_enabled": prefetch,
                "prefetch_capacity": prefetch_capacity,
                "writer_capacity": writer_capacity,
                "batch_oom_fallbacks": estimator.oom_fallbacks - oom_fallback_start,
                "batch_size_limit_fallbacks": (
                    estimator.size_limit_fallbacks - size_limit_fallback_start
                ),
                "checkpoint_sha256": checkpoint_sha256,
                "gpu_id": gpu_id,
                "inference_seconds": inference_seconds,
                "wall_seconds": time.perf_counter() - started,
            },
            "depth": {
                "storage": "uint16 millimeter PNG",
                "invalid_value": 0,
                "valid_range_m": [0.02, 10.0],
                "valid_fraction": writer.valid_pixels / writer.total_pixels,
                "zero_filled_source_frames": tail_missing_count,
                "path": str(destination.relative_to(output_dir)),
            },
        },
    )
    return ConversionResult(
        "written_tail_padded" if tail_missing_count else "written",
        tail_missing_count=tail_missing_count,
        tail_retry_recovered_frames=retry_recovered_frames,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--zed-settings-dir", type=Path, required=True)
    parser.add_argument("--fs-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--gpu-id", type=int, required=True)
    parser.add_argument("--worker-index", type=int, required=True)
    parser.add_argument("--num-workers", type=int, required=True)
    parser.add_argument("--episodes", type=int, nargs="*")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--iters", type=int, default=32)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--no-prefetch",
        action="store_true",
        help="Decode SVO frames synchronously for recovery after a worker failure.",
    )
    args = parser.parse_args()

    chunks = parse_chunks(args.chunks)
    rows = read_rows(args.input_dir, chunks)
    if args.episodes:
        selected = set(args.episodes)
        rows = [row for row in rows if int(row["episode_index"]) in selected]
        found = {int(row["episode_index"]) for row in rows}
        if found != selected:
            raise ValueError(f"Selected episodes are not in the chosen chunks: {sorted(selected - found)}")
    jobs = [(row, role) for row in rows for role in ROLES]
    all_job_count = len(jobs)
    if not args.overwrite:
        # Rebuild the shared queue from incomplete outputs before sharding it.
        # Sharding first makes a resumed run preserve the old imbalance: GPUs
        # whose original jobs are complete exit while one GPU keeps all of its
        # unfinished jobs. Every worker computes this same deterministic list,
        # then takes a disjoint slice below.
        jobs = [
            (row, role)
            for row, role in jobs
            if not (
                depth_is_valid(depth_dir(args.output_dir, row, role), int(row["length"]))
                and metadata_is_valid(
                    metadata_path(args.output_dir, row, role),
                    row,
                    role,
                    args.checkpoint_sha256,
                    args.iters,
                )
            )
        ]
    pending_job_count = len(jobs)
    jobs = jobs[args.worker_index :: args.num_workers]
    print(
        f"[GPU {args.gpu_id}] pending before sharding: "
        f"{pending_job_count}/{all_job_count}; assigned: {len(jobs)}",
        flush=True,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / "logs" / f"gpu-{args.gpu_id}.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if not jobs:
        return 0
    estimator = Estimator(args.fs_root, args.checkpoint, args.config, args.gpu_id, args.iters)
    failures = 0
    with log_path.open("a", encoding="utf-8", buffering=1) as log:
        for position, (row, role) in enumerate(jobs, start=1):
            event = {
                "episode_index": int(row["episode_index"]),
                "camera_role": role,
                "gpu_id": args.gpu_id,
                "time": time.time(),
            }
            try:
                result = convert_one(
                    estimator,
                    args.input_dir,
                    args.output_dir,
                    args.zed_settings_dir,
                    row,
                    role,
                    args.gpu_id,
                    args.batch_size,
                    args.checkpoint_sha256,
                    args.overwrite,
                    not args.no_prefetch,
                )
            # Do not swallow KeyboardInterrupt/SystemExit: Ctrl-C must stop the
            # worker instead of recording one failed episode and continuing.
            except Exception as error:
                failures += 1
                fatal_cuda = is_fatal_cuda_error(error)
                event.update(
                    status="failed",
                    error_type=type(error).__name__,
                    message=str(error),
                    traceback=traceback.format_exc(),
                    fatal_cuda=fatal_cuda,
                )
            else:
                event.update(
                    status=result.status,
                    tail_missing_count=result.tail_missing_count,
                    tail_retry_recovered_frames=result.tail_retry_recovered_frames,
                )
            log.write(json.dumps(event, ensure_ascii=False) + "\n")
            print(
                f"[GPU {args.gpu_id}] {position}/{len(jobs)} "
                f"episode={row['episode_index']} role={role} status={event['status']}",
                flush=True,
            )
            if event.get("fatal_cuda"):
                print(
                    f"[GPU {args.gpu_id}] CUDA context is corrupted; "
                    "exiting this worker for a clean process restart.",
                    flush=True,
                )
                return 86
    return int(failures > 0)


if __name__ == "__main__":
    raise SystemExit(main())
