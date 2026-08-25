#!/usr/bin/env python3
"""Create an RGB | metric-depth MP4 for one converted DROID episode."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Iterator

import imageio.v3 as iio
import numpy as np

from convert_droid_depth import DEPTH_KEYS, EXPECTED_SHAPE, ROLES, iter_stereo, source_svo


FPS = 15
NEAR_M = 0.2
FAR_M = 4.0
TURBO_COEFFICIENTS = np.array(
    [
        [
            0.13572138,
            4.61539260,
            -42.66032258,
            132.13108234,
            -152.94239396,
            59.28637943,
        ],
        [
            0.09140261,
            2.19418839,
            4.84296658,
            -14.18503333,
            4.27729857,
            2.82956604,
        ],
        [
            0.10667330,
            12.64194608,
            -60.58204836,
            110.36276771,
            -89.90310912,
            27.34824973,
        ],
    ],
    dtype=np.float32,
)


def load_episode(input_dir: Path, episode_index: int) -> dict[str, Any]:
    chunk = episode_index // 1000
    manifest = input_dir / "manifests/chunks" / f"chunk-{chunk:03d}.jsonl"
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    with manifest.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if int(row["episode_index"]) == episode_index:
                return row
    raise ValueError(f"Episode {episode_index} is not present in {manifest}")


def camera_roles(value: str) -> tuple[str, ...]:
    aliases = {
        "both": ROLES,
        "1": ("external_1",),
        "2": ("external_2",),
        "external_1": ("external_1",),
        "external_2": ("external_2",),
    }
    try:
        return tuple(aliases[value])
    except KeyError as error:
        raise ValueError("camera must be both, 1, 2, external_1 or external_2") from error


def depth_frames(output_dir: Path, row: dict[str, Any], role: str) -> list[Path]:
    episode_index = int(row["episode_index"])
    chunk = int(row["episode_chunk"])
    directory = (
        output_dir
        / "images"
        / f"chunk-{chunk:03d}"
        / DEPTH_KEYS[role]
        / f"episode_{episode_index:06d}"
    )
    frames = sorted(directory.glob("frame_*.png")) if directory.is_dir() else []
    expected = int(row["length"])
    if len(frames) != expected:
        raise ValueError(f"Expected {expected} depth PNGs in {directory}, found {len(frames)}")
    for index, frame in enumerate(frames):
        if frame.name != f"frame_{index:06d}.png":
            raise ValueError(f"Unexpected depth filename: {frame}")
    return frames


def visualization_path(output_dir: Path, row: dict[str, Any], role: str) -> Path:
    return (
        output_dir
        / "visualizations/foundation_stereo_depth"
        / f"chunk-{int(row['episode_chunk']):03d}"
        / DEPTH_KEYS[role]
        / f"episode_{int(row['episode_index']):06d}.mp4"
    )


def colorize_depth(depth_mm: np.ndarray) -> np.ndarray:
    if depth_mm.shape != EXPECTED_SHAPE or depth_mm.dtype != np.uint16:
        raise ValueError(f"Expected uint16 depth {EXPECTED_SHAPE}, got {depth_mm.shape}/{depth_mm.dtype}")
    valid = depth_mm > 0
    depth_m = depth_mm.astype(np.float32) / 1000.0
    normalized = (np.clip(depth_m, NEAR_M, FAR_M) - NEAR_M) / (FAR_M - NEAR_M)
    values = np.clip(1.0 - normalized, 0.0, 1.0)
    powers = np.stack([values**power for power in range(6)], axis=-1)
    rgb = np.einsum("...p,cp->...c", powers, TURBO_COEFFICIENTS)
    color = np.rint(np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
    color[~valid] = 0
    return color


class VideoWriter:
    def __init__(self, path: Path, frame_count: int, overwrite: bool) -> None:
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError("ffmpeg is not available")
        if path.exists() and not overwrite:
            raise FileExistsError(f"Visualization already exists: {path}; pass --overwrite to replace it")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.temporary = path.with_name(f".{path.stem}.tmp.{os.getpid()}.mp4")
        self.temporary.unlink(missing_ok=True)
        self.expected_frames = frame_count
        self.written_frames = 0
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
                "rgb24",
                "-s",
                "2560x720",
                "-r",
                str(FPS),
                "-i",
                "-",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "17",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(self.temporary),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

    def write(self, frame: np.ndarray) -> None:
        if frame.shape != (720, 2560, 3) or frame.dtype != np.uint8:
            raise ValueError(f"Unexpected visualization frame: {frame.shape}/{frame.dtype}")
        if self.process.stdin is None:
            raise RuntimeError("ffmpeg stdin is unavailable")
        self.process.stdin.write(np.ascontiguousarray(frame).tobytes())
        self.written_frames += 1

    def close(self) -> None:
        if self.process.stdin is not None:
            self.process.stdin.close()
        stderr = self.process.stderr.read().decode(errors="replace") if self.process.stderr else ""
        return_code = self.process.wait()
        if return_code or self.written_frames != self.expected_frames:
            self.cleanup()
            raise RuntimeError(
                stderr.strip()
                or f"Wrote {self.written_frames}/{self.expected_frames} visualization frames"
            )
        os.replace(self.temporary, self.path)

    def cleanup(self) -> None:
        if self.process.poll() is None:
            self.process.kill()
            self.process.wait()
        self.temporary.unlink(missing_ok=True)


def decoded_rgb_frames(
    input_dir: Path,
    row: dict[str, Any],
    role: str,
    gpu_id: int,
) -> Iterator[np.ndarray]:
    svo = source_svo(input_dir, row, role)
    settings = input_dir / "calibrations"
    frame_count = int(row["length"])
    decoded = 0
    for stereo in iter_stereo(svo, frame_count, gpu_id, settings):
        yield stereo.left_rgb
        decoded += 1
    if decoded < frame_count:
        for stereo in iter_stereo(
            svo,
            frame_count,
            gpu_id,
            settings,
            frame_start=decoded,
        ):
            yield stereo.left_rgb
            decoded += 1


def visualize_camera(
    input_dir: Path,
    output_dir: Path,
    row: dict[str, Any],
    role: str,
    gpu_id: int,
    overwrite: bool,
) -> Path:
    frames = depth_frames(output_dir, row, role)
    destination = visualization_path(output_dir, row, role)
    writer = VideoWriter(destination, len(frames), overwrite)
    rgb_iterator = iter(decoded_rgb_frames(input_dir, row, role, gpu_id))
    decoded = 0
    try:
        for index, depth_path in enumerate(frames):
            try:
                rgb = next(rgb_iterator)
                decoded += 1
            except StopIteration:
                rgb = np.zeros((*EXPECTED_SHAPE, 3), dtype=np.uint8)
            depth = iio.imread(depth_path)
            writer.write(np.concatenate((rgb, colorize_depth(depth)), axis=1))
            if (index + 1) % 100 == 0 or index + 1 == len(frames):
                print(f"  Frames: {index + 1}/{len(frames)}", flush=True)
        writer.close()
    except BaseException:
        writer.cleanup()
        raise
    finally:
        rgb_iterator.close()
    print(f"  RGB decoded: {decoded}/{len(frames)}", flush=True)
    print(f"  MP4: {destination}", flush=True)
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--camera", default="both")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.episode < 0 or args.gpu_id < 0:
        parser.error("episode and gpu-id must be non-negative")
    try:
        roles = camera_roles(args.camera)
    except ValueError as error:
        parser.error(str(error))
    row = load_episode(args.input_dir, args.episode)

    print("DROID depth visualization", flush=True)
    print(f"  Episode: {args.episode}", flush=True)
    print(f"  Cameras: {', '.join(roles)}", flush=True)
    print(f"  Layout: RGB | metric depth ({NEAR_M:.1f}-{FAR_M:.1f} m)", flush=True)
    for role in roles:
        print(f"Camera: {role}", flush=True)
        visualize_camera(
            args.input_dir,
            args.output_dir,
            row,
            role,
            args.gpu_id,
            args.overwrite,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
