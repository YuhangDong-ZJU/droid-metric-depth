#!/usr/bin/env python3
"""Report complete and resumable DROID metric-depth chunks."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from convert_droid_depth import (
    DEPTH_KEYS,
    ROLES,
    depth_dir,
    depth_is_valid,
    metadata_is_valid,
    metadata_path,
)


DEFAULT_CHECKPOINT_SHA256 = "60e79bde9c6a00acea551625ff814fe06e5a6806e2c0c9829baee248de87c5f1"


@dataclass(frozen=True)
class Job:
    chunk: int
    row: dict[str, Any]
    role: str


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


def read_jobs(input_dir: Path, chunks: Iterable[int]) -> list[Job]:
    jobs: list[Job] = []
    for chunk in chunks:
        manifest = input_dir / "manifests/chunks" / f"chunk-{chunk:03d}.jsonl"
        if not manifest.is_file():
            raise FileNotFoundError(f"Missing manifest: {manifest}")
        with manifest.open(encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        for row in rows:
            if int(row["episode_chunk"]) != chunk:
                raise RuntimeError(f"Chunk mismatch in {manifest}")
            if set(row["cameras"]) != set(ROLES):
                raise RuntimeError(
                    f"Unexpected camera set in episode {row['episode_index']}"
                )
            jobs.extend(Job(chunk, row, role) for role in ROLES)
    return jobs


def validate_job(
    job: Job,
    output_dir: Path,
    checkpoint_sha256: str,
    iters: int,
) -> tuple[int, bool, str]:
    frame_count = int(job.row["length"])
    destination = depth_dir(output_dir, job.row, job.role)
    if not destination.is_dir():
        return job.chunk, False, "missing_depth_dir"

    try:
        names = sorted(
            entry.name
            for entry in os.scandir(destination)
            if entry.is_file() and entry.name.startswith("frame_") and entry.name.endswith(".png")
        )
    except OSError:
        return job.chunk, False, "unreadable_depth_dir"
    if len(names) != frame_count:
        return job.chunk, False, "wrong_png_count"
    if any(name != f"frame_{index:06d}.png" for index, name in enumerate(names)):
        return job.chunk, False, "noncontiguous_png_names"

    # Reuse the converter's own validation so status and automatic resume agree.
    if not depth_is_valid(destination, frame_count):
        return job.chunk, False, "invalid_png"

    sidecar = metadata_path(output_dir, job.row, job.role)
    if not sidecar.is_file():
        return job.chunk, False, "missing_metadata"
    if not metadata_is_valid(
        sidecar,
        job.row,
        job.role,
        checkpoint_sha256,
        iters,
    ):
        return job.chunk, False, "incompatible_metadata"
    return job.chunk, True, "complete"


def format_chunks(chunks: Iterable[int]) -> str:
    values = sorted(set(chunks))
    if not values:
        return "none"
    ranges: list[str] = []
    start = previous = values[0]
    for value in values[1:]:
        if value == previous + 1:
            previous = value
            continue
        ranges.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = value
    ranges.append(str(start) if start == previous else f"{start}-{previous}")
    return ",".join(ranges)


def count_packed_tars(output_dir: Path, chunks: Iterable[int]) -> int:
    count = 0
    for chunk in chunks:
        for depth_key in DEPTH_KEYS.values():
            directory = output_dir / "images" / f"chunk-{chunk:03d}" / depth_key
            count += sum(1 for _ in directory.glob("episodes-*.tar"))
    return count


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check which DROID metric-depth chunks are complete or need resume."
    )
    parser.add_argument("--chunks", required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--checkpoint-sha256",
        default=DEFAULT_CHECKPOINT_SHA256,
        help="Must match the checkpoint used for conversion.",
    )
    parser.add_argument("--iters", type=int, default=32)
    parser.add_argument(
        "--workers",
        type=int,
        default=min(8, os.cpu_count() or 1),
        help="Parallel filesystem validation workers.",
    )
    parser.add_argument(
        "--fail-if-pending",
        action="store_true",
        help="Return exit code 1 when any task needs resume.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    chunks = parse_chunks(args.chunks)
    if args.workers < 1:
        raise ValueError("--workers must be positive")

    jobs = read_jobs(args.input_dir.resolve(), chunks)
    output_dir = args.output_dir.resolve()
    totals = Counter(job.chunk for job in jobs)
    complete: Counter[int] = Counter()
    reasons: dict[int, Counter[str]] = defaultdict(Counter)

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        results = executor.map(
            lambda job: validate_job(
                job,
                output_dir,
                args.checkpoint_sha256,
                args.iters,
            ),
            jobs,
        )
        for chunk, valid, reason in results:
            if valid:
                complete[chunk] += 1
            else:
                reasons[chunk][reason] += 1

    print("DROID metric-depth progress")
    print(f"  Input:          {args.input_dir.resolve()}")
    print(f"  Output:         {output_dir}")
    print(f"  Chunks:         {format_chunks(chunks)}")
    print(f"  Validation:     converter-equivalent PNG + metadata checks")
    print()

    complete_chunks: list[int] = []
    resume_chunks: list[int] = []
    for chunk in chunks:
        total = totals[chunk]
        done = complete[chunk]
        pending = total - done
        if pending == 0:
            complete_chunks.append(chunk)
            print(
                f"chunk-{chunk:03d} COMPLETE  "
                f"complete={done}/{total} pending=0"
            )
        else:
            resume_chunks.append(chunk)
            summary = ", ".join(
                f"{reason}={count}"
                for reason, count in reasons[chunk].most_common()
            )
            print(
                f"chunk-{chunk:03d} RESUME    "
                f"complete={done}/{total} pending={pending} | {summary}"
            )

    pending_tasks = sum(totals.values()) - sum(complete.values())
    print()
    print(f"Complete chunks: {format_chunks(complete_chunks)}")
    print(f"Resume chunks:   {format_chunks(resume_chunks)}")
    print(f"Pending tasks:   {pending_tasks}/{sum(totals.values())}")

    tar_count = count_packed_tars(output_dir, chunks)
    if tar_count:
        print()
        print(
            f"WARNING: found {tar_count} packed TAR file(s). The converter resumes "
            "from raw PNG directories and metadata, not TAR contents."
        )

    return 1 if args.fail_if_pending and pending_tasks else 0


if __name__ == "__main__":
    raise SystemExit(main())
