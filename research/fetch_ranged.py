"""Resumable parallel HTTP downloader for large public dataset files.

S3 (and most dataset hosts) support HTTP byte ranges. A single saturated
connection can be very slow on long-haul links, so this utility splits the
*remaining* bytes of a file into ordered ranges, downloads them concurrently
with ``curl`` into a temporary parts directory, appends them to the target
file and verifies the final size against ``Content-Length``.

Curl is used instead of Python's urllib because the latter stalled on the MIT
S3 endpoint from this network while curl sustained transfers.

Raw dataset bytes are never transformed. Existing partial files are resumed.
Parts are kept on failure so a re-run only re-fetches incomplete ranges.

Usage:
    python research/fetch_ranged.py URL OUTPUT [--connections 6]

Exit codes: 0 success, 1 failure.
"""

from __future__ import annotations

import argparse
import math
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

COPY_CHUNK = 8 * 1024 * 1024
CURL = shutil.which("curl") or "curl"


def content_length(url: str, timeout: int = 60) -> int:
    result = subprocess.run(
        [CURL, "-sSI", "-L", url],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    lengths = [
        int(line.split(":", 1)[1].strip())
        for line in result.stdout.splitlines()
        if line.lower().startswith("content-length:")
    ]
    if not lengths:
        raise RuntimeError(f"No Content-Length from {url}: {result.stderr.strip()}")
    return lengths[-1]


def download_range(
    url: str,
    start: int,
    end: int,
    path: Path,
    retries: int = 4,
    timeout: int = 900,
) -> Path:
    expected = end - start + 1
    if path.exists() and path.stat().st_size == expected:
        print(f"  reuse {path.name} ({expected:,} bytes)")
        return path

    command = [
        CURL,
        "-sS",
        "-L",
        "--fail",
        "--retry",
        "3",
        "--retry-all-errors",
        "--retry-delay",
        "2",
        "--connect-timeout",
        "30",
        "--max-time",
        str(timeout),
        "-r",
        f"{start}-{end}",
        "-o",
        str(path),
        url,
    ]
    for attempt in range(1, retries + 1):
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        actual = path.stat().st_size if path.exists() else 0
        if result.returncode == 0 and actual == expected:
            print(f"  done  {path.name} ({expected:,} bytes)")
            return path
        message = (result.stderr or result.stdout or "").strip().splitlines()
        detail = message[-1] if message else f"exit {result.returncode}"
        wait = min(2 * attempt, 20)
        print(
            f"  retry {attempt}/{retries} {path.name}: {detail} "
            f"({actual:,}/{expected:,} bytes, sleep {wait}s)",
            file=sys.stderr,
        )
        if attempt == retries:
            raise RuntimeError(f"range {start}-{end} failed: {detail}")
        time.sleep(wait)
    raise RuntimeError("unreachable")


def fetch(url: str, output: Path, connections: int) -> None:
    total = content_length(url)
    output.parent.mkdir(parents=True, exist_ok=True)
    current = output.stat().st_size if output.exists() else 0
    if current > total:
        raise RuntimeError(
            f"Existing file is larger than remote ({current:,} > {total:,})"
        )
    if current == total:
        print(f"already complete: {output} ({total:,} bytes)")
        return

    remaining = total - current
    parts_dir = output.parent / f".{output.name}.parts"
    parts_dir.mkdir(parents=True, exist_ok=True)

    minimum_part = 8 * 1024 * 1024
    n_parts = max(1, min(connections, math.ceil(remaining / minimum_part)))
    span = math.ceil(remaining / n_parts)
    ranges = []
    for index in range(n_parts):
        start = current + index * span
        end = min(start + span - 1, total - 1)
        if start <= end:
            ranges.append((index, start, end))

    print(
        f"target: {output}\n"
        f"remote: {total:,} bytes | present: {current:,} | "
        f"remaining: {remaining:,} in {len(ranges)} ranges"
    )

    started = time.time()
    with ThreadPoolExecutor(max_workers=len(ranges)) as pool:
        futures = {
            pool.submit(
                download_range, url, start, end, parts_dir / f"part{index:02d}"
            ): (index, start, end)
            for index, start, end in ranges
        }
        for future in as_completed(futures):
            future.result()  # propagate the first failure

    # Append in strict byte order; the pre-existing prefix stays untouched.
    with output.open("ab") as destination:
        for index, start, end in ranges:
            part = parts_dir / f"part{index:02d}"
            with part.open("rb") as source:
                shutil.copyfileobj(source, destination, COPY_CHUNK)
            part.unlink()

    try:
        parts_dir.rmdir()
    except OSError:
        pass

    final = output.stat().st_size
    if final != total:
        raise RuntimeError(f"size verification failed: {final:,} != {total:,}")
    elapsed = time.time() - started
    rate = remaining / elapsed / 1024 / 1024 if elapsed else 0
    print(f"complete: {output} ({final:,} bytes) in {elapsed:.1f}s ({rate:.2f} MB/s)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url")
    parser.add_argument("output", type=Path)
    parser.add_argument("--connections", type=int, default=6)
    args = parser.parse_args()
    try:
        fetch(args.url, args.output, max(1, args.connections))
    except Exception as error:  # noqa: BLE001 - CLI boundary
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
