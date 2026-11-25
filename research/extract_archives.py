"""Safely extract downloaded battery dataset archives.

Raw archive files are never modified. Extracted files are placed in
``data/raw/<source>/extracted/``. Nested ZIP archives (NASA) are extracted
into a directory named after the archive.
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW = PROJECT_ROOT / "data" / "raw"


def safe_extract(zip_path: Path, destination: Path) -> list[Path]:
    """Extract a ZIP while refusing path traversal and recording members."""
    destination.mkdir(parents=True, exist_ok=True)
    extracted: list[Path] = []
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if not str(target).startswith(str(destination.resolve())):
                raise ValueError(f"Unsafe archive member: {member.filename}")
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as src, target.open("wb") as dst:
                while chunk := src.read(1024 * 1024):
                    dst.write(chunk)
            extracted.append(target)
    return extracted


def extract_nasa() -> None:
    source = RAW / "NASA"
    outer = source / "5.+Battery+Data+Set.zip"
    if not outer.exists():
        print(f"SKIP (missing): {outer}")
        return
    stage = source / "extracted" / "nested"
    if not (stage / "5. Battery Data Set" / "1. BatteryAgingARC-FY08Q4.zip").exists():
        safe_extract(outer, stage)
    # Extract each nested archive into its own stable directory.
    for nested in sorted(stage.rglob("*.zip")):
        target = source / "extracted" / nested.stem.replace(".", "_")
        marker = target / ".extracted"
        if marker.exists():
            print(f"SKIP (already extracted): {target.name}")
            continue
        files = safe_extract(nested, target)
        marker.write_text(f"source={nested.name}\nfiles={len(files)}\n", encoding="utf-8")
        print(f"NASA: {nested.name} -> {len(files)} files")


def extract_generic(source_name: str) -> None:
    source = RAW / source_name
    for archive_path in sorted(source.glob("*.zip")):
        target = source / "extracted" / archive_path.stem
        marker = target / ".extracted"
        if marker.exists():
            print(f"SKIP (already extracted): {source_name}/{archive_path.name}")
            continue
        try:
            files = safe_extract(archive_path, target)
        except zipfile.BadZipFile:
            print(f"SKIP (incomplete or invalid ZIP): {source_name}/{archive_path.name}")
            continue
        marker.write_text(f"source={archive_path.name}\nfiles={len(files)}\n", encoding="utf-8")
        print(f"{source_name}: {archive_path.name} -> {len(files)} files")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sources",
        nargs="*",
        default=["NASA", "CALCE"],
        help="Raw archive groups to extract.",
    )
    args = parser.parse_args()
    for source in args.sources:
        if source == "NASA":
            extract_nasa()
        else:
            extract_generic(source)


if __name__ == "__main__":
    main()
