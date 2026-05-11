"""Import negative samples into the YOLO dataset without touching training code.

This script:
1. Validates the dataset split/images/labels structure.
2. Scans a negative sample directory for supported images.
3. Detects unreadable/corrupted files and duplicate images by content hash.
4. Splits valid negative samples into train/valid/test.
5. Copies images into split image directories without writing detection boxes.
6. Generates a markdown summary and a CSV mapping/report for traceability.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import random
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

from dataset_config import find_default_dataset_dir

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
EXPECTED_SPLITS = ("train", "valid", "test")
SUMMARY_FILENAME = "negative_import_summary.md"
MAPPING_FILENAME = "negative_import_mapping.csv"

try:
    from PIL import Image
    from PIL import UnidentifiedImageError
except ModuleNotFoundError:  # pragma: no cover - runtime fallback
    Image = None
    UnidentifiedImageError = Exception


@dataclass(frozen=True)
class SourceImage:
    """A validated source image ready for import."""

    path: Path
    sha256: str
    width: int
    height: int
    file_size: int


@dataclass(frozen=True)
class ExistingImage:
    """A pre-existing dataset image used for duplicate checks."""

    path: Path
    sha256: str


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    script_dir = Path(__file__).resolve().parent
    workspace_dir = script_dir.parent
    default_dataset_dir = find_default_dataset_dir(workspace_dir)

    parser = argparse.ArgumentParser(
        description="Import negative images into a YOLO dataset with reports."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=default_dataset_dir,
        help="YOLO dataset root directory. Default: %(default)s",
    )
    parser.add_argument(
        "--negative-dir",
        type=Path,
        default=workspace_dir / "negative",
        help="Source negative image directory. Default: %(default)s",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic split assignment.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.70,
        help="Train split ratio. Default: %(default)s",
    )
    parser.add_argument(
        "--valid-ratio",
        type=float,
        default=0.15,
        help="Valid split ratio. Default: %(default)s",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="Test split ratio. Default: %(default)s",
    )
    parser.add_argument(
        "--create-empty-labels",
        action="store_true",
        help="Create empty label files for imported negatives. Default: disabled.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan and report only without copying files.",
    )
    return parser.parse_args()


def print_status(level: str, message: str) -> None:
    """Print a compact status line."""
    print(f"[{level}] {message}")


def validate_ratios(train_ratio: float, valid_ratio: float, test_ratio: float) -> None:
    """Ensure the split ratios are usable."""
    ratios = (train_ratio, valid_ratio, test_ratio)
    if any(ratio < 0 for ratio in ratios):
        raise ValueError("Split ratios must be non-negative.")
    if not math.isclose(sum(ratios), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("Split ratios must sum to 1.0.")


def validate_dataset_structure(dataset_dir: Path) -> None:
    """Ensure the dataset matches train/valid/test + images/labels."""
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_dir}")

    missing_paths: list[Path] = []
    for split_name in EXPECTED_SPLITS:
        for child_name in ("images", "labels"):
            candidate = dataset_dir / split_name / child_name
            if not candidate.is_dir():
                missing_paths.append(candidate)

    if missing_paths:
        joined = "\n".join(str(path) for path in missing_paths)
        raise FileNotFoundError(
            "Dataset structure is incomplete. Missing:\n"
            f"{joined}"
        )


def iter_image_files(directory: Path) -> list[Path]:
    """Return supported image files from a directory."""
    if not directory.exists():
        raise FileNotFoundError(f"Negative image directory does not exist: {directory}")
    if not directory.is_dir():
        raise NotADirectoryError(f"Negative image path is not a directory: {directory}")

    return sorted(
        (
            path
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ),
        key=lambda path: path.name.lower(),
    )


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Hash a file with SHA-256."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def probe_image(path: Path) -> tuple[int, int]:
    """Open and verify an image, returning width and height."""
    if Image is None:
        raise RuntimeError(
            "Pillow is required for image validation but is not installed in this environment."
        )

    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            width, height = image.size
    except (OSError, UnidentifiedImageError) as exc:
        raise ValueError(str(exc)) from exc

    if width <= 0 or height <= 0:
        raise ValueError("Image dimensions must be positive.")

    return width, height


def collect_dataset_images(dataset_dir: Path) -> list[ExistingImage]:
    """Collect existing dataset images for duplicate checks."""
    existing: list[ExistingImage] = []
    for split_name in EXPECTED_SPLITS:
        image_dir = dataset_dir / split_name / "images"
        for path in sorted(image_dir.iterdir(), key=lambda item: item.name.lower()):
            if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            existing.append(ExistingImage(path=path, sha256=sha256_file(path)))
    return existing


def allocate_split_counts(total: int, train_ratio: float, valid_ratio: float) -> dict[str, int]:
    """Convert ratios into deterministic integer counts."""
    train_count = round(total * train_ratio)
    valid_count = round(total * valid_ratio)

    if train_count + valid_count > total:
        overflow = train_count + valid_count - total
        valid_reduce = min(valid_count, overflow)
        valid_count -= valid_reduce
        overflow -= valid_reduce
        train_count -= overflow

    test_count = total - train_count - valid_count
    return {"train": train_count, "valid": valid_count, "test": test_count}


def assign_splits(
    candidates: list[SourceImage],
    seed: int,
    train_ratio: float,
    valid_ratio: float,
) -> dict[str, list[SourceImage]]:
    """Shuffle candidates deterministically and partition them into splits."""
    shuffled = list(candidates)
    random.Random(seed).shuffle(shuffled)
    counts = allocate_split_counts(len(shuffled), train_ratio, valid_ratio)

    train_end = counts["train"]
    valid_end = train_end + counts["valid"]
    return {
        "train": shuffled[:train_end],
        "valid": shuffled[train_end:valid_end],
        "test": shuffled[valid_end:],
    }


def ensure_unique_destination_name(
    source_name: str,
    split_name: str,
    used_names: set[str],
    existing_names: set[str],
) -> tuple[str, bool]:
    """Preserve the original filename when possible, otherwise append a suffix."""
    if source_name not in used_names and source_name not in existing_names:
        return source_name, False

    stem = Path(source_name).stem
    suffix = Path(source_name).suffix.lower()
    counter = 1
    while True:
        candidate = f"{stem}__neg_{split_name}_{counter:03d}{suffix}"
        if candidate not in used_names and candidate not in existing_names:
            return candidate, True
        counter += 1


def create_empty_label(label_path: Path) -> None:
    """Create an empty YOLO label file."""
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text("", encoding="utf-8")


def format_examples(items: Iterable[str], limit: int = 10) -> str:
    """Format example items for markdown."""
    examples = list(items)[:limit]
    if not examples:
        return "- None"
    return "\n".join(f"- `{item}`" for item in examples)


def write_mapping_csv(rows: list[dict[str, str]], output_path: Path) -> None:
    """Write a flat import/skip report."""
    fieldnames = [
        "status",
        "source_name",
        "source_path",
        "destination_split",
        "destination_name",
        "destination_path",
        "sha256",
        "width",
        "height",
        "file_size",
        "renamed",
        "note",
    ]

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(
    summary_path: Path,
    dataset_dir: Path,
    negative_dir: Path,
    total_supported: int,
    imported_counts: dict[str, int],
    skipped_invalid: list[dict[str, str]],
    skipped_duplicate_source: list[dict[str, str]],
    skipped_duplicate_dataset: list[dict[str, str]],
    renamed_rows: list[dict[str, str]],
    dry_run: bool,
    create_empty_labels: bool,
) -> None:
    """Write the markdown summary report."""
    imported_total = sum(imported_counts.values())
    lines = [
        "# Negative Import Summary",
        "",
        f"- Generated at: `{datetime.now().isoformat(timespec='seconds')}`",
        f"- Dataset root: `{dataset_dir}`",
        f"- Negative source: `{negative_dir}`",
        f"- Mode: `{'dry-run' if dry_run else 'import'}`",
        f"- Empty label files created: `{'yes' if create_empty_labels else 'no'}`",
        "",
        "## Overall",
        "",
        f"- Supported source images scanned: `{total_supported}`",
        f"- Imported negative samples: `{imported_total}`",
        f"- Skipped unreadable/corrupted images: `{len(skipped_invalid)}`",
        f"- Skipped duplicate images inside source negatives: `{len(skipped_duplicate_source)}`",
        f"- Skipped images already duplicated in dataset by content hash: `{len(skipped_duplicate_dataset)}`",
        f"- Renamed on import due to filename conflict: `{len(renamed_rows)}`",
        "",
        "## Split Counts",
        "",
        "| Split | Imported |",
        "| --- | ---: |",
        f"| train | {imported_counts['train']} |",
        f"| valid | {imported_counts['valid']} |",
        f"| test | {imported_counts['test']} |",
        "",
        "## Notes",
        "",
        "- Imported negatives were copied into `images/` only and do not contain detection boxes.",
        "- Empty label files are disabled by default; Ultralytics can treat missing label files as background images.",
        "- Full per-file traceability is recorded in `negative_import_mapping.csv`.",
        "",
        "## Example Unreadable/Corrupted Files",
        "",
        format_examples((row["source_name"] for row in skipped_invalid)),
        "",
        "## Example Source Duplicates",
        "",
        format_examples((row["source_name"] for row in skipped_duplicate_source)),
        "",
        "## Example Dataset Duplicates",
        "",
        format_examples((row["source_name"] for row in skipped_duplicate_dataset)),
        "",
        "## Example Renamed Files",
        "",
        format_examples(
            (
                f"{row['source_name']} -> {row['destination_name']}"
                for row in renamed_rows
            )
        ),
        "",
    ]
    summary_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    """Run the negative import workflow."""
    args = parse_args()

    dataset_dir = args.dataset_dir.resolve()
    negative_dir = args.negative_dir.resolve()
    summary_path = dataset_dir / SUMMARY_FILENAME
    mapping_path = dataset_dir / MAPPING_FILENAME

    validate_ratios(args.train_ratio, args.valid_ratio, args.test_ratio)
    validate_dataset_structure(dataset_dir)

    source_files = iter_image_files(negative_dir)
    print_status("INFO", f"Found {len(source_files)} supported image files in {negative_dir}")

    dataset_existing = collect_dataset_images(dataset_dir)
    existing_hash_to_path = {image.sha256: image.path for image in dataset_existing}
    existing_name_sets = {
        split_name: {
            path.name
            for path in (dataset_dir / split_name / "images").iterdir()
            if path.is_file()
        }
        for split_name in EXPECTED_SPLITS
    }

    mapping_rows: list[dict[str, str]] = []
    valid_candidates: list[SourceImage] = []
    source_hash_to_path: dict[str, Path] = {}
    skipped_invalid: list[dict[str, str]] = []
    skipped_duplicate_source: list[dict[str, str]] = []
    skipped_duplicate_dataset: list[dict[str, str]] = []

    for path in source_files:
        try:
            width, height = probe_image(path)
            digest = sha256_file(path)
        except (OSError, RuntimeError, ValueError) as exc:
            row = {
                "status": "skipped_invalid",
                "source_name": path.name,
                "source_path": str(path),
                "destination_split": "",
                "destination_name": "",
                "destination_path": "",
                "sha256": "",
                "width": "",
                "height": "",
                "file_size": str(path.stat().st_size if path.exists() else 0),
                "renamed": "no",
                "note": str(exc),
            }
            mapping_rows.append(row)
            skipped_invalid.append(row)
            continue

        if digest in source_hash_to_path:
            original = source_hash_to_path[digest]
            row = {
                "status": "skipped_duplicate_source",
                "source_name": path.name,
                "source_path": str(path),
                "destination_split": "",
                "destination_name": "",
                "destination_path": "",
                "sha256": digest,
                "width": str(width),
                "height": str(height),
                "file_size": str(path.stat().st_size),
                "renamed": "no",
                "note": f"Duplicate of source image: {original.name}",
            }
            mapping_rows.append(row)
            skipped_duplicate_source.append(row)
            continue

        if digest in existing_hash_to_path:
            dataset_match = existing_hash_to_path[digest]
            row = {
                "status": "skipped_duplicate_dataset",
                "source_name": path.name,
                "source_path": str(path),
                "destination_split": "",
                "destination_name": "",
                "destination_path": "",
                "sha256": digest,
                "width": str(width),
                "height": str(height),
                "file_size": str(path.stat().st_size),
                "renamed": "no",
                "note": f"Duplicate of dataset image: {dataset_match.relative_to(dataset_dir)}",
            }
            mapping_rows.append(row)
            skipped_duplicate_dataset.append(row)
            continue

        source_hash_to_path[digest] = path
        valid_candidates.append(
            SourceImage(
                path=path,
                sha256=digest,
                width=width,
                height=height,
                file_size=path.stat().st_size,
            )
        )

    split_candidates = assign_splits(
        valid_candidates,
        seed=args.seed,
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
    )

    imported_counts = {"train": 0, "valid": 0, "test": 0}
    renamed_rows: list[dict[str, str]] = []

    for split_name in EXPECTED_SPLITS:
        used_names: set[str] = set()
        image_dir = dataset_dir / split_name / "images"
        label_dir = dataset_dir / split_name / "labels"
        for source in split_candidates[split_name]:
            destination_name, renamed = ensure_unique_destination_name(
                source_name=source.path.name,
                split_name=split_name,
                used_names=used_names,
                existing_names=existing_name_sets[split_name],
            )
            used_names.add(destination_name)

            destination_path = image_dir / destination_name
            if not args.dry_run:
                shutil.copy2(source.path, destination_path)
                if args.create_empty_labels:
                    create_empty_label(label_dir / f"{Path(destination_name).stem}.txt")

            row = {
                "status": "imported",
                "source_name": source.path.name,
                "source_path": str(source.path),
                "destination_split": split_name,
                "destination_name": destination_name,
                "destination_path": str(destination_path),
                "sha256": source.sha256,
                "width": str(source.width),
                "height": str(source.height),
                "file_size": str(source.file_size),
                "renamed": "yes" if renamed else "no",
                "note": "Copied without label file" if not args.create_empty_labels else "Copied with empty label file",
            }
            mapping_rows.append(row)
            imported_counts[split_name] += 1
            if renamed:
                renamed_rows.append(row)

    write_mapping_csv(mapping_rows, mapping_path)
    write_summary(
        summary_path=summary_path,
        dataset_dir=dataset_dir,
        negative_dir=negative_dir,
        total_supported=len(source_files),
        imported_counts=imported_counts,
        skipped_invalid=skipped_invalid,
        skipped_duplicate_source=skipped_duplicate_source,
        skipped_duplicate_dataset=skipped_duplicate_dataset,
        renamed_rows=renamed_rows,
        dry_run=args.dry_run,
        create_empty_labels=args.create_empty_labels,
    )

    print_status("OK", f"Imported negatives: {sum(imported_counts.values())}")
    for split_name in EXPECTED_SPLITS:
        print_status("OK", f"{split_name}: {imported_counts[split_name]}")
    print_status("OK", f"Summary written to {summary_path}")
    print_status("OK", f"Mapping written to {mapping_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # pragma: no cover - command line reporting
        print_status("ERROR", str(exc))
        raise
