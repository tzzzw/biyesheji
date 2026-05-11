"""Read-only dataset checker for the smoking detection project.

This script validates:
1. YOLO directory structure under the dataset root.
2. Image/label pairing for train/valid/test splits.
3. Basic YOLO label format and class id range.
4. data.yaml class settings and resolved train/val/test paths.

It does not modify the original dataset.
"""

from __future__ import annotations

import argparse
import ast
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from dataset_config import EXPECTED_CLASS_NAME
from dataset_config import are_class_names_compatible
from dataset_config import find_default_dataset_dir
from dataset_config import normalize_class_name

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
EXPECTED_SPLITS = ("train", "valid", "test")
MAX_ERROR_EXAMPLES = 5
MAX_WARNING_EXAMPLES = 10


def print_status(level: str, message: str) -> None:
    """Print a compact status line."""
    print(f"[{level}] {message}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    script_dir = Path(__file__).resolve().parent
    workspace_dir = script_dir.parent
    default_dataset_dir = find_default_dataset_dir(workspace_dir)

    parser = argparse.ArgumentParser(
        description="Check YOLO dataset structure and data.yaml for the smoker project."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=default_dataset_dir,
        help="Dataset root directory. Default: %(default)s",
    )
    parser.add_argument(
        "--yaml",
        type=Path,
        default=None,
        help="Path to data.yaml. Default: <dataset-dir>/data.yaml",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return a non-zero exit code when warnings are found.",
    )
    return parser.parse_args()


def parse_scalar(value: str) -> Any:
    """Parse a small YAML scalar or inline list/dict without extra dependencies."""
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"null", "none", "~"}:
        return None

    try:
        return ast.literal_eval(value)
    except (SyntaxError, ValueError):
        pass

    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value.strip("'\"")


def load_yaml_fallback(yaml_path: Path) -> dict[str, Any]:
    """Load a simple YAML file when PyYAML is unavailable.

    This parser only supports the small subset used by this project:
    flat key/value pairs, inline lists, and nested dictionaries.
    """
    data: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, data)]

    for raw_line in yaml_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue

        indent = len(raw_line) - len(raw_line.lstrip(" "))
        stripped = line.strip()
        if ":" not in stripped:
            continue

        key, value = stripped.split(":", 1)
        key = key.strip()
        value = value.strip()

        while len(stack) > 1 and indent <= stack[-1][0]:
            stack.pop()

        current = stack[-1][1]
        if value == "":
            new_dict: dict[str, Any] = {}
            current[key] = new_dict
            stack.append((indent, new_dict))
        else:
            current[key] = parse_scalar(value)

    return data


def load_yaml(yaml_path: Path) -> dict[str, Any]:
    """Load YAML with PyYAML when available, otherwise use a small fallback parser."""
    try:
        import yaml  # type: ignore

        loaded = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            return loaded
        return {}
    except ModuleNotFoundError:
        return load_yaml_fallback(yaml_path)


def normalize_names_field(raw_names: Any) -> list[str]:
    """Normalize YOLO names to an ordered list."""
    if isinstance(raw_names, list):
        return [str(item) for item in raw_names]
    if isinstance(raw_names, dict):
        ordered_keys = sorted(raw_names.keys(), key=lambda key: int(key))
        return [str(raw_names[key]) for key in ordered_keys]
    return []


def resolve_yaml_root(yaml_data: dict[str, Any], yaml_path: Path) -> Path:
    """Resolve the dataset root the same way a trainer usually does."""
    yaml_dir = yaml_path.parent.resolve()
    declared_root = yaml_data.get("path")

    if not declared_root:
        return yaml_dir

    candidate = Path(str(declared_root))
    if candidate.is_absolute():
        return candidate.resolve()
    return (yaml_dir / candidate).resolve()


def resolve_split_path(
    split_value: Any,
    dataset_root_from_yaml: Path,
    dataset_dir: Path,
) -> list[Path]:
    """Resolve a train/val/test entry into one or more absolute paths."""
    if split_value is None:
        return []

    if isinstance(split_value, (list, tuple)):
        return [resolve_single_path(item, dataset_root_from_yaml, dataset_dir) for item in split_value]

    return [resolve_single_path(split_value, dataset_root_from_yaml, dataset_dir)]


def resolve_single_path(value: Any, dataset_root_from_yaml: Path, dataset_dir: Path) -> Path:
    """Resolve a single path entry."""
    candidate = Path(str(value))
    if candidate.is_absolute():
        return candidate.resolve()

    primary = (dataset_root_from_yaml / candidate).resolve()
    if primary.exists():
        return primary

    fallback_parts = [part for part in candidate.parts if part != "."]
    while fallback_parts and fallback_parts[0] == "..":
        fallback_parts.pop(0)

    if fallback_parts:
        fallback = (dataset_dir / Path(*fallback_parts)).resolve()
        if fallback.exists():
            return fallback

    return primary


def collect_stems(files: list[Path]) -> set[str]:
    """Collect file stems for pairing checks."""
    return {file_path.stem for file_path in files}


def classify_label_parts(parts: list[str]) -> str:
    """Classify one YOLO annotation line by token count.

    detect  -> class + x_center + y_center + width + height
    segment -> class + polygon points (odd token count >= 7)
    invalid -> anything else
    """
    if len(parts) == 5:
        return "detect"
    if len(parts) >= 7 and len(parts) % 2 == 1:
        return "segment"
    return "invalid"


def validate_label_file(
    label_path: Path,
    class_count: int,
    recoverable_warnings: list[str],
    format_counter: Counter[str],
) -> None:
    """Validate a single YOLO detection label file."""
    file_formats: set[str] = set()

    for line_index, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue

        parts = stripped.split()
        label_format = classify_label_parts(parts)
        if label_format == "invalid":
            recoverable_warnings.append(
                f"{label_path}: line {line_index} has unsupported token count {len(parts)}."
            )
            if len(recoverable_warnings) >= MAX_WARNING_EXAMPLES:
                return
            continue

        file_formats.add(label_format)
        format_counter[label_format] += 1

        try:
            class_id = int(parts[0])
            coords = [float(value) for value in parts[1:]]
        except ValueError:
            recoverable_warnings.append(f"{label_path}: line {line_index} contains non-numeric values.")
            if len(recoverable_warnings) >= MAX_WARNING_EXAMPLES:
                return
            continue

        if class_id < 0 or class_id >= class_count:
            recoverable_warnings.append(
                f"{label_path}: line {line_index} has class id {class_id} outside [0, {class_count - 1}]."
            )
            if len(recoverable_warnings) >= MAX_WARNING_EXAMPLES:
                return

        if label_format == "detect":
            x_center, y_center, width, height = coords
            if not (0.0 <= x_center <= 1.0 and 0.0 <= y_center <= 1.0):
                recoverable_warnings.append(f"{label_path}: line {line_index} has center outside [0, 1].")
                if len(recoverable_warnings) >= MAX_WARNING_EXAMPLES:
                    return

            if not (0.0 < width <= 1.0 and 0.0 < height <= 1.0):
                recoverable_warnings.append(
                    f"{label_path}: line {line_index} has width/height outside (0, 1]."
                )
                if len(recoverable_warnings) >= MAX_WARNING_EXAMPLES:
                    return

        if label_format == "segment":
            if len(coords) < 6 or len(coords) % 2 != 0:
                recoverable_warnings.append(
                    f"{label_path}: line {line_index} has an invalid polygon point count."
                )
                if len(recoverable_warnings) >= MAX_WARNING_EXAMPLES:
                    return

            for coord in coords:
                if not 0.0 <= coord <= 1.0:
                    recoverable_warnings.append(
                        f"{label_path}: line {line_index} has polygon coordinates outside [0, 1]."
                    )
                    if len(recoverable_warnings) >= MAX_WARNING_EXAMPLES:
                        return
                    break

    if len(file_formats) > 1:
        recoverable_warnings.append(f"{label_path}: mixed detect/segment lines were found in one file.")


def validate_split(
    dataset_dir: Path,
    split_name: str,
    class_count: int,
    errors: list[str],
    recoverable_warnings: list[str],
    format_counter: Counter[str],
) -> None:
    """Validate one dataset split directory."""
    split_dir = dataset_dir / split_name
    images_dir = split_dir / "images"
    labels_dir = split_dir / "labels"

    if not split_dir.exists():
        errors.append(f"Missing split directory: {split_dir}")
        return
    if not images_dir.exists():
        errors.append(f"Missing images directory: {images_dir}")
        return
    if not labels_dir.exists():
        errors.append(f"Missing labels directory: {labels_dir}")
        return

    image_files = sorted(
        file_path for file_path in images_dir.iterdir() if file_path.suffix.lower() in IMAGE_EXTENSIONS
    )
    label_files = sorted(file_path for file_path in labels_dir.iterdir() if file_path.suffix.lower() == ".txt")

    print_status(
        "OK",
        f"{split_name}: images={len(image_files)}, labels={len(label_files)}",
    )

    if len(image_files) != len(label_files):
        errors.append(
            f"{split_name}: image/label count mismatch ({len(image_files)} vs {len(label_files)})."
        )

    image_stems = collect_stems(image_files)
    label_stems = collect_stems(label_files)
    missing_labels = sorted(image_stems - label_stems)
    missing_images = sorted(label_stems - image_stems)

    if missing_labels:
        preview = ", ".join(missing_labels[:MAX_ERROR_EXAMPLES])
        errors.append(f"{split_name}: images without labels: {preview}")
    if missing_images:
        preview = ", ".join(missing_images[:MAX_ERROR_EXAMPLES])
        errors.append(f"{split_name}: labels without images: {preview}")

    label_errors_before = len(errors)
    for label_file in label_files:
        validate_label_file(label_file, class_count, recoverable_warnings, format_counter)
        if len(errors) - label_errors_before >= MAX_ERROR_EXAMPLES:
            break


def build_recommended_yaml(dataset_dir: Path) -> str:
    """Return a stable recommendation for this dataset."""
    dataset_dir_posix = dataset_dir.resolve().as_posix()
    return "\n".join(
        [
            f"path: {dataset_dir_posix}",
            "train: train/images",
            "val: valid/images",
            "test: test/images",
            "names:",
            "  0: smoker",
        ]
    )


def main() -> int:
    """Run dataset checks and return an exit code."""
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()
    yaml_path = args.yaml.resolve() if args.yaml else (dataset_dir / "data.yaml").resolve()

    errors: list[str] = []
    warnings: list[str] = []
    format_counter: Counter[str] = Counter()

    print_status("INFO", f"Dataset directory: {dataset_dir}")
    print_status("INFO", f"YAML file: {yaml_path}")

    if not dataset_dir.exists():
        print_status("ERROR", f"Dataset directory does not exist: {dataset_dir}")
        return 1

    if not yaml_path.exists():
        print_status("ERROR", f"data.yaml does not exist: {yaml_path}")
        return 1

    yaml_data = load_yaml(yaml_path)
    if not yaml_data:
        print_status("ERROR", "Failed to parse data.yaml or it is empty.")
        return 1

    names = normalize_names_field(yaml_data.get("names"))
    nc_value = yaml_data.get("nc")
    recoverable_warnings: list[str] = []

    if nc_value != 1:
        errors.append(f"Expected nc=1, got {nc_value!r}.")
    else:
        print_status("OK", "data.yaml nc=1")

    if not are_class_names_compatible(names):
        errors.append(f"Expected names=['{EXPECTED_CLASS_NAME}'], got {names!r}.")
    else:
        normalized_name = normalize_class_name(names[0]) if names else EXPECTED_CLASS_NAME
        print_status("OK", f"data.yaml single class is compatible: {names!r}")
        if normalized_name != names[0].strip().lower():
            warnings.append(
                f"data.yaml class {names[0]!r} will be treated as '{EXPECTED_CLASS_NAME}' in this project."
            )

    class_count = len(names) if names else 0
    if class_count == 0:
        errors.append("No valid class names found in data.yaml.")
        class_count = 1

    for split_name in EXPECTED_SPLITS:
        validate_split(dataset_dir, split_name, class_count, errors, recoverable_warnings, format_counter)

    if format_counter["detect"]:
        print_status("INFO", f"Detection annotations found: {format_counter['detect']}")
    if format_counter["segment"]:
        print_status("INFO", f"Segmentation annotations found: {format_counter['segment']}")

    if format_counter["detect"] and format_counter["segment"]:
        errors.append(
            "Mixed annotation tasks detected: both detection and segmentation labels are present."
        )
    elif format_counter["segment"] and not format_counter["detect"]:
        errors.append(
            "This dataset looks like segmentation labels, not pure detection labels."
        )
    elif format_counter["detect"] and not format_counter["segment"]:
        print_status("OK", "Label task looks like pure detection.")

    yaml_root = resolve_yaml_root(yaml_data, yaml_path)
    print_status("INFO", f"Resolved YAML dataset root: {yaml_root}")

    yaml_split_mapping = {
        "train": yaml_data.get("train"),
        "val": yaml_data.get("val"),
        "test": yaml_data.get("test"),
    }

    for split_name, split_value in yaml_split_mapping.items():
        if split_value is None:
            warnings.append(f"data.yaml does not define '{split_name}'.")
            continue

        resolved_paths = resolve_split_path(split_value, yaml_root, dataset_dir)
        for resolved_path in resolved_paths:
            if resolved_path.exists():
                print_status("OK", f"data.yaml {split_name} -> {resolved_path}")
            else:
                warnings.append(
                    f"data.yaml {split_name} resolves to a missing path: {resolved_path}"
                )

    if recoverable_warnings:
        warnings.append(
            f"Found {len(recoverable_warnings)} recoverable annotation issue(s); the runtime builder will skip invalid lines."
        )
        warnings.extend(recoverable_warnings[:MAX_WARNING_EXAMPLES])

    if warnings:
        print_status("WARN", "Potential compatibility issues were found during dataset checks.")
        for warning in warnings:
            print_status("WARN", warning)
        print_status("INFO", "Recommended stable YAML configuration:")
        print(build_recommended_yaml(dataset_dir))

    if errors:
        print_status("ERROR", "Dataset validation failed.")
        for error in errors[:MAX_ERROR_EXAMPLES]:
            print_status("ERROR", error)
        if len(errors) > MAX_ERROR_EXAMPLES:
            print_status("ERROR", f"... and {len(errors) - MAX_ERROR_EXAMPLES} more issue(s).")
        return 1

    if warnings and args.strict:
        print_status("ERROR", "Validation finished with warnings in strict mode.")
        return 1

    if warnings:
        print_status("INFO", "Validation passed with warnings.")
    else:
        print_status("INFO", "Validation passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
