"""Shared dataset defaults and compatibility helpers."""

from __future__ import annotations

from pathlib import Path


EXPECTED_CLASS_NAME = "smoker"
ACCEPTED_CLASS_NAMES = ("smoker", "smoking")
DEFAULT_DATASET_DIR_CANDIDATES = ("Smoking_yolov8", "cigarette_smoker_yolov8")


def find_default_dataset_dir(workspace_dir: Path) -> Path:
    """Return the first existing preferred dataset directory, or the new default."""
    for directory_name in DEFAULT_DATASET_DIR_CANDIDATES:
        candidate = workspace_dir / directory_name
        if candidate.exists():
            return candidate
    return workspace_dir / DEFAULT_DATASET_DIR_CANDIDATES[0]


def normalize_class_name(name: str) -> str:
    """Normalize a dataset class label to the project's canonical class name."""
    lowered = name.strip().lower()
    if lowered in ACCEPTED_CLASS_NAMES:
        return EXPECTED_CLASS_NAME
    return lowered


def are_class_names_compatible(names: list[str]) -> bool:
    """Return True when the dataset classes can be treated as this project's smoker class."""
    if len(names) != 1:
        return False
    return normalize_class_name(names[0]) == EXPECTED_CLASS_NAME
