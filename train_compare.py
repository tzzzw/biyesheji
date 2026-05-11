"""Auxiliary YOLO comparison script for the smoker detection project.

This script does not modify the original dataset. Instead, it builds a
runtime detection-only dataset under smoke_project/generated_data by:
1. Hard-linking or copying images from the source dataset.
2. Converting any segmentation-style labels into detection boxes.
3. Writing a stable data.yaml for Ultralytics training.

It then trains one or two models for quick comparison, intended for
early-stage YOLOv8 vs YOLO26 checks and smoke runs.

This script is kept as an auxiliary tool. The current formal-training
mainline for the s-model experiments is search_train_config.py, which is
better suited for fixed-config verification, resume workflows, and
summary/report generation.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from dataset_config import EXPECTED_CLASS_NAME
from dataset_config import find_default_dataset_dir

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
EXPECTED_SPLITS = ("train", "valid", "test")


@dataclass
class PrepareStats:
    """Summary of runtime dataset preparation."""

    images_linked: int = 0
    images_copied: int = 0
    empty_labels: int = 0
    detect_lines_kept: int = 0
    segment_lines_converted: int = 0
    invalid_lines_skipped: int = 0
    files_with_mixed_annotations: int = 0


@dataclass
class TrainResult:
    """One model's training result summary."""

    label: str
    model: str
    status: str
    save_dir: str | None = None
    best_weights: str | None = None
    results_csv: str | None = None
    validation_save_dir: str | None = None
    validation_metrics: dict[str, float] | None = None
    error: str | None = None


def print_status(level: str, message: str) -> None:
    """Print a compact status line."""
    print(f"[{level}] {message}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    script_dir = Path(__file__).resolve().parent
    workspace_dir = script_dir.parent
    default_dataset_dir = find_default_dataset_dir(workspace_dir)

    parser = argparse.ArgumentParser(
        description=(
            "Prepare a detection-only runtime dataset and run an auxiliary "
            "one-model or two-model comparison. For the current formal "
            "s-model mainline, prefer search_train_config.py."
        )
    )
    parser.add_argument(
        "--mode",
        choices=("smoke", "formal"),
        default="formal",
        help="Training mode. 'smoke' is a short GPU test, 'formal' is the full training profile.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=default_dataset_dir,
        help="Source dataset root directory. Default: %(default)s",
    )
    parser.add_argument(
        "--runtime-dataset-dir",
        type=Path,
        default=script_dir / "generated_data" / "smoker_detection_runtime",
        help="Where the generated detection-only runtime dataset will be stored.",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=script_dir / "runs",
        help="Where training outputs and summaries will be stored.",
    )
    parser.add_argument(
        "--project",
        type=Path,
        default=None,
        help="Ultralytics project directory. Default: smoke_project/runs",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Run name under project. Default: auto-generated from mode and timestamp.",
    )
    parser.add_argument(
        "--model-a",
        default="yolov8n.pt",
        help="First model name or path. Default: yolov8n.pt",
    )
    parser.add_argument(
        "--model-b",
        default="yolo26n.pt",
        help="Second model name or path. Change this if your YOLO26 file uses a different name.",
    )
    parser.add_argument(
        "--label-a",
        default="yolov8",
        help="Display label for the first model in saved summaries.",
    )
    parser.add_argument(
        "--label-b",
        default="yolo26",
        help="Display label for the second model in saved summaries.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Training epochs for each model. Default depends on --mode.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=None,
        help="Input image size. Default depends on --mode.",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=None,
        help="Batch size. Default depends on --mode and is tuned for a single RTX 4090.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Dataloader worker count. Default depends on --mode.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Ultralytics device string, for example 0, cpu, or 0,1. Default: 0 for GPU training.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible training.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=None,
        help="Early stopping patience. Default depends on --mode.",
    )
    parser.add_argument(
        "--cpu-friendly",
        action="store_true",
        help="Use lighter defaults for CPU-only training: device=cpu, smaller imgsz/batch, workers=0.",
    )
    parser.add_argument(
        "--skip-model-b",
        action="store_true",
        help="Skip the second model so you can first verify a single-model training pipeline.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Only prepare the runtime dataset and stop before training.",
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Delete and rebuild the runtime dataset directory before preparing data.",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    """Create a directory if needed."""
    path.mkdir(parents=True, exist_ok=True)


def apply_cpu_friendly_settings(args: argparse.Namespace) -> None:
    """Lower resource usage for CPU-only environments."""
    if not args.cpu_friendly:
        return

    if not args.device:
        args.device = "cpu"
    if args.batch is None or args.batch > 4:
        args.batch = 4
    if args.imgsz is None or args.imgsz > 416:
        args.imgsz = 416
    if args.workers is None or args.workers != 0:
        args.workers = 0
    if args.epochs is None or args.epochs > 10:
        args.epochs = 10
    if args.patience is None or args.patience > 5:
        args.patience = 5

    print_status(
        "INFO",
        "CPU-friendly mode enabled: "
        "device="
        f"{args.device}, imgsz={args.imgsz}, batch={args.batch}, workers={args.workers}, "
        f"epochs={args.epochs}, patience={args.patience}",
    )


def apply_training_profile(args: argparse.Namespace) -> None:
    """Apply mode-aware defaults tuned for smoke tests and RTX 4090 training."""
    if args.cpu_friendly:
        return

    if args.mode == "smoke":
        if args.device is None:
            args.device = "0"
        if args.epochs is None:
            args.epochs = 5
        if args.imgsz is None:
            args.imgsz = 640
        if args.batch is None:
            args.batch = 96
        if args.workers is None:
            args.workers = 16
        if args.patience is None:
            args.patience = 5
    else:
        if args.device is None:
            args.device = "0"
        if args.epochs is None:
            args.epochs = 100
        if args.imgsz is None:
            args.imgsz = 640
        if args.batch is None:
            args.batch = 96
        if args.workers is None:
            args.workers = 16
        if args.patience is None:
            args.patience = 20

    print_status(
        "INFO",
        "Training profile ready: "
        f"mode={args.mode}, device={args.device}, epochs={args.epochs}, imgsz={args.imgsz}, "
        f"batch={args.batch}, workers={args.workers}, seed={args.seed}, patience={args.patience}",
    )


def resolve_project_and_name(args: argparse.Namespace, runs_dir: Path) -> tuple[Path, str]:
    """Resolve the Ultralytics project directory and run name."""
    project_dir = args.project.resolve() if args.project else runs_dir.resolve()
    run_name = args.name or datetime.now().strftime(f"{args.mode}_%Y%m%d_%H%M%S")
    return project_dir, run_name


def reset_runtime_dir(runtime_dir: Path) -> None:
    """Delete a previously generated runtime dataset directory."""
    if runtime_dir.exists():
        shutil.rmtree(runtime_dir)


def classify_annotation(parts: list[str]) -> str:
    """Classify a YOLO annotation line by token count."""
    if len(parts) == 5:
        return "detect"
    if len(parts) >= 7 and len(parts) % 2 == 1:
        return "segment"
    return "invalid"


def format_float(value: float) -> str:
    """Format floats compactly for YOLO text labels."""
    return f"{value:.6f}".rstrip("0").rstrip(".")


def segment_to_bbox(coords: list[float]) -> tuple[float, float, float, float]:
    """Convert normalized polygon coordinates into a normalized bbox."""
    xs = coords[0::2]
    ys = coords[1::2]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    x_center = (min_x + max_x) / 2.0
    y_center = (min_y + max_y) / 2.0
    width = max_x - min_x
    height = max_y - min_y
    return x_center, y_center, width, height


def convert_label_file(source_label: Path, target_label: Path, stats: PrepareStats) -> None:
    """Convert one source label file into a detection-only target label file."""
    output_lines: list[str] = []
    file_has_detect = False
    file_has_segment = False

    if not source_label.exists():
        target_label.write_text("", encoding="utf-8")
        stats.empty_labels += 1
        return

    lines = source_label.read_text(encoding="utf-8").splitlines()
    if not any(line.strip() for line in lines):
        target_label.write_text("", encoding="utf-8")
        stats.empty_labels += 1
        return

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        parts = stripped.split()
        annotation_type = classify_annotation(parts)
        if annotation_type == "invalid":
            stats.invalid_lines_skipped += 1
            continue

        try:
            class_id = int(parts[0])
            coords = [float(item) for item in parts[1:]]
        except ValueError:
            stats.invalid_lines_skipped += 1
            continue

        if class_id != 0:
            stats.invalid_lines_skipped += 1
            continue

        if annotation_type == "detect":
            if len(coords) != 4:
                stats.invalid_lines_skipped += 1
                continue

            x_center, y_center, width, height = coords
            if not (0.0 <= x_center <= 1.0 and 0.0 <= y_center <= 1.0):
                stats.invalid_lines_skipped += 1
                continue
            if not (0.0 < width <= 1.0 and 0.0 < height <= 1.0):
                stats.invalid_lines_skipped += 1
                continue

            output_lines.append(
                f"0 {format_float(x_center)} {format_float(y_center)} "
                f"{format_float(width)} {format_float(height)}"
            )
            stats.detect_lines_kept += 1
            file_has_detect = True
            continue

        if len(coords) < 6 or len(coords) % 2 != 0:
            stats.invalid_lines_skipped += 1
            continue

        if not all(0.0 <= value <= 1.0 for value in coords):
            stats.invalid_lines_skipped += 1
            continue

        x_center, y_center, width, height = segment_to_bbox(coords)
        if width <= 0.0 or height <= 0.0:
            stats.invalid_lines_skipped += 1
            continue

        output_lines.append(
            f"0 {format_float(x_center)} {format_float(y_center)} "
            f"{format_float(width)} {format_float(height)}"
        )
        stats.segment_lines_converted += 1
        file_has_segment = True

    if file_has_detect and file_has_segment:
        stats.files_with_mixed_annotations += 1

    target_label.write_text("\n".join(output_lines), encoding="utf-8")


def link_or_copy_image(source_image: Path, target_image: Path, stats: PrepareStats) -> None:
    """Hard-link images when possible, otherwise copy them."""
    if target_image.exists():
        return

    try:
        os.link(source_image, target_image)
        stats.images_linked += 1
    except OSError:
        shutil.copy2(source_image, target_image)
        stats.images_copied += 1


def iter_source_images(images_dir: Path) -> list[Path]:
    """Return sorted source image files."""
    return sorted(
        file_path
        for file_path in images_dir.iterdir()
        if file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTENSIONS
    )


def validate_source_dataset(dataset_dir: Path) -> None:
    """Check that the basic source dataset layout exists before runtime generation."""
    missing_paths: list[Path] = []
    for split_name in EXPECTED_SPLITS:
        missing_paths.extend(
            path
            for path in (
                dataset_dir / split_name,
                dataset_dir / split_name / "images",
                dataset_dir / split_name / "labels",
            )
            if not path.exists()
        )

    if missing_paths:
        missing_text = "\n".join(str(path) for path in missing_paths)
        raise FileNotFoundError(f"Source dataset is incomplete:\n{missing_text}")


def build_runtime_yaml(runtime_dataset_dir: Path) -> Path:
    """Write a stable Ultralytics data.yaml for the generated dataset."""
    yaml_path = runtime_dataset_dir / "data_runtime.yaml"
    yaml_text = "\n".join(
        [
            f"path: {runtime_dataset_dir.resolve().as_posix()}",
            "train: train/images",
            "val: valid/images",
            "test: test/images",
            "names:",
            f"  0: {EXPECTED_CLASS_NAME}",
            "",
        ]
    )
    yaml_path.write_text(yaml_text, encoding="utf-8")
    return yaml_path


def prepare_runtime_dataset(dataset_dir: Path, runtime_dataset_dir: Path, force_rebuild: bool) -> tuple[Path, PrepareStats]:
    """Generate a detection-only runtime dataset inside smoke_project."""
    validate_source_dataset(dataset_dir)
    if force_rebuild:
        print_status("INFO", f"Rebuilding runtime dataset: {runtime_dataset_dir}")
        reset_runtime_dir(runtime_dataset_dir)

    stats = PrepareStats()
    for split_name in EXPECTED_SPLITS:
        source_images_dir = dataset_dir / split_name / "images"
        source_labels_dir = dataset_dir / split_name / "labels"
        target_images_dir = runtime_dataset_dir / split_name / "images"
        target_labels_dir = runtime_dataset_dir / split_name / "labels"
        ensure_dir(target_images_dir)
        ensure_dir(target_labels_dir)

        source_images = iter_source_images(source_images_dir)
        print_status("INFO", f"Preparing split '{split_name}' with {len(source_images)} images.")

        for source_image in source_images:
            target_image = target_images_dir / source_image.name
            target_label = target_labels_dir / f"{source_image.stem}.txt"
            source_label = source_labels_dir / f"{source_image.stem}.txt"

            link_or_copy_image(source_image, target_image, stats)
            convert_label_file(source_label, target_label, stats)

    runtime_yaml = build_runtime_yaml(runtime_dataset_dir)
    manifest_path = runtime_dataset_dir / "prepare_summary.json"
    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source_dataset_dir": str(dataset_dir.resolve()),
        "runtime_dataset_dir": str(runtime_dataset_dir.resolve()),
        "runtime_yaml": str(runtime_yaml.resolve()),
        "stats": asdict(stats),
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print_status("OK", f"Runtime dataset ready: {runtime_dataset_dir}")
    print_status(
        "OK",
        "Conversion summary: "
        f"detect_kept={stats.detect_lines_kept}, "
        f"segment_converted={stats.segment_lines_converted}, "
        f"invalid_skipped={stats.invalid_lines_skipped}, "
        f"mixed_files={stats.files_with_mixed_annotations}",
    )
    print_status("OK", f"Runtime data.yaml: {runtime_yaml}")
    return runtime_yaml, stats


def import_ultralytics_yolo() -> Any:
    """Import Ultralytics lazily so --prepare-only works without the package."""
    try:
        from ultralytics import YOLO  # type: ignore
        from ultralytics import settings  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "ultralytics is not installed. Please run 'pip install ultralytics opencv-python' first."
        ) from exc
    settings.update(
        {
            "sync": False,
            "clearml": False,
            "comet": False,
            "dvc": False,
            "hub": False,
            "mlflow": False,
            "neptune": False,
            "raytune": False,
            "tensorboard": False,
            "wandb": False,
        }
    )
    return YOLO


def extract_numeric(value: Any) -> float | None:
    """Convert a metric value into float when possible."""
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def extract_metrics(val_result: Any) -> dict[str, float]:
    """Extract a compact set of validation metrics from Ultralytics results."""
    metrics: dict[str, float] = {}

    results_dict = getattr(val_result, "results_dict", None)
    if isinstance(results_dict, dict):
        preferred_keys = {
            "metrics/precision(B)": "precision",
            "metrics/recall(B)": "recall",
            "metrics/mAP50(B)": "map50",
            "metrics/mAP50-95(B)": "map50_95",
            "fitness": "fitness",
        }
        for raw_key, display_key in preferred_keys.items():
            value = extract_numeric(results_dict.get(raw_key))
            if value is not None:
                metrics[display_key] = value

    box_metrics = getattr(val_result, "box", None)
    if box_metrics is not None:
        fallback_mapping = {
            "mp": "precision",
            "mr": "recall",
            "map50": "map50",
            "map": "map50_95",
        }
        for raw_attr, display_key in fallback_mapping.items():
            if display_key in metrics:
                continue
            value = extract_numeric(getattr(box_metrics, raw_attr, None))
            if value is not None:
                metrics[display_key] = value

    return metrics


def latest_metrics_from_csv(results_csv: Path) -> dict[str, str]:
    """Read the last metrics row from Ultralytics results.csv when available."""
    if not results_csv.exists():
        return {}

    lines = results_csv.read_text(encoding="utf-8").splitlines()
    if len(lines) < 2:
        return {}

    headers = [item.strip() for item in lines[0].split(",")]
    values = [item.strip() for item in lines[-1].split(",")]
    return dict(zip(headers, values))


def validate_trained_model(
    best_weights: Path,
    label: str,
    runtime_yaml: Path,
    comparison_dir: Path,
    args: argparse.Namespace,
) -> tuple[str | None, dict[str, float] | None]:
    """Run validation automatically after each training job."""
    YOLO = import_ultralytics_yolo()
    validation_project = comparison_dir / "validation"
    ensure_dir(validation_project)
    print_status("INFO", f"Running automatic validation for {label}: {best_weights}")

    model = YOLO(str(best_weights))
    val_kwargs: dict[str, Any] = {
        "data": str(runtime_yaml),
        "split": "val",
        "imgsz": args.imgsz,
        "batch": args.batch,
        "project": str(validation_project),
        "name": label,
        "exist_ok": True,
    }
    if args.device:
        val_kwargs["device"] = args.device

    val_result = model.val(**val_kwargs)
    metrics = extract_metrics(val_result)
    validation_save_dir = validation_project / label
    if metrics:
        metric_preview = ", ".join(f"{key}={value:.4f}" for key, value in metrics.items())
        print_status("OK", f"{label} validation metrics: {metric_preview}")
    else:
        print_status("WARN", f"{label} validation finished, but no metrics were extracted.")

    return (
        str(validation_save_dir.resolve()) if validation_save_dir.exists() else None,
        metrics or None,
    )


def train_one_model(
    model_spec: str,
    label: str,
    runtime_yaml: Path,
    comparison_dir: Path,
    args: argparse.Namespace,
) -> TrainResult:
    """Train one model and return a summary result."""
    YOLO = import_ultralytics_yolo()
    model_run_dir = comparison_dir / label
    ensure_dir(comparison_dir)

    print_status("INFO", f"Starting training for {label}: {model_spec}")
    try:
        model = YOLO(model_spec)
        train_kwargs: dict[str, Any] = {
            "data": str(runtime_yaml),
            "epochs": args.epochs,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "workers": args.workers,
            "project": str(comparison_dir),
            "name": label,
            "exist_ok": True,
            "seed": args.seed,
            "patience": args.patience,
        }
        if args.device:
            train_kwargs["device"] = args.device

        model.train(**train_kwargs)
        best_weights = model_run_dir / "weights" / "best.pt"
        results_csv = model_run_dir / "results.csv"
        metrics = latest_metrics_from_csv(results_csv)
        validation_save_dir: str | None = None
        validation_metrics: dict[str, float] | None = None

        if best_weights.exists():
            validation_save_dir, validation_metrics = validate_trained_model(
                best_weights=best_weights,
                label=label,
                runtime_yaml=runtime_yaml,
                comparison_dir=comparison_dir,
                args=args,
            )

        if metrics:
            metric_preview = ", ".join(
                f"{key}={value}"
                for key, value in metrics.items()
                if key in {"epoch", "metrics/mAP50(B)", "metrics/mAP50-95(B)", "fitness"}
            )
            if metric_preview:
                print_status("OK", f"{label} final metrics: {metric_preview}")

        return TrainResult(
            label=label,
            model=model_spec,
            status="success",
            save_dir=str(model_run_dir.resolve()),
            best_weights=str(best_weights.resolve()) if best_weights.exists() else None,
            results_csv=str(results_csv.resolve()) if results_csv.exists() else None,
            validation_save_dir=validation_save_dir,
            validation_metrics=validation_metrics,
        )
    except Exception as exc:
        print_status("ERROR", f"{label} training failed: {exc}")
        return TrainResult(
            label=label,
            model=model_spec,
            status="failed",
            error=str(exc),
            save_dir=str(model_run_dir.resolve()) if model_run_dir.exists() else None,
        )


def save_comparison_summary(
    comparison_dir: Path,
    runtime_yaml: Path,
    prepare_stats: PrepareStats,
    results: list[TrainResult],
    args: argparse.Namespace,
) -> Path:
    """Persist a comparison summary JSON for later evaluation scripts."""
    summary_path = comparison_dir / "comparison_summary.json"
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "runtime_yaml": str(runtime_yaml.resolve()),
        "prepare_stats": asdict(prepare_stats),
        "training_args": {
            "mode": args.mode,
            "epochs": args.epochs,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "workers": args.workers,
            "device": args.device,
            "seed": args.seed,
            "patience": args.patience,
            "project": str(args.project) if args.project is not None else None,
            "name": args.name,
            "cpu_friendly": args.cpu_friendly,
            "skip_model_b": args.skip_model_b,
        },
        "models": [asdict(result) for result in results],
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary_path


def main() -> int:
    """Entry point for dataset preparation and training comparison."""
    args = parse_args()
    apply_training_profile(args)
    apply_cpu_friendly_settings(args)
    dataset_dir = args.dataset_dir.resolve()
    runtime_dataset_dir = args.runtime_dataset_dir.resolve()
    runs_dir = args.runs_dir.resolve()
    project_dir, run_name = resolve_project_and_name(args, runs_dir)

    print_status("INFO", f"Source dataset: {dataset_dir}")
    runtime_yaml, prepare_stats = prepare_runtime_dataset(
        dataset_dir=dataset_dir,
        runtime_dataset_dir=runtime_dataset_dir,
        force_rebuild=args.force_rebuild,
    )

    if args.prepare_only:
        print_status("INFO", "Preparation finished. Training was skipped because --prepare-only was used.")
        return 0

    comparison_dir = project_dir / run_name
    ensure_dir(comparison_dir)

    model_jobs = [(args.model_a, args.label_a)]
    if not args.skip_model_b:
        model_jobs.append((args.model_b, args.label_b))
    else:
        print_status("INFO", "Skipping model-b for this run.")

    results = [
        train_one_model(model_spec, label, runtime_yaml, comparison_dir, args)
        for model_spec, label in model_jobs
    ]
    summary_path = save_comparison_summary(comparison_dir, runtime_yaml, prepare_stats, results, args)

    success_count = sum(result.status == "success" for result in results)
    failed_count = len(results) - success_count
    print_status("INFO", f"Comparison summary saved to: {summary_path}")
    print_status("INFO", f"Training finished: success={success_count}, failed={failed_count}")

    if success_count == 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
