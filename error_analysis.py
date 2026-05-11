"""Run baseline-oriented error analysis for smoker detection models.

The script reuses the project's runtime-dataset preparation flow and exports:
1. image-level failure summaries
2. false-positive box details
3. false-negative box details
4. annotated typical failure examples
5. a markdown report for quick inspection
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import yaml

from dataset_config import find_default_dataset_dir
from train_compare import IMAGE_EXTENSIONS
from train_compare import ensure_dir
from train_compare import import_ultralytics_yolo
from train_compare import prepare_runtime_dataset
from train_compare import print_status


SMALL_BOX_AREA_RATIO = 0.01
MEDIUM_BOX_AREA_RATIO = 0.05


@dataclass
class DetectionBox:
    """Compact xyxy box representation."""

    x1: float
    y1: float
    x2: float
    y2: float
    cls: int = 0
    conf: float | None = None

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height


@dataclass
class ImageCase:
    """Image-level error-analysis result."""

    image_path: str
    label_path: str
    visualization_path: str | None
    gt_count: int
    pred_count: int
    tp: int
    fp: int
    fn: int
    failure_type: str
    small_fn: int
    medium_fn: int
    large_fn: int
    crowded_failure: bool
    max_pred_conf: float | None
    max_fp_conf: float | None


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    script_dir = Path(__file__).resolve().parent
    workspace_dir = script_dir.parent
    default_dataset_dir = find_default_dataset_dir(workspace_dir)
    default_name = datetime.now().strftime("error_analysis_%Y%m%d_%H%M%S")

    parser = argparse.ArgumentParser(
        description=(
            "Analyze false positives, false negatives, and typical failure cases "
            "for a frozen baseline or any explicit best.pt."
        )
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=default_dataset_dir,
        help="Source dataset root directory.",
    )
    parser.add_argument(
        "--runtime-dataset-dir",
        type=Path,
        default=script_dir / "generated_data" / "smoker_detection_runtime",
        help="Runtime detection-only dataset directory.",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=script_dir / "runs",
        help="Directory containing baseline snapshots and error-analysis outputs.",
    )
    parser.add_argument(
        "--project",
        type=Path,
        default=None,
        help="Optional explicit analysis directory. Default: smoke_project/runs/<name>",
    )
    parser.add_argument(
        "--name",
        default=default_name,
        help="Analysis directory name under --project or --runs-dir.",
    )
    parser.add_argument(
        "--baseline-manifest",
        type=Path,
        default=None,
        help="Optional baseline_manifest.json. Default: latest baseline manifest under --runs-dir.",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=None,
        help="Optional explicit best.pt path. Overrides --baseline-manifest.",
    )
    parser.add_argument(
        "--split",
        choices=("train", "val", "valid", "test"),
        default="test",
        help="Dataset split to analyze. Default: test.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=None,
        help="Inference image size. Default: infer from baseline args.yaml/manifest, else 896.",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Inference batch size for predict().",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Prediction confidence threshold.",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.7,
        help="NMS IoU threshold passed into Ultralytics predict().",
    )
    parser.add_argument(
        "--match-iou",
        type=float,
        default=0.5,
        help="IoU threshold used to match predictions to GT boxes.",
    )
    parser.add_argument(
        "--device",
        default="",
        help="Ultralytics device string, for example 0 or cpu. Empty means auto.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Optional cap for smoke testing the analysis pipeline.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=25,
        help="How many typical failure images to render per failure type.",
    )
    parser.add_argument(
        "--force-rebuild-runtime-data",
        action="store_true",
        help="Rebuild the runtime dataset before analysis.",
    )
    return parser.parse_args()


def find_latest_baseline_manifest(runs_dir: Path) -> Path | None:
    """Return the most recent baseline_manifest.json."""
    if not runs_dir.exists():
        return None

    manifests = sorted(
        runs_dir.glob("**/baseline_manifest.json"),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    return manifests[0] if manifests else None


def load_json_dict(path: Path) -> dict[str, Any]:
    """Read one JSON file into a dict."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in: {path}")
    return payload


def load_yaml_dict(path: Path) -> dict[str, Any]:
    """Read one YAML file into a dict."""
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def resolve_weights_and_imgsz(args: argparse.Namespace) -> tuple[Path, Path | None, int]:
    """Resolve which best.pt to analyze and what imgsz to use."""
    if args.weights is not None:
        return args.weights.resolve(), None, args.imgsz or 896

    manifest_path = args.baseline_manifest.resolve() if args.baseline_manifest else find_latest_baseline_manifest(args.runs_dir.resolve())
    if manifest_path is None:
        raise FileNotFoundError("No baseline_manifest.json found. Pass --weights or freeze a baseline first.")

    manifest = load_json_dict(manifest_path)
    frozen_artifacts = manifest.get("frozen_artifacts")
    training_args = manifest.get("training_args")
    if not isinstance(frozen_artifacts, dict):
        raise ValueError("baseline_manifest.json is missing frozen_artifacts.")

    weights = frozen_artifacts.get("weights")
    if not weights:
        raise ValueError("baseline_manifest.json does not contain the frozen weights path.")

    inferred_imgsz = args.imgsz
    if inferred_imgsz is None and isinstance(training_args, dict):
        raw_imgsz = training_args.get("imgsz")
        if isinstance(raw_imgsz, int):
            inferred_imgsz = raw_imgsz
        elif isinstance(raw_imgsz, float):
            inferred_imgsz = int(raw_imgsz)

    return Path(str(weights)).resolve(), manifest_path, inferred_imgsz or 896


def prepare_or_reuse_runtime_yaml(
    dataset_dir: Path,
    runtime_dataset_dir: Path,
    force_rebuild: bool,
) -> Path:
    """Reuse an existing runtime yaml when possible to avoid redundant relinking."""
    runtime_yaml = runtime_dataset_dir / "data_runtime.yaml"
    if runtime_yaml.exists() and not force_rebuild:
        print_status("INFO", f"Reusing runtime dataset: {runtime_yaml}")
        return runtime_yaml.resolve()

    prepared_runtime_yaml, _ = prepare_runtime_dataset(
        dataset_dir=dataset_dir,
        runtime_dataset_dir=runtime_dataset_dir,
        force_rebuild=force_rebuild,
    )
    return prepared_runtime_yaml


def resolve_split_dirs(runtime_yaml: Path, split: str) -> tuple[Path, Path]:
    """Resolve images and labels directories for one split from data_runtime.yaml."""
    config = load_yaml_dict(runtime_yaml)
    root = Path(str(config.get("path", runtime_yaml.parent))).resolve()

    split_key = "val" if split in {"val", "valid"} else split
    split_ref = config.get(split_key)
    if not isinstance(split_ref, str):
        raise ValueError(f"Split '{split}' is not defined in runtime yaml: {runtime_yaml}")

    images_dir = (root / split_ref).resolve() if not Path(split_ref).is_absolute() else Path(split_ref).resolve()
    labels_dir = images_dir.parent / "labels"
    if not images_dir.exists():
        raise FileNotFoundError(f"Split images dir does not exist: {images_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"Split labels dir does not exist: {labels_dir}")
    return images_dir, labels_dir


def collect_images(images_dir: Path, max_images: int | None) -> list[Path]:
    """Collect sorted image paths from one split."""
    images = sorted(
        path
        for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    return images[:max_images] if max_images is not None else images


def label_path_for_image(image_path: Path, labels_dir: Path) -> Path:
    """Resolve the paired label file for one image path."""
    return labels_dir / f"{image_path.stem}.txt"


def yolo_to_xyxy(parts: list[str], width: int, height: int) -> DetectionBox:
    """Convert one YOLO txt row into xyxy coordinates."""
    cls_id = int(float(parts[0]))
    cx = float(parts[1]) * width
    cy = float(parts[2]) * height
    box_w = float(parts[3]) * width
    box_h = float(parts[4]) * height
    x1 = cx - box_w / 2.0
    y1 = cy - box_h / 2.0
    x2 = cx + box_w / 2.0
    y2 = cy + box_h / 2.0
    return DetectionBox(x1=x1, y1=y1, x2=x2, y2=y2, cls=cls_id)


def read_gt_boxes(label_path: Path, width: int, height: int) -> list[DetectionBox]:
    """Read one detection label file as GT boxes."""
    if not label_path.exists():
        return []

    boxes: list[DetectionBox] = []
    for raw_line in label_path.read_text(encoding="utf-8").splitlines():
        parts = raw_line.strip().split()
        if len(parts) != 5:
            continue
        try:
            boxes.append(yolo_to_xyxy(parts, width=width, height=height))
        except ValueError:
            continue
    return boxes


def result_boxes_to_xyxy(result: Any) -> list[DetectionBox]:
    """Extract predicted boxes from one Ultralytics result."""
    boxes_obj = getattr(result, "boxes", None)
    if boxes_obj is None or len(boxes_obj) == 0:
        return []

    xyxy_items = boxes_obj.xyxy.cpu().tolist()
    cls_items = boxes_obj.cls.cpu().tolist() if getattr(boxes_obj, "cls", None) is not None else [0.0] * len(xyxy_items)
    conf_items = boxes_obj.conf.cpu().tolist() if getattr(boxes_obj, "conf", None) is not None else [None] * len(xyxy_items)

    boxes: list[DetectionBox] = []
    for xyxy, cls_id, conf in zip(xyxy_items, cls_items, conf_items):
        boxes.append(
            DetectionBox(
                x1=float(xyxy[0]),
                y1=float(xyxy[1]),
                x2=float(xyxy[2]),
                y2=float(xyxy[3]),
                cls=int(cls_id),
                conf=float(conf) if conf is not None else None,
            )
        )
    return boxes


def box_iou(box_a: DetectionBox, box_b: DetectionBox) -> float:
    """Compute IoU for two xyxy boxes."""
    inter_x1 = max(box_a.x1, box_b.x1)
    inter_y1 = max(box_a.y1, box_b.y1)
    inter_x2 = min(box_a.x2, box_b.x2)
    inter_y2 = min(box_a.y2, box_b.y2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    union = box_a.area + box_b.area - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def best_iou_to_any(box: DetectionBox, others: list[DetectionBox]) -> float:
    """Return the highest IoU from one box to a list."""
    if not others:
        return 0.0
    return max(box_iou(box, other) for other in others)


def size_bucket(box: DetectionBox, width: int, height: int) -> str:
    """Bucket one GT box by relative area."""
    if width <= 0 or height <= 0:
        return "unknown"
    area_ratio = box.area / float(width * height)
    if area_ratio < SMALL_BOX_AREA_RATIO:
        return "small"
    if area_ratio < MEDIUM_BOX_AREA_RATIO:
        return "medium"
    return "large"


def draw_box(image: Any, box: DetectionBox, color: tuple[int, int, int], label: str) -> None:
    """Draw one box and label on a cv2 image."""
    x1, y1, x2, y2 = (int(round(value)) for value in (box.x1, box.y1, box.x2, box.y2))
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    cv2.putText(
        image,
        label,
        (x1, max(16, y1 - 6)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv2.LINE_AA,
    )


def save_visualization(
    image_path: Path,
    save_path: Path,
    gt_boxes: list[DetectionBox],
    pred_boxes: list[DetectionBox],
    matched_pred_indices: set[int],
    unmatched_gt_indices: list[int],
    unmatched_pred_indices: list[int],
    summary_text: str,
) -> None:
    """Save one annotated failure visualization."""
    image = cv2.imread(str(image_path))
    if image is None:
        return

    for index, box in enumerate(gt_boxes):
        label = "GT" if index not in unmatched_gt_indices else "FN"
        color = (0, 200, 255) if index not in unmatched_gt_indices else (0, 165, 255)
        draw_box(image, box, color, label)

    for index, box in enumerate(pred_boxes):
        confidence = f"{box.conf:.2f}" if box.conf is not None else "-"
        if index in unmatched_pred_indices:
            draw_box(image, box, (0, 0, 255), f"FP {confidence}")
        elif index in matched_pred_indices:
            draw_box(image, box, (255, 120, 0), f"TP {confidence}")

    cv2.putText(
        image,
        summary_text,
        (16, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    ensure_dir(save_path.parent)
    cv2.imwrite(str(save_path), image)


def analyze_one_image(
    image_path: Path,
    label_path: Path,
    pred_boxes: list[DetectionBox],
    gt_boxes: list[DetectionBox],
    width: int,
    height: int,
    match_iou: float,
) -> tuple[ImageCase, list[dict[str, Any]], list[dict[str, Any]], set[int], list[int], list[int]]:
    """Match predictions to GT and produce image-level / box-level summaries."""
    matched_pred_indices: set[int] = set()
    matched_gt_indices: set[int] = set()
    matched_pairs: list[tuple[int, int]] = []

    pred_order = sorted(
        range(len(pred_boxes)),
        key=lambda index: pred_boxes[index].conf if pred_boxes[index].conf is not None else 0.0,
        reverse=True,
    )

    for pred_index in pred_order:
        best_gt_index: int | None = None
        best_iou = 0.0
        for gt_index, gt_box in enumerate(gt_boxes):
            if gt_index in matched_gt_indices:
                continue
            iou = box_iou(pred_boxes[pred_index], gt_box)
            if iou >= match_iou and iou > best_iou:
                best_iou = iou
                best_gt_index = gt_index

        if best_gt_index is None:
            continue
        matched_pred_indices.add(pred_index)
        matched_gt_indices.add(best_gt_index)
        matched_pairs.append((pred_index, best_gt_index))

    unmatched_pred_indices = [index for index in range(len(pred_boxes)) if index not in matched_pred_indices]
    unmatched_gt_indices = [index for index in range(len(gt_boxes)) if index not in matched_gt_indices]

    fp_records: list[dict[str, Any]] = []
    for pred_index in unmatched_pred_indices:
        pred_box = pred_boxes[pred_index]
        fp_records.append(
            {
                "image_path": str(image_path.resolve()),
                "label_path": str(label_path.resolve()),
                "confidence": pred_box.conf,
                "x1": pred_box.x1,
                "y1": pred_box.y1,
                "x2": pred_box.x2,
                "y2": pred_box.y2,
                "best_iou_to_gt": best_iou_to_any(pred_box, gt_boxes),
                "area_ratio": pred_box.area / float(width * height) if width > 0 and height > 0 else None,
            }
        )

    fn_records: list[dict[str, Any]] = []
    small_fn = 0
    medium_fn = 0
    large_fn = 0
    for gt_index in unmatched_gt_indices:
        gt_box = gt_boxes[gt_index]
        bucket = size_bucket(gt_box, width=width, height=height)
        if bucket == "small":
            small_fn += 1
        elif bucket == "medium":
            medium_fn += 1
        elif bucket == "large":
            large_fn += 1

        best_pred = None
        best_pred_iou = 0.0
        for pred_box in pred_boxes:
            iou = box_iou(gt_box, pred_box)
            if iou > best_pred_iou:
                best_pred_iou = iou
                best_pred = pred_box

        fn_records.append(
            {
                "image_path": str(image_path.resolve()),
                "label_path": str(label_path.resolve()),
                "x1": gt_box.x1,
                "y1": gt_box.y1,
                "x2": gt_box.x2,
                "y2": gt_box.y2,
                "best_iou_to_pred": best_pred_iou,
                "best_pred_conf": best_pred.conf if best_pred is not None else None,
                "area_ratio": gt_box.area / float(width * height) if width > 0 and height > 0 else None,
                "size_bucket": bucket,
            }
        )

    fp_count = len(unmatched_pred_indices)
    fn_count = len(unmatched_gt_indices)
    if fp_count == 0 and fn_count == 0:
        failure_type = "clean"
    elif fp_count > 0 and fn_count == 0:
        failure_type = "fp_only"
    elif fp_count == 0 and fn_count > 0:
        failure_type = "fn_only"
    else:
        failure_type = "mixed"

    max_pred_conf = max((box.conf for box in pred_boxes if box.conf is not None), default=None)
    max_fp_conf = max((pred_boxes[index].conf for index in unmatched_pred_indices if pred_boxes[index].conf is not None), default=None)

    case = ImageCase(
        image_path=str(image_path.resolve()),
        label_path=str(label_path.resolve()),
        visualization_path=None,
        gt_count=len(gt_boxes),
        pred_count=len(pred_boxes),
        tp=len(matched_pairs),
        fp=fp_count,
        fn=fn_count,
        failure_type=failure_type,
        small_fn=small_fn,
        medium_fn=medium_fn,
        large_fn=large_fn,
        crowded_failure=len(gt_boxes) >= 2 and (fp_count > 0 or fn_count > 0),
        max_pred_conf=max_pred_conf,
        max_fp_conf=max_fp_conf,
    )
    return case, fp_records, fn_records, matched_pred_indices, unmatched_gt_indices, unmatched_pred_indices


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    """Write one csv file."""
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def pick_top_cases(cases: list[ImageCase], failure_type: str, top_k: int) -> list[ImageCase]:
    """Choose top-k cases for one failure type."""
    filtered = [case for case in cases if case.failure_type == failure_type]
    return sorted(
        filtered,
        key=lambda case: (
            case.fn,
            case.fp,
            case.max_fp_conf if case.max_fp_conf is not None else 0.0,
            case.gt_count,
        ),
        reverse=True,
    )[:top_k]


def proxy_ratio(numerator: int, denominator: int) -> float | None:
    """Compute a safe ratio."""
    if denominator <= 0:
        return None
    return numerator / denominator


def format_ratio(value: float | None) -> str:
    """Format one ratio for markdown output."""
    return "-" if value is None else f"{value:.4f}"


def build_report_markdown(
    analysis_dir: Path,
    weights: Path,
    baseline_manifest: Path | None,
    split: str,
    imgsz: int,
    conf: float,
    match_iou: float,
    cases: list[ImageCase],
    fp_records: list[dict[str, Any]],
    fn_records: list[dict[str, Any]],
) -> str:
    """Compose typical_failures.md."""
    total_images = len(cases)
    total_gt = sum(case.gt_count for case in cases)
    total_pred = sum(case.pred_count for case in cases)
    total_tp = sum(case.tp for case in cases)
    total_fp = sum(case.fp for case in cases)
    total_fn = sum(case.fn for case in cases)

    fp_images = sum(1 for case in cases if case.fp > 0)
    fn_images = sum(1 for case in cases if case.fn > 0)
    mixed_images = sum(1 for case in cases if case.failure_type == "mixed")
    crowded_failures = sum(1 for case in cases if case.crowded_failure)
    high_conf_fp = sum(1 for record in fp_records if (record.get("confidence") or 0.0) >= 0.5)

    top_mixed = pick_top_cases(cases, "mixed", 10)
    top_fp_only = pick_top_cases(cases, "fp_only", 10)
    top_fn_only = pick_top_cases(cases, "fn_only", 10)

    def format_case_line(case: ImageCase) -> str:
        visual = case.visualization_path or "-"
        return (
            f"- image: `{case.image_path}` | tp={case.tp}, fp={case.fp}, fn={case.fn}, "
            f"gt={case.gt_count}, pred={case.pred_count}, visual=`{visual}`"
        )

    lines = [
        "# Error Analysis",
        "",
        "## Run",
        f"- created_at: `{datetime.now().isoformat(timespec='seconds')}`",
        f"- analysis_dir: `{analysis_dir.resolve()}`",
        f"- weights: `{weights.resolve()}`",
        f"- baseline_manifest: `{baseline_manifest.resolve() if baseline_manifest else '-'}`",
        f"- split: `{split}`",
        f"- imgsz: `{imgsz}`",
        f"- conf: `{conf}`",
        f"- match_iou: `{match_iou}`",
        "",
        "## Summary",
        f"- total_images: `{total_images}`",
        f"- total_gt_boxes: `{total_gt}`",
        f"- total_pred_boxes: `{total_pred}`",
        f"- total_tp: `{total_tp}`",
        f"- total_fp: `{total_fp}`",
        f"- total_fn: `{total_fn}`",
        f"- proxy_precision: `{format_ratio(proxy_ratio(total_tp, total_tp + total_fp))}`",
        f"- proxy_recall: `{format_ratio(proxy_ratio(total_tp, total_tp + total_fn))}`",
        f"- fp_images: `{fp_images}`",
        f"- fn_images: `{fn_images}`",
        f"- mixed_images: `{mixed_images}`",
        f"- crowded_failure_images: `{crowded_failures}`",
        f"- high_conf_fp_boxes(conf>=0.5): `{high_conf_fp}`",
        f"- missed_small_gt: `{sum(case.small_fn for case in cases)}`",
        f"- missed_medium_gt: `{sum(case.medium_fn for case in cases)}`",
        f"- missed_large_gt: `{sum(case.large_fn for case in cases)}`",
        "",
        "## Typical Mixed Failures",
    ]
    lines.extend(format_case_line(case) for case in top_mixed) if top_mixed else lines.append("- none")

    lines.extend(["", "## Typical False Positives"])
    lines.extend(format_case_line(case) for case in top_fp_only) if top_fp_only else lines.append("- none")

    lines.extend(["", "## Typical False Negatives"])
    lines.extend(format_case_line(case) for case in top_fn_only) if top_fn_only else lines.append("- none")

    lines.extend(
        [
            "",
            "## Exported Files",
            f"- image_summary.csv: `{(analysis_dir / 'image_summary.csv').resolve()}`",
            f"- fp_images.csv: `{(analysis_dir / 'fp_images.csv').resolve()}`",
            f"- fn_images.csv: `{(analysis_dir / 'fn_images.csv').resolve()}`",
            f"- fp_boxes.csv: `{(analysis_dir / 'fp_boxes.csv').resolve()}`",
            f"- fn_boxes.csv: `{(analysis_dir / 'fn_boxes.csv').resolve()}`",
            f"- visuals_dir: `{(analysis_dir / 'visuals').resolve()}`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    """Run the end-to-end error analysis flow."""
    args = parse_args()
    weights, baseline_manifest, imgsz = resolve_weights_and_imgsz(args)
    if not weights.exists():
        raise FileNotFoundError(f"Weights file does not exist: {weights}")

    analysis_root = args.project.resolve() if args.project else args.runs_dir.resolve() / args.name
    ensure_dir(analysis_root)
    visuals_root = analysis_root / "visuals"
    ensure_dir(visuals_root)

    runtime_yaml = prepare_or_reuse_runtime_yaml(
        dataset_dir=args.dataset_dir.resolve(),
        runtime_dataset_dir=args.runtime_dataset_dir.resolve(),
        force_rebuild=args.force_rebuild_runtime_data,
    )
    images_dir, labels_dir = resolve_split_dirs(runtime_yaml=runtime_yaml, split=args.split)
    image_paths = collect_images(images_dir=images_dir, max_images=args.max_images)
    if not image_paths:
        raise FileNotFoundError(f"No images found for split '{args.split}' under: {images_dir}")

    YOLO = import_ultralytics_yolo()
    model = YOLO(str(weights))

    cases: list[ImageCase] = []
    fp_records: list[dict[str, Any]] = []
    fn_records: list[dict[str, Any]] = []
    render_payloads: dict[str, tuple[list[DetectionBox], list[DetectionBox], set[int], list[int], list[int], Path]] = {}

    print_status("INFO", f"Running error analysis on {len(image_paths)} images from split={args.split}")
    for image_index, image_path in enumerate(image_paths, start=1):
        if image_index == 1 or image_index % 25 == 0 or image_index == len(image_paths):
            print_status("INFO", f"Analyzing image {image_index}/{len(image_paths)}: {image_path.name}")

        predict_kwargs: dict[str, Any] = {
            "source": str(image_path),
            "imgsz": imgsz,
            "conf": args.conf,
            "iou": args.iou,
            "verbose": False,
            "save": False,
        }
        if args.device:
            predict_kwargs["device"] = args.device

        results = model.predict(**predict_kwargs)
        if not results:
            continue

        result = results[0]
        image_path = Path(str(result.path)).resolve()
        height, width = result.orig_shape
        label_path = label_path_for_image(image_path=image_path, labels_dir=labels_dir)
        pred_boxes = result_boxes_to_xyxy(result)
        gt_boxes = read_gt_boxes(label_path=label_path, width=width, height=height)
        case, image_fp_records, image_fn_records, matched_pred_indices, unmatched_gt_indices, unmatched_pred_indices = analyze_one_image(
            image_path=image_path,
            label_path=label_path,
            pred_boxes=pred_boxes,
            gt_boxes=gt_boxes,
            width=width,
            height=height,
            match_iou=args.match_iou,
        )
        cases.append(case)
        fp_records.extend(image_fp_records)
        fn_records.extend(image_fn_records)
        render_payloads[str(image_path)] = (
            gt_boxes,
            pred_boxes,
            matched_pred_indices,
            unmatched_gt_indices,
            unmatched_pred_indices,
            image_path,
        )

    for failure_type in ("mixed", "fp_only", "fn_only"):
        for case in pick_top_cases(cases, failure_type=failure_type, top_k=args.top_k):
            payload = render_payloads.get(case.image_path)
            if payload is None:
                continue
            gt_boxes, pred_boxes, matched_pred_indices, unmatched_gt_indices, unmatched_pred_indices, image_path = payload
            visual_path = visuals_root / failure_type / f"{Path(case.image_path).stem}.jpg"
            summary_text = f"{failure_type} tp={case.tp} fp={case.fp} fn={case.fn}"
            save_visualization(
                image_path=image_path,
                save_path=visual_path,
                gt_boxes=gt_boxes,
                pred_boxes=pred_boxes,
                matched_pred_indices=matched_pred_indices,
                unmatched_gt_indices=unmatched_gt_indices,
                unmatched_pred_indices=unmatched_pred_indices,
                summary_text=summary_text,
            )
            case.visualization_path = str(visual_path.resolve())

    case_rows = [asdict(case) for case in cases]
    write_csv(
        analysis_root / "image_summary.csv",
        case_rows,
        fieldnames=[
            "image_path",
            "label_path",
            "visualization_path",
            "gt_count",
            "pred_count",
            "tp",
            "fp",
            "fn",
            "failure_type",
            "small_fn",
            "medium_fn",
            "large_fn",
            "crowded_failure",
            "max_pred_conf",
            "max_fp_conf",
        ],
    )
    write_csv(
        analysis_root / "fp_images.csv",
        [row for row in case_rows if row["fp"] > 0],
        fieldnames=[
            "image_path",
            "label_path",
            "visualization_path",
            "gt_count",
            "pred_count",
            "tp",
            "fp",
            "fn",
            "failure_type",
            "small_fn",
            "medium_fn",
            "large_fn",
            "crowded_failure",
            "max_pred_conf",
            "max_fp_conf",
        ],
    )
    write_csv(
        analysis_root / "fn_images.csv",
        [row for row in case_rows if row["fn"] > 0],
        fieldnames=[
            "image_path",
            "label_path",
            "visualization_path",
            "gt_count",
            "pred_count",
            "tp",
            "fp",
            "fn",
            "failure_type",
            "small_fn",
            "medium_fn",
            "large_fn",
            "crowded_failure",
            "max_pred_conf",
            "max_fp_conf",
        ],
    )
    write_csv(
        analysis_root / "fp_boxes.csv",
        fp_records,
        fieldnames=[
            "image_path",
            "label_path",
            "confidence",
            "x1",
            "y1",
            "x2",
            "y2",
            "best_iou_to_gt",
            "area_ratio",
        ],
    )
    write_csv(
        analysis_root / "fn_boxes.csv",
        fn_records,
        fieldnames=[
            "image_path",
            "label_path",
            "x1",
            "y1",
            "x2",
            "y2",
            "best_iou_to_pred",
            "best_pred_conf",
            "area_ratio",
            "size_bucket",
        ],
    )

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "analysis_dir": str(analysis_root.resolve()),
        "weights": str(weights.resolve()),
        "baseline_manifest": str(baseline_manifest.resolve()) if baseline_manifest else None,
        "split": args.split,
        "imgsz": imgsz,
        "batch": args.batch,
        "conf": args.conf,
        "iou": args.iou,
        "match_iou": args.match_iou,
        "max_images": args.max_images,
        "cases": case_rows,
    }
    (analysis_root / "analysis_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    report_markdown = build_report_markdown(
        analysis_dir=analysis_root,
        weights=weights,
        baseline_manifest=baseline_manifest,
        split=args.split,
        imgsz=imgsz,
        conf=args.conf,
        match_iou=args.match_iou,
        cases=cases,
        fp_records=fp_records,
        fn_records=fn_records,
    )
    (analysis_root / "typical_failures.md").write_text(report_markdown, encoding="utf-8")

    print_status("OK", f"Error analysis saved to: {analysis_root}")
    print_status("OK", f"Typical failure report: {analysis_root / 'typical_failures.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
