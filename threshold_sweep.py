"""Sweep confidence thresholds for formal detector evaluation.

The labeled-dataset mode runs inference once at a low confidence threshold,
then re-filters cached predictions to recompute precision, recall, AP50, and
AP50-95 for each threshold. Optional video mode exports proxy runtime stats
for the same threshold grid.
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
import numpy as np

from core.alarm_logic import FrameEvidence
from dataset_config import find_default_dataset_dir
from error_analysis import DetectionBox
from error_analysis import collect_images
from error_analysis import prepare_or_reuse_runtime_yaml
from error_analysis import read_gt_boxes
from error_analysis import resolve_split_dirs
from error_analysis import result_boxes_to_xyxy
from train_compare import ensure_dir
from train_compare import import_ultralytics_yolo
from train_compare import print_status
from core.zone_config import load_zone_config
from core.zone_config import resolve_zone_pixels


IOU_THRESHOLDS = [0.5 + index * 0.05 for index in range(10)]


@dataclass
class CachedImagePrediction:
    """One labeled image with cached predictions and GT."""

    image_path: str
    gt_boxes: list[DetectionBox]
    pred_boxes: list[DetectionBox]


@dataclass
class SweepRow:
    """One threshold-sweep metric row."""

    conf: float
    precision: float | None
    recall: float | None
    map50: float | None
    map50_95: float | None
    tp: int
    fp: int
    fn: int
    recommendation_reason: str | None = None


@dataclass
class CachedVideoFrame:
    """One video frame with cached predictions."""

    frame_index: int
    timestamp_sec: float
    width: int
    height: int
    pred_boxes: list[DetectionBox]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    script_dir = Path(__file__).resolve().parent
    workspace_dir = script_dir.parent
    default_dataset_dir = find_default_dataset_dir(workspace_dir)
    default_name = datetime.now().strftime("threshold_sweep_%Y%m%d_%H%M%S")

    parser = argparse.ArgumentParser(
        description=(
            "Sweep confidence thresholds on the labeled runtime dataset and "
            "optionally on a video source."
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
        help="Directory containing output artifacts.",
    )
    parser.add_argument(
        "--project",
        type=Path,
        default=None,
        help="Optional explicit output directory.",
    )
    parser.add_argument(
        "--name",
        default=default_name,
        help="Output directory name under --project or --runs-dir.",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        required=True,
        help="Weights used for the sweep.",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Explicit runtime data yaml. Default: prepare/reuse generated runtime dataset.",
    )
    parser.add_argument(
        "--split",
        choices=("train", "val", "valid", "test"),
        default="val",
        help="Labeled dataset split used for the formal threshold sweep.",
    )
    parser.add_argument("--imgsz", type=int, default=896, help="Inference image size.")
    parser.add_argument("--batch", type=int, default=48, help="Reserved for report metadata.")
    parser.add_argument(
        "--device",
        default="0",
        help="Ultralytics device string.",
    )
    parser.add_argument(
        "--conf-start",
        type=float,
        default=0.05,
        help="Sweep start confidence threshold.",
    )
    parser.add_argument(
        "--conf-stop",
        type=float,
        default=0.80,
        help="Sweep stop confidence threshold.",
    )
    parser.add_argument(
        "--conf-step",
        type=float,
        default=0.01,
        help="Sweep confidence step.",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.7,
        help="NMS IoU threshold for model.predict().",
    )
    parser.add_argument(
        "--match-iou",
        type=float,
        default=0.5,
        help="IoU threshold for the displayed precision/recall counts.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Optional cap for labeled-image inference.",
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=None,
        help="Optional video path for proxy runtime statistics.",
    )
    parser.add_argument(
        "--zone-config",
        type=Path,
        default=None,
        help="Optional zone polygon used for video proxy stats.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional cap for video frames.",
    )
    parser.add_argument(
        "--frame-step",
        type=int,
        default=1,
        help="Sample every Nth frame for video proxy stats.",
    )
    parser.add_argument(
        "--recommendation-strategy",
        choices=("balanced", "recall_first"),
        default="recall_first",
        help="Strategy used to recommend the final confidence threshold.",
    )
    parser.add_argument(
        "--precision-floor",
        type=float,
        default=0.75,
        help="Minimum precision used by the recall-first threshold recommendation.",
    )
    parser.add_argument(
        "--force-rebuild-runtime-data",
        action="store_true",
        help="Rebuild the runtime dataset before the sweep.",
    )
    return parser.parse_args()


def iter_conf_values(start: float, stop: float, step: float) -> list[float]:
    """Build a stable threshold grid."""
    if step <= 0:
        raise ValueError("--conf-step must be positive.")
    values: list[float] = []
    current = start
    while current <= stop + 1e-9:
        values.append(round(current, 6))
        current += step
    return values


def box_iou(box_a: DetectionBox, box_b: DetectionBox) -> float:
    """Compute IoU for two boxes."""
    inter_x1 = max(box_a.x1, box_b.x1)
    inter_y1 = max(box_a.y1, box_b.y1)
    inter_x2 = min(box_a.x2, box_b.x2)
    inter_y2 = min(box_a.y2, box_b.y2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    union_area = box_a.area + box_b.area - inter_area
    if union_area <= 0.0:
        return 0.0
    return inter_area / union_area


def evaluate_ap_at_iou(
    dataset: list[CachedImagePrediction],
    conf_threshold: float,
    match_iou: float,
) -> tuple[float, int, int, int]:
    """Compute AP and terminal TP/FP/FN at one IoU threshold."""
    detections: list[tuple[int, float, DetectionBox]] = []
    gt_count = 0
    for image_index, item in enumerate(dataset):
        gt_count += len(item.gt_boxes)
        for pred_box in item.pred_boxes:
            if pred_box.conf is None or pred_box.conf < conf_threshold:
                continue
            detections.append((image_index, float(pred_box.conf), pred_box))

    detections.sort(key=lambda item: item[1], reverse=True)
    matched_gt: dict[int, set[int]] = {index: set() for index in range(len(dataset))}
    tp_flags: list[int] = []
    fp_flags: list[int] = []

    for image_index, _conf, pred_box in detections:
        gt_boxes = dataset[image_index].gt_boxes
        best_gt_index: int | None = None
        best_iou = 0.0
        for gt_index, gt_box in enumerate(gt_boxes):
            if gt_index in matched_gt[image_index]:
                continue
            iou_value = box_iou(pred_box, gt_box)
            if iou_value >= match_iou and iou_value > best_iou:
                best_iou = iou_value
                best_gt_index = gt_index

        if best_gt_index is None:
            tp_flags.append(0)
            fp_flags.append(1)
            continue

        matched_gt[image_index].add(best_gt_index)
        tp_flags.append(1)
        fp_flags.append(0)

    cumulative_tp = 0
    cumulative_fp = 0
    precisions: list[float] = []
    recalls: list[float] = []
    for tp_flag, fp_flag in zip(tp_flags, fp_flags):
        cumulative_tp += tp_flag
        cumulative_fp += fp_flag
        precisions.append(cumulative_tp / max(cumulative_tp + cumulative_fp, 1))
        recalls.append(cumulative_tp / max(gt_count, 1))

    ap = 0.0
    for recall_level in range(101):
        threshold = recall_level / 100.0
        precision_candidates = [
            precision
            for precision, recall in zip(precisions, recalls)
            if recall >= threshold
        ]
        ap += (max(precision_candidates) if precision_candidates else 0.0) / 101.0

    tp_total = sum(tp_flags)
    fp_total = sum(fp_flags)
    fn_total = max(gt_count - tp_total, 0)
    return ap, tp_total, fp_total, fn_total


def evaluate_threshold(
    dataset: list[CachedImagePrediction],
    conf_threshold: float,
    display_match_iou: float,
) -> SweepRow:
    """Evaluate one confidence threshold."""
    ap50, _tp50, _fp50, _fn50 = evaluate_ap_at_iou(dataset, conf_threshold=conf_threshold, match_iou=0.5)
    _display_ap, display_tp, display_fp, display_fn = evaluate_ap_at_iou(
        dataset,
        conf_threshold=conf_threshold,
        match_iou=display_match_iou,
    )
    ap_values = [
        evaluate_ap_at_iou(dataset, conf_threshold=conf_threshold, match_iou=iou_threshold)[0]
        for iou_threshold in IOU_THRESHOLDS
    ]
    precision = display_tp / max(display_tp + display_fp, 1)
    recall = display_tp / max(display_tp + display_fn, 1)
    return SweepRow(
        conf=conf_threshold,
        precision=precision,
        recall=recall,
        map50=ap50,
        map50_95=sum(ap_values) / len(ap_values),
        tp=display_tp,
        fp=display_fp,
        fn=display_fn,
    )


def choose_recommended_row(rows: list[SweepRow], args: argparse.Namespace) -> SweepRow:
    """Choose the final threshold recommendation."""
    if not rows:
        raise ValueError("No sweep rows were produced.")

    if args.recommendation_strategy == "balanced":
        winner = max(
            rows,
            key=lambda item: (
                item.map50_95 or -1.0,
                item.map50 or -1.0,
                item.recall or -1.0,
                item.precision or -1.0,
                -item.conf,
            ),
        )
        winner.recommendation_reason = "max mAP50-95, then recall"
        return winner

    floor_candidates = [
        item
        for item in rows
        if item.precision is not None and item.precision >= args.precision_floor
    ]
    if floor_candidates:
        winner = max(
            floor_candidates,
            key=lambda item: (
                item.recall or -1.0,
                item.map50_95 or -1.0,
                item.map50 or -1.0,
                item.precision or -1.0,
                -item.conf,
            ),
        )
        winner.recommendation_reason = f"max recall with precision >= {args.precision_floor:.2f}"
        return winner

    winner = max(
        rows,
        key=lambda item: (
            item.map50_95 or -1.0,
            item.recall or -1.0,
            item.precision or -1.0,
            -item.conf,
        ),
    )
    winner.recommendation_reason = "fallback to max mAP50-95 because no threshold met the precision floor"
    return winner


def cache_labeled_predictions(
    weights: Path,
    runtime_yaml: Path,
    split: str,
    imgsz: int,
    conf_threshold: float,
    iou: float,
    device: str,
    max_images: int | None,
) -> list[CachedImagePrediction]:
    """Run inference once on the labeled split and cache all predictions."""
    images_dir, labels_dir = resolve_split_dirs(runtime_yaml=runtime_yaml, split=split)
    image_paths = collect_images(images_dir=images_dir, max_images=max_images)
    if not image_paths:
        raise FileNotFoundError(f"No images found under split '{split}': {images_dir}")

    YOLO = import_ultralytics_yolo()
    model = YOLO(str(weights))
    cached_items: list[CachedImagePrediction] = []

    print_status("INFO", f"Caching predictions for {len(image_paths)} labeled images.")
    for image_index, image_path in enumerate(image_paths, start=1):
        if image_index == 1 or image_index % 50 == 0 or image_index == len(image_paths):
            print_status("INFO", f"Inference {image_index}/{len(image_paths)}: {image_path.name}")

        predict_kwargs: dict[str, Any] = {
            "source": str(image_path),
            "imgsz": imgsz,
            "conf": conf_threshold,
            "iou": iou,
            "verbose": False,
            "save": False,
        }
        if device:
            predict_kwargs["device"] = device

        results = model.predict(**predict_kwargs)
        if not results:
            continue

        result = results[0]
        resolved_image_path = Path(str(result.path)).resolve()
        height, width = result.orig_shape
        label_path = labels_dir / f"{resolved_image_path.stem}.txt"
        cached_items.append(
            CachedImagePrediction(
                image_path=str(resolved_image_path),
                gt_boxes=read_gt_boxes(label_path=label_path, width=width, height=height),
                pred_boxes=result_boxes_to_xyxy(result),
            )
        )
    return cached_items


def point_inside_polygon_simple(point: tuple[int, int], polygon: list[tuple[int, int]]) -> bool:
    """Use cv2.pointPolygonTest without importing the realtime UI helpers."""
    contour = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))
    return cv2.pointPolygonTest(contour, point, False) >= 0


def extract_frame_evidence_from_boxes(
    pred_boxes: list[DetectionBox],
    conf_threshold: float,
    zone_polygon: list[tuple[int, int]],
) -> FrameEvidence:
    """Compute inside-zone evidence from cached boxes."""
    inside_count = 0
    confidence_sum = 0.0
    best_confidence = 0.0
    for pred_box in pred_boxes:
        if pred_box.conf is None or pred_box.conf < conf_threshold:
            continue
        center_point = (
            int(round((pred_box.x1 + pred_box.x2) / 2.0)),
            int(round((pred_box.y1 + pred_box.y2) / 2.0)),
        )
        if point_inside_polygon_simple(center_point, zone_polygon):
            inside_count += 1
            confidence_sum += float(pred_box.conf)
            best_confidence = max(best_confidence, float(pred_box.conf))
    frame_average_confidence = confidence_sum / inside_count if inside_count > 0 else 0.0
    return FrameEvidence(
        inside_count=inside_count,
        frame_average_confidence=frame_average_confidence,
        best_confidence=best_confidence,
    )


def resolve_zone_polygon(zone_config: Path | None, frame_width: int, frame_height: int) -> list[tuple[int, int]]:
    """Load or default the video zone polygon."""
    if zone_config is None:
        return [(0, 0), (frame_width - 1, 0), (frame_width - 1, frame_height - 1), (0, frame_height - 1)]
    normalized_points = load_zone_config(zone_config.resolve())
    return resolve_zone_pixels(normalized_points, frame_width, frame_height)


def cache_video_predictions(
    weights: Path,
    video_path: Path,
    imgsz: int,
    conf_threshold: float,
    iou: float,
    device: str,
    frame_step: int,
    max_frames: int | None,
) -> list[CachedVideoFrame]:
    """Run inference once on a video and cache frame predictions."""
    if frame_step < 1:
        raise ValueError("--frame-step must be at least 1.")
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Failed to open video: {video_path}")

    YOLO = import_ultralytics_yolo()
    model = YOLO(str(weights))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 20.0)
    cached_frames: list[CachedVideoFrame] = []
    frame_index = 0
    kept_count = 0

    print_status("INFO", f"Caching video predictions from: {video_path}")
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame_index += 1
            if (frame_index - 1) % frame_step != 0:
                continue
            kept_count += 1
            if max_frames is not None and kept_count > max_frames:
                break

            predict_kwargs: dict[str, Any] = {
                "source": frame,
                "imgsz": imgsz,
                "conf": conf_threshold,
                "iou": iou,
                "verbose": False,
                "save": False,
            }
            if device:
                predict_kwargs["device"] = device
            results = model.predict(**predict_kwargs)
            if not results:
                continue
            result = results[0]
            height, width = result.orig_shape
            timestamp_sec = frame_index / max(fps, 1e-6)
            cached_frames.append(
                CachedVideoFrame(
                    frame_index=frame_index,
                    timestamp_sec=timestamp_sec,
                    width=width,
                    height=height,
                    pred_boxes=result_boxes_to_xyxy(result),
                )
            )
            if kept_count == 1 or kept_count % 100 == 0:
                print_status("INFO", f"Cached video frame {kept_count}")
    finally:
        capture.release()

    return cached_frames


def build_video_proxy_rows(
    cached_frames: list[CachedVideoFrame],
    zone_config: Path | None,
    conf_values: list[float],
) -> list[dict[str, Any]]:
    """Build threshold-wise proxy video statistics."""
    if not cached_frames:
        return []

    first_frame = cached_frames[0]
    zone_polygon = resolve_zone_polygon(zone_config, first_frame.width, first_frame.height)
    rows: list[dict[str, Any]] = []
    for conf_value in conf_values:
        positive_frames = 0
        inside_positive_frames = 0
        mean_best_conf_sum = 0.0
        for frame in cached_frames:
            filtered = [
                box for box in frame.pred_boxes
                if box.conf is not None and box.conf >= conf_value
            ]
            if filtered:
                positive_frames += 1
                mean_best_conf_sum += max(float(box.conf or 0.0) for box in filtered)
            evidence = extract_frame_evidence_from_boxes(frame.pred_boxes, conf_value, zone_polygon)
            if evidence.hit:
                inside_positive_frames += 1

        rows.append(
            {
                "conf": conf_value,
                "video_frames": len(cached_frames),
                "positive_frames": positive_frames,
                "inside_zone_positive_frames": inside_positive_frames,
                "positive_frame_ratio": positive_frames / max(len(cached_frames), 1),
                "inside_zone_positive_ratio": inside_positive_frames / max(len(cached_frames), 1),
                "mean_best_conf_on_positive_frames": (
                    mean_best_conf_sum / positive_frames if positive_frames > 0 else 0.0
                ),
            }
        )
    return rows


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    """Write one CSV file."""
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def format_metric(value: float | None) -> str:
    """Format one metric."""
    return "-" if value is None else f"{value:.4f}"


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    """Build a markdown table."""
    header_line = "| " + " | ".join(headers) + " |"
    separator_line = "| " + " | ".join(["---"] * len(headers)) + " |"
    body_lines = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header_line, separator_line, *body_lines])


def main() -> int:
    """Run the confidence-threshold sweep."""
    args = parse_args()
    conf_values = iter_conf_values(args.conf_start, args.conf_stop, args.conf_step)
    output_dir = args.project.resolve() if args.project else args.runs_dir.resolve() / args.name
    ensure_dir(output_dir)

    if args.data is not None:
        runtime_yaml = args.data.resolve()
    else:
        runtime_yaml = prepare_or_reuse_runtime_yaml(
            dataset_dir=args.dataset_dir.resolve(),
            runtime_dataset_dir=args.runtime_dataset_dir.resolve(),
            force_rebuild=args.force_rebuild_runtime_data,
        )

    labeled_predictions = cache_labeled_predictions(
        weights=args.weights.resolve(),
        runtime_yaml=runtime_yaml,
        split=args.split,
        imgsz=args.imgsz,
        conf_threshold=min(conf_values),
        iou=args.iou,
        device=args.device,
        max_images=args.max_images,
    )
    sweep_rows = [
        evaluate_threshold(
            labeled_predictions,
            conf_threshold=value,
            display_match_iou=args.match_iou,
        )
        for value in conf_values
    ]
    recommended_row = choose_recommended_row(sweep_rows, args)

    video_rows: list[dict[str, Any]] = []
    if args.video is not None:
        cached_video = cache_video_predictions(
            weights=args.weights.resolve(),
            video_path=args.video.resolve(),
            imgsz=args.imgsz,
            conf_threshold=min(conf_values),
            iou=args.iou,
            device=args.device,
            frame_step=args.frame_step,
            max_frames=args.max_frames,
        )
        video_rows = build_video_proxy_rows(
            cached_frames=cached_video,
            zone_config=args.zone_config.resolve() if args.zone_config else None,
            conf_values=conf_values,
        )

    sweep_json = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "output_dir": str(output_dir.resolve()),
        "weights": str(args.weights.resolve()),
        "runtime_yaml": str(runtime_yaml.resolve()),
        "split": args.split,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "iou": args.iou,
        "match_iou": args.match_iou,
        "recommendation_strategy": args.recommendation_strategy,
        "precision_floor": args.precision_floor,
        "recommended_threshold": asdict(recommended_row),
        "rows": [asdict(item) for item in sweep_rows],
        "video_proxy_rows": video_rows,
    }
    (output_dir / "sweep.json").write_text(
        json.dumps(sweep_json, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    write_csv(
        output_dir / "sweep.csv",
        fieldnames=[
            "conf",
            "precision",
            "recall",
            "map50",
            "map50_95",
            "tp",
            "fp",
            "fn",
            "recommendation_reason",
        ],
        rows=[asdict(item) for item in sweep_rows],
    )
    if video_rows:
        write_csv(
            output_dir / "video_proxy.csv",
            fieldnames=list(video_rows[0].keys()),
            rows=video_rows,
        )

    top_rows = sorted(
        sweep_rows,
        key=lambda item: (item.map50_95 or -1.0, item.recall or -1.0),
        reverse=True,
    )[:10]
    summary_lines = [
        "# Threshold Sweep Summary",
        "",
        "## Run",
        f"- output_dir: `{output_dir.resolve()}`",
        f"- weights: `{args.weights.resolve()}`",
        f"- runtime_yaml: `{runtime_yaml.resolve()}`",
        f"- split: `{args.split}`",
        f"- conf_range: `{args.conf_start} -> {args.conf_stop} step {args.conf_step}`",
        f"- recommendation_strategy: `{args.recommendation_strategy}`",
        f"- precision_floor: `{args.precision_floor}`",
        "",
        "## Recommended Threshold",
        f"- conf: `{recommended_row.conf:.4f}`",
        f"- precision: `{format_metric(recommended_row.precision)}`",
        f"- recall: `{format_metric(recommended_row.recall)}`",
        f"- mAP50: `{format_metric(recommended_row.map50)}`",
        f"- mAP50-95: `{format_metric(recommended_row.map50_95)}`",
        f"- reason: `{recommended_row.recommendation_reason or '-'}`",
        "",
        "## Top Thresholds",
        markdown_table(
            headers=["conf", "Precision", "Recall", "mAP50", "mAP50-95"],
            rows=[
                [
                    f"{item.conf:.2f}",
                    format_metric(item.precision),
                    format_metric(item.recall),
                    format_metric(item.map50),
                    format_metric(item.map50_95),
                ]
                for item in top_rows
            ],
        ),
    ]
    if video_rows:
        best_video_row = max(video_rows, key=lambda item: item["inside_zone_positive_ratio"])
        summary_lines.extend(
            [
                "",
                "## Video Proxy",
                f"- source_video: `{args.video.resolve()}`",
                f"- sampled_frames: `{best_video_row['video_frames']}`",
                (
                    f"- highest_inside_zone_positive_ratio_conf: "
                    f"`{best_video_row['conf']:.4f}` "
                    f"(ratio={best_video_row['inside_zone_positive_ratio']:.4f})"
                ),
                "- note: video mode has no GT labels by default, so it exports proxy statistics instead of precision/recall/mAP.",
            ]
        )
    (output_dir / "summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print_status("OK", f"Threshold sweep saved to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
