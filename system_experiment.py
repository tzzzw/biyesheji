"""Run system-level alarm experiments on a video source.

The script evaluates eight alarm-system combinations:
1. single-frame alarm vs temporal alarm
2. unrestricted area vs configured no-smoking zone
3. no cooldown vs cooldown enabled

If an event-level truth JSON is provided, it also computes system-level
precision, recall, F1, and average trigger delay.
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
from core.alarm_logic import TemporalAlarmStateMachine
from error_analysis import DetectionBox
from error_analysis import result_boxes_to_xyxy
from train_compare import ensure_dir
from train_compare import import_ultralytics_yolo
from train_compare import print_status
from core.zone_config import load_zone_config
from core.zone_config import resolve_zone_pixels


@dataclass
class CachedFrame:
    """One cached video frame prediction."""

    frame_index: int
    timestamp_sec: float
    width: int
    height: int
    pred_boxes: list[DetectionBox]


@dataclass
class TruthEvent:
    """One ground-truth smoking event interval."""

    start_sec: float
    end_sec: float
    label: str = "smoking"


@dataclass
class PredictedEvent:
    """One predicted system alarm event."""

    frame_index: int
    timestamp_sec: float
    best_confidence: float
    inside_count: int


@dataclass
class PositiveSegment:
    """One contiguous positive-evidence interval in the video."""

    segment_index: int
    start_frame_index: int
    end_frame_index: int
    start_sec: float
    end_sec: float
    duration_sec: float


@dataclass
class EventDetail:
    """One predicted alarm event annotated with its source interval."""

    mode: str
    zone: str
    cooldown: str
    event_index: int
    frame_index: int
    timestamp_sec: float
    best_confidence: float
    inside_count: int
    segment_index: int | None
    segment_start_frame_index: int | None
    segment_end_frame_index: int | None
    segment_start_sec: float | None
    segment_end_sec: float | None
    event_duration_sec: float | None
    trigger_latency_sec: float | None
    is_repeated_alarm: bool


@dataclass
class ExperimentRow:
    """One system experiment result row."""

    mode: str
    zone: str
    cooldown: str
    conf: float
    positive_segments: int
    predicted_events: int
    unique_triggered_segments: int
    repeated_alarm_count: int
    avg_trigger_latency_sec: float | None
    avg_event_duration_sec: float | None
    total_event_duration_sec: float | None
    matched_tp: int | None
    fp_events: int | None
    fn_events: int | None
    precision: float | None
    recall: float | None
    f1: float | None
    avg_trigger_delay_sec: float | None


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    script_dir = Path(__file__).resolve().parent
    default_name = datetime.now().strftime("system_experiment_%Y%m%d_%H%M%S")

    parser = argparse.ArgumentParser(
        description="Run system-level single-frame/temporal/zone/cooldown experiments on a video."
    )
    parser.add_argument(
        "--weights",
        type=Path,
        required=True,
        help="Detector weights used for all system experiments.",
    )
    parser.add_argument(
        "--video",
        type=Path,
        required=True,
        help="Video used for the system experiments.",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=script_dir / "runs",
        help="Directory containing experiment outputs.",
    )
    parser.add_argument(
        "--project",
        type=Path,
        default=None,
        help="Optional explicit experiment directory.",
    )
    parser.add_argument(
        "--name",
        default=default_name,
        help="Experiment directory name under --project or --runs-dir.",
    )
    parser.add_argument("--conf", type=float, required=True, help="Recommended detector confidence threshold.")
    parser.add_argument("--imgsz", type=int, default=896, help="Inference image size.")
    parser.add_argument("--device", default="0", help="Ultralytics device string.")
    parser.add_argument(
        "--iou",
        type=float,
        default=0.7,
        help="NMS IoU threshold for model.predict().",
    )
    parser.add_argument(
        "--zone-config",
        type=Path,
        default=None,
        help="Optional no-smoking zone config JSON.",
    )
    parser.add_argument(
        "--temporal-frames",
        type=int,
        default=8,
        help="Temporal alarm requires this many consecutive positive frames.",
    )
    parser.add_argument(
        "--temporal-avg-conf",
        type=float,
        default=0.45,
        help="Temporal alarm requires this minimum streak-average confidence.",
    )
    parser.add_argument(
        "--cooldown-seconds",
        type=float,
        default=3.0,
        help="Cooldown duration used by the cooldown-enabled variants.",
    )
    parser.add_argument(
        "--frame-step",
        type=int,
        default=1,
        help="Sample every Nth frame from the input video.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional cap for sampled frames.",
    )
    parser.add_argument(
        "--truth-json",
        type=Path,
        default=None,
        help="Optional event-level truth JSON with smoking intervals.",
    )
    parser.add_argument(
        "--match-tolerance-sec",
        type=float,
        default=0.0,
        help="Match predicted events to truth intervals with this tolerance.",
    )
    return parser.parse_args()


def point_inside_polygon(point: tuple[int, int], polygon: list[tuple[int, int]]) -> bool:
    """Return whether a point is inside or on the polygon boundary."""
    contour = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))
    return cv2.pointPolygonTest(contour, point, False) >= 0


def extract_frame_evidence(
    pred_boxes: list[DetectionBox],
    conf_threshold: float,
    zone_polygon: list[tuple[int, int]],
) -> FrameEvidence:
    """Aggregate one frame's inside-zone evidence."""
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
        if point_inside_polygon(center_point, zone_polygon):
            inside_count += 1
            confidence_sum += float(pred_box.conf)
            best_confidence = max(best_confidence, float(pred_box.conf))
    frame_average_confidence = confidence_sum / inside_count if inside_count > 0 else 0.0
    return FrameEvidence(
        inside_count=inside_count,
        frame_average_confidence=frame_average_confidence,
        best_confidence=best_confidence,
    )


def load_truth_events(truth_json: Path | None) -> list[TruthEvent]:
    """Load event-level truth intervals when available."""
    if truth_json is None:
        return []
    payload = json.loads(truth_json.read_text(encoding="utf-8"))
    events = payload.get("events")
    if not isinstance(events, list):
        raise ValueError("truth_json must contain an 'events' list.")
    truth_events: list[TruthEvent] = []
    for item in events:
        if not isinstance(item, dict):
            continue
        truth_events.append(
            TruthEvent(
                start_sec=float(item["start_sec"]),
                end_sec=float(item["end_sec"]),
                label=str(item.get("label", "smoking")),
            )
        )
    return truth_events


def resolve_zone_polygons(zone_config: Path | None, width: int, height: int) -> dict[str, list[tuple[int, int]]]:
    """Build both unrestricted and restricted polygons."""
    unrestricted = [(0, 0), (width - 1, 0), (width - 1, height - 1), (0, height - 1)]
    if zone_config is None:
        return {"none": unrestricted, "restricted": unrestricted}
    normalized_points = load_zone_config(zone_config.resolve())
    restricted = resolve_zone_pixels(normalized_points, width, height)
    return {"none": unrestricted, "restricted": restricted}


def cache_video_predictions(
    weights: Path,
    video_path: Path,
    imgsz: int,
    conf_threshold: float,
    iou: float,
    device: str,
    frame_step: int,
    max_frames: int | None,
) -> list[CachedFrame]:
    """Run detector inference once and cache per-frame predictions."""
    if frame_step < 1:
        raise ValueError("--frame-step must be at least 1.")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Failed to open video: {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 20.0)
    YOLO = import_ultralytics_yolo()
    model = YOLO(str(weights))

    cached_frames: list[CachedFrame] = []
    frame_index = 0
    kept_count = 0
    print_status("INFO", f"Caching predictions for video: {video_path}")
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
            cached_frames.append(
                CachedFrame(
                    frame_index=frame_index,
                    timestamp_sec=frame_index / max(fps, 1e-6),
                    width=width,
                    height=height,
                    pred_boxes=result_boxes_to_xyxy(result),
                )
            )
            if kept_count == 1 or kept_count % 100 == 0:
                print_status("INFO", f"Cached {kept_count} frames")
    finally:
        capture.release()

    return cached_frames


def run_single_frame_mode(
    evidences: list[tuple[CachedFrame, FrameEvidence]],
    cooldown_seconds: float,
) -> list[PredictedEvent]:
    """Trigger an event for any positive frame, with optional cooldown."""
    events: list[PredictedEvent] = []
    last_trigger_time: float | None = None
    latched_until_clear = False
    clear_start_time: float | None = None
    for cached_frame, evidence in evidences:
        if cooldown_seconds > 0.0 and latched_until_clear:
            if evidence.hit:
                clear_start_time = None
                continue

            if clear_start_time is None:
                clear_start_time = cached_frame.timestamp_sec
            clear_elapsed = cached_frame.timestamp_sec - clear_start_time
            clear_remaining = max(cooldown_seconds - clear_elapsed, 0.0)
            cooldown_remaining = 0.0
            if last_trigger_time is not None:
                cooldown_remaining = max(cached_frame.timestamp_sec - last_trigger_time, 0.0)
                cooldown_remaining = max(cooldown_seconds - cooldown_remaining, 0.0)
            if max(clear_remaining, cooldown_remaining) > 0.0:
                continue
            latched_until_clear = False
            clear_start_time = None

        if not evidence.hit:
            continue
        if last_trigger_time is not None and cooldown_seconds > 0.0:
            if cached_frame.timestamp_sec - last_trigger_time < cooldown_seconds:
                continue
        events.append(
            PredictedEvent(
                frame_index=cached_frame.frame_index,
                timestamp_sec=cached_frame.timestamp_sec,
                best_confidence=evidence.best_confidence,
                inside_count=evidence.inside_count,
            )
        )
        last_trigger_time = cached_frame.timestamp_sec
        if cooldown_seconds > 0.0:
            latched_until_clear = True
            clear_start_time = None
    return events


def run_temporal_mode(
    evidences: list[tuple[CachedFrame, FrameEvidence]],
    min_hit_frames: int,
    min_average_confidence: float,
    cooldown_seconds: float,
) -> list[PredictedEvent]:
    """Trigger events using the temporal alarm state machine."""
    state_machine = TemporalAlarmStateMachine(
        min_hit_frames=min_hit_frames,
        min_average_confidence=min_average_confidence,
        cooldown_seconds=cooldown_seconds,
    )
    events: list[PredictedEvent] = []
    for cached_frame, evidence in evidences:
        decision = state_machine.update(evidence=evidence, current_time=cached_frame.timestamp_sec)
        if not decision.should_trigger_alarm:
            continue
        events.append(
            PredictedEvent(
                frame_index=cached_frame.frame_index,
                timestamp_sec=cached_frame.timestamp_sec,
                best_confidence=decision.best_confidence,
                inside_count=decision.inside_count,
            )
        )
    return events


def average_or_none(values: list[float]) -> float | None:
    """Return the average of one float list, or None when empty."""
    if not values:
        return None
    return sum(values) / len(values)


def estimate_sample_period_sec(evidences: list[tuple[CachedFrame, FrameEvidence]]) -> float:
    """Estimate the per-sample duration from cached frames."""
    if len(evidences) < 2:
        return 0.0
    deltas = [
        max(evidences[index][0].timestamp_sec - evidences[index - 1][0].timestamp_sec, 0.0)
        for index in range(1, len(evidences))
    ]
    positive_deltas = [value for value in deltas if value > 0.0]
    if not positive_deltas:
        return 0.0
    return average_or_none(positive_deltas) or 0.0


def build_positive_segments(
    evidences: list[tuple[CachedFrame, FrameEvidence]],
) -> list[PositiveSegment]:
    """Group contiguous positive frames into event-like intervals."""
    sample_period_sec = estimate_sample_period_sec(evidences)
    segments: list[PositiveSegment] = []
    start_frame: CachedFrame | None = None
    end_frame: CachedFrame | None = None

    for cached_frame, evidence in evidences:
        if evidence.hit:
            if start_frame is None:
                start_frame = cached_frame
            end_frame = cached_frame
            continue

        if start_frame is None or end_frame is None:
            continue
        segments.append(
            PositiveSegment(
                segment_index=len(segments) + 1,
                start_frame_index=start_frame.frame_index,
                end_frame_index=end_frame.frame_index,
                start_sec=start_frame.timestamp_sec,
                end_sec=end_frame.timestamp_sec,
                duration_sec=max(end_frame.timestamp_sec - start_frame.timestamp_sec + sample_period_sec, 0.0),
            )
        )
        start_frame = None
        end_frame = None

    if start_frame is not None and end_frame is not None:
        segments.append(
            PositiveSegment(
                segment_index=len(segments) + 1,
                start_frame_index=start_frame.frame_index,
                end_frame_index=end_frame.frame_index,
                start_sec=start_frame.timestamp_sec,
                end_sec=end_frame.timestamp_sec,
                duration_sec=max(end_frame.timestamp_sec - start_frame.timestamp_sec + sample_period_sec, 0.0),
            )
        )
    return segments


def find_segment_for_event(
    predicted_event: PredictedEvent,
    segments: list[PositiveSegment],
) -> PositiveSegment | None:
    """Locate which positive segment produced one predicted alarm event."""
    for segment in segments:
        if segment.start_frame_index <= predicted_event.frame_index <= segment.end_frame_index:
            return segment
    return None


def build_event_details(
    mode: str,
    zone: str,
    cooldown: str,
    predicted_events: list[PredictedEvent],
    positive_segments: list[PositiveSegment],
) -> list[EventDetail]:
    """Annotate predicted events with interval-level metadata."""
    per_segment_counts: dict[int, int] = {}
    details: list[EventDetail] = []

    for event_index, predicted_event in enumerate(predicted_events, start=1):
        segment = find_segment_for_event(predicted_event, positive_segments)
        segment_index = segment.segment_index if segment is not None else None
        if segment_index is not None:
            per_segment_counts[segment_index] = per_segment_counts.get(segment_index, 0) + 1
        repeat_rank = per_segment_counts.get(segment_index or -1, 1)
        details.append(
            EventDetail(
                mode=mode,
                zone=zone,
                cooldown=cooldown,
                event_index=event_index,
                frame_index=predicted_event.frame_index,
                timestamp_sec=predicted_event.timestamp_sec,
                best_confidence=predicted_event.best_confidence,
                inside_count=predicted_event.inside_count,
                segment_index=segment.segment_index if segment is not None else None,
                segment_start_frame_index=segment.start_frame_index if segment is not None else None,
                segment_end_frame_index=segment.end_frame_index if segment is not None else None,
                segment_start_sec=segment.start_sec if segment is not None else None,
                segment_end_sec=segment.end_sec if segment is not None else None,
                event_duration_sec=segment.duration_sec if segment is not None else None,
                trigger_latency_sec=(
                    max(predicted_event.timestamp_sec - segment.start_sec, 0.0)
                    if segment is not None
                    else None
                ),
                is_repeated_alarm=repeat_rank > 1,
            )
        )
    return details


def summarize_event_details(
    event_details: list[EventDetail],
) -> tuple[int, int, float | None, float | None, float | None]:
    """Summarize event details for one experiment row."""
    unique_segments: dict[int, float] = {}
    trigger_latencies: list[float] = []
    repeated_alarm_count = 0

    for detail in event_details:
        if detail.is_repeated_alarm:
            repeated_alarm_count += 1
        if detail.trigger_latency_sec is not None:
            trigger_latencies.append(detail.trigger_latency_sec)
        if detail.segment_index is not None and detail.event_duration_sec is not None:
            unique_segments.setdefault(detail.segment_index, detail.event_duration_sec)

    unique_triggered_segments = len(unique_segments)
    event_durations = list(unique_segments.values())
    avg_trigger_latency_sec = average_or_none(trigger_latencies)
    avg_event_duration_sec = average_or_none(event_durations)
    total_event_duration_sec = sum(event_durations) if event_durations else None
    return (
        unique_triggered_segments,
        repeated_alarm_count,
        avg_trigger_latency_sec,
        avg_event_duration_sec,
        total_event_duration_sec,
    )


def match_events(
    predicted_events: list[PredictedEvent],
    truth_events: list[TruthEvent],
    tolerance_sec: float,
) -> tuple[int, int, int, float | None]:
    """Match predicted events to truth intervals greedily in time order."""
    if not truth_events:
        return 0, 0, 0, None

    matched_truth_indices: set[int] = set()
    matched_delays: list[float] = []
    tp = 0
    fp = 0
    for predicted_event in predicted_events:
        matched_index: int | None = None
        for truth_index, truth_event in enumerate(truth_events):
            if truth_index in matched_truth_indices:
                continue
            if predicted_event.timestamp_sec < truth_event.start_sec - tolerance_sec:
                continue
            if predicted_event.timestamp_sec > truth_event.end_sec + tolerance_sec:
                continue
            matched_index = truth_index
            break

        if matched_index is None:
            fp += 1
            continue

        matched_truth_indices.add(matched_index)
        tp += 1
        matched_delays.append(max(predicted_event.timestamp_sec - truth_events[matched_index].start_sec, 0.0))

    fn = max(len(truth_events) - len(matched_truth_indices), 0)
    avg_delay = sum(matched_delays) / len(matched_delays) if matched_delays else None
    return tp, fp, fn, avg_delay


def safe_ratio(numerator: int, denominator: int) -> float | None:
    """Compute a safe ratio."""
    if denominator <= 0:
        return None
    return numerator / denominator


def safe_f1(precision: float | None, recall: float | None) -> float | None:
    """Compute F1 from precision and recall."""
    if precision is None or recall is None:
        return None
    if precision + recall <= 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def format_metric(value: float | None) -> str:
    """Format one metric for markdown."""
    return "-" if value is None else f"{value:.4f}"


def format_metric_or_int(value: float | int | None) -> str:
    """Format one summary value for markdown."""
    if value is None:
        return "-"
    if isinstance(value, int):
        return str(value)
    return f"{value:.4f}"


def build_comparison_conclusions(rows: list[ExperimentRow], truth_events: list[TruthEvent]) -> list[str]:
    """Build short comparison conclusions from the experiment matrix."""
    conclusions: list[str] = []
    mode_groups = {
        "single_frame": [row for row in rows if row.mode == "single_frame"],
        "temporal": [row for row in rows if row.mode == "temporal"],
    }
    zone_groups = {
        "none": [row for row in rows if row.zone == "none"],
        "restricted": [row for row in rows if row.zone == "restricted"],
    }
    cooldown_groups = {
        "disabled": [row for row in rows if row.cooldown == "disabled"],
        "enabled": [row for row in rows if row.cooldown == "enabled"],
    }

    def avg_row_metric(group: list[ExperimentRow], field_name: str) -> float | None:
        values = [
            getattr(row, field_name)
            for row in group
            if isinstance(getattr(row, field_name), (int, float))
        ]
        normalized = [float(value) for value in values]
        return average_or_none(normalized)

    single_events = avg_row_metric(mode_groups["single_frame"], "predicted_events")
    temporal_events = avg_row_metric(mode_groups["temporal"], "predicted_events")
    single_repeat = avg_row_metric(mode_groups["single_frame"], "repeated_alarm_count")
    temporal_repeat = avg_row_metric(mode_groups["temporal"], "repeated_alarm_count")
    single_latency = avg_row_metric(mode_groups["single_frame"], "avg_trigger_latency_sec")
    temporal_latency = avg_row_metric(mode_groups["temporal"], "avg_trigger_latency_sec")
    if single_events is not None and temporal_events is not None:
        conclusions.append(
            "Temporal alarm vs single-frame alarm: "
            f"avg alarms {temporal_events:.2f} vs {single_events:.2f}, "
            f"avg repeated alarms {format_metric(temporal_repeat)} vs {format_metric(single_repeat)}, "
            f"avg trigger latency {format_metric(temporal_latency)}s vs {format_metric(single_latency)}s."
        )

    none_events = avg_row_metric(zone_groups["none"], "predicted_events")
    restricted_events = avg_row_metric(zone_groups["restricted"], "predicted_events")
    none_repeat = avg_row_metric(zone_groups["none"], "repeated_alarm_count")
    restricted_repeat = avg_row_metric(zone_groups["restricted"], "repeated_alarm_count")
    if none_events is not None and restricted_events is not None:
        conclusions.append(
            "Restricted zone vs unrestricted area: "
            f"avg alarms {restricted_events:.2f} vs {none_events:.2f}, "
            f"avg repeated alarms {format_metric(restricted_repeat)} vs {format_metric(none_repeat)}."
        )

    disabled_events = avg_row_metric(cooldown_groups["disabled"], "predicted_events")
    enabled_events = avg_row_metric(cooldown_groups["enabled"], "predicted_events")
    disabled_repeat = avg_row_metric(cooldown_groups["disabled"], "repeated_alarm_count")
    enabled_repeat = avg_row_metric(cooldown_groups["enabled"], "repeated_alarm_count")
    if disabled_events is not None and enabled_events is not None:
        conclusions.append(
            "Cooldown enabled vs disabled: "
            f"avg alarms {enabled_events:.2f} vs {disabled_events:.2f}, "
            f"avg repeated alarms {format_metric(enabled_repeat)} vs {format_metric(disabled_repeat)}."
        )

    if not truth_events:
        conclusions.append(
            "No truth_json was provided, so the recommendation is heuristic and prioritizes fewer repeated alarms, "
            "reasonable alarm count, and shorter trigger latency instead of formal precision/recall."
        )
    return conclusions


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    """Build a small markdown table."""
    header_line = "| " + " | ".join(headers) + " |"
    separator_line = "| " + " | ".join(["---"] * len(headers)) + " |"
    body_lines = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header_line, separator_line, *body_lines])


def main() -> int:
    """Run the full 2x2x2 system experiment matrix."""
    args = parse_args()
    output_dir = args.project.resolve() if args.project else args.runs_dir.resolve() / args.name
    ensure_dir(output_dir)

    truth_events = load_truth_events(args.truth_json.resolve() if args.truth_json else None)
    cached_frames = cache_video_predictions(
        weights=args.weights.resolve(),
        video_path=args.video.resolve(),
        imgsz=args.imgsz,
        conf_threshold=args.conf,
        iou=args.iou,
        device=args.device,
        frame_step=args.frame_step,
        max_frames=args.max_frames,
    )
    if not cached_frames:
        raise RuntimeError("No video frames were cached. Check the video path or frame-step setting.")

    zone_polygons = resolve_zone_polygons(
        zone_config=args.zone_config.resolve() if args.zone_config else None,
        width=cached_frames[0].width,
        height=cached_frames[0].height,
    )

    experiment_rows: list[ExperimentRow] = []
    experiment_events: dict[str, list[dict[str, Any]]] = {}
    event_details_rows: list[EventDetail] = []
    segment_payload: dict[str, list[dict[str, Any]]] = {}
    for mode in ("single_frame", "temporal"):
        for zone_name in ("none", "restricted"):
            evidences = [
                (
                    cached_frame,
                    extract_frame_evidence(
                        pred_boxes=cached_frame.pred_boxes,
                        conf_threshold=args.conf,
                        zone_polygon=zone_polygons[zone_name],
                    ),
                )
                for cached_frame in cached_frames
            ]
            positive_segments = build_positive_segments(evidences)
            for cooldown_name, cooldown_seconds in (
                ("disabled", 0.0),
                ("enabled", args.cooldown_seconds),
            ):
                if mode == "single_frame":
                    predicted_events = run_single_frame_mode(
                        evidences=evidences,
                        cooldown_seconds=cooldown_seconds,
                    )
                else:
                    predicted_events = run_temporal_mode(
                        evidences=evidences,
                        min_hit_frames=args.temporal_frames,
                        min_average_confidence=args.temporal_avg_conf,
                        cooldown_seconds=cooldown_seconds,
                    )

                event_details = build_event_details(
                    mode=mode,
                    zone=zone_name,
                    cooldown=cooldown_name,
                    predicted_events=predicted_events,
                    positive_segments=positive_segments,
                )
                (
                    unique_triggered_segments,
                    repeated_alarm_count,
                    avg_trigger_latency_sec,
                    avg_event_duration_sec,
                    total_event_duration_sec,
                ) = summarize_event_details(event_details)
                tp: int | None = None
                fp: int | None = None
                fn: int | None = None
                precision: float | None = None
                recall: float | None = None
                avg_delay: float | None = None
                if truth_events:
                    tp, fp, fn, avg_delay = match_events(
                        predicted_events=predicted_events,
                        truth_events=truth_events,
                        tolerance_sec=args.match_tolerance_sec,
                    )
                    precision = safe_ratio(tp, tp + fp)
                    recall = safe_ratio(tp, tp + fn)

                row = ExperimentRow(
                    mode=mode,
                    zone=zone_name,
                    cooldown=cooldown_name,
                    conf=args.conf,
                    positive_segments=len(positive_segments),
                    predicted_events=len(predicted_events),
                    unique_triggered_segments=unique_triggered_segments,
                    repeated_alarm_count=repeated_alarm_count,
                    avg_trigger_latency_sec=avg_trigger_latency_sec,
                    avg_event_duration_sec=avg_event_duration_sec,
                    total_event_duration_sec=total_event_duration_sec,
                    matched_tp=tp,
                    fp_events=fp,
                    fn_events=fn,
                    precision=precision,
                    recall=recall,
                    f1=safe_f1(precision, recall),
                    avg_trigger_delay_sec=avg_delay,
                )
                experiment_rows.append(row)
                key = f"{mode}__{zone_name}__{cooldown_name}"
                experiment_events[key] = [asdict(item) for item in predicted_events]
                event_details_rows.extend(event_details)
                segment_payload[key] = [asdict(item) for item in positive_segments]

    comparison_conclusions = build_comparison_conclusions(experiment_rows, truth_events)

    json_payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "output_dir": str(output_dir.resolve()),
        "weights": str(args.weights.resolve()),
        "video": str(args.video.resolve()),
        "conf": args.conf,
        "imgsz": args.imgsz,
        "device": args.device,
        "iou": args.iou,
        "temporal_frames": args.temporal_frames,
        "temporal_avg_conf": args.temporal_avg_conf,
        "cooldown_seconds": args.cooldown_seconds,
        "truth_json": str(args.truth_json.resolve()) if args.truth_json else None,
        "truth_event_count": len(truth_events),
        "rows": [asdict(row) for row in experiment_rows],
        "predicted_events": experiment_events,
        "event_details": [asdict(item) for item in event_details_rows],
        "positive_segments": segment_payload,
        "comparison_conclusions": comparison_conclusions,
    }
    (output_dir / "system_results.json").write_text(
        json.dumps(json_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    with (output_dir / "system_results.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(experiment_rows[0]).keys()))
        writer.writeheader()
        for row in experiment_rows:
            writer.writerow(asdict(row))

    if event_details_rows:
        with (output_dir / "system_event_details.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(asdict(event_details_rows[0]).keys()))
            writer.writeheader()
            for row in event_details_rows:
                writer.writerow(asdict(row))

    best_row = None
    if truth_events:
        best_row = max(
            experiment_rows,
            key=lambda item: (
                item.recall or -1.0,
                item.precision or -1.0,
                -(item.repeated_alarm_count),
                -(item.avg_trigger_delay_sec or 0.0),
            ),
        )
    else:
        best_row = max(
            experiment_rows,
            key=lambda item: (
                item.unique_triggered_segments,
                -(item.repeated_alarm_count),
                -(item.avg_trigger_latency_sec or 9999.0),
                -(item.predicted_events),
            ),
        )

    summary_lines = [
        "# System Experiment Summary",
        "",
        "## Run",
        f"- output_dir: `{output_dir.resolve()}`",
        f"- weights: `{args.weights.resolve()}`",
        f"- video: `{args.video.resolve()}`",
        f"- conf: `{args.conf}`",
        f"- temporal_frames: `{args.temporal_frames}`",
        f"- temporal_avg_conf: `{args.temporal_avg_conf}`",
        f"- cooldown_seconds: `{args.cooldown_seconds}`",
        f"- truth_event_count: `{len(truth_events)}`",
        "",
        "## Result Table",
        markdown_table(
            headers=[
                "Mode",
                "Zone",
                "Cooldown",
                "Alarm Count",
                "Repeated",
                "Avg Trigger(s)",
                "Avg Event(s)",
                "Total Event(s)",
                "Precision",
                "Recall",
                "F1",
                "Truth Delay(s)",
            ],
            rows=[
                [
                    row.mode,
                    row.zone,
                    row.cooldown,
                    str(row.predicted_events),
                    str(row.repeated_alarm_count),
                    format_metric(row.avg_trigger_latency_sec),
                    format_metric(row.avg_event_duration_sec),
                    format_metric(row.total_event_duration_sec),
                    format_metric(row.precision),
                    format_metric(row.recall),
                    format_metric(row.f1),
                    format_metric(row.avg_trigger_delay_sec),
                ]
                for row in experiment_rows
            ],
        ),
        "",
        "## Recommended System Setting",
        f"- mode: `{best_row.mode}`",
        f"- zone: `{best_row.zone}`",
        f"- cooldown: `{best_row.cooldown}`",
        f"- predicted_events: `{best_row.predicted_events}`",
        f"- repeated_alarm_count: `{best_row.repeated_alarm_count}`",
        f"- avg_trigger_latency_sec: `{format_metric(best_row.avg_trigger_latency_sec)}`",
        f"- avg_event_duration_sec: `{format_metric(best_row.avg_event_duration_sec)}`",
        f"- precision: `{format_metric(best_row.precision)}`",
        f"- recall: `{format_metric(best_row.recall)}`",
        f"- f1: `{format_metric(best_row.f1)}`",
        f"- avg_trigger_delay_sec: `{format_metric(best_row.avg_trigger_delay_sec)}`",
        "",
        "## Comparison Conclusions",
        *[f"- {item}" for item in comparison_conclusions],
    ]
    if not truth_events:
        summary_lines.append("- note: no truth_json was provided, so system-level precision/recall are left blank.")

    (output_dir / "summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    (output_dir / "system_results.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    (output_dir / "system_conclusion.md").write_text(
        "# System Experiment Conclusions\n\n" + "\n".join(f"- {item}" for item in comparison_conclusions) + "\n",
        encoding="utf-8",
    )
    print_status("OK", f"System experiment saved to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
