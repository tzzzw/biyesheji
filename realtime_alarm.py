"""Realtime smoking detection with temporal warning and interactive zones.

The script keeps the original project workflow and now supports:
1. Consecutive multi-frame warning logic.
2. Streak-average confidence filtering.
3. Cooldown suppression and visible alarm states.
4. Interactive no-smoking-zone drawing.
5. Zone config save/load for repeated experiments.
6. Event closure with snapshots, clips, and structured logs.

The script remains CPU-friendly by default when no device is specified.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2

from core.alarm_logic import AlarmDecision
from core.alarm_logic import AlarmState
from core.alarm_logic import FrameEvidence
from core.alarm_logic import TemporalAlarmStateMachine
from core.event_recorder import EventRecorder
from train_compare import import_ultralytics_yolo
from train_compare import print_status
from core.zone_config import draw_zone_interactively
from core.zone_config import draw_zone_overlay
from core.zone_config import load_zone_config
from core.zone_config import parse_zone_points
from core.zone_config import polygon_array
from core.zone_config import resolve_zone_pixels
from core.zone_config import save_zone_config


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description="Run realtime smoker detection with a no-smoking-zone alarm."
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=None,
        help="Model weights path. Default: auto-find the latest trained best.pt.",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=script_dir / "runs",
        help="Directory used to auto-find the latest training result.",
    )
    parser.add_argument(
        "--source",
        default="0",
        help="Webcam index like 0 or a local video file path.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Ultralytics device string. Default: cpu",
    )
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold.")
    parser.add_argument("--imgsz", type=int, default=416, help="Inference image size.")
    parser.add_argument(
        "--alarm-frames",
        type=int,
        default=8,
        help="Alarm triggers after this many consecutive inside-zone frames.",
    )
    parser.add_argument(
        "--alarm-avg-conf",
        type=float,
        default=0.45,
        help="Alarm also requires the streak-average confidence to reach this threshold.",
    )
    parser.add_argument(
        "--alarm-min-conf",
        type=float,
        default=0.55,
        help="Alarm only counts inside-zone detections whose confidence reaches this threshold.",
    )
    parser.add_argument(
        "--alarm-cooldown",
        type=float,
        default=3.0,
        help="Minimum seconds between two alarm beeps/log records.",
    )
    parser.add_argument(
        "--zone",
        nargs="*",
        default=None,
        help=(
            "Polygon points like 100,100 500,100 500,400 100,400. "
            "If every value is between 0 and 1, they are treated as normalized coordinates."
        ),
    )
    parser.add_argument(
        "--zone-config",
        type=Path,
        default=None,
        help="Load a zone polygon from a JSON config file. When used with --draw-zone, it also becomes the default save path.",
    )
    parser.add_argument(
        "--save-zone-config",
        type=Path,
        default=None,
        help="Save the current zone polygon to a JSON config file.",
    )
    parser.add_argument(
        "--draw-zone",
        action="store_true",
        help="Draw the zone interactively on the first frame before realtime detection starts.",
    )
    parser.add_argument(
        "--event-pre-seconds",
        type=float,
        default=2.0,
        help="How many seconds before the alarm frame should be kept in each event clip.",
    )
    parser.add_argument(
        "--event-post-seconds",
        type=float,
        default=2.0,
        help="How many seconds after the alarm frame should be recorded in each event clip.",
    )
    parser.add_argument(
        "--save-event-snapshot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to save an annotated snapshot for each violation event. Default: enabled.",
    )
    parser.add_argument(
        "--save-event-clip",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to save an annotated short clip for each violation event. Default: enabled.",
    )
    parser.add_argument(
        "--save-video",
        action="store_true",
        help="Save the annotated result video under smoke_project/runs/realtime_*.",
    )
    return parser.parse_args()


def validate_alarm_settings(args: argparse.Namespace) -> None:
    """Validate temporal alarm parameters."""
    if args.alarm_frames < 1:
        raise ValueError("--alarm-frames must be at least 1.")
    if not 0.0 <= args.alarm_avg_conf <= 1.0:
        raise ValueError("--alarm-avg-conf must be between 0 and 1.")
    if not 0.0 <= args.alarm_min_conf <= 1.0:
        raise ValueError("--alarm-min-conf must be between 0 and 1.")
    if args.alarm_cooldown < 0.0:
        raise ValueError("--alarm-cooldown must be non-negative.")
    if args.event_pre_seconds < 0.0:
        raise ValueError("--event-pre-seconds must be non-negative.")
    if args.event_post_seconds < 0.0:
        raise ValueError("--event-post-seconds must be non-negative.")


def parse_source(source: str) -> int | str:
    """Interpret webcam indices and video paths."""
    return int(source) if source.isdigit() else source


def point_inside_polygon(point: tuple[int, int], polygon: list[tuple[int, int]]) -> bool:
    """Return True when a point is inside or on the polygon edge."""
    contour = polygon_array(polygon)
    return cv2.pointPolygonTest(contour, point, False) >= 0


def find_latest_summary(runs_dir: Path) -> Path | None:
    """Find the newest comparison_summary.json."""
    if not runs_dir.exists():
        return None

    summaries = sorted(
        runs_dir.glob("**/comparison_summary.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return summaries[0] if summaries else None


def find_latest_weights(runs_dir: Path) -> Path | None:
    """Resolve the latest successful trained weight file."""
    summary_path = find_latest_summary(runs_dir)
    if summary_path and summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        for model_info in summary.get("models", []):
            best_weights = model_info.get("best_weights")
            status = model_info.get("status")
            if status == "success" and best_weights:
                weight_path = Path(str(best_weights))
                if weight_path.exists():
                    return weight_path

    candidates = sorted(
        runs_dir.glob("**/weights/best.pt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def ensure_writer(
    output_dir: Path,
    frame_width: int,
    frame_height: int,
    fps: float,
) -> cv2.VideoWriter:
    """Create a video writer for saving annotated output."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "realtime_result.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    safe_fps = fps if fps > 1 else 20.0
    return cv2.VideoWriter(str(output_path), fourcc, safe_fps, (frame_width, frame_height))


def play_alarm() -> None:
    """Trigger a simple local alarm sound."""
    try:
        import winsound

        winsound.Beep(1200, 300)
    except Exception:
        print("\a", end="", flush=True)


def resolve_base_zone_points(args: argparse.Namespace) -> list[tuple[float, float]]:
    """Resolve a zone polygon from config, CLI points, or the built-in default."""
    if args.zone_config and args.zone_config.exists():
        zone_points = load_zone_config(args.zone_config)
        print_status("INFO", f"Loaded zone config: {args.zone_config.resolve()}")
        return zone_points

    if args.zone_config and not args.zone_config.exists() and not args.draw_zone:
        raise FileNotFoundError(f"Zone config not found: {args.zone_config}")

    return parse_zone_points(args.zone)


def choose_zone_save_path(args: argparse.Namespace) -> Path | None:
    """Choose where the current zone config should be saved, if requested."""
    if args.save_zone_config is not None:
        return args.save_zone_config
    if args.draw_zone and args.zone_config is not None:
        return args.zone_config
    return None


def resolve_zone_for_session(
    args: argparse.Namespace,
    first_frame: Any,
) -> tuple[list[tuple[int, int]], Path | None]:
    """Resolve the runtime zone polygon and optionally save it."""
    frame_height, frame_width = first_frame.shape[:2]
    base_zone_points = resolve_base_zone_points(args)

    if args.draw_zone:
        initial_points = resolve_zone_pixels(base_zone_points, frame_width, frame_height)
        pixel_points = draw_zone_interactively(
            first_frame,
            initial_points=initial_points,
        )
    else:
        pixel_points = resolve_zone_pixels(base_zone_points, frame_width, frame_height)

    save_path = choose_zone_save_path(args)
    if save_path is not None:
        saved_path = save_zone_config(save_path, pixel_points, frame_width, frame_height)
        print_status("OK", f"Zone config saved to: {saved_path.resolve()}")
    return pixel_points, save_path


def extract_frame_evidence(
    result: Any,
    zone_polygon: list[tuple[int, int]],
    frame: Any,
    alarm_min_conf: float = 0.0,
) -> FrameEvidence:
    """Draw detections and summarize inside-zone evidence for one frame."""
    inside_count = 0
    inside_confidence_sum = 0.0
    best_confidence = 0.0
    candidate_inside_count = 0

    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return FrameEvidence(
            inside_count=0,
            frame_average_confidence=0.0,
            best_confidence=0.0,
            candidate_inside_count=0,
        )

    xyxy_values = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else boxes.xyxy
    conf_values = boxes.conf.cpu().numpy() if hasattr(boxes.conf, "cpu") else boxes.conf
    cls_values = boxes.cls.cpu().numpy() if hasattr(boxes.cls, "cpu") else boxes.cls

    for xyxy, confidence, class_id in zip(xyxy_values, conf_values, cls_values):
        if int(class_id) != 0:
            continue

        confidence_value = float(confidence)
        x1, y1, x2, y2 = [int(value) for value in xyxy]
        center_point = ((x1 + x2) // 2, (y1 + y2) // 2)
        is_inside = point_inside_polygon(center_point, zone_polygon)
        counts_for_alarm = is_inside and confidence_value >= alarm_min_conf

        if is_inside:
            candidate_inside_count += 1
        if counts_for_alarm:
            inside_count += 1
            inside_confidence_sum += confidence_value
            best_confidence = max(best_confidence, confidence_value)

        if counts_for_alarm:
            box_color = (0, 0, 255)
        elif is_inside:
            box_color = (0, 165, 255)
        else:
            box_color = (0, 200, 0)
        label = f"smoker {confidence_value:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
        cv2.circle(frame, center_point, 4, box_color, -1)
        cv2.putText(
            frame,
            label,
            (x1, max(y1 - 8, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            box_color,
            2,
            cv2.LINE_AA,
        )

    frame_average_confidence = (
        inside_confidence_sum / inside_count if inside_count > 0 else 0.0
    )
    return FrameEvidence(
        inside_count=inside_count,
        frame_average_confidence=frame_average_confidence,
        best_confidence=best_confidence,
        candidate_inside_count=candidate_inside_count,
    )


def status_panel_style(state: AlarmState) -> tuple[int, int, int]:
    """Choose a panel color for the current warning state."""
    if state == AlarmState.ALARMED:
        return (0, 0, 255)
    if state == AlarmState.COOLDOWN:
        return (0, 165, 255)
    if state == AlarmState.OBSERVING:
        return (60, 120, 30)
    return (80, 80, 80)


def build_status_lines(decision: AlarmDecision, args: argparse.Namespace) -> list[str]:
    """Build readable on-screen state descriptions."""
    first_line = (
        f"State={decision.state.value} | inside={decision.inside_count} | "
        f"hit_frames={decision.consecutive_hit_frames}/{args.alarm_frames}"
    )
    second_line = (
        f"frame_avg_conf={decision.frame_average_confidence:.2f} | "
        f"streak_avg_conf={decision.streak_average_confidence:.2f}/{args.alarm_avg_conf:.2f} | "
        f"best_conf={decision.best_confidence:.2f}/{args.alarm_min_conf:.2f}"
    )

    if decision.state == AlarmState.ALARMED:
        second_line = (
            f"Triggered: hit_frames={decision.consecutive_hit_frames}, "
            f"streak_avg_conf={decision.streak_average_confidence:.2f}, "
            f"best_conf={decision.best_confidence:.2f}"
        )
    elif decision.state == AlarmState.OBSERVING:
        remaining_frames = max(args.alarm_frames - decision.consecutive_hit_frames, 0)
        remaining_conf = max(args.alarm_avg_conf - decision.streak_average_confidence, 0.0)
        second_line = (
            f"Observe: need_frames={remaining_frames}, need_avg_conf={remaining_conf:.2f}, "
            f"best_conf={decision.best_confidence:.2f}/{args.alarm_min_conf:.2f}"
        )
    elif decision.state == AlarmState.IDLE:
        second_line = (
            f"Ready: alarm_frames={args.alarm_frames}, "
            f"alarm_avg_conf={args.alarm_avg_conf:.2f}, alarm_min_conf={args.alarm_min_conf:.2f}"
        )
    elif decision.state == AlarmState.COOLDOWN:
        if decision.waiting_for_clear and decision.cooldown_remaining <= 0.0:
            second_line = (
                f"Cooldown: best_conf={decision.best_confidence:.2f}/{args.alarm_min_conf:.2f}, "
                "waiting for target to clear"
            )
        elif decision.waiting_for_clear:
            second_line = (
                f"Cooldown: clear_remaining={decision.cooldown_remaining:.1f}s, "
                f"best_conf={decision.best_confidence:.2f}/{args.alarm_min_conf:.2f}"
            )
        else:
            second_line = (
                f"Cooldown: best_conf={decision.best_confidence:.2f}/{args.alarm_min_conf:.2f}, "
                f"remaining={decision.cooldown_remaining:.1f}s"
            )

    return [first_line, second_line]


def draw_status_panel(frame: Any, lines: list[str], color: tuple[int, int, int]) -> None:
    """Draw a multi-line top status panel."""
    frame_width = int(frame.shape[1])
    line_height = 28
    panel_height = 16 + line_height * len(lines)
    panel_right = max(min(frame_width - 10, 940), 320)
    cv2.rectangle(frame, (10, 10), (panel_right, 10 + panel_height), color, -1)

    for index, line in enumerate(lines):
        y = 38 + index * line_height
        cv2.putText(
            frame,
            line,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )


def main() -> int:
    """Run realtime detection and alarm."""
    args = parse_args()
    try:
        validate_alarm_settings(args)
    except ValueError as exc:
        print_status("ERROR", str(exc))
        return 1

    source = parse_source(args.source)
    runs_dir = args.runs_dir.resolve()

    weights_path = args.weights.resolve() if args.weights else None
    if weights_path is None:
        weights_path = find_latest_weights(runs_dir)

    if weights_path is None or not weights_path.exists():
        print_status(
            "ERROR",
            "No model weights found. Train a model first or pass --weights explicitly.",
        )
        return 1

    print_status("INFO", f"Using weights: {weights_path}")
    YOLO = import_ultralytics_yolo()
    model = YOLO(str(weights_path))

    if isinstance(source, int):
        capture = cv2.VideoCapture(source, cv2.CAP_DSHOW)
    else:
        capture = cv2.VideoCapture(source)

    if not capture.isOpened():
        print_status("ERROR", f"Failed to open source: {args.source}")
        return 1

    ok, first_frame = capture.read()
    if not ok:
        print_status("ERROR", "Failed to read the first frame from the source.")
        capture.release()
        return 1

    frame_height, frame_width = first_frame.shape[:2]
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 20.0)

    try:
        zone_polygon, saved_zone_path = resolve_zone_for_session(args, first_frame)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print_status("ERROR", str(exc))
        capture.release()
        cv2.destroyAllWindows()
        return 1

    if saved_zone_path is not None and not args.draw_zone:
        print_status("INFO", f"Zone config export complete: {saved_zone_path.resolve()}")

    output_dir = runs_dir / datetime.now().strftime("realtime_%Y%m%d_%H%M%S")
    writer = ensure_writer(output_dir, frame_width, frame_height, fps) if args.save_video else None
    event_recorder = EventRecorder(
        output_dir=output_dir,
        fps=fps,
        frame_width=frame_width,
        frame_height=frame_height,
        pre_event_seconds=args.event_pre_seconds,
        post_event_seconds=args.event_post_seconds,
        save_snapshot=args.save_event_snapshot,
        save_clip=args.save_event_clip,
    )

    state_machine = TemporalAlarmStateMachine(
        min_hit_frames=args.alarm_frames,
        min_average_confidence=args.alarm_avg_conf,
        cooldown_seconds=args.alarm_cooldown,
    )
    frame_counter = 0
    start_time = time.time()
    pending_frame = first_frame

    print_status("INFO", "Realtime alarm started. Press 'q' to quit.")

    try:
        while True:
            if pending_frame is not None:
                frame = pending_frame
                pending_frame = None
            else:
                ok, frame = capture.read()
                if not ok:
                    print_status("INFO", "Video stream ended or frame read failed.")
                    break

            frame_counter += 1
            results = model.predict(
                source=frame,
                conf=args.conf,
                imgsz=args.imgsz,
                device=args.device,
                verbose=False,
            )
            result = results[0]

            evidence = extract_frame_evidence(
                result,
                zone_polygon,
                frame,
                alarm_min_conf=args.alarm_min_conf,
            )
            decision = state_machine.update(evidence=evidence, current_time=time.time())

            draw_zone_overlay(
                frame,
                zone_polygon,
                active=evidence.candidate_inside_count > 0 or decision.should_trigger_alarm,
            )

            elapsed = max(time.time() - start_time, 1e-6)
            fps_text = frame_counter / elapsed
            cv2.putText(
                frame,
                f"FPS: {fps_text:.1f}",
                (10, frame_height - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            draw_status_panel(
                frame,
                build_status_lines(decision, args),
                status_panel_style(decision.state),
            )

            event_recorder.register_frame(
                frame,
                frame_counter,
                inside_count=decision.inside_count,
                best_confidence=decision.best_confidence,
            )

            if decision.should_trigger_alarm:
                play_alarm()
                event = event_recorder.trigger_event(
                    timestamp=datetime.now().isoformat(timespec="seconds"),
                    source=str(args.source),
                    weight=str(weights_path),
                    state=decision.state.value,
                    inside_count=decision.inside_count,
                    hit_frames=decision.consecutive_hit_frames,
                    frame_average_confidence=decision.frame_average_confidence,
                    streak_average_confidence=decision.streak_average_confidence,
                    best_confidence=decision.best_confidence,
                    frame_index=frame_counter,
                )
                print_status(
                    "OK",
                    f"Violation event recorded: {event.event_id} "
                    f"(snapshot={bool(event.snapshot_path)}, clip={bool(event.clip_path)})",
                )

            if writer is not None:
                writer.write(frame)

            cv2.imshow("Smoker Realtime Alarm", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
    finally:
        capture.release()
        event_recorder.close()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

    print_status("INFO", "Realtime alarm stopped.")
    if args.save_video or event_recorder.event_count > 0:
        print_status("INFO", f"Annotated outputs saved to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
