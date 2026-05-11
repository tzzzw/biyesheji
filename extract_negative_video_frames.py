"""Extract reviewable negative frames from non-smoking videos.

This script:
1. Recursively scans a directory of videos.
2. Samples frames at a configurable interval.
3. Writes extracted frames into a flat image directory for manual review.
4. Writes CSV and Markdown reports for traceability.

It does not modify the YOLO training dataset.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2


VIDEO_EXTENSIONS = {
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".m4v",
    ".mpg",
    ".mpeg",
    ".wmv",
    ".flv",
    ".webm",
}
MAPPING_FILENAME = "negative_frame_extract_mapping.csv"
SUMMARY_FILENAME = "negative_frame_extract_summary.md"


@dataclass(frozen=True)
class VideoSummary:
    """Compact extraction summary for one source video."""

    video_path: Path
    fps: float
    frame_count: int | None
    duration_sec: float | None
    sample_step: int
    saved_frames: int


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    script_dir = Path(__file__).resolve().parent
    workspace_dir = script_dir.parent

    parser = argparse.ArgumentParser(
        description="Extract negative frames from videos for later manual review."
    )
    parser.add_argument(
        "--video-dir",
        type=Path,
        default=workspace_dir / "not_smoking_video",
        help="Directory containing non-smoking videos. Scanned recursively.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_dir / "generated_data" / "negative_video_frames_review",
        help="Output root for extracted review images and reports.",
    )
    parser.add_argument(
        "--frame-step",
        type=int,
        default=None,
        help="Save every Nth frame. Overrides --sample-seconds when provided.",
    )
    parser.add_argument(
        "--sample-seconds",
        type=float,
        default=1.0,
        help="Target sampling interval in seconds when --frame-step is not set.",
    )
    parser.add_argument(
        "--max-frames-per-video",
        type=int,
        default=None,
        help="Optional cap on saved frames per video.",
    )
    parser.add_argument(
        "--resize-max-side",
        type=int,
        default=None,
        help="Optionally resize frames so the longest side is at most this value.",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=95,
        help="JPEG quality for extracted frames. Default: %(default)s",
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Delete the existing output directory before extracting.",
    )
    return parser.parse_args()


def print_status(level: str, message: str) -> None:
    """Print a compact status line."""
    print(f"[{level}] {message}")


def validate_args(args: argparse.Namespace) -> None:
    """Validate user-provided options."""
    if args.frame_step is not None and args.frame_step < 1:
        raise ValueError("--frame-step must be at least 1.")
    if args.sample_seconds <= 0:
        raise ValueError("--sample-seconds must be positive.")
    if args.max_frames_per_video is not None and args.max_frames_per_video < 1:
        raise ValueError("--max-frames-per-video must be at least 1.")
    if args.resize_max_side is not None and args.resize_max_side < 32:
        raise ValueError("--resize-max-side must be at least 32 when provided.")
    if not 1 <= args.jpeg_quality <= 100:
        raise ValueError("--jpeg-quality must be between 1 and 100.")


def ensure_output_root(output_dir: Path, force_rebuild: bool) -> tuple[Path, Path, Path]:
    """Prepare the output directory layout."""
    if output_dir.exists() and force_rebuild:
        print_status("INFO", f"Removing existing output directory: {output_dir}")
        shutil.rmtree(output_dir)

    images_dir = output_dir / "images"
    mapping_path = output_dir / MAPPING_FILENAME
    summary_path = output_dir / SUMMARY_FILENAME

    if output_dir.exists() and any(output_dir.iterdir()) and not force_rebuild:
        raise FileExistsError(
            "Output directory already contains files. "
            "Use --force-rebuild to regenerate a clean review set:\n"
            f"{output_dir}"
        )

    images_dir.mkdir(parents=True, exist_ok=True)
    return images_dir, mapping_path, summary_path


def iter_video_files(video_dir: Path) -> list[Path]:
    """Collect supported video files recursively."""
    if not video_dir.exists():
        raise FileNotFoundError(f"Video directory does not exist: {video_dir}")
    if not video_dir.is_dir():
        raise NotADirectoryError(f"Video directory is not a folder: {video_dir}")

    video_paths = sorted(
        (
            path
            for path in video_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
        ),
        key=lambda path: str(path.relative_to(video_dir)).lower(),
    )
    if not video_paths:
        raise FileNotFoundError(f"No supported videos found under: {video_dir}")
    return video_paths


def sanitize_token(text: str) -> str:
    """Convert one path token into a filename-friendly piece."""
    cleaned = re.sub(r"[^\w\-]+", "_", text, flags=re.UNICODE).strip("_")
    return cleaned or "video"


def build_output_prefix(video_path: Path, video_dir: Path) -> str:
    """Build a stable prefix from the video's relative path."""
    relative = video_path.relative_to(video_dir).with_suffix("")
    return "__".join(sanitize_token(part) for part in relative.parts)


def compute_sample_step(fps: float, frame_step: int | None, sample_seconds: float) -> int:
    """Resolve the actual extraction interval in frames."""
    if frame_step is not None:
        return frame_step
    if fps > 1e-6 and math.isfinite(fps):
        return max(1, int(round(fps * sample_seconds)))
    return 30


def format_timestamp_tag(timestamp_sec: float | None) -> str:
    """Format a short timestamp tag for filenames."""
    if timestamp_sec is None or not math.isfinite(timestamp_sec):
        return "unknown"
    return f"{timestamp_sec:09.2f}".replace(".", "p")


def resize_frame(frame: object, resize_max_side: int | None) -> object:
    """Optionally resize a frame while preserving aspect ratio."""
    if resize_max_side is None:
        return frame

    height, width = frame.shape[:2]
    longest_side = max(width, height)
    if longest_side <= resize_max_side:
        return frame

    scale = resize_max_side / float(longest_side)
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    return cv2.resize(frame, (resized_width, resized_height), interpolation=cv2.INTER_AREA)


def make_unique_output_path(images_dir: Path, base_name: str) -> Path:
    """Avoid overwriting an already generated frame."""
    candidate = images_dir / f"{base_name}.jpg"
    if not candidate.exists():
        return candidate

    counter = 1
    while True:
        candidate = images_dir / f"{base_name}__{counter:03d}.jpg"
        if not candidate.exists():
            return candidate
        counter += 1


def save_frame_as_jpeg(frame: object, output_path: Path, jpeg_quality: int) -> None:
    """Write one frame as JPEG with Unicode-safe file output."""
    success, encoded = cv2.imencode(
        ".jpg",
        frame,
        [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)],
    )
    if not success:
        raise ValueError(f"failed to encode frame as JPEG: {output_path}")
    output_path.write_bytes(encoded.tobytes())


def extract_video_frames(
    video_path: Path,
    video_dir: Path,
    images_dir: Path,
    args: argparse.Namespace,
) -> tuple[VideoSummary | None, list[dict[str, str]], str | None]:
    """Extract review frames from one video and return manifest rows."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return None, [], "failed to open video"

    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count_raw = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        frame_count = frame_count_raw if frame_count_raw > 0 else None
        duration_sec = (frame_count / fps) if frame_count is not None and fps > 1e-6 else None
        sample_step = compute_sample_step(fps=fps, frame_step=args.frame_step, sample_seconds=args.sample_seconds)
        prefix = build_output_prefix(video_path=video_path, video_dir=video_dir)

        rows: list[dict[str, str]] = []
        saved_frames = 0
        frame_index = 0

        print_status("INFO", f"Extracting from video: {video_path.name} (step={sample_step})")
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            frame_index += 1
            if (frame_index - 1) % sample_step != 0:
                continue
            if args.max_frames_per_video is not None and saved_frames >= args.max_frames_per_video:
                break

            frame = resize_frame(frame=frame, resize_max_side=args.resize_max_side)
            height, width = frame.shape[:2]
            timestamp_sec = (frame_index - 1) / fps if fps > 1e-6 else None
            base_name = f"{prefix}__f{frame_index:06d}__t{format_timestamp_tag(timestamp_sec)}"
            output_path = make_unique_output_path(images_dir=images_dir, base_name=base_name)

            try:
                save_frame_as_jpeg(
                    frame=frame,
                    output_path=output_path,
                    jpeg_quality=args.jpeg_quality,
                )
            except (OSError, ValueError) as exc:
                return None, rows, f"failed to write frame image: {output_path} ({exc})"

            saved_frames += 1
            rows.append(
                {
                    "video_name": video_path.name,
                    "video_path": str(video_path.resolve()),
                    "frame_index": str(frame_index),
                    "timestamp_sec": "" if timestamp_sec is None else f"{timestamp_sec:.3f}",
                    "fps": "" if fps <= 1e-6 else f"{fps:.3f}",
                    "frame_count": "" if frame_count is None else str(frame_count),
                    "sample_step": str(sample_step),
                    "width": str(width),
                    "height": str(height),
                    "output_name": output_path.name,
                    "output_path": str(output_path.resolve()),
                }
            )

        summary = VideoSummary(
            video_path=video_path,
            fps=fps,
            frame_count=frame_count,
            duration_sec=duration_sec,
            sample_step=sample_step,
            saved_frames=saved_frames,
        )
        return summary, rows, None
    finally:
        capture.release()


def write_mapping_csv(rows: list[dict[str, str]], output_path: Path) -> None:
    """Write the per-frame manifest."""
    fieldnames = [
        "video_name",
        "video_path",
        "frame_index",
        "timestamp_sec",
        "fps",
        "frame_count",
        "sample_step",
        "width",
        "height",
        "output_name",
        "output_path",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(
    output_path: Path,
    video_dir: Path,
    output_dir: Path,
    images_dir: Path,
    args: argparse.Namespace,
    video_summaries: list[VideoSummary],
    failed_videos: list[tuple[Path, str]],
    total_frames: int,
) -> None:
    """Write a human-readable extraction summary."""
    lines: list[str] = [
        "# Negative Video Frame Extraction Summary",
        "",
        f"- Generated at: `{datetime.now().isoformat(timespec='seconds')}`",
        f"- Source video dir: `{video_dir.resolve()}`",
        f"- Review output dir: `{output_dir.resolve()}`",
        f"- Review images dir: `{images_dir.resolve()}`",
        f"- Sampling mode: `{'frame-step' if args.frame_step is not None else 'sample-seconds'}`",
        f"- Frame step override: `{args.frame_step if args.frame_step is not None else '-'}`",
        f"- Sample seconds: `{args.sample_seconds}`",
        f"- Max frames per video: `{args.max_frames_per_video if args.max_frames_per_video is not None else '-'}`",
        f"- Resize max side: `{args.resize_max_side if args.resize_max_side is not None else '-'}`",
        "",
        "## Overall",
        "",
        f"- Videos processed successfully: `{len(video_summaries)}`",
        f"- Videos failed: `{len(failed_videos)}`",
        f"- Extracted review images: `{total_frames}`",
        "",
        "## Per Video",
        "",
        "| Video | FPS | Frames | Duration(s) | Sample Step | Saved |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]

    for item in video_summaries:
        fps_text = "-" if item.fps <= 1e-6 else f"{item.fps:.2f}"
        frame_count_text = "-" if item.frame_count is None else str(item.frame_count)
        duration_text = "-" if item.duration_sec is None else f"{item.duration_sec:.2f}"
        lines.append(
            f"| `{item.video_path.name}` | {fps_text} | {frame_count_text} | "
            f"{duration_text} | {item.sample_step} | {item.saved_frames} |"
        )

    lines.extend(
        [
            "",
            "## Failed Videos",
            "",
        ]
    )
    if failed_videos:
        for path, reason in failed_videos:
            lines.append(f"- `{path.name}`: {reason}")
    else:
        lines.append("- None")

    lines.extend(
        [
            "",
            "## Next Step",
            "",
            "- 先人工检查 `images/` 里的抽帧结果，删除不想导入的数据。",
            "- 检查完成后，可以把 `images/` 目录作为后续负样本导入源。",
            (
                f"- 之后可运行：`python import_negative_samples.py --negative-dir "
                f"\"{images_dir.resolve()}\"`"
            ),
        ]
    )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    """Run frame extraction."""
    args = parse_args()
    validate_args(args)

    script_dir = Path(__file__).resolve().parent
    workspace_dir = script_dir.parent
    video_dir = args.video_dir.resolve() if args.video_dir.is_absolute() else (workspace_dir / args.video_dir).resolve()
    output_dir = args.output_dir.resolve() if args.output_dir.is_absolute() else (script_dir / args.output_dir).resolve()

    images_dir, mapping_path, summary_path = ensure_output_root(
        output_dir=output_dir,
        force_rebuild=args.force_rebuild,
    )
    video_paths = iter_video_files(video_dir)

    print_status("INFO", f"Found {len(video_paths)} videos under: {video_dir}")

    mapping_rows: list[dict[str, str]] = []
    video_summaries: list[VideoSummary] = []
    failed_videos: list[tuple[Path, str]] = []

    for video_path in video_paths:
        summary, rows, error = extract_video_frames(
            video_path=video_path,
            video_dir=video_dir,
            images_dir=images_dir,
            args=args,
        )
        mapping_rows.extend(rows)
        if error is not None:
            failed_videos.append((video_path, error))
            print_status("WARN", f"Skipped {video_path.name}: {error}")
            continue
        if summary is not None:
            video_summaries.append(summary)
            print_status("OK", f"{video_path.name}: saved {summary.saved_frames} frames")

    total_frames = sum(item.saved_frames for item in video_summaries)
    write_mapping_csv(rows=mapping_rows, output_path=mapping_path)
    write_summary(
        output_path=summary_path,
        video_dir=video_dir,
        output_dir=output_dir,
        images_dir=images_dir,
        args=args,
        video_summaries=video_summaries,
        failed_videos=failed_videos,
        total_frames=total_frames,
    )

    print_status("OK", f"Review images ready: {images_dir}")
    print_status("OK", f"Frame manifest written to: {mapping_path}")
    print_status("OK", f"Summary written to: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
