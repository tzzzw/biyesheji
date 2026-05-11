"""Current formal-training mainline for smoker-detection experiments.

This script keeps the existing project structure intact and reuses the
runtime-dataset preparation logic from train_compare.py. It searches over
imgsz/batch combinations, runs short training jobs, records resource and
metric summaries, and recommends a formal-training configuration.

For the current s-model workflow, this is the preferred entry point for
fixed-config smoke checks, formal verification, and resumable long runs.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import threading
import time
from dataclasses import asdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from dataset_config import find_default_dataset_dir
from train_compare import PrepareStats
from train_compare import ensure_dir
from train_compare import extract_metrics
from train_compare import import_ultralytics_yolo
from train_compare import prepare_runtime_dataset
from train_compare import print_status


@dataclass
class SearchResult:
    """One imgsz/batch experiment summary."""

    imgsz: int
    batch: int
    epochs: int
    workers: int
    status: str
    run_dir: str | None = None
    best_weights: str | None = None
    results_csv: str | None = None
    validation_dir: str | None = None
    max_gpu_mem_mb: int | None = None
    total_train_time_sec: float | None = None
    first_epoch_time_sec: float | None = None
    avg_epoch_time_sec: float | None = None
    precision: float | None = None
    recall: float | None = None
    map50: float | None = None
    map50_95: float | None = None
    resume_used: bool = False
    recommendation: str | None = None
    recommendation_reason: str | None = None
    error: str | None = None


@dataclass
class EpochTimingSummary:
    """Compact timing stats extracted from Ultralytics results.csv."""

    epoch_count: int
    total_time_sec: float
    first_epoch_time_sec: float
    avg_epoch_time_sec: float


def resolve_primary_device_index(device: str | None) -> int | None:
    """Resolve the primary CUDA device index from an Ultralytics device string."""
    if device is None:
        return 0
    normalized = str(device).strip()
    if normalized == "" or normalized.lower() == "cpu":
        return None
    if "," in normalized:
        normalized = normalized.split(",", 1)[0].strip()
    try:
        return int(normalized)
    except ValueError:
        return None


class GpuMemoryMonitor:
    """Poll one GPU's memory usage while training is running."""

    def __init__(self, device: str | None, poll_interval: float = 0.5) -> None:
        self.device = device
        self.poll_interval = poll_interval
        self.max_memory_mb = 0
        self.total_memory_mb: int | None = None
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def _resolve_gpu_index(self) -> str | None:
        device_index = resolve_primary_device_index(self.device)
        return None if device_index is None else str(device_index)

    def _query_memory(self, field: str) -> int | None:
        gpu_index = self._resolve_gpu_index()
        if gpu_index is None:
            return None

        cmd = [
            "nvidia-smi",
            f"--query-gpu={field}",
            "--format=csv,noheader,nounits",
            "-i",
            gpu_index,
        ]
        try:
            completed = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
            )
        except (FileNotFoundError, subprocess.CalledProcessError):
            return None

        output = completed.stdout.strip().splitlines()
        if not output:
            return None
        try:
            return int(float(output[0].strip()))
        except ValueError:
            return None

    def _poll(self) -> None:
        while not self._stop_event.is_set():
            used_memory = self._query_memory("memory.used")
            if used_memory is not None:
                self.max_memory_mb = max(self.max_memory_mb, used_memory)
            time.sleep(self.poll_interval)

    def start(self) -> None:
        self.total_memory_mb = self._query_memory("memory.total")
        if self._resolve_gpu_index() is None:
            return
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)


_LOW_PREFETCH_PATCH_INSTALLED = False


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    script_dir = Path(__file__).resolve().parent
    workspace_dir = script_dir.parent
    default_dataset_dir = find_default_dataset_dir(workspace_dir)

    parser = argparse.ArgumentParser(
        description="Run a small imgsz/batch resource search for smoker-detection training."
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
        help="Runtime detection-only dataset directory.",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=script_dir / "runs",
        help="Directory containing search outputs.",
    )
    parser.add_argument(
        "--project",
        type=Path,
        default=None,
        help="Optional explicit search project directory. Default: smoke_project/runs",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Optional search run directory name. Default: search_<timestamp>",
    )
    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help="Model spec or weights path used for every search run.",
    )
    parser.add_argument(
        "--label-prefix",
        default="search",
        help="Prefix used in each child run name.",
    )
    parser.add_argument(
        "--imgsz-values",
        type=int,
        nargs="+",
        default=[640, 768, 896],
        help="Candidate image sizes.",
    )
    parser.add_argument(
        "--batch-values",
        type=int,
        nargs="+",
        default=[16, 24, 32, 48, 64],
        help="Candidate batch sizes.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Epochs for each short experiment. Recommended: 3 to 5.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=16,
        help="Dataloader worker count. Kept relatively stable during the search.",
    )
    parser.add_argument(
        "--device",
        default="0",
        help="Ultralytics device string. Default: 0",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible comparisons.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Early stopping patience. Should stay above the short search epoch count.",
    )
    parser.add_argument(
        "--fraction",
        type=float,
        default=1.0,
        help="Fraction of the training dataset to use for each short experiment. Default: 1.0",
    )
    parser.add_argument(
        "--skip-final-val",
        action="store_true",
        help="Skip the extra post-training validation pass and reuse the last epoch metrics from results.csv.",
    )
    parser.add_argument(
        "--force-rebuild-runtime-data",
        action="store_true",
        help="Rebuild the runtime detection dataset before running the search.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume interrupted training or skip combos already marked successful in search_summary.json.",
    )
    parser.add_argument(
        "--formal-verify",
        action="store_true",
        help="Run a fixed-config formal verification stage and emit summary.md/result.csv outputs.",
    )
    parser.add_argument(
        "--low-prefetch",
        action="store_true",
        help="Lower Ultralytics dataloader prefetch_factor to reduce host-memory pressure without changing imgsz/batch/workers.",
    )
    return parser.parse_args()


def resolve_project_dir(args: argparse.Namespace) -> Path:
    """Resolve the top-level search output directory."""
    base_project = args.project.resolve() if args.project else args.runs_dir.resolve()
    default_prefix = "formal_verify" if args.formal_verify else "search"
    run_name = args.name or datetime.now().strftime(f"{default_prefix}_%Y%m%d_%H%M%S")
    return base_project / run_name


def install_low_prefetch_patch() -> None:
    """Monkey-patch Ultralytics dataloaders to use a lighter prefetch factor."""
    global _LOW_PREFETCH_PATCH_INSTALLED
    if _LOW_PREFETCH_PATCH_INSTALLED:
        return

    import torch
    from torch.utils.data import distributed
    import ultralytics.data.build as build_mod
    import ultralytics.models.yolo.detect.train as detect_train_mod
    import ultralytics.models.yolo.detect.val as detect_val_mod
    from ultralytics.data.build import ContiguousDistributedSampler
    from ultralytics.data.build import InfiniteDataLoader
    from ultralytics.data.build import RANK
    from ultralytics.data.build import seed_worker

    def build_dataloader_low_prefetch(
        dataset,
        batch: int,
        workers: int,
        shuffle: bool = True,
        rank: int = -1,
        drop_last: bool = False,
        pin_memory: bool = True,
    ) -> InfiniteDataLoader:
        batch = min(batch, len(dataset))
        nd = torch.cuda.device_count()
        nw = min(os.cpu_count() // max(nd, 1), workers)
        sampler = (
            None
            if rank == -1
            else distributed.DistributedSampler(dataset, shuffle=shuffle)
            if shuffle
            else ContiguousDistributedSampler(dataset)
        )
        generator = torch.Generator()
        generator.manual_seed(6148914691236517205 + RANK)
        return InfiniteDataLoader(
            dataset=dataset,
            batch_size=batch,
            shuffle=shuffle and sampler is None,
            num_workers=nw,
            sampler=sampler,
            prefetch_factor=1 if nw > 0 else None,
            pin_memory=nd > 0 and pin_memory,
            collate_fn=getattr(dataset, "collate_fn", None),
            worker_init_fn=seed_worker,
            generator=generator,
            drop_last=drop_last and len(dataset) % batch != 0,
        )

    build_mod.build_dataloader = build_dataloader_low_prefetch
    detect_train_mod.build_dataloader = build_dataloader_low_prefetch
    detect_val_mod.build_dataloader = build_dataloader_low_prefetch
    _LOW_PREFETCH_PATCH_INSTALLED = True


def format_seconds(seconds: float | None) -> str:
    """Format seconds into a short human-readable string."""
    if seconds is None:
        return "-"
    total_seconds = max(seconds, 0.0)
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    secs = total_seconds - hours * 3600 - minutes * 60
    if hours:
        return f"{hours}h {minutes}m {secs:.1f}s"
    if minutes:
        return f"{minutes}m {secs:.1f}s"
    return f"{secs:.1f}s"


def extract_epoch_time_stats(results_csv: Path) -> EpochTimingSummary | None:
    """Read results.csv and estimate per-epoch timings from cumulative time values."""
    if not results_csv.exists():
        return None

    try:
        rows = list(csv.DictReader(results_csv.open("r", encoding="utf-8")))
    except OSError:
        return None

    cumulative_times: list[float] = []
    for row in rows:
        raw_value = row.get("time")
        if raw_value in (None, ""):
            continue
        try:
            cumulative_times.append(float(raw_value))
        except ValueError:
            continue

    if not cumulative_times:
        return None

    epoch_times: list[float] = []
    previous = 0.0
    for current in cumulative_times:
        epoch_times.append(max(current - previous, 0.0))
        previous = current

    stable_times = epoch_times[1:] if len(epoch_times) >= 2 else epoch_times
    avg_epoch_time_sec = sum(stable_times) / max(len(stable_times), 1)
    return EpochTimingSummary(
        epoch_count=len(epoch_times),
        total_time_sec=cumulative_times[-1],
        first_epoch_time_sec=epoch_times[0],
        avg_epoch_time_sec=avg_epoch_time_sec,
    )


def load_prepare_stats(runtime_dataset_dir: Path) -> PrepareStats:
    """Load preparation stats from an existing runtime dataset when available."""
    summary_path = runtime_dataset_dir / "prepare_summary.json"
    if not summary_path.exists():
        return PrepareStats()

    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return PrepareStats()

    stats_payload = payload.get("stats")
    if not isinstance(stats_payload, dict):
        return PrepareStats()

    defaults = asdict(PrepareStats())
    normalized_stats = {key: stats_payload.get(key, value) for key, value in defaults.items()}
    return PrepareStats(**normalized_stats)


def resolve_runtime_dataset(
    dataset_dir: Path,
    runtime_dataset_dir: Path,
    force_rebuild: bool,
) -> tuple[Path, PrepareStats]:
    """Reuse the existing runtime dataset when possible, otherwise rebuild it."""
    runtime_yaml = runtime_dataset_dir / "data_runtime.yaml"
    if runtime_yaml.exists() and not force_rebuild:
        print_status("INFO", f"Reusing existing runtime dataset: {runtime_dataset_dir}")
        return runtime_yaml.resolve(), load_prepare_stats(runtime_dataset_dir)

    print_status("INFO", f"Preparing runtime dataset under: {runtime_dataset_dir}")
    return prepare_runtime_dataset(
        dataset_dir=dataset_dir,
        runtime_dataset_dir=runtime_dataset_dir,
        force_rebuild=force_rebuild,
    )


def evaluate_best_weights(
    best_weights: Path,
    runtime_yaml: Path,
    result_dir: Path,
    run_name: str,
    imgsz: int,
    batch: int,
    args: argparse.Namespace,
) -> tuple[str | None, dict[str, float]]:
    """Validate one trained checkpoint and return compact metrics."""
    YOLO = import_ultralytics_yolo()
    model = YOLO(str(best_weights))
    validation_dir = result_dir / "validation" / run_name
    val_kwargs: dict[str, Any] = {
        "data": str(runtime_yaml),
        "split": "val",
        "imgsz": imgsz,
        "batch": batch,
        "project": str(result_dir / "validation"),
        "name": run_name,
        "exist_ok": True,
    }
    if args.device:
        val_kwargs["device"] = args.device
    val_result = model.val(**val_kwargs)
    return (
        str(validation_dir.resolve()) if validation_dir.exists() else None,
        extract_metrics(val_result),
    )


def format_run_name(label_prefix: str, imgsz: int, batch: int) -> str:
    """Build a readable run name for one search point."""
    return f"{label_prefix}_imgsz{imgsz}_batch{batch}"


def extract_metrics_from_results_csv(results_csv: Path) -> dict[str, float]:
    """Read the last row of Ultralytics results.csv and extract key metrics."""
    if not results_csv.exists():
        return {}

    try:
        rows = list(csv.DictReader(results_csv.open("r", encoding="utf-8")))
    except OSError:
        return {}

    if not rows:
        return {}

    last_row = rows[-1]
    metric_mapping = {
        "metrics/precision(B)": "precision",
        "metrics/recall(B)": "recall",
        "metrics/mAP50(B)": "map50",
        "metrics/mAP50-95(B)": "map50_95",
    }
    metrics: dict[str, float] = {}
    for source_key, target_key in metric_mapping.items():
        raw_value = last_row.get(source_key)
        if raw_value in (None, ""):
            continue
        try:
            metrics[target_key] = float(raw_value)
        except ValueError:
            continue
    return metrics


def assess_long_train_readiness(
    result: SearchResult,
    total_memory_mb: int | None,
) -> tuple[str, str]:
    """Return a concise recommendation on whether to move to the full long run."""
    blockers: list[str] = []
    signals: list[str] = []

    if result.status != "success":
        blockers.append("训练未成功完成")
    else:
        signals.append("训练和自动验证已完成")

    if result.precision is None or result.recall is None or result.map50 is None or result.map50_95 is None:
        blockers.append("验证指标不完整")
    else:
        if result.precision >= 0.65:
            signals.append(f"precision={result.precision:.4f}")
        else:
            blockers.append(f"precision 偏低 ({result.precision:.4f})")
        if result.recall >= 0.50:
            signals.append(f"recall={result.recall:.4f}")
        else:
            blockers.append(f"recall 偏低 ({result.recall:.4f})")
        if result.map50 >= 0.60:
            signals.append(f"mAP50={result.map50:.4f}")
        else:
            blockers.append(f"mAP50 偏低 ({result.map50:.4f})")
        if result.map50_95 >= 0.35:
            signals.append(f"mAP50-95={result.map50_95:.4f}")
        else:
            blockers.append(f"mAP50-95 偏低 ({result.map50_95:.4f})")

    if result.avg_epoch_time_sec is not None:
        signals.append(f"平均单 epoch 时间 {format_seconds(result.avg_epoch_time_sec)}")

    if total_memory_mb is not None and result.max_gpu_mem_mb is not None:
        usage_ratio = result.max_gpu_mem_mb / max(total_memory_mb, 1)
        if usage_ratio <= 0.90:
            signals.append(f"峰值显存占用 {result.max_gpu_mem_mb}MB ({usage_ratio:.1%})")
        else:
            blockers.append(f"峰值显存占用过高 {result.max_gpu_mem_mb}MB ({usage_ratio:.1%})")
    elif result.max_gpu_mem_mb is not None:
        signals.append(f"峰值显存占用 {result.max_gpu_mem_mb}MB")

    if blockers:
        return "建议暂不直接进入正式长训", "；".join(blockers)
    return "建议直接进入正式长训", "；".join(signals)


def format_metric(value: float | None) -> str:
    """Format a float metric for compact reports."""
    return "-" if value is None else f"{value:.4f}"


def write_markdown_summary(
    markdown_path: Path,
    search_dir: Path,
    dataset_dir: Path,
    runtime_yaml: Path,
    args: argparse.Namespace,
    results: list[SearchResult],
    total_memory_mb: int | None,
) -> None:
    """Write a human-readable markdown summary."""
    lines = [
        "# Summary",
        "",
        "## Run",
        f"- stage: {'formal verify' if args.formal_verify else 'search'}",
        f"- search_dir: `{search_dir.resolve()}`",
        f"- dataset_dir: `{dataset_dir}`",
        f"- runtime_yaml: `{runtime_yaml.resolve()}`",
        f"- model: `{args.model}`",
        f"- imgsz_values: `{args.imgsz_values}`",
        f"- batch_values: `{args.batch_values}`",
        f"- epochs: `{args.epochs}`",
        f"- workers: `{args.workers}`",
        f"- device: `{args.device}`",
        f"- resume: `{args.resume}`",
        f"- low_prefetch: `{args.low_prefetch}`",
        "",
        "## Result",
    ]

    if not results:
        lines.extend(["- no result yet", ""])
        markdown_path.write_text("\n".join(lines), encoding="utf-8")
        return

    lines.extend(
        [
            "| imgsz | batch | status | precision | recall | mAP50 | mAP50-95 | epoch time | peak GPU mem | decision |",
            "|---:|---:|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for item in results:
        peak_gpu_text = f"{item.max_gpu_mem_mb}MB" if item.max_gpu_mem_mb is not None else "-"
        lines.append(
            f"| {item.imgsz} | {item.batch} | {item.status} | "
            f"{format_metric(item.precision)} | {format_metric(item.recall)} | "
            f"{format_metric(item.map50)} | {format_metric(item.map50_95)} | "
            f"{format_seconds(item.avg_epoch_time_sec)} | {peak_gpu_text} | "
            f"{item.recommendation or '-'} |"
        )

    lines.extend(["", "## Decision"])
    if len(results) == 1:
        item = results[0]
        lines.append(f"- decision: {item.recommendation or '-'}")
        lines.append(f"- reason: {item.recommendation_reason or '-'}")
    else:
        for item in results:
            reason_suffix = f" ({item.recommendation_reason})" if item.recommendation_reason else ""
            lines.append(
                f"- imgsz={item.imgsz}, batch={item.batch}: {item.recommendation or '-'}{reason_suffix}"
            )

    lines.extend(["", "## Paths"])
    for item in results:
        lines.append(f"- run_dir[{item.imgsz}/{item.batch}]: `{item.run_dir or '-'}`")
        lines.append(f"- results_csv[{item.imgsz}/{item.batch}]: `{item.results_csv or '-'}`")
        lines.append(f"- validation_dir[{item.imgsz}/{item.batch}]: `{item.validation_dir or '-'}`")

    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def train_one_combo(
    imgsz: int,
    batch: int,
    runtime_yaml: Path,
    search_dir: Path,
    args: argparse.Namespace,
) -> SearchResult:
    """Train one imgsz/batch combination and collect metrics."""
    if args.low_prefetch:
        install_low_prefetch_patch()

    YOLO = import_ultralytics_yolo()
    run_name = format_run_name(args.label_prefix, imgsz, batch)
    run_dir = search_dir / run_name
    best_weights = run_dir / "weights" / "best.pt"
    last_weights = run_dir / "weights" / "last.pt"
    results_csv = run_dir / "results.csv"

    if args.resume and best_weights.exists() and results_csv.exists():
        timing_summary = extract_epoch_time_stats(results_csv)
        if timing_summary is not None and timing_summary.epoch_count >= args.epochs:
            print_status("INFO", f"Found completed on-disk run, reusing results: imgsz={imgsz}, batch={batch}")
            validation_dir: str | None = None
            metrics: dict[str, float] = {}
            if not args.skip_final_val:
                validation_dir, metrics = evaluate_best_weights(
                    best_weights=best_weights,
                    runtime_yaml=runtime_yaml,
                    result_dir=search_dir,
                    run_name=run_name,
                    imgsz=imgsz,
                    batch=batch,
                    args=args,
                )
            if not metrics:
                metrics = extract_metrics_from_results_csv(results_csv)
            recommendation, recommendation_reason = assess_long_train_readiness(
                SearchResult(
                    imgsz=imgsz,
                    batch=batch,
                    epochs=args.epochs,
                    workers=args.workers,
                    status="success",
                    avg_epoch_time_sec=timing_summary.avg_epoch_time_sec,
                    precision=metrics.get("precision"),
                    recall=metrics.get("recall"),
                    map50=metrics.get("map50"),
                    map50_95=metrics.get("map50_95"),
                ),
                None,
            )
            return SearchResult(
                imgsz=imgsz,
                batch=batch,
                epochs=args.epochs,
                workers=args.workers,
                status="success",
                run_dir=str(run_dir.resolve()) if run_dir.exists() else None,
                best_weights=str(best_weights.resolve()),
                results_csv=str(results_csv.resolve()),
                validation_dir=validation_dir,
                total_train_time_sec=timing_summary.total_time_sec,
                first_epoch_time_sec=timing_summary.first_epoch_time_sec,
                avg_epoch_time_sec=timing_summary.avg_epoch_time_sec,
                precision=metrics.get("precision"),
                recall=metrics.get("recall"),
                map50=metrics.get("map50"),
                map50_95=metrics.get("map50_95"),
                resume_used=True,
                recommendation=recommendation,
                recommendation_reason=recommendation_reason,
            )

    monitor = GpuMemoryMonitor(args.device)
    start_time = time.perf_counter()
    resume_used = args.resume and last_weights.exists()
    print_status("INFO", f"Search run started: imgsz={imgsz}, batch={batch}")

    try:
        torch_peak_mem_mb: int | None = None
        torch_module: Any | None = None
        device_index = resolve_primary_device_index(args.device)
        if device_index is not None:
            try:
                import torch as torch_module

                if torch_module.cuda.is_available():
                    torch_module.cuda.empty_cache()
                    torch_module.cuda.reset_peak_memory_stats(device_index)
            except Exception:
                torch_module = None

        if resume_used:
            print_status("INFO", f"[stage] Resuming from checkpoint for imgsz={imgsz}, batch={batch}: {last_weights}")
            model = YOLO(str(last_weights))
            train_kwargs: dict[str, Any] = {"resume": True}
        else:
            print_status("INFO", f"[stage] Loading YOLO model for imgsz={imgsz}, batch={batch}")
            model = YOLO(args.model)
            print_status("INFO", f"[stage] YOLO model loaded for imgsz={imgsz}, batch={batch}")
            train_kwargs = {
                "data": str(runtime_yaml),
                "epochs": args.epochs,
                "imgsz": imgsz,
                "batch": batch,
                "workers": args.workers,
                "project": str(search_dir),
                "name": run_name,
                "exist_ok": True,
                "seed": args.seed,
                "patience": args.patience,
                "fraction": args.fraction,
                "plots": False,
            }
            if args.device:
                train_kwargs["device"] = args.device

        print_status("INFO", f"[stage] Starting GPU monitor for imgsz={imgsz}, batch={batch}")
        monitor.start()
        print_status("INFO", f"[stage] Calling model.train for imgsz={imgsz}, batch={batch}")
        model.train(**train_kwargs)
        print_status("INFO", f"[stage] model.train finished for imgsz={imgsz}, batch={batch}")
        monitor.stop()
        if torch_module is not None and device_index is not None and torch_module.cuda.is_available():
            torch_module.cuda.synchronize(device_index)
            torch_peak_mem_mb = int(torch_module.cuda.max_memory_reserved(device_index) / (1024 ** 2))

        elapsed = time.perf_counter() - start_time
        timing_summary = extract_epoch_time_stats(results_csv)
        avg_epoch_time_sec = timing_summary.avg_epoch_time_sec if timing_summary else elapsed / max(args.epochs, 1)
        first_epoch_time_sec = timing_summary.first_epoch_time_sec if timing_summary else None
        total_train_time_sec = timing_summary.total_time_sec if timing_summary else elapsed
        metrics: dict[str, float] = {}
        validation_dir: str | None = None
        if not args.skip_final_val and best_weights.exists():
            print_status("INFO", f"[stage] Starting validation for imgsz={imgsz}, batch={batch}")
            validation_dir, metrics = evaluate_best_weights(
                best_weights=best_weights,
                runtime_yaml=runtime_yaml,
                result_dir=search_dir,
                run_name=run_name,
                imgsz=imgsz,
                batch=batch,
                args=args,
            )
        if not metrics:
            metrics = extract_metrics_from_results_csv(results_csv)

        max_gpu_candidates = [
            value
            for value in (monitor.max_memory_mb, torch_peak_mem_mb)
            if value is not None and value > 0
        ]
        max_gpu_mem_mb = max(max_gpu_candidates) if max_gpu_candidates else None
        recommendation, recommendation_reason = assess_long_train_readiness(
            SearchResult(
                imgsz=imgsz,
                batch=batch,
                epochs=args.epochs,
                workers=args.workers,
                status="success",
                max_gpu_mem_mb=max_gpu_mem_mb,
                avg_epoch_time_sec=avg_epoch_time_sec,
                precision=metrics.get("precision"),
                recall=metrics.get("recall"),
                map50=metrics.get("map50"),
                map50_95=metrics.get("map50_95"),
            ),
            monitor.total_memory_mb,
        )

        result = SearchResult(
            imgsz=imgsz,
            batch=batch,
            epochs=args.epochs,
            workers=args.workers,
            status="success",
            run_dir=str(run_dir.resolve()) if run_dir.exists() else None,
            best_weights=str(best_weights.resolve()) if best_weights.exists() else None,
            results_csv=str(results_csv.resolve()) if results_csv.exists() else None,
            validation_dir=validation_dir,
            max_gpu_mem_mb=max_gpu_mem_mb,
            total_train_time_sec=total_train_time_sec,
            first_epoch_time_sec=first_epoch_time_sec,
            avg_epoch_time_sec=avg_epoch_time_sec,
            precision=metrics.get("precision"),
            recall=metrics.get("recall"),
            map50=metrics.get("map50"),
            map50_95=metrics.get("map50_95"),
            resume_used=resume_used,
            recommendation=recommendation,
            recommendation_reason=recommendation_reason,
        )
        metric_preview = ", ".join(
            f"{key}={value:.4f}"
            for key, value in {
                "precision": result.precision,
                "recall": result.recall,
                "map50": result.map50,
                "map50_95": result.map50_95,
            }.items()
            if value is not None
        )
        print_status(
            "OK",
            f"Search run finished: imgsz={imgsz}, batch={batch}, "
            f"gpu_mem={result.max_gpu_mem_mb}MB, epoch_time={avg_epoch_time_sec:.2f}s, "
            f"{metric_preview}, decision={result.recommendation}",
        )
        return result
    except Exception as exc:
        monitor.stop()
        print_status("ERROR", f"Search run failed for imgsz={imgsz}, batch={batch}: {exc}")
        return SearchResult(
            imgsz=imgsz,
            batch=batch,
            epochs=args.epochs,
            workers=args.workers,
            status="failed",
            run_dir=str(run_dir.resolve()) if run_dir.exists() else None,
            results_csv=str(results_csv.resolve()) if results_csv.exists() else None,
            max_gpu_mem_mb=monitor.max_memory_mb or None,
            resume_used=resume_used,
            recommendation="建议暂不直接进入正式长训",
            recommendation_reason="训练阶段异常退出",
            error=str(exc),
        )


def choose_recommendations(
    results: list[SearchResult],
    total_memory_mb: int | None,
) -> dict[str, dict[str, Any] | None]:
    """Recommend quality-first and balance-first formal-training settings."""
    successful = [item for item in results if item.status == "success" and item.map50_95 is not None]
    if not successful:
        return {"best_quality": None, "best_balance": None}

    safe_candidates = successful
    if total_memory_mb is not None:
        safe_candidates = [
            item
            for item in successful
            if item.max_gpu_mem_mb is None or item.max_gpu_mem_mb <= int(total_memory_mb * 0.9)
        ] or successful

    best_quality = max(
        safe_candidates,
        key=lambda item: (
            item.map50_95 or -1.0,
            item.map50 or -1.0,
            item.recall or -1.0,
            -(item.avg_epoch_time_sec or float("inf")),
        ),
    )

    best_balance = max(
        safe_candidates,
        key=lambda item: (
            round(item.map50_95 or -1.0, 4),
            -(item.avg_epoch_time_sec or float("inf")),
            -(item.max_gpu_mem_mb or 0),
        ),
    )

    def to_payload(item: SearchResult) -> dict[str, Any]:
        return {
            "imgsz": item.imgsz,
            "batch": item.batch,
            "workers": item.workers,
            "precision": item.precision,
            "recall": item.recall,
            "map50": item.map50,
            "map50_95": item.map50_95,
            "avg_epoch_time_sec": item.avg_epoch_time_sec,
            "max_gpu_mem_mb": item.max_gpu_mem_mb,
            "recommended_formal_args": {
                "imgsz": item.imgsz,
                "batch": item.batch,
                "workers": item.workers,
            },
        }

    return {
        "best_quality": to_payload(best_quality),
        "best_balance": to_payload(best_balance),
    }


def write_csv(results: list[SearchResult], csv_path: Path) -> None:
    """Write a compact CSV report."""
    fieldnames = [
        "imgsz",
        "batch",
        "epochs",
        "workers",
        "status",
        "resume_used",
        "max_gpu_mem_mb",
        "total_train_time_sec",
        "first_epoch_time_sec",
        "avg_epoch_time_sec",
        "precision",
        "recall",
        "map50",
        "map50_95",
        "run_dir",
        "best_weights",
        "results_csv",
        "validation_dir",
        "recommendation",
        "recommendation_reason",
        "error",
    ]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))


def load_existing_results(summary_path: Path) -> list[SearchResult]:
    """Load previously saved search results for resume mode."""
    if not summary_path.exists():
        return []

    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []

    items = payload.get("results")
    if not isinstance(items, list):
        return []

    loaded_results: list[SearchResult] = []
    valid_fields = SearchResult.__dataclass_fields__.keys()
    for item in items:
        if not isinstance(item, dict):
            continue
        normalized_item = {key: item.get(key) for key in valid_fields}
        try:
            loaded_results.append(SearchResult(**normalized_item))
        except TypeError:
            continue
    return loaded_results


def save_search_state(
    search_dir: Path,
    dataset_dir: Path,
    runtime_yaml: Path,
    prepare_stats: PrepareStats,
    args: argparse.Namespace,
    results: list[SearchResult],
    total_memory_mb: int | None,
) -> tuple[Path, Path, Path, Path, dict[str, dict[str, Any] | None]]:
    """Persist the current search state so long searches can be resumed safely."""
    recommendations = choose_recommendations(results, total_memory_mb)
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "search_dir": str(search_dir.resolve()),
        "dataset_dir": str(dataset_dir),
        "runtime_yaml": str(runtime_yaml.resolve()),
        "prepare_stats": asdict(prepare_stats),
        "search_args": {
            "model": args.model,
            "imgsz_values": args.imgsz_values,
            "batch_values": args.batch_values,
            "epochs": args.epochs,
            "workers": args.workers,
            "device": args.device,
            "seed": args.seed,
            "patience": args.patience,
            "fraction": args.fraction,
            "skip_final_val": args.skip_final_val,
            "resume": args.resume,
            "formal_verify": args.formal_verify,
            "low_prefetch": args.low_prefetch,
        },
        "results": [asdict(result) for result in results],
        "recommendations": recommendations,
    }

    summary_path = search_dir / "search_summary.json"
    csv_path = search_dir / "search_summary.csv"
    result_csv_path = search_dir / "result.csv"
    markdown_path = search_dir / "summary.md"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(results, csv_path)
    write_csv(results, result_csv_path)
    write_markdown_summary(
        markdown_path=markdown_path,
        search_dir=search_dir,
        dataset_dir=dataset_dir,
        runtime_yaml=runtime_yaml,
        args=args,
        results=results,
        total_memory_mb=total_memory_mb,
    )
    return summary_path, csv_path, result_csv_path, markdown_path, recommendations


def main() -> int:
    """Run the imgsz/batch resource search."""
    args = parse_args()
    if args.formal_verify and args.label_prefix == "search":
        args.label_prefix = "formalverify"
    if args.low_prefetch:
        install_low_prefetch_patch()
    dataset_dir = args.dataset_dir.resolve()
    runtime_dataset_dir = args.runtime_dataset_dir.resolve()
    search_dir = resolve_project_dir(args)
    ensure_dir(search_dir)

    print_status("INFO", f"Source dataset: {dataset_dir}")
    runtime_yaml, prepare_stats = resolve_runtime_dataset(
        dataset_dir=dataset_dir,
        runtime_dataset_dir=runtime_dataset_dir,
        force_rebuild=args.force_rebuild_runtime_data,
    )

    combos = [(imgsz, batch) for imgsz in args.imgsz_values for batch in args.batch_values]
    if args.formal_verify and len(combos) != 1:
        print_status("ERROR", "Formal verify stage expects exactly one imgsz/batch combination.")
        return 2
    print_status("INFO", f"Resource search started with {len(combos)} combinations.")

    total_memory_mb = GpuMemoryMonitor(args.device)._query_memory("memory.total")
    summary_path = search_dir / "search_summary.json"
    loaded_results = load_existing_results(summary_path) if args.resume else []
    if loaded_results:
        success_count = sum(result.status == "success" for result in loaded_results)
        print_status(
            "INFO",
            f"Resume mode loaded {len(loaded_results)} saved combinations "
            f"({success_count} successful, {len(loaded_results) - success_count} unfinished).",
        )

    results_by_combo: dict[tuple[int, int], SearchResult] = {
        (result.imgsz, result.batch): result for result in loaded_results
    }
    for imgsz, batch in combos:
        combo_key = (imgsz, batch)
        existing_result = results_by_combo.get(combo_key)
        if existing_result is not None and existing_result.status == "success":
            existing_run_dir = Path(existing_result.run_dir) if existing_result.run_dir else None
            if existing_run_dir is None or existing_run_dir.exists():
                print_status("INFO", f"Skipping successful combo from summary: imgsz={imgsz}, batch={batch}")
                continue
        if existing_result is not None and existing_result.status != "success":
            print_status("INFO", f"Retrying unfinished combo: imgsz={imgsz}, batch={batch}")
        result = train_one_combo(imgsz, batch, runtime_yaml, search_dir, args)
        results_by_combo[combo_key] = result
        ordered_results = [results_by_combo[key] for key in combos if key in results_by_combo]
        summary_path, csv_path, result_csv_path, markdown_path, _ = save_search_state(
            search_dir=search_dir,
            dataset_dir=dataset_dir,
            runtime_yaml=runtime_yaml,
            prepare_stats=prepare_stats,
            args=args,
            results=ordered_results,
            total_memory_mb=total_memory_mb,
        )

    results = [results_by_combo[key] for key in combos if key in results_by_combo]
    summary_path, csv_path, result_csv_path, markdown_path, recommendations = save_search_state(
        search_dir=search_dir,
        dataset_dir=dataset_dir,
        runtime_yaml=runtime_yaml,
        prepare_stats=prepare_stats,
        args=args,
        results=results,
        total_memory_mb=total_memory_mb,
    )

    print_status("INFO", f"Search summary saved to: {summary_path}")
    print_status("INFO", f"Search CSV saved to: {csv_path}")
    print_status("INFO", f"Result CSV saved to: {result_csv_path}")
    print_status("INFO", f"Summary markdown saved to: {markdown_path}")
    if recommendations["best_quality"] is not None:
        print_status("OK", f"Best quality recommendation: {recommendations['best_quality']}")
    if recommendations["best_balance"] is not None:
        print_status("OK", f"Best balance recommendation: {recommendations['best_balance']}")

    success_count = sum(result.status == "success" for result in results)
    if success_count == 0:
        print_status("ERROR", "No successful search runs were completed.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
