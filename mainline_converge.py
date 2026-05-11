"""Resume the selected mainline model and generate thesis-ready reports.

This script keeps the existing 30-epoch model-selection evidence intact:
1. Preserve the current YOLOv8s run as the baseline result.
2. Preserve the current 30-epoch YOLO26s result as the selection evidence.
3. Resume YOLO26s only, continuing training to the target epoch count.
4. Run formal validation/test evaluation automatically after long training.
5. Write markdown/csv/json reports suitable for thesis writing.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import signal
import shutil
import subprocess
from dataclasses import asdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from dataset_config import find_default_dataset_dir
from search_train_config import EpochTimingSummary
from search_train_config import extract_epoch_time_stats
from search_train_config import format_seconds
from search_train_config import resolve_runtime_dataset
from train_compare import PrepareStats
from train_compare import ensure_dir
from train_compare import extract_metrics
from train_compare import import_ultralytics_yolo
from train_compare import print_status


@dataclass
class RunSnapshot:
    """Compact snapshot of one training run directory."""

    label: str
    run_dir: str
    results_csv: str
    best_weights: str
    last_weights: str
    imgsz: int | None
    batch: int | None
    workers: int | None
    device: str | None
    epoch_count: int
    selection_epoch: int
    selection_precision: float | None
    selection_recall: float | None
    selection_map50: float | None
    selection_map50_95: float | None
    latest_precision: float | None
    latest_recall: float | None
    latest_map50: float | None
    latest_map50_95: float | None
    total_train_time_sec: float | None
    avg_epoch_time_sec: float | None


@dataclass
class EvalSnapshot:
    """One formal evaluation result."""

    split: str
    label: str
    weights: str
    save_dir: str | None
    precision: float | None
    recall: float | None
    map50: float | None
    map50_95: float | None


@dataclass
class ResumeCheckpoint:
    """One candidate checkpoint for resumable long training."""

    path: str
    epoch_index: int | None
    human_epoch: int | None
    resumable: bool
    reason: str
    optimizer_name: str | None


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    script_dir = Path(__file__).resolve().parent
    workspace_dir = script_dir.parent
    default_dataset_dir = find_default_dataset_dir(workspace_dir)

    parser = argparse.ArgumentParser(
        description=(
            "Resume YOLO26s as the selected mainline model, then run formal "
            "evaluation and export thesis-ready summaries."
        )
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
        help="Directory containing training/evaluation/report outputs.",
    )
    parser.add_argument(
        "--baseline-run-dir",
        type=Path,
        default=script_dir / "runs" / "s_high_mem_formal_20260424" / "yolov8s_formal_e30_img896_b48_w16",
        help="Existing YOLOv8s 30-epoch baseline run directory.",
    )
    parser.add_argument(
        "--mainline-run-dir",
        type=Path,
        default=script_dir / "runs" / "s_high_mem_formal_20260424" / "yolo26s_formal_e30_img896_b48_w16",
        help="Existing YOLO26s run directory to resume.",
    )
    parser.add_argument(
        "--baseline-label",
        default="YOLOv8s",
        help="Display label for the retained baseline model.",
    )
    parser.add_argument(
        "--mainline-label",
        default="YOLO26s",
        help="Display label for the selected mainline model.",
    )
    parser.add_argument(
        "--selection-epoch",
        type=int,
        default=30,
        help="Epoch used as the stage-wise model-selection comparison point.",
    )
    parser.add_argument(
        "--target-epochs",
        type=int,
        default=100,
        help="Target total epoch count for the resumed mainline run.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Optional device override for resume/evaluation, e.g. 0 or cpu.",
    )
    parser.add_argument(
        "--eval-imgsz",
        type=int,
        default=None,
        help="Optional evaluation image size. Default: reuse the mainline training imgsz.",
    )
    parser.add_argument(
        "--eval-batch",
        type=int,
        default=None,
        help="Optional evaluation batch size. Default: reuse the mainline training batch.",
    )
    parser.add_argument(
        "--force-rebuild-runtime-data",
        action="store_true",
        help="Rebuild the runtime dataset before resume/evaluation.",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip the resume step and only regenerate evaluation/report outputs.",
    )
    parser.add_argument(
        "--report-name",
        default=None,
        help="Optional report directory name under runs-dir. Default: mainline_converge_<timestamp>.",
    )
    return parser.parse_args()


def load_simple_yaml(path: Path) -> dict[str, Any]:
    """Load one small YAML file with a PyYAML-first strategy."""
    try:
        import yaml  # type: ignore

        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except ModuleNotFoundError:
        print_status("WARN", "PyYAML is unavailable; falling back to a minimal YAML reader.")
    except Exception as exc:
        print_status("WARN", f"Failed to parse YAML with PyYAML ({path}): {exc}")

    payload: dict[str, Any] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        payload[key.strip()] = value.strip().strip("'").strip('"')
    return payload


def as_int(value: Any) -> int | None:
    """Convert a scalar into int when possible."""
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def as_float(value: Any) -> float | None:
    """Convert a scalar into float when possible."""
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def format_metric(value: float | None) -> str:
    """Format one metric for markdown/csv output."""
    return "-" if value is None else f"{value:.4f}"


def slugify_label(text: str) -> str:
    """Convert a label into a filesystem-friendly slug."""
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip().lower()).strip("_")
    return slug or "run"


def cleanup_stale_mainline_processes(script_path: Path) -> int:
    """Kill orphaned processes from previous failed mainline runs."""
    try:
        output = subprocess.check_output(["ps", "-eo", "pid=,ppid=,args="], text=True)
    except Exception:
        return 0

    killed = 0
    current_pid = os.getpid()
    script_text = str(script_path.resolve())
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split(None, 2)
        if len(parts) < 3:
            continue
        pid = int(parts[0])
        ppid = int(parts[1])
        cmd = parts[2]
        if pid == current_pid:
            continue
        if ppid == 1 and script_text in cmd:
            try:
                os.kill(pid, signal.SIGKILL)
                killed += 1
            except ProcessLookupError:
                continue
    return killed


def choose_safe_workers(requested_workers: int | None) -> int | None:
    """Cap worker count conservatively for the current host."""
    if requested_workers is None:
        return None
    if requested_workers <= 0:
        return requested_workers

    cpu_total = os.cpu_count() or 4
    safe_cap = 4 if cpu_total >= 4 else max(1, cpu_total)
    return min(requested_workers, safe_cap)


def build_batch_retry_sequence(requested_batch: int | None) -> list[int | None]:
    """Build a conservative retry sequence for CUDA resource failures."""
    if requested_batch is None:
        return [None]

    candidates = [
        requested_batch,
        min(requested_batch, 40),
        min(requested_batch, 32),
        min(requested_batch, 24),
        min(requested_batch, 16),
        min(requested_batch, 12),
        min(requested_batch, 8),
    ]
    ordered: list[int | None] = []
    for item in candidates:
        if item is None or item <= 0:
            continue
        if item not in ordered:
            ordered.append(item)
    return ordered or [requested_batch]


def build_worker_retry_sequence(requested_workers: int | None) -> list[int | None]:
    """Build a conservative retry sequence for host-side dataloader pressure."""
    if requested_workers is None:
        return [None]
    if requested_workers <= 0:
        return [requested_workers]

    candidates = [requested_workers, min(requested_workers, 2), 0]
    ordered: list[int | None] = []
    for item in candidates:
        if item not in ordered:
            ordered.append(item)
    return ordered


def build_resource_attempts(batch: int | None, workers: int | None) -> list[tuple[int | None, int | None]]:
    """Build retry attempts from aggressive to conservative resource settings."""
    batch_candidates = build_batch_retry_sequence(batch)
    worker_candidates = build_worker_retry_sequence(workers)

    attempts: list[tuple[int | None, int | None]] = []
    for batch_value in batch_candidates:
        if batch_value is not None and batch is not None and batch_value >= max(batch, 1) // 2:
            attempt_workers = [worker_candidates[0]]
        else:
            attempt_workers = worker_candidates
        for worker_value in attempt_workers:
            pair = (batch_value, worker_value)
            if pair not in attempts:
                attempts.append(pair)
    return attempts


def is_cuda_resource_error(exc: Exception) -> bool:
    """Return whether an exception looks like a CUDA resource/capacity failure."""
    message = str(exc).lower()
    patterns = [
        "cuda out of memory",
        "cublas_status_alloc_failed",
        "cudnn_status_alloc_failed",
        "cuda error",
        "out of memory",
    ]
    return any(pattern in message for pattern in patterns)


def cleanup_partial_run(output_run_dir: Path, patched_checkpoint: Path) -> None:
    """Remove generated artifacts from a failed attempt before retrying."""
    if output_run_dir.exists():
        shutil.rmtree(output_run_dir, ignore_errors=True)
    if patched_checkpoint.exists():
        try:
            patched_checkpoint.unlink()
        except OSError:
            pass


def read_results_rows(results_csv: Path) -> list[dict[str, str]]:
    """Read Ultralytics results.csv."""
    try:
        with results_csv.open("r", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    except OSError as exc:
        raise FileNotFoundError(f"Failed to read results.csv: {results_csv}") from exc


def find_row_for_epoch(rows: list[dict[str, str]], epoch: int) -> dict[str, str] | None:
    """Find the row recorded for a specific 1-based epoch."""
    for row in rows:
        row_epoch = as_int(row.get("epoch"))
        if row_epoch == epoch:
            return row
    if 0 < epoch <= len(rows):
        return rows[epoch - 1]
    return None


def metrics_from_row(row: dict[str, str] | None) -> dict[str, float | None]:
    """Extract compact metrics from one CSV row."""
    if row is None:
        return {
            "precision": None,
            "recall": None,
            "map50": None,
            "map50_95": None,
        }
    return {
        "precision": as_float(row.get("metrics/precision(B)")),
        "recall": as_float(row.get("metrics/recall(B)")),
        "map50": as_float(row.get("metrics/mAP50(B)")),
        "map50_95": as_float(row.get("metrics/mAP50-95(B)")),
    }


def load_run_snapshot(run_dir: Path, label: str, selection_epoch: int) -> RunSnapshot:
    """Collect one run directory snapshot."""
    args_path = run_dir / "args.yaml"
    results_csv = run_dir / "results.csv"
    best_weights = run_dir / "weights" / "best.pt"
    last_weights = run_dir / "weights" / "last.pt"

    missing_paths = [
        path
        for path in (args_path, results_csv, best_weights, last_weights)
        if not path.exists()
    ]
    if missing_paths:
        missing_text = "\n".join(str(path) for path in missing_paths)
        raise FileNotFoundError(f"Run directory is incomplete:\n{missing_text}")

    args_payload = load_simple_yaml(args_path)
    rows = read_results_rows(results_csv)
    latest_row = rows[-1] if rows else None
    selection_row = find_row_for_epoch(rows, selection_epoch)
    latest_metrics = metrics_from_row(latest_row)
    selection_metrics = metrics_from_row(selection_row)
    timing: EpochTimingSummary | None = extract_epoch_time_stats(results_csv)

    epoch_count = timing.epoch_count if timing is not None else len(rows)

    return RunSnapshot(
        label=label,
        run_dir=str(run_dir.resolve()),
        results_csv=str(results_csv.resolve()),
        best_weights=str(best_weights.resolve()),
        last_weights=str(last_weights.resolve()),
        imgsz=as_int(args_payload.get("imgsz")),
        batch=as_int(args_payload.get("batch")),
        workers=as_int(args_payload.get("workers")),
        device=str(args_payload.get("device")) if args_payload.get("device") is not None else None,
        epoch_count=epoch_count,
        selection_epoch=selection_epoch,
        selection_precision=selection_metrics["precision"],
        selection_recall=selection_metrics["recall"],
        selection_map50=selection_metrics["map50"],
        selection_map50_95=selection_metrics["map50_95"],
        latest_precision=latest_metrics["precision"],
        latest_recall=latest_metrics["recall"],
        latest_map50=latest_metrics["map50"],
        latest_map50_95=latest_metrics["map50_95"],
        total_train_time_sec=timing.total_time_sec if timing is not None else None,
        avg_epoch_time_sec=timing.avg_epoch_time_sec if timing is not None else None,
    )


def inspect_resume_checkpoint(ckpt_path: Path) -> ResumeCheckpoint:
    """Inspect whether a checkpoint can be used for true Ultralytics resume."""
    import torch

    ckpt = torch.load(ckpt_path, map_location="cpu")
    epoch_index = as_int(ckpt.get("epoch"))
    optimizer_state = ckpt.get("optimizer")
    ema_state = ckpt.get("ema")
    resumable = (
        epoch_index is not None
        and epoch_index >= 0
        and optimizer_state is not None
        and ema_state is not None
    )
    optimizer_name = infer_optimizer_name(optimizer_state)
    reason = (
        "checkpoint contains epoch/optimizer/ema state"
        if resumable
        else "checkpoint was stripped or does not contain full resume state"
    )
    return ResumeCheckpoint(
        path=str(ckpt_path.resolve()),
        epoch_index=epoch_index,
        human_epoch=(epoch_index + 1) if epoch_index is not None and epoch_index >= 0 else None,
        resumable=resumable,
        reason=reason,
        optimizer_name=optimizer_name,
    )


def infer_optimizer_name(optimizer_state: Any) -> str | None:
    """Infer a stable optimizer name from a serialized optimizer state dict."""
    if not isinstance(optimizer_state, dict):
        return None

    param_groups = optimizer_state.get("param_groups")
    if not isinstance(param_groups, list) or not param_groups:
        return None

    sample_group = param_groups[0]
    if not isinstance(sample_group, dict):
        return None

    if "betas" in sample_group:
        return "AdamW"
    if "momentum" in sample_group:
        return "SGD"
    return None


def choose_resume_checkpoint(run_dir: Path) -> ResumeCheckpoint:
    """Choose the best available resumable checkpoint from one run directory."""
    weights_dir = run_dir / "weights"
    candidates: list[Path] = []

    preferred = [weights_dir / "last.pt", weights_dir / "best.pt"]
    candidates.extend(path for path in preferred if path.exists())

    epoch_paths = sorted(
        weights_dir.glob("epoch*.pt"),
        key=lambda path: as_int(re.search(r"epoch(\d+)\.pt$", path.name).group(1)) if re.search(r"epoch(\d+)\.pt$", path.name) else -1,
        reverse=True,
    )
    candidates.extend(path for path in epoch_paths if path not in candidates)

    inspected: list[ResumeCheckpoint] = []
    for candidate in candidates:
        info = inspect_resume_checkpoint(candidate)
        inspected.append(info)
        if info.resumable:
            return info

    if not inspected:
        raise FileNotFoundError(f"No checkpoint file was found under: {weights_dir}")

    reason_text = "；".join(f"{Path(item.path).name}: {item.reason}" for item in inspected[:5])
    raise RuntimeError(f"No resumable checkpoint is available. {reason_text}")


def create_resume_checkpoint_copy(
    source_checkpoint: ResumeCheckpoint,
    output_checkpoint: Path,
    output_project_dir: Path,
    output_run_name: str,
    output_run_dir: Path,
    trainer_total_epochs: int,
    device: str | None,
    imgsz: int | None,
    batch: int | None,
    workers: int | None,
) -> Path:
    """Copy and patch a resumable checkpoint so resumed training writes into a new run dir."""
    import torch

    ckpt = torch.load(source_checkpoint.path, map_location="cpu")
    train_args = dict(ckpt.get("train_args") or {})
    train_args["project"] = str(output_project_dir.resolve())
    train_args["name"] = output_run_name
    train_args["exist_ok"] = True
    train_args["save_dir"] = str(output_run_dir.resolve())
    train_args["epochs"] = trainer_total_epochs
    if source_checkpoint.optimizer_name is not None:
        train_args["optimizer"] = source_checkpoint.optimizer_name
    if device is not None:
        train_args["device"] = device
    if imgsz is not None:
        train_args["imgsz"] = imgsz
    if batch is not None:
        train_args["batch"] = batch
    if workers is not None:
        train_args["workers"] = workers
    ckpt["train_args"] = train_args

    ensure_dir(output_checkpoint.parent)
    torch.save(ckpt, output_checkpoint)
    return output_checkpoint


def resume_mainline_training(
    run_dir: Path,
    target_epochs: int,
    current_epoch_count: int,
    device: str | None,
    output_project_dir: Path,
    output_run_name: str,
    output_run_dir: Path,
    imgsz: int | None,
    batch: int | None,
    workers: int | None,
) -> tuple[Path, ResumeCheckpoint, int]:
    """Resume the selected mainline run to the target epoch count.

    In this environment, final last.pt/best.pt files may be stripped during
    Ultralytics final_eval and become non-resumable. We therefore locate the
    latest checkpoint that still contains optimizer/EMA state, patch its
    saved project/name so that resumed training writes into a fresh run dir,
    and then continue to the target epoch count.
    """
    resume_source = choose_resume_checkpoint(run_dir)
    additional_epochs_needed = max(target_epochs - current_epoch_count, 0)
    if resume_source.human_epoch is None:
        raise RuntimeError("Resume source checkpoint does not record a valid completed epoch count.")
    trainer_total_epochs = resume_source.human_epoch + additional_epochs_needed
    patched_checkpoint = create_resume_checkpoint_copy(
        source_checkpoint=resume_source,
        output_checkpoint=output_project_dir / "_resume_source" / f"{output_run_name}.pt",
        output_project_dir=output_project_dir,
        output_run_name=output_run_name,
        output_run_dir=output_run_dir,
        trainer_total_epochs=trainer_total_epochs,
        device=device,
        imgsz=imgsz,
        batch=batch,
        workers=workers,
    )
    train_kwargs: dict[str, Any] = {
        "epochs": trainer_total_epochs,
    }

    attempts = build_resource_attempts(batch=batch, workers=workers)
    last_error: Exception | None = None
    for attempt_index, (attempt_batch, attempt_workers) in enumerate(attempts, start=1):
        cleanup_partial_run(output_run_dir, patched_checkpoint)
        patched_checkpoint = create_resume_checkpoint_copy(
            source_checkpoint=resume_source,
            output_checkpoint=output_project_dir / "_resume_source" / f"{output_run_name}.pt",
            output_project_dir=output_project_dir,
            output_run_name=output_run_name,
            output_run_dir=output_run_dir,
            trainer_total_epochs=trainer_total_epochs,
            device=device,
            imgsz=imgsz,
            batch=attempt_batch,
            workers=attempt_workers,
        )
        YOLO = import_ultralytics_yolo()
        model = YOLO(str(patched_checkpoint))

        attempt_kwargs = dict(train_kwargs)
        attempt_kwargs["resume"] = str(patched_checkpoint)
        if device:
            attempt_kwargs["device"] = device
        if imgsz is not None:
            attempt_kwargs["imgsz"] = imgsz
        if attempt_batch is not None:
            attempt_kwargs["batch"] = attempt_batch
        if attempt_workers is not None:
            attempt_kwargs["workers"] = attempt_workers

        print_status(
            "INFO",
            "Resuming mainline training from "
            f"{resume_source.path} (human_epoch={resume_source.human_epoch}) "
            f"with optimizer={resume_source.optimizer_name or 'checkpoint_default'} "
            f"to trainer_total_epochs={trainer_total_epochs} "
            f"(target selection-equivalent total={target_epochs}), "
            f"attempt={attempt_index}/{len(attempts)}, batch={attempt_batch}, workers={attempt_workers}.",
        )

        try:
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

            model.train(**attempt_kwargs)
            print_status("OK", "Mainline resume training finished.")
            return output_run_dir, resume_source, trainer_total_epochs
        except Exception as exc:
            last_error = exc
            if not is_cuda_resource_error(exc) or attempt_index == len(attempts):
                raise
            print_status(
                "WARN",
                f"Training failed with a CUDA resource error at batch={attempt_batch}, workers={attempt_workers}: {exc}",
            )
            print_status("WARN", "Retrying with a more conservative batch/workers configuration.")

    if last_error is not None:
        raise last_error
    raise RuntimeError("Unexpected empty retry flow while resuming mainline training.")


def evaluate_one_split(
    weights: Path,
    runtime_yaml: Path,
    report_dir: Path,
    label: str,
    split: str,
    imgsz: int,
    batch: int,
    device: str | None,
) -> EvalSnapshot:
    """Evaluate one weight file on one split."""
    YOLO = import_ultralytics_yolo()
    model = YOLO(str(weights))

    evaluation_project = report_dir / "evaluation"
    ensure_dir(evaluation_project)
    run_name = f"{label.lower()}_{split}"
    eval_kwargs: dict[str, Any] = {
        "data": str(runtime_yaml),
        "split": split,
        "imgsz": imgsz,
        "batch": batch,
        "project": str(evaluation_project),
        "name": run_name,
        "exist_ok": True,
    }
    if device:
        eval_kwargs["device"] = device

    print_status("INFO", f"Running formal evaluation: {label}, split={split}")
    val_result = model.val(**eval_kwargs)
    metrics = extract_metrics(val_result)
    save_dir = evaluation_project / run_name

    metric_preview = ", ".join(
        f"{key}={value:.4f}" for key, value in metrics.items() if key in {"precision", "recall", "map50", "map50_95"}
    )
    if metric_preview:
        print_status("OK", f"{label} {split} metrics: {metric_preview}")

    return EvalSnapshot(
        split=split,
        label=label,
        weights=str(weights.resolve()),
        save_dir=str(save_dir.resolve()) if save_dir.exists() else None,
        precision=metrics.get("precision"),
        recall=metrics.get("recall"),
        map50=metrics.get("map50"),
        map50_95=metrics.get("map50_95"),
    )


def write_csv_rows(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    """Write a UTF-8 CSV report."""
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    """Build a small markdown table."""
    header_line = "| " + " | ".join(headers) + " |"
    separator_line = "| " + " | ".join(["---"] * len(headers)) + " |"
    body_lines = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header_line, separator_line, *body_lines])


def build_selection_rows(
    baseline: RunSnapshot,
    mainline: RunSnapshot,
) -> list[dict[str, Any]]:
    """Build the retained model-selection comparison rows."""
    return [
        {
            "model": baseline.label,
            "epoch": baseline.selection_epoch,
            "imgsz": baseline.imgsz,
            "batch": baseline.batch,
            "workers": baseline.workers,
            "precision": baseline.selection_precision,
            "recall": baseline.selection_recall,
            "map50": baseline.selection_map50,
            "map50_95": baseline.selection_map50_95,
            "decision": "保留为基线对比结果",
            "source_run_dir": baseline.run_dir,
        },
        {
            "model": mainline.label,
            "epoch": mainline.selection_epoch,
            "imgsz": mainline.imgsz,
            "batch": mainline.batch,
            "workers": mainline.workers,
            "precision": mainline.selection_precision,
            "recall": mainline.selection_recall,
            "map50": mainline.selection_map50,
            "map50_95": mainline.selection_map50_95,
            "decision": "作为主线模型继续正式收敛训练",
            "source_run_dir": mainline.run_dir,
        },
    ]


def build_eval_rows(evaluations: list[EvalSnapshot]) -> list[dict[str, Any]]:
    """Build the final evaluation rows."""
    return [
        {
            "model": item.label,
            "split": item.split,
            "precision": item.precision,
            "recall": item.recall,
            "map50": item.map50,
            "map50_95": item.map50_95,
            "weights": item.weights,
            "save_dir": item.save_dir,
        }
        for item in evaluations
    ]


def write_markdown_reports(
    report_dir: Path,
    baseline: RunSnapshot,
    mainline_before: RunSnapshot,
    mainline_after: RunSnapshot,
    resumed_run_dir: Path,
    resume_source: ResumeCheckpoint | None,
    trainer_total_epochs: int | None,
    evaluations: list[EvalSnapshot],
    args: argparse.Namespace,
    runtime_yaml: Path,
    prepare_stats: PrepareStats,
) -> None:
    """Write thesis-ready markdown reports."""
    selection_rows = build_selection_rows(baseline, mainline_before)
    evaluation_rows = build_eval_rows(evaluations)

    selection_table = markdown_table(
        headers=["模型", "Epoch", "imgsz", "batch", "workers", "Precision", "Recall", "mAP50", "mAP50-95", "处理结论"],
        rows=[
            [
                str(row["model"]),
                str(row["epoch"]),
                str(row["imgsz"]),
                str(row["batch"]),
                str(row["workers"]),
                format_metric(as_float(row["precision"])),
                format_metric(as_float(row["recall"])),
                format_metric(as_float(row["map50"])),
                format_metric(as_float(row["map50_95"])),
                str(row["decision"]),
            ]
            for row in selection_rows
        ],
    )
    final_eval_table = markdown_table(
        headers=["模型", "数据划分", "Precision", "Recall", "mAP50", "mAP50-95"],
        rows=[
            [
                row["model"],
                row["split"],
                format_metric(as_float(row["precision"])),
                format_metric(as_float(row["recall"])),
                format_metric(as_float(row["map50"])),
                format_metric(as_float(row["map50_95"])),
            ]
            for row in evaluation_rows
        ],
    )

    selection_md = report_dir / "model_selection_table.md"
    selection_md.write_text("# 模型筛选对比表\n\n" + selection_table + "\n", encoding="utf-8")

    final_eval_md = report_dir / "yolo26s_final_eval.md"
    final_eval_md.write_text("# YOLO26s 最终评估结果\n\n" + final_eval_table + "\n", encoding="utf-8")

    summary_lines = [
        "# Summary",
        "",
        "## 阶段说明",
        "- 当前阶段定义为“主线模型收敛训练阶段”。",
        f"- 基线模型保留为：`{baseline.label}`，其 30 epoch 结果仅用于论文基线对比，不再继续训练。",
        f"- 主线模型为：`{mainline_after.label}`，采用 resume 从已有 checkpoint 继续训练到 `{args.target_epochs}` epoch。",
        "",
        "## 数据与路径",
        f"- dataset_dir: `{args.dataset_dir.resolve()}`",
        f"- runtime_yaml: `{runtime_yaml.resolve()}`",
        f"- baseline_run_dir: `{baseline.run_dir}`",
        f"- model_selection_run_dir: `{mainline_before.run_dir}`",
        f"- resumed_long_train_run_dir: `{resumed_run_dir.resolve()}`",
        f"- report_dir: `{report_dir.resolve()}`",
        "",
        "## 模型筛选实验结论",
        "- 在相同训练条件下，YOLO26s 在 precision、recall、mAP50、mAP50-95 四项指标上整体优于 YOLOv8s。",
        "- 因此后续正式训练与系统实现仅保留 YOLO26s 作为主线模型，YOLOv8s 当前结果保留为论文基线对比结果。",
        "",
        "## 模型筛选对比表",
        selection_table,
        "",
        "## 主线续训记录",
        f"- selection_epoch: `{args.selection_epoch}`",
        f"- mainline_epoch_before_resume: `{mainline_before.epoch_count}`",
        f"- mainline_epoch_after_resume: `{mainline_after.epoch_count}`",
        f"- target_selection_equivalent_epochs: `{args.target_epochs}`",
        f"- actual_trainer_total_epochs: `{trainer_total_epochs if trainer_total_epochs is not None else '-'}`",
        f"- resume_source_checkpoint: `{resume_source.path if resume_source is not None else '-'}`",
        f"- resume_source_human_epoch: `{resume_source.human_epoch if resume_source is not None else '-'}`",
        f"- resume_source_optimizer: `{resume_source.optimizer_name if resume_source is not None else '-'}`",
        (
            "- note: 原始 last.pt/best.pt 已被 Ultralytics strip，"
            "脚本已自动选择最近一个仍保留 optimizer/EMA 状态的中间 checkpoint 继续，"
            "并通过调整 trainer_total_epochs 来补足目标追加轮数。"
            if resume_source is not None and Path(resume_source.path).name != "last.pt"
            else "- note: 本次长训直接从原始可恢复 checkpoint 继续。"
        ),
        f"- avg_epoch_time: `{format_seconds(mainline_after.avg_epoch_time_sec)}`",
        f"- total_train_time: `{format_seconds(mainline_after.total_train_time_sec)}`",
        f"- best_weights: `{mainline_after.best_weights}`",
        f"- last_weights: `{mainline_after.last_weights}`",
        "",
        "## YOLO26s 最终评估结果",
        final_eval_table,
        "",
        "## 运行时数据集准备摘要",
        f"- images_linked: `{prepare_stats.images_linked}`",
        f"- images_copied: `{prepare_stats.images_copied}`",
        f"- empty_labels: `{prepare_stats.empty_labels}`",
        f"- detect_lines_kept: `{prepare_stats.detect_lines_kept}`",
        f"- segment_lines_converted: `{prepare_stats.segment_lines_converted}`",
        f"- invalid_lines_skipped: `{prepare_stats.invalid_lines_skipped}`",
        "",
        "## 输出文件",
        f"- model_selection_table.md: `{selection_md.resolve()}`",
        f"- model_selection_comparison.csv: `{(report_dir / 'model_selection_comparison.csv').resolve()}`",
        f"- yolo26s_final_eval.md: `{final_eval_md.resolve()}`",
        f"- yolo26s_final_evaluation.csv: `{(report_dir / 'yolo26s_final_evaluation.csv').resolve()}`",
        f"- mainline_summary.json: `{(report_dir / 'mainline_summary.json').resolve()}`",
        "",
        "## 论文建议表述",
        "- 为确定后续正式训练主线，本文在相同训练条件下对 YOLOv8s 与 YOLO26s 进行了阶段性模型筛选实验。",
        "- 实验统一采用相同数据集、相同输入尺寸、相同批大小、相同训练轮数与相同随机种子，仅调整模型结构。",
        "- 结果表明，YOLO26s 在四项核心检测指标上整体优于 YOLOv8s，因此选定 YOLO26s 作为后续正式训练与系统实现的主线模型。",
        "",
    ]
    (report_dir / "summary.md").write_text("\n".join(summary_lines), encoding="utf-8")


def main() -> int:
    """Run the mainline convergence stage."""
    args = parse_args()
    stale_kill_count = cleanup_stale_mainline_processes(Path(__file__))
    if stale_kill_count:
        print_status("WARN", f"Cleaned up {stale_kill_count} orphaned mainline worker/process entries from previous runs.")

    dataset_dir = args.dataset_dir.resolve()
    runtime_dataset_dir = args.runtime_dataset_dir.resolve()
    runs_dir = args.runs_dir.resolve()
    baseline_run_dir = args.baseline_run_dir.resolve()
    mainline_run_dir = args.mainline_run_dir.resolve()
    report_name = args.report_name or datetime.now().strftime("mainline_converge_%Y%m%d_%H%M%S")
    report_dir = runs_dir / report_name
    ensure_dir(report_dir)
    long_train_project_dir = report_dir / "training"
    long_train_run_name = f"{slugify_label(args.mainline_label)}_resume_e{args.target_epochs}"
    long_train_run_dir = long_train_project_dir / long_train_run_name

    print_status("INFO", f"Source dataset: {dataset_dir}")
    runtime_yaml, prepare_stats = resolve_runtime_dataset(
        dataset_dir=dataset_dir,
        runtime_dataset_dir=runtime_dataset_dir,
        force_rebuild=args.force_rebuild_runtime_data,
    )

    baseline_snapshot = load_run_snapshot(
        run_dir=baseline_run_dir,
        label=args.baseline_label,
        selection_epoch=args.selection_epoch,
    )
    mainline_before = load_run_snapshot(
        run_dir=mainline_run_dir,
        label=args.mainline_label,
        selection_epoch=args.selection_epoch,
    )
    effective_workers = choose_safe_workers(mainline_before.workers)
    if effective_workers != mainline_before.workers:
        print_status(
            "WARN",
            f"Reducing workers from {mainline_before.workers} to {effective_workers} for stability on this host.",
        )
    resume_source: ResumeCheckpoint | None = None
    trainer_total_epochs: int | None = None

    if args.skip_train:
        print_status("INFO", "Skip-train mode enabled; reusing the existing mainline run.")
        mainline_after_run_dir = mainline_run_dir
    elif mainline_before.epoch_count >= args.target_epochs:
        print_status(
            "INFO",
            f"Mainline run already reached epoch_count={mainline_before.epoch_count}; skipping resume.",
        )
        mainline_after_run_dir = mainline_run_dir
    else:
        mainline_after_run_dir, resume_source, trainer_total_epochs = resume_mainline_training(
            run_dir=mainline_run_dir,
            target_epochs=args.target_epochs,
            current_epoch_count=mainline_before.epoch_count,
            device=args.device,
            output_project_dir=long_train_project_dir,
            output_run_name=long_train_run_name,
            output_run_dir=long_train_run_dir,
            imgsz=mainline_before.imgsz,
            batch=mainline_before.batch,
            workers=effective_workers,
        )

    mainline_after = load_run_snapshot(
        run_dir=mainline_after_run_dir,
        label=args.mainline_label,
        selection_epoch=args.selection_epoch,
    )

    eval_imgsz = args.eval_imgsz or mainline_after.imgsz
    eval_batch = args.eval_batch or mainline_after.batch
    if eval_imgsz is None or eval_batch is None:
        raise ValueError("Failed to resolve eval imgsz/batch from args.yaml; please pass --eval-imgsz and --eval-batch.")

    eval_device = args.device or mainline_after.device
    best_weights = Path(mainline_after.best_weights)
    evaluations = [
        evaluate_one_split(
            weights=best_weights,
            runtime_yaml=runtime_yaml,
            report_dir=report_dir,
            label=args.mainline_label,
            split=split_name,
            imgsz=eval_imgsz,
            batch=eval_batch,
            device=eval_device,
        )
        for split_name in ("val", "test")
    ]

    selection_rows = build_selection_rows(baseline_snapshot, mainline_before)
    evaluation_rows = build_eval_rows(evaluations)
    write_csv_rows(
        path=report_dir / "model_selection_comparison.csv",
        fieldnames=[
            "model",
            "epoch",
            "imgsz",
            "batch",
            "workers",
            "precision",
            "recall",
            "map50",
            "map50_95",
            "decision",
            "source_run_dir",
        ],
        rows=selection_rows,
    )
    write_csv_rows(
        path=report_dir / "yolo26s_final_evaluation.csv",
        fieldnames=["model", "split", "precision", "recall", "map50", "map50_95", "weights", "save_dir"],
        rows=evaluation_rows,
    )

    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_dir": str(dataset_dir),
        "runtime_yaml": str(runtime_yaml.resolve()),
        "selection_epoch": args.selection_epoch,
        "target_epochs": args.target_epochs,
        "trainer_total_epochs": trainer_total_epochs,
        "long_train_run_dir": str(mainline_after_run_dir.resolve()),
        "resume_source": asdict(resume_source) if resume_source is not None else None,
        "baseline": asdict(baseline_snapshot),
        "mainline_before": asdict(mainline_before),
        "mainline_after": asdict(mainline_after),
        "evaluations": [asdict(item) for item in evaluations],
    }
    (report_dir / "mainline_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    write_markdown_reports(
        report_dir=report_dir,
        baseline=baseline_snapshot,
        mainline_before=mainline_before,
        mainline_after=mainline_after,
        resumed_run_dir=mainline_after_run_dir,
        resume_source=resume_source,
        trainer_total_epochs=trainer_total_epochs,
        evaluations=evaluations,
        args=args,
        runtime_yaml=runtime_yaml,
        prepare_stats=prepare_stats,
    )

    print_status("OK", f"Mainline stage summary saved to: {report_dir / 'summary.md'}")
    print_status("OK", f"Model-selection table saved to: {report_dir / 'model_selection_table.md'}")
    print_status("OK", f"Final evaluation table saved to: {report_dir / 'yolo26s_final_eval.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
