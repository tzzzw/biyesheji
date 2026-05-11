"""Fresh formal long-training entrypoint for the selected YOLO26s setup.

This script intentionally starts a brand-new run instead of resuming from any
previous checkpoint chain. It then runs formal val/test evaluation
automatically and exports thesis-ready summaries.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from dataset_config import find_default_dataset_dir
from search_train_config import GpuMemoryMonitor
from search_train_config import extract_epoch_time_stats
from search_train_config import format_seconds
from search_train_config import install_low_prefetch_patch
from search_train_config import resolve_primary_device_index
from search_train_config import resolve_runtime_dataset
from train_compare import PrepareStats
from train_compare import ensure_dir
from train_compare import extract_metrics
from train_compare import import_ultralytics_yolo
from train_compare import print_status


DEFAULT_MODEL_PATH = Path("/mnt/毕设/yolo26s.pt")


@dataclass
class EvalSnapshot:
    """Compact formal evaluation result for one split."""

    split: str
    precision: float | None
    recall: float | None
    map50: float | None
    map50_95: float | None
    save_dir: str | None


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    script_dir = Path(__file__).resolve().parent
    workspace_dir = script_dir.parent
    default_dataset_dir = find_default_dataset_dir(workspace_dir)
    default_name = datetime.now().strftime(
        "yolo26s_formal_long_%Y%m%d_e100_img896_b48_sgd_lr0_0p005_wd_0p001_mosaic_0p5_seed123"
    )

    parser = argparse.ArgumentParser(
        description=(
            "Run a fresh formal long-training job for YOLO26s, then perform "
            "formal val/test evaluation automatically."
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
        help="Directory containing all training and reporting outputs.",
    )
    parser.add_argument(
        "--name",
        default=default_name,
        help="Fresh run directory name under --runs-dir.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Fresh initialization weights or model spec for the long run.",
    )
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs.")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience.")
    parser.add_argument("--imgsz", type=int, default=896, help="Training/eval image size.")
    parser.add_argument("--batch", type=int, default=48, help="Training/eval batch size.")
    parser.add_argument("--workers", type=int, default=16, help="Training dataloader workers.")
    parser.add_argument("--device", default="0", help="Ultralytics device string.")
    parser.add_argument("--optimizer", default="SGD", help="Optimizer name.")
    parser.add_argument("--lr0", type=float, default=0.005, help="Initial learning rate.")
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.001,
        help="Weight decay.",
    )
    parser.add_argument(
        "--warmup-epochs",
        type=float,
        default=3.0,
        help="Warmup epochs.",
    )
    parser.add_argument("--mosaic", type=float, default=0.5, help="Mosaic probability.")
    parser.add_argument(
        "--close-mosaic",
        type=int,
        default=2,
        help="Disable mosaic augmentation for the last N epochs.",
    )
    parser.add_argument("--seed", type=int, default=123, help="Random seed.")
    parser.add_argument(
        "--fraction",
        type=float,
        default=1.0,
        help="Fraction of the training set to use.",
    )
    parser.add_argument(
        "--eval-splits",
        nargs="+",
        choices=("val", "test"),
        default=["val", "test"],
        help="Evaluation splits to run after training.",
    )
    parser.add_argument(
        "--force-rebuild-runtime-data",
        action="store_true",
        help="Rebuild the runtime dataset before training.",
    )
    parser.add_argument(
        "--low-prefetch",
        action="store_true",
        help="Lower Ultralytics dataloader prefetch_factor to reduce host-memory pressure.",
    )
    return parser.parse_args()


def evaluate_split(
    best_weights: Path,
    runtime_yaml: Path,
    project_dir: Path,
    split: str,
    args: argparse.Namespace,
) -> EvalSnapshot:
    """Run one formal evaluation split."""
    evaluation_dir = project_dir / "evaluation"
    ensure_dir(evaluation_dir)
    YOLO = import_ultralytics_yolo()
    model = YOLO(str(best_weights))
    run_name = f"best_{split}"
    val_kwargs: dict[str, Any] = {
        "data": str(runtime_yaml),
        "split": split,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "project": str(evaluation_dir),
        "name": run_name,
        "exist_ok": True,
    }
    if args.device:
        val_kwargs["device"] = args.device

    print_status("INFO", f"Running formal evaluation on split={split}")
    val_result = model.val(**val_kwargs)
    metrics = extract_metrics(val_result)
    save_dir = evaluation_dir / run_name
    return EvalSnapshot(
        split=split,
        precision=metrics.get("precision"),
        recall=metrics.get("recall"),
        map50=metrics.get("map50"),
        map50_95=metrics.get("map50_95"),
        save_dir=str(save_dir.resolve()) if save_dir.exists() else None,
    )


def write_csv_rows(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    """Write a UTF-8 CSV file."""
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def format_metric(value: float | None) -> str:
    """Format one metric safely."""
    return "-" if value is None else f"{value:.4f}"


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    """Build a small markdown table."""
    header_line = "| " + " | ".join(headers) + " |"
    separator_line = "| " + " | ".join(["---"] * len(headers)) + " |"
    body_lines = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header_line, separator_line, *body_lines])


def save_outputs(
    project_dir: Path,
    runtime_yaml: Path,
    prepare_stats: PrepareStats,
    args: argparse.Namespace,
    best_weights: Path,
    results_csv: Path,
    max_gpu_mem_mb: int | None,
    train_time_sec: float | None,
    epoch_time_sec: float | None,
    eval_results: list[EvalSnapshot],
) -> None:
    """Persist JSON/CSV/Markdown outputs."""
    summary_json = project_dir / "summary.json"
    result_csv = project_dir / "result.csv"
    summary_md = project_dir / "summary.md"
    train_args_path = project_dir / "train_args.json"
    timing_summary = extract_epoch_time_stats(results_csv)

    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "project_dir": str(project_dir.resolve()),
        "runtime_yaml": str(runtime_yaml.resolve()),
        "best_weights": str(best_weights.resolve()),
        "results_csv": str(results_csv.resolve()),
        "training": {
            "model": str(args.model),
            "epochs": args.epochs,
            "patience": args.patience,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "workers": args.workers,
            "device": args.device,
            "optimizer": args.optimizer,
            "lr0": args.lr0,
            "weight_decay": args.weight_decay,
            "warmup_epochs": args.warmup_epochs,
            "mosaic": args.mosaic,
            "close_mosaic": args.close_mosaic,
            "seed": args.seed,
            "fraction": args.fraction,
            "low_prefetch": args.low_prefetch,
            "peak_gpu_mem_mb": max_gpu_mem_mb,
            "total_train_time_sec": train_time_sec,
            "avg_epoch_time_sec": epoch_time_sec,
            "epoch_timing": asdict(timing_summary) if timing_summary is not None else None,
        },
        "prepare_stats": asdict(prepare_stats),
        "evaluations": [asdict(item) for item in eval_results],
    }
    summary_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    train_args_path.write_text(
        json.dumps(payload["training"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    csv_rows = [
        {
            "split": item.split,
            "precision": item.precision,
            "recall": item.recall,
            "map50": item.map50,
            "map50_95": item.map50_95,
            "save_dir": item.save_dir,
        }
        for item in eval_results
    ]
    write_csv_rows(
        result_csv,
        fieldnames=["split", "precision", "recall", "map50", "map50_95", "save_dir"],
        rows=csv_rows,
    )

    eval_table = markdown_table(
        headers=["Split", "Precision", "Recall", "mAP50", "mAP50-95"],
        rows=[
            [
                item.split,
                format_metric(item.precision),
                format_metric(item.recall),
                format_metric(item.map50),
                format_metric(item.map50_95),
            ]
            for item in eval_results
        ],
    )
    summary_lines = [
        "# Formal Long-Train Summary",
        "",
        "## Run",
        f"- project_dir: `{project_dir.resolve()}`",
        f"- runtime_yaml: `{runtime_yaml.resolve()}`",
        f"- model: `{args.model}`",
        f"- epochs: `{args.epochs}`",
        f"- patience: `{args.patience}`",
        f"- imgsz: `{args.imgsz}`",
        f"- batch: `{args.batch}`",
        f"- workers: `{args.workers}`",
        f"- device: `{args.device}`",
        f"- optimizer: `{args.optimizer}`",
        f"- lr0: `{args.lr0}`",
        f"- weight_decay: `{args.weight_decay}`",
        f"- warmup_epochs: `{args.warmup_epochs}`",
        f"- mosaic: `{args.mosaic}`",
        f"- close_mosaic: `{args.close_mosaic}`",
        f"- seed: `{args.seed}`",
        f"- fresh_start: `True`",
        "",
        "## Training",
        f"- best_weights: `{best_weights.resolve()}`",
        f"- results_csv: `{results_csv.resolve()}`",
        f"- peak_gpu_mem_mb: `{max_gpu_mem_mb if max_gpu_mem_mb is not None else '-'}`",
        f"- total_train_time: `{format_seconds(train_time_sec)}`",
        f"- avg_epoch_time: `{format_seconds(epoch_time_sec)}`",
        "",
        "## Formal Evaluation",
        eval_table,
        "",
        "## Runtime Dataset Preparation",
        f"- images_linked: `{prepare_stats.images_linked}`",
        f"- images_copied: `{prepare_stats.images_copied}`",
        f"- detect_lines_kept: `{prepare_stats.detect_lines_kept}`",
        f"- segment_lines_converted: `{prepare_stats.segment_lines_converted}`",
        f"- invalid_lines_skipped: `{prepare_stats.invalid_lines_skipped}`",
        "",
    ]
    summary_md.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")


def main() -> int:
    """Run the full fresh long-training flow."""
    args = parse_args()
    if args.low_prefetch:
        install_low_prefetch_patch()

    dataset_dir = args.dataset_dir.resolve()
    runtime_dataset_dir = args.runtime_dataset_dir.resolve()
    runs_dir = args.runs_dir.resolve()
    project_dir = runs_dir / args.name
    training_project = project_dir
    training_run_name = "training"
    training_run_dir = training_project / training_run_name
    ensure_dir(project_dir)

    print_status("INFO", f"Fresh formal run dir: {project_dir}")
    runtime_yaml, prepare_stats = resolve_runtime_dataset(
        dataset_dir=dataset_dir,
        runtime_dataset_dir=runtime_dataset_dir,
        force_rebuild=args.force_rebuild_runtime_data,
    )

    YOLO = import_ultralytics_yolo()
    model = YOLO(str(args.model))
    train_kwargs: dict[str, Any] = {
        "data": str(runtime_yaml),
        "epochs": args.epochs,
        "patience": args.patience,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "project": str(training_project),
        "name": training_run_name,
        "exist_ok": True,
        "seed": args.seed,
        "fraction": args.fraction,
        "plots": False,
        "optimizer": args.optimizer,
        "lr0": args.lr0,
        "weight_decay": args.weight_decay,
        "warmup_epochs": args.warmup_epochs,
        "mosaic": args.mosaic,
        "close_mosaic": args.close_mosaic,
        "resume": False,
    }
    if args.device:
        train_kwargs["device"] = args.device

    monitor = GpuMemoryMonitor(args.device)
    train_start = time.perf_counter()
    torch_peak_mem_mb: int | None = None
    device_index = resolve_primary_device_index(args.device)
    torch_module: Any | None = None
    try:
        if device_index is not None:
            try:
                import torch as torch_module

                if torch_module.cuda.is_available():
                    torch_module.cuda.empty_cache()
                    torch_module.cuda.reset_peak_memory_stats(device_index)
            except Exception:
                torch_module = None

        print_status("INFO", "Starting fresh formal long training.")
        monitor.start()
        model.train(**train_kwargs)
        monitor.stop()
        if torch_module is not None and device_index is not None and torch_module.cuda.is_available():
            torch_module.cuda.synchronize(device_index)
            torch_peak_mem_mb = int(torch_module.cuda.max_memory_reserved(device_index) / (1024 ** 2))
    except Exception:
        monitor.stop()
        raise

    results_csv = training_run_dir / "results.csv"
    best_weights = training_run_dir / "weights" / "best.pt"
    if not results_csv.exists():
        raise FileNotFoundError(f"Training results.csv not found: {results_csv}")
    if not best_weights.exists():
        raise FileNotFoundError(f"Training best.pt not found: {best_weights}")

    elapsed = time.perf_counter() - train_start
    timing_summary = extract_epoch_time_stats(results_csv)
    avg_epoch_time_sec = timing_summary.avg_epoch_time_sec if timing_summary is not None else elapsed / max(args.epochs, 1)
    total_train_time_sec = timing_summary.total_time_sec if timing_summary is not None else elapsed
    peak_gpu_mem_mb = max(
        [
            value
            for value in (monitor.max_memory_mb, torch_peak_mem_mb)
            if value is not None and value > 0
        ],
        default=None,
    )

    eval_results = [
        evaluate_split(
            best_weights=best_weights,
            runtime_yaml=runtime_yaml,
            project_dir=project_dir,
            split=split,
            args=args,
        )
        for split in args.eval_splits
    ]
    save_outputs(
        project_dir=project_dir,
        runtime_yaml=runtime_yaml,
        prepare_stats=prepare_stats,
        args=args,
        best_weights=best_weights,
        results_csv=results_csv,
        max_gpu_mem_mb=peak_gpu_mem_mb,
        train_time_sec=total_train_time_sec,
        epoch_time_sec=avg_epoch_time_sec,
        eval_results=eval_results,
    )

    print_status("OK", f"Formal long-train summary saved to: {project_dir / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
