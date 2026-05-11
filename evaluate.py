"""Evaluate trained smoker-detection models on the runtime dataset.

This script is designed to work with train_compare.py:
1. It can read the latest comparison_summary.json automatically.
2. It can prepare or rebuild the runtime detection dataset if needed.
3. It evaluates one or more trained weight files on val/test split.

The original dataset is never modified.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from dataset_config import find_default_dataset_dir
from train_compare import extract_metrics
from train_compare import ensure_dir
from train_compare import import_ultralytics_yolo
from train_compare import prepare_runtime_dataset
from train_compare import print_status


@dataclass
class EvalTarget:
    """One model weight to evaluate."""

    label: str
    weight: str


@dataclass
class EvalResult:
    """Evaluation result summary for one model."""

    label: str
    weight: str
    status: str
    split: str
    save_dir: str | None = None
    metrics: dict[str, float] | None = None
    error: str | None = None


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    script_dir = Path(__file__).resolve().parent
    workspace_dir = script_dir.parent
    default_dataset_dir = find_default_dataset_dir(workspace_dir)

    parser = argparse.ArgumentParser(
        description="Evaluate trained YOLO smoker-detection models."
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
        help="Directory containing training and evaluation outputs.",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=None,
        help="Path to comparison_summary.json. Default: auto-find the latest one under runs-dir.",
    )
    parser.add_argument(
        "--weights",
        nargs="*",
        default=None,
        help="Optional explicit weight paths. If provided, they override --summary.",
    )
    parser.add_argument(
        "--labels",
        nargs="*",
        default=None,
        help="Optional display labels for --weights. Count must match --weights.",
    )
    parser.add_argument(
        "--split",
        choices=("val", "test"),
        default="test",
        help="Dataset split for evaluation.",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size.")
    parser.add_argument("--batch", type=int, default=16, help="Batch size.")
    parser.add_argument(
        "--device",
        default="",
        help="Ultralytics device string, for example 0 or cpu. Empty means auto.",
    )
    parser.add_argument(
        "--force-rebuild-runtime-data",
        action="store_true",
        help="Rebuild the runtime detection dataset before evaluation.",
    )
    return parser.parse_args()


def find_latest_summary(runs_dir: Path) -> Path | None:
    """Find the most recent comparison summary under runs_dir."""
    if not runs_dir.exists():
        return None

    summaries = sorted(
        runs_dir.glob("**/comparison_summary.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return summaries[0] if summaries else None


def infer_label_from_weight(weight: str, index: int) -> str:
    """Build a readable label from a weight path or model string."""
    path_like = Path(weight)
    if path_like.suffix:
        return path_like.stem
    return f"model_{index}"


def resolve_targets_from_weights(args: argparse.Namespace) -> list[EvalTarget]:
    """Build evaluation targets from explicit --weights arguments."""
    if not args.weights:
        return []

    if args.labels and len(args.labels) != len(args.weights):
        raise ValueError("The number of --labels must match the number of --weights.")

    targets: list[EvalTarget] = []
    for index, weight in enumerate(args.weights, start=1):
        label = args.labels[index - 1] if args.labels else infer_label_from_weight(weight, index)
        targets.append(EvalTarget(label=label, weight=weight))
    return targets


def resolve_targets_from_summary(summary_path: Path) -> list[EvalTarget]:
    """Build evaluation targets from a saved training comparison summary."""
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    targets: list[EvalTarget] = []

    for model_info in data.get("models", []):
        status = model_info.get("status")
        best_weights = model_info.get("best_weights")
        label = model_info.get("label") or model_info.get("model") or "model"
        if status != "success":
            continue
        if not best_weights:
            continue
        targets.append(EvalTarget(label=str(label), weight=str(best_weights)))

    return targets


def evaluate_one_model(
    target: EvalTarget,
    runtime_yaml: Path,
    evaluation_dir: Path,
    args: argparse.Namespace,
) -> EvalResult:
    """Run Ultralytics validation for one target model."""
    YOLO = import_ultralytics_yolo()
    print_status("INFO", f"Evaluating {target.label}: {target.weight}")

    try:
        model = YOLO(target.weight)
        val_kwargs: dict[str, Any] = {
            "data": str(runtime_yaml),
            "split": args.split,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "project": str(evaluation_dir),
            "name": target.label,
            "exist_ok": True,
        }
        if args.device:
            val_kwargs["device"] = args.device

        val_result = model.val(**val_kwargs)
        metrics = extract_metrics(val_result)
        if metrics:
            metric_text = ", ".join(f"{key}={value:.4f}" for key, value in metrics.items())
            print_status("OK", f"{target.label} metrics: {metric_text}")
        else:
            print_status("WARN", f"{target.label} evaluation finished, but no metrics were extracted.")

        save_dir = evaluation_dir / target.label
        return EvalResult(
            label=target.label,
            weight=target.weight,
            status="success",
            split=args.split,
            save_dir=str(save_dir.resolve()) if save_dir.exists() else None,
            metrics=metrics or None,
        )
    except Exception as exc:
        print_status("ERROR", f"{target.label} evaluation failed: {exc}")
        return EvalResult(
            label=target.label,
            weight=target.weight,
            status="failed",
            split=args.split,
            error=str(exc),
        )


def save_evaluation_summary(
    summary_path: Path,
    evaluation_dir: Path,
    runtime_yaml: Path,
    args: argparse.Namespace,
    results: list[EvalResult],
) -> Path:
    """Save evaluation results for later reporting."""
    output_path = evaluation_dir / "evaluation_summary.json"
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_summary": str(summary_path.resolve()) if summary_path.exists() else None,
        "runtime_yaml": str(runtime_yaml.resolve()),
        "split": args.split,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "results": [asdict(result) for result in results],
    }
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_path


def resolve_targets(args: argparse.Namespace) -> tuple[list[EvalTarget], Path | None]:
    """Choose evaluation targets from explicit weights or a training summary."""
    explicit_targets = resolve_targets_from_weights(args)
    if explicit_targets:
        return explicit_targets, None

    summary_path = args.summary.resolve() if args.summary else None
    if summary_path is None:
        summary_path = find_latest_summary(args.runs_dir.resolve())

    if summary_path is None:
        raise FileNotFoundError(
            "No comparison_summary.json was found. Train a model first or pass --weights explicitly."
        )
    if not summary_path.exists():
        raise FileNotFoundError(f"Summary file does not exist: {summary_path}")

    targets = resolve_targets_from_summary(summary_path)
    if not targets:
        raise RuntimeError(
            "The summary file does not contain any successful models with best_weights."
        )
    return targets, summary_path


def main() -> int:
    """Entry point for evaluation."""
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()
    runtime_dataset_dir = args.runtime_dataset_dir.resolve()
    runs_dir = args.runs_dir.resolve()

    print_status("INFO", f"Source dataset: {dataset_dir}")
    runtime_yaml, _ = prepare_runtime_dataset(
        dataset_dir=dataset_dir,
        runtime_dataset_dir=runtime_dataset_dir,
        force_rebuild=args.force_rebuild_runtime_data,
    )

    try:
        targets, summary_path = resolve_targets(args)
    except Exception as exc:
        print_status("ERROR", str(exc))
        return 1

    evaluation_name = datetime.now().strftime("evaluate_%Y%m%d_%H%M%S")
    evaluation_dir = runs_dir / evaluation_name
    ensure_dir(evaluation_dir)

    results = [evaluate_one_model(target, runtime_yaml, evaluation_dir, args) for target in targets]
    summary_reference = summary_path if summary_path is not None else Path(".")
    saved_summary = save_evaluation_summary(summary_reference, evaluation_dir, runtime_yaml, args, results)

    success_count = sum(result.status == "success" for result in results)
    failed_count = len(results) - success_count
    print_status("INFO", f"Evaluation summary saved to: {saved_summary}")
    print_status("INFO", f"Evaluation finished: success={success_count}, failed={failed_count}")

    if success_count == 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
