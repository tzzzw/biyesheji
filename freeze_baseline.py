"""Freeze one selected training run as the current baseline snapshot.

This script copies the chosen best.pt, results.csv, and existing validation
artifacts into a dedicated baseline directory, then writes a human-readable
baseline_summary.md for later comparison and paper-writing.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from dataclasses import asdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from train_compare import ensure_dir
from train_compare import print_status


@dataclass
class FrozenSource:
    """Resolved source files and metadata for one frozen baseline."""

    source_summary: str | None
    source_run_dir: str | None
    source_result_index: int | None
    weights: str
    results_csv: str
    validation_dir: str | None
    args_yaml: str | None
    source_result_csv: str | None
    source_markdown: str | None
    summary_metrics: dict[str, float] | None
    summary_decision: str | None
    summary_reason: str | None


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    script_dir = Path(__file__).resolve().parent
    runs_dir = script_dir / "runs"
    default_name = datetime.now().strftime("baseline_freeze_%Y%m%d_%H%M%S")

    parser = argparse.ArgumentParser(
        description=(
            "Freeze the current baseline model by copying best.pt, results.csv, "
            "and existing evaluation outputs into a dedicated snapshot directory."
        )
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=runs_dir,
        help="Directory containing training runs and summaries.",
    )
    parser.add_argument(
        "--project",
        type=Path,
        default=None,
        help="Optional explicit baseline snapshot directory. Default: smoke_project/runs/<name>",
    )
    parser.add_argument(
        "--name",
        default=default_name,
        help="Baseline snapshot name under --project or --runs-dir.",
    )
    parser.add_argument(
        "--label",
        default="current_baseline",
        help="Display label written into baseline_summary.md.",
    )
    parser.add_argument(
        "--source-summary",
        type=Path,
        default=None,
        help="Optional search_summary.json to freeze from. Default: latest one under --runs-dir.",
    )
    parser.add_argument(
        "--source-run-dir",
        type=Path,
        default=None,
        help="Optional explicit Ultralytics run directory containing weights/best.pt and results.csv.",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=None,
        help="Optional explicit best.pt path. When set, --results-csv must also be set.",
    )
    parser.add_argument(
        "--results-csv",
        type=Path,
        default=None,
        help="Optional explicit results.csv path matching --weights.",
    )
    parser.add_argument(
        "--validation-dir",
        type=Path,
        default=None,
        help="Optional explicit existing evaluation directory to preserve.",
    )
    parser.add_argument(
        "--result-index",
        type=int,
        default=None,
        help=(
            "When --source-summary contains multiple successful results, choose "
            "which one to freeze. Default: auto-pick the recommended/best result."
        ),
    )
    return parser.parse_args()


def find_latest_search_summary(runs_dir: Path) -> Path | None:
    """Return the most recent search_summary.json under runs_dir."""
    if not runs_dir.exists():
        return None

    candidates = sorted(
        runs_dir.glob("**/search_summary.json"),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def as_float(value: Any) -> float | None:
    """Convert one metric-like value into float when possible."""
    try:
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def load_yaml_dict(path: Path | None) -> dict[str, Any]:
    """Read one YAML file into a dict when it exists."""
    if path is None or not path.exists():
        return {}
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return {}
    return payload if isinstance(payload, dict) else {}


def infer_validation_dir(run_dir: Path) -> Path | None:
    """Infer the matching validation directory from one Ultralytics run dir."""
    candidate = run_dir.parent / "validation" / run_dir.name
    return candidate if candidate.exists() else None


def copy_file(src: Path, dst: Path) -> None:
    """Copy one file while preserving metadata."""
    ensure_dir(dst.parent)
    shutil.copy2(src, dst)


def copy_directory(src: Path, dst: Path) -> None:
    """Copy one directory tree."""
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def resolve_from_run_dir(run_dir: Path, validation_dir: Path | None) -> FrozenSource:
    """Resolve source files from an explicit Ultralytics run directory."""
    weights = run_dir / "weights" / "best.pt"
    results_csv = run_dir / "results.csv"
    args_yaml = run_dir / "args.yaml"
    resolved_validation_dir = validation_dir or infer_validation_dir(run_dir)

    if not weights.exists():
        raise FileNotFoundError(f"best.pt not found under run dir: {weights}")
    if not results_csv.exists():
        raise FileNotFoundError(f"results.csv not found under run dir: {results_csv}")

    return FrozenSource(
        source_summary=None,
        source_run_dir=str(run_dir.resolve()),
        source_result_index=None,
        weights=str(weights.resolve()),
        results_csv=str(results_csv.resolve()),
        validation_dir=str(resolved_validation_dir.resolve()) if resolved_validation_dir else None,
        args_yaml=str(args_yaml.resolve()) if args_yaml.exists() else None,
        source_result_csv=None,
        source_markdown=None,
        summary_metrics=None,
        summary_decision=None,
        summary_reason=None,
    )


def resolve_from_explicit_paths(
    weights: Path,
    results_csv: Path,
    validation_dir: Path | None,
) -> FrozenSource:
    """Resolve source files from explicit best.pt and results.csv paths."""
    if not weights.exists():
        raise FileNotFoundError(f"best.pt not found: {weights}")
    if not results_csv.exists():
        raise FileNotFoundError(f"results.csv not found: {results_csv}")

    run_dir = results_csv.parent
    args_yaml = run_dir / "args.yaml"
    resolved_validation_dir = validation_dir or infer_validation_dir(run_dir)
    return FrozenSource(
        source_summary=None,
        source_run_dir=str(run_dir.resolve()) if run_dir.exists() else None,
        source_result_index=None,
        weights=str(weights.resolve()),
        results_csv=str(results_csv.resolve()),
        validation_dir=str(resolved_validation_dir.resolve()) if resolved_validation_dir else None,
        args_yaml=str(args_yaml.resolve()) if args_yaml.exists() else None,
        source_result_csv=None,
        source_markdown=None,
        summary_metrics=None,
        summary_decision=None,
        summary_reason=None,
    )


def find_recommended_result_index(summary: dict[str, Any], successful: list[tuple[int, dict[str, Any]]]) -> int | None:
    """Match recommendations.best_quality back to one successful result index."""
    recommendations = summary.get("recommendations")
    if not isinstance(recommendations, dict):
        return None

    best_quality = recommendations.get("best_quality")
    if not isinstance(best_quality, dict):
        return None

    candidate_imgsz = best_quality.get("imgsz")
    candidate_batch = best_quality.get("batch")
    candidate_workers = best_quality.get("workers")
    for raw_index, item in successful:
        if (
            item.get("imgsz") == candidate_imgsz
            and item.get("batch") == candidate_batch
            and item.get("workers") == candidate_workers
        ):
            return raw_index
    return None


def pick_result_from_summary(summary: dict[str, Any], result_index: int | None) -> tuple[int, dict[str, Any]]:
    """Choose one successful result entry from search_summary.json."""
    raw_results = summary.get("results")
    if not isinstance(raw_results, list):
        raise ValueError("search_summary.json does not contain a valid results list.")

    successful: list[tuple[int, dict[str, Any]]] = []
    for index, item in enumerate(raw_results):
        if not isinstance(item, dict):
            continue
        if item.get("status") != "success":
            continue
        if not item.get("best_weights") or not item.get("results_csv"):
            continue
        successful.append((index, item))

    if not successful:
        raise ValueError("No successful result with best_weights/results_csv was found in search_summary.json.")

    if result_index is None:
        recommended_raw_index = find_recommended_result_index(summary, successful)
        if recommended_raw_index is not None:
            for raw_index, item in successful:
                if raw_index == recommended_raw_index:
                    return raw_index, item
        return max(
            successful,
            key=lambda pair: (
                as_float(pair[1].get("map50_95")) or -1.0,
                as_float(pair[1].get("map50")) or -1.0,
                as_float(pair[1].get("recall")) or -1.0,
            ),
        )

    if result_index < 0 or result_index >= len(successful):
        raise IndexError(
            f"Requested --result-index={result_index}, but only {len(successful)} successful results are available."
        )

    return successful[result_index]


def resolve_from_summary(summary_path: Path, result_index: int | None) -> FrozenSource:
    """Resolve source files from one saved search_summary.json."""
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    chosen_index, chosen = pick_result_from_summary(summary, result_index)

    weights = Path(str(chosen["best_weights"]))
    results_csv = Path(str(chosen["results_csv"]))
    run_dir = Path(str(chosen["run_dir"])) if chosen.get("run_dir") else results_csv.parent
    validation_dir = Path(str(chosen["validation_dir"])) if chosen.get("validation_dir") else infer_validation_dir(run_dir)
    args_yaml = run_dir / "args.yaml"

    if not weights.exists():
        raise FileNotFoundError(f"best.pt referenced by summary does not exist: {weights}")
    if not results_csv.exists():
        raise FileNotFoundError(f"results.csv referenced by summary does not exist: {results_csv}")

    summary_metrics = {
        "precision": as_float(chosen.get("precision")),
        "recall": as_float(chosen.get("recall")),
        "map50": as_float(chosen.get("map50")),
        "map50_95": as_float(chosen.get("map50_95")),
    }
    if not any(value is not None for value in summary_metrics.values()):
        summary_metrics = None

    return FrozenSource(
        source_summary=str(summary_path.resolve()),
        source_run_dir=str(run_dir.resolve()) if run_dir.exists() else None,
        source_result_index=chosen_index,
        weights=str(weights.resolve()),
        results_csv=str(results_csv.resolve()),
        validation_dir=str(validation_dir.resolve()) if validation_dir and validation_dir.exists() else None,
        args_yaml=str(args_yaml.resolve()) if args_yaml.exists() else None,
        source_result_csv=str((summary_path.parent / "result.csv").resolve())
        if (summary_path.parent / "result.csv").exists()
        else None,
        source_markdown=str((summary_path.parent / "summary.md").resolve())
        if (summary_path.parent / "summary.md").exists()
        else None,
        summary_metrics=summary_metrics,
        summary_decision=str(chosen.get("recommendation")) if chosen.get("recommendation") else None,
        summary_reason=str(chosen.get("recommendation_reason")) if chosen.get("recommendation_reason") else None,
    )


def resolve_source(args: argparse.Namespace) -> FrozenSource:
    """Resolve which files should be frozen as the baseline snapshot."""
    if args.weights or args.results_csv:
        if args.weights is None or args.results_csv is None:
            raise ValueError("--weights and --results-csv must be provided together.")
        return resolve_from_explicit_paths(
            weights=args.weights.resolve(),
            results_csv=args.results_csv.resolve(),
            validation_dir=args.validation_dir.resolve() if args.validation_dir else None,
        )

    if args.source_run_dir is not None:
        return resolve_from_run_dir(
            run_dir=args.source_run_dir.resolve(),
            validation_dir=args.validation_dir.resolve() if args.validation_dir else None,
        )

    summary_path = args.source_summary.resolve() if args.source_summary else find_latest_search_summary(args.runs_dir.resolve())
    if summary_path is None:
        raise FileNotFoundError("No search_summary.json found. Pass --source-summary, --source-run-dir, or explicit paths.")
    return resolve_from_summary(summary_path=summary_path, result_index=args.result_index)


def read_results_rows(results_csv: Path) -> list[dict[str, str]]:
    """Read Ultralytics results.csv rows."""
    with results_csv.open("r", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def extract_row_metrics(row: dict[str, str]) -> dict[str, float | int | None]:
    """Extract compact metrics from one results.csv row."""
    return {
        "epoch": int(float(row.get("epoch", "0") or 0)),
        "precision": as_float(row.get("metrics/precision(B)")),
        "recall": as_float(row.get("metrics/recall(B)")),
        "map50": as_float(row.get("metrics/mAP50(B)")),
        "map50_95": as_float(row.get("metrics/mAP50-95(B)")),
        "time": as_float(row.get("time")),
    }


def choose_best_row(rows: list[dict[str, str]]) -> dict[str, str]:
    """Choose the best row by mAP50-95, then mAP50."""
    return max(
        rows,
        key=lambda row: (
            as_float(row.get("metrics/mAP50-95(B)")) or -1.0,
            as_float(row.get("metrics/mAP50(B)")) or -1.0,
            as_float(row.get("metrics/recall(B)")) or -1.0,
        ),
    )


def format_metric(value: float | int | None) -> str:
    """Format one compact metric for markdown output."""
    if value is None:
        return "-"
    if isinstance(value, int):
        return str(value)
    return f"{value:.4f}"


def build_summary_markdown(
    baseline_dir: Path,
    label: str,
    source: FrozenSource,
    args_yaml_payload: dict[str, Any],
    row_count: int,
    best_row_metrics: dict[str, float | int | None],
    last_row_metrics: dict[str, float | int | None],
) -> str:
    """Compose baseline_summary.md."""
    target_epochs = as_float(args_yaml_payload.get("epochs"))
    observed_epochs = last_row_metrics.get("epoch")
    early_stopped = (
        isinstance(observed_epochs, int)
        and target_epochs is not None
        and float(observed_epochs) < float(target_epochs)
    )

    table_lines = [
        "| source | epoch | precision | recall | mAP50 | mAP50-95 | note |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    if source.summary_metrics is not None:
        table_lines.append(
            "| existing validation / best.pt | - | "
            f"{format_metric(source.summary_metrics.get('precision'))} | "
            f"{format_metric(source.summary_metrics.get('recall'))} | "
            f"{format_metric(source.summary_metrics.get('map50'))} | "
            f"{format_metric(source.summary_metrics.get('map50_95'))} | "
            "preserved from search_summary.json |"
        )
    table_lines.append(
        "| results.csv best epoch | "
        f"{format_metric(best_row_metrics.get('epoch'))} | "
        f"{format_metric(best_row_metrics.get('precision'))} | "
        f"{format_metric(best_row_metrics.get('recall'))} | "
        f"{format_metric(best_row_metrics.get('map50'))} | "
        f"{format_metric(best_row_metrics.get('map50_95'))} | "
        "best row in training curve |"
    )
    table_lines.append(
        "| results.csv last epoch | "
        f"{format_metric(last_row_metrics.get('epoch'))} | "
        f"{format_metric(last_row_metrics.get('precision'))} | "
        f"{format_metric(last_row_metrics.get('recall'))} | "
        f"{format_metric(last_row_metrics.get('map50'))} | "
        f"{format_metric(last_row_metrics.get('map50_95'))} | "
        "last observed epoch before stop |"
    )

    lines = [
        "# Baseline Summary",
        "",
        "## Snapshot",
        f"- label: `{label}`",
        f"- frozen_at: `{datetime.now().isoformat(timespec='seconds')}`",
        f"- baseline_dir: `{baseline_dir.resolve()}`",
        f"- source_summary: `{source.source_summary or '-'}`",
        f"- source_run_dir: `{source.source_run_dir or '-'}`",
        f"- source_result_index: `{source.source_result_index if source.source_result_index is not None else '-'}`",
        "",
        "## Frozen Artifacts",
        f"- weights: `{(baseline_dir / 'weights' / 'best.pt').resolve()}`",
        f"- results_csv: `{(baseline_dir / 'results.csv').resolve()}`",
        f"- evaluation_dir: `{(baseline_dir / 'evaluation').resolve() if (baseline_dir / 'evaluation').exists() else '-'}`",
        f"- source_args_yaml: `{(baseline_dir / 'source' / 'args.yaml').resolve() if (baseline_dir / 'source' / 'args.yaml').exists() else '-'}`",
        "",
        "## Training Snapshot",
        f"- optimizer: `{args_yaml_payload.get('optimizer', '-')}`",
        f"- imgsz: `{args_yaml_payload.get('imgsz', '-')}`",
        f"- batch: `{args_yaml_payload.get('batch', '-')}`",
        f"- workers: `{args_yaml_payload.get('workers', '-')}`",
        f"- lr0: `{args_yaml_payload.get('lr0', '-')}`",
        f"- weight_decay: `{args_yaml_payload.get('weight_decay', '-')}`",
        f"- warmup_epochs: `{args_yaml_payload.get('warmup_epochs', '-')}`",
        f"- mosaic: `{args_yaml_payload.get('mosaic', '-')}`",
        f"- target_epochs: `{int(target_epochs) if target_epochs is not None else '-'}`",
        f"- observed_epochs: `{observed_epochs if observed_epochs is not None else '-'}`",
        f"- early_stopped_before_target: `{early_stopped if observed_epochs is not None else '-'}`",
        f"- results_row_count: `{row_count}`",
        "",
        "## Metrics",
        *table_lines,
        "",
        "## Decision Snapshot",
        f"- summary_decision: `{source.summary_decision or '-'}`",
        f"- summary_reason: `{source.summary_reason or '-'}`",
        "",
        "## Notes",
        "- This baseline snapshot is frozen for later error analysis and small-budget tuning comparisons.",
        "- No additional long training is launched by this script.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    """Freeze one selected run into a standalone baseline snapshot."""
    args = parse_args()
    runs_dir = args.runs_dir.resolve()
    source = resolve_source(args)

    project_root = args.project.resolve() if args.project else runs_dir
    baseline_dir = project_root if args.project else project_root / args.name
    ensure_dir(baseline_dir)
    ensure_dir(baseline_dir / "weights")
    ensure_dir(baseline_dir / "source")

    weights_path = Path(source.weights)
    results_csv_path = Path(source.results_csv)
    validation_dir_path = Path(source.validation_dir) if source.validation_dir else None
    args_yaml_path = Path(source.args_yaml) if source.args_yaml else None

    copy_file(weights_path, baseline_dir / "weights" / "best.pt")
    copy_file(results_csv_path, baseline_dir / "results.csv")

    if validation_dir_path is not None and validation_dir_path.exists():
        copy_directory(validation_dir_path, baseline_dir / "evaluation")
    if args_yaml_path is not None and args_yaml_path.exists():
        copy_file(args_yaml_path, baseline_dir / "source" / "args.yaml")
    if source.source_summary:
        copy_file(Path(source.source_summary), baseline_dir / "source" / "search_summary.json")
    if source.source_result_csv:
        copy_file(Path(source.source_result_csv), baseline_dir / "source" / "result.csv")
    if source.source_markdown:
        copy_file(Path(source.source_markdown), baseline_dir / "source" / "summary.md")

    rows = read_results_rows(results_csv_path)
    if not rows:
        raise ValueError(f"results.csv contains no training rows: {results_csv_path}")

    best_row_metrics = extract_row_metrics(choose_best_row(rows))
    last_row_metrics = extract_row_metrics(rows[-1])
    args_yaml_payload = load_yaml_dict(args_yaml_path)

    summary_markdown = build_summary_markdown(
        baseline_dir=baseline_dir,
        label=args.label,
        source=source,
        args_yaml_payload=args_yaml_payload,
        row_count=len(rows),
        best_row_metrics=best_row_metrics,
        last_row_metrics=last_row_metrics,
    )
    (baseline_dir / "baseline_summary.md").write_text(summary_markdown, encoding="utf-8")

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "label": args.label,
        "baseline_dir": str(baseline_dir.resolve()),
        "source": asdict(source),
        "frozen_artifacts": {
            "weights": str((baseline_dir / "weights" / "best.pt").resolve()),
            "results_csv": str((baseline_dir / "results.csv").resolve()),
            "evaluation_dir": str((baseline_dir / "evaluation").resolve())
            if (baseline_dir / "evaluation").exists()
            else None,
            "summary_markdown": str((baseline_dir / "baseline_summary.md").resolve()),
        },
        "training_args": {
            key: args_yaml_payload.get(key)
            for key in (
                "model",
                "epochs",
                "patience",
                "imgsz",
                "batch",
                "workers",
                "optimizer",
                "lr0",
                "weight_decay",
                "warmup_epochs",
                "mosaic",
            )
        },
        "metrics": {
            "summary_best_pt_validation": source.summary_metrics,
            "results_csv_best_epoch": best_row_metrics,
            "results_csv_last_epoch": last_row_metrics,
        },
    }
    (baseline_dir / "baseline_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print_status("OK", f"Baseline snapshot saved to: {baseline_dir}")
    print_status("OK", f"Baseline summary saved to: {baseline_dir / 'baseline_summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
