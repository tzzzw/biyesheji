"""Run a staged small-budget hyperparameter tuning workflow.

This script is designed for the "freeze baseline, then improve" stage:
1. Reuse the frozen baseline best.pt as the common starting point.
2. Keep imgsz/batch/workers fixed unless explicitly overridden.
3. Tune only a small set of high-priority hyperparameters in order:
   - round 1: optimizer + lr0
   - round 2: weight_decay + warmup_epochs
   - round 3: mosaic

All trials start from the same frozen baseline weights. Later rounds inherit
only the winning hyperparameter choices from earlier rounds, not their tuned
weights. This keeps the budget small and the comparisons fair.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import time
from argparse import BooleanOptionalAction
from dataclasses import asdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from search_train_config import GpuMemoryMonitor
from search_train_config import extract_epoch_time_stats
from search_train_config import extract_metrics_from_results_csv
from search_train_config import format_seconds
from search_train_config import install_low_prefetch_patch
from search_train_config import resolve_primary_device_index
from search_train_config import resolve_runtime_dataset
from train_compare import PrepareStats
from train_compare import ensure_dir
from train_compare import extract_metrics
from train_compare import import_ultralytics_yolo
from train_compare import print_status


DEFAULT_ROUND1_LRS = [0.005, 0.01]
DEFAULT_ROUND1_OPTIMIZERS = ["AdamW", "SGD"]
DEFAULT_ROUND3_MOSAIC = [1.0, 0.5, 0.2]


@dataclass
class BaselineContext:
    """Resolved frozen baseline metadata."""

    manifest_path: str | None
    weights: str
    label: str
    precision: float | None
    recall: float | None
    map50: float | None
    map50_95: float | None
    imgsz: int
    batch: int
    workers: int
    weight_decay: float
    warmup_epochs: float
    mosaic: float


@dataclass
class TrialSpec:
    """One planned trial inside the small-budget workflow."""

    round_index: int
    round_name: str
    round_title: str
    trial_name: str
    description: str
    source_weights: str
    optimizer: str
    lr0: float
    weight_decay: float
    warmup_epochs: float
    mosaic: float
    inherited_from: str | None = None
    candidate_role: str | None = None
    seed: int | None = None
    epochs_override: int | None = None
    patience_override: int | None = None


@dataclass
class TrialResult:
    """One completed or failed trial result."""

    round_index: int
    round_name: str
    round_title: str
    trial_name: str
    description: str
    inherited_from: str | None
    source_weights: str
    optimizer: str
    lr0: float
    weight_decay: float
    warmup_epochs: float
    mosaic: float
    status: str
    candidate_role: str | None = None
    seed: int | None = None
    epochs: int | None = None
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
    delta_precision_vs_baseline: float | None = None
    delta_recall_vs_baseline: float | None = None
    delta_map50_vs_baseline: float | None = None
    delta_map50_95_vs_baseline: float | None = None
    resume_used: bool = False
    error: str | None = None


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    script_dir = Path(__file__).resolve().parent
    default_runs_dir = script_dir / "runs"
    default_name = datetime.now().strftime("hparam_budget_%Y%m%d_%H%M%S")

    parser = argparse.ArgumentParser(
        description=(
            "Run a staged small-budget hyperparameter tuning workflow based on "
            "the frozen baseline model."
        )
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=None,
        help="Source dataset root directory. Default: infer from project defaults.",
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
        default=default_runs_dir,
        help="Directory containing baseline snapshots and tuning outputs.",
    )
    parser.add_argument(
        "--project",
        type=Path,
        default=None,
        help="Optional explicit tuning output directory. Default: smoke_project/runs/<name>",
    )
    parser.add_argument(
        "--name",
        default=default_name,
        help="Tuning directory name under --runs-dir.",
    )
    parser.add_argument(
        "--baseline-manifest",
        type=Path,
        default=None,
        help="Optional explicit baseline_manifest.json. Default: latest one under --runs-dir.",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=None,
        help="Optional explicit source best.pt. Overrides --baseline-manifest.",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=1,
        help="How many tuning rounds to run. 1=optimizer/lr0, 2=+regularization, 3=+mosaic.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=12,
        help="Epochs for each small-budget trial.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=6,
        help="Early-stopping patience for each trial.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=None,
        help="Override imgsz. Default: inherit from baseline manifest.",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=None,
        help="Override batch size. Default: inherit from baseline manifest.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Override dataloader workers. Default: inherit from baseline manifest.",
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
        "--fraction",
        type=float,
        default=1.0,
        help="Fraction of the training dataset used by each trial.",
    )
    parser.add_argument(
        "--val-split",
        choices=("val", "test"),
        default="val",
        help="Split used for post-training validation.",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Only write the planned round/trial layout without launching training.",
    )
    parser.add_argument(
        "--resume",
        action=BooleanOptionalAction,
        default=True,
        help="Reuse completed trials and resume interrupted trials when possible.",
    )
    parser.add_argument(
        "--low-prefetch",
        action=BooleanOptionalAction,
        default=True,
        help="Lower Ultralytics dataloader prefetch_factor to reduce host-memory pressure.",
    )
    parser.add_argument(
        "--skip-final-val",
        action=BooleanOptionalAction,
        default=False,
        help="Skip the extra validation pass and read metrics from results.csv only.",
    )
    parser.add_argument(
        "--force-rebuild-runtime-data",
        action="store_true",
        help="Rebuild the runtime detection dataset before tuning.",
    )
    parser.add_argument(
        "--round1-lr-values",
        type=float,
        nargs="+",
        default=DEFAULT_ROUND1_LRS,
        help="Candidate lr0 values used in round 1.",
    )
    parser.add_argument(
        "--round1-optimizers",
        nargs="+",
        default=DEFAULT_ROUND1_OPTIMIZERS,
        help="Candidate optimizers used in round 1.",
    )
    parser.add_argument(
        "--round3-mosaic-values",
        type=float,
        nargs="+",
        default=DEFAULT_ROUND3_MOSAIC,
        help="Candidate mosaic values used in round 3.",
    )
    parser.add_argument(
        "--confirmation",
        action=BooleanOptionalAction,
        default=True,
        help="After round 3, run a confirmation stage for the current best group and the backup candidate.",
    )
    parser.add_argument(
        "--confirmation-seed",
        type=int,
        default=123,
        help="Alternate seed used by the confirmation stage.",
    )
    parser.add_argument(
        "--confirmation-epochs",
        type=int,
        default=24,
        help="Epochs used by the confirmation stage. Recommended: 20 to 30.",
    )
    parser.add_argument(
        "--confirmation-patience",
        type=int,
        default=10,
        help="Early-stopping patience used by the confirmation stage.",
    )
    parser.add_argument(
        "--backup-weight-decay",
        type=float,
        default=0.001,
        help="Reserved backup candidate weight_decay used after round 3.",
    )
    parser.add_argument(
        "--backup-warmup-epochs",
        type=float,
        default=5.0,
        help="Reserved backup candidate warmup_epochs used after round 3.",
    )
    return parser.parse_args()


def as_float(value: Any) -> float | None:
    """Convert one metric-like value into float when possible."""
    try:
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def format_metric(value: float | None) -> str:
    """Format one metric for markdown output."""
    return "-" if value is None else f"{value:.4f}"


def format_delta(value: float | None) -> str:
    """Format one delta metric for markdown output."""
    if value is None:
        return "-"
    return f"{value:+.4f}"


def find_latest_baseline_manifest(runs_dir: Path) -> Path | None:
    """Return the most recent baseline_manifest.json under runs_dir."""
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


def resolve_baseline_context(args: argparse.Namespace) -> BaselineContext:
    """Resolve the frozen baseline weights and comparison metrics."""
    if args.weights is not None:
        weights = args.weights.resolve()
        if not weights.exists():
            raise FileNotFoundError(f"Weights file does not exist: {weights}")
        return BaselineContext(
            manifest_path=None,
            weights=str(weights),
            label="explicit_weights",
            precision=None,
            recall=None,
            map50=None,
            map50_95=None,
            imgsz=args.imgsz or 896,
            batch=args.batch or 48,
            workers=args.workers or 16,
            weight_decay=0.0005,
            warmup_epochs=3.0,
            mosaic=1.0,
        )

    manifest_path = args.baseline_manifest.resolve() if args.baseline_manifest else find_latest_baseline_manifest(args.runs_dir.resolve())
    if manifest_path is None:
        raise FileNotFoundError("No baseline_manifest.json found. Freeze a baseline first or pass --weights.")

    manifest = load_json_dict(manifest_path)
    frozen_artifacts = manifest.get("frozen_artifacts")
    training_args = manifest.get("training_args")
    metrics = manifest.get("metrics")
    if not isinstance(frozen_artifacts, dict):
        raise ValueError("baseline_manifest.json is missing frozen_artifacts.")
    if not isinstance(training_args, dict):
        training_args = {}
    if not isinstance(metrics, dict):
        metrics = {}

    weights = frozen_artifacts.get("weights")
    if not isinstance(weights, str):
        raise ValueError("baseline_manifest.json is missing the frozen weights path.")

    reference_metrics = metrics.get("summary_best_pt_validation")
    if not isinstance(reference_metrics, dict):
        reference_metrics = metrics.get("results_csv_best_epoch")
    if not isinstance(reference_metrics, dict):
        reference_metrics = {}

    return BaselineContext(
        manifest_path=str(manifest_path.resolve()),
        weights=str(Path(weights).resolve()),
        label=str(manifest.get("label") or "frozen_baseline"),
        precision=as_float(reference_metrics.get("precision")),
        recall=as_float(reference_metrics.get("recall")),
        map50=as_float(reference_metrics.get("map50")),
        map50_95=as_float(reference_metrics.get("map50_95")),
        imgsz=args.imgsz or int(as_float(training_args.get("imgsz")) or 896),
        batch=args.batch or int(as_float(training_args.get("batch")) or 48),
        workers=args.workers or int(as_float(training_args.get("workers")) or 16),
        weight_decay=as_float(training_args.get("weight_decay")) or 0.0005,
        warmup_epochs=as_float(training_args.get("warmup_epochs")) or 3.0,
        mosaic=as_float(training_args.get("mosaic")) or 1.0,
    )


def sanitize_float_token(value: float) -> str:
    """Convert one float into a filesystem-safe token."""
    text = f"{value:.6f}".rstrip("0").rstrip(".")
    return text.replace("-", "m").replace(".", "p")


def make_project_dir(args: argparse.Namespace) -> Path:
    """Resolve the top-level tuning output directory."""
    if args.project is not None:
        return args.project.resolve()
    return args.runs_dir.resolve() / args.name


def trial_score(result: TrialResult) -> tuple[float, float, float, float, float, float]:
    """Quality-first comparison score for one completed trial."""
    return (
        result.map50_95 if result.map50_95 is not None else -1.0,
        result.recall if result.recall is not None else -1.0,
        result.map50 if result.map50 is not None else -1.0,
        result.precision if result.precision is not None else -1.0,
        -(result.max_gpu_mem_mb or 0),
        -(result.avg_epoch_time_sec or float("inf")),
    )


def select_round_winner(results: list[TrialResult]) -> TrialResult | None:
    """Select the best successful trial from one round."""
    successful = [item for item in results if item.status == "success" and item.map50_95 is not None]
    if not successful:
        return None
    return max(successful, key=trial_score)


def clamp_mosaic(value: float) -> float:
    """Clamp one mosaic factor into the legal [0, 1] range."""
    return min(max(value, 0.0), 1.0)


def resolve_close_mosaic(epochs: int, mosaic: float) -> int:
    """Scale close_mosaic for short runs so mosaic remains observable."""
    if mosaic <= 0.0:
        return 0
    return max(1, min(10, int(round(max(epochs, 1) * 0.1))))


def build_round1_specs(baseline: BaselineContext, args: argparse.Namespace) -> list[TrialSpec]:
    """Create round-1 optimizer/lr0 trial specs."""
    specs: list[TrialSpec] = []
    for optimizer in args.round1_optimizers:
        normalized_optimizer = str(optimizer).strip()
        if not normalized_optimizer:
            continue
        for lr0 in args.round1_lr_values:
            optimizer_token = normalized_optimizer.lower()
            lr_token = sanitize_float_token(lr0)
            specs.append(
                TrialSpec(
                    round_index=1,
                    round_name="optimizer_lr0",
                    round_title="Round 1: Optimizer + lr0",
                    trial_name=f"round1_{optimizer_token}_lr0_{lr_token}",
                    description=f"Test optimizer={normalized_optimizer}, lr0={lr0:g}",
                    source_weights=baseline.weights,
                    optimizer=normalized_optimizer,
                    lr0=lr0,
                    weight_decay=baseline.weight_decay,
                    warmup_epochs=baseline.warmup_epochs,
                    mosaic=baseline.mosaic,
                )
            )
    return specs


def build_round2_specs(
    baseline: BaselineContext,
    round1_winner: TrialResult,
) -> list[TrialSpec]:
    """Create round-2 weight_decay/warmup_epochs trial specs."""
    base_wd = baseline.weight_decay
    base_warmup = baseline.warmup_epochs
    low_wd = round(max(base_wd * 0.2, 1e-5), 6)
    high_wd = round(max(base_wd * 2.0, low_wd), 6)
    short_warmup = round(max(1.0, base_warmup - 1.0), 2)
    long_warmup = round(base_warmup + 2.0, 2)
    inherited = f"{round1_winner.optimizer}/lr0={round1_winner.lr0:g}"

    return [
        TrialSpec(
            round_index=2,
            round_name="weight_decay_warmup",
            round_title="Round 2: weight_decay + warmup_epochs",
            trial_name=(
                "round2_"
                f"{round1_winner.optimizer.lower()}_lr0_{sanitize_float_token(round1_winner.lr0)}_"
                f"wd_{sanitize_float_token(low_wd)}_warmup_{sanitize_float_token(base_warmup)}"
            ),
            description=f"Keep {inherited}, lower weight_decay to {low_wd:g}",
            source_weights=baseline.weights,
            optimizer=round1_winner.optimizer,
            lr0=round1_winner.lr0,
            weight_decay=low_wd,
            warmup_epochs=base_warmup,
            mosaic=baseline.mosaic,
            inherited_from=inherited,
        ),
        TrialSpec(
            round_index=2,
            round_name="weight_decay_warmup",
            round_title="Round 2: weight_decay + warmup_epochs",
            trial_name=(
                "round2_"
                f"{round1_winner.optimizer.lower()}_lr0_{sanitize_float_token(round1_winner.lr0)}_"
                f"wd_{sanitize_float_token(high_wd)}_warmup_{sanitize_float_token(base_warmup)}"
            ),
            description=f"Keep {inherited}, raise weight_decay to {high_wd:g}",
            source_weights=baseline.weights,
            optimizer=round1_winner.optimizer,
            lr0=round1_winner.lr0,
            weight_decay=high_wd,
            warmup_epochs=base_warmup,
            mosaic=baseline.mosaic,
            inherited_from=inherited,
        ),
        TrialSpec(
            round_index=2,
            round_name="weight_decay_warmup",
            round_title="Round 2: weight_decay + warmup_epochs",
            trial_name=(
                "round2_"
                f"{round1_winner.optimizer.lower()}_lr0_{sanitize_float_token(round1_winner.lr0)}_"
                f"wd_{sanitize_float_token(base_wd)}_warmup_{sanitize_float_token(short_warmup)}"
            ),
            description=f"Keep {inherited}, shorten warmup_epochs to {short_warmup:g}",
            source_weights=baseline.weights,
            optimizer=round1_winner.optimizer,
            lr0=round1_winner.lr0,
            weight_decay=base_wd,
            warmup_epochs=short_warmup,
            mosaic=baseline.mosaic,
            inherited_from=inherited,
        ),
        TrialSpec(
            round_index=2,
            round_name="weight_decay_warmup",
            round_title="Round 2: weight_decay + warmup_epochs",
            trial_name=(
                "round2_"
                f"{round1_winner.optimizer.lower()}_lr0_{sanitize_float_token(round1_winner.lr0)}_"
                f"wd_{sanitize_float_token(base_wd)}_warmup_{sanitize_float_token(long_warmup)}"
            ),
            description=f"Keep {inherited}, extend warmup_epochs to {long_warmup:g}",
            source_weights=baseline.weights,
            optimizer=round1_winner.optimizer,
            lr0=round1_winner.lr0,
            weight_decay=base_wd,
            warmup_epochs=long_warmup,
            mosaic=baseline.mosaic,
            inherited_from=inherited,
        ),
    ]


def build_round3_specs(
    baseline: BaselineContext,
    round2_winner: TrialResult,
    args: argparse.Namespace,
) -> list[TrialSpec]:
    """Create round-3 mosaic trial specs."""
    mosaic_values: list[float] = []
    for raw_value in [baseline.mosaic, *args.round3_mosaic_values]:
        value = round(clamp_mosaic(raw_value), 4)
        if value not in mosaic_values:
            mosaic_values.append(value)

    inherited = (
        f"{round2_winner.optimizer}/lr0={round2_winner.lr0:g}/"
        f"wd={round2_winner.weight_decay:g}/warmup={round2_winner.warmup_epochs:g}"
    )
    specs: list[TrialSpec] = []
    for mosaic in mosaic_values:
        specs.append(
            TrialSpec(
                round_index=3,
                round_name="mosaic",
                round_title="Round 3: mosaic",
                trial_name=(
                    "round3_"
                    f"{round2_winner.optimizer.lower()}_lr0_{sanitize_float_token(round2_winner.lr0)}_"
                    f"wd_{sanitize_float_token(round2_winner.weight_decay)}_"
                    f"warmup_{sanitize_float_token(round2_winner.warmup_epochs)}_"
                    f"mosaic_{sanitize_float_token(mosaic)}"
                ),
                description=f"Keep {inherited}, test mosaic={mosaic:g}",
                source_weights=baseline.weights,
                optimizer=round2_winner.optimizer,
                lr0=round2_winner.lr0,
                weight_decay=round2_winner.weight_decay,
                warmup_epochs=round2_winner.warmup_epochs,
                mosaic=mosaic,
                inherited_from=inherited,
            )
        )
    return specs


def build_confirmation_specs(
    round3_winner: TrialResult,
    args: argparse.Namespace,
) -> list[TrialSpec]:
    """Create confirmation-stage specs for the main and backup candidates."""
    confirmation_inherited = f"round3 winner: {round3_winner.trial_name}"
    seed_token = sanitize_float_token(float(args.confirmation_seed))
    mosaic_token = sanitize_float_token(round3_winner.mosaic)
    primary_weight_decay = round3_winner.weight_decay
    backup_weight_decay = args.backup_weight_decay
    backup_warmup = args.backup_warmup_epochs

    return [
        TrialSpec(
            round_index=4,
            round_name="confirmation",
            round_title="Confirmation",
            trial_name=(
                "confirm_primary_"
                f"{round3_winner.optimizer.lower()}_"
                f"lr0_{sanitize_float_token(round3_winner.lr0)}_"
                f"wd_{sanitize_float_token(primary_weight_decay)}_"
                f"warmup_{sanitize_float_token(round3_winner.warmup_epochs)}_"
                f"mosaic_{mosaic_token}_seed_{seed_token}"
            ),
            description=(
                "Confirm the current best group with an alternate seed before "
                "moving to any formal long training."
            ),
            source_weights=round3_winner.source_weights,
            optimizer=round3_winner.optimizer,
            lr0=round3_winner.lr0,
            weight_decay=primary_weight_decay,
            warmup_epochs=round3_winner.warmup_epochs,
            mosaic=round3_winner.mosaic,
            inherited_from=confirmation_inherited,
            candidate_role="primary",
            seed=args.confirmation_seed,
            epochs_override=args.confirmation_epochs,
            patience_override=args.confirmation_patience,
        ),
        TrialSpec(
            round_index=4,
            round_name="confirmation",
            round_title="Confirmation",
            trial_name=(
                "confirm_backup_"
                f"{round3_winner.optimizer.lower()}_"
                f"lr0_{sanitize_float_token(round3_winner.lr0)}_"
                f"wd_{sanitize_float_token(backup_weight_decay)}_"
                f"warmup_{sanitize_float_token(backup_warmup)}_"
                f"mosaic_{mosaic_token}_seed_{seed_token}"
            ),
            description=(
                "Evaluate the reserved backup candidate under the same "
                "confirmation setting so the final recommendation can compare "
                "mAP50-95 and recall directly."
            ),
            source_weights=round3_winner.source_weights,
            optimizer=round3_winner.optimizer,
            lr0=round3_winner.lr0,
            weight_decay=backup_weight_decay,
            warmup_epochs=backup_warmup,
            mosaic=round3_winner.mosaic,
            inherited_from=confirmation_inherited,
            candidate_role="backup",
            seed=args.confirmation_seed,
            epochs_override=args.confirmation_epochs,
            patience_override=args.confirmation_patience,
        ),
    ]


def serialize_plan_specs(round_specs: dict[int, list[TrialSpec]]) -> dict[str, list[dict[str, Any]]]:
    """Convert the staged trial plan into JSON-friendly payloads."""
    return {f"round_{round_index}": [asdict(spec) for spec in specs] for round_index, specs in round_specs.items()}


def load_existing_results(summary_path: Path) -> list[TrialResult]:
    """Load previously saved small-budget results when available."""
    if not summary_path.exists():
        return []

    try:
        payload = load_json_dict(summary_path)
    except (OSError, json.JSONDecodeError, ValueError):
        return []

    items = payload.get("results")
    if not isinstance(items, list):
        return []

    results: list[TrialResult] = []
    valid_fields = TrialResult.__dataclass_fields__.keys()
    for item in items:
        if not isinstance(item, dict):
            continue
        normalized = {field_name: item.get(field_name) for field_name in valid_fields}
        try:
            results.append(TrialResult(**normalized))
        except TypeError:
            continue
    return results


def resolve_round_dir(project_dir: Path, spec: TrialSpec) -> Path:
    """Return the filesystem directory for one round."""
    return project_dir / f"round{spec.round_index}_{spec.round_name}"


def validate_trial(
    best_weights: Path,
    runtime_yaml: Path,
    round_dir: Path,
    spec: TrialSpec,
    args: argparse.Namespace,
    baseline: BaselineContext,
) -> tuple[str | None, dict[str, float]]:
    """Validate one tuned checkpoint and return compact metrics."""
    YOLO = import_ultralytics_yolo()
    model = YOLO(str(best_weights))
    validation_root = round_dir / "validation"
    validation_dir = validation_root / spec.trial_name
    val_kwargs: dict[str, Any] = {
        "data": str(runtime_yaml),
        "split": args.val_split,
        "imgsz": baseline.imgsz,
        "batch": baseline.batch,
        "project": str(validation_root),
        "name": spec.trial_name,
        "exist_ok": True,
    }
    if args.device:
        val_kwargs["device"] = args.device
    val_result = model.val(**val_kwargs)
    return (
        str(validation_dir.resolve()) if validation_dir.exists() else None,
        extract_metrics(val_result),
    )


def trial_metrics_delta(metric_value: float | None, baseline_value: float | None) -> float | None:
    """Compute one metric delta relative to the frozen baseline."""
    if metric_value is None or baseline_value is None:
        return None
    return metric_value - baseline_value


def reuse_saved_result(
    existing_result: TrialResult,
    project_dir: Path,
    spec: TrialSpec,
) -> TrialResult | None:
    """Return one already completed trial result when it can be reused safely."""
    if existing_result.status != "success":
        return None
    if not existing_result.best_weights or not existing_result.results_csv:
        return None

    best_weights = Path(existing_result.best_weights)
    results_csv = Path(existing_result.results_csv)
    if not best_weights.exists() or not results_csv.exists():
        return None

    print_status("INFO", f"Reusing completed trial: {spec.trial_name}")
    return existing_result


def prepare_clean_run_dir(run_dir: Path) -> None:
    """Remove stale unfinished output before relaunching one trial."""
    if run_dir.exists():
        shutil.rmtree(run_dir)


def train_one_trial(
    spec: TrialSpec,
    runtime_yaml: Path,
    project_dir: Path,
    args: argparse.Namespace,
    baseline: BaselineContext,
    existing_result: TrialResult | None,
) -> TrialResult:
    """Run one hyperparameter trial and collect metrics."""
    round_dir = resolve_round_dir(project_dir, spec)
    run_dir = round_dir / spec.trial_name
    best_weights = run_dir / "weights" / "best.pt"
    last_weights = run_dir / "weights" / "last.pt"
    results_csv = run_dir / "results.csv"
    trial_seed = spec.seed if spec.seed is not None else args.seed
    trial_epochs = spec.epochs_override if spec.epochs_override is not None else args.epochs
    trial_patience = spec.patience_override if spec.patience_override is not None else args.patience

    reusable = existing_result and reuse_saved_result(existing_result=existing_result, project_dir=project_dir, spec=spec)
    if reusable is not None:
        return reusable

    if args.low_prefetch:
        install_low_prefetch_patch()

    ensure_dir(round_dir)
    YOLO = import_ultralytics_yolo()
    monitor = GpuMemoryMonitor(args.device)
    start_time = time.perf_counter()

    resume_used = False
    if args.resume and last_weights.exists() and results_csv.exists():
        timing_summary = extract_epoch_time_stats(results_csv)
        if timing_summary is not None and timing_summary.epoch_count < trial_epochs:
            resume_used = True

    print_status("INFO", f"Trial started: {spec.trial_name}")

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
            print_status("INFO", f"[trial] Resuming {spec.trial_name} from: {last_weights}")
            model = YOLO(str(last_weights))
            train_kwargs: dict[str, Any] = {"resume": True}
        else:
            if run_dir.exists() and not best_weights.exists():
                prepare_clean_run_dir(run_dir)
            print_status("INFO", f"[trial] Loading source weights for {spec.trial_name}: {spec.source_weights}")
            model = YOLO(spec.source_weights)
            train_kwargs = {
                "data": str(runtime_yaml),
                "epochs": trial_epochs,
                "imgsz": baseline.imgsz,
                "batch": baseline.batch,
                "workers": baseline.workers,
                "project": str(round_dir),
                "name": spec.trial_name,
                "exist_ok": True,
                "seed": trial_seed,
                "patience": trial_patience,
                "fraction": args.fraction,
                "plots": False,
                "optimizer": spec.optimizer,
                "lr0": spec.lr0,
                "weight_decay": spec.weight_decay,
                "warmup_epochs": spec.warmup_epochs,
                "mosaic": spec.mosaic,
                "close_mosaic": resolve_close_mosaic(trial_epochs, spec.mosaic),
            }
            if args.device:
                train_kwargs["device"] = args.device

        monitor.start()
        model.train(**train_kwargs)
        monitor.stop()
        if torch_module is not None and device_index is not None and torch_module.cuda.is_available():
            torch_module.cuda.synchronize(device_index)
            torch_peak_mem_mb = int(torch_module.cuda.max_memory_reserved(device_index) / (1024 ** 2))

        elapsed = time.perf_counter() - start_time
        timing_summary = extract_epoch_time_stats(results_csv)
        avg_epoch_time_sec = timing_summary.avg_epoch_time_sec if timing_summary else elapsed / max(trial_epochs, 1)
        first_epoch_time_sec = timing_summary.first_epoch_time_sec if timing_summary else None
        total_train_time_sec = timing_summary.total_time_sec if timing_summary else elapsed

        validation_dir: str | None = None
        metrics: dict[str, float] = {}
        if not args.skip_final_val and best_weights.exists():
            validation_dir, metrics = validate_trial(
                best_weights=best_weights,
                runtime_yaml=runtime_yaml,
                round_dir=round_dir,
                spec=spec,
                args=args,
                baseline=baseline,
            )
        if not metrics:
            metrics = extract_metrics_from_results_csv(results_csv)

        max_gpu_candidates = [
            value
            for value in (monitor.max_memory_mb, torch_peak_mem_mb)
            if value is not None and value > 0
        ]
        max_gpu_mem_mb = max(max_gpu_candidates) if max_gpu_candidates else None

        result = TrialResult(
            round_index=spec.round_index,
            round_name=spec.round_name,
            round_title=spec.round_title,
            trial_name=spec.trial_name,
            description=spec.description,
            inherited_from=spec.inherited_from,
            source_weights=spec.source_weights,
            optimizer=spec.optimizer,
            lr0=spec.lr0,
            weight_decay=spec.weight_decay,
            warmup_epochs=spec.warmup_epochs,
            mosaic=spec.mosaic,
            candidate_role=spec.candidate_role,
            seed=trial_seed,
            epochs=trial_epochs,
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
            delta_precision_vs_baseline=trial_metrics_delta(metrics.get("precision"), baseline.precision),
            delta_recall_vs_baseline=trial_metrics_delta(metrics.get("recall"), baseline.recall),
            delta_map50_vs_baseline=trial_metrics_delta(metrics.get("map50"), baseline.map50),
            delta_map50_95_vs_baseline=trial_metrics_delta(metrics.get("map50_95"), baseline.map50_95),
            resume_used=resume_used,
        )
        print_status(
            "OK",
            f"Trial finished: {spec.trial_name}, "
            f"map50-95={format_metric(result.map50_95)}, "
            f"delta={format_delta(result.delta_map50_95_vs_baseline)}, "
            f"gpu_mem={result.max_gpu_mem_mb}MB",
        )
        return result
    except Exception as exc:
        monitor.stop()
        print_status("ERROR", f"Trial failed: {spec.trial_name}: {exc}")
        return TrialResult(
            round_index=spec.round_index,
            round_name=spec.round_name,
            round_title=spec.round_title,
            trial_name=spec.trial_name,
            description=spec.description,
            inherited_from=spec.inherited_from,
            source_weights=spec.source_weights,
            optimizer=spec.optimizer,
            lr0=spec.lr0,
            weight_decay=spec.weight_decay,
            warmup_epochs=spec.warmup_epochs,
            mosaic=spec.mosaic,
            candidate_role=spec.candidate_role,
            seed=trial_seed,
            epochs=trial_epochs,
            status="failed",
            run_dir=str(run_dir.resolve()) if run_dir.exists() else None,
            results_csv=str(results_csv.resolve()) if results_csv.exists() else None,
            max_gpu_mem_mb=monitor.max_memory_mb or None,
            resume_used=resume_used,
            error=str(exc),
        )


def write_results_csv(results: list[TrialResult], csv_path: Path) -> None:
    """Write the flattened trial results table."""
    fieldnames = list(TrialResult.__dataclass_fields__.keys())
    with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))


def build_round_lookup(results: list[TrialResult]) -> dict[int, list[TrialResult]]:
    """Group results by round index."""
    grouped: dict[int, list[TrialResult]] = {}
    for result in results:
        grouped.setdefault(result.round_index, []).append(result)
    for round_results in grouped.values():
        round_results.sort(key=lambda item: item.trial_name)
    return grouped


def resolve_round_title(
    round_index: int,
    planned_rounds: dict[int, list[TrialSpec]],
    round_results: list[TrialResult],
) -> str:
    """Resolve the display title for one round index."""
    specs = planned_rounds.get(round_index, [])
    if specs:
        return specs[0].round_title
    if round_results:
        return round_results[0].round_title
    return f"Round {round_index}"


def find_confirmation_result(results: list[TrialResult], candidate_role: str) -> TrialResult | None:
    """Return the best confirmation result for one candidate role."""
    candidates = [
        item
        for item in results
        if item.round_name == "confirmation"
        and item.candidate_role == candidate_role
        and item.status == "success"
    ]
    if not candidates:
        return None
    return max(candidates, key=trial_score)


def build_candidate_payload(
    label: str,
    result: TrialResult | None,
    fallback: TrialSpec | None = None,
) -> dict[str, Any] | None:
    """Convert one result or spec into a recommendation-friendly payload."""
    if result is None and fallback is None:
        return None

    if result is not None:
        return {
            "label": label,
            "trial_name": result.trial_name,
            "candidate_role": result.candidate_role,
            "optimizer": result.optimizer,
            "lr0": result.lr0,
            "weight_decay": result.weight_decay,
            "warmup_epochs": result.warmup_epochs,
            "mosaic": result.mosaic,
            "precision": result.precision,
            "recall": result.recall,
            "map50": result.map50,
            "map50_95": result.map50_95,
            "delta_map50_95_vs_baseline": result.delta_map50_95_vs_baseline,
            "seed": result.seed,
            "epochs": result.epochs,
            "validation_dir": result.validation_dir,
            "run_dir": result.run_dir,
            "source": "result",
        }

    assert fallback is not None
    return {
        "label": label,
        "trial_name": fallback.trial_name,
        "candidate_role": fallback.candidate_role,
        "optimizer": fallback.optimizer,
        "lr0": fallback.lr0,
        "weight_decay": fallback.weight_decay,
        "warmup_epochs": fallback.warmup_epochs,
        "mosaic": fallback.mosaic,
        "precision": None,
        "recall": None,
        "map50": None,
        "map50_95": None,
        "delta_map50_95_vs_baseline": None,
        "seed": fallback.seed,
        "epochs": fallback.epochs_override,
        "validation_dir": None,
        "run_dir": None,
        "source": "planned_only",
    }


def choose_final_recommendation(
    results: list[TrialResult],
    planned_rounds: dict[int, list[TrialSpec]],
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    """Choose the final main/backup recommendation payload."""
    round3_winner = select_round_winner([item for item in results if item.round_index == 3])
    confirmation_specs = planned_rounds.get(4, [])
    primary_confirmation = find_confirmation_result(results, "primary")
    backup_confirmation = find_confirmation_result(results, "backup")

    primary_payload = build_candidate_payload(
        label="current_best_group",
        result=primary_confirmation or round3_winner,
        fallback=confirmation_specs[0] if confirmation_specs else None,
    )
    backup_payload = build_candidate_payload(
        label="backup_group",
        result=backup_confirmation,
        fallback=confirmation_specs[1] if len(confirmation_specs) >= 2 else None,
    )
    if primary_payload is None:
        return None

    reason_lines: list[str] = []
    confirmation_completed = primary_confirmation is not None
    backup_confirmed = backup_confirmation is not None
    recommended_payload = primary_payload
    alternative_payload = backup_payload

    if confirmation_completed and backup_confirmed and backup_payload is not None:
        primary_map = primary_payload.get("map50_95")
        backup_map = backup_payload.get("map50_95")
        primary_recall = primary_payload.get("recall")
        backup_recall = backup_payload.get("recall")
        map_gap = (primary_map - backup_map) if primary_map is not None and backup_map is not None else None
        recall_gap = (backup_recall - primary_recall) if primary_recall is not None and backup_recall is not None else None

        if map_gap is not None and map_gap < 0:
            recommended_payload = backup_payload
            alternative_payload = primary_payload
            reason_lines.append(
                f"备选组在确认实验中的 mAP50-95 更高 ({backup_map:.4f} vs {primary_map:.4f})，"
                "因此升级为主正式训练参数。"
            )
        elif (
            map_gap is not None
            and recall_gap is not None
            and map_gap <= 0.002
            and recall_gap >= 0.005
        ):
            recommended_payload = backup_payload
            alternative_payload = primary_payload
            reason_lines.append(
                f"两组 mAP50-95 非常接近 (差值 {map_gap:+.4f})，"
                f"但备选组 recall 更高 ({backup_recall:.4f} vs {primary_recall:.4f})，"
                "更适合当前漏检更敏感的目标。"
            )
        else:
            reason_lines.append(
                f"当前最优组在确认实验中保持更高的 mAP50-95 "
                f"({primary_map:.4f} vs {backup_map:.4f})。"
            )
            if recall_gap is not None and recall_gap > 0:
                reason_lines.append(
                    f"备选组 recall 略高 ({backup_recall:.4f} vs {primary_recall:.4f})，"
                    "因此保留为强调漏检控制时的备选参数。"
                )
            elif recall_gap is not None:
                reason_lines.append(
                    f"当前最优组的 recall 也不低于备选组 ({primary_recall:.4f} vs {backup_recall:.4f})，"
                    "综合质量更稳。"
                )
    else:
        if confirmation_completed:
            reason_lines.append(
                "已完成当前最优组的确认实验，但备选组尚无同条件确认结果；"
                "因此主推荐优先沿用已确认的当前最优组。"
            )
        else:
            reason_lines.append(
                "当前仅完成搜索阶段，尚未完成确认实验；"
                "主推荐仍以 round 3 当前最优组为准。"
            )
        if backup_payload is not None:
            reason_lines.append(
                "备选组保留为更高 warmup 的稳健方案，用于正式长训前的最终人工复核。"
            )

    return {
        "confirmation_completed": confirmation_completed,
        "backup_confirmation_completed": backup_confirmed,
        "main_formal": recommended_payload,
        "backup_formal": alternative_payload,
        "reason_lines": reason_lines,
    }


def build_confirmation_markdown(results: list[TrialResult]) -> str:
    """Write a focused markdown view for the confirmation stage."""
    confirmation_results = [item for item in results if item.round_name == "confirmation"]
    lines = ["# Confirmation Summary", ""]
    if not confirmation_results:
        lines.append("- confirmation stage has not produced any result yet")
        return "\n".join(lines) + "\n"

    lines.extend(
        [
            "| role | trial | status | seed | epochs | optimizer | lr0 | weight_decay | warmup_epochs | mosaic | precision | recall | mAP50 | mAP50-95 | ΔmAP50-95 |",
            "|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for result in sorted(confirmation_results, key=lambda item: (item.candidate_role or "", item.trial_name)):
        lines.append(
            f"| {result.candidate_role or '-'} | {result.trial_name} | {result.status} | "
            f"{result.seed if result.seed is not None else '-'} | "
            f"{result.epochs if result.epochs is not None else '-'} | "
            f"{result.optimizer} | {result.lr0:.6f} | {result.weight_decay:.6f} | "
            f"{result.warmup_epochs:.2f} | {result.mosaic:.2f} | "
            f"{format_metric(result.precision)} | {format_metric(result.recall)} | "
            f"{format_metric(result.map50)} | {format_metric(result.map50_95)} | "
            f"{format_delta(result.delta_map50_95_vs_baseline)} |"
        )
    return "\n".join(lines) + "\n"


def build_final_recommendation_markdown(recommendation: dict[str, Any] | None) -> str:
    """Write the final main/backup parameter recommendation."""
    lines = ["# Final Recommendation", ""]
    if recommendation is None:
        lines.append("- final recommendation is not available yet")
        return "\n".join(lines) + "\n"

    def append_candidate(title: str, payload: dict[str, Any] | None) -> None:
        lines.append(f"## {title}")
        if payload is None:
            lines.append("- none")
            lines.append("")
            return
        lines.append(f"- optimizer: `{payload['optimizer']}`")
        lines.append(f"- lr0: `{payload['lr0']}`")
        lines.append(f"- weight_decay: `{payload['weight_decay']}`")
        lines.append(f"- warmup_epochs: `{payload['warmup_epochs']}`")
        lines.append(f"- mosaic: `{payload['mosaic']}`")
        lines.append(f"- source_trial: `{payload['trial_name']}`")
        lines.append(f"- seed: `{payload['seed'] if payload['seed'] is not None else '-'}`")
        lines.append(f"- epochs: `{payload['epochs'] if payload['epochs'] is not None else '-'}`")
        lines.append(f"- precision: `{format_metric(payload.get('precision'))}`")
        lines.append(f"- recall: `{format_metric(payload.get('recall'))}`")
        lines.append(f"- mAP50: `{format_metric(payload.get('map50'))}`")
        lines.append(f"- mAP50-95: `{format_metric(payload.get('map50_95'))}`")
        lines.append(f"- delta_mAP50-95_vs_baseline: `{format_delta(payload.get('delta_map50_95_vs_baseline'))}`")
        lines.append("")

    append_candidate("Main Formal Training Params", recommendation.get("main_formal"))
    append_candidate("Backup Params", recommendation.get("backup_formal"))
    lines.append("## Reason")
    for reason in recommendation.get("reason_lines", []):
        lines.append(f"- {reason}")
    return "\n".join(lines) + "\n"


def build_summary_markdown(
    project_dir: Path,
    baseline: BaselineContext,
    args: argparse.Namespace,
    runtime_yaml: Path,
    prepare_stats: PrepareStats,
    results: list[TrialResult],
    planned_rounds: dict[int, list[TrialSpec]],
    total_memory_mb: int | None,
    recommendation: dict[str, Any] | None,
) -> str:
    """Compose the top-level markdown summary."""
    round_lookup = build_round_lookup(results)
    overall_winner = select_round_winner(results)

    lines = [
        "# Small-Budget Tuning Summary",
        "",
        "## Baseline",
        f"- project_dir: `{project_dir.resolve()}`",
        f"- baseline_label: `{baseline.label}`",
        f"- baseline_manifest: `{baseline.manifest_path or '-'}`",
        f"- source_weights: `{baseline.weights}`",
        f"- runtime_yaml: `{runtime_yaml.resolve()}`",
        f"- target_search_rounds: `{args.rounds}`",
        f"- confirmation_stage: `{args.confirmation and args.rounds >= 3}`",
        f"- fixed_imgsz: `{baseline.imgsz}`",
        f"- fixed_batch: `{baseline.batch}`",
        f"- fixed_workers: `{baseline.workers}`",
        f"- trial_epochs: `{args.epochs}`",
        f"- trial_patience: `{args.patience}`",
        f"- trial_fraction: `{args.fraction}`",
        f"- val_split: `{args.val_split}`",
        f"- low_prefetch: `{args.low_prefetch}`",
        f"- prepare_detect_kept: `{prepare_stats.detect_lines_kept}`",
        f"- baseline_precision: `{format_metric(baseline.precision)}`",
        f"- baseline_recall: `{format_metric(baseline.recall)}`",
        f"- baseline_mAP50: `{format_metric(baseline.map50)}`",
        f"- baseline_mAP50-95: `{format_metric(baseline.map50_95)}`",
    ]

    if total_memory_mb is not None:
        lines.append(f"- total_gpu_mem_mb: `{total_memory_mb}`")

    for round_index in sorted(planned_rounds):
        round_results = round_lookup.get(round_index, [])
        lines.extend(["", f"## {resolve_round_title(round_index, planned_rounds, round_results)}"])
        if not round_results and not planned_rounds.get(round_index):
            lines.append("- deferred until the previous winner is available")
            continue
        if not round_results:
            lines.append("- no result yet")
            continue

        winner = select_round_winner(round_results)
        lines.extend(
            [
                "| trial | role | status | seed | epochs | optimizer | lr0 | weight_decay | warmup_epochs | mosaic | precision | recall | mAP50 | mAP50-95 | ΔmAP50-95 | epoch time | peak GPU mem |",
                "|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for result in sorted(round_results, key=lambda item: (item.status != "success", -trial_score(item)[0], item.trial_name)):
            peak_gpu_text = f"{result.max_gpu_mem_mb}MB" if result.max_gpu_mem_mb is not None else "-"
            lines.append(
                f"| {result.trial_name} | {result.candidate_role or '-'} | {result.status} | "
                f"{result.seed if result.seed is not None else '-'} | "
                f"{result.epochs if result.epochs is not None else '-'} | "
                f"{result.optimizer} | {result.lr0:.6f} | {result.weight_decay:.6f} | {result.warmup_epochs:.2f} | "
                f"{result.mosaic:.2f} | {format_metric(result.precision)} | {format_metric(result.recall)} | "
                f"{format_metric(result.map50)} | {format_metric(result.map50_95)} | "
                f"{format_delta(result.delta_map50_95_vs_baseline)} | {format_seconds(result.avg_epoch_time_sec)} | "
                f"{peak_gpu_text} |"
            )
        lines.append("")
        if winner is None:
            lines.append("- winner: `-`")
        else:
            lines.append(
                "- winner: "
                f"`{winner.trial_name}` "
                f"(optimizer={winner.optimizer}, lr0={winner.lr0:g}, wd={winner.weight_decay:g}, "
                f"warmup={winner.warmup_epochs:g}, mosaic={winner.mosaic:g}, "
                f"mAP50-95={format_metric(winner.map50_95)}, "
                f"delta={format_delta(winner.delta_map50_95_vs_baseline)})"
            )

    lines.extend(["", "## Overall"])
    if overall_winner is None:
        lines.append("- overall_winner: `-`")
        lines.append("- recommendation: 暂未得到成功试验结果，建议先检查训练日志或资源配置。")
    else:
        lines.append(
            "- overall_winner: "
            f"`{overall_winner.trial_name}` "
            f"(mAP50-95={format_metric(overall_winner.map50_95)}, "
            f"delta={format_delta(overall_winner.delta_map50_95_vs_baseline)}, "
            f"recall={format_metric(overall_winner.recall)})"
        )
        if overall_winner.delta_map50_95_vs_baseline is not None and overall_winner.delta_map50_95_vs_baseline > 0:
            lines.append("- recommendation: 已出现优于冻结基线的候选结果，建议先做 20 到 30 epoch 确认实验，再决定是否重启正式长训。")
        else:
            lines.append("- recommendation: 当前试验尚未稳定超越冻结基线，建议继续围绕误差分析结果微调或补数据。")

    lines.extend(["", "## Final Recommendation"])
    if recommendation is None:
        lines.append("- final recommendation: `-`")
    else:
        main_formal = recommendation.get("main_formal")
        backup_formal = recommendation.get("backup_formal")
        if main_formal is not None:
            lines.append(
                "- main_formal: "
                f"`{main_formal['optimizer']}, lr0={main_formal['lr0']}, wd={main_formal['weight_decay']}, "
                f"warmup={main_formal['warmup_epochs']}, mosaic={main_formal['mosaic']}`"
            )
        else:
            lines.append("- main_formal: `-`")
        if backup_formal is not None:
            lines.append(
                "- backup_formal: "
                f"`{backup_formal['optimizer']}, lr0={backup_formal['lr0']}, wd={backup_formal['weight_decay']}, "
                f"warmup={backup_formal['warmup_epochs']}, mosaic={backup_formal['mosaic']}`"
            )
        else:
            lines.append("- backup_formal: `-`")
        for reason in recommendation.get("reason_lines", []):
            lines.append(f"- reason: {reason}")

    return "\n".join(lines) + "\n"


def save_state(
    project_dir: Path,
    baseline: BaselineContext,
    args: argparse.Namespace,
    runtime_yaml: Path,
    prepare_stats: PrepareStats,
    results: list[TrialResult],
    planned_rounds: dict[int, list[TrialSpec]],
    total_memory_mb: int | None,
    target_rounds: int,
) -> tuple[Path, Path, Path, Path, Path]:
    """Persist the current staged tuning state."""
    summary_path = project_dir / "budget_summary.json"
    csv_path = project_dir / "budget_summary.csv"
    markdown_path = project_dir / "summary.md"
    confirmation_markdown_path = project_dir / "confirmation_summary.md"
    recommendation_markdown_path = project_dir / "final_recommendation.md"

    winners: dict[str, dict[str, Any] | None] = {}
    round_lookup = build_round_lookup(results)
    for round_index, round_results in round_lookup.items():
        winner = select_round_winner(round_results)
        winners[f"round_{round_index}"] = asdict(winner) if winner is not None else None
    overall_winner = select_round_winner(results)
    recommendation = choose_final_recommendation(results=results, planned_rounds=planned_rounds, args=args)

    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "project_dir": str(project_dir.resolve()),
        "baseline": asdict(baseline),
        "runtime_yaml": str(runtime_yaml.resolve()),
        "prepare_stats": asdict(prepare_stats),
        "tuning_args": {
            "rounds": target_rounds,
            "epochs": args.epochs,
            "patience": args.patience,
            "device": args.device,
            "seed": args.seed,
            "fraction": args.fraction,
            "val_split": args.val_split,
            "resume": args.resume,
            "low_prefetch": args.low_prefetch,
            "skip_final_val": args.skip_final_val,
            "plan_only": args.plan_only,
            "confirmation": args.confirmation,
            "confirmation_seed": args.confirmation_seed,
            "confirmation_epochs": args.confirmation_epochs,
            "confirmation_patience": args.confirmation_patience,
            "backup_weight_decay": args.backup_weight_decay,
            "backup_warmup_epochs": args.backup_warmup_epochs,
        },
        "planned_rounds": serialize_plan_specs(planned_rounds),
        "results": [asdict(result) for result in results],
        "winners": winners,
        "overall_winner": asdict(overall_winner) if overall_winner is not None else None,
        "final_recommendation": recommendation,
    }
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_results_csv(results, csv_path)
    markdown_path.write_text(
        build_summary_markdown(
            project_dir=project_dir,
            baseline=baseline,
            args=args,
            runtime_yaml=runtime_yaml,
            prepare_stats=prepare_stats,
            results=results,
            planned_rounds=planned_rounds,
            total_memory_mb=total_memory_mb,
            recommendation=recommendation,
        ),
        encoding="utf-8",
    )
    confirmation_markdown_path.write_text(build_confirmation_markdown(results), encoding="utf-8")
    recommendation_markdown_path.write_text(
        build_final_recommendation_markdown(recommendation),
        encoding="utf-8",
    )
    return summary_path, csv_path, markdown_path, confirmation_markdown_path, recommendation_markdown_path


def write_plan_only_files(project_dir: Path, baseline: BaselineContext, round_specs: dict[int, list[TrialSpec]], args: argparse.Namespace) -> tuple[Path, Path]:
    """Write a lightweight plan preview without launching any training."""
    ensure_dir(project_dir)
    plan_json_path = project_dir / "tuning_plan.json"
    plan_md_path = project_dir / "tuning_plan.md"

    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "project_dir": str(project_dir.resolve()),
        "baseline": asdict(baseline),
        "target_rounds": args.rounds,
        "confirmation": args.confirmation and args.rounds >= 3,
        "planned_rounds": serialize_plan_specs(round_specs),
    }
    plan_json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Small-Budget Tuning Plan",
        "",
        "## Baseline",
        f"- baseline_label: `{baseline.label}`",
        f"- baseline_manifest: `{baseline.manifest_path or '-'}`",
        f"- source_weights: `{baseline.weights}`",
        f"- fixed_imgsz: `{baseline.imgsz}`",
        f"- fixed_batch: `{baseline.batch}`",
        f"- fixed_workers: `{baseline.workers}`",
        f"- trial_epochs: `{args.epochs}`",
        f"- confirmation_stage: `{args.confirmation and args.rounds >= 3}`",
        "",
    ]

    for round_index in sorted(round_specs):
        if args.confirmation and args.rounds >= 3 and round_index == 4:
            lines.append("## Confirmation")
        else:
            lines.append(f"## Round {round_index}")
        specs = round_specs[round_index]
        if not specs:
            lines.append("- deferred until previous-round winner is known")
            lines.append("")
            continue
        lines.extend(
            [
                "| trial | optimizer | lr0 | weight_decay | warmup_epochs | mosaic | description |",
                "|---|---|---:|---:|---:|---:|---|",
            ]
        )
        for spec in specs:
            lines.append(
                f"| {spec.trial_name} | {spec.optimizer} | {spec.lr0:.6f} | "
                f"{spec.weight_decay:.6f} | {spec.warmup_epochs:.2f} | {spec.mosaic:.2f} | "
                f"{spec.description} |"
            )
        lines.append("")

    plan_md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return plan_json_path, plan_md_path


def main() -> int:
    """Run the staged small-budget tuning workflow."""
    args = parse_args()
    if args.rounds < 1 or args.rounds > 3:
        print_status("ERROR", "--rounds must be between 1 and 3.")
        return 2

    baseline = resolve_baseline_context(args)
    project_dir = make_project_dir(args)
    ensure_dir(project_dir)

    from dataset_config import find_default_dataset_dir

    script_dir = Path(__file__).resolve().parent
    workspace_dir = script_dir.parent
    dataset_dir = (args.dataset_dir or find_default_dataset_dir(workspace_dir)).resolve()
    runtime_yaml, prepare_stats = resolve_runtime_dataset(
        dataset_dir=dataset_dir,
        runtime_dataset_dir=args.runtime_dataset_dir.resolve(),
        force_rebuild=args.force_rebuild_runtime_data,
    )

    existing_results = load_existing_results(project_dir / "budget_summary.json") if args.resume else []
    results_by_trial_name = {item.trial_name: item for item in existing_results}
    results: list[TrialResult] = list(existing_results)

    round_specs: dict[int, list[TrialSpec]] = {1: build_round1_specs(baseline, args)}

    if args.plan_only:
        for round_index in range(2, args.rounds + 1):
            round_specs.setdefault(round_index, [])
        if args.confirmation and args.rounds >= 3:
            round_specs.setdefault(4, [])
        plan_json_path, plan_md_path = write_plan_only_files(
            project_dir=project_dir,
            baseline=baseline,
            round_specs=round_specs,
            args=args,
        )
        print_status("OK", f"Tuning plan JSON saved to: {plan_json_path}")
        print_status("OK", f"Tuning plan markdown saved to: {plan_md_path}")
        return 0

    total_memory_mb = GpuMemoryMonitor(args.device)._query_memory("memory.total")

    for round_index in range(1, args.rounds + 1):
        if round_index == 2:
            round1_results = [item for item in results if item.round_index == 1]
            round1_winner = select_round_winner(round1_results)
            if round1_winner is None:
                print_status("WARN", "Round 2 skipped because round 1 has no successful winner yet.")
                break
            round_specs[2] = build_round2_specs(baseline=baseline, round1_winner=round1_winner)
        elif round_index == 3:
            round2_results = [item for item in results if item.round_index == 2]
            round2_winner = select_round_winner(round2_results)
            if round2_winner is None:
                print_status("WARN", "Round 3 skipped because round 2 has no successful winner yet.")
                break
            round_specs[3] = build_round3_specs(baseline=baseline, round2_winner=round2_winner, args=args)

        specs = round_specs.get(round_index, [])
        if not specs:
            continue

        print_status("INFO", f"Starting round {round_index} with {len(specs)} trials.")
        for spec in specs:
            existing_result = results_by_trial_name.get(spec.trial_name)
            if existing_result is not None and existing_result.status == "success":
                reusable = reuse_saved_result(existing_result=existing_result, project_dir=project_dir, spec=spec)
                if reusable is not None:
                    continue

            result = train_one_trial(
                spec=spec,
                runtime_yaml=runtime_yaml,
                project_dir=project_dir,
                args=args,
                baseline=baseline,
                existing_result=existing_result,
            )
            results_by_trial_name[spec.trial_name] = result
            results = list(results_by_trial_name.values())
            summary_path, csv_path, markdown_path, confirmation_md_path, recommendation_md_path = save_state(
                project_dir=project_dir,
                baseline=baseline,
                args=args,
                runtime_yaml=runtime_yaml,
                prepare_stats=prepare_stats,
                results=results,
                planned_rounds=round_specs,
                total_memory_mb=total_memory_mb,
                target_rounds=args.rounds,
            )
            print_status("INFO", f"Saved tuning summary to: {summary_path}")
            print_status("INFO", f"Saved tuning CSV to: {csv_path}")
            print_status("INFO", f"Saved tuning markdown to: {markdown_path}")
            print_status("INFO", f"Saved confirmation markdown to: {confirmation_md_path}")
            print_status("INFO", f"Saved recommendation markdown to: {recommendation_md_path}")

        round_results = [item for item in results_by_trial_name.values() if item.round_index == round_index]
        round_winner = select_round_winner(round_results)
        if round_winner is not None:
            print_status(
                "OK",
                f"Round {round_index} winner: {round_winner.trial_name} "
                f"(mAP50-95={format_metric(round_winner.map50_95)}, "
                f"delta={format_delta(round_winner.delta_map50_95_vs_baseline)})",
            )

    if args.confirmation and args.rounds >= 3:
        round3_results = [item for item in results_by_trial_name.values() if item.round_index == 3]
        round3_winner = select_round_winner(round3_results)
        if round3_winner is None:
            print_status("WARN", "Confirmation stage skipped because round 3 has no successful winner yet.")
        else:
            round_specs[4] = build_confirmation_specs(round3_winner=round3_winner, args=args)
            print_status("INFO", f"Starting confirmation stage with {len(round_specs[4])} trials.")
            for spec in round_specs[4]:
                existing_result = results_by_trial_name.get(spec.trial_name)
                if existing_result is not None and existing_result.status == "success":
                    reusable = reuse_saved_result(existing_result=existing_result, project_dir=project_dir, spec=spec)
                    if reusable is not None:
                        continue

                result = train_one_trial(
                    spec=spec,
                    runtime_yaml=runtime_yaml,
                    project_dir=project_dir,
                    args=args,
                    baseline=baseline,
                    existing_result=existing_result,
                )
                results_by_trial_name[spec.trial_name] = result
                results = list(results_by_trial_name.values())
                summary_path, csv_path, markdown_path, confirmation_md_path, recommendation_md_path = save_state(
                    project_dir=project_dir,
                    baseline=baseline,
                    args=args,
                    runtime_yaml=runtime_yaml,
                    prepare_stats=prepare_stats,
                    results=results,
                    planned_rounds=round_specs,
                    total_memory_mb=total_memory_mb,
                    target_rounds=args.rounds,
                )
                print_status("INFO", f"Saved tuning summary to: {summary_path}")
                print_status("INFO", f"Saved tuning CSV to: {csv_path}")
                print_status("INFO", f"Saved tuning markdown to: {markdown_path}")
                print_status("INFO", f"Saved confirmation markdown to: {confirmation_md_path}")
                print_status("INFO", f"Saved recommendation markdown to: {recommendation_md_path}")

            confirmation_results = [item for item in results_by_trial_name.values() if item.round_name == "confirmation"]
            confirmation_winner = select_round_winner(confirmation_results)
            if confirmation_winner is not None:
                print_status(
                    "OK",
                    f"Confirmation-stage top result: {confirmation_winner.trial_name} "
                    f"(mAP50-95={format_metric(confirmation_winner.map50_95)}, "
                    f"delta={format_delta(confirmation_winner.delta_map50_95_vs_baseline)})",
                )

    summary_path, csv_path, markdown_path, confirmation_md_path, recommendation_md_path = save_state(
        project_dir=project_dir,
        baseline=baseline,
        args=args,
        runtime_yaml=runtime_yaml,
        prepare_stats=prepare_stats,
        results=list(results_by_trial_name.values()),
        planned_rounds=round_specs,
        total_memory_mb=total_memory_mb,
        target_rounds=args.rounds,
    )
    print_status("INFO", f"Tuning summary saved to: {summary_path}")
    print_status("INFO", f"Tuning CSV saved to: {csv_path}")
    print_status("INFO", f"Tuning markdown saved to: {markdown_path}")
    print_status("INFO", f"Confirmation markdown saved to: {confirmation_md_path}")
    print_status("INFO", f"Recommendation markdown saved to: {recommendation_md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
