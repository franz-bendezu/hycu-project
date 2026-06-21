from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml


TRAINING_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INFERENCE_MODELS_DIR = TRAINING_ROOT.parents[0] / "inference" / "models"


@dataclass
class PipelineConfig:
    config_path: Path | None
    prepare_dataset: bool
    prepare_class_profile: str
    prepare_clean: bool
    validate_dataset: bool
    dataset_yaml: Path
    yolo_model: str
    epochs: int
    imgsz: int
    batch: int
    patience: int
    workers: int
    project_dir: Path
    experiment: str
    regression_csv: Path | None
    regression_out_dir: Path
    export_out_dir: Path
    deploy_to_inference: bool
    inference_models_dir: Path
    val_ratio: float
    mode: str
    production_mode: bool
    min_train_images: int
    min_val_images: int
    min_non_empty_train_labels: int
    min_non_empty_val_labels: int
    max_class_imbalance_ratio: float
    min_boxes_per_required_class: int
    required_box_classes: tuple[str, ...]
    min_map50: float
    min_map50_95: float
    min_precision: float
    min_recall: float
    dry_run: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Standardized training pipeline: optionally prepare dataset, validate labels, train detector, "
            "export ONNX, and deploy to inference models."
        )
    )
    parser.add_argument("--config", type=Path, default=None, help="YAML config file for reproducible pipeline runs")
    parser.add_argument(
        "--prepare-dataset",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Run dataset preparation before training",
    )
    parser.add_argument(
        "--prepare-class-profile",
        choices=("target3", "components"),
        default=None,
        help="Class profile to emit when preparing dataset",
    )
    parser.add_argument(
        "--prepare-clean",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Rebuild yolo/images and yolo/labels",
    )
    parser.add_argument(
        "--validate-dataset",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Validate label schema and class IDs before training",
    )
    parser.add_argument("--dataset-yaml", type=Path, default=None)
    parser.add_argument("--yolo-model", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--project-dir", type=Path, default=None)
    parser.add_argument("--experiment", default=None)
    parser.add_argument("--regression-csv", type=Path, default=None)
    parser.add_argument("--regression-out-dir", type=Path, default=None)
    parser.add_argument("--export-out-dir", type=Path, default=None)
    parser.add_argument(
        "--deploy-to-inference",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Export ONNX directly to inference models directory",
    )
    parser.add_argument("--inference-models-dir", type=Path, default=None)
    parser.add_argument("--val-ratio", type=float, default=None)
    parser.add_argument("--mode", choices=("copy", "symlink"), default=None)
    parser.add_argument(
        "--production-mode",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable stricter production quality and promotion gates",
    )
    parser.add_argument("--min-train-images", type=int, default=None)
    parser.add_argument("--min-val-images", type=int, default=None)
    parser.add_argument("--min-non-empty-train-labels", type=int, default=None)
    parser.add_argument("--min-non-empty-val-labels", type=int, default=None)
    parser.add_argument("--max-class-imbalance-ratio", type=float, default=None)
    parser.add_argument("--min-boxes-per-required-class", type=int, default=None)
    parser.add_argument(
        "--required-box-classes",
        default=None,
        help="Comma-separated class names that must each meet min box count",
    )
    parser.add_argument("--min-map50", type=float, default=None)
    parser.add_argument("--min-map50-95", type=float, default=None)
    parser.add_argument("--min-precision", type=float, default=None)
    parser.add_argument("--min-recall", type=float, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    return parser.parse_args()


def _resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return TRAINING_ROOT / path


def _load_yaml_config(config_path: Path | None) -> dict:
    if config_path is None:
        return {}
    resolved = _resolve_path(config_path)
    if not resolved.exists():
        raise FileNotFoundError(f"Pipeline config not found: {resolved}")
    data = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError("Pipeline config must be a YAML mapping")
    return data


def _pick(cli_value, cfg: dict, key: str, default):
    if cli_value is not None:
        return cli_value
    value = cfg.get(key)
    if value is not None:
        return value
    return default


def build_config(args: argparse.Namespace) -> PipelineConfig:
    cfg = _load_yaml_config(args.config)
    default_experiment = f"furniture_components_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"

    config_path = _resolve_path(args.config) if args.config else None
    prepare_dataset = bool(_pick(args.prepare_dataset, cfg, "prepare_dataset", False))
    prepare_class_profile = str(_pick(args.prepare_class_profile, cfg, "prepare_class_profile", "components"))
    prepare_clean = bool(_pick(args.prepare_clean, cfg, "prepare_clean", False))
    validate_dataset = bool(_pick(args.validate_dataset, cfg, "validate_dataset", True))
    dataset_yaml = _resolve_path(Path(_pick(args.dataset_yaml, cfg, "dataset_yaml", "datasets/yolo_dataset_components.yaml")))
    yolo_model = str(_pick(args.yolo_model, cfg, "yolo_model", "yolo11n.pt"))
    epochs = int(_pick(args.epochs, cfg, "epochs", 40))
    imgsz = int(_pick(args.imgsz, cfg, "imgsz", 640))
    batch = int(_pick(args.batch, cfg, "batch", 8))
    patience = int(_pick(args.patience, cfg, "patience", 20))
    workers = int(_pick(args.workers, cfg, "workers", 0))
    project_dir = _resolve_path(Path(_pick(args.project_dir, cfg, "project_dir", "runs/train")))
    experiment = str(_pick(args.experiment, cfg, "experiment", default_experiment))
    regression_csv_value = _pick(args.regression_csv, cfg, "regression_csv", None)
    regression_csv = _resolve_path(Path(regression_csv_value)) if regression_csv_value else None
    regression_out_dir = _resolve_path(Path(_pick(args.regression_out_dir, cfg, "regression_out_dir", "models/regression")))
    export_out_dir = _resolve_path(Path(_pick(args.export_out_dir, cfg, "export_out_dir", "models/export")))
    deploy_to_inference = bool(_pick(args.deploy_to_inference, cfg, "deploy_to_inference", False))
    inference_models_dir = _resolve_path(
        Path(_pick(args.inference_models_dir, cfg, "inference_models_dir", str(DEFAULT_INFERENCE_MODELS_DIR)))
    )
    val_ratio = float(_pick(args.val_ratio, cfg, "val_ratio", 0.2))
    mode = str(_pick(args.mode, cfg, "mode", "copy"))
    production_mode = bool(_pick(args.production_mode, cfg, "production_mode", False))
    min_train_images = int(_pick(args.min_train_images, cfg, "min_train_images", 50 if production_mode else 1))
    min_val_images = int(_pick(args.min_val_images, cfg, "min_val_images", 10 if production_mode else 0))
    min_non_empty_train_labels = int(
        _pick(
            args.min_non_empty_train_labels,
            cfg,
            "min_non_empty_train_labels",
            max(1, int(min_train_images * 0.7)) if production_mode else 1,
        )
    )
    min_non_empty_val_labels = int(
        _pick(
            args.min_non_empty_val_labels,
            cfg,
            "min_non_empty_val_labels",
            int(min_val_images * 0.6) if production_mode else 0,
        )
    )
    max_class_imbalance_ratio = float(
        _pick(args.max_class_imbalance_ratio, cfg, "max_class_imbalance_ratio", 3.0 if production_mode else 999.0)
    )
    min_boxes_per_required_class = int(
        _pick(
            args.min_boxes_per_required_class,
            cfg,
            "min_boxes_per_required_class",
            10 if production_mode else 1,
        )
    )
    raw_required_box_classes = _pick(
        args.required_box_classes,
        cfg,
        "required_box_classes",
        "cabinet,desk,shelf",
    )
    if isinstance(raw_required_box_classes, str):
        required_box_classes = tuple(
            item.strip() for item in raw_required_box_classes.split(",") if item.strip()
        )
    elif isinstance(raw_required_box_classes, list):
        required_box_classes = tuple(str(item).strip() for item in raw_required_box_classes if str(item).strip())
    else:
        required_box_classes = ("cabinet", "desk", "shelf")
    min_map50 = float(_pick(args.min_map50, cfg, "min_map50", 0.35 if production_mode else 0.0))
    min_map50_95 = float(_pick(args.min_map50_95, cfg, "min_map50_95", 0.20 if production_mode else 0.0))
    min_precision = float(_pick(args.min_precision, cfg, "min_precision", 0.30 if production_mode else 0.0))
    min_recall = float(_pick(args.min_recall, cfg, "min_recall", 0.25 if production_mode else 0.0))

    return PipelineConfig(
        config_path=config_path,
        prepare_dataset=prepare_dataset,
        prepare_class_profile=prepare_class_profile,
        prepare_clean=prepare_clean,
        validate_dataset=validate_dataset,
        dataset_yaml=dataset_yaml,
        yolo_model=yolo_model,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        patience=patience,
        workers=workers,
        project_dir=project_dir,
        experiment=experiment,
        regression_csv=regression_csv,
        regression_out_dir=regression_out_dir,
        export_out_dir=export_out_dir,
        deploy_to_inference=deploy_to_inference,
        inference_models_dir=inference_models_dir,
        val_ratio=val_ratio,
        mode=mode,
        production_mode=production_mode,
        min_train_images=min_train_images,
        min_val_images=min_val_images,
        min_non_empty_train_labels=min_non_empty_train_labels,
        min_non_empty_val_labels=min_non_empty_val_labels,
        max_class_imbalance_ratio=max_class_imbalance_ratio,
        min_boxes_per_required_class=min_boxes_per_required_class,
        required_box_classes=required_box_classes,
        min_map50=min_map50,
        min_map50_95=min_map50_95,
        min_precision=min_precision,
        min_recall=min_recall,
        dry_run=args.dry_run,
    )


def _resolve_dataset_root(dataset_yaml: Path, yaml_doc: dict[str, Any]) -> Path:
    dataset_root = yaml_doc.get("path")
    if not isinstance(dataset_root, str) or not dataset_root.strip():
        raise ValueError("Dataset YAML must define a non-empty 'path'")
    root = Path(dataset_root)
    if root.is_absolute():
        return root

    candidates = [
        (dataset_yaml.parent / root).resolve(),
        (dataset_yaml.parents[1] / root).resolve(),
        (Path.cwd() / root).resolve(),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _category_from_label_name(label_path: Path) -> str:
    stem = label_path.stem
    if "__" in stem:
        return stem.split("__", 1)[0]
    return "unknown"


def _dataset_label_stats(dataset_yaml: Path) -> dict[str, Any]:
    yaml_doc = yaml.safe_load(dataset_yaml.read_text(encoding="utf-8"))
    if not isinstance(yaml_doc, dict):
        raise ValueError("Dataset YAML must parse into a mapping")

    dataset_root = _resolve_dataset_root(dataset_yaml, yaml_doc)
    labels_root = dataset_root / "labels"
    if not labels_root.exists():
        raise FileNotFoundError(f"Labels directory not found: {labels_root}")

    names_value = yaml_doc.get("names")
    class_name_by_id: dict[int, str] = {}
    if isinstance(names_value, list):
        class_name_by_id = {idx: str(name) for idx, name in enumerate(names_value)}
    elif isinstance(names_value, dict):
        for key, value in names_value.items():
            try:
                class_name_by_id[int(key)] = str(value)
            except Exception:
                continue

    split_counts: dict[str, int] = {"train": 0, "val": 0}
    split_non_empty_counts: dict[str, int] = {"train": 0, "val": 0}
    by_category: dict[str, int] = {}
    non_empty_by_category: dict[str, int] = {}
    boxes_per_class: dict[str, int] = {}
    total_boxes = 0
    for split in ("train", "val"):
        split_dir = labels_root / split
        if not split_dir.exists():
            continue
        for label_file in sorted(split_dir.glob("*.txt")):
            split_counts[split] += 1
            category = _category_from_label_name(label_file)
            by_category[category] = by_category.get(category, 0) + 1

            text = label_file.read_text(encoding="utf-8").strip()
            if not text:
                continue

            split_non_empty_counts[split] += 1
            non_empty_by_category[category] = non_empty_by_category.get(category, 0) + 1

            for line in text.splitlines():
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                try:
                    class_id = int(parts[0])
                except ValueError:
                    continue
                if class_id < 0:
                    continue
                class_name = class_name_by_id.get(class_id, f"class_{class_id}")
                boxes_per_class[class_name] = boxes_per_class.get(class_name, 0) + 1
                total_boxes += 1

    non_zero = [count for count in by_category.values() if count > 0]
    if non_zero:
        imbalance_ratio = max(non_zero) / max(1, min(non_zero))
    else:
        imbalance_ratio = 0.0

    return {
        "dataset_root": str(dataset_root),
        "labels_root": str(labels_root),
        "train_images": split_counts["train"],
        "val_images": split_counts["val"],
        "total_images": split_counts["train"] + split_counts["val"],
        "class_counts": by_category,
        "non_empty_train_labels": split_non_empty_counts["train"],
        "non_empty_val_labels": split_non_empty_counts["val"],
        "total_non_empty_labels": split_non_empty_counts["train"] + split_non_empty_counts["val"],
        "non_empty_category_counts": non_empty_by_category,
        "total_boxes": total_boxes,
        "boxes_per_class": boxes_per_class,
        "class_imbalance_ratio": imbalance_ratio,
    }


def enforce_data_quality_gates(config: PipelineConfig) -> dict[str, Any]:
    stats = _dataset_label_stats(config.dataset_yaml)

    if stats["train_images"] < config.min_train_images:
        raise RuntimeError(
            f"Data quality gate failed: train_images={stats['train_images']} < min_train_images={config.min_train_images}"
        )
    if stats["val_images"] < config.min_val_images:
        raise RuntimeError(
            f"Data quality gate failed: val_images={stats['val_images']} < min_val_images={config.min_val_images}"
        )
    if stats["non_empty_train_labels"] < config.min_non_empty_train_labels:
        raise RuntimeError(
            "Data quality gate failed: non_empty_train_labels="
            f"{stats['non_empty_train_labels']} < min_non_empty_train_labels={config.min_non_empty_train_labels}"
        )
    if stats["non_empty_val_labels"] < config.min_non_empty_val_labels:
        raise RuntimeError(
            "Data quality gate failed: non_empty_val_labels="
            f"{stats['non_empty_val_labels']} < min_non_empty_val_labels={config.min_non_empty_val_labels}"
        )
    if stats["class_imbalance_ratio"] > config.max_class_imbalance_ratio:
        raise RuntimeError(
            "Data quality gate failed: class imbalance ratio "
            f"{stats['class_imbalance_ratio']:.2f} > max_class_imbalance_ratio={config.max_class_imbalance_ratio:.2f}"
        )

    boxes_per_class = stats.get("boxes_per_class", {})
    missing_required: list[str] = []
    for class_name in config.required_box_classes:
        count = int(boxes_per_class.get(class_name, 0))
        if count < config.min_boxes_per_required_class:
            missing_required.append(f"{class_name}={count}")
    if missing_required:
        joined = ", ".join(missing_required)
        raise RuntimeError(
            "Data quality gate failed: insufficient boxes for required classes; "
            f"need >= {config.min_boxes_per_required_class} each, got {joined}"
        )

    print(
        "Data quality gates passed: "
        f"train={stats['train_images']}, val={stats['val_images']}, "
        f"non_empty_train={stats['non_empty_train_labels']}, "
        f"non_empty_val={stats['non_empty_val_labels']}, "
        f"boxes={stats['total_boxes']}, imbalance_ratio={stats['class_imbalance_ratio']:.2f}"
    )
    return stats


def _read_training_metrics(run_dir: Path) -> dict[str, float]:
    results_csv = run_dir / "results.csv"
    if not results_csv.exists():
        raise FileNotFoundError(f"Training metrics file not found: {results_csv}")

    with results_csv.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"Training metrics file is empty: {results_csv}")

    # Use best-epoch validation metrics, not final epoch, to match saved best.pt behavior.
    score_keys = ("metrics/mAP50-95(B)", "metrics/mAP50-95", "metrics/mAP50(B)", "metrics/mAP50")
    best_row = rows[-1]
    best_score = -1.0
    for row in rows:
        score = None
        for key in score_keys:
            value = row.get(key)
            if value not in (None, ""):
                try:
                    score = float(value)
                    break
                except ValueError:
                    continue
        if score is not None and score > best_score:
            best_score = score
            best_row = row

    metrics: dict[str, float] = {}
    for key, value in best_row.items():
        if value is None or value == "":
            continue
        try:
            metrics[key] = float(value)
        except ValueError:
            continue
    return metrics


def _metric_value(metrics: dict[str, float], keys: tuple[str, ...], label: str) -> float:
    for key in keys:
        if key in metrics:
            return metrics[key]
    raise KeyError(f"Could not find metric '{label}' in training results")


def enforce_promotion_gates(config: PipelineConfig, run_dir: Path) -> dict[str, float]:
    metrics = _read_training_metrics(run_dir)
    map50 = _metric_value(metrics, ("metrics/mAP50(B)", "metrics/mAP50"), "mAP50")
    map50_95 = _metric_value(metrics, ("metrics/mAP50-95(B)", "metrics/mAP50-95"), "mAP50-95")
    precision = _metric_value(metrics, ("metrics/precision(B)", "metrics/precision"), "precision")
    recall = _metric_value(metrics, ("metrics/recall(B)", "metrics/recall"), "recall")

    if map50 < config.min_map50:
        raise RuntimeError(f"Promotion gate failed: mAP50={map50:.4f} < min_map50={config.min_map50:.4f}")
    if map50_95 < config.min_map50_95:
        raise RuntimeError(
            f"Promotion gate failed: mAP50-95={map50_95:.4f} < min_map50_95={config.min_map50_95:.4f}"
        )
    if precision < config.min_precision:
        raise RuntimeError(f"Promotion gate failed: precision={precision:.4f} < min_precision={config.min_precision:.4f}")
    if recall < config.min_recall:
        raise RuntimeError(f"Promotion gate failed: recall={recall:.4f} < min_recall={config.min_recall:.4f}")

    print(
        "Promotion gates passed: "
        f"mAP50={map50:.4f}, mAP50-95={map50_95:.4f}, precision={precision:.4f}, recall={recall:.4f}"
    )

    return {
        "mAP50": map50,
        "mAP50_95": map50_95,
        "precision": precision,
        "recall": recall,
    }


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _artifact_metadata(path: Path) -> dict[str, Any]:
    info = {
        "path": str(path),
        "exists": path.exists(),
    }
    if not path.exists():
        return info
    info["size_bytes"] = path.stat().st_size
    info["sha256"] = _sha256(path)
    return info


def _git_commit_hash() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(TRAINING_ROOT),
        check=False,
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def run_cmd(cmd: list[str], *, cwd: Path, dry_run: bool) -> subprocess.CompletedProcess[str]:
    print("$", " ".join(cmd))
    if dry_run:
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")
    return subprocess.run(cmd, cwd=str(cwd), check=False, text=True, capture_output=True)


def _print_result(result: subprocess.CompletedProcess[str]) -> None:
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)


def _has_shared_memory_error(output: str) -> bool:
    low = output.lower()
    return "shared memory" in low or "no space left on device" in low or "unable to allocate shared memory" in low


def prepare_dataset(config: PipelineConfig) -> None:
    cmd = [
        sys.executable,
        str(TRAINING_ROOT / "src" / "data_collection" / "prepare_yolo_dataset.py"),
        "--raw-root",
        "datasets/raw/images",
        "--out-root",
        "datasets/yolo",
        "--val-ratio",
        str(config.val_ratio),
        "--mode",
        config.mode,
        "--class-profile",
        config.prepare_class_profile,
    ]
    if config.prepare_clean:
        cmd.append("--clean")
    result = run_cmd(cmd, cwd=TRAINING_ROOT, dry_run=config.dry_run)
    _print_result(result)
    if result.returncode != 0:
        raise RuntimeError("Dataset preparation failed")


def validate_dataset(config: PipelineConfig) -> None:
    cmd = [
        sys.executable,
        str(TRAINING_ROOT / "src" / "model" / "validate_dataset.py"),
        "--dataset-yaml",
        str(config.dataset_yaml),
    ]
    result = run_cmd(cmd, cwd=TRAINING_ROOT, dry_run=config.dry_run)
    _print_result(result)
    if result.returncode != 0:
        raise RuntimeError("Dataset validation failed")


def run_training(config: PipelineConfig) -> tuple[Path, bool]:
    cmd = [
        sys.executable,
        str(TRAINING_ROOT / "src" / "model" / "train.py"),
        "--dataset-yaml",
        str(config.dataset_yaml),
        "--yolo-model",
        config.yolo_model,
        "--epochs",
        str(config.epochs),
        "--imgsz",
        str(config.imgsz),
        "--batch",
        str(config.batch),
        "--patience",
        str(config.patience),
        "--workers",
        str(config.workers),
        "--project-dir",
        str(config.project_dir),
        "--regression-out-dir",
        str(config.regression_out_dir),
        "--experiment",
        config.experiment,
        "--exist-ok",
    ]
    if config.regression_csv is not None:
        cmd.extend(["--regression-csv", str(config.regression_csv)])

    result = run_cmd(cmd, cwd=TRAINING_ROOT, dry_run=config.dry_run)
    output = (result.stdout or "") + "\n" + (result.stderr or "")
    _print_result(result)
    used_worker_fallback = False

    if result.returncode != 0 and config.workers > 0 and _has_shared_memory_error(output):
        fallback_cmd = cmd[:]
        worker_index = fallback_cmd.index("--workers")
        fallback_cmd[worker_index + 1] = "0"
        print("Detected shared-memory issue; retrying training with --workers 0")
        fallback = run_cmd(fallback_cmd, cwd=TRAINING_ROOT, dry_run=config.dry_run)
        _print_result(fallback)
        if fallback.returncode != 0:
            raise RuntimeError("Training failed even after --workers 0 fallback")
        used_worker_fallback = True
    elif result.returncode != 0:
        raise RuntimeError("Training failed")

    best_weights = config.project_dir / config.experiment / "weights" / "best.pt"
    if not config.dry_run and not best_weights.exists():
        raise FileNotFoundError(f"Training finished but checkpoint not found: {best_weights}")

    return best_weights, used_worker_fallback


def run_export(config: PipelineConfig, best_weights: Path) -> Path:
    export_dir = config.inference_models_dir if config.deploy_to_inference else config.export_out_dir
    cmd = [
        sys.executable,
        str(TRAINING_ROOT / "src" / "model" / "export_model.py"),
        "--yolo-weights",
        str(best_weights),
        "--out-dir",
        str(export_dir),
        "--imgsz",
        str(config.imgsz),
    ]

    regression_model = config.regression_out_dir / "dimension_regressor.joblib"
    if config.regression_csv is not None and (config.dry_run or regression_model.exists()):
        cmd.extend(["--regression-model", str(regression_model)])

    result = run_cmd(cmd, cwd=TRAINING_ROOT, dry_run=config.dry_run)
    _print_result(result)
    if result.returncode != 0:
        raise RuntimeError("Model export failed")

    return export_dir / "detector.onnx"


def write_manifest(
    config: PipelineConfig,
    best_weights: Path,
    detector_onnx: Path,
    used_worker_fallback: bool,
    dataset_stats: dict[str, Any] | None,
    promotion_metrics: dict[str, float] | None,
) -> Path:
    run_dir = config.project_dir / config.experiment
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "pipeline_manifest.json"
    manifest = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "git_commit": _git_commit_hash(),
        "config": {
            **asdict(config),
            "config_path": str(config.config_path) if config.config_path else None,
            "dataset_yaml": str(config.dataset_yaml),
            "project_dir": str(config.project_dir),
            "regression_csv": str(config.regression_csv) if config.regression_csv else None,
            "regression_out_dir": str(config.regression_out_dir),
            "export_out_dir": str(config.export_out_dir),
            "inference_models_dir": str(config.inference_models_dir),
        },
        "artifacts": {
            "best_weights": _artifact_metadata(best_weights),
            "detector_onnx": _artifact_metadata(detector_onnx),
        },
        "used_workers_fallback": used_worker_fallback,
        "dataset_quality": dataset_stats,
        "promotion_metrics": promotion_metrics,
    }
    regression_model = config.regression_out_dir / "dimension_regressor.joblib"
    if regression_model.exists():
        manifest["artifacts"]["dimension_regressor"] = _artifact_metadata(regression_model)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    latest_marker = config.project_dir / "LATEST_RUN.txt"
    latest_marker.write_text(str(run_dir), encoding="utf-8")
    return manifest_path


def main() -> None:
    config = build_config(parse_args())
    dataset_stats: dict[str, Any] | None = None
    promotion_metrics: dict[str, float] | None = None

    if config.prepare_dataset:
        prepare_dataset(config)

    if config.validate_dataset:
        validate_dataset(config)

    if not config.dry_run:
        dataset_stats = enforce_data_quality_gates(config)

    best_weights, used_worker_fallback = run_training(config)
    detector_onnx = run_export(config, best_weights)

    if config.production_mode and not config.dry_run:
        run_dir = config.project_dir / config.experiment
        promotion_metrics = enforce_promotion_gates(config, run_dir)

    manifest_path = write_manifest(
        config,
        best_weights,
        detector_onnx,
        used_worker_fallback,
        dataset_stats,
        promotion_metrics,
    )

    print(f"Best checkpoint: {best_weights}")
    print(f"Exported detector: {detector_onnx}")
    print(f"Run manifest: {manifest_path}")


if __name__ == "__main__":
    main()
