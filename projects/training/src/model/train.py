from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import joblib
import pandas as pd
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.multioutput import MultiOutputRegressor
from ultralytics import YOLO


TRAINING_ROOT = Path(__file__).resolve().parents[2]
RUNS_ROOT = TRAINING_ROOT / "runs"


def _resolve_from_training_root(path: Path) -> Path:
    return path if path.is_absolute() else (TRAINING_ROOT / path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLOv11 detector and material regression model")
    parser.add_argument("--dataset-yaml", type=Path, required=True, help="Path to YOLO dataset YAML")
    parser.add_argument("--regression-csv", type=Path, default=None, help="CSV for regression training")
    parser.add_argument("--yolo-model", default="yolo11n.pt", help="YOLO checkpoint name or path")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16, help="Batch size for YOLO training")
    parser.add_argument("--patience", type=int, default=20, help="Early-stopping patience for YOLO")
    parser.add_argument("--workers", type=int, default=8, help="Data loader workers for YOLO")
    parser.add_argument("--project-dir", type=Path, default=TRAINING_ROOT / "runs" / "train")
    parser.add_argument("--regression-out-dir", type=Path, default=TRAINING_ROOT / "models" / "regression")
    parser.add_argument("--experiment", default="yolo11_furniture")
    parser.add_argument("--resume", action="store_true", help="Resume interrupted YOLO run if available")
    parser.add_argument("--exist-ok", action="store_true", help="Allow reusing an existing experiment directory")
    return parser.parse_args()


def train_detector(args: argparse.Namespace) -> Path:
    if not args.dataset_yaml.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {args.dataset_yaml}")

    model = YOLO(args.yolo_model)
    result = model.train(
        data=str(args.dataset_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        workers=args.workers,
        project=str(args.project_dir),
        name=args.experiment,
        exist_ok=args.exist_ok,
        resume=args.resume,
        # Optimization for small furniture components (hinges, slides, etc.)
        box=7.5,      # Precise box localization
        cls=1.0,      # Give more weight to correct classification
        overlap_mask=True, # Better for components that overlap
        mosaic=1.0,   # Keep mosaic at 1.0 to help with context
        mixup=0.1,    # Add light mixup for generalization
    )

    run_dir = Path(result.save_dir)
    expected_root = RUNS_ROOT.resolve()
    try:
        run_dir.resolve().relative_to(expected_root)
    except ValueError:
        fallback_dir = (args.project_dir / run_dir.name).resolve()
        fallback_dir.parent.mkdir(parents=True, exist_ok=True)
        if fallback_dir.exists():
            shutil.rmtree(fallback_dir)
        shutil.move(str(run_dir), str(fallback_dir))
        run_dir = fallback_dir

    best_weights = run_dir / "weights" / "best.pt"
    if not best_weights.exists():
        raise FileNotFoundError(f"YOLO training completed but best checkpoint missing: {best_weights}")
    return best_weights


def train_regression(args: argparse.Namespace) -> Path | None:
    if args.regression_csv is None:
        print("Skipping regression training: --regression-csv not provided")
        return None

    if not args.regression_csv.exists():
        raise FileNotFoundError(f"Regression CSV not found: {args.regression_csv}")

    df = pd.read_csv(args.regression_csv)
    feature_columns = ["detected_width", "detected_height", "detected_depth", "confidence", "shelf_count"]
    target_columns = ["target_width", "target_height", "target_depth"]

    missing_features = [c for c in feature_columns if c not in df.columns]
    missing_targets = [c for c in target_columns if c not in df.columns]
    if missing_features or missing_targets:
        raise ValueError(
            "Regression CSV missing required columns. "
            f"Missing features: {missing_features}; missing targets: {missing_targets}"
        )

    x = df[feature_columns]
    y = df[target_columns]

    reg = MultiOutputRegressor(RandomForestRegressor(n_estimators=200, random_state=42))
    reg.fit(x, y)
    pred = reg.predict(x)
    mae = mean_absolute_error(y, pred, multioutput="raw_values")

    regression_dir = args.regression_out_dir
    regression_dir.mkdir(parents=True, exist_ok=True)
    model_path = regression_dir / "dimension_regressor.joblib"
    metrics_path = regression_dir / "metrics.json"

    joblib.dump(reg, model_path)
    metrics_path.write_text(
        json.dumps(
            {
                "feature_columns": feature_columns,
                "target_columns": target_columns,
                "mae": {
                    "target_width": float(mae[0]),
                    "target_height": float(mae[1]),
                    "target_depth": float(mae[2]),
                },
            },
            indent=2,
        )
    )
    return model_path


def main() -> None:
    args = parse_args()
    args.dataset_yaml = _resolve_from_training_root(args.dataset_yaml)
    args.project_dir = _resolve_from_training_root(args.project_dir)
    if args.regression_csv is not None:
        args.regression_csv = _resolve_from_training_root(args.regression_csv)
    args.regression_out_dir = _resolve_from_training_root(args.regression_out_dir)

    args.project_dir.mkdir(parents=True, exist_ok=True)

    best_detector = train_detector(args)
    regression_model = train_regression(args)

    print(f"Detector checkpoint: {best_detector}")
    if regression_model:
        print(f"Regression model: {regression_model}")


if __name__ == "__main__":
    main()
