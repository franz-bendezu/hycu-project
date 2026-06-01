from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.multioutput import MultiOutputRegressor
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLOv11 detector and material regression model")
    parser.add_argument("--dataset-yaml", type=Path, required=True, help="Path to YOLO dataset YAML")
    parser.add_argument("--regression-csv", type=Path, default=None, help="CSV for regression training")
    parser.add_argument("--yolo-model", default="yolo11n.pt", help="YOLO checkpoint name or path")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--project-dir", type=Path, default=Path("models"))
    parser.add_argument("--experiment", default="yolo11_furniture")
    return parser.parse_args()


def train_detector(args: argparse.Namespace) -> Path:
    if not args.dataset_yaml.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {args.dataset_yaml}")

    model = YOLO(args.yolo_model)
    result = model.train(
        data=str(args.dataset_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        project=str(args.project_dir),
        name=args.experiment,
    )

    run_dir = Path(result.save_dir)
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

    regression_dir = args.project_dir / "regression"
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
    args.project_dir.mkdir(parents=True, exist_ok=True)

    best_detector = train_detector(args)
    regression_model = train_regression(args)

    print(f"Detector checkpoint: {best_detector}")
    if regression_model:
        print(f"Regression model: {regression_model}")


if __name__ == "__main__":
    main()
