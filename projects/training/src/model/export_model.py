from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export trained models for inference service")
    parser.add_argument("--yolo-weights", type=Path, required=True, help="Path to YOLO .pt checkpoint")
    parser.add_argument("--regression-model", type=Path, default=None, help="Path to regression .joblib model")
    parser.add_argument("--out-dir", type=Path, default=Path("models/export"))
    parser.add_argument("--imgsz", type=int, default=640)
    return parser.parse_args()


def export_yolo(weights_path: Path, out_dir: Path, imgsz: int) -> Path:
    if not weights_path.exists():
        raise FileNotFoundError(f"YOLO weights not found: {weights_path}")

    model = YOLO(str(weights_path))
    exported_path = Path(model.export(format="onnx", imgsz=imgsz))
    out_dir.mkdir(parents=True, exist_ok=True)
    target_path = out_dir / "detector.onnx"
    shutil.copy2(exported_path, target_path)
    return target_path


def export_regressor(regression_model: Path | None, out_dir: Path) -> Path | None:
    if regression_model is None:
        return None
    if not regression_model.exists():
        raise FileNotFoundError(f"Regression model not found: {regression_model}")

    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / "dimension_regressor.joblib"
    shutil.copy2(regression_model, target)
    return target


def main() -> None:
    args = parse_args()
    detector_path = export_yolo(args.yolo_weights, args.out_dir, args.imgsz)
    regressor_path = export_regressor(args.regression_model, args.out_dir)

    print(f"Exported YOLO model: {detector_path}")
    if regressor_path:
        print(f"Exported regression model: {regressor_path}")
    else:
        print("No regression model exported")


if __name__ == "__main__":
    main()
