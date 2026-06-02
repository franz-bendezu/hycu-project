from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort  # type: ignore[import-not-found]
from fastapi import HTTPException
from PIL import Image

from app.core.config import get_image_size_fallback, get_execution_providers
from app.utils.image import preprocess

class YoloDetector:
    def __init__(self, model_path: Path, labels: tuple[str, ...], score_threshold: float) -> None:
        if not model_path.exists():
            raise RuntimeError(f"YOLO model not found at {model_path}")

        try:
            providers = get_execution_providers()
            self._session = ort.InferenceSession(str(model_path), providers=providers)
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(f"Failed to load YOLO model from {model_path}: {exc}") from exc

        model_inputs = self._session.get_inputs()
        if len(model_inputs) != 1:
            raise RuntimeError("YOLO model must expose exactly one input tensor")

        input_tensor = model_inputs[0]
        self.input_name = input_tensor.name
        self.input_height, self.input_width = self._resolve_input_size(input_tensor.shape)
        self.labels = labels
        self.model_path = model_path
        self.score_threshold = score_threshold
        self.active_providers = self._session.get_providers()

    def _resolve_input_size(self, shape: list[int | str | None]) -> tuple[int, int]:
        if len(shape) != 4:
            raise RuntimeError(f"Unexpected YOLO input tensor rank: {len(shape)}")

        height = shape[2]
        width = shape[3]
        if isinstance(height, int) and isinstance(width, int):
            return height, width

        fallback_size = get_image_size_fallback()
        return fallback_size, fallback_size

    def analyze(self, image: Image.Image) -> tuple[str, float, list[tuple[int, float]]]:
        tensor = preprocess(image, self.input_width, self.input_height)
        outputs = self._session.run(None, {self.input_name: tensor})
        return self._decode_prediction(outputs, self.labels)

    def _decode_prediction(self, outputs: list[np.ndarray], labels: tuple[str, ...]) -> tuple[str, float, list[tuple[int, float]]]:
        if not outputs:
            raise HTTPException(status_code=500, detail="YOLO model returned no outputs")

        detections = self._extract_detections(outputs[0], len(labels))
        if not detections:
            raise HTTPException(status_code=422, detail="YOLO model produced no detections")

        category_scores: dict[str, float] = {label: 0.0 for label in labels}
        for class_id, score in detections:
            if class_id < 0 or class_id >= len(labels):
                continue
            category = labels[class_id]
            category_scores[category] = max(category_scores[category], score)

        best_category = max(category_scores, key=category_scores.get)
        best_score = category_scores[best_category]

        return best_category, round(float(best_score), 3), detections

    def _extract_detections(self, output: np.ndarray, num_labels: int) -> list[tuple[int, float]]:
        tensor = np.array(output)
        if tensor.ndim == 3 and tensor.shape[0] == 1:
            tensor = tensor[0]

        if tensor.ndim != 2:
            return []

        if tensor.shape[0] < tensor.shape[1]:
            tensor = tensor.T

        rows, cols = tensor.shape
        detections: list[tuple[int, float]] = []

        if cols == 6:
            for row in tensor:
                score = float(row[4])
                class_id = int(row[5])
                detections.append((class_id, score))
            return detections

        if cols < 4 + num_labels:
            return []

        for row in tensor:
            if cols >= 5 + num_labels:
                objectness = float(row[4])
                class_scores = row[5 : 5 + num_labels]
                scores = class_scores * objectness
            else:
                scores = row[4 : 4 + num_labels]

            class_id = int(np.argmax(scores))
            score = float(scores[class_id])
            detections.append((class_id, score))

        return detections
