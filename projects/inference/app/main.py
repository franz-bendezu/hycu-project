from __future__ import annotations

import base64
import io
import os
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from statistics import mean
from urllib.parse import unquote_to_bytes

import httpx
import numpy as np
import onnxruntime as ort  # type: ignore[import-not-found]
from fastapi import FastAPI, HTTPException
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, Field, model_validator


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Fail fast during service boot when the detector artifact is missing or invalid.
    _detector()
    yield


app = FastAPI(title="Vision Inference Service", version="0.1.0", lifespan=lifespan)

SUPPORTED_TYPES = ("cabinet", "desk", "shelf")


class InferRequest(BaseModel):
    image_urls: list[str] = Field(..., min_length=1)

    @model_validator(mode="after")
    def validate_sources(self) -> "InferRequest":
        if len(self.image_urls) > 12:
            raise ValueError("Maximum 12 images per request")
        for idx, value in enumerate(self.image_urls):
            if len(value) < 8:
                raise ValueError(f"image_urls[{idx}] is too short")
        return self


class Component(BaseModel):
    name: str
    kind: str
    quantity: int = Field(..., ge=1)


class InferImageResult(BaseModel):
    detected_type: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    suggested_width: float = Field(..., gt=0)
    suggested_height: float = Field(..., gt=0)
    suggested_depth: float = Field(..., gt=0)
    components: list[Component]
    image_url: str = Field(..., min_length=8)


class InferResponse(BaseModel):
    detected_type: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    suggested_width: float = Field(..., gt=0)
    suggested_height: float = Field(..., gt=0)
    suggested_depth: float = Field(..., gt=0)
    components: list[Component]
    image_url: str = Field(..., min_length=8)
    images_analyzed: int = Field(default=1, ge=1)
    image_results: list[InferImageResult]


class YoloDetector:
    def __init__(self, model_path: Path, labels: tuple[str, ...], score_threshold: float) -> None:
        if not model_path.exists():
            raise RuntimeError(f"YOLO model not found at {model_path}")

        try:
            self._session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        except Exception as exc:  # pragma: no cover - provider/runtime error path
            raise RuntimeError(f"Failed to load YOLO model from {model_path}: {exc}") from exc

        model_inputs = self._session.get_inputs()
        if len(model_inputs) != 1:
            raise RuntimeError("YOLO model must expose exactly one input tensor")

        input_tensor = model_inputs[0]
        self.input_name = input_tensor.name
        self.input_height, self.input_width = _resolve_input_size(input_tensor.shape)
        self.labels = labels
        self.model_path = model_path
        self.score_threshold = score_threshold

    def analyze(self, image: Image.Image) -> tuple[str, float, list[tuple[int, float]]]:
        tensor = _preprocess(image, self.input_width, self.input_height)
        outputs = self._session.run(None, {self.input_name: tensor})
        return _decode_prediction(outputs, self.labels, self.score_threshold)

    def classify(self, image: Image.Image) -> tuple[str, float]:
        detected_type, confidence, _ = self.analyze(image)
        return detected_type, confidence


def _detector_model_path() -> Path:
    override = os.getenv("INFERENCE_DETECTOR_ONNX")
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[1] / "models" / "detector.onnx"


def _detector_labels() -> tuple[str, ...]:
    raw = os.getenv("INFERENCE_LABELS")
    if not raw:
        return SUPPORTED_TYPES
    labels = tuple(label.strip() for label in raw.split(",") if label.strip())
    if not labels:
        raise RuntimeError("INFERENCE_LABELS is set but empty")
    return labels


def _resolve_input_size(shape: list[int | str | None]) -> tuple[int, int]:
    if len(shape) != 4:
        raise RuntimeError(f"Unexpected YOLO input tensor rank: {len(shape)}")

    height = shape[2]
    width = shape[3]
    if isinstance(height, int) and isinstance(width, int):
        return height, width

    fallback_size = int(os.getenv("INFERENCE_IMAGE_SIZE", "640"))
    return fallback_size, fallback_size


def _decode_data_url(image_url: str) -> bytes:
    header, payload = image_url.split(",", 1)
    if ";base64" in header:
        return base64.b64decode(payload)
    return unquote_to_bytes(payload)


def _load_image_bytes(image_url: str) -> bytes:
    if image_url.startswith("data:"):
        return _decode_data_url(image_url)

    try:
        response = httpx.get(image_url, follow_redirects=True, timeout=20.0)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail="Failed to fetch input image") from exc
    return response.content


def _open_image(image_bytes: bytes) -> Image.Image:
    try:
        image = Image.open(io.BytesIO(image_bytes))
        image.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=422, detail="Input is not a valid image") from exc
    return ImageOps.exif_transpose(image).convert("RGB")


def _preprocess(image: Image.Image, width: int, height: int) -> np.ndarray:
    resized = image.resize((width, height), Image.Resampling.BILINEAR)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    chw = np.transpose(array, (2, 0, 1))
    return np.expand_dims(chw, axis=0)


def _decode_prediction(
    outputs: list[np.ndarray], labels: tuple[str, ...], threshold: float
) -> tuple[str, float, list[tuple[int, float]]]:
    if not outputs:
        raise HTTPException(status_code=500, detail="YOLO model returned no outputs")

    detections = _extract_detections(outputs[0], len(labels))
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

    # Low confidence should not fail the full workflow; return the best-effort prediction
    # and let downstream logic/UI decide how to handle weak confidence.
    _ = threshold
    return best_category, round(float(best_score), 3), detections


def _extract_detections(output: np.ndarray, num_labels: int) -> list[tuple[int, float]]:
    tensor = np.array(output)
    if tensor.ndim == 3 and tensor.shape[0] == 1:
        tensor = tensor[0]

    if tensor.ndim != 2:
        return []

    # Normalize to rows = detections, cols = features.
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
        # YOLO exports commonly emit either [x,y,w,h,cls...] or [x,y,w,h,obj,cls...].
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


@lru_cache(maxsize=1)
def _detector() -> YoloDetector:
    threshold = float(os.getenv("INFERENCE_CONFIDENCE_THRESHOLD", "0.25"))
    return YoloDetector(
        model_path=_detector_model_path(),
        labels=_detector_labels(),
        score_threshold=threshold,
    )


def _estimate_dimensions(category: str, image: Image.Image) -> tuple[float, float, float]:
    width_px, height_px = image.size
    aspect = width_px / max(height_px, 1)

    if category == "desk":
        suggested_width = 1200 + max(0, (aspect - 1.2) * 220)
        return round(suggested_width, 1), 750.0, 600.0

    if category == "shelf":
        suggested_height = 1650 + max(0, (1.0 / max(aspect, 0.35) - 1.0) * 180)
        return 900.0, round(suggested_height, 1), 300.0

    suggested_height = 1200 + max(0, (1.0 / max(aspect, 0.4) - 1.0) * 120)
    return 800.0, round(suggested_height, 1), 450.0


def _normalize_component_name(label: str) -> str:
    normalized = "_".join(label.strip().lower().replace("-", " ").split())
    return normalized or "component"


def _component_kind(label: str) -> str:
    normalized = label.strip().lower()
    if any(token in normalized for token in ("panel", "door", "shelf", "top", "bottom", "back", "side")):
        return "panel"
    if any(token in normalized for token in ("leg", "support", "brace", "apron", "rail")):
        return "support"
    return "assembly"


def _components_from_detections(
    detections: list[tuple[int, float]],
    labels: tuple[str, ...],
    threshold: float,
    detected_type: str,
) -> list[Component]:
    counts: dict[str, tuple[str, int]] = {}
    for class_id, score in detections:
        if class_id < 0 or class_id >= len(labels) or score < threshold:
            continue
        label = labels[class_id]
        name = _normalize_component_name(label)
        kind = _component_kind(label)
        current = counts.get(name)
        if current is None:
            counts[name] = (kind, 1)
        else:
            counts[name] = (current[0], current[1] + 1)

    if not counts:
        fallback_name = _normalize_component_name(detected_type)
        return [Component(name=fallback_name, kind="assembly", quantity=1)]

    return [
        Component(name=name, kind=kind, quantity=quantity)
        for name, (kind, quantity) in sorted(counts.items())
    ]


def _suggest_components(category: str) -> list[Component]:
    if category == "desk":
        return [
            Component(name="desktop", kind="panel", quantity=1),
            Component(name="left_side_panel", kind="panel", quantity=1),
            Component(name="right_side_panel", kind="panel", quantity=1),
            Component(name="back_panel", kind="panel", quantity=1),
            Component(name="front_apron", kind="support", quantity=1),
        ]

    if category == "shelf":
        return [
            Component(name="top_panel", kind="panel", quantity=1),
            Component(name="bottom_panel", kind="panel", quantity=1),
            Component(name="left_side_panel", kind="panel", quantity=1),
            Component(name="right_side_panel", kind="panel", quantity=1),
            Component(name="shelf_panel", kind="panel", quantity=3),
            Component(name="back_panel", kind="panel", quantity=1),
        ]

    return [
        Component(name="top_panel", kind="panel", quantity=1),
        Component(name="bottom_panel", kind="panel", quantity=1),
        Component(name="left_side_panel", kind="panel", quantity=1),
        Component(name="right_side_panel", kind="panel", quantity=1),
        Component(name="door_panel", kind="panel", quantity=2),
        Component(name="shelf_panel", kind="panel", quantity=3),
        Component(name="back_panel", kind="panel", quantity=1),
    ]


def _classify(image: Image.Image) -> tuple[str, float]:
    return _detector().classify(image)


def _infer_single(image_url: str) -> InferImageResult:
    image_bytes = _load_image_bytes(image_url)
    image = _open_image(image_bytes)
    detector = _detector()
    if hasattr(detector, "analyze"):
        detected_type, confidence, detections = detector.analyze(image)
        labels = getattr(detector, "labels", SUPPORTED_TYPES)
        threshold = float(getattr(detector, "score_threshold", 0.25))
        components = _components_from_detections(
            detections,
            labels,
            threshold,
            detected_type,
        )
    else:
        detected_type, confidence = detector.classify(image)
        components = _suggest_components(detected_type)
    suggested_width, suggested_height, suggested_depth = _estimate_dimensions(detected_type, image)
    return InferImageResult(
        detected_type=detected_type,
        confidence=confidence,
        suggested_width=suggested_width,
        suggested_height=suggested_height,
        suggested_depth=suggested_depth,
        components=components,
        image_url=image_url,
    )


def _aggregate_components(results: list[InferImageResult]) -> list[Component]:
    merged: dict[str, Component] = {}
    for result in results:
        for component in result.components:
            current = merged.get(component.name)
            if current is None or component.quantity > current.quantity:
                merged[component.name] = Component(
                    name=component.name,
                    kind=component.kind,
                    quantity=component.quantity,
                )
    return [merged[key] for key in sorted(merged)]


def _aggregate_results(results: list[InferImageResult]) -> InferResponse:
    scores_by_type: dict[str, float] = {}
    for result in results:
        scores_by_type[result.detected_type] = scores_by_type.get(result.detected_type, 0.0) + result.confidence

    detected_type = max(scores_by_type, key=scores_by_type.get)
    matching = [result for result in results if result.detected_type == detected_type]
    confidence_source = matching if matching else results

    return InferResponse(
        detected_type=detected_type,
        confidence=round(mean(item.confidence for item in confidence_source), 3),
        suggested_width=round(mean(item.suggested_width for item in results), 1),
        suggested_height=round(mean(item.suggested_height for item in results), 1),
        suggested_depth=round(mean(item.suggested_depth for item in results), 1),
        components=_aggregate_components(results),
        image_url=results[0].image_url,
        images_analyzed=len(results),
        image_results=results,
    )


@app.get("/health")
def health() -> dict:
    try:
        detector = _detector()
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {
        "status": "ok",
        "model_path": str(detector.model_path),
        "labels": list(detector.labels),
        "confidence_threshold": detector.score_threshold,
    }


@app.post("/infer", response_model=InferResponse)
def infer(payload: InferRequest) -> InferResponse:
    results = [_infer_single(image_url=url) for url in payload.image_urls]
    return _aggregate_results(results)
