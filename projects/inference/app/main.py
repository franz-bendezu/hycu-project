from __future__ import annotations

from contextlib import asynccontextmanager
from functools import lru_cache

from fastapi import FastAPI, HTTPException
from PIL import Image

from app.core.config import (
    get_detector_model_path,
    get_detector_labels,
    get_confidence_threshold,
)
from app.schemas import (
    InferRequest,
    InferResponse,
)
from app.services.detector import YoloDetector
from app.services.segmenter import Segmenter
from app.services.tracker import MultiViewTracker
from app.utils.image import load_image_bytes, open_image


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Fail fast during service boot when artifacts are missing or invalid.
    _detector()
    _segmenter()
    yield


app = FastAPI(title="Vision Inference Service", version="0.1.0", lifespan=lifespan)


@lru_cache(maxsize=1)
def _detector() -> YoloDetector:
    return YoloDetector(
        model_path=get_detector_model_path(),
        labels=get_detector_labels(),
        score_threshold=get_confidence_threshold(),
    )


@lru_cache(maxsize=1)
def _segmenter() -> Segmenter:
    # Placeholder for model path configuration
    model_path = "path/to/sam2/model.pth"
    return Segmenter(model_path=model_path)


@lru_cache(maxsize=1)
def _tracker() -> MultiViewTracker:
    return MultiViewTracker(iou_threshold=0.35, max_age=2)


def load_all_images(image_urls: list[str]) -> list[tuple[Image.Image, str]]:
    results = []
    for url in image_urls:
        try:
            image_bytes = load_image_bytes(url)
            image = open_image(image_bytes)
            results.append((image, url))
        except Exception as exc:
            raise HTTPException(
                status_code=400, detail=f"Could not load image from {url}: {str(exc)}"
            )
    return results


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
        "active_providers": detector.active_providers,
    }


@app.post("/infer", response_model=InferResponse)
def infer(payload: InferRequest) -> InferResponse:
    images_with_urls = load_all_images(payload.image_urls)

    # Stage 1: Object Detection
    detector = _detector()
    detection_result = detector.analyze(images_with_urls)

    # Stage 2: Multi-view tracking + segmentation
    segmenter = _segmenter()
    tracker = _tracker()
    for idx, evidence in enumerate(detection_result.evidence):
        image, _ = images_with_urls[idx]

        normalized_detections: list[dict] = []
        for detection in evidence.raw_detections:
            box = detection.get("box")
            if isinstance(box, (tuple, list)) and len(box) == 4:
                normalized_detections.append(detection)

        tracked = tracker.update(normalized_detections)
        boxes = [tuple(item["box"]) for item in tracked if isinstance(item.get("box"), (list, tuple))]
        masks = segmenter.predict(image, boxes) if boxes else []

        for det_idx, item in enumerate(tracked):
            if det_idx < len(masks):
                item["mask_area_px"] = int(masks[det_idx].sum())

        evidence.raw_detections = tracked

    return detection_result
