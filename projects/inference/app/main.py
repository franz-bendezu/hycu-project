from __future__ import annotations

from collections import Counter
from contextlib import asynccontextmanager
from functools import lru_cache
from statistics import mean

from fastapi import FastAPI, HTTPException
from PIL import Image

from app.core.config import (
    get_detector_model_path,
    get_detector_labels,
    get_confidence_threshold,
    get_sam2_model_path,
    get_segmentation_backend,
)
from app.schemas import (
    BenchmarkItemSummary,
    BenchmarkRequest,
    BenchmarkResponse,
    InferRequest,
    InferResponse,
)
from app.services.detector import YoloDetector
from app.services.segmenter import Segmenter
from app.services.refiner import HeavyRefiner
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
    return Segmenter(
        model_path=get_sam2_model_path(),
        backend=get_segmentation_backend(),
    )


@lru_cache(maxsize=1)
def _refiner() -> HeavyRefiner:
    return HeavyRefiner()


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
        segmenter = _segmenter()
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {
        "status": "ok",
        "model_path": str(detector.model_path),
        "labels": list(detector.labels),
        "confidence_threshold": detector.score_threshold,
        "active_providers": detector.active_providers,
        "segmentation_backend": segmenter.active_backend,
    }


def _infer_once(image_urls: list[str]) -> InferResponse:
    from app.services.processor import assemble_project

    images_with_urls = load_all_images(image_urls)

    # Stage 1: Object Detection
    detector = _detector()
    detection_result = detector.analyze(images_with_urls)

    # Stage 2: Multi-view tracking + segmentation
    segmenter = _segmenter()
    tracker = MultiViewTracker(iou_threshold=0.35, max_age=2)
    for idx, evidence in enumerate(detection_result.evidence):
        image, _ = images_with_urls[idx]

        normalized_detections: list[dict] = []
        for detection in evidence.raw_detections:
            box = detection.get("box")
            if isinstance(box, (tuple, list)) and len(box) == 4:
                enriched = dict(detection)
                enriched["view_index"] = idx
                normalized_detections.append(enriched)

        tracked = tracker.update(normalized_detections)
        boxes = [tuple(item["box"]) for item in tracked if isinstance(item.get("box"), (list, tuple))]
        masks = segmenter.predict(image, boxes) if boxes else []

        for det_idx, item in enumerate(tracked):
            item.setdefault("image_width_px", image.width)
            item.setdefault("image_height_px", image.height)
            if det_idx < len(masks):
                mask_area = int(masks[det_idx].sum())
                item["mask_area_px"] = mask_area
                item["mask_fill_ratio"] = round(mask_area / max(image.width * image.height, 1), 5)

        evidence.raw_detections = tracked

    response = assemble_project(detection_result.evidence, detector.labels, detector.score_threshold)
    response = _refiner().maybe_refine(response, detection_result.evidence)
    return response


@app.post("/infer", response_model=InferResponse)
def infer(payload: InferRequest) -> InferResponse:
    return _infer_once(payload.image_urls)


@app.post("/benchmark", response_model=BenchmarkResponse)
def benchmark(payload: BenchmarkRequest) -> BenchmarkResponse:
    item_results: list[BenchmarkItemSummary] = []
    strategy_counts: Counter[str] = Counter()

    for idx, item in enumerate(payload.items, start=1):
        response = _infer_once(item.image_urls)

        component_coverage = float(response.validation_metrics.get("component_coverage", 0.0))
        physical_validity = float(response.validation_metrics.get("physical_validity_score", 0.0))
        escalation_strategy = str(response.escalation.get("strategy", "fast_2d_fusion"))
        human_review = bool(response.escalation.get("human_review_required", False))

        item_results.append(
            BenchmarkItemSummary(
                item_id=item.item_id or f"item_{idx}",
                detected_type=response.detected_type,
                confidence=response.confidence,
                component_coverage=component_coverage,
                physical_validity_score=physical_validity,
                escalation_strategy=escalation_strategy,
                human_review_required=human_review,
            )
        )
        strategy_counts[escalation_strategy] += 1

    total_items = len(item_results)
    human_review_count = sum(1 for item in item_results if item.human_review_required)
    return BenchmarkResponse(
        items_analyzed=total_items,
        avg_confidence=round(mean(item.confidence for item in item_results), 4),
        avg_component_coverage=round(mean(item.component_coverage for item in item_results), 4),
        avg_physical_validity=round(mean(item.physical_validity_score for item in item_results), 4),
        human_review_rate=round(human_review_count / max(total_items, 1), 4),
        escalation_strategy_counts=dict(strategy_counts),
        item_results=item_results,
    )
