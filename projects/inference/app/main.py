from __future__ import annotations

from contextlib import asynccontextmanager
from functools import lru_cache

from fastapi import FastAPI, HTTPException

from app.core.config import (
    get_detector_model_path,
    get_detector_labels,
    get_confidence_threshold,
)
from app.schemas import (
    InferRequest,
    InferResponse,
    InferImageResult,
    InteriorAssessment,
    DoorAssessment,
    UncertaintyAssessment,
)
from app.services.detector import YoloDetector
from app.services.processor import (
    estimate_dimensions,
    fallback_type_from_aspect,
    components_from_detections,
    split_components,
    interior_visibility,
    door_metadata,
    has_uncertain_hardware,
    build_joints,
    build_hardware,
    component_index,
    aggregate_results,
)
from app.utils.image import load_image_bytes, open_image


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Fail fast during service boot when the detector artifact is missing or invalid.
    _detector()
    yield


app = FastAPI(title="Vision Inference Service", version="0.1.0", lifespan=lifespan)


@lru_cache(maxsize=1)
def _detector() -> YoloDetector:
    return YoloDetector(
        model_path=get_detector_model_path(),
        labels=get_detector_labels(),
        score_threshold=get_confidence_threshold(),
    )


def _infer_single(image_url: str) -> InferImageResult:
    image_bytes = load_image_bytes(image_url)
    image = open_image(image_bytes)
    detector = _detector()
    detected_type, confidence, detections = detector.analyze(image)
    
    if confidence < 0.2:
        detected_type = fallback_type_from_aspect(image)
        
    components = components_from_detections(
        detections,
        detector.labels,
        detector.score_threshold,
        detected_type,
    )
    structural_components, hardware_components = split_components(components)
    visibility, coverage_ratio, unknown_interior = interior_visibility(
        structural_components,
        hardware_components,
    )
    door_type, door_count_uncertain = door_metadata(structural_components, hardware_components, visibility)
    uncertain_hardware = has_uncertain_hardware(detections, detector.labels, detector.score_threshold)
    joints = build_joints(structural_components, hardware_components, door_type)
    hardware = build_hardware(joints, uncertain_hardware)
    suggested_width, suggested_height, suggested_depth = estimate_dimensions(detected_type, image)
    index = component_index(components)
    
    return InferImageResult(
        detected_type=detected_type,
        confidence=confidence,
        suggested_width=suggested_width,
        suggested_height=suggested_height,
        suggested_depth=suggested_depth,
        components=components,
        component_index=index,
        interior=InteriorAssessment(
            visibility=visibility,
            coverage_ratio=coverage_ratio,
            unknown_interior=unknown_interior,
        ),
        door=DoorAssessment(type=door_type, count_uncertain=door_count_uncertain),
        uncertainty=UncertaintyAssessment(hardware_uncertain=uncertain_hardware),
        joints=joints,
        hardware=hardware,
        image_url=image_url,
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
        "active_providers": detector.active_providers,
    }


@app.post("/infer", response_model=InferResponse)
def infer(payload: InferRequest) -> InferResponse:
    results = [_infer_single(image_url=url) for url in payload.image_urls]
    return aggregate_results(results)
