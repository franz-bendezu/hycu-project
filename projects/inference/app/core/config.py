from __future__ import annotations

import os
from pathlib import Path

SUPPORTED_TYPES = ("cabinet", "desk", "shelf")

KNOWN_INTERIOR_PARTS = {"shelf_panel", "drawer_box", "divider_panel", "telescopic_slide", "rail", "hinge"}
HARDWARE_PARTS = {"telescopic_slide", "hinge", "handle_pull", "bracket", "rail", "sliding_door_track"}

COMPONENT_ALIAS_EXACT = {
    "cabinet": "cabinet_body",
    "wardrobe": "cabinet_body",
    "closet": "cabinet_body",
    "desk": "desk_frame",
    "table": "top_panel",
    "shelf": "shelf_panel",
    "bookcase": "shelf_panel",
}

COMPONENT_ALIAS_CONTAINS = (
    ("door", "door_panel"),
    ("drawer_front", "drawer_front"),
    ("drawer", "drawer_box"),
    ("side", "side_panel"),
    ("back", "back_panel"),
    ("bottom", "bottom_panel"),
    ("top", "top_panel"),
    ("shelf", "shelf_panel"),
    ("divider", "divider_panel"),
    ("leg", "leg"),
    ("handle", "handle_pull"),
    ("hinge", "hinge"),
    ("slide", "telescopic_slide"),
    ("rail", "rail"),
    ("track", "sliding_door_track"),
    ("apron", "front_apron"),
)

def get_detector_model_path() -> Path:
    override = os.getenv("INFERENCE_DETECTOR_ONNX")
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[2] / "models" / "detector.onnx"

def get_detector_labels() -> tuple[str, ...]:
    raw = os.getenv("INFERENCE_LABELS")
    if not raw:
        # Default labels for the components profile
        return (
            "cabinet", "desk", "shelf", "cabinet_body", "desk_frame",
            "side_panel", "top_panel", "bottom_panel", "back_panel",
            "door_panel", "shelf_panel", "divider_panel", "drawer_front",
            "drawer_box", "leg", "front_apron", "handle_pull", "hinge",
            "telescopic_slide", "rail", "sliding_door_track"
        )
    labels = tuple(label.strip() for label in raw.split(",") if label.strip())
    if not labels:
        raise RuntimeError("INFERENCE_LABELS is set but empty")
    return labels

def get_confidence_threshold() -> float:
    # Lowered default for prototype stage to see more potential detections
    return float(os.getenv("INFERENCE_CONFIDENCE_THRESHOLD", "0.10"))

def get_image_size_fallback() -> int:
    return int(os.getenv("INFERENCE_IMAGE_SIZE", "640"))

def get_execution_providers() -> list[str]:
    raw = os.getenv("INFERENCE_PROVIDERS")
    if not raw:
        # Default to trying CUDA then falling back to CPU
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return [p.strip() for p in raw.split(",") if p.strip()]
