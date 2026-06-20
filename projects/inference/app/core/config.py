from __future__ import annotations

import os
from pathlib import Path

import yaml

SUPPORTED_TYPES = ("cabinet", "desk", "shelf")


def _taxonomy_path() -> Path:
    return Path(__file__).resolve().parent / "furniture_taxonomy.yaml"


def _load_taxonomy_file() -> dict:
    path = _taxonomy_path()
    if not path.exists():
        raise RuntimeError(f"Taxonomy file not found: {path}")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Could not parse taxonomy YAML at {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise RuntimeError(f"Taxonomy YAML root must be a mapping: {path}")
    return data


_taxonomy = _load_taxonomy_file()


def _as_aliases(value: object) -> dict[str, str]:
    if not isinstance(value, dict):
        raise RuntimeError("taxonomy.product_type_aliases must be a mapping")
    out: dict[str, str] = {}
    for key, mapped in value.items():
        out[str(key)] = str(mapped)
    if not out:
        raise RuntimeError("taxonomy.product_type_aliases cannot be empty")
    return out


def _as_hints(value: object) -> dict[str, tuple[str, ...]]:
    if not isinstance(value, dict):
        raise RuntimeError("taxonomy.product_type_component_hints must be a mapping")
    out: dict[str, tuple[str, ...]] = {}
    for product_type, hints in value.items():
        if not isinstance(hints, (list, tuple)):
            continue
        out[str(product_type)] = tuple(str(item) for item in hints if str(item))
    if not out:
        raise RuntimeError("taxonomy.product_type_component_hints cannot be empty")
    return out


def _as_minimums(value: object) -> dict[str, dict[str, int]]:
    if not isinstance(value, dict):
        raise RuntimeError("taxonomy.product_type_component_minimums must be a mapping")
    out: dict[str, dict[str, int]] = {}
    for product_type, minimums in value.items():
        if not isinstance(minimums, dict):
            continue
        item: dict[str, int] = {}
        for component_name, quantity in minimums.items():
            try:
                item[str(component_name)] = max(1, int(quantity))
            except Exception:
                continue
        if item:
            out[str(product_type)] = item
    if not out:
        raise RuntimeError("taxonomy.product_type_component_minimums cannot be empty")
    return out


def _as_confidence_multipliers(value: object) -> dict[str, float]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise RuntimeError("taxonomy.product_type_confidence_multipliers must be a mapping")
    out: dict[str, float] = {}
    for product_type, raw in value.items():
        try:
            parsed = float(raw)
        except Exception:
            continue
        out[str(product_type)] = min(2.0, max(0.5, parsed))
    return out


# Extensible taxonomy for project-level furniture categorization.
PRODUCT_TYPE_ALIASES = _as_aliases(_taxonomy.get("product_type_aliases"))

# Product signatures used to infer product type from detected components.
PRODUCT_TYPE_COMPONENT_HINTS = _as_hints(_taxonomy.get("product_type_component_hints"))

# Minimal component priors for sparse detections.
PRODUCT_TYPE_COMPONENT_MINIMUMS = _as_minimums(_taxonomy.get("product_type_component_minimums"))

# Per-product threshold multiplier for confidence fallback logic.
PRODUCT_TYPE_CONFIDENCE_MULTIPLIERS = _as_confidence_multipliers(
    _taxonomy.get("product_type_confidence_multipliers")
)

KNOWN_INTERIOR_PARTS = {"shelf_panel", "drawer_box", "divider_panel", "telescopic_slide", "rail", "hinge"}
HARDWARE_PARTS = {"telescopic_slide", "hinge", "handle_pull", "bracket", "rail", "sliding_door_track"}

COMPONENT_ALIAS_EXACT = {
    "cabinet": "cabinet_body",
    "wardrobe": "cabinet_body",
    "closet": "cabinet_body",
    "armoire": "cabinet_body",
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
    ("rod", "hanger_rod"),
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
