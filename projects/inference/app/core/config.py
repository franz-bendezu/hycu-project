from __future__ import annotations

import os
from pathlib import Path

import yaml

from app.schemas import SegmentationBackend

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


def _resolve_component_dataset_yaml_path() -> Path | None:
    configured = os.getenv("INFERENCE_COMPONENT_DATASET_YAML", "").strip()
    if configured:
        return Path(configured)

    default_path = (
        Path(__file__).resolve().parents[3]
        / "training"
        / "datasets"
        / "yolo_dataset_components_labeled.yaml"
    )
    return default_path


def _resolve_dataset_root(dataset_yaml: Path, yaml_doc: dict) -> Path:
    dataset_root = yaml_doc.get("path")
    if not isinstance(dataset_root, str) or not dataset_root.strip():
        raise RuntimeError("Component dataset YAML must define a non-empty 'path'")

    root = Path(dataset_root)
    if root.is_absolute():
        return root

    candidates = [
        (dataset_yaml.parent / root).resolve(),
        (dataset_yaml.parents[1] / root).resolve(),
        (Path.cwd() / root).resolve(),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _component_supervision_available() -> bool:
    dataset_yaml = _resolve_component_dataset_yaml_path()
    if dataset_yaml is None or not dataset_yaml.exists():
        return True

    try:
        yaml_doc = yaml.safe_load(dataset_yaml.read_text(encoding="utf-8"))
        if not isinstance(yaml_doc, dict):
            return True

        dataset_root = _resolve_dataset_root(dataset_yaml, yaml_doc)
        labels_root = dataset_root / "labels"
        if not labels_root.exists():
            return True

        for split in ("train", "val"):
            split_dir = labels_root / split
            if not split_dir.exists():
                continue
            for label_file in split_dir.glob("*.txt"):
                text = label_file.read_text(encoding="utf-8").strip()
                if not text:
                    continue
                for line in text.splitlines():
                    parts = line.strip().split()
                    if not parts:
                        continue
                    try:
                        class_id = int(parts[0])
                    except ValueError:
                        continue
                    if class_id >= 3:
                        return True
    except Exception:
        # If dataset probing fails, avoid blocking component inference.
        return True

    return False


COMPONENT_SUPERVISION_AVAILABLE = _component_supervision_available()


def _as_bool(raw: str | None, default: bool = False) -> bool:
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _resolve_latest_training_onnx() -> Path | None:
    training_root = Path(__file__).resolve().parents[3] / "training"
    latest_pointer = training_root / "runs" / "train" / "LATEST_RUN.txt"

    # First try the explicit latest-run pointer when it exists.
    if latest_pointer.exists():
        try:
            pointed = latest_pointer.read_text(encoding="utf-8").strip()
            if pointed:
                run_dir = Path(pointed)
                if not run_dir.is_absolute():
                    run_dir = (latest_pointer.parent / pointed).resolve()
                candidate = run_dir / "weights" / "best.onnx"
                if candidate.exists():
                    return candidate
        except Exception:
            pass

    # Fallback to scanning for newest exported ONNX in training runs.
    runs_root = training_root / "runs" / "train"
    if not runs_root.exists():
        return None

    candidates = list(runs_root.glob("*/weights/best.onnx"))
    if not candidates:
        return None

    try:
        return max(candidates, key=lambda path: path.stat().st_mtime)
    except Exception:
        return None

def get_detector_model_path() -> Path:
    override = os.getenv("INFERENCE_DETECTOR_ONNX")
    if override:
        return Path(override)

    bundled = Path(__file__).resolve().parents[2] / "models" / "detector.onnx"
    prefer_latest = _as_bool(os.getenv("INFERENCE_PREFER_LATEST_TRAINING_ONNX"), default=True)
    if not prefer_latest:
        return bundled

    latest_training = _resolve_latest_training_onnx()
    if latest_training is None:
        return bundled

    if not bundled.exists():
        return latest_training

    try:
        if latest_training.stat().st_mtime > bundled.stat().st_mtime:
            return latest_training
    except Exception:
        # If stat fails for any reason, keep deterministic bundled fallback.
        return bundled

    return bundled

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
        requested = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        requested = [p.strip() for p in raw.split(",") if p.strip()]

    try:
        import onnxruntime as ort

        available = set(ort.get_available_providers())
        filtered = [provider for provider in requested if provider in available]
        if filtered:
            return filtered
        if "CPUExecutionProvider" in available:
            return ["CPUExecutionProvider"]
        if available:
            return [next(iter(available))]
    except Exception:
        # Keep requested order when provider discovery is unavailable.
        pass

    return requested


def get_segmentation_backend() -> SegmentationBackend:
    raw = os.getenv("INFERENCE_SEGMENTATION_BACKEND", SegmentationBackend.SAM2.value)
    try:
        return SegmentationBackend(raw.strip().lower())
    except ValueError:
        return SegmentationBackend.SAM2


def get_sam2_model_path() -> Path | None:
    raw = os.getenv("INFERENCE_SAM2_MODEL_PATH", "").strip()
    if raw:
        return Path(raw)

    bundled = Path(__file__).resolve().parents[2] / "models" / "sam2_hiera_tiny.pt"
    if bundled.exists():
        return bundled

    return None


def get_escalation_geometry_threshold() -> float:
    raw = os.getenv("INFERENCE_ESCALATE_GEOMETRY_THRESHOLD", "0.45")
    try:
        value = float(raw)
    except ValueError:
        value = 0.45
    return min(1.0, max(0.0, value))


def get_escalation_mvs_threshold() -> float:
    raw = os.getenv("INFERENCE_ESCALATE_MVS_THRESHOLD", "0.65")
    try:
        value = float(raw)
    except ValueError:
        value = 0.65
    return min(1.0, max(0.0, value))


def get_enable_heavy_refinement() -> bool:
    raw = os.getenv("INFERENCE_ENABLE_HEAVY_REFINEMENT", "false").strip().lower()
    return raw in {"1", "true", "yes", "on"}
