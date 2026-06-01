from __future__ import annotations

import base64
import io
import os
from dataclasses import dataclass
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from urllib.parse import unquote_to_bytes

import httpx
from fastapi import FastAPI, HTTPException
from PIL import Image, ImageFilter, ImageOps, ImageStat, UnidentifiedImageError
from pydantic import BaseModel, Field

app = FastAPI(title="Vision Inference Service", version="0.1.0")

SUPPORTED_TYPES = ("cabinet", "desk", "shelf")
HASH_SIZE = 16


class InferRequest(BaseModel):
    image_url: str = Field(..., min_length=8)


class InferResponse(BaseModel):
    detected_type: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    suggested_width: float = Field(..., gt=0)
    suggested_height: float = Field(..., gt=0)
    suggested_depth: float = Field(..., gt=0)
    image_url: str = Field(..., min_length=8)


@dataclass(frozen=True)
class ReferenceImage:
    category: str
    average_hash: int
    difference_hash: int
    edge_density: float
    aspect_ratio: float
    brightness: float


def _gallery_root() -> Path:
    override = os.getenv("INFERENCE_GALLERY_ROOT")
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[2] / "training" / "datasets" / "raw" / "images"


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


def _average_hash(image: Image.Image) -> int:
    grayscale = image.convert("L").resize((HASH_SIZE, HASH_SIZE), Image.Resampling.LANCZOS)
    pixels = list(grayscale.getdata())
    mean = sum(pixels) / len(pixels)
    value = 0
    for pixel in pixels:
        value = (value << 1) | int(pixel >= mean)
    return value


def _difference_hash(image: Image.Image) -> int:
    grayscale = image.convert("L").resize((HASH_SIZE + 1, HASH_SIZE), Image.Resampling.LANCZOS)
    pixels = list(grayscale.getdata())
    value = 0
    for row in range(HASH_SIZE):
        row_offset = row * (HASH_SIZE + 1)
        for col in range(HASH_SIZE):
            left = pixels[row_offset + col]
            right = pixels[row_offset + col + 1]
            value = (value << 1) | int(left >= right)
    return value


def _hamming_distance(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def _edge_density(image: Image.Image) -> float:
    edges = image.convert("L").filter(ImageFilter.FIND_EDGES)
    pixels = list(edges.getdata())
    if not pixels:
        return 0.0
    strong_edges = sum(1 for value in pixels if value >= 48)
    return strong_edges / len(pixels)


def _image_features(image: Image.Image) -> tuple[int, int, float, float, float]:
    width_px, height_px = image.size
    aspect_ratio = width_px / max(height_px, 1)
    brightness = ImageStat.Stat(image.convert("L")).mean[0] / 255.0
    return (
        _average_hash(image),
        _difference_hash(image),
        _edge_density(image),
        aspect_ratio,
        brightness,
    )


def _distance(query: tuple[int, int, float, float, float], ref: ReferenceImage) -> float:
    query_ahash, query_dhash, query_edge, query_aspect, query_brightness = query
    max_distance = HASH_SIZE * HASH_SIZE
    ahash_distance = _hamming_distance(query_ahash, ref.average_hash) / max_distance
    dhash_distance = _hamming_distance(query_dhash, ref.difference_hash) / max_distance
    edge_distance = abs(query_edge - ref.edge_density)
    aspect_distance = min(1.0, abs(query_aspect - ref.aspect_ratio) / 1.8)
    brightness_distance = abs(query_brightness - ref.brightness)

    return (
        ahash_distance * 0.42
        + dhash_distance * 0.34
        + edge_distance * 0.12
        + aspect_distance * 0.08
        + brightness_distance * 0.04
    )


@lru_cache(maxsize=1)
def _reference_gallery() -> tuple[ReferenceImage, ...]:
    root = _gallery_root()
    references: list[ReferenceImage] = []

    for category in SUPPORTED_TYPES:
        folder = root / category
        if not folder.exists():
            continue
        for image_path in sorted(folder.iterdir()):
            if not image_path.is_file():
                continue
            try:
                with Image.open(image_path) as sample:
                    normalized = ImageOps.exif_transpose(sample).convert("RGB")
                    average_hash, difference_hash, edge_density, aspect_ratio, brightness = _image_features(normalized)
            except (UnidentifiedImageError, OSError):
                continue
            references.append(
                ReferenceImage(
                    category=category,
                    average_hash=average_hash,
                    difference_hash=difference_hash,
                    edge_density=edge_density,
                    aspect_ratio=aspect_ratio,
                    brightness=brightness,
                )
            )

    if not references:
        raise RuntimeError(f"No reference images found under {root}")

    return tuple(references)


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


def _classify(image: Image.Image) -> tuple[str, float]:
    query = _image_features(image)
    scores_by_category: dict[str, list[float]] = defaultdict(list)

    for reference in _reference_gallery():
        scores_by_category[reference.category].append(_distance(query, reference))

    aggregated = []
    for category, scores in scores_by_category.items():
        top_scores = sorted(scores)[: min(3, len(scores))]
        aggregated.append((sum(top_scores) / len(top_scores), category))

    aggregated.sort(key=lambda item: item[0])
    best_score, best_category = aggregated[0]
    second_score = aggregated[1][0] if len(aggregated) > 1 else best_score

    separation_bonus = max(0.0, second_score - best_score)
    confidence = 1.0 - best_score * 0.9 + separation_bonus * 0.4
    confidence = max(0.2, min(0.98, confidence))
    return best_category, round(confidence, 3)


@app.get("/health")
def health() -> dict:
    try:
        gallery_size = len(_reference_gallery())
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {"status": "ok", "reference_images": gallery_size}


@app.post("/infer", response_model=InferResponse)
def infer(payload: InferRequest) -> InferResponse:
    image_bytes = _load_image_bytes(payload.image_url)
    image = _open_image(image_bytes)
    detected_type, confidence = _classify(image)
    suggested_width, suggested_height, suggested_depth = _estimate_dimensions(detected_type, image)

    return InferResponse(
        detected_type=detected_type,
        confidence=confidence,
        suggested_width=suggested_width,
        suggested_height=suggested_height,
        suggested_depth=suggested_depth,
        image_url=payload.image_url,
    )
