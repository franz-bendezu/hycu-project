from __future__ import annotations

import base64
import io
from urllib.parse import unquote_to_bytes

import httpx
import numpy as np
from fastapi import HTTPException
from PIL import Image, ImageOps, UnidentifiedImageError

def decode_data_url(image_url: str) -> bytes:
    header, payload = image_url.split(",", 1)
    if ";base64" in header:
        return base64.b64decode(payload)
    return unquote_to_bytes(payload)


def load_image_bytes(image_url: str) -> bytes:
    if image_url.startswith("data:"):
        return decode_data_url(image_url)

    try:
        response = httpx.get(image_url, follow_redirects=True, timeout=20.0)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail="Failed to fetch input image") from exc
    return response.content


def open_image(image_bytes: bytes) -> Image.Image:
    try:
        image = Image.open(io.BytesIO(image_bytes))
        image.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=422, detail="Input is not a valid image") from exc
    return ImageOps.exif_transpose(image).convert("RGB")


def preprocess(img: Image.Image, w: int, h: int) -> np.ndarray:
    """Pre-process image for YOLOv8/11 inference."""
    # Resize directly to model input size to keep coordinate mapping deterministic.
    img = img.resize((w, h), Image.Resampling.BILINEAR)

    img_data = np.array(img, dtype=np.float32) / 255.0
    if len(img_data.shape) == 2:  # Grayscale
        img_data = np.stack([img_data] * 3, axis=-1)
    img_data = np.transpose(img_data, (2, 0, 1))
    return np.expand_dims(img_data, axis=0)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
    """
    Performs Non-Maximum Suppression (NMS) on bounding boxes.
    """
    if boxes.size == 0:
        return []

    # Sort by score
    sorted_indices = np.argsort(scores)[::-1]

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)

    keep: list[int] = []
    while sorted_indices.size > 0:
        current_idx = sorted_indices[0]
        keep.append(current_idx)

        remaining_indices = sorted_indices[1:]
        if remaining_indices.size == 0:
            break

        xx1 = np.maximum(x1[current_idx], x1[remaining_indices])
        yy1 = np.maximum(y1[current_idx], y1[remaining_indices])
        xx2 = np.minimum(x2[current_idx], x2[remaining_indices])
        yy2 = np.minimum(y2[current_idx], y2[remaining_indices])

        intersection = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        union = areas[current_idx] + areas[remaining_indices] - intersection
        iou = intersection / (union + 1e-6)

        low_iou_indices = np.where(iou < iou_threshold)[0]
        sorted_indices = remaining_indices[low_iou_indices]

    return keep
