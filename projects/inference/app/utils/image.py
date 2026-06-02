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


def preprocess(image: Image.Image, width: int, height: int) -> np.ndarray:
    resized = image.resize((width, height), Image.Resampling.BILINEAR)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    chw = np.transpose(array, (2, 0, 1))
    return np.expand_dims(chw, axis=0)
