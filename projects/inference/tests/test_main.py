import base64
import io

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from PIL import Image

from app.main import _classify, app


def _solid_image(width: int, height: int, color: tuple[int, int, int]) -> Image.Image:
    return Image.new("RGB", (width, height), color=color)


def _png_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


class _DetectorStub:
    def __init__(self, category: str, confidence: float) -> None:
        self.category = category
        self.confidence = confidence

    def classify(self, image: Image.Image) -> tuple[str, float]:
        _ = image
        return self.category, self.confidence


class _LowConfidenceDetectorStub:
    def classify(self, image: Image.Image) -> tuple[str, float]:
        _ = image
        return "desk", 0.12


def test_classify_uses_yolo_detector(monkeypatch) -> None:
    query = _solid_image(320, 560, (230, 230, 230))
    monkeypatch.setattr("app.main._detector", lambda: _DetectorStub("shelf", 0.91))

    detected_type, confidence = _classify(query)

    assert detected_type == "shelf"
    assert confidence == 0.91


def test_classify_returns_supported_category(monkeypatch) -> None:
    query = _solid_image(640, 380, (130, 120, 110))
    monkeypatch.setattr("app.main._detector", lambda: _DetectorStub("desk", 0.77))

    detected_type, confidence = _classify(query)

    assert detected_type in {"desk", "cabinet", "shelf"}
    assert 0.0 <= confidence <= 1.0


def test_startup_fails_when_detector_unavailable(monkeypatch) -> None:
    def _raise_runtime_error() -> _DetectorStub:
        raise RuntimeError("YOLO model not found")

    monkeypatch.setattr("app.main._detector", _raise_runtime_error)

    with pytest.raises(RuntimeError, match="YOLO model not found"):
        with TestClient(app):
            pass


def test_infer_returns_low_confidence_prediction(monkeypatch) -> None:
    payload = {"image_urls": [_png_data_url(_solid_image(320, 240, (120, 120, 120)))]}
    monkeypatch.setattr("app.main._detector", lambda: _LowConfidenceDetectorStub())

    with TestClient(app) as client:
        response = client.post("/infer", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["detected_type"] == "desk"
    assert body["confidence"] == 0.12
    assert body["images_analyzed"] == 1
    assert len(body["image_results"]) == 1


class _SequentialDetectorStub:
    def __init__(self, outputs: list[tuple[str, float]]) -> None:
        self._outputs = outputs
        self._idx = 0

    def classify(self, image: Image.Image) -> tuple[str, float]:
        _ = image
        value = self._outputs[self._idx % len(self._outputs)]
        self._idx += 1
        return value


def test_infer_accepts_batch_images_and_returns_aggregate(monkeypatch) -> None:
    image_a = _png_data_url(_solid_image(320, 240, (120, 120, 120)))
    image_b = _png_data_url(_solid_image(320, 240, (180, 180, 180)))
    monkeypatch.setattr(
        "app.main._detector",
        lambda: _SequentialDetectorStub([("desk", 0.7), ("cabinet", 0.9)]),
    )

    with TestClient(app) as client:
        response = client.post("/infer", json={"image_urls": [image_a, image_b]})

    assert response.status_code == 200
    body = response.json()
    assert body["images_analyzed"] == 2
    assert isinstance(body["image_results"], list)
    assert len(body["image_results"]) == 2
