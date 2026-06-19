import base64
import io
import uuid

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app
from app.schemas import ImageEvidence
from app.services.processor import assemble_project


def _solid_image(width: int, height: int, color: tuple[int, int, int]) -> Image.Image:
    return Image.new("RGB", (width, height), color=color)


def _png_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


class _AnalyzeDetectorStub:
    def __init__(
        self,
        outputs: list[tuple[str, float, list[dict]]],
        labels: tuple[str, ...],
        score_threshold: float,
    ) -> None:
        self._outputs = outputs
        self.labels = labels
        self.score_threshold = score_threshold

    def analyze(self, images_with_urls: list[tuple[Image.Image, str]]):
        evidence: list[ImageEvidence] = []
        for idx, (image, url) in enumerate(images_with_urls):
            out_idx = min(idx, len(self._outputs) - 1)
            detected_type, confidence, raw_detections = self._outputs[out_idx]
            evidence.append(
                ImageEvidence(
                    image_url=url,
                    width_px=image.width,
                    height_px=image.height,
                    detected_type=detected_type,
                    confidence=confidence,
                    raw_detections=raw_detections,
                )
            )
        return assemble_project(evidence, self.labels, self.score_threshold)


def test_startup_fails_when_detector_unavailable(monkeypatch) -> None:
    def _raise_runtime_error() -> _AnalyzeDetectorStub:
        raise RuntimeError("YOLO model not found")

    monkeypatch.setattr("app.main._detector", _raise_runtime_error)

    with pytest.raises(RuntimeError, match="YOLO model not found"):
        with TestClient(app):
            pass


def test_infer_uses_aspect_fallback_on_low_confidence(monkeypatch) -> None:
    payload = {"image_urls": [_png_data_url(_solid_image(320, 640, (120, 120, 120)))]}
    monkeypatch.setattr(
        "app.main._detector",
        lambda: _AnalyzeDetectorStub(
            outputs=[("desk", 0.12, [])],
            labels=("cabinet", "desk", "shelf"),
            score_threshold=0.25,
        ),
    )

    with TestClient(app) as client:
        response = client.post("/infer", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["detected_type"] == "shelf"
    assert body["confidence"] == 0.12
    assert body["images_analyzed"] == 1
    assert len(body["image_results"]) == 1


def test_infer_accepts_batch_images_and_returns_aggregate(monkeypatch) -> None:
    image_a = _png_data_url(_solid_image(320, 240, (120, 120, 120)))
    image_b = _png_data_url(_solid_image(320, 240, (180, 180, 180)))
    monkeypatch.setattr(
        "app.main._detector",
        lambda: _AnalyzeDetectorStub(
            outputs=[
                ("desk", 0.7, [{"class_id": 1, "score": 0.7, "box": (10.0, 10.0, 80.0, 80.0)}]),
                ("cabinet", 0.9, [{"class_id": 0, "score": 0.9, "box": (20.0, 20.0, 90.0, 90.0)}]),
            ],
            labels=("cabinet", "desk", "shelf"),
            score_threshold=0.25,
        ),
    )

    with TestClient(app) as client:
        response = client.post("/infer", json={"image_urls": [image_a, image_b]})

    assert response.status_code == 200
    body = response.json()
    assert body["images_analyzed"] == 2
    assert isinstance(body["image_results"], list)
    assert len(body["image_results"]) == 2


def test_infer_counts_component_quantities_from_detections(monkeypatch) -> None:
    payload = {"image_urls": [_png_data_url(_solid_image(420, 320, (100, 100, 100)))]}
    monkeypatch.setattr(
        "app.main._detector",
        lambda: _AnalyzeDetectorStub(
            outputs=[(
                "cabinet",
                0.82,
                [
                    {"class_id": 0, "score": 0.88, "box": (10.0, 10.0, 70.0, 120.0)},
                    {"class_id": 0, "score": 0.79, "box": (72.0, 10.0, 130.0, 120.0)},
                    {"class_id": 0, "score": 0.33, "box": (132.0, 10.0, 190.0, 120.0)},
                ],
            )],
            labels=("door_panel", "shelf_panel", "back_panel"),
            score_threshold=0.25,
        ),
    )

    with TestClient(app) as client:
        response = client.post("/infer", json=payload)

    assert response.status_code == 200
    body = response.json()
    components = {component["name"]: component for component in body["components"]}
    assert components["door_panel"]["quantity"] == 3
    uuid.UUID(components["door_panel"]["id"])
    assert "component_index" in body and isinstance(body["component_index"], dict)
    door_component_id = components["door_panel"]["id"]
    assert body["component_index"][door_component_id]["name"] == "door_panel"
    assert "joints" in body and isinstance(body["joints"], list)
    assert all("parent_component_id" in item and "child_component_id" in item for item in body["joints"])
    assert "hardware" in body and isinstance(body["hardware"], list)


def test_infer_maps_label_aliases_to_component_and_hardware(monkeypatch) -> None:
    payload = {"image_urls": [_png_data_url(_solid_image(400, 400, (90, 90, 90)))]}
    monkeypatch.setattr(
        "app.main._detector",
        lambda: _AnalyzeDetectorStub(
            outputs=[(
                "cabinet",
                0.9,
                [
                    {"class_id": 0, "score": 0.9, "box": (5.0, 5.0, 200.0, 300.0)},
                    {"class_id": 1, "score": 0.8, "box": (15.0, 35.0, 40.0, 120.0)},
                    {"class_id": 2, "score": 0.8, "box": (45.0, 40.0, 160.0, 100.0)},
                ],
            )],
            labels=("wardrobe", "hinge", "drawer_front"),
            score_threshold=0.25,
        ),
    )

    with TestClient(app) as client:
        response = client.post("/infer", json=payload)

    assert response.status_code == 200
    body = response.json()
    names = {component["name"] for component in body["components"]}
    assert "cabinet_body" in names
    assert "hinge" in names
    assert any(item["code"] == "HINGE_SOFT_CLOSE_110" for item in body["hardware"])
