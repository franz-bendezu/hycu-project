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


class _AnalyzeDetectorSequentialStub:
    def __init__(
        self,
        outputs_per_call: list[list[tuple[str, float, list[dict]]]],
        labels: tuple[str, ...],
        score_threshold: float,
    ) -> None:
        self._outputs_per_call = outputs_per_call
        self.labels = labels
        self.score_threshold = score_threshold
        self._call_idx = 0

    def analyze(self, images_with_urls: list[tuple[Image.Image, str]]):
        call_idx = min(self._call_idx, len(self._outputs_per_call) - 1)
        call_outputs = self._outputs_per_call[call_idx]
        self._call_idx += 1

        evidence: list[ImageEvidence] = []
        for idx, (image, url) in enumerate(images_with_urls):
            out_idx = min(idx, len(call_outputs) - 1)
            detected_type, confidence, raw_detections = call_outputs[out_idx]
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
    assert body["detected_type"] in {"shelf", "bookcase"}
    assert 0.1 <= body["confidence"] <= 1.0
    assert body["images_analyzed"] == 1
    assert len(body["image_results"]) == 1


def test_infer_does_not_force_desk_without_desk_signal(monkeypatch) -> None:
    payload = {"image_urls": [_png_data_url(_solid_image(640, 420, (120, 120, 120)))]}
    monkeypatch.setattr(
        "app.main._detector",
        lambda: _AnalyzeDetectorStub(
            outputs=[(
                "cabinet",
                0.11,
                [
                    {"class_id": 0, "score": 0.8, "box": (30.0, 20.0, 100.0, 180.0)},
                    {"class_id": 0, "score": 0.81, "box": (540.0, 20.0, 610.0, 180.0)},
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
    assert body["detected_type"] != "desk"


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
    assert body["schema_version"] == "1.1.0"
    assert body["coordinate_frame"] == "furniture_local_v1"
    assert isinstance(body["deterministic_hash"], str) and len(body["deterministic_hash"]) == 64
    assert isinstance(body["constraints_report"], dict)
    assert isinstance(body["review_flags"], list)
    assert isinstance(body["validation_metrics"], dict)
    assert "graph_edit_distance_proxy" in body["validation_metrics"]
    assert isinstance(body["escalation"], dict)
    assert "strategy" in body["escalation"]


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
    assert "relative_position" in components["door_panel"]
    assert "bbox_3d" in components["door_panel"]
    assert "confidence" in components["door_panel"]
    assert "uncertainty" in components["door_panel"]
    assert "joints" in body and isinstance(body["joints"], list)
    assert all("parent_component_id" in item and "child_component_id" in item for item in body["joints"])
    assert all("anchor_parent" in item and "anchor_child" in item for item in body["joints"])
    assert isinstance(body["validation_metrics"]["component_coverage"], float)
    assert "human_review_required" in body["escalation"]
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


def test_assemble_project_recognizes_wardrobe_alias_and_component() -> None:
    image = _solid_image(600, 600, (100, 110, 120))
    evidence = [
        ImageEvidence(
            image_url=_png_data_url(image),
            width_px=image.width,
            height_px=image.height,
            detected_type="cabinet",
            confidence=0.2,
            raw_detections=[{"class_id": 0, "score": 0.95, "box": (20.0, 20.0, 580.0, 580.0)}],
        )
    ]

    result = assemble_project(evidence, labels=("wardrobe",), threshold=0.25)

    assert result.detected_type == "wardrobe"
    names = {component.name for component in result.components}
    assert "cabinet_body" in names


def test_assemble_project_uses_type_confidence_multiplier() -> None:
    image = _solid_image(500, 500, (130, 130, 130))
    evidence = [
        ImageEvidence(
            image_url=_png_data_url(image),
            width_px=image.width,
            height_px=image.height,
            detected_type="wardrobe",
            confidence=0.23,
            raw_detections=[],
        )
    ]

    result = assemble_project(evidence, labels=(), threshold=0.25)

    assert result.detected_type == "wardrobe"


def test_infer_supports_tv_stand_product_type(monkeypatch) -> None:
    payload = {"image_urls": [_png_data_url(_solid_image(640, 420, (95, 95, 95)))]}
    monkeypatch.setattr(
        "app.main._detector",
        lambda: _AnalyzeDetectorStub(
            outputs=[(
                "cabinet",
                0.2,
                [
                    {"class_id": 0, "score": 0.92, "box": (25.0, 30.0, 620.0, 390.0)},
                ],
            )],
            labels=("tv_stand",),
            score_threshold=0.25,
        ),
    )

    with TestClient(app) as client:
        response = client.post("/infer", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["detected_type"] == "tv_stand"


def test_assemble_project_prefers_bookcase_for_open_front_shelf_evidence() -> None:
    image = _solid_image(620, 900, (140, 140, 140))
    evidence = [
        ImageEvidence(
            image_url=_png_data_url(image),
            width_px=image.width,
            height_px=image.height,
            detected_type="cabinet",
            confidence=0.55,
            raw_detections=[
                {"class_id": 0, "score": 0.92, "box": (20.0, 20.0, 70.0, 880.0)},   # side_panel
                {"class_id": 0, "score": 0.88, "box": (540.0, 20.0, 600.0, 880.0)}, # side_panel
                {"class_id": 1, "score": 0.91, "box": (70.0, 80.0, 540.0, 130.0)},  # shelf_panel
                {"class_id": 1, "score": 0.89, "box": (70.0, 340.0, 540.0, 390.0)}, # shelf_panel
                {"class_id": 2, "score": 0.86, "box": (70.0, 130.0, 540.0, 880.0)}, # back_panel
            ],
        )
    ]

    result = assemble_project(
        evidence,
        labels=("side_panel", "shelf_panel", "back_panel"),
        threshold=0.25,
    )

    assert result.detected_type == "bookcase"


def test_assemble_project_prefers_dresser_when_drawers_dominate() -> None:
    image = _solid_image(700, 850, (150, 150, 150))
    evidence = [
        ImageEvidence(
            image_url=_png_data_url(image),
            width_px=image.width,
            height_px=image.height,
            detected_type="cabinet",
            confidence=0.45,
            raw_detections=[
                {"class_id": 0, "score": 0.9, "box": (100.0, 80.0, 600.0, 160.0)},
                {"class_id": 0, "score": 0.91, "box": (100.0, 290.0, 600.0, 370.0)},
                {"class_id": 0, "score": 0.9, "box": (100.0, 500.0, 600.0, 580.0)},
                {"class_id": 1, "score": 0.88, "box": (100.0, 160.0, 600.0, 280.0)},
                {"class_id": 1, "score": 0.87, "box": (100.0, 370.0, 600.0, 490.0)},
                {"class_id": 1, "score": 0.89, "box": (100.0, 580.0, 600.0, 760.0)},
            ],
        )
    ]

    result = assemble_project(
        evidence,
        labels=("drawer_front", "drawer_box"),
        threshold=0.25,
    )

    assert result.detected_type == "dresser"


def test_assemble_project_uses_minimums_when_component_supervision_missing(monkeypatch) -> None:
    image = _solid_image(640, 480, (120, 120, 120))
    evidence = [
        ImageEvidence(
            image_url=_png_data_url(image),
            width_px=image.width,
            height_px=image.height,
            detected_type="cabinet",
            confidence=0.9,
            raw_detections=[
                {"class_id": 0, "score": 0.95, "box": (20.0, 20.0, 620.0, 460.0)},
                {"class_id": 0, "score": 0.92, "box": (30.0, 30.0, 610.0, 450.0)},
            ],
        )
    ]

    monkeypatch.setattr("app.services.processor.COMPONENT_SUPERVISION_AVAILABLE", False)
    result = assemble_project(evidence, labels=("cabinet",), threshold=0.25)

    component_by_name = {component.name: component for component in result.components}
    assert component_by_name["cabinet_body"].quantity == 1
    assert component_by_name["side_panel"].quantity == 2
    assert component_by_name["top_panel"].quantity == 1
    assert component_by_name["bottom_panel"].quantity == 1
    assert component_by_name["back_panel"].quantity == 1


def test_assemble_project_geometry_relabels_ambiguous_panels() -> None:
    image = _solid_image(600, 900, (125, 125, 125))
    evidence = [
        ImageEvidence(
            image_url=_png_data_url(image),
            width_px=image.width,
            height_px=image.height,
            detected_type="cabinet",
            confidence=0.4,
            raw_detections=[
                {
                    "class_id": 0,
                    "score": 0.55,
                    "box": (10.0, 30.0, 95.0, 870.0),
                    "image_width_px": 600,
                    "image_height_px": 900,
                },
                {
                    "class_id": 0,
                    "score": 0.53,
                    "box": (505.0, 30.0, 590.0, 870.0),
                    "image_width_px": 600,
                    "image_height_px": 900,
                },
                {
                    "class_id": 0,
                    "score": 0.57,
                    "box": (95.0, 120.0, 505.0, 180.0),
                    "image_width_px": 600,
                    "image_height_px": 900,
                },
                {
                    "class_id": 0,
                    "score": 0.56,
                    "box": (95.0, 360.0, 505.0, 420.0),
                    "image_width_px": 600,
                    "image_height_px": 900,
                },
                {
                    "class_id": 0,
                    "score": 0.58,
                    "box": (95.0, 180.0, 505.0, 860.0),
                    "image_width_px": 600,
                    "image_height_px": 900,
                },
            ],
        )
    ]

    result = assemble_project(evidence, labels=("shelf_panel",), threshold=0.25)
    counts = {component.name: component.quantity for component in result.components}

    assert counts.get("side_panel", 0) >= 2
    assert counts.get("shelf_panel", 0) >= 2
    assert counts.get("back_panel", 0) >= 1


def test_assemble_project_avoids_single_detection_collapse_for_product_label() -> None:
    image = _solid_image(1000, 1000, (128, 128, 128))
    evidence = [
        ImageEvidence(
            image_url=_png_data_url(image),
            width_px=image.width,
            height_px=image.height,
            detected_type="cabinet",
            confidence=0.68,
            raw_detections=[
                {
                    "class_id": 0,
                    "score": 0.68,
                    "box": (290.3, -10.1, 984.1, 830.8),
                    "image_width_px": 1000,
                    "image_height_px": 1000,
                }
            ],
        )
    ]

    result = assemble_project(evidence, labels=("cabinet",), threshold=0.25)
    names = {component.name for component in result.components}

    assert "cabinet_body" in names
    assert names != {"back_panel"}
    assert any(name in names for name in {"side_panel", "top_panel", "bottom_panel"})


def test_assemble_project_uses_richer_desk_minimum_components_when_product_only() -> None:
    image = _solid_image(1200, 700, (110, 110, 110))
    evidence = [
        ImageEvidence(
            image_url=_png_data_url(image),
            width_px=image.width,
            height_px=image.height,
            detected_type="desk",
            confidence=0.7,
            raw_detections=[
                {
                    "class_id": 0,
                    "score": 0.74,
                    "box": (80.0, 90.0, 1120.0, 660.0),
                    "image_width_px": 1200,
                    "image_height_px": 700,
                }
            ],
        )
    ]

    result = assemble_project(evidence, labels=("desk",), threshold=0.25)
    counts = {component.name: component.quantity for component in result.components}

    assert counts.get("top_panel", 0) >= 1
    assert counts.get("side_panel", 0) >= 2
    assert counts.get("drawer_front", 0) >= 2
    assert counts.get("drawer_box", 0) == 0


def test_infer_track_voting_stabilizes_class_jitter(monkeypatch) -> None:
    image_a = _png_data_url(_solid_image(640, 480, (105, 105, 105)))
    image_b = _png_data_url(_solid_image(640, 480, (115, 115, 115)))

    monkeypatch.setattr(
        "app.main._detector",
        lambda: _AnalyzeDetectorStub(
            outputs=[
                ("cabinet", 0.8, [{"class_id": 0, "score": 0.86, "box": (100.0, 80.0, 520.0, 420.0)}]),
                ("cabinet", 0.8, [{"class_id": 1, "score": 0.31, "box": (102.0, 82.0, 522.0, 422.0)}]),
            ],
            labels=("door_panel", "side_panel"),
            score_threshold=0.25,
        ),
    )

    with TestClient(app) as client:
        response = client.post("/infer", json={"image_urls": [image_a, image_b]})

    assert response.status_code == 200
    body = response.json()
    components = {component["name"]: component for component in body["components"]}

    # One physical object across views should keep one stable component label.
    assert "door_panel" in components
    assert components["door_panel"]["quantity"] == 1


def test_assemble_project_exposes_escalation_thresholds() -> None:
    image = _solid_image(640, 480, (120, 120, 120))
    evidence = [
        ImageEvidence(
            image_url=_png_data_url(image),
            width_px=image.width,
            height_px=image.height,
            detected_type="cabinet",
            confidence=0.8,
            raw_detections=[{"class_id": 0, "score": 0.9, "box": (20.0, 20.0, 600.0, 460.0)}],
        )
    ]

    result = assemble_project(evidence, labels=("cabinet",), threshold=0.25)

    assert "geometry_threshold" in result.escalation
    assert "mvs_threshold" in result.escalation


def test_infer_applies_heavy_refiner_when_escalated(monkeypatch) -> None:
    payload = {"image_urls": [_png_data_url(_solid_image(320, 640, (120, 120, 120)))]}

    monkeypatch.setattr(
        "app.main._detector",
        lambda: _AnalyzeDetectorStub(
            outputs=[("desk", 0.12, [])],
            labels=("cabinet", "desk", "shelf"),
            score_threshold=0.25,
        ),
    )
    monkeypatch.setattr("app.services.processor.get_escalation_geometry_threshold", lambda: 0.1)
    monkeypatch.setattr("app.services.processor.get_escalation_mvs_threshold", lambda: 0.2)
    monkeypatch.setattr("app.main._refiner", lambda: __import__("app.services.refiner", fromlist=["HeavyRefiner"]).HeavyRefiner(enabled=True))

    with TestClient(app) as client:
        response = client.post("/infer", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["escalation"]["strategy"] in {"mvs_refinement_applied", "escalate_mvs_refinement"}
    assert body["constraints_report"].get("heavy_refinement_attempted") in {True, False}


def test_benchmark_returns_aggregate_metrics(monkeypatch) -> None:
    item_a = _png_data_url(_solid_image(400, 320, (120, 120, 120)))
    item_b = _png_data_url(_solid_image(400, 640, (150, 150, 150)))

    sequential_detector = _AnalyzeDetectorSequentialStub(
        outputs_per_call=[
            [("desk", 0.85, [{"class_id": 0, "score": 0.9, "box": (20.0, 20.0, 300.0, 200.0)}])],
            [("cabinet", 0.25, [])],
        ],
        labels=("door_panel", "shelf_panel", "back_panel"),
        score_threshold=0.25,
    )
    monkeypatch.setattr("app.main._detector", lambda: sequential_detector)

    with TestClient(app) as client:
        response = client.post(
            "/benchmark",
            json={
                "items": [
                    {"item_id": "sku_a", "image_urls": [item_a]},
                    {"item_id": "sku_b", "image_urls": [item_b]},
                ]
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["items_analyzed"] == 2
    assert isinstance(body["escalation_strategy_counts"], dict)
    assert isinstance(body["item_results"], list)
    assert len(body["item_results"]) == 2
    assert 0.0 <= body["avg_confidence"] <= 1.0
    assert 0.0 <= body["avg_component_coverage"] <= 1.0
    assert 0.0 <= body["avg_physical_validity"] <= 1.0
    assert 0.0 <= body["human_review_rate"] <= 1.0
