from __future__ import annotations

import uuid

from fastapi.testclient import TestClient

from app.api.routers import analysis, projects
from app.main import app
from app.presentation.schemas.inference import InferenceOutput


def _component(kind: str, suffix: str) -> dict:
    return {
        "id": f"cmp_{kind}_{suffix}",
        "kind": kind,
        "category": "structural",
        "width": 100.0,
        "height": 100.0,
        "depth": 18.0,
    }


def _components_for_type(detected_type: str) -> list[dict]:
    if detected_type == "desk":
        return [
            _component("top_panel", "1"),
            _component("left_side", "1"),
            _component("right_side", "1"),
            _component("back_panel", "1"),
            _component("front_panel", "1"),
        ]

    if detected_type == "shelf":
        return [
            _component("top_panel", "1"),
            _component("bottom_panel", "1"),
            _component("left_side", "1"),
            _component("right_side", "1"),
            _component("shelf", "1"),
            _component("shelf", "2"),
            _component("shelf", "3"),
            _component("back_panel", "1"),
        ]

    return [
        _component("top_panel", "1"),
        _component("bottom_panel", "1"),
        _component("left_side", "1"),
        _component("right_side", "1"),
        _component("door_panel", "1"),
        _component("door_panel", "2"),
        _component("shelf", "1"),
        _component("shelf", "2"),
        _component("shelf", "3"),
        _component("back_panel", "1"),
    ]


def _faces_for_components(components: list[dict]) -> list[dict]:
    normals = ["+x", "-x", "+y", "-y", "+z", "-z"]
    faces: list[dict] = []
    for component in components:
        component_id = component["id"]
        for normal in normals:
            faces.append(
                {
                    "id": f"{component_id}:{normal}",
                    "component_id": component_id,
                    "normal": normal,
                }
            )
    return faces


def _joints_for_components(components: list[dict]) -> list[dict]:
    if len(components) < 2:
        return []
    parent_id = components[0]["id"]
    child_id = components[1]["id"]
    return [
        {
            "id": f"joint_{parent_id}_{child_id}",
            "parent_face_id": f"{parent_id}:+x",
            "child_face_id": f"{child_id}:-x",
            "joint_rule": "overlap",
            "offset_u": 0.0,
            "offset_v": 0.0,
            "clearance": 0.0,
        }
    ]


class TestInferenceGateway:
    def infer(self, image_url: str) -> dict:
        detected_type = "cabinet"
        width, height, depth = 800.0, 1200.0, 450.0

        lower = image_url.lower()
        if "desk" in lower:
            detected_type = "desk"
            width, height, depth = 1200.0, 750.0, 600.0
        elif "shelf" in lower:
            detected_type = "shelf"
            width, height, depth = 900.0, 1700.0, 300.0

        components = _components_for_type(detected_type)
        faces = _faces_for_components(components)
        joints = _joints_for_components(components)
        return {
            "detected_type": detected_type,
            "confidence": 0.94,
            "suggested_width": width,
            "suggested_height": height,
            "suggested_depth": depth,
            "components": components,
            "faces": faces,
            "joints": joints,
            "image_url": image_url,
        }

    def infer_many(self, image_urls: list[str]) -> InferenceOutput:
        results = [self.infer(url) for url in image_urls]
        best = max(results, key=lambda item: float(item["confidence"]))
        payload = dict(best)
        payload["images_analyzed"] = len(results)
        payload["image_results"] = [
            {
                "image_url": item["image_url"],
                "width_px": 640,
                "height_px": 480,
                "detected_type": item["detected_type"],
                "confidence": item["confidence"],
                "raw_detections": [],
            }
            for item in results
        ]
        payload["image_url"] = image_urls[0]
        return InferenceOutput.model_validate(payload)

    def submit(self, image_urls: list[str]) -> tuple[str, InferenceOutput]:
        return str(uuid.uuid4()), self.infer_many(image_urls)


def test_analyze_to_validate_happy_path(monkeypatch) -> None:
    monkeypatch.setattr(analysis, "client", TestInferenceGateway())
    client = TestClient(app)

    analyze_response = client.post(
        "/api/v1/analyze",
        json={
            "image_urls": ["https://example.com/cabinet.jpg"],
            "project_name": "Kitchen Cabinet",
        },
    )
    assert analyze_response.status_code == 200
    analyze_data = analyze_response.json()
    assert analyze_data["status"] == "complete"
    job_id = analyze_data["job_id"]

    job_response = client.get(f"/api/v1/jobs/{job_id}")
    assert job_response.status_code == 200
    assert job_response.json()["result"]["detected_type"] == "cabinet"
    assert len(job_response.json()["result"]["components"]) > 0

    create_response = client.post("/api/v1/projects", json={"job_id": job_id})
    assert create_response.status_code == 200
    create_data = create_response.json()
    project_id = create_data["project_id"]
    assert create_data["model"]["product"]["name"] == "Kitchen Cabinet"

    patch_response = client.patch(
        f"/api/v1/projects/{project_id}",
        json={
            "target_width": 900,
            "target_height": 1300,
            "target_depth": 480,
            "shelf_count": 4,
        },
    )
    assert patch_response.status_code == 200
    updated_product = patch_response.json()["model"]["product"]
    assert updated_product["target_width"] == 900
    assert updated_product["shelf_count"] == 4

    validate_response = client.post(f"/api/v1/projects/{project_id}/validate")
    assert validate_response.status_code == 200
    validate_data = validate_response.json()
    assert validate_data["valid"] is True
    assert validate_data["errors"] == []


def test_error_envelope_for_missing_job() -> None:
    client = TestClient(app)
    response = client.get("/api/v1/jobs/does-not-exist")
    assert response.status_code == 404
    payload = response.json()
    assert payload["error"]["code"] == "HTTP_404"
    assert payload["error"]["message"] == "Job not found"


def test_analyze_upload_batch_happy_path(monkeypatch) -> None:
    monkeypatch.setattr(analysis, "client", TestInferenceGateway())
    client = TestClient(app)

    response = client.post(
        "/api/v1/analyze-upload-batch",
        files=[("files", ("cabinet.jpg", b"fake-image-bytes", "image/jpeg"))],
        data={"project_name": "Upload Cabinet"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "complete"

    job_response = client.get(f"/api/v1/jobs/{payload['job_id']}")
    assert job_response.status_code == 200
    job_data = job_response.json()
    assert job_data["result"]["image_url"].startswith("data:image/jpeg;base64,")


def test_analyze_upload_batch_rejects_non_image(monkeypatch) -> None:
    monkeypatch.setattr(analysis, "client", TestInferenceGateway())
    client = TestClient(app)

    response = client.post(
        "/api/v1/analyze-upload-batch",
        files=[("files", ("notes.txt", b"hello", "text/plain"))],
    )

    assert response.status_code == 415
    payload = response.json()
    assert payload["error"]["code"] == "HTTP_415"
    assert payload["error"]["message"] == "Uploaded files must be images"


def test_analyze_accepts_multiple_urls(monkeypatch) -> None:
    monkeypatch.setattr(analysis, "client", TestInferenceGateway())
    client = TestClient(app)

    response = client.post(
        "/api/v1/analyze",
        json={
            "image_urls": [
                "https://example.com/cabinet.jpg",
                "https://example.com/desk.jpg",
            ],
            "project_name": "Batch URLs",
        },
    )
    assert response.status_code == 200
    job_id = response.json()["job_id"]

    job_response = client.get(f"/api/v1/jobs/{job_id}")
    assert job_response.status_code == 200
    result = job_response.json()["result"]
    assert result["images_analyzed"] == 2
    assert len(result["image_results"]) == 2


def test_project_first_asset_then_job_flow(monkeypatch) -> None:
    monkeypatch.setattr(analysis, "client", TestInferenceGateway())
    monkeypatch.setattr(projects, "client", TestInferenceGateway())
    client = TestClient(app)

    create_project = client.post("/api/v1/projects", json={"name": "Project First"})
    assert create_project.status_code == 200
    project_id = create_project.json()["project_id"]

    upload_asset = client.post(
        f"/api/v1/projects/{project_id}/assets",
        files={"file": ("cabinet.jpg", b"fake-image-bytes", "image/jpeg")},
    )
    assert upload_asset.status_code == 200

    upload_second_asset = client.post(
        f"/api/v1/projects/{project_id}/assets",
        files={"file": ("desk.jpg", b"fake-image-bytes-2", "image/jpeg")},
    )
    assert upload_second_asset.status_code == 200

    create_job = client.post(
        f"/api/v1/projects/{project_id}/jobs",
        json={},
    )
    assert create_job.status_code == 200
    job_payload = create_job.json()
    assert job_payload["status"] == "complete"
    assert job_payload["project_id"] == project_id
    assert job_payload["asset_id"] is None
    assert job_payload["asset_count"] == 2

    job_id = job_payload["job_id"]
    fetched_job = client.get(f"/api/v1/projects/{project_id}/jobs/{job_id}")
    assert fetched_job.status_code == 200
    fetched_data = fetched_job.json()
    assert fetched_data["project_id"] == project_id
    assert fetched_data["asset_id"] is None
    assert fetched_data["result"]["images_analyzed"] == 2
    assert fetched_data["result"]["detected_type"] == "cabinet"
    assert len(fetched_data["result"]["components"]) > 0
    assert fetched_data["asset_results"] is not None
    assert len(fetched_data["asset_results"]) == 2
    assert all(row["job_id"] == job_id for row in fetched_data["asset_results"])
    assert all(row["status"] == "complete" for row in fetched_data["asset_results"])
    assert all(row["result"] is not None for row in fetched_data["asset_results"])


def test_deleted_asset_is_excluded_from_project_job(monkeypatch) -> None:
    monkeypatch.setattr(projects, "client", TestInferenceGateway())
    client = TestClient(app)

    create_project = client.post("/api/v1/projects", json={"name": "Delete Asset Flow"})
    assert create_project.status_code == 200
    project_id = create_project.json()["project_id"]

    upload_a = client.post(
        f"/api/v1/projects/{project_id}/assets",
        files={"file": ("cabinet.jpg", b"fake-a", "image/jpeg")},
    )
    assert upload_a.status_code == 200
    asset_a = upload_a.json()["asset_id"]

    upload_b = client.post(
        f"/api/v1/projects/{project_id}/assets",
        files={"file": ("desk.jpg", b"fake-b", "image/jpeg")},
    )
    assert upload_b.status_code == 200

    delete_a = client.delete(f"/api/v1/projects/{project_id}/assets/{asset_a}")
    assert delete_a.status_code == 204

    listed_assets = client.get(f"/api/v1/projects/{project_id}/assets")
    assert listed_assets.status_code == 200
    assert len(listed_assets.json()["assets"]) == 1

    create_job = client.post(f"/api/v1/projects/{project_id}/jobs", json={})
    assert create_job.status_code == 200
    assert create_job.json()["asset_count"] == 1


def test_create_project_uses_inference_component_shelf_quantity(monkeypatch) -> None:
    class CustomGateway(TestInferenceGateway):
        def infer(self, image_url: str) -> dict:
            payload = super().infer(image_url)
            payload["detected_type"] = "cabinet"
            payload["components"] = [
                {"name": "top_panel", "kind": "panel", "quantity": 1},
                {"name": "shelf_panel", "kind": "panel", "quantity": 5},
                {"name": "back_panel", "kind": "panel", "quantity": 1},
            ]
            return payload

    monkeypatch.setattr(analysis, "client", CustomGateway())
    client = TestClient(app)

    analyze_response = client.post(
        "/api/v1/analyze",
        json={"image_urls": ["https://example.com/cabinet.jpg"], "project_name": "Component-Driven"},
    )
    assert analyze_response.status_code == 200
    job_id = analyze_response.json()["job_id"]

    create_response = client.post("/api/v1/projects", json={"job_id": job_id})
    assert create_response.status_code == 200
    model = create_response.json()["model"]
    assert model["product"]["shelf_count"] == 5


def test_create_project_uses_components_for_inferred_type(monkeypatch) -> None:
    class CustomGateway(TestInferenceGateway):
        def infer(self, image_url: str) -> dict:
            payload = super().infer(image_url)
            payload["detected_type"] = "cabinet"
            payload["components"] = [
                {"name": "desktop", "kind": "panel", "quantity": 1},
                {"name": "front_apron", "kind": "support", "quantity": 1},
            ]
            return payload

    monkeypatch.setattr(analysis, "client", CustomGateway())
    client = TestClient(app)

    analyze_response = client.post(
        "/api/v1/analyze",
        json={"image_urls": ["https://example.com/desk.jpg"], "project_name": "Desk from Components"},
    )
    assert analyze_response.status_code == 200
    job_id = analyze_response.json()["job_id"]

    create_response = client.post("/api/v1/projects", json={"job_id": job_id})
    assert create_response.status_code == 200
    product = create_response.json()["model"]["product"]
    assert product["inferred_type"] == "desk"
    assert product["shelf_count"] == 0


def test_create_project_maps_inference_facade_counts(monkeypatch) -> None:
    class CustomGateway(TestInferenceGateway):
        def infer(self, image_url: str) -> dict:
            payload = super().infer(image_url)
            payload["detected_type"] = "cabinet"
            payload["components"] = [
                {"name": "door_panel", "kind": "panel", "quantity": 3},
                {"name": "drawer_front", "kind": "panel", "quantity": 2},
                {"name": "shelf_panel", "kind": "panel", "quantity": 2},
            ]
            return payload

    monkeypatch.setattr(analysis, "client", CustomGateway())
    client = TestClient(app)

    analyze_response = client.post(
        "/api/v1/analyze",
        json={"image_urls": ["https://example.com/cabinet-combo.jpg"], "project_name": "Facade Counts"},
    )
    assert analyze_response.status_code == 200
    job_id = analyze_response.json()["job_id"]

    create_response = client.post("/api/v1/projects", json={"job_id": job_id})
    assert create_response.status_code == 200
    product = create_response.json()["model"]["product"]
    components = create_response.json()["model"]["components"]

    assert product["door_count"] == 3
    assert product["drawer_count"] == 2
    assert sum(1 for component in components if component["kind"] == "door_panel") == 3
    assert sum(1 for component in components if component["kind"] == "drawer_front") == 2


def test_create_project_prefers_facade_components_over_wrong_detected_type(monkeypatch) -> None:
    class CustomGateway(TestInferenceGateway):
        def infer(self, image_url: str) -> dict:
            payload = super().infer(image_url)
            payload["detected_type"] = "shelf"
            payload["confidence"] = 0.91
            payload["components"] = [
                {"name": "door_panel", "kind": "panel", "quantity": 2},
                {"name": "drawer_front", "kind": "panel", "quantity": 1},
                {"name": "shelf_panel", "kind": "panel", "quantity": 2},
            ]
            return payload

    monkeypatch.setattr(analysis, "client", CustomGateway())
    client = TestClient(app)

    analyze_response = client.post(
        "/api/v1/analyze",
        json={"image_urls": ["https://example.com/mixed-facade.jpg"], "project_name": "Facade Override"},
    )
    assert analyze_response.status_code == 200
    job_id = analyze_response.json()["job_id"]

    create_response = client.post("/api/v1/projects", json={"job_id": job_id})
    assert create_response.status_code == 200
    product = create_response.json()["model"]["product"]

    assert product["inferred_type"] == "cabinet"
    assert product["door_count"] == 2
    assert product["drawer_count"] == 1


def test_project_job_updates_model_with_facade_counts(monkeypatch) -> None:
    class CustomGateway(TestInferenceGateway):
        def infer_many(self, image_urls: list[str]) -> dict:
            payload = super().infer_many(image_urls)
            payload["detected_type"] = "cabinet"
            payload["components"] = [
                {"id": "cmp_door_panel", "name": "door_panel", "kind": "panel", "quantity": 2},
                {"id": "cmp_drawer_front", "name": "drawer_front", "kind": "panel", "quantity": 2},
                {"id": "cmp_shelf_panel", "name": "shelf_panel", "kind": "panel", "quantity": 2},
            ]
            return payload

    monkeypatch.setattr(projects, "client", CustomGateway())
    client = TestClient(app)

    create_project = client.post("/api/v1/projects", json={"name": "Facade Project Flow"})
    assert create_project.status_code == 200
    project_id = create_project.json()["project_id"]

    upload_asset = client.post(
        f"/api/v1/projects/{project_id}/assets",
        files={"file": ("cabinet.jpg", b"fake-image-bytes", "image/jpeg")},
    )
    assert upload_asset.status_code == 200

    create_job = client.post(f"/api/v1/projects/{project_id}/jobs", json={})
    assert create_job.status_code == 200

    project_response = client.get(f"/api/v1/projects/{project_id}")
    assert project_response.status_code == 200
    model = project_response.json()["model"]

    assert model["product"]["door_count"] == 2
    assert model["product"]["drawer_count"] == 2
    assert sum(1 for component in model["components"] if component["kind"] == "door_panel") == 2
    assert sum(1 for component in model["components"] if component["kind"] == "drawer_front") == 2


def test_create_project_maps_divider_count_from_inference(monkeypatch) -> None:
    class CustomGateway(TestInferenceGateway):
        def infer(self, image_url: str) -> dict:
            payload = super().infer(image_url)
            payload["detected_type"] = "cabinet"
            payload["components"] = [
                {"name": "divider_panel", "kind": "panel", "quantity": 1},
                {"name": "door_panel", "kind": "panel", "quantity": 2},
                {"name": "shelf_panel", "kind": "panel", "quantity": 2},
            ]
            return payload

    monkeypatch.setattr(analysis, "client", CustomGateway())
    client = TestClient(app)

    analyze_response = client.post(
        "/api/v1/analyze",
        json={"image_urls": ["https://example.com/divider-cabinet.jpg"], "project_name": "Divider Cabinet"},
    )
    assert analyze_response.status_code == 200
    job_id = analyze_response.json()["job_id"]

    create_response = client.post("/api/v1/projects", json={"job_id": job_id})
    assert create_response.status_code == 200
    model = create_response.json()["model"]

    assert model["product"]["divider_count"] == 1
    assert sum(1 for component in model["components"] if component["kind"] == "divider_panel") == 1


def test_create_project_infers_divider_from_two_door_layout(monkeypatch) -> None:
    class CustomGateway(TestInferenceGateway):
        def infer(self, image_url: str) -> dict:
            payload = super().infer(image_url)
            payload["detected_type"] = "cabinet"
            payload["components"] = [
                {"name": "door_panel", "kind": "panel", "quantity": 2},
                {"name": "shelf_panel", "kind": "panel", "quantity": 2},
                {"name": "back_panel", "kind": "panel", "quantity": 1},
            ]
            return payload

    monkeypatch.setattr(analysis, "client", CustomGateway())
    client = TestClient(app)

    analyze_response = client.post(
        "/api/v1/analyze",
        json={"image_urls": ["https://example.com/two-door-cabinet.jpg"], "project_name": "Two Door"},
    )
    assert analyze_response.status_code == 200
    job_id = analyze_response.json()["job_id"]

    create_response = client.post("/api/v1/projects", json={"job_id": job_id})
    assert create_response.status_code == 200
    model = create_response.json()["model"]

    assert model["product"]["door_count"] == 2
    assert model["product"]["divider_count"] == 1
    assert sum(1 for component in model["components"] if component["kind"] == "divider_panel") == 1


def test_create_project_maps_generic_fronts_to_mixed_facades(monkeypatch) -> None:
    class CustomGateway(TestInferenceGateway):
        def infer(self, image_url: str) -> dict:
            payload = super().infer(image_url)
            payload["detected_type"] = "cabinet"
            payload["components"] = [
                {"name": "front_panel", "kind": "panel", "quantity": 4},
                {"name": "shelf_panel", "kind": "panel", "quantity": 2},
            ]
            return payload

    monkeypatch.setattr(analysis, "client", CustomGateway())
    client = TestClient(app)

    analyze_response = client.post(
        "/api/v1/analyze",
        json={"image_urls": ["https://example.com/mixed-front-cabinet.jpg"], "project_name": "Mixed Fronts"},
    )
    assert analyze_response.status_code == 200
    job_id = analyze_response.json()["job_id"]

    create_response = client.post("/api/v1/projects", json={"job_id": job_id})
    assert create_response.status_code == 200
    model = create_response.json()["model"]

    assert model["product"]["door_count"] == 3
    assert model["product"]["drawer_count"] == 1
    assert sum(1 for component in model["components"] if component["kind"] == "door_panel") == 3
    assert sum(1 for component in model["components"] if component["kind"] == "drawer_front") == 1


def test_create_project_normalizes_two_door_divider_pattern_to_mixed_facade(monkeypatch) -> None:
    class CustomGateway(TestInferenceGateway):
        def infer(self, image_url: str) -> dict:
            payload = super().infer(image_url)
            payload["detected_type"] = "cabinet"
            payload["components"] = [
                {"name": "door_panel", "kind": "panel", "quantity": 2},
                {"name": "divider_panel", "kind": "panel", "quantity": 1},
                {"name": "shelf_panel", "kind": "panel", "quantity": 2},
            ]
            return payload

    monkeypatch.setattr(analysis, "client", CustomGateway())
    client = TestClient(app)

    analyze_response = client.post(
        "/api/v1/analyze",
        json={"image_urls": ["https://example.com/asymmetric-cabinet.jpg"], "project_name": "Asymmetric Cabinet"},
    )
    assert analyze_response.status_code == 200
    job_id = analyze_response.json()["job_id"]

    create_response = client.post("/api/v1/projects", json={"job_id": job_id})
    assert create_response.status_code == 200
    model = create_response.json()["model"]

    assert model["product"]["door_count"] == 3
    assert model["product"]["drawer_count"] == 1
    assert sum(1 for component in model["components"] if component["kind"] == "door_panel") == 3
    assert sum(1 for component in model["components"] if component["kind"] == "drawer_front") == 1


def test_create_project_keeps_multiview_two_door_pattern_without_explicit_divider(monkeypatch) -> None:
    class CustomGateway(TestInferenceGateway):
        def infer_many(self, image_urls: list[str]) -> dict:
            payload = super().infer_many(image_urls)
            payload["detected_type"] = "cabinet"
            payload["images_analyzed"] = 2
            payload["components"] = [
                {"name": "door_panel", "kind": "panel", "quantity": 2},
                {"name": "shelf_panel", "kind": "panel", "quantity": 2},
            ]
            payload["image_results"] = [
                {
                    "image_url": image_urls[0],
                    "width_px": 720,
                    "height_px": 1600,
                    "detected_type": "cabinet",
                    "confidence": 0.91,
                    "raw_detections": [],
                },
                {
                    "image_url": image_urls[1],
                    "width_px": 720,
                    "height_px": 1600,
                    "detected_type": "cabinet",
                    "confidence": 0.89,
                    "raw_detections": [],
                },
            ]
            return payload

    monkeypatch.setattr(analysis, "client", CustomGateway())
    client = TestClient(app)

    analyze_response = client.post(
        "/api/v1/analyze",
        json={
            "image_urls": [
                "https://example.com/cabinet-closed.jpg",
                "https://example.com/cabinet-open.jpg",
            ],
            "project_name": "MultiView Cabinet",
        },
    )
    assert analyze_response.status_code == 200
    job_id = analyze_response.json()["job_id"]

    create_response = client.post("/api/v1/projects", json={"job_id": job_id})
    assert create_response.status_code == 200
    model = create_response.json()["model"]

    assert model["product"]["door_count"] == 2
    assert model["product"]["drawer_count"] == 0
    assert sum(1 for component in model["components"] if component["kind"] == "door_panel") == 2
    assert sum(1 for component in model["components"] if component["kind"] == "drawer_front") == 0


def test_create_project_normalizes_overdetected_doors_to_mixed_facade(monkeypatch) -> None:
    class CustomGateway(TestInferenceGateway):
        def infer(self, image_url: str) -> dict:
            payload = super().infer(image_url)
            payload["detected_type"] = "cabinet"
            payload["components"] = [
                {"name": "door_panel", "kind": "panel", "quantity": 4},
                {"name": "shelf_panel", "kind": "panel", "quantity": 2},
            ]
            return payload

    monkeypatch.setattr(analysis, "client", CustomGateway())
    client = TestClient(app)

    analyze_response = client.post(
        "/api/v1/analyze",
        json={"image_urls": ["https://example.com/overdetect-cabinet.jpg"], "project_name": "Overdetect"},
    )
    assert analyze_response.status_code == 200
    job_id = analyze_response.json()["job_id"]

    create_response = client.post("/api/v1/projects", json={"job_id": job_id})
    assert create_response.status_code == 200
    model = create_response.json()["model"]

    assert model["product"]["door_count"] == 3
    assert model["product"]["drawer_count"] == 1
    assert sum(1 for component in model["components"] if component["kind"] == "door_panel") == 3
    assert sum(1 for component in model["components"] if component["kind"] == "drawer_front") == 1


def test_update_project_allows_divider_override_without_reinference(monkeypatch) -> None:
    monkeypatch.setattr(analysis, "client", TestInferenceGateway())
    client = TestClient(app)

    analyze_response = client.post(
        "/api/v1/analyze",
        json={"image_urls": ["https://example.com/cabinet.jpg"], "project_name": "Manual Divider"},
    )
    assert analyze_response.status_code == 200
    job_id = analyze_response.json()["job_id"]

    create_response = client.post("/api/v1/projects", json={"job_id": job_id})
    assert create_response.status_code == 200
    project_id = create_response.json()["project_id"]

    patch_response = client.patch(
        f"/api/v1/projects/{project_id}",
        json={"divider_count": 1, "door_count": 2, "drawer_count": 0},
    )
    assert patch_response.status_code == 200
    model = patch_response.json()["model"]

    assert model["product"]["divider_count"] == 1
    assert sum(1 for component in model["components"] if component["kind"] == "divider_panel") == 1


def test_create_project_maps_detection_layout_into_joint_positions(monkeypatch) -> None:
    class CustomGateway(TestInferenceGateway):
        def infer(self, image_url: str) -> dict:
            payload = super().infer(image_url)
            payload["detected_type"] = "cabinet"
            payload["components"] = [
                {
                    "name": "door_panel_left",
                    "kind": "panel",
                    "quantity": 1,
                    "box_corners": [0.1, 0.1, 0.4, 0.9],
                },
                {
                    "name": "door_panel_right",
                    "kind": "panel",
                    "quantity": 1,
                    "box_corners": [0.6, 0.1, 0.9, 0.9],
                },
                {
                    "name": "shelf_panel",
                    "kind": "panel",
                    "quantity": 2,
                },
            ]
            payload["image_results"] = [
                {
                    "image_url": image_url,
                    "width_px": 800,
                    "height_px": 1200,
                    "detected_type": "cabinet",
                    "confidence": 0.95,
                    "raw_detections": [
                        {"label": "door_panel", "box": [80, 120, 320, 1080]},
                        {"label": "door_panel", "box": [480, 120, 720, 1080]},
                    ],
                }
            ]
            return payload

    monkeypatch.setattr(analysis, "client", CustomGateway())
    client = TestClient(app)

    analyze_response = client.post(
        "/api/v1/analyze",
        json={"image_urls": ["https://example.com/layout-cabinet.jpg"], "project_name": "Layout Cabinet"},
    )
    assert analyze_response.status_code == 200
    job_id = analyze_response.json()["job_id"]

    create_response = client.post("/api/v1/projects", json={"job_id": job_id})
    assert create_response.status_code == 200
    model = create_response.json()["model"]

    components_by_id = {component["id"]: component for component in model["components"]}
    def _component_id_from_face_id(face_id: str) -> str:
        return face_id.split(":", 1)[0]

    door_joint_positions = [
        joint["offset_u"]
        for joint in model["joints"]
        if components_by_id.get(_component_id_from_face_id(joint["child_face_id"]), {}).get("kind") == "door_panel"
    ]

    assert len(door_joint_positions) == 2
    assert min(door_joint_positions) < 0
    assert max(door_joint_positions) > 0