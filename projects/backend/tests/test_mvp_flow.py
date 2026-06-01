from __future__ import annotations

import uuid

from fastapi.testclient import TestClient

from app.api.routers import analysis, projects
from app.main import app


def _components_for_type(detected_type: str) -> list[dict]:
    if detected_type == "desk":
        return [
            {"name": "desktop", "kind": "panel", "quantity": 1},
            {"name": "left_side_panel", "kind": "panel", "quantity": 1},
            {"name": "right_side_panel", "kind": "panel", "quantity": 1},
            {"name": "back_panel", "kind": "panel", "quantity": 1},
            {"name": "front_apron", "kind": "support", "quantity": 1},
        ]

    if detected_type == "shelf":
        return [
            {"name": "top_panel", "kind": "panel", "quantity": 1},
            {"name": "bottom_panel", "kind": "panel", "quantity": 1},
            {"name": "left_side_panel", "kind": "panel", "quantity": 1},
            {"name": "right_side_panel", "kind": "panel", "quantity": 1},
            {"name": "shelf_panel", "kind": "panel", "quantity": 3},
            {"name": "back_panel", "kind": "panel", "quantity": 1},
        ]

    return [
        {"name": "top_panel", "kind": "panel", "quantity": 1},
        {"name": "bottom_panel", "kind": "panel", "quantity": 1},
        {"name": "left_side_panel", "kind": "panel", "quantity": 1},
        {"name": "right_side_panel", "kind": "panel", "quantity": 1},
        {"name": "door_panel", "kind": "panel", "quantity": 2},
        {"name": "shelf_panel", "kind": "panel", "quantity": 3},
        {"name": "back_panel", "kind": "panel", "quantity": 1},
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

        return {
            "detected_type": detected_type,
            "confidence": 0.94,
            "suggested_width": width,
            "suggested_height": height,
            "suggested_depth": depth,
            "components": _components_for_type(detected_type),
            "image_url": image_url,
        }

    def infer_many(self, image_urls: list[str]) -> dict:
        results = [self.infer(url) for url in image_urls]
        # Keep compatibility with existing downstream project generation by returning
        # the same top-level fields while exposing batch metadata.
        best = max(results, key=lambda item: float(item["confidence"]))
        merged_components: dict[str, dict] = {}
        for result in results:
            for component in result["components"]:
                current = merged_components.get(component["name"])
                if current is None or component["quantity"] > current["quantity"]:
                    merged_components[component["name"]] = component

        payload = dict(best)
        payload["components"] = [merged_components[key] for key in sorted(merged_components)]
        payload["images_analyzed"] = len(results)
        payload["image_results"] = results
        payload["image_url"] = image_urls[0]
        return payload

    def submit(self, image_urls: list[str]) -> tuple[str, dict]:
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