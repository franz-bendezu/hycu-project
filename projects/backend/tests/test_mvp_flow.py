from __future__ import annotations

from fastapi.testclient import TestClient

from app.api.routers import analysis, projects
from app.infrastructure.gateways.inference_gateway import FakeInferenceGateway
from app.main import app


def test_analyze_to_validate_happy_path(monkeypatch) -> None:
    monkeypatch.setattr(analysis, "client", FakeInferenceGateway())
    client = TestClient(app)

    analyze_response = client.post(
        "/api/v1/analyze",
        json={
            "image_url": "https://example.com/cabinet.jpg",
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


def test_analyze_upload_happy_path(monkeypatch) -> None:
    monkeypatch.setattr(analysis, "client", FakeInferenceGateway())
    client = TestClient(app)

    response = client.post(
        "/api/v1/analyze-upload",
        files={"file": ("cabinet.jpg", b"fake-image-bytes", "image/jpeg")},
        data={"project_name": "Upload Cabinet"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "complete"

    job_response = client.get(f"/api/v1/jobs/{payload['job_id']}")
    assert job_response.status_code == 200
    job_data = job_response.json()
    assert job_data["result"]["image_url"].startswith("data:image/jpeg;base64,")


def test_analyze_upload_rejects_non_image(monkeypatch) -> None:
    monkeypatch.setattr(analysis, "client", FakeInferenceGateway())
    client = TestClient(app)

    response = client.post(
        "/api/v1/analyze-upload",
        files={"file": ("notes.txt", b"hello", "text/plain")},
    )

    assert response.status_code == 415
    payload = response.json()
    assert payload["error"]["code"] == "HTTP_415"
    assert payload["error"]["message"] == "Uploaded file must be an image"


def test_project_first_asset_then_job_flow(monkeypatch) -> None:
    monkeypatch.setattr(analysis, "client", FakeInferenceGateway())
    monkeypatch.setattr(projects, "client", FakeInferenceGateway())
    client = TestClient(app)

    create_project = client.post("/api/v1/projects", json={"name": "Project First"})
    assert create_project.status_code == 200
    project_id = create_project.json()["project_id"]

    upload_asset = client.post(
        f"/api/v1/projects/{project_id}/assets",
        files={"file": ("cabinet.jpg", b"fake-image-bytes", "image/jpeg")},
    )
    assert upload_asset.status_code == 200
    asset_id = upload_asset.json()["asset_id"]

    create_job = client.post(
        f"/api/v1/projects/{project_id}/jobs",
        json={"asset_id": asset_id},
    )
    assert create_job.status_code == 200
    job_payload = create_job.json()
    assert job_payload["status"] == "complete"
    assert job_payload["project_id"] == project_id

    job_id = job_payload["job_id"]
    fetched_job = client.get(f"/api/v1/projects/{project_id}/jobs/{job_id}")
    assert fetched_job.status_code == 200
    fetched_data = fetched_job.json()
    assert fetched_data["project_id"] == project_id
    assert fetched_data["asset_id"] == asset_id
    assert fetched_data["result"]["detected_type"] == "cabinet"