from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app


def test_project_artifact_download_endpoints() -> None:
    client = TestClient(app)

    create_response = client.post("/api/v1/projects", json={"name": "Artifact Project"})
    assert create_response.status_code == 200
    project_id = create_response.json()["project_id"]

    blueprint = client.get(f"/api/v1/projects/{project_id}/blueprint.pdf")
    assert blueprint.status_code == 200
    assert blueprint.headers["content-type"].startswith("application/pdf")
    assert "attachment;" in blueprint.headers.get("content-disposition", "")
    assert len(blueprint.content) > 50
    assert b"Furniture Plan" in blueprint.content
    assert b" re S" in blueprint.content

    bom = client.get(f"/api/v1/projects/{project_id}/bom.csv")
    assert bom.status_code == 200
    assert bom.headers["content-type"].startswith("text/csv")
    assert "section,id_or_code,kind,width,height,depth,qty" in bom.text

    nesting = client.get(f"/api/v1/projects/{project_id}/nesting.dxf")
    assert nesting.status_code == 200
    assert nesting.headers["content-type"].startswith("application/dxf")
    assert "SECTION" in nesting.text
    assert "ENTITIES" in nesting.text

    package = client.get(f"/api/v1/projects/{project_id}/export")
    assert package.status_code == 200
    assert package.headers["content-type"].startswith("application/zip")
    assert len(package.content) > 100


def test_update_project_accepts_material_thickness() -> None:
    client = TestClient(app)

    create_response = client.post("/api/v1/projects", json={"name": "Thickness Project"})
    assert create_response.status_code == 200
    project_id = create_response.json()["project_id"]

    update_response = client.patch(
        f"/api/v1/projects/{project_id}",
        json={
            "target_width": 920,
            "target_height": 1300,
            "target_depth": 500,
            "material_thickness": 22,
            "shelf_count": 3,
        },
    )
    assert update_response.status_code == 200
    product = update_response.json()["model"]["product"]
    assert product["material_thickness"] == 22
