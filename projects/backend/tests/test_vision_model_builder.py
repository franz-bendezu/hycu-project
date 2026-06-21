from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.domain.services.vision_model_builder import build_project_model_from_inference
from app.presentation.schemas.inference import InferenceOutput


def _strict_payload() -> dict:
    return {
        "detected_type": "cabinet",
        "confidence": 0.94,
        "suggested_width": 900,
        "suggested_height": 1600,
        "suggested_depth": 500,
        "image_url": "https://example.com/cabinet.jpg",
        "components": [
            {
                "id": "11111111-1111-4111-8111-111111111111",
                "kind": "left_side",
                "width": 18.0,
                "height": 1600.0,
                "depth": 500.0,
            },
            {
                "id": "22222222-2222-4222-8222-222222222222",
                "kind": "door_panel",
                "width": 430.0,
                "height": 1500.0,
                "depth": 9.0,
            },
        ],
        "faces": [
            {
                "id": "11111111-1111-4111-8111-111111111111:+z:aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
                "component_id": "11111111-1111-4111-8111-111111111111",
                "normal": "+z",
            },
            {
                "id": "11111111-1111-4111-8111-111111111111:-z:bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb",
                "component_id": "11111111-1111-4111-8111-111111111111",
                "normal": "-z",
            },
            {
                "id": "22222222-2222-4222-8222-222222222222:+z:cccccccc-cccc-4ccc-8ccc-cccccccccccc",
                "component_id": "22222222-2222-4222-8222-222222222222",
                "normal": "+z",
            },
            {
                "id": "22222222-2222-4222-8222-222222222222:-z:dddddddd-dddd-4ddd-8ddd-dddddddddddd",
                "component_id": "22222222-2222-4222-8222-222222222222",
                "normal": "-z",
            },
        ],
        "joints": [
            {
                "id": "33333333-3333-4333-8333-333333333333",
                "parent_face_id": "11111111-1111-4111-8111-111111111111:+z:aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
                "child_face_id": "22222222-2222-4222-8222-222222222222:-z:dddddddd-dddd-4ddd-8ddd-dddddddddddd",
                "joint_rule": "mount",
                "offset_u": -110.0,
                "offset_v": 0.0,
                "clearance": 0.0,
            }
        ],
    }


def test_build_project_model_accepts_strict_face_contract() -> None:
    inference = InferenceOutput.model_validate(_strict_payload())

    model = build_project_model_from_inference(inference, project_name="Strict Vision")

    assert model.product.inferred_type == "cabinet"
    assert len(model.components) == 2
    assert len(model.joints) == 1
    assert model.joints[0].joint_rule == "mount"


def test_build_project_model_requires_faces() -> None:
    result = _strict_payload()
    result.pop("faces")

    with pytest.raises(ValidationError):
        InferenceOutput.model_validate(result)


def test_build_project_model_rejects_joint_with_unknown_face() -> None:
    payload = _strict_payload()
    payload["joints"][0]["child_face_id"] = "missing:face:id"
    inference = InferenceOutput.model_validate(payload)

    with pytest.raises(ValueError, match="unknown faces"):
        build_project_model_from_inference(inference, project_name="Strict Vision")


def test_build_project_model_synthesizes_structure_and_connectivity_for_sparse_desk() -> None:
    payload = {
        "detected_type": "desk",
        "confidence": 0.88,
        "suggested_width": 1000,
        "suggested_height": 975,
        "suggested_depth": 525,
        "image_url": "https://example.com/desk.jpg",
        "components": [
            {"id": "51f6e560-4557-56a3-afbd-be12ddeb4268", "kind": "left_side", "width": 18.0, "height": 897.0, "depth": 483.0},
            {"id": "c8ddeb2f-c887-570e-bc8e-547b6f78d203", "kind": "top_panel", "width": 1000.0, "height": 18.0, "depth": 525.0},
            {"id": "32034ade-c825-591b-adfd-12fb3fd368f0", "kind": "divider_panel", "width": 18.0, "height": 760.5, "depth": 472.5},
            {"id": "8221b393-fd44-52bf-b535-115f3d1338db", "kind": "drawer_front", "width": 220.0, "height": 136.5, "depth": 18.0},
            {"id": "42b3f577-32fb-5684-bf69-a312bbe44249", "kind": "front_panel", "width": 220.0, "height": 136.5, "depth": 18.0},
        ],
        "faces": [
            {"id": "51f6e560-4557-56a3-afbd-be12ddeb4268:+x:eba52720-6a1d-5b0c-8713-fb40a8475b34", "component_id": "51f6e560-4557-56a3-afbd-be12ddeb4268", "normal": "+x"},
            {"id": "51f6e560-4557-56a3-afbd-be12ddeb4268:-x:c62026e0-0ac8-59d8-882e-77aff323fe86", "component_id": "51f6e560-4557-56a3-afbd-be12ddeb4268", "normal": "-x"},
            {"id": "c8ddeb2f-c887-570e-bc8e-547b6f78d203:+x:37787f10-b65a-5c17-b764-4871165a3628", "component_id": "c8ddeb2f-c887-570e-bc8e-547b6f78d203", "normal": "+x"},
            {"id": "c8ddeb2f-c887-570e-bc8e-547b6f78d203:-x:45026063-0aae-59b1-9df9-05f9297aa1bd", "component_id": "c8ddeb2f-c887-570e-bc8e-547b6f78d203", "normal": "-x"},
            {"id": "32034ade-c825-591b-adfd-12fb3fd368f0:+x:1989b7cc-ca90-5420-9fcd-dda9ee65046d", "component_id": "32034ade-c825-591b-adfd-12fb3fd368f0", "normal": "+x"},
            {"id": "32034ade-c825-591b-adfd-12fb3fd368f0:-x:fb31e3e6-531f-5db2-b9db-8d7fbf5c17fa", "component_id": "32034ade-c825-591b-adfd-12fb3fd368f0", "normal": "-x"},
            {"id": "8221b393-fd44-52bf-b535-115f3d1338db:+z:05bd39a9-c650-5772-a1f5-36877e5cd9c6", "component_id": "8221b393-fd44-52bf-b535-115f3d1338db", "normal": "+z"},
            {"id": "8221b393-fd44-52bf-b535-115f3d1338db:-z:63ad0850-e10a-5b77-854b-1ec0985d7ab7", "component_id": "8221b393-fd44-52bf-b535-115f3d1338db", "normal": "-z"},
            {"id": "42b3f577-32fb-5684-bf69-a312bbe44249:+z:6c1ec9d2-764e-5d9e-b29f-a46a600ea6f5", "component_id": "42b3f577-32fb-5684-bf69-a312bbe44249", "normal": "+z"},
            {"id": "42b3f577-32fb-5684-bf69-a312bbe44249:-z:4ceb625b-3e58-54f6-bccd-f8174a974b0f", "component_id": "42b3f577-32fb-5684-bf69-a312bbe44249", "normal": "-z"},
        ],
        "joints": [
            {
                "id": "7480e1e4-938a-5145-9e23-441ada2578e0",
                "parent_face_id": "51f6e560-4557-56a3-afbd-be12ddeb4268:+x:eba52720-6a1d-5b0c-8713-fb40a8475b34",
                "child_face_id": "c8ddeb2f-c887-570e-bc8e-547b6f78d203:-x:45026063-0aae-59b1-9df9-05f9297aa1bd",
                "joint_rule": "overlap",
                "offset_u": 0,
                "offset_v": 0,
                "clearance": 0,
            }
        ],
    }

    inference = InferenceOutput.model_validate(payload)
    model = build_project_model_from_inference(inference, project_name="Sparse Desk")

    right_sides = [component for component in model.components if component.kind.value == "right_side"]
    assert len(right_sides) == 1
    assert len(model.joints) >= 3
