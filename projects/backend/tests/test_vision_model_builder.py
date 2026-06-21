from __future__ import annotations

from app.domain.services.vision_model_builder import build_project_model_from_inference


def test_build_project_model_distributes_joints_for_repeated_components() -> None:
    result = {
        "detected_type": "cabinet",
        "suggested_width": 900,
        "suggested_height": 1600,
        "suggested_depth": 500,
        "components": [
            {"id": "side_src", "name": "side_panel", "kind": "panel", "quantity": 2},
            {"id": "door_src", "name": "door_panel", "kind": "panel", "quantity": 3},
            {"id": "top_src", "name": "top_panel", "kind": "panel", "quantity": 1},
        ],
        "joints": [
            {
                "parent_component_id": "side_src",
                "child_component_id": "door_src",
                "joint_type": "hinge",
                "count": 6,
            },
            {
                "parent_component_id": "side_src",
                "child_component_id": "top_src",
                "joint_type": "cam_lock",
                "count": 2,
            },
        ],
    }

    model = build_project_model_from_inference(result, project_name="Joint Mapping")
    components_by_id = {component.id: component for component in model.components}

    door_ids = {component.id for component in model.components if component.kind == "door_panel"}
    side_ids = {component.id for component in model.components if component.kind in {"left_side", "right_side"}}
    top_ids = {component.id for component in model.components if component.kind == "top_panel"}

    assert len(door_ids) == 3
    assert len(side_ids) == 2
    assert len(top_ids) == 1

    door_joints = [joint for joint in model.joints if joint.child_id in door_ids]
    assert len({joint.child_id for joint in door_joints}) == 3
    assert {joint.parent_id for joint in door_joints}.issubset(side_ids)
    assert len({components_by_id[joint.parent_id].kind for joint in door_joints}) == 2

    top_joints = [joint for joint in model.joints if joint.child_id in top_ids]
    assert len(top_joints) == 2
    assert {joint.parent_id for joint in top_joints} == side_ids


def test_build_project_model_preserves_extended_inferred_type() -> None:
    result = {
        "detected_type": "wardrobe",
        "suggested_width": 1000,
        "suggested_height": 2100,
        "suggested_depth": 550,
        "components": [],
    }

    model = build_project_model_from_inference(result, project_name="Wardrobe Type")

    assert model.product.inferred_type == "wardrobe"


def test_build_project_model_does_not_double_count_drawer_box_as_front() -> None:
    result = {
        "detected_type": "desk",
        "suggested_width": 1000,
        "suggested_height": 975,
        "suggested_depth": 525,
        "components": [
            {"name": "top_panel", "kind": "panel", "quantity": 1},
            {"name": "side_panel", "kind": "panel", "quantity": 2},
            {"name": "front_apron", "kind": "panel", "quantity": 1},
            {"name": "drawer_front", "kind": "panel", "quantity": 4},
            {"name": "drawer_box", "kind": "panel", "quantity": 4},
        ],
    }

    model = build_project_model_from_inference(result, project_name="Desk Drawers")

    drawer_front_components = [component for component in model.components if component.kind == "drawer_front"]
    assert len(drawer_front_components) == 4
    assert model.product.drawer_count == 4
