import uuid

from app.domain import ProductSpec
from app.domain.services import ModelGenerator


def test_generate_desk_profile_has_legs_and_no_shelves() -> None:
    generator = ModelGenerator()
    spec = ProductSpec(
        name="Desk A",
        inferred_type="desk",
        target_width=1200,
        target_height=750,
        target_depth=600,
        shelf_count=2,
    )

    model = generator.generate(spec)
    assert sum(1 for component in model.components if component.kind == "left_leg_front") == 1
    assert sum(1 for component in model.components if component.kind == "right_leg_front") == 1
    assert sum(1 for component in model.components if component.kind == "left_leg_back") == 1
    assert sum(1 for component in model.components if component.kind == "right_leg_back") == 1
    assert all(component.kind != "shelf" for component in model.components)
    assert len({component.id for component in model.components}) == len(model.components)
    for component in model.components:
        uuid.UUID(component.id)


def test_generate_shelf_profile_has_multiple_shelves() -> None:
    generator = ModelGenerator()
    spec = ProductSpec(
        name="Shelf A",
        inferred_type="shelf",
        target_width=900,
        target_height=1800,
        target_depth=320,
        shelf_count=4,
    )

    model = generator.generate(spec)
    shelf_components = [component for component in model.components if component.kind == "shelf"]

    assert len(shelf_components) == 4
    assert any(component.kind == "left_side" for component in model.components)
    assert any(component.kind == "right_side" for component in model.components)


def test_generate_cabinet_profile_has_back_panel() -> None:
    generator = ModelGenerator()
    spec = ProductSpec(
        name="Cabinet A",
        inferred_type="cabinet",
        target_width=800,
        target_height=1200,
        target_depth=450,
        shelf_count=2,
    )

    model = generator.generate(spec)

    assert any(component.kind == "back_panel" for component in model.components)


def test_generate_cabinet_profile_adds_facade_components_when_requested() -> None:
    generator = ModelGenerator()
    spec = ProductSpec(
        name="Cabinet Facade",
        inferred_type="cabinet",
        target_width=1000,
        target_height=2000,
        target_depth=500,
        shelf_count=3,
        door_count=3,
        drawer_count=2,
    )

    model = generator.generate(spec)
    assert sum(1 for component in model.components if component.kind == "door_panel") == 3
    assert sum(1 for component in model.components if component.kind == "drawer_front") == 2
    assert len({component.id for component in model.components}) == len(model.components)
    for component in model.components:
        uuid.UUID(component.id)


def test_generated_model_references_are_consistent() -> None:
    generator = ModelGenerator()
    spec = ProductSpec(
        name="Reference Integrity",
        inferred_type="cabinet",
        target_width=850,
        target_height=1300,
        target_depth=470,
        shelf_count=2,
        door_count=2,
        drawer_count=1,
    )

    model = generator.generate(spec)
    component_ids = {component.id for component in model.components}

    assert len(component_ids) == len(model.components)
    assert all(joint.parent_id in component_ids and joint.child_id in component_ids for joint in model.joints)
    assert all(feature.component_id in component_ids for feature in model.features)


def test_generated_cabinet_joints_distribute_repeated_parts() -> None:
    generator = ModelGenerator()
    spec = ProductSpec(
        name="Joint Distribution",
        inferred_type="cabinet",
        target_width=800,
        target_height=1200,
        target_depth=450,
        shelf_count=2,
        door_count=2,
        divider_count=1,
        drawer_count=0,
    )

    model = generator.generate(spec)
    components_by_id = {component.id: component for component in model.components}

    shelf_joint_y = sorted(
        {
            joint.pos_y
            for joint in model.joints
            if components_by_id.get(joint.child_id) and components_by_id[joint.child_id].kind == "shelf"
        }
    )
    door_joint_x = sorted(
        {
            joint.pos_x
            for joint in model.joints
            if components_by_id.get(joint.child_id) and components_by_id[joint.child_id].kind == "door_panel"
        }
    )
    divider_joint_x = [
        joint.pos_x
        for joint in model.joints
        if components_by_id.get(joint.child_id) and components_by_id[joint.child_id].kind == "divider_panel"
    ]

    assert len(shelf_joint_y) == 2
    assert shelf_joint_y[0] != shelf_joint_y[1]
    assert len(door_joint_x) == 2
    assert door_joint_x[0] < 0 < door_joint_x[1]
    assert len(divider_joint_x) == 1
    assert abs(divider_joint_x[0]) <= 1e-3


def test_generated_mixed_facade_layout_matches_asymmetric_profile() -> None:
    generator = ModelGenerator()
    spec = ProductSpec(
        name="Mixed Facade",
        inferred_type="cabinet",
        target_width=720,
        target_height=1600,
        target_depth=300,
        shelf_count=3,
        door_count=3,
        drawer_count=1,
        divider_count=1,
    )

    model = generator.generate(spec)
    components_by_id = {component.id: component for component in model.components}

    door_components = [component for component in model.components if component.kind == "door_panel"]
    drawer_components = [component for component in model.components if component.kind == "drawer_front"]
    assert len(door_components) == 3
    assert len(drawer_components) == 1

    # Mixed profile should produce non-uniform door heights.
    door_heights = sorted(component.height for component in door_components)
    assert door_heights[0] < door_heights[-1]

    door_joint_positions = sorted(
        (
            joint.pos_x,
            joint.pos_y,
        )
        for joint in model.joints
        if components_by_id.get(joint.child_id) and components_by_id[joint.child_id].kind == "door_panel"
    )
    drawer_joint_positions = [
        (joint.pos_x, joint.pos_y)
        for joint in model.joints
        if components_by_id.get(joint.child_id) and components_by_id[joint.child_id].kind == "drawer_front"
    ]

    assert len(door_joint_positions) == 3
    assert len(drawer_joint_positions) == 1
    # One right-column door and two left-column fronts should produce at least two distinct x anchors.
    assert len({round(value[0], 3) for value in door_joint_positions}) >= 2
    # Top-left door should be above bottom-left door (different y anchors).
    assert len({round(value[1], 3) for value in door_joint_positions}) >= 2

    divider_joint_positions = [
        (joint.pos_x, joint.pos_y)
        for joint in model.joints
        if components_by_id.get(joint.child_id) and components_by_id[joint.child_id].kind == "divider_panel"
    ]
    assert len(divider_joint_positions) == 1
    assert divider_joint_positions[0][0] > 0

    door_joints = [
        joint
        for joint in model.joints
        if components_by_id.get(joint.child_id) and components_by_id[joint.child_id].kind == "door_panel"
    ]
    assert len(door_joints) == 3

    # Mixed layout semantics: highest door top-hinged, lower-left door left-hinged, right door right-hinged.
    highest_door_joint = max(door_joints, key=lambda joint: joint.pos_y)
    highest_parent_kind = components_by_id[highest_door_joint.parent_id].kind
    assert highest_parent_kind == "top_panel"

    left_column_lower = [
        joint
        for joint in door_joints
        if joint.pos_x < 0 and joint.pos_y < highest_door_joint.pos_y
    ]
    assert left_column_lower
    assert components_by_id[left_column_lower[0].parent_id].kind == "left_side"

    right_column = [joint for joint in door_joints if joint.pos_x > 0]
    assert right_column
    assert components_by_id[right_column[0].parent_id].kind == "right_side"


def test_two_door_layout_uses_left_and_right_hinges() -> None:
    generator = ModelGenerator()
    spec = ProductSpec(
        name="Two Door Hinges",
        inferred_type="cabinet",
        target_width=800,
        target_height=1200,
        target_depth=450,
        shelf_count=2,
        door_count=2,
        drawer_count=0,
        divider_count=1,
    )

    model = generator.generate(spec)
    components_by_id = {component.id: component for component in model.components}
    door_joints = [
        joint
        for joint in model.joints
        if components_by_id.get(joint.child_id) and components_by_id[joint.child_id].kind == "door_panel"
    ]

    assert len(door_joints) == 2
    parent_kinds = {components_by_id[joint.parent_id].kind for joint in door_joints}
    assert "left_side" in parent_kinds
    assert "right_side" in parent_kinds
