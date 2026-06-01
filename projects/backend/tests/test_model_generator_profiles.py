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
    ids = {component.id for component in model.components}

    assert "left_leg_front" in ids
    assert "right_leg_front" in ids
    assert "left_leg_back" in ids
    assert "right_leg_back" in ids
    assert all("shelf_" not in component.id for component in model.components)


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
    shelf_components = [component for component in model.components if component.id.startswith("shelf_")]

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

    assert any(component.id == "back_panel" for component in model.components)
