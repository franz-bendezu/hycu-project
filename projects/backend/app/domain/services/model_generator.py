from __future__ import annotations

from app.domain import ComponentSpec, HardwareSpec, ProductSpec, ProjectDesign
from app.domain.services.rule_validator import ProjectRuleValidator, ValidationResult


class ModelGenerator:
    def __init__(self) -> None:
        self.validator = ProjectRuleValidator()

    def generate(self, spec: ProductSpec) -> ProjectDesign:
        components = self._build_components(spec)
        hardware = self._build_hardware(spec)
        warnings = self._generate_warnings(spec)
        return ProjectDesign(product=spec, components=components, hardware=hardware, warnings=warnings)

    def validate(self, model: ProjectDesign) -> ValidationResult:
        return self.validator.validate(model)

    def apply_update(self, model: ProjectDesign, **updates: float) -> ProjectDesign:
        spec = model.product.model_copy(deep=True)
        for key, value in updates.items():
            if value is not None:
                setattr(spec, key, value)
        return self.generate(spec)

    def _build_components(self, spec: ProductSpec) -> list[ComponentSpec]:
        profile = self._profile_for_spec(spec)
        if profile == "desk":
            return self._build_desk_components(spec)
        if profile == "shelf":
            return self._build_shelf_components(spec)
        return self._build_cabinet_components(spec)

    def _profile_for_spec(self, spec: ProductSpec) -> str:
        profile = (getattr(spec, "inferred_type", "") or "").strip().lower()
        if profile in {"cabinet", "desk", "shelf"}:
            return profile
        # Legacy fallback for models created before inferred_type existed.
        if spec.shelf_count == 0:
            return "desk"
        if spec.shelf_count >= 4:
            return "shelf"
        return "cabinet"

    def _build_cabinet_components(self, spec: ProductSpec) -> list[ComponentSpec]:
        inner_width = spec.target_width - (2 * spec.material_thickness)
        shelf_depth = max(spec.target_depth - 2, 1)
        components = [
            ComponentSpec(
                id="left_side",
                kind="left_side",
                width=spec.material_thickness,
                height=spec.target_height,
                depth=spec.target_depth,
            ),
            ComponentSpec(
                id="right_side",
                kind="right_side",
                width=spec.material_thickness,
                height=spec.target_height,
                depth=spec.target_depth,
            ),
            ComponentSpec(
                id="top",
                kind="top_panel",
                width=inner_width,
                height=spec.material_thickness,
                depth=spec.target_depth,
            ),
            ComponentSpec(
                id="bottom",
                kind="bottom_panel",
                width=inner_width,
                height=spec.material_thickness,
                depth=spec.target_depth,
            ),
            ComponentSpec(
                id="back_panel",
                kind="back_panel",
                width=max(inner_width, 1),
                height=max(spec.target_height - (2 * spec.material_thickness), 1),
                depth=spec.material_thickness,
            ),
        ]

        for idx in range(spec.shelf_count):
            components.append(
                ComponentSpec(
                    id=f"shelf_{idx + 1}",
                    kind="shelf",
                    width=inner_width,
                    height=spec.material_thickness,
                    depth=shelf_depth,
                )
            )
        return components

    def _build_shelf_components(self, spec: ProductSpec) -> list[ComponentSpec]:
        # Open shelf variant: no full back panel by default, more shelf levels.
        inner_width = spec.target_width - (2 * spec.material_thickness)
        shelf_depth = max(spec.target_depth - 2, 1)
        components = [
            ComponentSpec(
                id="left_side",
                kind="left_side",
                width=spec.material_thickness,
                height=spec.target_height,
                depth=spec.target_depth,
            ),
            ComponentSpec(
                id="right_side",
                kind="right_side",
                width=spec.material_thickness,
                height=spec.target_height,
                depth=spec.target_depth,
            ),
            ComponentSpec(
                id="top",
                kind="top_panel",
                width=inner_width,
                height=spec.material_thickness,
                depth=spec.target_depth,
            ),
            ComponentSpec(
                id="bottom",
                kind="bottom_panel",
                width=inner_width,
                height=spec.material_thickness,
                depth=spec.target_depth,
            ),
        ]

        for idx in range(spec.shelf_count):
            components.append(
                ComponentSpec(
                    id=f"shelf_{idx + 1}",
                    kind="shelf",
                    width=inner_width,
                    height=spec.material_thickness,
                    depth=shelf_depth,
                )
            )
        return components

    def _build_desk_components(self, spec: ProductSpec) -> list[ComponentSpec]:
        top_thickness = spec.material_thickness
        leg_width = max(spec.material_thickness * 1.4, 40)
        leg_depth = max(spec.material_thickness * 1.4, 40)
        leg_height = max(spec.target_height - top_thickness, 1)
        inset_x = max(spec.material_thickness * 1.8, 30)
        inset_z = max(spec.material_thickness * 1.2, 20)
        span_width = max(spec.target_width - (2 * inset_x + 2 * leg_width), 80)

        return [
            ComponentSpec(
                id="top",
                kind="top_panel",
                width=spec.target_width,
                height=top_thickness,
                depth=spec.target_depth,
            ),
            ComponentSpec(
                id="left_leg_front",
                kind="left_leg_front",
                width=leg_width,
                height=leg_height,
                depth=leg_depth,
            ),
            ComponentSpec(
                id="right_leg_front",
                kind="right_leg_front",
                width=leg_width,
                height=leg_height,
                depth=leg_depth,
            ),
            ComponentSpec(
                id="left_leg_back",
                kind="left_leg_back",
                width=leg_width,
                height=leg_height,
                depth=leg_depth,
            ),
            ComponentSpec(
                id="right_leg_back",
                kind="right_leg_back",
                width=leg_width,
                height=leg_height,
                depth=leg_depth,
            ),
            ComponentSpec(
                id="back_apron",
                kind="back_panel",
                width=span_width,
                height=max(spec.material_thickness * 2, 36),
                depth=spec.material_thickness,
            ),
            ComponentSpec(
                id="front_apron",
                kind="front_panel",
                width=span_width,
                height=max(spec.material_thickness * 2, 36),
                depth=spec.material_thickness,
            ),
        ]

    def _build_hardware(self, spec: ProductSpec) -> list[HardwareSpec]:
        profile = self._profile_for_spec(spec)
        if profile == "desk":
            return [
                HardwareSpec(code="CORNER_BRACKET_40", qty=8),
                HardwareSpec(code="WOOD_SCREW_4X40", qty=24),
            ]
        if profile == "shelf":
            return [
                HardwareSpec(code="CAM_LOCK_15MM", qty=max(12, spec.shelf_count * 4)),
                HardwareSpec(code="WOOD_SCREW_4X40", qty=max(20, spec.shelf_count * 8)),
            ]
        return [
            HardwareSpec(code="CAM_LOCK_15MM", qty=max(8, spec.shelf_count * 4)),
            HardwareSpec(code="WOOD_SCREW_4X40", qty=max(16, spec.shelf_count * 8)),
        ]

    def _generate_warnings(self, spec: ProductSpec) -> list[str]:
        shelf_spacing = (spec.target_height - (2 * spec.material_thickness)) / max(spec.shelf_count + 1, 1)
        warnings: list[str] = []
        if shelf_spacing > 420:
            warnings.append("Shelf spacing is large; consider adding a divider or extra shelf.")
        return warnings
