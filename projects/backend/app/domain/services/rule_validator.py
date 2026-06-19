from __future__ import annotations

from dataclasses import dataclass
import uuid

from app.domain import ComponentSpec, HardwareSpec, ProductSpec, ProjectDesign


@dataclass
class ValidationResult:
    valid: bool
    errors: list[str]
    warnings: list[str]


class ProjectRuleValidator:
    def validate(self, model: ProjectDesign) -> ValidationResult:
        errors: list[str] = []
        warnings = list(model.warnings)

        self._validate_product(model.product, errors, warnings)
        self._validate_components(model.components, errors)
        self._validate_model_references(model, errors)
        self._validate_facade_consistency(model.product, model.components, errors)
        self._validate_hardware(model.hardware, errors)

        return ValidationResult(valid=len(errors) == 0, errors=errors, warnings=warnings)

    def _validate_product(self, product: ProductSpec, errors: list[str], warnings: list[str]) -> None:
        if product.target_width <= 0 or product.target_height <= 0 or product.target_depth <= 0:
            errors.append("All target dimensions must be positive.")
        if product.material_thickness < 12:
            warnings.append("Material thickness below 12mm may reduce rigidity.")
        if product.shelf_count < 0:
            errors.append("Shelf count cannot be negative.")
        if product.target_width <= 2 * product.material_thickness:
            errors.append("Target width must exceed two times the material thickness.")
        if product.target_height <= 2 * product.material_thickness:
            errors.append("Target height must exceed two times the material thickness.")

    def _validate_components(self, components: list[ComponentSpec], errors: list[str]) -> None:
        seen_ids: set[str] = set()
        for component in components:
            if component.id in seen_ids:
                errors.append(f"Duplicate component id '{component.id}' detected.")
            else:
                seen_ids.add(component.id)
            try:
                uuid.UUID(component.id)
            except (ValueError, TypeError, AttributeError):
                errors.append(f"Component '{component.id}' must use UUID format.")
            if component.width <= 0 or component.height <= 0 or component.depth <= 0:
                errors.append(f"Component '{component.id}' has a non-positive dimension.")

    def _validate_model_references(self, model: ProjectDesign, errors: list[str]) -> None:
        component_ids = {component.id for component in model.components}

        for joint in model.joints:
            if joint.parent_id not in component_ids:
                errors.append(
                    f"Joint references unknown parent component id '{joint.parent_id}'."
                )
            if joint.child_id not in component_ids:
                errors.append(
                    f"Joint references unknown child component id '{joint.child_id}'."
                )

        for feature in model.features:
            if feature.component_id not in component_ids:
                errors.append(
                    f"Feature references unknown component id '{feature.component_id}'."
                )

    def _validate_facade_consistency(
        self, product: ProductSpec, components: list[ComponentSpec], errors: list[str]
    ) -> None:
        door_count = sum(1 for component in components if component.kind == "door_panel")
        drawer_count = sum(1 for component in components if component.kind == "drawer_front")

        if door_count != product.door_count:
            errors.append(
                "Door facade count mismatch: "
                f"product expects {product.door_count}, model has {door_count}."
            )

        if drawer_count != product.drawer_count:
            errors.append(
                "Drawer facade count mismatch: "
                f"product expects {product.drawer_count}, model has {drawer_count}."
            )

    def _validate_hardware(self, hardware: list[HardwareSpec], errors: list[str]) -> None:
        for item in hardware:
            if item.qty <= 0:
                errors.append(f"Hardware '{item.code}' has non-positive quantity.")
