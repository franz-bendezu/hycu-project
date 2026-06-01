from __future__ import annotations

from dataclasses import dataclass

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
        for component in components:
            if component.width <= 0 or component.height <= 0 or component.depth <= 0:
                errors.append(f"Component '{component.id}' has a non-positive dimension.")

    def _validate_hardware(self, hardware: list[HardwareSpec], errors: list[str]) -> None:
        for item in hardware:
            if item.qty <= 0:
                errors.append(f"Hardware '{item.code}' has non-positive quantity.")
