from __future__ import annotations

from app.domain.services.product_types import normalize_detected_product_type, shelf_default_for_type
from app.presentation.schemas.project_design import ProjectModel


class InferenceProjectionService:
    @staticmethod
    def component_quantity(result: dict, component_name: str) -> int:
        components = result.get("components")
        if not isinstance(components, list):
            return 0

        total = 0
        for component in components:
            if not isinstance(component, dict):
                continue
            name = str(component.get("name", "")).strip().lower()
            if name != component_name:
                continue
            try:
                qty = int(component.get("quantity", 0))
            except (TypeError, ValueError):
                qty = 0
            total += max(qty, 0)
        return total

    @staticmethod
    def inferred_type_from_components(result: dict) -> str | None:
        components = result.get("components")
        if not isinstance(components, list):
            return None

        names = {
            str(component.get("name", "")).strip().lower()
            for component in components
            if isinstance(component, dict)
        }

        if any(name in {"desktop", "front_apron"} for name in names):
            return "desk"
        if any(name in {"tv_stand_body", "tv_stand"} for name in names):
            return "tv_stand"
        if any(name in {"sideboard_body", "sideboard"} for name in names):
            return "sideboard"
        if any("door" in name for name in names) or any("drawer" in name for name in names):
            return "cabinet"
        if "shelf_panel" in names and not any("door" in name for name in names):
            return "shelf"
        return None

    @staticmethod
    def inferred_type_from_raw_detections(result: dict) -> str | None:
        image_results = result.get("image_results")
        if not isinstance(image_results, list):
            return None

        labels: list[str] = []
        for evidence in image_results:
            if not isinstance(evidence, dict):
                continue
            raw_detections = evidence.get("raw_detections")
            if not isinstance(raw_detections, list):
                continue
            for det in raw_detections:
                if not isinstance(det, dict):
                    continue
                label = str(det.get("label", "")).strip().lower()
                if label:
                    labels.append(label)

        if not labels:
            return None

        if any(label in {"tv_stand_body", "tv_stand"} for label in labels):
            return "tv_stand"
        if any(label in {"sideboard_body", "sideboard"} for label in labels):
            return "sideboard"
        if any("drawer" in label for label in labels) or any("door" in label for label in labels):
            return "cabinet"
        if any(label in {"desktop", "front_apron"} for label in labels):
            return "desk"
        if any("shelf" in label for label in labels):
            return "shelf"
        return None

    @staticmethod
    def inferred_type_from_detected_type(detected_type: str | None) -> str | None:
        return normalize_detected_product_type(detected_type)

    @classmethod
    def infer_model_type(cls, result: dict, fallback: str | None = None) -> str:
        from_components = cls.inferred_type_from_components(result)
        if from_components:
            return from_components

        from_raw = cls.inferred_type_from_raw_detections(result)
        if from_raw:
            return from_raw

        detected = cls.inferred_type_from_detected_type(result.get("detected_type"))
        if detected:
            return detected

        if fallback is not None:
            return fallback
        raise ValueError("Unable to infer model type from components, detections, or detected_type")

    @staticmethod
    def shelf_count_from_detected_type(detected_type: str | None) -> int:
        return shelf_default_for_type(detected_type)

    @classmethod
    def shelf_count_from_inference(cls, result: dict, inferred_type: str) -> int:
        shelf_qty = cls.component_quantity(result, "shelf_panel")
        if shelf_qty > 0:
            return shelf_qty
        return cls.shelf_count_from_detected_type(inferred_type)

    @staticmethod
    def safe_float(value: object, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def normalized_center_from_box(
        box: object,
        image_width: float,
        image_height: float,
    ) -> tuple[float, float] | None:
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            return None

        try:
            a, b, c, d = (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
        except (TypeError, ValueError):
            return None

        if c >= a and d >= b:
            x_center = (a + c) / 2.0
            y_center = (b + d) / 2.0
        else:
            x_center = a
            y_center = b

        width = max(float(image_width), 1.0)
        height = max(float(image_height), 1.0)
        return (
            min(max(x_center / width, 0.0), 1.0),
            min(max(y_center / height, 0.0), 1.0),
        )

    @classmethod
    def layout_axis_from_raw_detections(cls, result: dict, tokens: tuple[str, ...], axis: str) -> list[float]:
        image_results = result.get("image_results")
        if not isinstance(image_results, list):
            return []

        values: list[float] = []
        for evidence in image_results:
            if not isinstance(evidence, dict):
                continue
            raw_detections = evidence.get("raw_detections")
            if not isinstance(raw_detections, list):
                continue
            width_px = cls.safe_float(evidence.get("width_px"), default=1.0)
            height_px = cls.safe_float(evidence.get("height_px"), default=1.0)

            for detection in raw_detections:
                if not isinstance(detection, dict):
                    continue
                label = str(detection.get("label", "")).strip().lower()
                if not label or not any(token in label for token in tokens):
                    continue
                center = cls.normalized_center_from_box(detection.get("box"), width_px, height_px)
                if center is None:
                    continue
                values.append(center[0] if axis == "x" else center[1])

        return sorted(values)

    @classmethod
    def layout_axis_from_components(cls, result: dict, tokens: tuple[str, ...], axis: str) -> list[float]:
        components = result.get("components")
        if not isinstance(components, list):
            return []

        values: list[float] = []
        for component in components:
            if not isinstance(component, dict):
                continue
            name = str(component.get("name", "")).strip().lower()
            if not name or not any(token in name for token in tokens):
                continue

            center = cls.normalized_center_from_box(component.get("box_corners"), 1.0, 1.0)
            if center is None:
                continue

            try:
                quantity = max(int(component.get("quantity", 1)), 1)
            except (TypeError, ValueError):
                quantity = 1
            values.extend([center[0] if axis == "x" else center[1]] * quantity)

        return sorted(values)

    @staticmethod
    def fit_anchor_count(anchors: list[float], count: int) -> list[float]:
        if count <= 0:
            return []
        if not anchors:
            return []

        ordered = sorted(anchors)
        if len(ordered) == count:
            return ordered
        if len(ordered) == 1:
            return [ordered[0]] * count

        result: list[float] = []
        source_last = len(ordered) - 1
        target_last = max(count - 1, 1)
        for idx in range(count):
            source_idx = int(round((idx * source_last) / target_last))
            source_idx = min(max(source_idx, 0), source_last)
            result.append(ordered[source_idx])
        return result

    @classmethod
    def apply_inference_layout_to_joints(cls, model: ProjectModel, result: dict) -> None:
        components_by_id = {component.id: component for component in model.components}

        def _component_id_from_face_id(face_id: str) -> str:
            return face_id.split(":", 1)[0]

        inner_width = max(model.product.target_width - (2 * model.product.material_thickness), 1.0)
        usable_height = max(model.product.target_height - (2 * model.product.material_thickness), 1.0)

        kind_tokens_axis: list[tuple[str, tuple[str, ...], str]] = [
            ("door_panel", ("door", "wardrobe"), "x"),
            ("divider_panel", ("divider", "partition", "vertical"), "x"),
            ("shelf", ("shelf",), "y"),
            ("drawer_front", ("drawer",), "y"),
        ]

        target_values: dict[str, dict[str, float]] = {}
        for kind_name, tokens, axis in kind_tokens_axis:
            child_ids = sorted(
                component.id
                for component in model.components
                if getattr(component.kind, "value", str(component.kind)) == kind_name
            )
            if not child_ids:
                continue

            anchors = cls.layout_axis_from_raw_detections(result, tokens=tokens, axis=axis)
            if not anchors:
                anchors = cls.layout_axis_from_components(result, tokens=tokens, axis=axis)
            fitted = cls.fit_anchor_count(anchors, len(child_ids))
            if not fitted:
                continue

            axis_map = {child_id: fitted[idx] for idx, child_id in enumerate(child_ids)}
            target_values[kind_name] = axis_map

        for joint in model.joints:
            component = components_by_id.get(_component_id_from_face_id(joint.child_face_id))
            if component is None:
                continue
            kind_name = getattr(component.kind, "value", str(component.kind))
            axis_map = target_values.get(kind_name)
            if not axis_map:
                continue
            anchor = axis_map.get(component.id)
            if anchor is None:
                continue

            if kind_name in {"door_panel", "divider_panel"}:
                joint.offset_u = round((anchor - 0.5) * inner_width, 3)
            elif kind_name in {"shelf", "drawer_front"}:
                joint.offset_v = round((0.5 - anchor) * usable_height, 3)


