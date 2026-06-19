from __future__ import annotations

import base64
import csv
import datetime
import io
import json
import uuid
import zipfile

from fastapi import APIRouter, Depends, File, HTTPException, Response, UploadFile
from sqlalchemy.orm import Session

from app.infrastructure.persistence.database import get_db
from app.infrastructure.gateways.inference_gateway import (
    InferenceGatewayError,
    get_inference_gateway,
)
from app.domain.services import ModelGenerator
from app.infrastructure.persistence.models import (
    Job as JobModel,
    JobAssetResult,
    Project,
    ProjectAsset,
)
from app.presentation.schemas.project_design import ProductSpec, ProjectModel
from app.presentation.schemas.projects import (
    CreateProjectAssetResponse,
    CreateProjectJobRequest,
    CreateProjectJobResponse,
    CreateProjectRequest,
    CreateProjectResponse,
    ProjectAssetResponse,
    ProjectAssetsResponse,
    ProjectJobsResponse,
    ProjectResponse,
    ProjectSummary,
    ProjectsListResponse,
    UpdateProjectRequest,
    ValidateResponse,
)
from app.presentation.schemas.jobs import JobResponse

router = APIRouter(prefix="/api/v1/projects", tags=["projects"])
generator = ModelGenerator()
client = get_inference_gateway()


def _shelf_count_from_detected_type(detected_type: str | None) -> int:
    normalized = (detected_type or "").strip().lower()
    if normalized == "desk":
        return 0
    if normalized == "cabinet":
        return 2
    if normalized == "shelf":
        return 4
    return 0


def _component_quantity(result: dict, component_name: str) -> int:
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


def _component_quantity_by_tokens(result: dict, tokens: tuple[str, ...]) -> int:
    components = result.get("components")
    if not isinstance(components, list):
        return 0

    total = 0
    for component in components:
        if not isinstance(component, dict):
            continue
        name = str(component.get("name", "")).strip().lower()
        if not name or not any(token in name for token in tokens):
            continue
        try:
            qty = int(component.get("quantity", 0))
        except (TypeError, ValueError):
            qty = 0
        total += max(qty, 0)

    return total


def _raw_detection_count_by_tokens(result: dict, tokens: tuple[str, ...]) -> int:
    image_results = result.get("image_results")
    if not isinstance(image_results, list):
        return 0

    count = 0
    for evidence in image_results:
        if not isinstance(evidence, dict):
            continue
        raw_detections = evidence.get("raw_detections")
        if not isinstance(raw_detections, list):
            continue
        for detection in raw_detections:
            if not isinstance(detection, dict):
                continue
            label = str(detection.get("label", "")).strip().lower()
            if label and any(token in label for token in tokens):
                count += 1

    return count


def _facade_counts_from_inference(result: dict) -> tuple[int, int]:
    door_tokens = ("door", "wardrobe", "porta")
    drawer_tokens = ("drawer", "gaveta", "gavetete")
    generic_front_tokens = ("front_panel", "frontpanel", "facade", "frente", "panel_front")

    door_count = _component_quantity_by_tokens(result, door_tokens)
    drawer_count = _component_quantity_by_tokens(result, drawer_tokens)
    generic_front_count = _component_quantity_by_tokens(result, generic_front_tokens)

    if door_count == 0:
        door_count = _raw_detection_count_by_tokens(result, door_tokens)
    if drawer_count == 0:
        drawer_count = _raw_detection_count_by_tokens(result, drawer_tokens)
    if generic_front_count == 0:
        generic_front_count = _raw_detection_count_by_tokens(result, generic_front_tokens)

    explicit_total = door_count + drawer_count
    if generic_front_count > explicit_total:
        remaining = generic_front_count - explicit_total
        # Mixed cabinets often yield generic front detections. Keep at least one
        # drawer for 4+ fronts when no explicit drawer evidence is present.
        if drawer_count == 0 and generic_front_count >= 4:
            drawer_count = 1
            remaining = max(generic_front_count - (door_count + drawer_count), 0)
        door_count += remaining

    # Keep output bounded to avoid exploding model complexity on noisy detections.
    return min(max(door_count, 0), 8), min(max(drawer_count, 0), 8)


def _facade_evidence_from_inference(result: dict) -> dict[str, int]:
    door_tokens = ("door", "wardrobe", "porta")
    drawer_tokens = ("drawer", "gaveta", "gavetete")
    generic_front_tokens = ("front_panel", "frontpanel", "facade", "frente", "panel_front")

    explicit_door_count = _component_quantity_by_tokens(result, door_tokens)
    explicit_drawer_count = _component_quantity_by_tokens(result, drawer_tokens)
    generic_front_count = _component_quantity_by_tokens(result, generic_front_tokens)

    raw_door_count = _raw_detection_count_by_tokens(result, door_tokens)
    raw_drawer_count = _raw_detection_count_by_tokens(result, drawer_tokens)
    raw_generic_front_count = _raw_detection_count_by_tokens(result, generic_front_tokens)

    return {
        "explicit_door_count": max(explicit_door_count, 0),
        "explicit_drawer_count": max(explicit_drawer_count, 0),
        "generic_front_count": max(generic_front_count, 0),
        "raw_door_count": max(raw_door_count, 0),
        "raw_drawer_count": max(raw_drawer_count, 0),
        "raw_generic_front_count": max(raw_generic_front_count, 0),
    }


def _divider_count_from_inference(
    result: dict,
    inferred_type: str,
    door_count: int,
    drawer_count: int,
) -> int:
    divider_tokens = ("divider", "partition", "vertical")
    divider_count = _component_quantity_by_tokens(result, divider_tokens)
    if divider_count == 0:
        divider_count = _raw_detection_count_by_tokens(result, divider_tokens)

    # Fallback: many cabinet photos expose multiple door fronts but the detector
    # may miss explicit divider labels. Approximate internal partitions from door layout.
    if divider_count == 0 and inferred_type == "cabinet" and drawer_count == 0 and door_count >= 2:
        divider_count = max(1, door_count // 2)

    return min(max(divider_count, 0), 6)


def _has_divider_evidence(result: dict) -> bool:
    divider_tokens = ("divider", "partition", "vertical")
    component_hits = _component_quantity_by_tokens(result, divider_tokens)
    if component_hits > 0:
        return True
    return _raw_detection_count_by_tokens(result, divider_tokens) > 0


def _normalize_facade_counts(
    *,
    inferred_type: str,
    door_count: int,
    drawer_count: int,
    divider_count: int,
    has_divider_evidence: bool,
    shelf_count: int,
    evidence: dict[str, int],
    images_analyzed: int,
) -> tuple[int, int]:
    normalized_doors = max(door_count, 0)
    normalized_drawers = max(drawer_count, 0)

    explicit_doors = max(evidence.get("explicit_door_count", 0), 0)
    explicit_drawers = max(evidence.get("explicit_drawer_count", 0), 0)
    generic_fronts = max(evidence.get("generic_front_count", 0), 0)
    raw_drawers = max(evidence.get("raw_drawer_count", 0), 0)
    raw_generic_fronts = max(evidence.get("raw_generic_front_count", 0), 0)

    # Heuristic 1: Strong explicit mixed evidence from generic fronts.
    if inferred_type == "cabinet" and normalized_drawers == 0:
        front_signal = max(generic_fronts, raw_generic_fronts)
        if front_signal >= 4:
            normalized_drawers = 1
            normalized_doors = max(normalized_doors, front_signal - 1)

    # Heuristic 2: Divider-backed mixed layout with weak drawer detections.
    if (
        inferred_type == "cabinet"
        and normalized_doors == 2
        and normalized_drawers == 0
        and divider_count >= 1
        and has_divider_evidence
        and shelf_count >= 2
    ):
        normalized_doors = 3
        normalized_drawers = 1

    # Heuristic 3: Over-detected doors on mixed facades often include one drawer front.
    if (
        inferred_type == "cabinet"
        and normalized_drawers == 0
        and normalized_doors >= 4
        and shelf_count >= 2
    ):
        normalized_drawers = 1
        normalized_doors = max(2, normalized_doors - 1)

    return min(normalized_doors, 8), min(normalized_drawers, 8)


def _inferred_type_from_components(result: dict) -> str | None:
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
    if any("door" in name for name in names) or any("drawer" in name for name in names):
        return "cabinet"
    if "shelf_panel" in names and not any("door" in name for name in names):
        return "shelf"
    return None


def _inferred_type_from_raw_detections(result: dict) -> str | None:
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

    if any("drawer" in label for label in labels) or any("door" in label for label in labels):
        return "cabinet"
    if any(label in {"desktop", "front_apron"} for label in labels):
        return "desk"
    if any("shelf" in label for label in labels):
        return "shelf"
    return None


def _inferred_type_from_detected_type(detected_type: str | None) -> str | None:
    normalized = (detected_type or "").strip().lower()
    if normalized in {"cabinet", "desk", "shelf"}:
        return normalized
    return None


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _infer_model_type(result: dict, fallback: str = "cabinet") -> str:
    from_components = _inferred_type_from_components(result)
    if from_components:
        return from_components

    from_raw_detections = _inferred_type_from_raw_detections(result)
    if from_raw_detections:
        return from_raw_detections

    detected = _inferred_type_from_detected_type(result.get("detected_type"))
    if detected:
        return detected

    return fallback


def _shelf_count_from_inference(result: dict, inferred_type: str, fallback: int = 3) -> int:
    shelf_qty = _component_quantity(result, "shelf_panel")
    if shelf_qty > 0:
        return shelf_qty
    return _shelf_count_from_detected_type(inferred_type)


def _normalized_center_from_box(
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


def _layout_axis_from_raw_detections(result: dict, tokens: tuple[str, ...], axis: str) -> list[float]:
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
        width_px = _safe_float(evidence.get("width_px"), default=1.0)
        height_px = _safe_float(evidence.get("height_px"), default=1.0)

        for detection in raw_detections:
            if not isinstance(detection, dict):
                continue
            label = str(detection.get("label", "")).strip().lower()
            if not label or not any(token in label for token in tokens):
                continue
            center = _normalized_center_from_box(detection.get("box"), width_px, height_px)
            if center is None:
                continue
            values.append(center[0] if axis == "x" else center[1])

    return sorted(values)


def _layout_axis_from_components(result: dict, tokens: tuple[str, ...], axis: str) -> list[float]:
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

        center = _normalized_center_from_box(component.get("box_corners"), 1.0, 1.0)
        if center is None:
            continue

        try:
            quantity = max(int(component.get("quantity", 1)), 1)
        except (TypeError, ValueError):
            quantity = 1
        values.extend([center[0] if axis == "x" else center[1]] * quantity)

    return sorted(values)


def _fit_anchor_count(anchors: list[float], count: int) -> list[float]:
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


def _apply_inference_layout_to_joints(model: ProjectModel, result: dict) -> None:
    components_by_id = {component.id: component for component in model.components}

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

        anchors = _layout_axis_from_raw_detections(result, tokens=tokens, axis=axis)
        if not anchors:
            anchors = _layout_axis_from_components(result, tokens=tokens, axis=axis)
        fitted = _fit_anchor_count(anchors, len(child_ids))
        if not fitted:
            continue

        axis_map = {child_id: fitted[idx] for idx, child_id in enumerate(child_ids)}
        target_values[kind_name] = axis_map

    for joint in model.joints:
        component = components_by_id.get(joint.child_id)
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
            joint.pos_x = round((anchor - 0.5) * inner_width, 3)
        elif kind_name in {"shelf", "drawer_front"}:
            joint.pos_y = round((0.5 - anchor) * usable_height, 3)


def _content_disposition(file_name: str) -> dict[str, str]:
    return {"Content-Disposition": f'attachment; filename="{file_name}"'}


def _build_bom_csv(model: ProjectModel) -> bytes:
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["section", "id_or_code", "kind", "width", "height", "depth", "qty"])
    for component in model.components:
        writer.writerow([
            "panel",
            component.id,
            component.kind,
            component.width,
            component.height,
            component.depth,
            1,
        ])
    for hardware in model.hardware:
        writer.writerow(["hardware", hardware.code, "", "", "", "", hardware.qty])
    return buffer.getvalue().encode("utf-8")


def _build_nesting_dxf(model: ProjectModel) -> bytes:
    # Lightweight placeholder DXF output used by frontend download flow.
    lines = [
        "0",
        "SECTION",
        "2",
        "ENTITIES",
    ]
    cursor_x = 0.0
    cursor_y = 0.0
    row_height = 0.0
    sheet_width = 2440.0

    for component in model.components:
        width = float(max(component.width, component.depth, 1))
        height = float(max(component.height, 1))
        if cursor_x + width > sheet_width:
            cursor_x = 0.0
            cursor_y += row_height + 20.0
            row_height = 0.0

        x1 = cursor_x
        y1 = cursor_y
        x2 = cursor_x + width
        y2 = cursor_y + height

        lines.extend(
            [
                "0",
                "LWPOLYLINE",
                "8",
                "PANELS",
                "90",
                "4",
                "70",
                "1",
                "10",
                f"{x1}",
                "20",
                f"{y1}",
                "10",
                f"{x2}",
                "20",
                f"{y1}",
                "10",
                f"{x2}",
                "20",
                f"{y2}",
                "10",
                f"{x1}",
                "20",
                f"{y2}",
            ]
        )

        cursor_x += width + 10.0
        row_height = max(row_height, height)

    lines.extend(["0", "ENDSEC", "0", "EOF"])
    return "\n".join(lines).encode("utf-8")


def _pdf_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _build_blueprint_pdf(project_id: str, model: ProjectModel) -> bytes:
    width_mm = max(float(model.product.target_width), 1.0)
    height_mm = max(float(model.product.target_height), 1.0)
    depth_mm = max(float(model.product.target_depth), 1.0)

    def fit_scale(src_w: float, src_h: float, box_w: float, box_h: float) -> float:
        return min(box_w / max(src_w, 1.0), box_h / max(src_h, 1.0))

    front_scale = fit_scale(width_mm, height_mm, 150.0, 180.0)
    side_scale = fit_scale(depth_mm, height_mm, 120.0, 180.0)
    top_scale = fit_scale(width_mm, depth_mm, 150.0, 120.0)

    front_w = width_mm * front_scale
    front_h = height_mm * front_scale
    side_w = depth_mm * side_scale
    side_h = height_mm * side_scale
    top_w = width_mm * top_scale
    top_h = depth_mm * top_scale

    front_x = 68.0
    front_y = 500.0
    side_x = 276.0
    side_y = 500.0
    top_x = 430.0
    top_y = 560.0

    shelf_count = max(int(model.product.shelf_count), 0)
    component_rows = sorted(
        model.components,
        key=lambda comp: comp.id,
    )[:10]

    ops: list[str] = []

    # Header
    ops.extend(
        [
            "0.1 0.2 0.3 rg",
            "BT /F1 18 Tf 56 760 Td (Vision to Blueprint - Furniture Plan) Tj ET",
            "0 0 0 rg",
            "BT /F1 10 Tf 56 742 Td "
            f"(Project: {_pdf_escape(project_id)} | Product: {_pdf_escape(model.product.name)} ({_pdf_escape(model.product.inferred_type)})) Tj ET",
            "BT /F1 10 Tf 56 728 Td "
            f"(Overall dimensions: {width_mm:.0f} x {height_mm:.0f} x {depth_mm:.0f} mm | Material thickness: {model.product.material_thickness:.0f} mm) Tj ET",
        ]
    )

    # View titles
    ops.extend(
        [
            f"BT /F1 10 Tf {front_x:.2f} 690 Td (Front View) Tj ET",
            f"BT /F1 10 Tf {side_x:.2f} 690 Td (Side View) Tj ET",
            f"BT /F1 10 Tf {top_x:.2f} 690 Td (Top View) Tj ET",
        ]
    )

    # Main outlines
    ops.extend(
        [
            "0.15 0.15 0.15 RG",
            "1.2 w",
            f"{front_x:.2f} {front_y:.2f} {front_w:.2f} {front_h:.2f} re S",
            f"{side_x:.2f} {side_y:.2f} {side_w:.2f} {side_h:.2f} re S",
            f"{top_x:.2f} {top_y:.2f} {top_w:.2f} {top_h:.2f} re S",
        ]
    )

    # Shelf lines in front view
    if shelf_count > 0:
        spacing = front_h / (shelf_count + 1)
        for idx in range(shelf_count):
            y = front_y + spacing * (idx + 1)
            ops.append(f"{front_x:.2f} {y:.2f} m {(front_x + front_w):.2f} {y:.2f} l S")

    # Desk legs in front/top views
    if model.product.inferred_type == "desk":
        leg_w = max(10.0, front_w * 0.12)
        leg_h = max(24.0, front_h * 0.82)
        leg_top = front_y
        ops.extend(
            [
                f"{front_x:.2f} {leg_top:.2f} {leg_w:.2f} {leg_h:.2f} re S",
                f"{(front_x + front_w - leg_w):.2f} {leg_top:.2f} {leg_w:.2f} {leg_h:.2f} re S",
            ]
        )

        top_leg = max(8.0, top_h * 0.16)
        inset = max(6.0, top_w * 0.08)
        ops.extend(
            [
                f"{(top_x + inset):.2f} {(top_y + inset):.2f} {top_leg:.2f} {top_leg:.2f} re S",
                f"{(top_x + top_w - inset - top_leg):.2f} {(top_y + inset):.2f} {top_leg:.2f} {top_leg:.2f} re S",
                f"{(top_x + inset):.2f} {(top_y + top_h - inset - top_leg):.2f} {top_leg:.2f} {top_leg:.2f} re S",
                f"{(top_x + top_w - inset - top_leg):.2f} {(top_y + top_h - inset - top_leg):.2f} {top_leg:.2f} {top_leg:.2f} re S",
            ]
        )

    # Dimension lines + labels
    dim_y = front_y - 20.0
    ops.extend(
        [
            "0.35 0.35 0.35 RG",
            "0.8 w",
            f"{front_x:.2f} {dim_y:.2f} m {(front_x + front_w):.2f} {dim_y:.2f} l S",
            f"{front_x:.2f} {(dim_y - 4):.2f} m {front_x:.2f} {(dim_y + 4):.2f} l S",
            f"{(front_x + front_w):.2f} {(dim_y - 4):.2f} m {(front_x + front_w):.2f} {(dim_y + 4):.2f} l S",
            f"BT /F1 9 Tf {(front_x + front_w / 2 - 18):.2f} {(dim_y - 14):.2f} Td ({width_mm:.0f} mm) Tj ET",
            f"{(front_x - 20):.2f} {front_y:.2f} m {(front_x - 20):.2f} {(front_y + front_h):.2f} l S",
            f"{(front_x - 24):.2f} {front_y:.2f} m {(front_x - 16):.2f} {front_y:.2f} l S",
            f"{(front_x - 24):.2f} {(front_y + front_h):.2f} m {(front_x - 16):.2f} {(front_y + front_h):.2f} l S",
            f"BT /F1 9 Tf {(front_x - 46):.2f} {(front_y + front_h / 2):.2f} Td ({height_mm:.0f} mm) Tj ET",
            f"{side_x:.2f} {(side_y - 20):.2f} m {(side_x + side_w):.2f} {(side_y - 20):.2f} l S",
            f"{side_x:.2f} {(side_y - 24):.2f} m {side_x:.2f} {(side_y - 16):.2f} l S",
            f"{(side_x + side_w):.2f} {(side_y - 24):.2f} m {(side_x + side_w):.2f} {(side_y - 16):.2f} l S",
            f"BT /F1 9 Tf {(side_x + side_w / 2 - 18):.2f} {(side_y - 34):.2f} Td ({depth_mm:.0f} mm) Tj ET",
        ]
    )

    # Cut list table area
    table_x = 56.0
    table_y = 300.0
    table_w = 500.0
    row_h = 16.0
    header_h = 18.0
    visible_rows = min(len(component_rows), 10)
    table_h = header_h + row_h * max(visible_rows, 1)

    ops.extend(
        [
            "0.1 0.2 0.3 RG",
            "1 w",
            f"{table_x:.2f} {table_y:.2f} {table_w:.2f} {table_h:.2f} re S",
            f"{table_x:.2f} {(table_y + table_h - header_h):.2f} {table_w:.2f} {header_h:.2f} re S",
            f"BT /F1 10 Tf {(table_x + 6):.2f} {(table_y + table_h - 13):.2f} Td (Cut List \(Top 10 Components\)) Tj ET",
        ]
    )

    col_id = table_x + 8.0
    col_kind = table_x + 160.0
    col_dims = table_x + 320.0
    col_thickness = table_x + 442.0

    ops.extend(
        [
            f"BT /F1 8 Tf {col_id:.2f} {(table_y + table_h - 29):.2f} Td (ID) Tj ET",
            f"BT /F1 8 Tf {col_kind:.2f} {(table_y + table_h - 29):.2f} Td (Kind) Tj ET",
            f"BT /F1 8 Tf {col_dims:.2f} {(table_y + table_h - 29):.2f} Td (W x H x D \(mm\)) Tj ET",
            f"BT /F1 8 Tf {col_thickness:.2f} {(table_y + table_h - 29):.2f} Td (Qty) Tj ET",
        ]
    )

    for idx, component in enumerate(component_rows):
        y = table_y + table_h - header_h - row_h * (idx + 1)
        ops.append(f"{table_x:.2f} {y:.2f} m {(table_x + table_w):.2f} {y:.2f} l S")
        ops.append(
            f"BT /F1 8 Tf {col_id:.2f} {(y + 4):.2f} Td ({_pdf_escape(component.id)}) Tj ET"
        )
        ops.append(
            f"BT /F1 8 Tf {col_kind:.2f} {(y + 4):.2f} Td ({_pdf_escape(component.kind)}) Tj ET"
        )
        ops.append(
            "BT /F1 8 Tf "
            f"{col_dims:.2f} {(y + 4):.2f} Td "
            f"({_pdf_escape(f'{component.width:.0f} x {component.height:.0f} x {component.depth:.0f}')}) Tj ET"
        )
        ops.append(f"BT /F1 8 Tf {col_thickness:.2f} {(y + 4):.2f} Td (1) Tj ET")

    # Footer notes
    ops.extend(
        [
            "0 0 0 rg",
            "BT /F1 8 Tf 56 112 Td (Note: Dimensions are generated from the current inferred model and should be verified before cutting.) Tj ET",
            "BT /F1 8 Tf 56 98 Td (Hardware lines: "
            f"{len(model.hardware)} | Generated by Vision to Blueprint backend) Tj ET",
        ]
    )

    stream = "\n".join(ops).encode("utf-8")

    objects = [
        b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n",
        b"2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n",
        b"3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >> endobj\n",
        b"4 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj\n",
        b"5 0 obj << /Length " + str(len(stream)).encode("ascii") + b" >> stream\n" + stream + b"\nendstream endobj\n",
    ]

    payload = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for obj in objects:
        offsets.append(len(payload))
        payload.extend(obj)
    xref_offset = len(payload)
    payload.extend(f"xref\n0 {len(offsets)}\n".encode("ascii"))
    payload.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        payload.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    payload.extend(
        f"trailer << /Size {len(offsets)} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n".encode(
            "ascii"
        )
    )
    return bytes(payload)


def _build_export_zip(project_id: str, model: ProjectModel) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("blueprint.pdf", _build_blueprint_pdf(project_id, model))
        archive.writestr("bom.csv", _build_bom_csv(model))
        archive.writestr("nesting.dxf", _build_nesting_dxf(model))
        archive.writestr("model.json", model.model_dump_json(indent=2))
    return buffer.getvalue()


def _get_project_or_404(project_id: str, db: Session) -> Project:
    project = db.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return project


@router.get("", response_model=ProjectsListResponse)
def list_projects(db: Session = Depends(get_db)) -> ProjectsListResponse:
    rows = db.query(Project).order_by(Project.created_at.desc()).all()
    return ProjectsListResponse(
        projects=[
            ProjectSummary(
                project_id=p.id,
                name=p.name,
                created_at=p.created_at.isoformat(),
            )
            for p in rows
        ]
    )


@router.post("", response_model=CreateProjectResponse)
def create_project(
    payload: CreateProjectRequest, db: Session = Depends(get_db)
) -> CreateProjectResponse:
    default_name = payload.name or "Generated Project"
    result: dict | None = None

    if payload.job_id:
        job = db.get(JobModel, payload.job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        if job.status != "complete":
            raise HTTPException(status_code=409, detail="Job is not complete")

        result = json.loads(job.result_json or "{}")
        default_name = payload.name or job.project_name or default_name
        inferred_type = _infer_model_type(result, fallback="cabinet")
        door_count, drawer_count = _facade_counts_from_inference(result)
        if door_count + drawer_count > 0:
            inferred_type = "cabinet"
        shelf_count = _shelf_count_from_inference(result, inferred_type, fallback=3)
        divider_count = _divider_count_from_inference(
            result,
            inferred_type=inferred_type,
            door_count=door_count,
            drawer_count=drawer_count,
        )
        has_divider_evidence = _has_divider_evidence(result)
        evidence = _facade_evidence_from_inference(result)
        images_analyzed = max(int(result.get("images_analyzed", 1) or 1), 1)
        door_count, drawer_count = _normalize_facade_counts(
            inferred_type=inferred_type,
            door_count=door_count,
            drawer_count=drawer_count,
            divider_count=divider_count,
            has_divider_evidence=has_divider_evidence,
            shelf_count=shelf_count,
            evidence=evidence,
            images_analyzed=images_analyzed,
        )
        spec = ProductSpec(
            name=default_name,
            inferred_type=inferred_type,
            target_width=result["suggested_width"],
            target_height=result["suggested_height"],
            target_depth=result["suggested_depth"],
            shelf_count=shelf_count,
            divider_count=divider_count,
            door_count=door_count,
            drawer_count=drawer_count,
        )
    else:
        spec = ProductSpec(name=default_name)

    model = generator.generate(spec)
    if result is not None:
        _apply_inference_layout_to_joints(model, result)
    report = generator.validate(model)
    if not report.valid:
        raise HTTPException(status_code=422, detail="Generated model failed validation")

    project_id = str(uuid.uuid4())
    project = Project(id=project_id, name=default_name, model_json=model.model_dump_json())
    db.add(project)
    db.commit()

    return CreateProjectResponse(
        project_id=project_id,
        model=model,
        validation=ValidateResponse(valid=report.valid, errors=report.errors, warnings=report.warnings),
    )


@router.post("/{project_id}/assets", response_model=CreateProjectAssetResponse)
async def upload_project_asset(
    project_id: str,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
) -> CreateProjectAssetResponse:
    _get_project_or_404(project_id, db)
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Uploaded file must be an image")

    payload = await file.read()
    if not payload:
        raise HTTPException(status_code=422, detail="Uploaded file is empty")

    encoded = base64.b64encode(payload).decode("ascii")
    asset_id = str(uuid.uuid4())
    file_name = file.filename or "uploaded-image"
    content_type = file.content_type

    asset = ProjectAsset(
        id=asset_id,
        project_id=project_id,
        file_name=file_name,
        content_type=content_type,
        size_bytes=len(payload),
        image_data=f"data:{content_type};base64,{encoded}",
    )
    db.add(asset)
    db.commit()

    return CreateProjectAssetResponse(
        asset_id=asset_id,
        file_name=file_name,
        content_type=content_type,
        size_bytes=len(payload),
    )


@router.get("/{project_id}/assets", response_model=ProjectAssetsResponse)
def list_project_assets(
    project_id: str, db: Session = Depends(get_db)
) -> ProjectAssetsResponse:
    _get_project_or_404(project_id, db)
    assets = (
        db.query(ProjectAsset)
        .filter(ProjectAsset.project_id == project_id, ProjectAsset.deleted_at.is_(None))
        .all()
    )
    return ProjectAssetsResponse(
        project_id=project_id,
        assets=[
            ProjectAssetResponse(
                asset_id=a.id,
                file_name=a.file_name,
                content_type=a.content_type,
                size_bytes=a.size_bytes,
                image_url=a.image_data,
            )
            for a in assets
        ],
    )


@router.delete("/{project_id}/assets/{asset_id}", status_code=204)
def delete_project_asset(project_id: str, asset_id: str, db: Session = Depends(get_db)) -> Response:
    _get_project_or_404(project_id, db)
    asset = db.get(ProjectAsset, asset_id)
    if not asset or asset.project_id != project_id or asset.deleted_at is not None:
        raise HTTPException(status_code=404, detail="Project asset not found")

    asset.deleted_at = datetime.datetime.now(datetime.UTC)
    db.commit()
    return Response(status_code=204)


@router.post("/{project_id}/jobs", response_model=CreateProjectJobResponse)
def create_project_job(
    project_id: str,
    payload: CreateProjectJobRequest,
    db: Session = Depends(get_db),
) -> CreateProjectJobResponse:
    project = _get_project_or_404(project_id, db)
    _ = payload
    assets = (
        db.query(ProjectAsset)
        .filter(ProjectAsset.project_id == project_id, ProjectAsset.deleted_at.is_(None))
        .all()
    )
    if not assets:
        raise HTTPException(status_code=404, detail="Project has no assets")

    image_data_list = [asset.image_data for asset in assets]

    try:
        job_id, result = client.submit(image_data_list)
    except InferenceGatewayError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    result["source_asset_ids"] = [asset.id for asset in assets]

    job = JobModel(
        id=job_id,
        status="complete",
        result_json=json.dumps(result),
        project_name=project.name,
        project_id=project_id,
        asset_id=None,
    )
    db.add(job)

    asset_ids = result.get("source_asset_ids")
    image_results = result.get("image_results")
    if isinstance(asset_ids, list) and isinstance(image_results, list):
        pair_count = min(len(asset_ids), len(image_results))
        for idx in range(pair_count):
            asset_id = asset_ids[idx]
            image_result = image_results[idx]
            if not isinstance(asset_id, str) or not isinstance(image_result, dict):
                continue
            db.add(
                JobAssetResult(
                    job_id=job_id,
                    asset_id=asset_id,
                    status="complete",
                    result_json=json.dumps(image_result),
                )
            )

    current_model = ProjectModel.model_validate_json(project.model_json)
    inferred_type = _infer_model_type(result, fallback=current_model.product.inferred_type)
    door_count, drawer_count = _facade_counts_from_inference(result)
    if door_count + drawer_count > 0:
        inferred_type = "cabinet"
    shelf_count = _shelf_count_from_inference(
        result,
        inferred_type,
        fallback=current_model.product.shelf_count,
    )
    divider_count = _divider_count_from_inference(
        result,
        inferred_type=inferred_type,
        door_count=door_count,
        drawer_count=drawer_count,
    )
    has_divider_evidence = _has_divider_evidence(result)
    evidence = _facade_evidence_from_inference(result)
    images_analyzed = max(int(result.get("images_analyzed", 1) or 1), 1)
    door_count, drawer_count = _normalize_facade_counts(
        inferred_type=inferred_type,
        door_count=door_count,
        drawer_count=drawer_count,
        divider_count=divider_count,
        has_divider_evidence=has_divider_evidence,
        shelf_count=shelf_count,
        evidence=evidence,
        images_analyzed=images_analyzed,
    )
    spec = ProductSpec(
        name=current_model.product.name,
        inferred_type=inferred_type,
        target_width=result["suggested_width"],
        target_height=result["suggested_height"],
        target_depth=result["suggested_depth"],
        shelf_count=shelf_count,
        divider_count=divider_count,
        door_count=door_count,
        drawer_count=drawer_count,
    )
    updated_model = generator.generate(spec)
    _apply_inference_layout_to_joints(updated_model, result)
    report = generator.validate(updated_model)
    if not report.valid:
        raise HTTPException(status_code=422, detail="Generated model failed validation")
    project.model_json = updated_model.model_dump_json()
    db.commit()

    return CreateProjectJobResponse(
        project_id=project_id,
        asset_id=None,
        asset_count=len(assets),
        job_id=job_id,
        status="complete",
        validation=ValidateResponse(valid=report.valid, errors=report.errors, warnings=report.warnings),
    )


@router.get("/{project_id}/jobs", response_model=ProjectJobsResponse)
def list_project_jobs(project_id: str, db: Session = Depends(get_db)) -> ProjectJobsResponse:
    _get_project_or_404(project_id, db)

    jobs = (
        db.query(JobModel)
        .filter(JobModel.project_id == project_id)
        .order_by(JobModel.id.desc())
        .all()
    )

    return ProjectJobsResponse(
        project_id=project_id,
        jobs=[
            JobResponse(
                job_id=job.id,
                status=job.status,
                result=json.loads(job.result_json) if job.result_json else None,
                project_id=job.project_id,
                asset_id=job.asset_id,
            )
            for job in jobs
        ],
    )


@router.get("/{project_id}/jobs/{job_id}", response_model=JobResponse)
def get_project_job(
    project_id: str, job_id: str, db: Session = Depends(get_db)
) -> JobResponse:
    _get_project_or_404(project_id, db)
    job = db.get(JobModel, job_id)
    if not job or job.project_id != project_id:
        raise HTTPException(status_code=404, detail="Project job not found")

    result = json.loads(job.result_json) if job.result_json else None
    asset_results_rows = (
        db.query(JobAssetResult)
        .filter(JobAssetResult.job_id == job_id)
        .order_by(JobAssetResult.asset_id.asc())
        .all()
    )
    return JobResponse(
        job_id=job_id,
        status=job.status,
        result=result,
        project_id=job.project_id,
        asset_id=job.asset_id,
        asset_results=[
            {
                "job_id": row.job_id,
                "asset_id": row.asset_id,
                "status": row.status,
                "result": json.loads(row.result_json) if row.result_json else None,
            }
            for row in asset_results_rows
        ]
        if asset_results_rows
        else None,
    )


@router.get("/{project_id}", response_model=ProjectResponse)
def get_project(project_id: str, db: Session = Depends(get_db)) -> ProjectResponse:
    project = _get_project_or_404(project_id, db)
    model = ProjectModel.model_validate_json(project.model_json)
    report = generator.validate(model)
    return ProjectResponse(
        project_id=project_id,
        model=model,
        validation=ValidateResponse(valid=report.valid, errors=report.errors, warnings=report.warnings),
    )


@router.patch("/{project_id}", response_model=ProjectResponse)
def update_project(
    project_id: str,
    payload: UpdateProjectRequest,
    db: Session = Depends(get_db),
) -> ProjectResponse:
    project = _get_project_or_404(project_id, db)
    current_model = ProjectModel.model_validate_json(project.model_json)

    updated = generator.apply_update(current_model, **payload.model_dump())
    report = generator.validate(updated)
    if not report.valid:
        raise HTTPException(status_code=422, detail="Project update failed validation")

    project.model_json = updated.model_dump_json()
    db.commit()

    return ProjectResponse(
        project_id=project_id,
        model=updated,
        validation=ValidateResponse(valid=report.valid, errors=report.errors, warnings=report.warnings),
    )


@router.post("/{project_id}/validate", response_model=ValidateResponse)
def validate_project(
    project_id: str, db: Session = Depends(get_db)
) -> ValidateResponse:
    project = _get_project_or_404(project_id, db)
    model = ProjectModel.model_validate_json(project.model_json)
    report = generator.validate(model)
    return ValidateResponse(valid=report.valid, errors=report.errors, warnings=report.warnings)


@router.get("/{project_id}/blueprint.pdf")
def download_blueprint(project_id: str, db: Session = Depends(get_db)) -> Response:
    project = _get_project_or_404(project_id, db)
    model = ProjectModel.model_validate_json(project.model_json)
    payload = _build_blueprint_pdf(project_id, model)
    return Response(
        content=payload,
        media_type="application/pdf",
        headers=_content_disposition(f"{project_id}-blueprint.pdf"),
    )


@router.get("/{project_id}/bom.csv")
def download_bom(project_id: str, db: Session = Depends(get_db)) -> Response:
    project = _get_project_or_404(project_id, db)
    model = ProjectModel.model_validate_json(project.model_json)
    payload = _build_bom_csv(model)
    return Response(
        content=payload,
        media_type="text/csv",
        headers=_content_disposition(f"{project_id}-bom.csv"),
    )


@router.get("/{project_id}/nesting.dxf")
def download_nesting(project_id: str, db: Session = Depends(get_db)) -> Response:
    project = _get_project_or_404(project_id, db)
    model = ProjectModel.model_validate_json(project.model_json)
    payload = _build_nesting_dxf(model)
    return Response(
        content=payload,
        media_type="application/dxf",
        headers=_content_disposition(f"{project_id}-nesting.dxf"),
    )


@router.get("/{project_id}/export")
def download_export_package(project_id: str, db: Session = Depends(get_db)) -> Response:
    project = _get_project_or_404(project_id, db)
    model = ProjectModel.model_validate_json(project.model_json)
    payload = _build_export_zip(project_id, model)
    return Response(
        content=payload,
        media_type="application/zip",
        headers=_content_disposition(f"{project_id}-fabrication-package.zip"),
    )
