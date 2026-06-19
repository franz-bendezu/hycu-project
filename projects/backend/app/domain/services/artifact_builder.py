from __future__ import annotations

import csv
import io
import zipfile

from app.presentation.schemas.project_design import ProjectModel


def content_disposition(file_name: str) -> dict[str, str]:
    return {"Content-Disposition": f'attachment; filename="{file_name}"'}


def build_bom_csv(model: ProjectModel) -> bytes:
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


def build_nesting_dxf(model: ProjectModel) -> bytes:
    lines = ["0", "SECTION", "2", "ENTITIES"]
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


def _pdf_escape(value: object) -> str:
    raw = value
    if hasattr(raw, "value"):
        raw = getattr(raw, "value")
    text = str(raw)
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def build_blueprint_pdf(project_id: str, model: ProjectModel) -> bytes:
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
        key=lambda comp: (getattr(comp.kind, "value", str(comp.kind)), comp.id),
    )[:10]

    ops_page1: list[str] = []
    ops_page2: list[str] = []

    ops_page1.extend(
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

    ops_page1.extend(
        [
            f"BT /F1 10 Tf {front_x:.2f} 690 Td (Front View) Tj ET",
            f"BT /F1 10 Tf {side_x:.2f} 690 Td (Side View) Tj ET",
            f"BT /F1 10 Tf {top_x:.2f} 690 Td (Top View) Tj ET",
        ]
    )

    ops_page1.extend(
        [
            "0.15 0.15 0.15 RG",
            "1.2 w",
            f"{front_x:.2f} {front_y:.2f} {front_w:.2f} {front_h:.2f} re S",
            f"{side_x:.2f} {side_y:.2f} {side_w:.2f} {side_h:.2f} re S",
            f"{top_x:.2f} {top_y:.2f} {top_w:.2f} {top_h:.2f} re S",
        ]
    )

    if shelf_count > 0:
        spacing = front_h / (shelf_count + 1)
        for idx in range(shelf_count):
            y = front_y + spacing * (idx + 1)
            ops_page1.append(f"{front_x:.2f} {y:.2f} m {(front_x + front_w):.2f} {y:.2f} l S")

    if model.product.inferred_type == "desk":
        leg_w = max(10.0, front_w * 0.12)
        leg_h = max(24.0, front_h * 0.82)
        leg_top = front_y
        ops_page1.extend(
            [
                f"{front_x:.2f} {leg_top:.2f} {leg_w:.2f} {leg_h:.2f} re S",
                f"{(front_x + front_w - leg_w):.2f} {leg_top:.2f} {leg_w:.2f} {leg_h:.2f} re S",
            ]
        )

    dim_y = front_y - 20.0
    ops_page1.extend(
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
            f"BT /F1 9 Tf {(front_x - 56):.2f} {(front_y + front_h / 2):.2f} Td ({height_mm:.0f} mm) Tj ET",
            f"{side_x:.2f} {(side_y - 20):.2f} m {(side_x + side_w):.2f} {(side_y - 20):.2f} l S",
            f"{side_x:.2f} {(side_y - 24):.2f} m {side_x:.2f} {(side_y - 16):.2f} l S",
            f"{(side_x + side_w):.2f} {(side_y - 24):.2f} m {(side_x + side_w):.2f} {(side_y - 16):.2f} l S",
            f"BT /F1 9 Tf {(side_x + side_w / 2 - 18):.2f} {(side_y - 34):.2f} Td ({depth_mm:.0f} mm) Tj ET",
            "BT /F1 9 Tf 56 468 Td (See page 2 for component figures and full cut list.) Tj ET",
        ]
    )

    ops_page2.extend(
        [
            "0.1 0.2 0.3 rg",
            "BT /F1 16 Tf 56 760 Td (Vision to Blueprint - Component Details) Tj ET",
            "0 0 0 rg",
            "BT /F1 10 Tf 56 742 Td "
            f"(Project: {_pdf_escape(project_id)} | Product: {_pdf_escape(model.product.name)} ({_pdf_escape(model.product.inferred_type)})) Tj ET",
        ]
    )

    figure_components = component_rows[:6]
    if figure_components:
        fig_x = 56.0
        fig_y = 610.0
        fig_w = 500.0
        fig_h = 112.0
        cell_w = fig_w / len(figure_components)

        ops_page2.extend(
            [
                "0.12 0.12 0.12 RG",
                "0.8 w",
                f"{fig_x:.2f} {fig_y:.2f} {fig_w:.2f} {fig_h:.2f} re S",
                f"BT /F1 9 Tf {(fig_x + 6):.2f} {(fig_y + fig_h - 13):.2f} Td (Component Figures) Tj ET",
            ]
        )

        drawable_top = fig_y + fig_h - 20.0
        drawable_bottom = fig_y + 20.0
        drawable_h = max(drawable_top - drawable_bottom, 1.0)

        for idx, component in enumerate(figure_components):
            cell_x = fig_x + idx * cell_w
            if idx > 0:
                ops_page2.append(f"{cell_x:.2f} {fig_y:.2f} m {cell_x:.2f} {(fig_y + fig_h):.2f} l S")

            comp_w = max(float(component.width), 1.0)
            comp_h = max(float(component.height), 1.0)
            scale = min((cell_w - 12.0) / comp_w, (drawable_h - 8.0) / comp_h)
            scale = max(scale, 0.01)
            draw_w = max(comp_w * scale, 3.0)
            draw_h = max(comp_h * scale, 3.0)

            rect_x = cell_x + (cell_w - draw_w) / 2.0
            rect_y = drawable_bottom + (drawable_h - draw_h) / 2.0

            ops_page2.append(f"{rect_x:.2f} {rect_y:.2f} {draw_w:.2f} {draw_h:.2f} re S")

            label = getattr(component.kind, "value", str(component.kind))
            if len(label) > 14:
                label = label[:11] + "..."
            dim_text = f"{component.width:.0f}x{component.height:.0f}"

            ops_page2.append(
                f"BT /F1 7 Tf {(cell_x + 4):.2f} {(fig_y + 8):.2f} Td ({_pdf_escape(label)}) Tj ET"
            )
            ops_page2.append(
                f"BT /F1 7 Tf {(cell_x + 4):.2f} {(fig_y + 2):.2f} Td ({_pdf_escape(dim_text)}) Tj ET"
            )

    table_x = 56.0
    table_y = 340.0
    table_w = 500.0
    row_h = 18.0
    header_h = 34.0
    visible_rows = min(len(component_rows), 10)
    table_h = header_h + row_h * max(visible_rows, 1)

    ops_page2.extend(
        [
            "0.1 0.2 0.3 RG",
            "1 w",
            f"{table_x:.2f} {table_y:.2f} {table_w:.2f} {table_h:.2f} re S",
            f"{table_x:.2f} {(table_y + table_h - header_h):.2f} {table_w:.2f} {header_h:.2f} re S",
            f"BT /F1 10 Tf {(table_x + 6):.2f} {(table_y + table_h - 13):.2f} Td (Cut List - Top 10 Components) Tj ET",
        ]
    )

    col_id = table_x + 8.0
    col_kind = table_x + 160.0
    col_dims = table_x + 300.0
    col_qty = table_x + 456.0
    col_header_y = table_y + table_h - 29.0

    ops_page2.extend(
        [
            f"BT /F1 8 Tf {col_id:.2f} {col_header_y:.2f} Td (ID) Tj ET",
            f"BT /F1 8 Tf {col_kind:.2f} {col_header_y:.2f} Td (Kind) Tj ET",
            f"BT /F1 8 Tf {col_dims:.2f} {col_header_y:.2f} Td (W x H x D mm) Tj ET",
            f"BT /F1 8 Tf {col_qty:.2f} {col_header_y:.2f} Td (Qty) Tj ET",
            f"{table_x:.2f} {(table_y + table_h - header_h):.2f} m {(table_x + table_w):.2f} {(table_y + table_h - header_h):.2f} l S",
        ]
    )

    for idx, component in enumerate(component_rows):
        y = table_y + table_h - header_h - row_h * (idx + 1)
        display_id = component.id
        if len(display_id) > 14:
            display_id = f"{display_id[:8]}...{display_id[-4:]}"
        display_kind = getattr(component.kind, "value", str(component.kind))

        ops_page2.append(f"{table_x:.2f} {y:.2f} m {(table_x + table_w):.2f} {y:.2f} l S")
        ops_page2.append(
            f"BT /F1 8 Tf {col_id:.2f} {(y + 4):.2f} Td ({_pdf_escape(display_id)}) Tj ET"
        )
        ops_page2.append(
            f"BT /F1 8 Tf {col_kind:.2f} {(y + 4):.2f} Td ({_pdf_escape(display_kind)}) Tj ET"
        )
        ops_page2.append(
            "BT /F1 8 Tf "
            f"{col_dims:.2f} {(y + 4):.2f} Td "
            f"({_pdf_escape(f'{component.width:.0f} x {component.height:.0f} x {component.depth:.0f}')}) Tj ET"
        )
        ops_page2.append(f"BT /F1 8 Tf {col_qty:.2f} {(y + 4):.2f} Td (1) Tj ET")

    ops_page2.extend(
        [
            "0 0 0 rg",
            "BT /F1 8 Tf 56 112 Td (Note: Dimensions are generated from the current inferred model and should be verified before cutting.) Tj ET",
            "BT /F1 8 Tf 56 98 Td (Hardware lines: "
            f"{len(model.hardware)} | Generated by Vision to Blueprint backend) Tj ET",
        ]
    )

    stream_page1 = "\n".join(ops_page1).encode("utf-8")
    stream_page2 = "\n".join(ops_page2).encode("utf-8")

    objects = [
        b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n",
        b"2 0 obj << /Type /Pages /Kids [3 0 R 6 0 R] /Count 2 >> endobj\n",
        b"3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >> endobj\n",
        b"4 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj\n",
        b"5 0 obj << /Length " + str(len(stream_page1)).encode("ascii") + b" >> stream\n" + stream_page1 + b"\nendstream endobj\n",
        b"6 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 7 0 R >> endobj\n",
        b"7 0 obj << /Length " + str(len(stream_page2)).encode("ascii") + b" >> stream\n" + stream_page2 + b"\nendstream endobj\n",
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


def build_export_zip(project_id: str, model: ProjectModel) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("blueprint.pdf", build_blueprint_pdf(project_id, model))
        archive.writestr("bom.csv", build_bom_csv(model))
        archive.writestr("nesting.dxf", build_nesting_dxf(model))
        archive.writestr("model.json", model.model_dump_json(indent=2))
    return buffer.getvalue()
