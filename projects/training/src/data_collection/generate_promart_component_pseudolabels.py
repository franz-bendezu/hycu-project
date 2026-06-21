from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path

from PIL import Image


TRAINING_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = TRAINING_ROOT.parents[1]
INFERENCE_ROOT = REPO_ROOT / "projects" / "inference"

DEFAULT_QUEUE_CSV = TRAINING_ROOT / "datasets" / "yolo_components_labeled" / "component_labeling_queue.csv"
DEFAULT_SUMMARY_JSON = TRAINING_ROOT / "datasets" / "yolo_components_labeled" / "component_pseudolabels.summary.json"


@dataclass
class GenerationStats:
    scanned: int = 0
    selected: int = 0
    wrote_labels: int = 0
    skipped_existing_label: int = 0
    skipped_no_component_detection: int = 0
    heuristic_labels_written: int = 0
    skipped_missing_image: int = 0
    skipped_row_filter: int = 0
    boxes_written: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate component pseudo-labels (YOLO txt) using the current inference detector "
            "and rows from a labeling queue CSV."
        )
    )
    parser.add_argument("--queue-csv", type=Path, default=DEFAULT_QUEUE_CSV)
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY_JSON)
    parser.add_argument("--limit", type=int, default=0, help="Max rows to process (0 = no limit)")
    parser.add_argument("--min-score", type=float, default=0.25, help="Minimum detection score to keep")
    parser.add_argument("--component-class-min", type=int, default=3, help="Minimum class id considered a component")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing label files")
    parser.add_argument("--only-promart", action="store_true", default=True)
    parser.add_argument("--no-only-promart", action="store_false", dest="only_promart")
    parser.add_argument("--update-queue-status", action="store_true", help="Update queue status to pseudo_labeled")
    parser.add_argument("--heuristic-fallback", action="store_true", default=True)
    parser.add_argument("--no-heuristic-fallback", action="store_false", dest="heuristic_fallback")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolve_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path

    candidates = (
        TRAINING_ROOT / path,
        REPO_ROOT / path,
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def to_rel_training(path: Path) -> str:
    try:
        return path.relative_to(TRAINING_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def should_select_row(row: dict[str, str], args: argparse.Namespace) -> bool:
    image_path = row.get("image_path", "")
    source_image_path = row.get("source_image_path", "")
    if args.only_promart and (
        "datasets/raw/promart_products/" not in image_path
        and "datasets/raw/promart_products/" not in source_image_path
    ):
        return False

    status = row.get("status", "").strip().lower()
    return status in {"todo_component_labels", "labeled"}


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def best_furniture_box(evidence: object, row: dict[str, str]) -> tuple[float, float, float, float] | None:
    detections = sorted(getattr(evidence, "raw_detections", []), key=lambda x: x.score, reverse=True)
    for det in detections:
        if det.class_id in {0, 1, 2}:
            return det.box

    category_hint = (row.get("category_hint", "") or "").strip().lower()
    if category_hint in {"cabinet", "desk", "shelf"}:
        width = float(getattr(evidence, "width_px", 0) or 0)
        height = float(getattr(evidence, "height_px", 0) or 0)
        if width > 0 and height > 0:
            return (0.0, 0.0, width, height)
    return None


def _box_to_line(
    box: tuple[float, float, float, float],
    class_name: str,
    label_to_id: dict[str, int],
    width_px: int,
    height_px: int,
) -> str | None:
    class_id = label_to_id.get(class_name)
    if class_id is None:
        return None
    return detection_to_yolo_line(box, class_id, width_px, height_px)


def heuristic_component_lines(
    row: dict[str, str],
    evidence: object,
    label_to_id: dict[str, int],
) -> list[str]:
    base_box = best_furniture_box(evidence, row)
    if base_box is None:
        return []

    x1, y1, x2, y2 = base_box
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    if w < 20.0 or h < 20.0:
        return []

    t_side = max(2.0, w * 0.08)
    t_top = max(2.0, h * 0.08)
    t_bottom = max(2.0, h * 0.08)
    t_shelf = max(2.0, h * 0.06)
    t_leg_w = max(2.0, w * 0.08)
    t_leg_h = max(2.0, h * 0.18)

    category_hint = (row.get("category_hint", "") or "").strip().lower()
    width_px = int(getattr(evidence, "width_px", 0) or 0)
    height_px = int(getattr(evidence, "height_px", 0) or 0)
    if width_px <= 0 or height_px <= 0:
        return []

    candidate_boxes: list[tuple[str, tuple[float, float, float, float]]] = []

    candidate_boxes.append(("side_panel", (x1, y1, x1 + t_side, y2)))
    candidate_boxes.append(("side_panel", (x2 - t_side, y1, x2, y2)))
    candidate_boxes.append(("top_panel", (x1, y1, x2, y1 + t_top)))
    candidate_boxes.append(("bottom_panel", (x1, y2 - t_bottom, x2, y2)))

    if category_hint in {"cabinet", "shelf"}:
        shelf_y = y1 + h * 0.55
        candidate_boxes.append(("shelf_panel", (x1 + t_side, shelf_y, x2 - t_side, shelf_y + t_shelf)))

    if category_hint == "cabinet":
        candidate_boxes.append(("door_panel", (x1 + t_side, y1 + t_top, x2 - t_side, y2 - t_bottom)))
        handle_x = x2 - t_side - (w * 0.05)
        handle_y = y1 + h * 0.45
        candidate_boxes.append(("handle_pull", (handle_x, handle_y, handle_x + (w * 0.02), handle_y + (h * 0.12))))
        hinge_x = x1 + t_side + (w * 0.02)
        candidate_boxes.append(("hinge", (hinge_x, y1 + h * 0.18, hinge_x + (w * 0.02), y1 + h * 0.24)))
        candidate_boxes.append(("hinge", (hinge_x, y1 + h * 0.72, hinge_x + (w * 0.02), y1 + h * 0.78)))

    if category_hint == "desk":
        candidate_boxes.append(("leg", (x1 + (w * 0.08), y2 - t_leg_h, x1 + (w * 0.08) + t_leg_w, y2)))
        candidate_boxes.append(("leg", (x2 - (w * 0.08) - t_leg_w, y2 - t_leg_h, x2 - (w * 0.08), y2)))

    lines: list[str] = []
    for class_name, box in candidate_boxes:
        line = _box_to_line(box, class_name, label_to_id, width_px, height_px)
        if line is not None:
            lines.append(line)
    return lines


def detection_to_yolo_line(
    box: tuple[float, float, float, float],
    class_id: int,
    width_px: int,
    height_px: int,
) -> str | None:
    x1, y1, x2, y2 = box
    x1 = clamp(x1, 0.0, float(width_px))
    x2 = clamp(x2, 0.0, float(width_px))
    y1 = clamp(y1, 0.0, float(height_px))
    y2 = clamp(y2, 0.0, float(height_px))

    bw = x2 - x1
    bh = y2 - y1
    if bw < 2.0 or bh < 2.0:
        return None

    cx = (x1 + x2) / 2.0 / float(width_px)
    cy = (y1 + y2) / 2.0 / float(height_px)
    nw = bw / float(width_px)
    nh = bh / float(height_px)

    if nw <= 0.0 or nh <= 0.0:
        return None

    return f"{class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"


def load_detector() -> object:
    os.environ.setdefault("INFERENCE_PROVIDERS", "CPUExecutionProvider")

    import sys

    if str(INFERENCE_ROOT) not in sys.path:
        sys.path.insert(0, str(INFERENCE_ROOT))

    from app.core.config import get_confidence_threshold, get_detector_labels, get_detector_model_path
    from app.services.detector import YoloDetector
    from app.services.processor import assemble_project

    threshold = get_confidence_threshold()
    return YoloDetector(
        model_path=get_detector_model_path(),
        labels=get_detector_labels(),
        score_threshold=threshold,
        assemble_project=assemble_project,
    )


def main() -> None:
    args = parse_args()
    if args.limit < 0:
        raise ValueError("--limit must be >= 0")
    if not 0.0 <= args.min_score <= 1.0:
        raise ValueError("--min-score must be in [0, 1]")

    if not args.queue_csv.exists():
        raise FileNotFoundError(f"Queue CSV not found: {args.queue_csv}")

    with args.queue_csv.open("r", newline="", encoding="utf-8") as f:
        rows = [dict(row) for row in csv.DictReader(f)]

    detector = load_detector()
    labels = tuple(detector.labels)
    label_to_id = {name: idx for idx, name in enumerate(labels)}
    stats = GenerationStats(scanned=len(rows))

    for row in rows:
        if args.limit > 0 and stats.selected >= args.limit:
            break

        if not should_select_row(row, args):
            stats.skipped_row_filter += 1
            continue

        image_path = resolve_path(row.get("image_path", ""))
        label_path = resolve_path(row.get("label_path", ""))

        if not image_path.exists():
            stats.skipped_missing_image += 1
            continue

        if label_path.exists() and not args.overwrite and label_path.read_text(encoding="utf-8").strip():
            stats.skipped_existing_label += 1
            continue

        stats.selected += 1

        image = Image.open(image_path).convert("RGB")
        infer_response = detector.analyze([(image, f"file://{image_path}")])
        evidence_list = infer_response.image_results or infer_response.evidence
        evidence = evidence_list[0] if evidence_list else None
        if evidence is None:
            stats.skipped_no_component_detection += 1
            continue

        lines: list[str] = []
        for det in evidence.raw_detections:
            if det.class_id < args.component_class_min:
                continue
            if det.score < args.min_score:
                continue
            line = detection_to_yolo_line(det.box, det.class_id, evidence.width_px, evidence.height_px)
            if line is None:
                continue
            lines.append(line)

        if not lines and args.heuristic_fallback:
            lines = heuristic_component_lines(row=row, evidence=evidence, label_to_id=label_to_id)
            if lines:
                stats.heuristic_labels_written += 1

        if not lines:
            stats.skipped_no_component_detection += 1
            continue

        stats.boxes_written += len(lines)
        stats.wrote_labels += 1
        if args.update_queue_status:
            row["status"] = "pseudo_labeled"

        if args.dry_run:
            continue

        label_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    if args.update_queue_status and not args.dry_run and rows:
        with args.queue_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    summary = {
        "queue_csv": str(args.queue_csv),
        "only_promart": args.only_promart,
        "limit": args.limit,
        "min_score": args.min_score,
        "component_class_min": args.component_class_min,
        "overwrite": args.overwrite,
        "update_queue_status": args.update_queue_status,
        "heuristic_fallback": args.heuristic_fallback,
        "dry_run": args.dry_run,
        "stats": {
            "scanned": stats.scanned,
            "selected": stats.selected,
            "wrote_labels": stats.wrote_labels,
            "boxes_written": stats.boxes_written,
            "heuristic_labels_written": stats.heuristic_labels_written,
            "skipped_existing_label": stats.skipped_existing_label,
            "skipped_no_component_detection": stats.skipped_no_component_detection,
            "skipped_missing_image": stats.skipped_missing_image,
            "skipped_row_filter": stats.skipped_row_filter,
        },
    }

    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    if not args.dry_run:
        args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Pseudo-label generation finished")
    print(f"Selected rows: {stats.selected}")
    print(f"Label files written: {stats.wrote_labels}")
    print(f"Boxes written: {stats.boxes_written}")
    print(f"Summary JSON: {to_rel_training(args.summary_json)}")


if __name__ == "__main__":
    main()
