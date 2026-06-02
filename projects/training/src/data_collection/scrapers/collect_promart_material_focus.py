from __future__ import annotations

import argparse
import csv
import json
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

import requests

USER_AGENT = "VisionToBlueprintMVP/0.1 (material-focused dataset collector; local-dev)"
TARGET_CLASSES = ("cabinet", "desk", "shelf")
SEMANTIC_CLASSES = (
    "wardrobe",
    "kitchen_cabinet",
    "bookcase",
    "shelf",
    "desk",
    "nightstand",
    "dresser",
    "tv_stand",
    "sideboard",
    "cabinet",
)
SEMANTIC_TO_TARGET = {
    "wardrobe": "cabinet",
    "kitchen_cabinet": "cabinet",
    "bookcase": "shelf",
    "shelf": "shelf",
    "desk": "desk",
    "nightstand": "cabinet",
    "dresser": "cabinet",
    "tv_stand": "cabinet",
    "sideboard": "cabinet",
    "cabinet": "cabinet",
}
MATERIAL_TOKENS = (
    "melamina",
    "mdf",
    "mdp",
    "aglomerado",
    "laminad",
    "triplay",
    "plywood",
    "hdf",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Promart images focused on melamine/MDF-like materials and balanced furniture types"
    )
    parser.add_argument(
        "--metadata-csv",
        type=Path,
        default=Path("datasets/raw/promart_flat_wood_metadata.csv"),
        help="Input metadata CSV from collect_promart_furniture_metadata.py",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("datasets/raw/images"),
        help="Output root containing class folders",
    )
    parser.add_argument(
        "--output-metadata-csv",
        type=Path,
        default=Path("datasets/raw/metadata_promart_material_focus.csv"),
        help="Output CSV for downloaded material-focused samples",
    )
    parser.add_argument("--per-category", type=int, default=80, help="Max downloads per inferred category")
    parser.add_argument(
        "--category-mode",
        choices=("target3", "semantic"),
        default="target3",
        help="Write images grouped by 3 target classes or richer semantic classes",
    )
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--sleep", type=float, default=0.3)
    return parser.parse_args()


def _normalize(value: str) -> str:
    lowered = value.lower()
    lowered = (
        lowered.replace("á", "a")
        .replace("é", "e")
        .replace("í", "i")
        .replace("ó", "o")
        .replace("ú", "u")
        .replace("ñ", "n")
    )
    return lowered


def infer_semantic_category(product_name: str, category_path: str) -> str:
    text = _normalize(f"{product_name} {category_path}")
    if any(token in text for token in ("ropero", "closet", "armario", "guardarropa", "wardrobe")):
        return "wardrobe"
    if any(token in text for token in ("alacena", "cocina", "kitchen cabinet", "mueble cocina")):
        return "kitchen_cabinet"
    if any(token in text for token in ("librero", "biblioteca", "bookcase")):
        return "bookcase"
    if any(token in text for token in ("estante", "repisa", "shelf")):
        return "shelf"
    if any(token in text for token in ("escritorio", "desk", "mesa de estudio", "mesa escritorio")):
        return "desk"
    if any(token in text for token in ("velador", "mesa de noche", "nightstand", "bedside")):
        return "nightstand"
    if any(token in text for token in ("cajonera", "comoda", "dresser", "chest of drawers")):
        return "dresser"
    if any(token in text for token in ("mueble tv", "rack tv", "tv stand", "entertainment center")):
        return "tv_stand"
    if any(token in text for token in ("aparador", "vitrina", "sideboard", "buffet")):
        return "sideboard"
    return "cabinet"


def choose_output_category(semantic_category: str, category_mode: str) -> str:
    if category_mode == "semantic":
        return semantic_category
    return SEMANTIC_TO_TARGET.get(semantic_category, "cabinet")


def has_target_material(material_text: str, material_fields_json: str) -> bool:
    blob = _normalize(f"{material_text} {material_fields_json}")
    return any(token in blob for token in MATERIAL_TOKENS)


def safe_file_name(product_id: str, product_name: str, image_url: str) -> str:
    stem = re.sub(r"[^a-zA-Z0-9_-]+", "_", _normalize(product_name)).strip("_")[:64]
    if not stem:
        stem = "product"
    ext = ".jpg"
    url_lower = image_url.lower()
    for candidate in (".jpg", ".jpeg", ".png", ".webp"):
        if candidate in url_lower:
            ext = candidate
            break
    return f"{product_id}_{stem}{ext}"


def request_with_backoff(url: str, timeout: float, attempts: int = 5) -> bytes | None:
    for attempt in range(attempts):
        try:
            response = requests.get(url, timeout=timeout, headers={"User-Agent": USER_AGENT})
            if response.status_code == 429:
                wait = min(2**attempt, 15)
                time.sleep(wait)
                continue
            response.raise_for_status()
            return response.content
        except Exception:
            if attempt == attempts - 1:
                return None
            time.sleep(min(2**attempt, 15))
    return None


def main() -> None:
    args = parse_args()
    if not args.metadata_csv.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {args.metadata_csv}")

    rows = list(csv.DictReader(args.metadata_csv.open("r", encoding="utf-8")))
    selected: list[dict[str, Any]] = []
    for row in rows:
        if not has_target_material(
            row.get("material_text", ""),
            row.get("material_fields_json", ""),
        ):
            continue
        image_url = (row.get("image_url") or "").strip()
        if not image_url:
            continue

        semantic_category = infer_semantic_category(
            row.get("product_name", ""),
            row.get("category_path", ""),
        )
        row["semantic_category"] = semantic_category
        row["target_category"] = SEMANTIC_TO_TARGET.get(semantic_category, "cabinet")
        row["output_category"] = choose_output_category(semantic_category, args.category_mode)
        selected.append(row)

    category_names = SEMANTIC_CLASSES if args.category_mode == "semantic" else TARGET_CLASSES
    by_category: dict[str, list[dict[str, Any]]] = {name: [] for name in category_names}
    for row in selected:
        category = row.get("output_category", "cabinet")
        if category not in by_category:
            continue
        by_category[category].append(row)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    downloaded_rows: list[dict[str, str]] = []
    download_counts: Counter[str] = Counter()

    semantic_counts: Counter[str] = Counter()
    target_counts: Counter[str] = Counter()

    for category in category_names:
        class_dir = args.output_dir / category
        class_dir.mkdir(parents=True, exist_ok=True)
        candidates = by_category.get(category, [])

        for row in candidates:
            if download_counts[category] >= args.per_category:
                break

            product_id = str(row.get("product_id", "unknown"))
            product_name = str(row.get("product_name", "product"))
            image_url = str(row.get("image_url", ""))
            file_name = safe_file_name(product_id, product_name, image_url)
            local_path = class_dir / file_name

            if local_path.exists():
                downloaded_rows.append(
                    {
                        "category": category,
                        "semantic_category": str(row.get("semantic_category", "cabinet")),
                        "target_category": str(row.get("target_category", "cabinet")),
                        "product_id": product_id,
                        "product_name": product_name,
                        "image_url": image_url,
                        "local_path": str(local_path),
                        "material_text": str(row.get("material_text", "")),
                        "status": "existing",
                    }
                )
                download_counts[category] += 1
                semantic_counts[str(row.get("semantic_category", "cabinet"))] += 1
                target_counts[str(row.get("target_category", "cabinet"))] += 1
                continue

            content = request_with_backoff(image_url, timeout=args.timeout)
            if not content:
                continue

            local_path.write_bytes(content)
            downloaded_rows.append(
                {
                    "category": category,
                    "semantic_category": str(row.get("semantic_category", "cabinet")),
                    "target_category": str(row.get("target_category", "cabinet")),
                    "product_id": product_id,
                    "product_name": product_name,
                    "image_url": image_url,
                    "local_path": str(local_path),
                    "material_text": str(row.get("material_text", "")),
                    "status": "downloaded",
                }
            )
            download_counts[category] += 1
            semantic_counts[str(row.get("semantic_category", "cabinet"))] += 1
            target_counts[str(row.get("target_category", "cabinet"))] += 1
            time.sleep(args.sleep)

    args.output_metadata_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_metadata_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "category",
                "semantic_category",
                "target_category",
                "product_id",
                "product_name",
                "image_url",
                "local_path",
                "material_text",
                "status",
            ],
        )
        writer.writeheader()
        writer.writerows(downloaded_rows)

    summary = {
        "input_rows": len(rows),
        "selected_rows": len(selected),
        "downloaded_total": len(downloaded_rows),
        "category_mode": args.category_mode,
        "downloaded_by_category": dict(download_counts),
        "downloaded_by_semantic_category": dict(semantic_counts),
        "downloaded_by_target_category": dict(target_counts),
        "output_metadata_csv": str(args.output_metadata_csv),
        "output_dir": str(args.output_dir),
    }
    summary_path = args.output_metadata_csv.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Selected rows with target materials: {len(selected)}")
    print(f"Downloaded/linked rows: {len(downloaded_rows)}")
    print(f"By category: {dict(download_counts)}")
    print(f"Metadata CSV: {args.output_metadata_csv}")
    print(f"Summary JSON: {summary_path}")


if __name__ == "__main__":
    main()
