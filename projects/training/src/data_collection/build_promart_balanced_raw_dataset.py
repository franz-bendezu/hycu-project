from __future__ import annotations

import argparse
import csv
import json
import random
import re
import shutil
from pathlib import Path

TARGET_CATEGORIES = ("cabinet", "desk", "shelf")

CATEGORY_TOKENS: dict[str, tuple[str, ...]] = {
    "desk": (
        "desk",
        "escritorio",
        "office desk",
        "writing desk",
        "study desk",
        "computer desk",
        "gamer desk",
        "tocador",
    ),
    "shelf": (
        "shelf",
        "bookcase",
        "bookshelf",
        "estante",
        "biblioteca",
        "librero",
        "repisa",
    ),
    "cabinet": (
        "cabinet",
        "wardrobe",
        "closet",
        "ropero",
        "armario",
        "comoda",
        "cajonera",
        "sideboard",
        "vitrina",
        "alacena",
        "nightstand",
        "velador",
        "mesa de noche",
        "tv stand",
        "centro de entretenimiento",
        "storage",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a balanced cabinet/desk/shelf raw image dataset from Promart metadata. "
            "Outputs files under datasets/raw/images/<category>/ for downstream YOLO preparation."
        )
    )
    parser.add_argument(
        "--images-metadata-csv",
        type=Path,
        default=Path("datasets/raw/promart_product_images_metadata.csv"),
        help="CSV produced by collect_promart_furniture_metadata.py with local image paths",
    )
    parser.add_argument(
        "--products-metadata-csv",
        type=Path,
        default=Path("datasets/raw/promart_flat_wood_metadata.csv"),
        help="CSV produced by collect_promart_furniture_metadata.py with product-level fields",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("datasets/raw/images"),
        help="Target root folder. Category subfolders are created under this path.",
    )
    parser.add_argument(
        "--mode",
        choices=("copy", "symlink"),
        default="copy",
        help="How to stage selected images into output-root",
    )
    parser.add_argument(
        "--strategy",
        choices=("cap", "strict-balance"),
        default="cap",
        help=(
            "Sampling strategy: cap keeps up to max-per-class by availability; "
            "strict-balance uses the same per-class target based on the smallest available class."
        ),
    )
    parser.add_argument("--max-per-class", type=int, default=800)
    parser.add_argument(
        "--min-per-class",
        type=int,
        default=0,
        help="If > 0, fail when any class selects fewer images than this threshold",
    )
    parser.add_argument("--max-per-brand-per-class", type=int, default=80)
    parser.add_argument("--max-images-per-product", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clean", action="store_true", help="Remove existing target category folders before writing")
    parser.add_argument(
        "--selected-csv",
        type=Path,
        default=Path("datasets/raw/promart_balanced_selected.csv"),
        help="CSV summary with selected source/target pairs",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=Path("datasets/raw/promart_balanced_summary.json"),
        help="Run summary JSON",
    )
    return parser.parse_args()


def normalize_text(value: str) -> str:
    text = value.lower().strip()
    replacements = (
        ("á", "a"),
        ("é", "e"),
        ("í", "i"),
        ("ó", "o"),
        ("ú", "u"),
        ("ñ", "n"),
    )
    for old, new in replacements:
        text = text.replace(old, new)
    text = re.sub(r"\s+", " ", text)
    return text


def load_products(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Products metadata CSV not found: {path}")
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    out: dict[str, dict[str, str]] = {}
    for row in rows:
        product_id = str(row.get("product_id", "")).strip()
        if not product_id:
            continue
        out[product_id] = {k: str(v) for k, v in row.items()}
    return out


def classify_category(product: dict[str, str]) -> tuple[str | None, dict[str, int]]:
    joined = " ".join(
        [
            product.get("product_name", ""),
            product.get("category_path", ""),
            product.get("model", ""),
            product.get("specs_json", ""),
        ]
    )
    text = normalize_text(joined)

    scores: dict[str, int] = {category: 0 for category in TARGET_CATEGORIES}
    for category, tokens in CATEGORY_TOKENS.items():
        for token in tokens:
            if token in text:
                scores[category] += 1

    best_category = max(scores, key=scores.get)
    best_score = scores[best_category]
    if best_score == 0:
        return None, scores

    # Avoid unstable ties.
    top_count = sum(1 for value in scores.values() if value == best_score)
    if top_count > 1:
        return None, scores

    return best_category, scores


def load_image_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Image metadata CSV not found: {path}")
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = [dict(row) for row in csv.DictReader(f)]

    usable: list[dict[str, str]] = []
    for row in rows:
        status = str(row.get("status", "")).strip().lower()
        local_path = Path(str(row.get("local_path", "")).strip())
        if status.startswith("error"):
            continue
        if not local_path.exists() or not local_path.is_file():
            continue
        if local_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
            continue
        usable.append(row)
    return usable


def _stage_file(source: Path, target: Path, mode: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        target.unlink()

    if mode == "symlink":
        target.symlink_to(source.resolve())
        return
    shutil.copy2(source, target)


def _safe_name(value: str, fallback: str) -> str:
    norm = normalize_text(value)
    norm = re.sub(r"[^a-z0-9_.-]+", "_", norm).strip("_")
    return norm or fallback


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    products = load_products(args.products_metadata_csv)
    image_rows = load_image_rows(args.images_metadata_csv)

    grouped: dict[str, dict[str, dict[str, list[dict[str, str]]]]] = {
        category: {} for category in TARGET_CATEGORIES
    }

    skipped_missing_product = 0
    skipped_unclassified = 0

    for row in image_rows:
        product_id = str(row.get("product_id", "")).strip()
        product = products.get(product_id)
        if product is None:
            skipped_missing_product += 1
            continue

        category, _scores = classify_category(product)
        if category is None:
            skipped_unclassified += 1
            continue

        brand = str(product.get("brand", "")).strip() or "unknown"
        brand_bucket = grouped[category].setdefault(brand, {})
        brand_bucket.setdefault(product_id, []).append(row)

    available_by_class: dict[str, int] = {}
    for category in TARGET_CATEGORIES:
        count = 0
        for product_map in grouped[category].values():
            for images in product_map.values():
                count += min(len(images), max(1, args.max_images_per_product))
        available_by_class[category] = count

    if args.strategy == "strict-balance":
        target_per_class = min(args.max_per_class, min(available_by_class.values()))
    else:
        target_per_class = args.max_per_class

    if args.clean:
        for category in TARGET_CATEGORIES:
            category_dir = args.output_root / category
            if category_dir.exists():
                shutil.rmtree(category_dir)

    selected_rows: list[dict[str, str]] = []
    class_counts: dict[str, int] = {category: 0 for category in TARGET_CATEGORIES}
    brand_counts: dict[str, dict[str, int]] = {category: {} for category in TARGET_CATEGORIES}

    for category in TARGET_CATEGORIES:
        brands = list(grouped[category].keys())
        rng.shuffle(brands)

        candidates: list[tuple[str, str, dict[str, str]]] = []
        for brand in brands:
            product_map = grouped[category][brand]
            product_ids = list(product_map.keys())
            rng.shuffle(product_ids)
            for product_id in product_ids:
                images = product_map[product_id][:]
                rng.shuffle(images)
                for row in images[: max(1, args.max_images_per_product)]:
                    candidates.append((brand, product_id, row))

        rng.shuffle(candidates)
        seen_source_paths: set[str] = set()

        for brand, product_id, row in candidates:
            if class_counts[category] >= target_per_class:
                break

            brand_count = brand_counts[category].get(brand, 0)
            if brand_count >= args.max_per_brand_per_class:
                continue

            source_path = Path(str(row.get("local_path", "")).strip())
            if not source_path.exists() or not source_path.is_file():
                continue
            source_key = str(source_path.resolve())
            if source_key in seen_source_paths:
                continue
            seen_source_paths.add(source_key)

            product_name = str(row.get("product_name", "")).strip() or product_id
            stem = _safe_name(f"{product_id}_{product_name}_{class_counts[category] + 1}", fallback=product_id)
            target_name = f"{stem}{source_path.suffix.lower()}"
            target_path = args.output_root / category / target_name
            _stage_file(source_path, target_path, args.mode)

            class_counts[category] += 1
            brand_counts[category][brand] = brand_count + 1
            selected_rows.append(
                {
                    "category": category,
                    "brand": brand,
                    "product_id": product_id,
                    "source_path": str(source_path),
                    "target_path": str(target_path),
                    "mode": args.mode,
                }
            )

    if args.min_per_class > 0:
        underfilled = {
            category: count
            for category, count in class_counts.items()
            if count < args.min_per_class
        }
        if underfilled:
            details = ", ".join(f"{k}={v}" for k, v in sorted(underfilled.items()))
            raise RuntimeError(
                "Balanced dataset generation failed: selected counts below min-per-class "
                f"({args.min_per_class}). Underfilled: {details}"
            )

    args.selected_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.selected_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["category", "brand", "product_id", "source_path", "target_path", "mode"],
        )
        writer.writeheader()
        writer.writerows(selected_rows)

    summary = {
        "images_metadata_csv": str(args.images_metadata_csv),
        "products_metadata_csv": str(args.products_metadata_csv),
        "output_root": str(args.output_root),
        "selected_csv": str(args.selected_csv),
        "mode": args.mode,
        "strategy": args.strategy,
        "max_per_class": args.max_per_class,
        "target_per_class": target_per_class,
        "min_per_class": args.min_per_class,
        "max_per_brand_per_class": args.max_per_brand_per_class,
        "max_images_per_product": args.max_images_per_product,
        "available_by_class": available_by_class,
        "selected_total": len(selected_rows),
        "selected_by_class": class_counts,
        "selected_brand_counts": brand_counts,
        "skipped_missing_product": skipped_missing_product,
        "skipped_unclassified": skipped_unclassified,
    }

    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Balanced Promart raw dataset prepared")
    print(f"Selected images: {len(selected_rows)}")
    print(f"Selected by class: {class_counts}")
    print(f"Selected CSV: {args.selected_csv}")
    print(f"Summary JSON: {args.summary_json}")


if __name__ == "__main__":
    main()
