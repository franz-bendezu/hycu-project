from __future__ import annotations

import argparse
import csv
import random
import shutil
from collections import defaultdict
from pathlib import Path


CLASS_MAP = {"cabinet": 0, "desk": 1, "shelf": 2}
CLASS_NAMES = {0: "cabinet", 1: "desk", 2: "shelf"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a class-balanced yolo_active dataset by combining existing yolo_active rows "
            "with newly collected shelf images from Promart."
        )
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-per-class", type=int, default=1000)
    parser.add_argument(
        "--base-queue-csv",
        type=Path,
        default=Path("datasets/yolo_components_labeled/component_labeling_queue.csv"),
        help="Stable source queue CSV (recommended: yolo_components_labeled)",
    )
    parser.add_argument(
        "--shelves-images-csv",
        type=Path,
        default=Path("datasets/raw/promart_product_images_metadata_shelves.csv"),
        help="Collected shelves image metadata CSV",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("datasets/yolo_active"),
        help="Output yolo_active root",
    )
    return parser.parse_args()


def ensure_dirs(root: Path) -> None:
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)


def reset_dataset(root: Path) -> None:
    if root.exists():
        shutil.rmtree(root)
    ensure_dirs(root)


def safe_split(index: int) -> str:
    return "val" if index % 5 == 0 else "train"


def load_base_rows(path: Path) -> dict[str, list[dict[str, str]]]:
    if not path.exists():
        raise FileNotFoundError(f"Base queue CSV not found: {path}")

    rows = list(csv.DictReader(path.open("r", newline="", encoding="utf-8")))
    grouped: dict[str, list[dict[str, str]]] = {k: [] for k in CLASS_MAP}

    for row in rows:
        category = (row.get("category_hint") or "").strip().lower()
        image_path = Path(row.get("image_path") or "")
        if category not in CLASS_MAP:
            continue
        if not image_path.exists():
            continue
        grouped[category].append(
            {
                "category_hint": category,
                "image_path": str(image_path),
                "source": "yolo_components_labeled",
            }
        )

    return grouped


def load_shelf_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []

    rows = list(csv.DictReader(path.open("r", newline="", encoding="utf-8")))
    out: list[dict[str, str]] = []
    seen: set[str] = set()
    for row in rows:
        status = str(row.get("status", "")).strip().lower()
        if status.startswith("error"):
            continue
        image_path = Path(str(row.get("local_path", "")).strip())
        if not image_path.exists() or not image_path.is_file():
            continue
        image_key = str(image_path.resolve())
        if image_key in seen:
            continue
        seen.add(image_key)
        out.append(
            {
                "category_hint": "shelf",
                "image_path": str(image_path),
                "source": "promart_shelves_collection",
            }
        )
    return out


def write_sample(
    *,
    out_root: Path,
    split: str,
    category: str,
    image_src: Path,
    sample_id: str,
) -> tuple[str, str]:
    image_name = f"{sample_id}{image_src.suffix.lower() or '.jpg'}"
    image_dst = out_root / "images" / split / image_name
    label_dst = out_root / "labels" / split / f"{sample_id}.txt"

    if image_dst.exists() or image_dst.is_symlink():
        image_dst.unlink()
    image_dst.symlink_to(image_src.resolve())

    class_id = CLASS_MAP[category]
    label_dst.write_text(f"{class_id} 0.500000 0.500000 1.000000 1.000000\n", encoding="utf-8")

    return str(image_dst), str(label_dst)


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    grouped = load_base_rows(args.base_queue_csv)
    grouped["shelf"].extend(load_shelf_rows(args.shelves_images_csv))

    for category in CLASS_MAP:
        rng.shuffle(grouped[category])

    reset_dataset(args.out_root)

    queue_rows: list[dict[str, str]] = []
    counters = defaultdict(int)

    for category in ("cabinet", "desk", "shelf"):
        available = grouped[category]
        if not available:
            continue

        if len(available) >= args.target_per_class:
            selected = available[: args.target_per_class]
        else:
            selected = available.copy()
            while len(selected) < args.target_per_class:
                selected.append(rng.choice(available))

        for idx, item in enumerate(selected, start=1):
            counters[category] += 1
            split = safe_split(counters[category])
            image_src = Path(item["image_path"])
            stem = image_src.stem
            sample_id = f"active_{category}_{counters[category]:05d}_{stem}"
            image_path, label_path = write_sample(
                out_root=args.out_root,
                split=split,
                category=category,
                image_src=image_src,
                sample_id=sample_id,
            )
            queue_rows.append(
                {
                    "split": split,
                    "category_hint": category,
                    "image_path": image_path,
                    "label_path": label_path,
                    "status": "labeled",
                    "source": item.get("source", "unknown"),
                }
            )

    queue_csv = args.out_root / "labeling_queue.csv"
    with queue_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["split", "category_hint", "image_path", "label_path", "status", "source"],
        )
        writer.writeheader()
        writer.writerows(queue_rows)

    print("Balanced yolo_active dataset created")
    for class_id in (0, 1, 2):
        name = CLASS_NAMES[class_id]
        print(f"{name}: {counters[name]}")
    print(f"total: {len(queue_rows)}")
    print(f"queue: {queue_csv}")


if __name__ == "__main__":
    main()
