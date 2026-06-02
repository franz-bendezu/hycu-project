from __future__ import annotations

import argparse
import csv
import random
import shutil
from dataclasses import dataclass
from pathlib import Path


ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
SUPPORTED_CATEGORIES = ("cabinet", "desk", "shelf")
TARGET3_CLASSES = ("cabinet", "desk", "shelf")
COMPONENT_CLASSES = (
    "cabinet_body",
    "desk_frame",
    "side_panel",
    "top_panel",
    "bottom_panel",
    "back_panel",
    "door_panel",
    "shelf_panel",
    "divider_panel",
    "drawer_front",
    "drawer_box",
    "leg",
    "front_apron",
    "handle",
    "hinge",
    "telescopic_slide",
    "rail",
    "sliding_door_track",
)
COMPONENT_PROFILE_CLASSES = TARGET3_CLASSES + COMPONENT_CLASSES


@dataclass
class ImageRow:
    category: str
    source_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare YOLO train/val image splits and labeling queue from raw category images"
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("datasets/raw/images"),
        help="Root folder containing category subfolders (cabinet/desk/shelf)",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("datasets/yolo"),
        help="YOLO dataset root output folder",
    )
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation ratio in [0.05, 0.5]")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--mode",
        choices=("copy", "symlink"),
        default="copy",
        help="How to stage images into yolo/images train and val",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove existing yolo/images and yolo/labels before rebuilding",
    )
    parser.add_argument(
        "--class-profile",
        choices=("target3", "components"),
        default="target3",
        help="Dataset class schema to emit in generated YAML",
    )
    return parser.parse_args()


def collect_images(raw_root: Path) -> list[ImageRow]:
    rows: list[ImageRow] = []
    for category in SUPPORTED_CATEGORIES:
        class_dir = raw_root / category
        if not class_dir.exists():
            continue
        for path in sorted(class_dir.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix.lower() not in ALLOWED_EXTENSIONS:
                continue
            rows.append(ImageRow(category=category, source_path=path))
    return rows


def split_rows(rows: list[ImageRow], val_ratio: float, seed: int) -> tuple[list[ImageRow], list[ImageRow]]:
    by_category: dict[str, list[ImageRow]] = {category: [] for category in SUPPORTED_CATEGORIES}
    for row in rows:
        by_category[row.category].append(row)

    rng = random.Random(seed)
    train: list[ImageRow] = []
    val: list[ImageRow] = []
    for category, category_rows in by_category.items():
        if not category_rows:
            continue
        pool = category_rows[:]
        rng.shuffle(pool)
        val_count = max(1, int(round(len(pool) * val_ratio))) if len(pool) > 1 else 0
        val.extend(pool[:val_count])
        train.extend(pool[val_count:])
        print(f"category={category} total={len(pool)} train={len(pool[val_count:])} val={len(pool[:val_count])}")

    return train, val


def stage_image(source: Path, target: Path, mode: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        target.unlink()

    if mode == "symlink":
        target.symlink_to(source.resolve())
        return

    shutil.copy2(source, target)


def build_dataset(
    out_root: Path,
    train_rows: list[ImageRow],
    val_rows: list[ImageRow],
    *,
    mode: str,
) -> tuple[int, int]:
    images_train = out_root / "images" / "train"
    images_val = out_root / "images" / "val"
    labels_train = out_root / "labels" / "train"
    labels_val = out_root / "labels" / "val"

    count_images = 0
    count_labels = 0

    for split_name, split_rows, image_dir, label_dir in (
        ("train", train_rows, images_train, labels_train),
        ("val", val_rows, images_val, labels_val),
    ):
        for row in split_rows:
            file_name = f"{row.category}__{row.source_path.name}"
            image_path = image_dir / file_name
            label_path = label_dir / f"{Path(file_name).stem}.txt"

            stage_image(row.source_path, image_path, mode)
            count_images += 1

            # Empty label files are intentional placeholders for manual annotation.
            label_path.parent.mkdir(parents=True, exist_ok=True)
            if not label_path.exists():
                label_path.write_text("", encoding="utf-8")
            count_labels += 1

        print(f"staged split={split_name} images={len(split_rows)}")

    return count_images, count_labels


def write_queue_csv(out_root: Path) -> Path:
    images_root = out_root / "images"
    labels_root = out_root / "labels"
    queue_path = out_root / "labeling_queue.csv"

    rows: list[dict[str, str]] = []
    for split in ("train", "val"):
        split_dir = images_root / split
        if not split_dir.exists():
            continue
        for image_path in sorted(split_dir.glob("*")):
            if image_path.suffix.lower() not in ALLOWED_EXTENSIONS:
                continue
            category = image_path.name.split("__", 1)[0]
            label_path = labels_root / split / f"{image_path.stem}.txt"
            status = "todo"
            if label_path.exists() and label_path.read_text(encoding="utf-8").strip():
                status = "labeled"
            rows.append(
                {
                    "split": split,
                    "category_hint": category,
                    "image_path": str(image_path),
                    "label_path": str(label_path),
                    "status": status,
                }
            )

    with queue_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["split", "category_hint", "image_path", "label_path", "status"],
        )
        writer.writeheader()
        writer.writerows(rows)

    return queue_path


def write_dataset_yaml(yaml_path: Path, class_names: tuple[str, ...]) -> None:
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    names_lines = [f"  {idx}: {name}" for idx, name in enumerate(class_names)]
    yaml_path.write_text(
        "\n".join(
            [
                "path: datasets/yolo",
                "train: images/train",
                "val: images/val",
                "",
                "auto_nc: false",
                f"nc: {len(class_names)}",
                "names:",
                *names_lines,
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    if not 0.05 <= args.val_ratio <= 0.5:
        raise ValueError("--val-ratio must be between 0.05 and 0.5")

    if args.clean and args.out_root.exists():
        shutil.rmtree(args.out_root)

    rows = collect_images(args.raw_root)
    if not rows:
        raise FileNotFoundError(f"No source images found under {args.raw_root}")

    train_rows, val_rows = split_rows(rows, val_ratio=args.val_ratio, seed=args.seed)
    image_count, label_count = build_dataset(args.out_root, train_rows, val_rows, mode=args.mode)

    queue_path = write_queue_csv(args.out_root)
    if args.class_profile == "components":
        class_names = COMPONENT_PROFILE_CLASSES
        yaml_name = "yolo_dataset_components.yaml"
    else:
        class_names = TARGET3_CLASSES
        yaml_name = "yolo_dataset.yaml"
    dataset_yaml_path = args.out_root.parent / yaml_name
    write_dataset_yaml(dataset_yaml_path, class_names)

    print(f"Prepared images: {image_count}")
    print(f"Prepared label placeholders: {label_count}")
    print(f"Labeling queue: {queue_path}")
    print(f"Dataset YAML: {dataset_yaml_path}")


if __name__ == "__main__":
    main()