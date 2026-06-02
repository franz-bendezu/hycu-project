from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class ValidationStats:
    files_total: int = 0
    files_empty: int = 0
    lines_total: int = 0
    boxes_total: int = 0
    errors_total: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate YOLO dataset labels against dataset YAML schema")
    parser.add_argument("--dataset-yaml", type=Path, required=True)
    return parser.parse_args()


def _resolve_dataset_root(dataset_yaml: Path, yaml_doc: dict) -> Path:
    dataset_root = yaml_doc.get("path")
    if not isinstance(dataset_root, str) or not dataset_root.strip():
        raise ValueError("Dataset YAML must define a non-empty 'path'")
    root = Path(dataset_root)
    if root.is_absolute():
        return root

    # Accept both common conventions:
    # 1) dataset path relative to the YAML file directory
    # 2) dataset path relative to the training project root
    candidates = [
        (dataset_yaml.parent / root).resolve(),
        (dataset_yaml.parents[1] / root).resolve(),
        (Path.cwd() / root).resolve(),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _class_count(yaml_doc: dict) -> int:
    names = yaml_doc.get("names")
    if isinstance(names, dict):
        return len(names)
    if isinstance(names, list):
        return len(names)
    nc = yaml_doc.get("nc")
    if isinstance(nc, int) and nc > 0:
        return nc
    raise ValueError("Dataset YAML must define names (list/dict) or nc")


def _validate_label_file(path: Path, num_classes: int, stats: ValidationStats) -> None:
    stats.files_total += 1
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        stats.files_empty += 1
        return

    for line_no, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        stats.lines_total += 1
        parts = line.split()
        if len(parts) != 5:
            print(f"ERROR {path}:{line_no}: expected 5 fields, got {len(parts)}")
            stats.errors_total += 1
            continue

        try:
            class_id = int(parts[0])
        except ValueError:
            print(f"ERROR {path}:{line_no}: class id is not int: {parts[0]}")
            stats.errors_total += 1
            continue

        if class_id < 0 or class_id >= num_classes:
            print(f"ERROR {path}:{line_no}: class id {class_id} out of range [0, {num_classes - 1}]")
            stats.errors_total += 1
            continue

        try:
            x_center, y_center, width, height = (float(value) for value in parts[1:])
        except ValueError:
            print(f"ERROR {path}:{line_no}: box values must be float")
            stats.errors_total += 1
            continue

        if not (0.0 <= x_center <= 1.0 and 0.0 <= y_center <= 1.0):
            print(f"ERROR {path}:{line_no}: x/y center must be in [0,1]")
            stats.errors_total += 1
            continue
        if not (0.0 < width <= 1.0 and 0.0 < height <= 1.0):
            print(f"ERROR {path}:{line_no}: width/height must be in (0,1]")
            stats.errors_total += 1
            continue

        stats.boxes_total += 1


def main() -> None:
    args = parse_args()
    dataset_yaml = args.dataset_yaml.resolve()
    if not dataset_yaml.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {dataset_yaml}")

    yaml_doc = yaml.safe_load(dataset_yaml.read_text(encoding="utf-8"))
    if not isinstance(yaml_doc, dict):
        raise ValueError("Dataset YAML must parse into a mapping")

    dataset_root = _resolve_dataset_root(dataset_yaml, yaml_doc)
    num_classes = _class_count(yaml_doc)
    labels_root = dataset_root / "labels"
    if not labels_root.exists():
        raise FileNotFoundError(f"Labels directory not found: {labels_root}")

    stats = ValidationStats()
    for split in ("train", "val"):
        split_dir = labels_root / split
        if not split_dir.exists():
            continue
        for label_file in sorted(split_dir.glob("*.txt")):
            _validate_label_file(label_file, num_classes, stats)

    print(
        "Validation summary: "
        f"files={stats.files_total}, empty_files={stats.files_empty}, "
        f"lines={stats.lines_total}, boxes={stats.boxes_total}, errors={stats.errors_total}"
    )

    if stats.errors_total > 0:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
