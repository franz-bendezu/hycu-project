from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests

OPENVERSE_API = "https://api.openverse.org/v1/images/"
USER_AGENT = "VisionToBlueprintMVP/0.1 (training dataset collector; educational use; open-license only)"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect open-licensed furniture images from Openverse")
    parser.add_argument("--queries-json", type=Path, required=True, help="JSON file with category and query")
    parser.add_argument("--output-dir", type=Path, required=True, help="Folder to store downloaded images")
    parser.add_argument("--metadata-csv", type=Path, required=True, help="CSV file to write image metadata")
    parser.add_argument("--per-category", type=int, default=20, help="Max downloaded images per category")
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--sleep", type=float, default=0.2, help="Delay between requests")
    return parser.parse_args()


def load_existing_rows(metadata_csv: Path) -> list[dict[str, str]]:
    if not metadata_csv.exists():
        return []
    with metadata_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [dict(row) for row in reader]


def load_queries(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Queries file not found: {path}")
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise ValueError("Queries file must be a JSON list")
    parsed: list[dict[str, str]] = []
    for item in data:
        category = str(item.get("category", "unknown")).strip()
        query = str(item.get("query", "")).strip()
        if not category or not query:
            continue
        parsed.append({"category": category, "query": query})
    if not parsed:
        raise ValueError("No valid query rows found")
    return parsed


def license_allows_download(license_code: str) -> bool:
    license_code = license_code.lower().strip()
    allowed = {"cc0", "by", "by-sa", "pdm", "cc-by", "cc-by-sa"}
    return license_code in allowed


def search_files(query: str, limit: int, timeout: float, sleep: float) -> list[dict[str, Any]]:
    collected: list[dict[str, Any]] = []
    next_url = OPENVERSE_API
    page = 1
    page_size = min(max(limit * 2, 20), 80)

    while next_url and len(collected) < limit * 3:
        if next_url == OPENVERSE_API:
            params = {
                "q": query,
                "license_type": "all",
                "extension": "jpg,jpeg,png,webp",
                "page": page,
                "page_size": page_size,
            }
        else:
            params = None

        response = requests.get(
            next_url,
            params=params,
            timeout=timeout,
            headers={"User-Agent": USER_AGENT},
        )
        response.raise_for_status()

        payload = response.json()
        collected.extend(payload.get("results", []))
        next_url = payload.get("next")
        page += 1
        time.sleep(sleep)

        if page > 6:
            break

    return collected


def safe_name(title: str) -> str:
    name = title.replace("/", "_").replace(" ", "_")
    return "".join(ch for ch in name if ch.isalnum() or ch in {"_", ".", "-"})


def guess_extension(image_url: str) -> str:
    path = urlparse(image_url).path.lower()
    if path.endswith(".png"):
        return ".png"
    if path.endswith(".webp"):
        return ".webp"
    if path.endswith(".jpeg"):
        return ".jpeg"
    return ".jpg"


def download_image(url: str, target_path: Path, timeout: float) -> bool:
    try:
        response = requests.get(url, timeout=timeout, headers={"User-Agent": USER_AGENT})
        response.raise_for_status()
    except Exception as exc:
        print(f"Download failed: {url} ({exc})")
        return False

    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_bytes(response.content)
    return True


def main() -> None:
    args = parse_args()
    queries = load_queries(args.queries_json)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.metadata_csv.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = load_existing_rows(args.metadata_csv)
    existing_urls = {row.get("download_url", "") for row in rows}
    category_counts: dict[str, int] = {}
    for row in rows:
        category = row.get("category", "")
        if category:
            category_counts[category] = category_counts.get(category, 0) + 1

    total_downloaded = 0
    seen_run_urls: set[str] = set()

    for item in queries:
        category = item["category"]
        query = item["query"]
        if category_counts.get(category, 0) >= args.per_category:
            print(
                f"Skipping category='{category}' query='{query}' because target is already met "
                f"({category_counts.get(category, 0)}/{args.per_category})"
            )
            continue

        print(f"Collecting category='{category}' query='{query}'")
        try:
            candidates = search_files(query, args.per_category, args.timeout, args.sleep)
        except Exception as exc:
            print(f"Search failed for category {category}: {exc}")
            continue

        for page in candidates:
            if category_counts.get(category, 0) >= args.per_category:
                break

            title = str(page.get("title", "") or f"image_{page.get('id', 'unknown')}")
            image_url = page.get("url")
            source_url = page.get("foreign_landing_url") or page.get("foreign_landing_url") or ""
            license_code = str(page.get("license", "")).lower()

            if not image_url:
                continue
            if not license_allows_download(license_code):
                continue
            image_url_str = str(image_url)
            if image_url_str in existing_urls or image_url_str in seen_run_urls:
                continue

            next_index = category_counts.get(category, 0) + 1
            file_name = f"{category}_{next_index}_{safe_name(title)}{guess_extension(image_url_str)}"
            target_path = args.output_dir / category / file_name
            if not download_image(image_url_str, target_path, args.timeout):
                continue

            rows.append(
                {
                    "category": category,
                    "title": title,
                    "query": query,
                    "download_url": image_url_str,
                    "source_url": str(source_url),
                    "license": str(page.get("license", "")),
                    "creator": str(page.get("creator", "")),
                    "provider": str(page.get("provider", "")),
                    "local_path": str(target_path),
                }
            )
            seen_run_urls.add(image_url_str)
            category_counts[category] = category_counts.get(category, 0) + 1
            total_downloaded += 1
            time.sleep(args.sleep)

        print(
            f"Category '{category}' now has {category_counts.get(category, 0)} images "
            f"(target {args.per_category})"
        )

    with args.metadata_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "category",
                "title",
                "query",
                "download_url",
                "source_url",
                "license",
                "creator",
                "provider",
                "local_path",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "newly_downloaded": total_downloaded,
        "total_rows": len(rows),
        "categories": sorted({row["category"] for row in rows}),
        "category_counts": category_counts,
        "metadata_csv": str(args.metadata_csv),
        "output_dir": str(args.output_dir),
    }
    summary_path = args.metadata_csv.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Newly downloaded in this run: {total_downloaded}")
    print(f"Total rows in metadata: {len(rows)}")
    print(f"Metadata CSV: {args.metadata_csv}")
    print(f"Summary JSON: {summary_path}")


if __name__ == "__main__":
    main()
