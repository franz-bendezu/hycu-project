from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import requests

COMMONS_API = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "VisionToBlueprintMVP/0.1 (educational dataset collector; contact: local-dev)"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect CC/open furniture images from Wikimedia Commons")
    parser.add_argument("--queries-json", type=Path, required=True, help="JSON with category and query fields")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--metadata-csv", type=Path, required=True)
    parser.add_argument("--per-category", type=int, default=8)
    parser.add_argument("--thumb-width", type=int, default=320)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--sleep", type=float, default=1.8)
    parser.add_argument("--max-pages", type=int, default=4)
    return parser.parse_args()


def load_queries(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Queries file not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("queries json must be a list")
    rows: list[dict[str, str]] = []
    for item in data:
        category = str(item.get("category", "")).strip()
        query = str(item.get("query", "")).strip()
        if category and query:
            rows.append({"category": category, "query": query})
    if not rows:
        raise ValueError("No valid rows in query file")
    return rows


def load_existing_rows(metadata_csv: Path) -> list[dict[str, str]]:
    if not metadata_csv.exists():
        return []
    with metadata_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [dict(row) for row in reader]


def license_allowed(ext: dict[str, Any]) -> bool:
    short = str(ext.get("LicenseShortName", {}).get("value", "")).lower()
    usage = str(ext.get("UsageTerms", {}).get("value", "")).lower()
    text = f"{short} {usage}"
    allow_tokens = ["cc", "creative commons", "public domain", "pdm"]
    return any(token in text for token in allow_tokens)


def request_with_backoff(url: str, *, params: dict[str, Any] | None, timeout: float, attempts: int = 5) -> requests.Response:
    for i in range(attempts):
        response = requests.get(url, params=params, timeout=timeout, headers={"User-Agent": USER_AGENT})
        if response.status_code != 429:
            response.raise_for_status()
            return response
        wait = 2 ** i
        print(f"Rate-limited by server, retrying in {wait}s")
        time.sleep(wait)
    response.raise_for_status()
    return response


def fetch_candidates(query: str, thumb_width: int, timeout: float, max_pages: int, sleep: float) -> list[dict[str, Any]]:
    cont: str | None = None
    pages: list[dict[str, Any]] = []

    for _ in range(max_pages):
        params: dict[str, Any] = {
            "action": "query",
            "format": "json",
            "generator": "search",
            "gsrsearch": query,
            "gsrnamespace": 6,
            "gsrlimit": 25,
            "prop": "imageinfo",
            "iiprop": "url|extmetadata|size",
            "iiurlwidth": thumb_width,
        }
        if cont:
            params["gsroffset"] = cont

        response = request_with_backoff(COMMONS_API, params=params, timeout=timeout)
        payload = response.json()
        query_pages = payload.get("query", {}).get("pages", {})
        pages.extend(query_pages.values())

        continue_payload = payload.get("continue", {})
        if "gsroffset" not in continue_payload:
            break
        cont = str(continue_payload["gsroffset"])
        time.sleep(sleep)

    return pages


def clean_name(value: str) -> str:
    out = value.replace("File:", "").replace("/", "_").replace(" ", "_")
    return "".join(c for c in out if c.isalnum() or c in {"_", ".", "-"})


def download_binary(url: str, path: Path, timeout: float) -> bool:
    try:
        response = request_with_backoff(url, params=None, timeout=timeout)
    except Exception as exc:
        print(f"Download failed: {url} ({exc})")
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(response.content)
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

    total_new = 0
    seen_run_urls: set[str] = set()

    for row in queries:
        category = row["category"]
        query = row["query"]
        if category_counts.get(category, 0) >= args.per_category:
            print(
                f"Skipping category='{category}' query='{query}' because target is already met "
                f"({category_counts.get(category, 0)}/{args.per_category})"
            )
            continue
        print(f"Collecting {category}: {query}")

        try:
            candidates = fetch_candidates(query, args.thumb_width, args.timeout, args.max_pages, args.sleep)
        except Exception as exc:
            print(f"Search failed for {category}: {exc}")
            continue

        category_new = 0
        for page in candidates:
            if category_counts.get(category, 0) >= args.per_category:
                break

            title = str(page.get("title", ""))
            imageinfo = page.get("imageinfo", [])
            if not imageinfo:
                continue

            info = imageinfo[0]
            ext = info.get("extmetadata", {})
            if not license_allowed(ext):
                continue

            image_url = info.get("thumburl") or info.get("url")
            if not image_url:
                continue
            image_url_str = str(image_url)
            if image_url_str in existing_urls or image_url_str in seen_run_urls:
                continue

            next_index = category_counts.get(category, 0) + 1
            file_name = f"{category}_{next_index}_{clean_name(title)}"
            local_path = args.output_dir / category / file_name
            if not download_binary(image_url_str, local_path, args.timeout):
                continue

            rows.append(
                {
                    "category": category,
                    "title": title,
                    "query": query,
                    "download_url": image_url_str,
                    "source_url": str(info.get("descriptionurl", "")),
                    "license": str(ext.get("LicenseShortName", {}).get("value", "")),
                    "creator": str(ext.get("Artist", {}).get("value", "")),
                    "provider": "wikimedia_commons",
                    "local_path": str(local_path),
                }
            )
            seen_run_urls.add(image_url_str)
            category_counts[category] = category_counts.get(category, 0) + 1
            category_new += 1
            total_new += 1
            time.sleep(args.sleep)

        print(
            f"Category '{category}' now has {category_counts.get(category, 0)} images "
            f"(added {category_new} in this run, target {args.per_category})"
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
        "newly_downloaded": total_new,
        "total_rows": len(rows),
        "metadata_csv": str(args.metadata_csv),
        "output_dir": str(args.output_dir),
        "categories": sorted({r["category"] for r in rows}),
        "category_counts": category_counts,
    }
    summary_path = args.metadata_csv.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Newly downloaded in this run: {total_new}")
    print(f"Total rows in metadata: {len(rows)}")
    print(f"Metadata CSV: {args.metadata_csv}")
    print(f"Summary JSON: {summary_path}")


if __name__ == "__main__":
    main()
