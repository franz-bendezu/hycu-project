from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect furniture images and metadata with Requests/Selenium-ready flow")
    parser.add_argument("--sources-json", type=Path, required=True, help="JSON file with category and product_urls list")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to save downloaded images")
    parser.add_argument("--metadata-csv", type=Path, required=True, help="CSV path for collected metadata")
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument(
        "--use-selenium",
        action="store_true",
        help="Render product pages with Selenium before image extraction",
    )
    return parser.parse_args()


def load_sources(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Sources file not found: {path}")
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise ValueError("Sources JSON must be a list")
    return data


def extract_image_urls_from_html(base_url: str, html: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    candidates: list[str] = []

    for img in soup.find_all("img"):
        src = img.get("src") or img.get("data-src")
        if src:
            candidates.append(urljoin(base_url, src))

    # Include common image links from script blobs as a fallback.
    candidates.extend(re.findall(r"https?://[^\"'\s>]+\.(?:jpg|jpeg|png|webp)", html, flags=re.IGNORECASE))

    normalized: list[str] = []
    seen: set[str] = set()
    for c in candidates:
        if c.lower().endswith((".jpg", ".jpeg", ".png", ".webp")) and c not in seen:
            seen.add(c)
            normalized.append(c)
    return normalized


def download_file(url: str, target_path: Path, timeout: float) -> bool:
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
    except Exception as exc:
        print(f"Skip download failed: {url} ({exc})")
        return False

    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_bytes(response.content)
    return True


def iter_sources(sources: list[dict]) -> Iterable[tuple[str, str]]:
    for item in sources:
        category = str(item.get("category", "unknown"))
        for url in item.get("product_urls", []):
            yield category, str(url)


def safe_basename_from_url(url: str) -> str:
    parsed = urlparse(url)
    base = Path(parsed.path).name or "image"
    base = re.sub(r"[^a-zA-Z0-9_.-]", "_", base)
    if not base.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
        base += ".jpg"
    return base


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.metadata_csv.parent.mkdir(parents=True, exist_ok=True)

    sources = load_sources(args.sources_json)
    rows: list[dict[str, str]] = []
    downloaded = 0

    browser = None
    if args.use_selenium:
        browser = build_selenium_driver()

    try:
        for category, product_url in iter_sources(sources):
            try:
                if browser:
                    browser.get(product_url)
                    html = browser.page_source
                else:
                    page = requests.get(product_url, timeout=args.timeout)
                    page.raise_for_status()
                    html = page.text
            except Exception as exc:
                print(f"Skip page failed: {product_url} ({exc})")
                continue

            image_urls = extract_image_urls_from_html(product_url, html)
            for idx, image_url in enumerate(image_urls, start=1):
                file_name = f"{category}_{idx}_{safe_basename_from_url(image_url)}"
                local_path = args.output_dir / category / file_name
                if download_file(image_url, local_path, args.timeout):
                    downloaded += 1
                    rows.append(
                        {
                            "category": category,
                            "product_url": product_url,
                            "image_url": image_url,
                            "local_path": str(local_path),
                        }
                    )
    finally:
        if browser is not None:
            browser.quit()

    with args.metadata_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["category", "product_url", "image_url", "local_path"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Collected {downloaded} images")
    print(f"Metadata written to {args.metadata_csv}")


def build_selenium_driver():
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options

    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    return webdriver.Chrome(options=options)


if __name__ == "__main__":
    main()
