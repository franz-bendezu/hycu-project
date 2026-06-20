from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse

import requests

PROMART_BASE = "https://www.promart.pe"
USER_AGENT = "VisionToBlueprintMVP/0.1 (metadata collector; contact: local-dev)"

WOOD_TOKENS = {
    "madera",
    "melamina",
    "mdf",
    "mdp",
    "aglomerado",
    "tablero",
    "laminado",
    "triplay",
    "plywood",
    "enchapado",
    "chapado",
    "osb",
    "hdf",
    "fibra de madera",
}

FLAT_BOARD_INCLUDE_TOKENS = {
    "ropero",
    "closet",
    "clset",
    "cajonera",
    "comoda",
    "comod",
    "velador",
    "mesa de noche",
    "zapatera",
    "estante",
    "librero",
    "biblioteca",
    "escritorio",
    "mueble tv",
    "centro de entretenimiento",
    "aparador",
    "vitrina",
    "rack",
    "alacena",
    "repostero",
    "bar",
    "mesa auxiliar",
    "mesa de centro",
    "mesa de comedor",
}

FLAT_BOARD_EXCLUDE_TOKENS = {
    "sofa",
    "sillon",
    "butaca",
    "poltrona",
    "puff",
    "banca",
    "banqueta",
    "colchon",
    "colch",
    "cama",
    "hamaca",
    "plastico",
    "resina",
}

EN_TEXT_MAP = {
    "muebles": "furniture",
    "dormitorio": "bedroom",
    "oficina": "office",
    "cocina": "kitchen",
    "roperos": "wardrobes",
    "ropero": "wardrobe",
    "escritorios": "desks",
    "escritorio": "desk",
    "estantes": "shelves",
    "estante": "shelf",
    "archivadores": "file cabinets",
    "archivador": "file cabinet",
    "veladores y mesas de noche": "nightstands and bedside tables",
    "velador": "nightstand",
    "mesas de noche": "bedside tables",
    "comodas": "dressers",
    "comodas": "dressers",
    "comod": "dresser",
    "cajones": "drawers",
    "cajon": "drawer",
    "puertas": "doors",
    "puerta": "door",
    "mueble de cocina": "kitchen cabinet",
    "muebles de cocina": "kitchen cabinets",
    "alacenas de cocina": "kitchen wall cabinets",
    "aparadores y vitrinas": "sideboards and display cabinets",
    "aparador": "sideboard",
    "vitrina": "display cabinet",
    "muebles auxiliares": "auxiliary furniture",
    "marron": "brown",
    "blanco": "white",
    "castano": "chestnut",
    "roble": "oak",
    "canela": "cinnamon",
    "beige": "beige",
    "si": "yes",
    "no": "no",
    "material principal": "main material",
    "material predominante de empaque primario": "primary packaging main material",
    "material de acabado": "finish material",
    "material de base del cajon": "drawer base material",
    "material del fondo": "back panel material",
    "material del cuerpo": "body material",
    "material del tablero": "board material",
    "material de tablero": "board material",
    "material de bisagras": "hinge material",
    "material de correderas": "drawer slide material",
    "material de la barra": "rod material",
    "material de repisas": "shelf material",
    "material de la estructura": "structure material",
    "material de tiradores": "handle material",
    "material de puertas": "door material",
    "material de las patas": "leg material",
    "material de la base": "base material",
    "material de la base": "base material",
    "carton": "cardboard",
    "melamina": "melamine",
    "pintura": "paint",
    "madera": "wood",
    "aglomerado": "particleboard",
    "plastico": "plastic",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect Promart furniture metadata and keep only flat-board wood-derived furniture"
    )
    parser.add_argument("--category-url", default=f"{PROMART_BASE}/muebles")
    parser.add_argument(
        "--category-urls-json",
        type=Path,
        default=None,
        help="Optional JSON file with a list of category URLs to crawl",
    )
    parser.add_argument(
        "--category-ids-json",
        type=Path,
        default=None,
        help="Optional JSON file with category ids to query directly via catalog API",
    )
    parser.add_argument(
        "--category-id",
        default="",
        help="Optional single category id (example: 890) to query via catalog API",
    )
    parser.add_argument(
        "--api-fq",
        action="append",
        default=[],
        help="Extra API fq filters, repeat flag for multiple filters (example: specificationFilter_10973:MDP)",
    )
    parser.add_argument(
        "--api-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use only catalog API pagination and skip HTML listing discovery",
    )
    parser.add_argument(
        "--discover-subcategories",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Auto-discover category links under each category URL",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("datasets/raw/promart_flat_wood_metadata.csv"),
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=Path("datasets/raw/promart_flat_wood_metadata.jsonl"),
    )
    parser.add_argument(
        "--download-images-during-collect",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Download product images while collecting accepted products",
    )
    parser.add_argument(
        "--images-output-dir",
        type=Path,
        default=Path("datasets/raw/promart_products"),
        help="Root output directory for per-product image folders",
    )
    parser.add_argument(
        "--images-metadata-csv",
        type=Path,
        default=Path("datasets/raw/promart_product_images_metadata.csv"),
        help="CSV output with one row per downloaded image",
    )
    parser.add_argument(
        "--max-images-per-product",
        type=int,
        default=0,
        help="Maximum images to save per product (0 means keep all)",
    )
    parser.add_argument("--page-size", type=int, default=24)
    parser.add_argument("--max-pages", type=int, default=60)
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=50,
        help="Persist CSV/JSONL/images metadata every N inspected products",
    )
    parser.add_argument("--sleep", type=float, default=1.5)
    parser.add_argument("--timeout", type=float, default=25.0)
    parser.add_argument("--limit-products", type=int, default=0, help="0 means no limit")
    return parser.parse_args()


def normalize_text(value: str) -> str:
    cleaned = value.lower().strip()
    repl = (
        ("á", "a"),
        ("é", "e"),
        ("í", "i"),
        ("ó", "o"),
        ("ú", "u"),
        ("ñ", "n"),
    )
    for old, new in repl:
        cleaned = cleaned.replace(old, new)
    return cleaned


def translate_to_english(value: str) -> str:
    text = normalize_text(value)
    for src in sorted(EN_TEXT_MAP.keys(), key=len, reverse=True):
        text = re.sub(rf"\b{re.escape(src)}\b", EN_TEXT_MAP[src], text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def request_with_backoff(
    session: requests.Session,
    url: str,
    *,
    timeout: float,
    params: Any = None,
    attempts: int = 5,
) -> requests.Response:
    for i in range(attempts):
        response = session.get(url, params=params, timeout=timeout)
        if response.status_code != 429:
            response.raise_for_status()
            return response
        wait = min(2 ** i, 20)
        print(f"Rate-limited by server, retrying in {wait}s")
        time.sleep(wait)
    response.raise_for_status()
    return response


def discover_listing_config(session: requests.Session, category_url: str, timeout: float) -> tuple[str, str]:
    html = request_with_backoff(session, category_url, timeout=timeout).text
    match = re.search(
        r"/buscapagina\?fq=C%3a%2f(?P<cat>\d+)%2f&PS=1&sl=(?P<sl>[a-z0-9\-]+)",
        html,
        re.IGNORECASE,
    )
    if not match:
        raise RuntimeError("Could not discover buscapagina listing config from category page")
    return match.group("cat"), match.group("sl")


def discover_subcategory_urls(session: requests.Session, category_url: str, timeout: float) -> list[str]:
    html = request_with_backoff(session, category_url, timeout=timeout).text
    hrefs = re.findall(r'href=["\']([^"\']+)["\']', html, flags=re.IGNORECASE)
    parsed_base = urlparse(PROMART_BASE)

    discovered: list[str] = []
    seen: set[str] = set()
    candidates = [category_url, *hrefs]
    for href in candidates:
        absolute = urljoin(category_url, href)
        parsed = urlparse(absolute)
        if parsed.scheme not in {"http", "https"}:
            continue
        if parsed.netloc != parsed_base.netloc:
            continue
        normalized_path = parsed.path.rstrip("/")
        if "/muebles" not in normalized_path:
            continue
        cleaned = f"{parsed.scheme}://{parsed.netloc}{normalized_path}"
        if cleaned in seen:
            continue
        seen.add(cleaned)
        discovered.append(cleaned)
    return discovered


def load_category_urls(args: argparse.Namespace) -> list[str]:
    if args.category_urls_json is not None:
        if not args.category_urls_json.exists():
            raise FileNotFoundError(f"Category URLs JSON not found: {args.category_urls_json}")
        data = json.loads(args.category_urls_json.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError("--category-urls-json must contain a JSON array")
        out = [str(item).strip() for item in data if str(item).strip()]
        if not out:
            raise ValueError("--category-urls-json did not contain valid URLs")
        return out
    return [str(args.category_url).strip()]


def load_category_ids(args: argparse.Namespace) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()

    if args.category_id.strip():
        value = args.category_id.strip()
        if value not in seen:
            seen.add(value)
            out.append(value)

    if args.category_ids_json is not None:
        if not args.category_ids_json.exists():
            raise FileNotFoundError(f"Category IDs JSON not found: {args.category_ids_json}")
        data = json.loads(args.category_ids_json.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError("--category-ids-json must contain a JSON array")
        for item in data:
            value = str(item).strip()
            if not value or value in seen:
                continue
            seen.add(value)
            out.append(value)

    return out


def iter_product_ids(
    session: requests.Session,
    *,
    category_id: str,
    sl: str,
    page_size: int,
    max_pages: int,
    timeout: float,
    sleep: float,
) -> list[str]:
    ids: list[str] = []
    seen: set[str] = set()

    for page in range(1, max_pages + 1):
        url = (
            f"{PROMART_BASE}/buscapagina?fq=C%3a%2f{category_id}%2f&PS={page_size}"
            f"&sl={sl}&cc={page_size}&sm=0&PageNumber={page}"
        )
        html = request_with_backoff(session, url, timeout=timeout).text
        page_ids = re.findall(r"helperComplement_(\d+)", html)
        if not page_ids:
            print(f"No product ids found on page {page}; stopping pagination.")
            break

        new_in_page = 0
        for pid in page_ids:
            if pid in seen:
                continue
            seen.add(pid)
            ids.append(pid)
            new_in_page += 1

        print(f"Page {page}: found {len(page_ids)} ids, new unique {new_in_page}, total unique {len(ids)}")
        if new_in_page == 0:
            break
        time.sleep(sleep)

    return ids


def iter_products_from_api(
    session: requests.Session,
    *,
    category_id: str,
    extra_fq: list[str],
    timeout: float,
    sleep: float,
    page_size: int,
    max_pages: int,
) -> list[dict[str, Any]]:
    products: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    base_url = f"{PROMART_BASE}/api/catalog_system/pub/products/search/"

    for page in range(max_pages):
        start = page * page_size
        end = start + page_size - 1
        params: list[tuple[str, str]] = [
            ("_from", str(start)),
            ("_to", str(end)),
            ("sc", "2"),
            ("O", "OrderByScoreDESC"),
            ("fq", f"C:/{category_id}/"),
            ("fq", "isAvailablePerSalesChannel_2:1"),
        ]
        for fq in extra_fq:
            params.append(("fq", fq))

        try:
            payload = request_with_backoff(session, base_url, timeout=timeout, params=params).json()
        except Exception as exc:
            print(f"Failed API listing page {page + 1} for category {category_id}: {exc}")
            break

        if not isinstance(payload, list) or not payload:
            print(f"Category {category_id}: no products in page {page + 1}; stopping pagination.")
            break

        new_count = 0
        for product in payload:
            if not isinstance(product, dict):
                continue
            pid = str(product.get("productId", "")).strip()
            if not pid or pid in seen_ids:
                continue
            seen_ids.add(pid)
            products.append(product)
            new_count += 1

        print(
            f"Category {category_id} page {page + 1}: received {len(payload)} products, "
            f"new unique {new_count}, total unique {len(products)}"
        )
        if len(payload) < page_size or new_count == 0:
            break
        time.sleep(sleep)

    return products


def collect_material_fields(product: dict[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    for key, value in product.items():
        if "material" in normalize_text(str(key)):
            if isinstance(value, list):
                out[str(key)] = " | ".join(str(v) for v in value if str(v).strip())
            else:
                out[str(key)] = str(value)
    return out


def classify_flat_wood_furniture(product: dict[str, Any]) -> tuple[bool, str, str]:
    name = str(product.get("productName", ""))
    categories = " | ".join(str(c) for c in product.get("categories", []))
    text = normalize_text(f"{name} {categories}")

    material_fields = collect_material_fields(product)
    material_text = normalize_text(" ".join(v for v in material_fields.values() if v))

    has_wood_material = any(tok in material_text for tok in WOOD_TOKENS)
    has_flat_include = any(tok in text for tok in FLAT_BOARD_INCLUDE_TOKENS)
    has_flat_exclude = any(tok in text for tok in FLAT_BOARD_EXCLUDE_TOKENS)

    accepted = has_wood_material and has_flat_include and not has_flat_exclude
    return accepted, material_text, text


def first_seller_offer(product: dict[str, Any]) -> tuple[str, str]:
    items = product.get("items", [])
    if not items:
        return "", ""
    sellers = items[0].get("sellers", [])
    if not sellers:
        return "", ""
    comm = sellers[0].get("commertialOffer", {})
    list_price = str(comm.get("ListPrice", ""))
    price = str(comm.get("Price", ""))
    return list_price, price


def scalar_text(value: Any) -> str:
    if isinstance(value, list):
        return " | ".join(str(v).strip() for v in value if str(v).strip())
    if value is None:
        return ""
    return str(value).strip()


def full_link(link: str) -> str:
    cleaned = link.strip()
    if not cleaned:
        return ""
    if cleaned.startswith("http://") or cleaned.startswith("https://"):
        return cleaned
    return f"{PROMART_BASE}{cleaned}"


def safe_slug(value: str, max_len: int = 80) -> str:
    slug = normalize_text(value)
    slug = re.sub(r"[^a-z0-9_-]+", "_", slug).strip("_")
    if not slug:
        return "product"
    return slug[:max_len]


def first_image(product: dict[str, Any]) -> str:
    items = product.get("items", [])
    if not items:
        return ""
    images = items[0].get("images", [])
    if not images:
        return ""
    return str(images[0].get("imageUrl", ""))


def all_item_images(product: dict[str, Any]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    seen: set[str] = set()
    items = product.get("items", [])
    if not isinstance(items, list):
        return out

    for item in items:
        if not isinstance(item, dict):
            continue
        item_id = str(item.get("itemId", "")).strip()
        images = item.get("images", [])
        if not isinstance(images, list):
            continue
        for image in images:
            if not isinstance(image, dict):
                continue
            image_url = str(image.get("imageUrl", "")).strip()
            if not image_url or image_url in seen:
                continue
            seen.add(image_url)
            out.append(
                {
                    "item_id": item_id,
                    "image_id": str(image.get("imageId", "")).strip(),
                    "image_label": str(image.get("imageLabel", "")).strip(),
                    "image_tag": str(image.get("imageTag", "")).strip(),
                    "image_text": str(image.get("imageText", "")).strip(),
                    "image_url": image_url,
                }
            )
    return out


def extract_specs(product: dict[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    specs = product.get("specificationGroups", [])
    if not isinstance(specs, list):
        return out
    for group in specs:
        if not isinstance(group, dict):
            continue
        for spec in group.get("specifications", []):
            if not isinstance(spec, dict):
                continue
            key = str(spec.get("name", "")).strip()
            if not key:
                continue
            values = spec.get("values", [])
            if isinstance(values, list):
                out[key] = " | ".join(str(v).strip() for v in values if str(v).strip())
            else:
                out[key] = str(values).strip()
    return out


def infer_image_extension(image_url: str) -> str:
    lowered = image_url.lower()
    for candidate in (".jpg", ".jpeg", ".png", ".webp"):
        if candidate in lowered:
            return candidate
    return ".jpg"


def download_product_images(
    session: requests.Session,
    *,
    product: dict[str, Any],
    images_root: Path,
    timeout: float,
    source_category_url: str,
    max_images_per_product: int,
) -> list[dict[str, str]]:
    product_id = str(product.get("productId", "")).strip() or "unknown"
    product_name = str(product.get("productName", "")).strip() or "product"
    folder_name = f"{product_id}_{safe_slug(product_name)}"
    product_dir = images_root / folder_name
    product_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    images = all_item_images(product)
    if max_images_per_product > 0:
        images = images[:max_images_per_product]

    for idx, image in enumerate(images, start=1):
        image_url = image.get("image_url", "").strip()
        if not image_url:
            continue

        ext = infer_image_extension(image_url)
        url_hash = hashlib.md5(image_url.encode("utf-8")).hexdigest()[:10]
        file_name = f"{idx:03d}_{url_hash}{ext}"
        local_path = product_dir / file_name

        status = "existing"
        if not local_path.exists():
            try:
                content = request_with_backoff(session, image_url, timeout=timeout).content
                local_path.write_bytes(content)
                status = "downloaded"
            except Exception as exc:
                rows.append(
                    {
                        "product_id": product_id,
                        "product_name": product_name,
                        "item_id": image.get("item_id", ""),
                        "image_id": image.get("image_id", ""),
                        "image_label": image.get("image_label", ""),
                        "image_tag": image.get("image_tag", ""),
                        "image_text": image.get("image_text", ""),
                        "image_url": image_url,
                        "local_path": str(local_path),
                        "status": f"error:{exc}",
                        "source_category_url": source_category_url,
                        "source": "promart.pe",
                    }
                )
                continue

        rows.append(
            {
                "product_id": product_id,
                "product_name": product_name,
                "item_id": image.get("item_id", ""),
                "image_id": image.get("image_id", ""),
                "image_label": image.get("image_label", ""),
                "image_tag": image.get("image_tag", ""),
                "image_text": image.get("image_text", ""),
                "image_url": image_url,
                "local_path": str(local_path),
                "status": status,
                "source_category_url": source_category_url,
                "source": "promart.pe",
            }
        )

    return rows


def product_to_row(product: dict[str, Any], material_text: str, source_category_url: str) -> dict[str, str]:
    list_price, best_price = first_seller_offer(product)
    material_fields = collect_material_fields(product)
    translated_material_fields = {
        translate_to_english(k): translate_to_english(v) for k, v in material_fields.items()
    }
    specs = extract_specs(product)
    translated_specs = {translate_to_english(k): translate_to_english(v) for k, v in specs.items()}
    item_images = all_item_images(product)
    image_urls = [img.get("image_url", "") for img in item_images if img.get("image_url", "")]

    return {
        "product_id": str(product.get("productId", "")),
        "product_name": translate_to_english(str(product.get("productName", ""))),
        "brand": str(product.get("brand", "")),
        "model": translate_to_english(scalar_text(product.get("Modelo", ""))),
        "link": full_link(str(product.get("link", ""))),
        "image_url": first_image(product),
        "category_path": translate_to_english(" | ".join(str(c) for c in product.get("categories", []))),
        "material_text": translate_to_english(material_text),
        "material_fields_json": json.dumps(translated_material_fields, ensure_ascii=True),
        "specs_json": json.dumps(translated_specs, ensure_ascii=True),
        "items_images_json": json.dumps(item_images, ensure_ascii=True),
        "image_urls_json": json.dumps(image_urls, ensure_ascii=True),
        "width": scalar_text(product.get("Ancho Del Producto", "")),
        "height": scalar_text(product.get("Altura Del Producto", "")),
        "depth": scalar_text(product.get("Profundidad Del Producto", "")),
        "weight": scalar_text(product.get("Peso Del Producto", "")),
        "requires_installation": translate_to_english(scalar_text(product.get("Requiere instalación", ""))),
        "list_price": list_price,
        "best_price": best_price,
        "source_category_url": source_category_url,
        "source": "promart.pe",
    }


def fetch_product_by_id(session: requests.Session, product_id: str, timeout: float) -> dict[str, Any] | None:
    url = f"{PROMART_BASE}/api/catalog_system/pub/products/search/?fq=productId:{product_id}"
    try:
        payload = request_with_backoff(session, url, timeout=timeout).json()
    except Exception as exc:
        print(f"Failed product id {product_id}: {exc}")
        return None
    if not payload:
        return None
    if not isinstance(payload, list):
        return None
    return dict(payload[0])


def write_outputs(
    csv_path: Path,
    jsonl_path: Path,
    rows: list[dict[str, str]],
    raw_products: list[dict[str, Any]],
    summary: dict[str, Any],
    image_rows: list[dict[str, str]],
    images_metadata_csv: Path,
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "product_id",
        "product_name",
        "brand",
        "model",
        "link",
        "image_url",
        "category_path",
        "material_text",
        "material_fields_json",
        "specs_json",
        "items_images_json",
        "image_urls_json",
        "width",
        "height",
        "depth",
        "weight",
        "requires_installation",
        "list_price",
        "best_price",
        "source_category_url",
        "source",
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with jsonl_path.open("w", encoding="utf-8") as f:
        for product in raw_products:
            f.write(json.dumps(product, ensure_ascii=True) + "\n")

    images_metadata_csv.parent.mkdir(parents=True, exist_ok=True)
    with images_metadata_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "product_id",
                "product_name",
                "item_id",
                "image_id",
                "image_label",
                "image_tag",
                "image_text",
                "image_url",
                "local_path",
                "status",
                "source_category_url",
                "source",
            ],
        )
        writer.writeheader()
        writer.writerows(image_rows)

    summary_path = csv_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")


def write_checkpoint_outputs(
    csv_path: Path,
    jsonl_path: Path,
    rows: list[dict[str, str]],
    raw_products: list[dict[str, Any]],
    summary: dict[str, Any],
    image_rows: list[dict[str, str]],
    images_metadata_csv: Path,
) -> None:
    tmp_csv = csv_path.with_suffix(".checkpoint.csv")
    tmp_jsonl = jsonl_path.with_suffix(".checkpoint.jsonl")
    tmp_images_csv = images_metadata_csv.with_suffix(".checkpoint.csv")
    write_outputs(tmp_csv, tmp_jsonl, rows, raw_products, summary, image_rows, tmp_images_csv)


def main() -> None:
    args = parse_args()

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json,text/html,*/*"})

    seed_category_urls = load_category_urls(args)
    explicit_category_ids = load_category_ids(args)
    all_category_urls: list[str] = []
    seen_category_urls: set[str] = set()
    if not args.api_only:
        for category_url in seed_category_urls:
            candidates = [category_url]
            if args.discover_subcategories:
                try:
                    candidates = discover_subcategory_urls(session, category_url, args.timeout)
                except Exception as exc:
                    print(f"Subcategory discovery failed for {category_url}: {exc}")
            for candidate in candidates:
                if candidate in seen_category_urls:
                    continue
                seen_category_urls.add(candidate)
                all_category_urls.append(candidate)

    if args.api_only and not explicit_category_ids:
        raise ValueError("When --api-only is enabled, provide --category-id or --category-ids-json")

    product_ids: list[str] = []
    source_category_by_product_id: dict[str, str] = {}
    listing_configs: dict[str, dict[str, str]] = {}
    seen_ids: set[str] = set()
    products_by_id: dict[str, dict[str, Any]] = {}

    for category_id in explicit_category_ids:
        products = iter_products_from_api(
            session,
            category_id=category_id,
            extra_fq=args.api_fq,
            timeout=args.timeout,
            sleep=args.sleep,
            page_size=args.page_size,
            max_pages=args.max_pages,
        )
        listing_configs[f"api://category/{category_id}"] = {
            "category_id": category_id,
            "layout_id": "api",
        }
        for product in products:
            pid = str(product.get("productId", "")).strip()
            if not pid or pid in seen_ids:
                continue
            seen_ids.add(pid)
            product_ids.append(pid)
            products_by_id[pid] = product
            source_category_by_product_id[pid] = f"api://category/{category_id}"

    for category_url in all_category_urls:
        try:
            category_id, sl = discover_listing_config(session, category_url, args.timeout)
        except Exception as exc:
            print(f"Skip category {category_url}: could not discover listing config ({exc})")
            continue

        listing_configs[category_url] = {"category_id": category_id, "layout_id": sl}
        print(f"Discovered listing config: url={category_url} category_id={category_id}, sl={sl}")
        ids = iter_product_ids(
            session,
            category_id=category_id,
            sl=sl,
            page_size=args.page_size,
            max_pages=args.max_pages,
            timeout=args.timeout,
            sleep=args.sleep,
        )

        for pid in ids:
            if pid in seen_ids:
                continue
            seen_ids.add(pid)
            product_ids.append(pid)
            source_category_by_product_id[pid] = category_url

    if args.limit_products > 0:
        product_ids = product_ids[: args.limit_products]

    kept_rows: list[dict[str, str]] = []
    kept_raw: list[dict[str, Any]] = []
    image_rows: list[dict[str, str]] = []
    inspected = 0

    for product_id in product_ids:
        product = products_by_id.get(product_id)
        if product is None:
            product = fetch_product_by_id(session, product_id, args.timeout)
        inspected += 1
        if not product:
            time.sleep(args.sleep)
            continue

        accepted, material_text, _flat_text = classify_flat_wood_furniture(product)
        if accepted:
            source_category_url = source_category_by_product_id.get(product_id, args.category_url)
            kept_rows.append(product_to_row(product, material_text, source_category_url))
            kept_raw.append(product)
            if args.download_images_during_collect:
                image_rows.extend(
                    download_product_images(
                        session,
                        product=product,
                        images_root=args.images_output_dir,
                        timeout=args.timeout,
                        source_category_url=source_category_url,
                        max_images_per_product=args.max_images_per_product,
                    )
                )

        if inspected % 25 == 0:
            print(f"Inspected {inspected}/{len(product_ids)} products; accepted {len(kept_rows)}")

        if args.checkpoint_every > 0 and inspected % args.checkpoint_every == 0:
            checkpoint_summary = {
                "seed_category_urls": seed_category_urls,
                "crawled_category_urls": all_category_urls,
                "listing_configs": listing_configs,
                "total_product_ids": len(product_ids),
                "inspected_products": inspected,
                "accepted_products": len(kept_rows),
                "downloaded_or_indexed_images": len(image_rows),
                "unique_brands": sorted({row["brand"] for row in kept_rows if row["brand"]}),
                "output_csv": str(args.output_csv.with_suffix(".checkpoint.csv")),
                "output_jsonl": str(args.output_jsonl.with_suffix(".checkpoint.jsonl")),
                "images_metadata_csv": str(args.images_metadata_csv.with_suffix(".checkpoint.csv")),
                "images_output_dir": str(args.images_output_dir),
                "is_checkpoint": True,
            }
            write_checkpoint_outputs(
                args.output_csv,
                args.output_jsonl,
                kept_rows,
                kept_raw,
                checkpoint_summary,
                image_rows,
                args.images_metadata_csv,
            )

        time.sleep(args.sleep)

    unique_brands = sorted({row["brand"] for row in kept_rows if row["brand"]})
    summary = {
        "seed_category_urls": seed_category_urls,
        "crawled_category_urls": all_category_urls,
        "listing_configs": listing_configs,
        "total_product_ids": len(product_ids),
        "inspected_products": inspected,
        "accepted_products": len(kept_rows),
        "downloaded_or_indexed_images": len(image_rows),
        "unique_brands": unique_brands,
        "output_csv": str(args.output_csv),
        "output_jsonl": str(args.output_jsonl),
        "images_metadata_csv": str(args.images_metadata_csv),
        "images_output_dir": str(args.images_output_dir),
    }

    write_outputs(
        args.output_csv,
        args.output_jsonl,
        kept_rows,
        kept_raw,
        summary,
        image_rows,
        args.images_metadata_csv,
    )

    print(f"Inspected products: {inspected}")
    print(f"Accepted flat-board wood products: {len(kept_rows)}")
    print(f"CSV: {args.output_csv}")
    print(f"JSONL: {args.output_jsonl}")
    print(f"Summary: {args.output_csv.with_suffix('.summary.json')}")
    print(f"Images metadata CSV: {args.images_metadata_csv}")
    print(f"Images root dir: {args.images_output_dir}")


if __name__ == "__main__":
    main()
