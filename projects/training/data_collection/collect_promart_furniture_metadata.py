from __future__ import annotations

import argparse
import csv
import json
import re
import time
from pathlib import Path
from typing import Any

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
        "--output-csv",
        type=Path,
        default=Path("datasets/raw/promart_flat_wood_metadata.csv"),
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=Path("datasets/raw/promart_flat_wood_metadata.jsonl"),
    )
    parser.add_argument("--page-size", type=int, default=24)
    parser.add_argument("--max-pages", type=int, default=60)
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
    params: dict[str, Any] | None = None,
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


def first_image(product: dict[str, Any]) -> str:
    items = product.get("items", [])
    if not items:
        return ""
    images = items[0].get("images", [])
    if not images:
        return ""
    return str(images[0].get("imageUrl", ""))


def product_to_row(product: dict[str, Any], material_text: str) -> dict[str, str]:
    list_price, best_price = first_seller_offer(product)
    material_fields = collect_material_fields(product)
    translated_material_fields = {
        translate_to_english(k): translate_to_english(v) for k, v in material_fields.items()
    }

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
        "width": scalar_text(product.get("Ancho Del Producto", "")),
        "height": scalar_text(product.get("Altura Del Producto", "")),
        "depth": scalar_text(product.get("Profundidad Del Producto", "")),
        "weight": scalar_text(product.get("Peso Del Producto", "")),
        "requires_installation": translate_to_english(scalar_text(product.get("Requiere instalación", ""))),
        "list_price": list_price,
        "best_price": best_price,
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
        "width",
        "height",
        "depth",
        "weight",
        "requires_installation",
        "list_price",
        "best_price",
        "source",
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with jsonl_path.open("w", encoding="utf-8") as f:
        for product in raw_products:
            f.write(json.dumps(product, ensure_ascii=True) + "\n")

    summary_path = csv_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")


def main() -> None:
    args = parse_args()

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json,text/html,*/*"})

    category_id, sl = discover_listing_config(session, args.category_url, args.timeout)
    print(f"Discovered listing config: category_id={category_id}, sl={sl}")

    product_ids = iter_product_ids(
        session,
        category_id=category_id,
        sl=sl,
        page_size=args.page_size,
        max_pages=args.max_pages,
        timeout=args.timeout,
        sleep=args.sleep,
    )

    if args.limit_products > 0:
        product_ids = product_ids[: args.limit_products]

    kept_rows: list[dict[str, str]] = []
    kept_raw: list[dict[str, Any]] = []
    inspected = 0

    for product_id in product_ids:
        product = fetch_product_by_id(session, product_id, args.timeout)
        inspected += 1
        if not product:
            time.sleep(args.sleep)
            continue

        accepted, material_text, _flat_text = classify_flat_wood_furniture(product)
        if accepted:
            kept_rows.append(product_to_row(product, material_text))
            kept_raw.append(product)

        if inspected % 25 == 0:
            print(f"Inspected {inspected}/{len(product_ids)} products; accepted {len(kept_rows)}")

        time.sleep(args.sleep)

    unique_brands = sorted({row["brand"] for row in kept_rows if row["brand"]})
    summary = {
        "category_url": args.category_url,
        "category_id": category_id,
        "layout_id": sl,
        "total_product_ids": len(product_ids),
        "inspected_products": inspected,
        "accepted_products": len(kept_rows),
        "unique_brands": unique_brands,
        "output_csv": str(args.output_csv),
        "output_jsonl": str(args.output_jsonl),
    }

    write_outputs(args.output_csv, args.output_jsonl, kept_rows, kept_raw, summary)

    print(f"Inspected products: {inspected}")
    print(f"Accepted flat-board wood products: {len(kept_rows)}")
    print(f"CSV: {args.output_csv}")
    print(f"JSONL: {args.output_jsonl}")
    print(f"Summary: {args.output_csv.with_suffix('.summary.json')}")


if __name__ == "__main__":
    main()
