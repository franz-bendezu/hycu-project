from __future__ import annotations

import argparse
import base64
from pathlib import Path
import sys

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.main import app


def file_to_data_url(path: Path) -> str:
    raw = path.read_bytes()
    suffix = path.suffix.lower()
    if suffix == ".jpg":
        suffix = ".jpeg"
    mime = {
        ".png": "image/png",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
    }.get(suffix, "application/octet-stream")
    encoded = base64.b64encode(raw).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local E2E inference for an image file")
    parser.add_argument("image", type=Path, help="Path to local image file")
    args = parser.parse_args()

    if not args.image.exists() or not args.image.is_file():
        raise SystemExit(f"Image not found: {args.image}")

    data_url = file_to_data_url(args.image)

    with TestClient(app) as client:
        response = client.post("/infer", json={"image_urls": [data_url]})

    print(f"status={response.status_code}")
    body = response.json()
    print(f"detected_type={body.get('detected_type')}")
    print(f"confidence={body.get('confidence')}")
    print(f"images_analyzed={body.get('images_analyzed')}")
    print(f"components={len(body.get('components', []))}")

    evidence = body.get("evidence") or []
    detections = evidence[0].get("raw_detections", []) if evidence else []
    print(f"detections={len(detections)}")
    if detections:
        print(f"first_detection_keys={sorted(detections[0].keys())}")


if __name__ == "__main__":
    main()
