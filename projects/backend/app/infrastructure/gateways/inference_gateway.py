from __future__ import annotations

import os
import uuid
from typing import Any

import httpx


class InferenceGatewayError(Exception):
    pass


class FakeInferenceGateway:
    def infer(self, image_url: str) -> dict[str, Any]:
        detected_type = "cabinet"
        width, height, depth = 800.0, 1200.0, 450.0

        lower = image_url.lower()
        if "desk" in lower:
            detected_type = "desk"
            width, height, depth = 1200.0, 750.0, 600.0
        elif "shelf" in lower:
            detected_type = "shelf"
            width, height, depth = 900.0, 1700.0, 300.0

        return {
            "detected_type": detected_type,
            "confidence": 0.94,
            "suggested_width": width,
            "suggested_height": height,
            "suggested_depth": depth,
            "image_url": image_url,
        }

    def submit(self, image_url: str) -> tuple[str, dict[str, Any]]:
        return str(uuid.uuid4()), self.infer(image_url)


class HttpInferenceGateway:
    def __init__(self, base_url: str, timeout_seconds: float = 25.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def infer(self, image_url: str) -> dict[str, Any]:
        try:
            response = httpx.post(
                f"{self.base_url}/infer",
                json={"image_url": image_url},
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
        except httpx.HTTPStatusError as exc:
            detail = ""
            try:
                body = exc.response.json()
                detail = body.get("detail") if isinstance(body, dict) else None
            except Exception:
                detail = None
            suffix = f": {detail}" if detail else ""
            raise InferenceGatewayError(
                f"Inference service returned {exc.response.status_code} at {self.base_url}{suffix}"
            ) from exc
        except httpx.HTTPError as exc:
            raise InferenceGatewayError(
                f"Cannot reach inference service at {self.base_url}: {exc}"
            ) from exc

        return {
            "detected_type": payload["detected_type"],
            "confidence": float(payload["confidence"]),
            "suggested_width": float(payload["suggested_width"]),
            "suggested_height": float(payload["suggested_height"]),
            "suggested_depth": float(payload["suggested_depth"]),
            "image_url": payload.get("image_url") or image_url,
        }

    def submit(self, image_url: str) -> tuple[str, dict[str, Any]]:
        try:
            result = self.infer(image_url)
            return str(uuid.uuid4()), result
        except InferenceGatewayError as primary_error:
            fallback_url = "http://127.0.0.1:8001"
            if self.base_url != fallback_url and "inference" in self.base_url:
                try:
                    fallback = HttpInferenceGateway(fallback_url, self.timeout_seconds)
                    result = fallback.infer(image_url)
                    return str(uuid.uuid4()), result
                except InferenceGatewayError:
                    pass
            raise primary_error


def get_inference_gateway() -> FakeInferenceGateway | HttpInferenceGateway:
    mode = os.getenv("INFERENCE_MODE", "service").strip().lower()
    if mode == "fake":
        return FakeInferenceGateway()

    service_url = os.getenv("INFERENCE_SERVICE_URL", "http://127.0.0.1:8001")
    return HttpInferenceGateway(service_url)
