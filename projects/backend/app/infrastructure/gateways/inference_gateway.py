from __future__ import annotations

import os
import uuid
from typing import Any

import httpx


class InferenceGatewayError(Exception):
    pass


class HttpInferenceGateway:
    def __init__(self, base_url: str, timeout_seconds: float = 25.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def _post_infer(self, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            response = httpx.post(
                f"{self.base_url}/infer",
                json=payload,
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            response_payload = response.json()
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

        return response_payload

    def _normalize_result(self, payload: dict[str, Any]) -> dict[str, Any]:
        required_fields = (
            "detected_type",
            "confidence",
            "suggested_width",
            "suggested_height",
            "suggested_depth",
            "components",
            "image_url",
        )
        missing = [field for field in required_fields if field not in payload]
        if missing:
            raise InferenceGatewayError(
                f"Inference service payload missing required fields: {', '.join(missing)}"
            )

        components = payload["components"]
        if not isinstance(components, list):
            raise InferenceGatewayError("Inference service payload field 'components' must be a list")

        normalized: dict[str, Any] = {
            "detected_type": payload["detected_type"],
            "confidence": float(payload["confidence"]),
            "suggested_width": float(payload["suggested_width"]),
            "suggested_height": float(payload["suggested_height"]),
            "suggested_depth": float(payload["suggested_depth"]),
            "components": components,
            "image_url": payload["image_url"],
        }

        passthrough_fields = (
            "interior",
            "door",
            "uncertainty",
            "joints",
            "hardware",
        )
        for field in passthrough_fields:
            if field in payload:
                normalized[field] = payload[field]

        image_results = payload.get("image_results")
        if image_results is not None and not isinstance(image_results, list):
            raise InferenceGatewayError("Inference service payload field 'image_results' must be a list when present")

        if image_results is not None:
            normalized["image_results"] = image_results

        images_analyzed = payload.get("images_analyzed")
        if images_analyzed is not None:
            normalized["images_analyzed"] = int(images_analyzed)

        return normalized

    def infer_many(self, image_urls: list[str]) -> dict[str, Any]:
        payload = self._post_infer({"image_urls": image_urls})
        return self._normalize_result(payload)

    def submit(self, image_urls: list[str]) -> tuple[str, dict[str, Any]]:
        result = self.infer_many(image_urls)
        return str(uuid.uuid4()), result


def get_inference_gateway() -> HttpInferenceGateway:
    service_url = os.getenv("INFERENCE_SERVICE_URL", "http://127.0.0.1:8001")
    return HttpInferenceGateway(service_url)
