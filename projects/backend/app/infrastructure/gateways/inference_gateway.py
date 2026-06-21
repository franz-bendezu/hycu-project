from __future__ import annotations

import os
import uuid
from typing import Any

import httpx
from pydantic import ValidationError

from app.presentation.schemas.inference import InferenceOutput


class InferenceGatewayError(Exception):
    pass


class HttpInferenceGateway:
    def __init__(self, base_url: str, timeout_seconds: float = 25.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def _post_infer(self, payload: dict[str, Any]) -> InferenceOutput:
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

        try:
            return InferenceOutput.model_validate(response_payload)
        except ValidationError as exc:
            first = exc.errors()[0]
            loc = ".".join(str(part) for part in first.get("loc", ()))
            msg = first.get("msg", "Invalid inference payload")
            where = f" at '{loc}'" if loc else ""
            raise InferenceGatewayError(f"Inference service payload does not match DTO schema{where}: {msg}") from exc

    def infer_many(self, image_urls: list[str]) -> InferenceOutput:
        return self._post_infer({"image_urls": image_urls})

    def submit(self, image_urls: list[str]) -> tuple[str, InferenceOutput]:
        result = self.infer_many(image_urls)
        return str(uuid.uuid4()), result


def get_inference_gateway() -> HttpInferenceGateway:
    service_url = os.getenv("INFERENCE_SERVICE_URL", "http://127.0.0.1:8001")
    return HttpInferenceGateway(service_url)
