from __future__ import annotations

from typing import Protocol


class InferenceGateway(Protocol):
    def infer(self, image_url: str) -> dict: ...
