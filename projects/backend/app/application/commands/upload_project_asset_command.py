from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class UploadProjectAssetCommand:
    project_id: str
    file_name: str
    content_type: str | None
    payload: bytes
