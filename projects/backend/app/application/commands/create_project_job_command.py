from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CreateProjectJobCommand:
    project_id: str
    asset_id: str
