from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CreateProjectCommand:
    name: str | None
    job_id: str | None
