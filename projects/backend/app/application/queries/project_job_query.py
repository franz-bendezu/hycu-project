from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProjectJobQuery:
    project_id: str
    job_id: str
