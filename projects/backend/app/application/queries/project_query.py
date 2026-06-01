from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProjectQuery:
    project_id: str
