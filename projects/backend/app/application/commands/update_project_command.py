from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class UpdateProjectCommand:
    project_id: str
    target_width: float | None = None
    target_height: float | None = None
    target_depth: float | None = None
    shelf_count: int | None = None

    def as_updates(self) -> dict:
        return {
            "target_width": self.target_width,
            "target_height": self.target_height,
            "target_depth": self.target_depth,
            "shelf_count": self.shelf_count,
        }
