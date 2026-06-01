from __future__ import annotations

import datetime

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.infrastructure.persistence.database import Base


class Project(Base):
    __tablename__ = "projects"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    model_json: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow
    )


class ProjectAsset(Base):
    __tablename__ = "project_assets"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    project_id: Mapped[str] = mapped_column(
        String, ForeignKey("projects.id", ondelete="CASCADE"), nullable=False
    )
    file_name: Mapped[str] = mapped_column(String, nullable=False)
    content_type: Mapped[str] = mapped_column(String, nullable=False)
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    image_data: Mapped[str] = mapped_column(Text, nullable=False)


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    status: Mapped[str] = mapped_column(String, nullable=False)
    result_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    project_name: Mapped[str | None] = mapped_column(String, nullable=True)
    project_id: Mapped[str | None] = mapped_column(
        String, ForeignKey("projects.id", ondelete="SET NULL"), nullable=True
    )
    asset_id: Mapped[str | None] = mapped_column(
        String, ForeignKey("project_assets.id", ondelete="SET NULL"), nullable=True
    )
