"""
SQLAlchemy ORM model definitions for the joke database.

Requires:
  - PostgreSQL 16.14 with pgvector extension installed
  - SQLAlchemy 2.0.50
  - pgvector[sqlalchemy] (python package)
"""

from __future__ import annotations

import uuid
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, ForeignKey, Text, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

EMB_DIMS = 1024


class Base(DeclarativeBase):
    pass


class Joke(Base):
    """Table storing joke texts and their LLM semantic embeddings."""

    __tablename__ = "joke"

    uuid: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        comment="Unique ID for this joke row",
    )
    content: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        comment="Text of the joke",
    )
    embed: Mapped[list[float] | None] = mapped_column(
        Vector(EMB_DIMS),
        nullable=True,
        default=None,
        comment="pgvector embedding vector for the given joke text",
    )
    added: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        comment="Date the row was added to table",
    )
    modified: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
        comment="Date the row was last modified",
    )

    media: Mapped[list["Media"]] = relationship(
        "Media", back_populates="joke", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Joke uuid={self.uuid} content={self.content[:40]!r}>"


class Media(Base):
    """Table storing paths to joke-related media such as audio files."""

    __tablename__ = "media"

    uuid: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        comment="Unique ID for a given media row",
    )
    joke_uuid: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("joke.uuid", ondelete="CASCADE"),
        nullable=False,
        comment="Foreign key linking to joke.uuid",
    )
    hash: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        default=None,
        comment="Hex text of the SHA256 hash of the media file content",
    )
    path: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        default=None,
        comment="Platform-appropriate pathname to the media file",
    )
    added: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        comment="Date the row was added to table",
    )
    modified: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
        comment="Date the row was last modified",
    )

    joke: Mapped["Joke"] = relationship("Joke", back_populates="media")

    def __repr__(self) -> str:
        return f"<Media uuid={self.uuid} joke_uuid={self.joke_uuid} path={self.path!r}>"
