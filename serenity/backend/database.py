"""Database models and session helpers for the Serenity backend."""

from __future__ import annotations

import os
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator, Iterable

from sqlalchemy import DateTime, Integer, String, Text, create_engine, delete, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker


def _default_database_url() -> str:
    """Return the default SQLite database URL for local development."""

    database_path = Path(__file__).resolve().parent.parent / "serenity.db"
    return f"sqlite:///{database_path.as_posix()}"


DATABASE_URL = os.getenv("SERENITY_DATABASE_URL", _default_database_url())


class Base(DeclarativeBase):
    """Base declarative class for SQLAlchemy models."""


class SessionMessage(Base):
    """Persisted chat message row."""

    __tablename__ = "sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    role: Mapped[str] = mapped_column(String(32), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )


connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args, future=True)
# expire_on_commit=False keeps attributes loaded after the session closes, so rows
# returned by fetch_history stay readable once they are detached.
SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
    expire_on_commit=False,
    future=True,
)


def init_db() -> None:
    """Create database tables if they do not already exist."""

    Base.metadata.create_all(bind=engine)


@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """Yield a managed SQLAlchemy session."""

    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def save_messages(session_id: str, messages: Iterable[tuple[str, str]]) -> None:
    """Persist one or more chat messages for a session."""

    rows = [
        SessionMessage(
            session_id=session_id,
            role=role,
            content=content,
            timestamp=datetime.now(timezone.utc),
        )
        for role, content in messages
    ]
    with get_db_session() as session:
        session.add_all(rows)


def fetch_history(session_id: str) -> list[SessionMessage]:
    """Fetch full conversation history for a session."""

    with get_db_session() as session:
        stmt = (
            select(SessionMessage)
            .where(SessionMessage.session_id == session_id)
            .order_by(SessionMessage.timestamp.asc(), SessionMessage.id.asc())
        )
        return list(session.scalars(stmt).all())


def delete_history(session_id: str) -> int:
    """Delete all rows associated with a session and return the deleted count."""

    with get_db_session() as session:
        stmt = delete(SessionMessage).where(SessionMessage.session_id == session_id)
        result = session.execute(stmt)
        return int(result.rowcount or 0)
