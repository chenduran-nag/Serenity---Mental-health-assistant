"""Shared pytest configuration for the Serenity test suite."""

from __future__ import annotations

import os
import sys
import tempfile
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Point the backend at a throwaway database before anything imports it, so tests
# never touch the developer's serenity.db.
_TEST_DB = Path(tempfile.gettempdir()) / f"serenity_test_{uuid.uuid4().hex}.db"
os.environ["SERENITY_DATABASE_URL"] = f"sqlite:///{_TEST_DB.as_posix()}"


def pytest_sessionfinish(session, exitstatus):
    """Release the database connection pool and remove the throwaway database."""

    try:
        from backend.database import engine

        engine.dispose()
    except Exception:  # pragma: no cover - nothing to dispose if import failed
        pass

    try:
        _TEST_DB.unlink(missing_ok=True)
    except PermissionError:  # pragma: no cover - Windows may still hold the handle
        pass
