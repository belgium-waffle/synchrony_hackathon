"""PostgreSQL + pgvector connectivity.

Connection details come from DATABASE_URL only. If the database is unreachable the
service logs it and continues in degraded mode with the in-memory policy store, so
a demo never dies on a missing container. `/health` reports which store is live.
"""
from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from .config import get_settings

log = logging.getLogger(__name__)

_engine: Optional[Engine] = None
_SessionLocal: Optional[sessionmaker] = None


def init_engine() -> Optional[Engine]:
    """Create the pooled engine and verify pgvector is installed. None if unavailable."""
    global _engine, _SessionLocal
    settings = get_settings()
    if not settings.DATABASE_URL:
        log.warning("DATABASE_URL not set — falling back to the in-memory policy store.")
        return None

    try:
        _engine = create_engine(
            settings.DATABASE_URL,
            pool_size=settings.DB_POOL_SIZE,
            pool_pre_ping=True,                       # survives idle-connection drops
            connect_args={"connect_timeout": settings.DB_CONNECT_TIMEOUT_S},
            future=True,
        )
        with _engine.connect() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            conn.commit()
            ext = conn.execute(
                text("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
            ).scalar()
        log.info("PostgreSQL connected; pgvector %s", ext)
        _SessionLocal = sessionmaker(bind=_engine, expire_on_commit=False, future=True)
        return _engine
    except SQLAlchemyError as exc:
        log.error("PostgreSQL unavailable (%s). Continuing in degraded mode.", exc.__class__.__name__)
        _engine, _SessionLocal = None, None
        return None


def get_engine() -> Optional[Engine]:
    return _engine


@contextmanager
def session_scope() -> Iterator[Session]:
    """Transactional session. Commits on success, rolls back on any exception."""
    if _SessionLocal is None:
        raise RuntimeError("Database is not configured.")
    session = _SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def dispose_engine() -> None:
    global _engine, _SessionLocal
    if _engine is not None:
        _engine.dispose()
    _engine, _SessionLocal = None, None


# ---------------------------------------------------------------------------
# Compliance audit trail (decision_audit_log — see sql/001_init.sql)
# ---------------------------------------------------------------------------
_AUDIT_INSERT_SQL = text("""
    INSERT INTO decision_audit_log (
        request_id, applicant_id, model_version, verifier_version,
        risk_score, probability_of_default, approval_status, recommended_limit,
        shap_attributions, retrieved_policy_ids, llm_narrative,
        verification_passed, verification_failures, recourse, latency_ms
    ) VALUES (
        :request_id, :applicant_id, :model_version, :verifier_version,
        :risk_score, :probability_of_default, :approval_status, :recommended_limit,
        CAST(:shap_attributions AS JSONB), :retrieved_policy_ids, CAST(:llm_narrative AS JSONB),
        :verification_passed, CAST(:verification_failures AS JSONB),
        CAST(:recourse AS JSONB), CAST(:latency_ms AS JSONB)
    )
""")


def write_audit_log(record: Dict[str, Any]) -> None:
    """Persist one decision — or one blocked attempt — to the compliance audit trail.

    `record["retrieved_policy_ids"]` must be a plain Python list of strings; psycopg2
    adapts a Python list to a PostgreSQL TEXT[] automatically for the target column.

    Raises `RuntimeError` if no database is configured at all (degraded / laptop-demo
    mode) — the caller is expected to catch that specific case and treat it as
    "logging unavailable", distinct from "logging failed", since only the latter
    should ever cause an already-computed decision to be withheld.

    Raises the underlying `SQLAlchemyError` if PostgreSQL *is* configured but the
    write fails (connection drop, constraint violation, etc.), so a caller for whom
    a durable audit record is mandatory can fail closed rather than release an
    unaudited decision.
    """
    if get_engine() is None:
        raise RuntimeError("No database configured; audit log write skipped.")
    with session_scope() as session:
        session.execute(_AUDIT_INSERT_SQL, record)
