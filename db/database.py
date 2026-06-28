"""
SQLAlchemy database initialisation and session factory.
The DATABASE_URL environment variable must be set before importing this module,
or a URL may be passed directly to `init_db()`.
Example DATABASE_URL:
    postgresql+psycopg2://user:password@localhost:5432/jokes
"""
import os
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker
from db.models import Base

# ---------------------------------------------------------------------------
# Engine / session factory – populated by init_db()
# ---------------------------------------------------------------------------
engine = None
SessionLocal: sessionmaker[Session] | None = None


def init_db(database_url: str | None = None) -> None:
    """
    Initialise the SQLAlchemy engine and session factory, then create all
    tables (including the pgvector extension) if they do not already exist.
    :param database_url: A SQLAlchemy-compatible connection URL.  Falls back
        to the DATABASE_URL environment variable when omitted.
    """
    global engine, SessionLocal
    url = database_url or os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError(
            "No database URL provided.  "
            "Pass one to init_db() or set the DATABASE_URL environment variable."
        )
    engine = create_engine(url, echo=False, future=True)
    # Ensure the pgvector extension exists before creating tables.
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


def get_session() -> Session:
    """
    Return a new SQLAlchemy Session.
    The caller is responsible for committing / rolling back and closing the
    session.  Prefer using it as a context manager::
        with get_session() as session:
            ...
    """
    if SessionLocal is None:
        raise RuntimeError("Database has not been initialised.  Call init_db() first.")
    return SessionLocal()
