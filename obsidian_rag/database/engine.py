"""Database engine and session management for obsidian-rag."""

import logging
import re
from collections.abc import Generator
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from obsidian_rag.database.models import Base

log = logging.getLogger(__name__)


def _normalize_postgres_url(url: str) -> str:
    """Normalize PostgreSQL URL to use psycopg (v3) driver.

    Converts URLs to use the psycopg driver explicitly by:
    - Replacing 'postgres://' or 'postgresql://' with 'postgresql+psycopg://'
    - Preserving URLs that already specify a driver

    Args:
        url: The database URL to normalize.

    Returns:
        The normalized URL with psycopg driver specified.

    """
    _msg = f"Normalizing database URL: {url}"
    log.debug(_msg)

    # Skip if URL already specifies a driver (contains +)
    if re.search(r"postgresql\+", url):
        _msg = "URL already has driver specified, skipping normalization"
        log.debug(_msg)
        return url

    # Replace postgres:// or postgresql:// with postgresql+psycopg://
    normalized = re.sub(r"^(postgres(?:ql)?://)", r"postgresql+psycopg://", url)

    _msg = f"Normalized URL: {normalized}"
    log.debug(_msg)
    return normalized


class DatabaseManager:
    """Manages database connections and sessions.

    This class provides a centralized way to manage database connections,
    create sessions, and handle schema creation.

    Args:
        database_url: The PostgreSQL connection URL.

    Attributes:
        engine: The SQLAlchemy engine instance.
        SessionLocal: Session factory bound to the engine.

    """

    def __init__(self, database_url: str) -> None:
        """Initialize the database manager with a connection URL."""
        normalized_url = _normalize_postgres_url(database_url)
        _msg = f"Initializing DatabaseManager with URL: {normalized_url}"
        log.debug(_msg)
        self.engine = create_engine(normalized_url)
        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine,
        )

    def create_tables(self) -> None:
        """Create all database tables."""
        _msg = "Creating all database tables"
        log.debug(_msg)
        Base.metadata.create_all(bind=self.engine)

    def drop_tables(self) -> None:
        """Drop all database tables."""
        _msg = "Dropping all database tables"
        log.debug(_msg)
        Base.metadata.drop_all(bind=self.engine)

    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """Provide a transactional scope around a series of operations.

        Yields:
            Session: A SQLAlchemy session.

        Example:
            with db_manager.get_session() as session:
                document = session.query(Document).first()

        """
        _msg = "Creating new database session"
        log.debug(_msg)
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def close(self) -> None:
        """Close the database engine."""
        _msg = "Closing database engine"
        log.debug(_msg)
        self.engine.dispose()
