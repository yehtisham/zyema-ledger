"""
database.py

SQLAlchemy database connection.

Local development : SQLite  (zero setup, file-based)
Production        : PostgreSQL  (set DATABASE_URL env var)

Switching is one environment variable:
  DATABASE_URL=postgresql://user:pass@host/dbname

Everything else in the codebase stays the same.
"""

import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase

# ── Connection URL ────────────────────────────────────────────────────
# Defaults to SQLite stored at data/ledger.db for local development.
# Override with DATABASE_URL env var for production PostgreSQL.
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "sqlite:////data/ledger.db"
)

# SQLite needs check_same_thread=False for FastAPI's multi-threaded env
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)

# ── Base class for all ORM models ─────────────────────────────────────
class Base(DeclarativeBase):
    pass

# ── FastAPI dependency: yields a DB session per request ──────────────
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
