"""
migrate.py

One-time migration: loads full_cashbook.csv into the SQLite database.

Run once:
    python src/migrate.py

Safe to re-run — clears and reloads the transactions table each time,
so it also works as a "refresh from CSV" command.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
from sqlalchemy import text
from database import engine, Base
from models import Transaction

BASE = os.path.dirname(os.path.dirname(__file__))
FULL_CSV = os.path.join(BASE, "data", "full_cashbook.csv")

# Ensure /data directory exists (required for Render persistent disk)
os.makedirs("/data", exist_ok=True)


def migrate():
    # ── 1. Create tables if they don't exist ─────────────────────────
    Base.metadata.create_all(bind=engine)
    print("Tables created (or already exist)")

    # ── 2. Load CSV ───────────────────────────────────────────────────
    df = pd.read_csv(FULL_CSV)
    df["date"]        = pd.to_datetime(df["date"]).dt.date
    df["description"] = df["description"].fillna("")
    df["counterparty"]= df["counterparty"].fillna("")
    df["account"]     = df["account"].fillna("Bank")
    df["debit"]       = pd.to_numeric(df["debit"],  errors="coerce").fillna(0)
    df["credit"]      = pd.to_numeric(df["credit"], errors="coerce").fillna(0)
    df["category"]    = df["category"].fillna("Miscellaneous")

    # Detect source from counterparty / existing source column
    if "source" not in df.columns:
        df["source"] = df["counterparty"].apply(
            lambda x: "htg_petro" if str(x).strip() == "HTG Petro" else "xlsm"
        )

    # ── 3. Clear existing rows and reload ────────────────────────────
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM transactions"))

    rows = [
        Transaction(
            date=        row["date"],
            description= row["description"],
            counterparty=row["counterparty"],
            account=     row["account"],
            debit=       row["debit"],
            credit=      row["credit"],
            category=    row["category"],
            source=      row["source"],
        )
        for _, row in df.iterrows()
    ]

    with engine.begin() as conn:
        from sqlalchemy.orm import Session
        with Session(bind=engine) as session:
            session.bulk_save_objects(rows)
            session.commit()

    print(f"Migrated {len(rows)} transactions")

    # ── 4. Verify ─────────────────────────────────────────────────────
    with engine.connect() as conn:
        count = conn.execute(text("SELECT COUNT(*) FROM transactions")).scalar()
        earliest = conn.execute(text("SELECT MIN(date) FROM transactions")).scalar()
        latest   = conn.execute(text("SELECT MAX(date) FROM transactions")).scalar()

    print(f"Database: {count} rows  |  {earliest} → {latest}")


if __name__ == "__main__":
    migrate()
