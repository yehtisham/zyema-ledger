"""
models.py

Database table definitions (SQLAlchemy ORM).

Tables
------
  Transaction  — every ledger entry (purchase, sale, payment, expense)
"""

from datetime import date, datetime
from sqlalchemy import (
    Column, Integer, String, Numeric,
    Date, DateTime, Text, Index
)
from sqlalchemy.sql import func
from database import Base


class Transaction(Base):
    __tablename__ = "transactions"

    id           = Column(Integer, primary_key=True, index=True)

    # ── Core fields ───────────────────────────────────────────────────
    date         = Column(Date, nullable=False, index=True)
    description  = Column(Text, default="")
    counterparty = Column(String(255), default="", index=True)
    account      = Column(String(100), default="Bank")

    # ── Amounts (PKR, 2 decimal places) ───────────────────────────────
    debit        = Column(Numeric(15, 2), default=0)
    credit       = Column(Numeric(15, 2), default=0)

    # ── Classification ────────────────────────────────────────────────
    category     = Column(String(100), index=True)

    # ── Audit ─────────────────────────────────────────────────────────
    # source: where this row came from
    #   'htg_petro' — cleaned from page-1_table-1.csv
    #   'xlsm'      — ingested from Zyema_Ledger.xlsm
    #   'manual'    — added via POST /transactions in the app
    #   'ocr'       — extracted from a photo (future)
    source       = Column(String(50), default="manual")
    created_at   = Column(DateTime, server_default=func.now())
    updated_at   = Column(DateTime, server_default=func.now(), onupdate=func.now())

    # ── Composite indexes for common query patterns ───────────────────
    __table_args__ = (
        Index("ix_txn_date_category", "date", "category"),
        Index("ix_txn_counterparty_date", "counterparty", "date"),
    )

    def to_dict(self) -> dict:
        return {
            "id":           self.id,
            "date":         str(self.date),
            "description":  self.description,
            "counterparty": self.counterparty,
            "account":      self.account,
            "debit":        float(self.debit or 0),
            "credit":       float(self.credit or 0),
            "category":     self.category,
            "source":       self.source,
            "created_at":   str(self.created_at),
        }
