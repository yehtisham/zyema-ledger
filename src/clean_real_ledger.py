"""
clean_real_ledger.py

Cleans the raw scanned supplier ledger (page-1_table-1.csv) into a
standardised cashbook CSV that the rest of the pipeline can consume.

Ledger context
--------------
This is a SUPPLIER / PURCHASE ledger tracking what the business owes
to a single drum/IBC supplier (HTG Petro and related parties).

Row types
---------
  purchase  – goods received; INCLUSIVE OF GST BILL is the amount owed
  payment   – cash paid to supplier; PAID Amount is the outflow
  freight   – conveyance / delivery charge; INCLUSIVE OF GST BILL
  return    – goods returned; GOODS RETURN column
  blank     – header / month rows, dropped
"""

import pandas as pd
import re


# ── Helpers ──────────────────────────────────────────────────────────

def _clean_number(val) -> float:
    """Convert strings like '1,234.5' or '(48,999)' to float."""
    if pd.isna(val):
        return 0.0
    s = str(val).strip()
    if s in ("", "-", "—"):
        return 0.0
    negative = s.startswith("(") and s.endswith(")")
    s = re.sub(r"[,()\"]", "", s)
    try:
        result = float(s)
        return -result if negative else result
    except ValueError:
        return 0.0


def _detect_row_type(detail) -> str:
    """Classify a row from its DETAIL text."""
    if pd.isna(detail) or str(detail).strip() == "":
        return "blank"
    d = str(detail).strip().lower()
    if "purchase" in d:
        if "steel" in d:
            return "purchase_steel"
        elif "plastic" in d:
            return "purchase_plastic"
        elif "ibc" in d:
            return "purchase_ibc"
        else:
            return "purchase_other"
    if any(kw in d for kw in ["paid", "dep cash", "dep ", "deposit"]):
        return "payment"
    if any(kw in d for kw in ["convaynce", "conveyance", "freight", "transport"]):
        return "freight"
    if "return" in d:
        return "return"
    return "other"


# Row-type → ledger category (must match chart_of_accounts.py)
_CATEGORY_MAP = {
    "purchase_steel":   "Purchases - Steel Drums",
    "purchase_plastic": "Purchases - Plastic Drums",
    "purchase_ibc":     "Purchases - IBC Tanks",
    "purchase_other":   "Purchases - Steel Drums",   # safe fallback
    "payment":          "Accounts Payable Payment",
    "freight":          "Freight & Conveyance",
    "return":           "Goods Return",
    "other":            "Miscellaneous",
}


# ── Main function ─────────────────────────────────────────────────────

def clean_ledger(
    raw_path: str = "real_ledger/page-1_table-1.csv",
    output_path: str = "data/cleaned_cashbook.csv",
) -> pd.DataFrame:

    df = pd.read_csv(raw_path)

    # ── Standardise column names ──────────────────────────────────────
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(r"[\s/\'\"]", "_", regex=True)
        .str.replace(r"_+", "_", regex=True)
        .str.strip("_")
    )

    # ── Numeric columns ───────────────────────────────────────────────
    num_cols = [
        "exclusive_of_gst_rate", "exclusive_of_gst_bill",
        "gst_18%", "inclusive_of_gst_bill",
        "paid_amount", "goods_return", "balance",
        "qty_steel_drums", "qty_plastic_drums",
        "qty_ibc_s_tanks", "qty_steel_drums_return",
    ]
    for col in num_cols:
        if col in df.columns:
            df[col] = df[col].apply(_clean_number)

    # ── Date column ───────────────────────────────────────────────────
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    df = df[df["date"].notna()].copy()   # drop month-header / blank rows

    # ── Row classification ────────────────────────────────────────────
    df["row_type"] = df["detail"].apply(_detect_row_type)
    df = df[df["row_type"] != "blank"].copy()

    # ── Debit / Credit mapping ────────────────────────────────────────
    # debit  = cost / liability incurred (purchase or freight)
    # credit = cash actually paid out to supplier
    def _debit(row) -> float:
        if row["row_type"] in (
            "purchase_steel", "purchase_plastic",
            "purchase_ibc", "purchase_other", "freight",
        ):
            return row.get("inclusive_of_gst_bill", 0.0) or 0.0
        return 0.0

    def _credit(row) -> float:
        if row["row_type"] == "payment":
            return row.get("paid_amount", 0.0) or 0.0
        return 0.0

    df["debit"]  = df.apply(_debit,  axis=1)
    df["credit"] = df.apply(_credit, axis=1)

    # ── Standard cashbook columns ─────────────────────────────────────
    df["description"]      = df["detail"].fillna("")
    df["counterparty"]     = "HTG Petro"
    df["account"]          = "Bank"
    df["supplier_balance"] = df["balance"].fillna(0.0)
    df["category"]         = df["row_type"].map(_CATEGORY_MAP)

    # Quantity columns (rename for clarity)
    qty_map = {
        "qty_steel_drums":        "qty_steel_drums",
        "qty_plastic_drums":      "qty_plastic_drums",
        "qty_ibc_s_tanks":        "qty_ibc_tanks",
        "qty_steel_drums_return": "qty_steel_drums_return",
    }
    for src, dst in qty_map.items():
        df[dst] = df[src].fillna(0.0) if src in df.columns else 0.0

    # ── Final selection & filter ──────────────────────────────────────
    out_cols = [
        "date", "description", "counterparty", "account",
        "debit", "credit", "supplier_balance",
        "qty_steel_drums", "qty_plastic_drums",
        "qty_ibc_tanks", "qty_steel_drums_return",
        "category",
    ]
    out = df[out_cols].copy()
    out = out[out["category"].notna()]
    out = out[(out["debit"] != 0) | (out["credit"] != 0)]  # drop zero rows

    out.to_csv(output_path, index=False)
    print(f"Cleaned ledger saved → {output_path}  ({len(out)} rows)")
    print(out.to_string(index=False))
    return out


if __name__ == "__main__":
    clean_ledger()
