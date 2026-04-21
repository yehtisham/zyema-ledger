"""
ingest_xlsm.py

Reads Zyema_Ledger.xlsm and converts all vendor / customer transactions
into the standard cashbook schema used by financial_statements.py.

Output
------
  data/xlsm_cashbook.csv   — transactions from the xlsm only
  data/full_cashbook.csv   — xlsm + HTG Petro purchase history merged
"""

import pandas as pd
import warnings
warnings.filterwarnings("ignore")

XLSM_PATH = "/Users/yahya/Desktop/Zyema_Ledger.xlsm"

# ── Product → chart-of-accounts category ─────────────────────────────
def _map_product(product_id: str, txn_type: str) -> str:
    p = str(product_id).lower().strip()
    prefix = "Sales" if txn_type in ("Sold",) else "Purchases"

    if "ibc" in p:
        return f"{prefix} - IBC Tanks"
    if "plastic drum" in p:
        return f"{prefix} - Plastic Drums"
    if "steel drum" in p or "steel" in p:
        return f"{prefix} - Steel Drums"
    return f"{prefix} - Steel Drums"   # safe fallback


# ── Sheet reader ──────────────────────────────────────────────────────
def _read_sheet(xl, sheet_name: str, sheet_type: str) -> pd.DataFrame | None:
    df = xl.parse(sheet_name, header=None)
    account_name = str(df.iloc[0, 0]).strip()
    account_id   = str(df.iloc[1, 0]).strip()

    # Locate header row (contains both "Date" and "TxnID")
    header_row = None
    for i, row in df.iterrows():
        vals = str(row.values)
        if "Date" in vals and "TxnID" in vals:
            header_row = i
            break
    if header_row is None:
        return None

    data = df.iloc[header_row + 1:].copy()

    # Deduplicate column names
    seen: dict[str, int] = {}
    clean = []
    for h in df.iloc[header_row].tolist():
        h = str(h).strip() if str(h) != "nan" else f"_col{len(clean)}"
        count = seen.get(h, 0)
        clean.append(h if count == 0 else f"{h}_{count}")
        seen[h] = count + 1
    data.columns = clean

    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    data = data[data["Date"].notna()].copy()
    if len(data) == 0:
        return None

    # Normalise amount columns across vendor / customer schema differences
    for candidate in ["Total after tax", "Total After tax", "Total"]:
        if candidate in data.columns:
            data["_amount_total"] = pd.to_numeric(data[candidate], errors="coerce")
            break
    else:
        data["_amount_total"] = 0.0

    data["_amount_paid"] = pd.to_numeric(
        data.get("Amount Paid", pd.Series(dtype=float)), errors="coerce"
    ).fillna(0)
    data["_qty"] = pd.to_numeric(
        data.get("Qty (+/-)", pd.Series(dtype=float)), errors="coerce"
    ).fillna(0)
    data["_product"] = data.get("ProductID", pd.Series(dtype=str)).fillna("")

    data["account_name"] = account_name
    data["account_id"]   = account_id
    data["sheet_type"]   = sheet_type
    return data


# ── Convert to cashbook schema ────────────────────────────────────────
def _to_cashbook_rows(df: pd.DataFrame) -> list[dict]:
    """
    Cashbook convention used throughout this project:
      debit  = value of liability / cost INCURRED (purchases, expenses)
      credit = value of revenue EARNED or CASH payment made/received

    For income-statement purposes:
      Revenue rows  → credit = amount_total, category = Sales - *
      COGS rows     → debit  = amount_total, category = Purchases - *

    For cash-flow purposes:
      Vendor payment made      → credit = amount_paid, cat = Accounts Payable Payment
      Customer payment received → debit = amount_paid, cat = Customer Payment Received
    """
    rows = []
    for _, r in df.iterrows():
        txn_type = str(r.get("Type", "")).strip()
        if txn_type not in ("Sold", "Purchase", "Payment"):
            continue

        base = {
            "date":         r["Date"].date(),
            "counterparty": r["account_name"],
            "account":      "Bank",
            "qty":          r["_qty"],
            "product":      str(r["_product"]).strip(),
        }

        if txn_type == "Sold":
            rows.append({**base,
                "description": f"Sold {r['_product']} to {r['account_name']}",
                "debit":       0,
                "credit":      r["_amount_total"],
                "category":    _map_product(r["_product"], "Sold"),
            })

        elif txn_type == "Purchase":
            rows.append({**base,
                "description": f"Purchase {r['_product']} from {r['account_name']}",
                "debit":       r["_amount_total"],
                "credit":      0,
                "category":    _map_product(r["_product"], "Purchase"),
            })

        elif txn_type == "Payment":
            if r["sheet_type"] == "vendor":
                rows.append({**base,
                    "description": f"Payment to {r['account_name']}",
                    "debit":       0,
                    "credit":      r["_amount_paid"],
                    "category":    "Accounts Payable Payment",
                })
            else:  # customer payment received
                rows.append({**base,
                    "description": f"Payment received from {r['account_name']}",
                    "debit":       r["_amount_paid"],
                    "credit":      0,
                    "category":    "Customer Payment Received",
                })
    return rows


# ── Main ingestion function ───────────────────────────────────────────
def ingest(
    xlsm_path: str = XLSM_PATH,
    output_xlsm:  str = "data/xlsm_cashbook.csv",
    output_full:  str = "data/full_cashbook.csv",
    htg_path:     str = "data/cleaned_cashbook.csv",
) -> pd.DataFrame:

    xl = pd.ExcelFile(xlsm_path, engine="openpyxl")

    all_rows = []
    for sheet in xl.sheet_names:
        if sheet.startswith("V") and sheet[1:].isdigit():
            df = _read_sheet(xl, sheet, "vendor")
            if df is not None:
                all_rows.extend(_to_cashbook_rows(df))
        elif sheet.startswith("C") and sheet[1:].isdigit():
            df = _read_sheet(xl, sheet, "customer")
            if df is not None:
                all_rows.extend(_to_cashbook_rows(df))

    xlsm_df = pd.DataFrame(all_rows)
    xlsm_df["date"] = pd.to_datetime(xlsm_df["date"])

    # Drop obvious date-entry errors (year < 2020)
    xlsm_df = xlsm_df[xlsm_df["date"].dt.year >= 2020]

    # Drop zero-amount rows
    xlsm_df = xlsm_df[(xlsm_df["debit"] != 0) | (xlsm_df["credit"] != 0)]

    xlsm_df = xlsm_df.sort_values("date").reset_index(drop=True)
    xlsm_df.to_csv(output_xlsm, index=False)
    print(f"xlsm cashbook saved → {output_xlsm}  ({len(xlsm_df)} rows)")

    # Merge with HTG Petro purchase history
    htg_df = pd.read_csv(htg_path)
    htg_df["date"] = pd.to_datetime(htg_df["date"])

    # Align columns
    shared_cols = ["date", "description", "counterparty", "account",
                   "debit", "credit", "category"]
    full_df = pd.concat(
        [htg_df[shared_cols], xlsm_df[shared_cols]],
        ignore_index=True
    ).sort_values("date").reset_index(drop=True)

    full_df.to_csv(output_full, index=False)
    print(f"Full cashbook saved  → {output_full}  ({len(full_df)} rows)")
    return full_df


# ── Summary ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = ingest()

    print(f"\n{'='*60}")
    print("  FULL CASHBOOK SUMMARY")
    print(f"{'='*60}")
    print(f"  Date range : {df['date'].min().date()} → {df['date'].max().date()}")
    print(f"  Total rows : {len(df)}")

    print(f"\n  By category:")
    cat_summary = df.groupby("category").agg(
        rows=("date", "count"),
        debit=("debit", "sum"),
        credit=("credit", "sum"),
    )
    for cat, row in cat_summary.iterrows():
        print(f"    {cat:<35}  rows={row['rows']:>3}  "
              f"debit={row['debit']:>14,.0f}  credit={row['credit']:>14,.0f}")

    sales      = df[df["category"].str.startswith("Sales",  na=False)]["credit"].sum()
    purchases  = df[df["category"].str.startswith("Purch",  na=False)]["debit"].sum()
    gross      = sales - purchases
    print(f"\n  Total Sales     : PKR {sales:>14,.0f}")
    print(f"  Total Purchases : PKR {purchases:>14,.0f}")
    print(f"  Gross Profit    : PKR {gross:>14,.0f}")
    print(f"  Gross Margin    : {gross/sales*100:.1f}%" if sales else "  Gross Margin    : N/A")
    print(f"{'='*60}\n")
