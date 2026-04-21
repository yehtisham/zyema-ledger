"""
chart_of_accounts.py

Chart of Accounts for a drums / IBC-tanks sole-trader business (PKR).

Account types
-------------
  Asset     – things the business owns (debit-normal)
  Liability – things the business owes (credit-normal)
  Equity    – owner's stake (credit-normal)
  Revenue   – income earned (credit-normal)
  COGS      – cost of goods sold (debit-normal)
  Expense   – operating expenses (debit-normal)

Each account has:
  code   – 4-digit reference number
  type   – one of the six types above
  normal – which side increases the account (debit / credit)
"""

ACCOUNTS: dict[str, dict] = {
    # ── Assets (1xxx) ─────────────────────────────────────────────────
    "Cash":                       {"code": "1001", "type": "Asset",     "normal": "debit"},
    "Bank":                       {"code": "1002", "type": "Asset",     "normal": "debit"},
    "Accounts Receivable":        {"code": "1003", "type": "Asset",     "normal": "debit"},
    "Inventory - Steel Drums":    {"code": "1004", "type": "Asset",     "normal": "debit"},
    "Inventory - Plastic Drums":  {"code": "1005", "type": "Asset",     "normal": "debit"},
    "Inventory - IBC Tanks":      {"code": "1006", "type": "Asset",     "normal": "debit"},
    "Prepaid Expenses":           {"code": "1007", "type": "Asset",     "normal": "debit"},

    # ── Liabilities (2xxx) ────────────────────────────────────────────
    "Accounts Payable":           {"code": "2001", "type": "Liability", "normal": "credit"},
    "GST Payable":                {"code": "2002", "type": "Liability", "normal": "credit"},

    # ── Equity (3xxx) ─────────────────────────────────────────────────
    "Owner's Capital":            {"code": "3001", "type": "Equity",    "normal": "credit"},
    "Owner's Drawings":           {"code": "3002", "type": "Equity",    "normal": "debit"},

    # ── Revenue (4xxx) ────────────────────────────────────────────────
    "Sales - Steel Drums":        {"code": "4001", "type": "Revenue",   "normal": "credit"},
    "Sales - Plastic Drums":      {"code": "4002", "type": "Revenue",   "normal": "credit"},
    "Sales - IBC Tanks":          {"code": "4003", "type": "Revenue",   "normal": "credit"},
    "Sales - Refurbished Drums":  {"code": "4004", "type": "Revenue",   "normal": "credit"},

    # ── Cost of Goods Sold (5xxx) ─────────────────────────────────────
    "Purchases - Steel Drums":    {"code": "5001", "type": "COGS",      "normal": "debit"},
    "Purchases - Plastic Drums":  {"code": "5002", "type": "COGS",      "normal": "debit"},
    "Purchases - IBC Tanks":      {"code": "5003", "type": "COGS",      "normal": "debit"},
    "Refurbishing Costs":         {"code": "5004", "type": "COGS",      "normal": "debit"},

    # ── Expenses (6xxx) ───────────────────────────────────────────────
    "Freight & Conveyance":       {"code": "6001", "type": "Expense",   "normal": "debit"},
    "Rent":                       {"code": "6002", "type": "Expense",   "normal": "debit"},
    "Utilities":                  {"code": "6003", "type": "Expense",   "normal": "debit"},
    "Office Supplies":            {"code": "6004", "type": "Expense",   "normal": "debit"},
    "Bank Charges":               {"code": "6005", "type": "Expense",   "normal": "debit"},
    "Miscellaneous":              {"code": "6006", "type": "Expense",   "normal": "debit"},

    # ── Pass-through (not on P&L or Balance Sheet) ────────────────────
    "Accounts Payable Payment":   {"code": "9001", "type": "Payable",   "normal": "credit"},
    "Goods Return":               {"code": "9002", "type": "Contra",    "normal": "debit"},
}

# ── Lookup helpers ────────────────────────────────────────────────────

def get_account_type(name: str) -> str:
    """Return the type string for an account name, or 'Unknown'."""
    return ACCOUNTS.get(name, {}).get("type", "Unknown")


def get_accounts_by_type(account_type: str) -> list[str]:
    """Return all account names that belong to a given type."""
    return [n for n, info in ACCOUNTS.items() if info["type"] == account_type]


def get_code(name: str) -> str:
    return ACCOUNTS.get(name, {}).get("code", "????")


# ── Grouped views used by financial_statements.py ────────────────────

INCOME_STATEMENT_GROUPS = {
    "revenue":  get_accounts_by_type("Revenue"),
    "cogs":     get_accounts_by_type("COGS"),
    "expense":  get_accounts_by_type("Expense"),
}

BALANCE_SHEET_GROUPS = {
    "asset":     get_accounts_by_type("Asset"),
    "liability": get_accounts_by_type("Liability"),
    "equity":    get_accounts_by_type("Equity"),
}


if __name__ == "__main__":
    print(f"{'CODE':<6} {'ACCOUNT':<35} {'TYPE':<12} {'NORMAL'}")
    print("-" * 65)
    for name, info in ACCOUNTS.items():
        print(f"{info['code']:<6} {name:<35} {info['type']:<12} {info['normal']}")
