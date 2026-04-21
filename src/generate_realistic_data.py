"""
generate_realistic_data.py

Generates a realistic synthetic cashbook for a drums / IBC-tanks
sole-trader business (PKR).  Categories match the chart of accounts
so the trained ML model predicts correctly on real data.

Fixes vs old version
--------------------
- Categories updated to match drums trading business
- Running balance is cumulative (not per-row reset)
- 15 % text-noise injection kept for robustness
"""

import pandas as pd
import random
from datetime import datetime, timedelta

# ── Category → description templates ─────────────────────────────────
CATEGORIES = {
    # Revenue
    "Sales - Steel Drums": [
        "Sold steel drums", "Steel drum sale", "Steel drums dispatched",
        "Customer pickup steel drums", "Invoice steel drums",
    ],
    "Sales - Plastic Drums": [
        "Sold plastic drums", "Plastic drum sale", "Plastic drums delivered",
        "Invoice plastic drums", "Customer payment plastic drums",
    ],
    "Sales - IBC Tanks": [
        "Sold IBC tanks", "IBC tank sale", "IBC small tank invoice",
        "Customer IBC payment", "IBC tanks dispatched",
    ],
    "Sales - Refurbished Drums": [
        "Refurbished drum sale", "Sold reconditioned drums",
        "Refurbished steel drums invoice", "Reconditioned barrel sale",
    ],
    # COGS
    "Purchases - Steel Drums": [
        "Purchase steel drums", "Steel drum procurement",
        "Bought steel drums HTG Petro", "Steel drums received",
    ],
    "Purchases - Plastic Drums": [
        "Purchase plastic drums", "Plastic drum procurement",
        "Bought plastic drums", "Plastic drums received",
    ],
    "Purchases - IBC Tanks": [
        "Purchase IBC small tanks", "IBC tank procurement",
        "Bought IBC tanks HTG Petro", "IBC tanks received bill",
    ],
    "Refurbishing Costs": [
        "Drum refurbishing labour", "Sandblasting cost",
        "Paint and coating drums", "Drum repair materials",
        "Reconditioning expense",
    ],
    # Expenses
    "Freight & Conveyance": [
        "Drums conveyance", "Freight charges", "Transport expense",
        "Truck hire drums", "Delivery charges",
    ],
    "Rent": [
        "Monthly warehouse rent", "Godown rent", "Storage rent paid",
        "Office rent", "Shop rent",
    ],
    "Utilities": [
        "Electricity bill", "Gas charges", "Water bill",
        "Internet bill", "Phone bill",
    ],
    "Office Supplies": [
        "Stationery purchase", "Printer ink", "Paper stock",
        "Office furniture", "Notebook order",
    ],
    "Bank Charges": [
        "Bank service charges", "Cheque book charges",
        "Online transfer fee", "Bank commission",
    ],
    "Miscellaneous": [
        "Miscellaneous expense", "Petty cash expense",
        "Sundry charges", "General expense",
    ],
}

COUNTERPARTIES = [
    "HTG Petro", "Pakistan Drums Co", "Al-Baraka Bank", "Meezan Bank",
    "Habib Bank", "Bank Al Falah", "TCS Courier", "M Ehtisham",
    "Local Supplier", "Walk-in Customer", "Regular Client",
    "Metro Traders", "Punjab Chemicals", "City Freight",
]

ACCOUNTS = ["Cash", "Bank"]

# ── Noise injection ───────────────────────────────────────────────────
def _add_noise(text: str) -> str:
    if random.random() < 0.15:
        pos = random.randint(0, max(len(text) - 1, 0))
        text = text[:pos] + random.choice("abcdefghijklmnopqrstuvwxyz") + text[pos:]
    if random.random() < 0.10:
        text = text.lower()
    return text


# ── Amount ranges per category ────────────────────────────────────────
AMOUNT_RANGES = {
    "Sales - Steel Drums":        (50_000,  900_000),
    "Sales - Plastic Drums":      (30_000,  200_000),
    "Sales - IBC Tanks":          (100_000, 800_000),
    "Sales - Refurbished Drums":  (20_000,  150_000),
    "Purchases - Steel Drums":    (40_000,  850_000),
    "Purchases - Plastic Drums":  (30_000,  200_000),
    "Purchases - IBC Tanks":      (80_000,  700_000),
    "Refurbishing Costs":         (5_000,   80_000),
    "Freight & Conveyance":       (5_000,   60_000),
    "Rent":                       (20_000,  80_000),
    "Utilities":                  (2_000,   20_000),
    "Office Supplies":            (1_000,   15_000),
    "Bank Charges":               (500,     5_000),
    "Miscellaneous":              (500,     10_000),
}

REVENUE_CATEGORIES = {c for c in CATEGORIES if c.startswith("Sales")}


def generate_data(n: int = 5000, starting_balance: float = 500_000) -> pd.DataFrame:
    start_date = datetime(2024, 1, 1)
    rows = []
    balance = starting_balance

    for _ in range(n):
        category    = random.choice(list(CATEGORIES.keys()))
        description = _add_noise(random.choice(CATEGORIES[category]))
        counterparty = _add_noise(random.choice(COUNTERPARTIES))
        account     = random.choice(ACCOUNTS)
        date        = start_date + timedelta(days=random.randint(0, 365))

        lo, hi = AMOUNT_RANGES[category]
        amount = random.randint(lo, hi)

        if category in REVENUE_CATEGORIES:
            debit, credit = 0, amount
        else:
            debit, credit = amount, 0

        balance += credit - debit

        rows.append([
            date.strftime("%Y-%m-%d"), description, counterparty,
            debit, credit, round(balance, 2), account, category,
        ])

    df = pd.DataFrame(rows, columns=[
        "date", "description", "counterparty",
        "debit", "credit", "balance", "account", "category",
    ])
    return df.sort_values("date").reset_index(drop=True)


if __name__ == "__main__":
    df = generate_data(5000)
    df.to_csv("data/cashbook_realistic.csv", index=False)
    print(f"Generated realistic dataset: {len(df)} rows → data/cashbook_realistic.csv")
    print(df.head(10).to_string(index=False))
