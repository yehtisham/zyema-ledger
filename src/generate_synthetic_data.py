"""
generate_synthetic_data.py

Generates a clean (no noise) synthetic cashbook for a drums / IBC-tanks
sole-trader business (PKR).  Useful as a quick sanity-check dataset.

Fixes vs old version
--------------------
- Categories updated to match drums trading business
- Running balance is cumulative (not per-row reset)
"""

import pandas as pd
import random
from datetime import datetime, timedelta

CATEGORIES = {
    "Sales - Steel Drums":        ["Sold steel drums", "Steel drum sale", "Steel drums invoice"],
    "Sales - Plastic Drums":      ["Sold plastic drums", "Plastic drum sale", "Plastic drums invoice"],
    "Sales - IBC Tanks":          ["Sold IBC tanks", "IBC tank sale", "IBC tanks invoice"],
    "Sales - Refurbished Drums":  ["Refurbished drum sale", "Reconditioned drums sold"],
    "Purchases - Steel Drums":    ["Purchase steel drums", "Steel drums received", "Steel drum bill"],
    "Purchases - Plastic Drums":  ["Purchase plastic drums", "Plastic drums received"],
    "Purchases - IBC Tanks":      ["Purchase IBC tanks", "IBC tanks received", "IBC bill"],
    "Refurbishing Costs":         ["Drum refurbishing", "Reconditioning cost", "Repair materials"],
    "Freight & Conveyance":       ["Freight charges", "Drums conveyance", "Transport expense"],
    "Rent":                       ["Warehouse rent", "Godown rent", "Office rent"],
    "Utilities":                  ["Electricity bill", "Gas charges", "Internet bill"],
    "Office Supplies":            ["Stationery purchase", "Printer ink", "Paper stock"],
    "Bank Charges":               ["Bank charges", "Transfer fee", "Cheque book fee"],
    "Miscellaneous":              ["Miscellaneous expense", "Petty cash", "Sundry charges"],
}

COUNTERPARTIES = [
    "HTG Petro", "Pakistan Drums Co", "Al-Baraka Bank",
    "Meezan Bank", "Walk-in Customer", "Regular Client",
    "Metro Traders", "City Freight", "Local Supplier",
]

ACCOUNTS = ["Cash", "Bank"]

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
        category     = random.choice(list(CATEGORIES.keys()))
        description  = random.choice(CATEGORIES[category])
        counterparty = random.choice(COUNTERPARTIES)
        account      = random.choice(ACCOUNTS)
        date         = start_date + timedelta(days=random.randint(0, 300))

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
    df.to_csv("data/cashbook_large.csv", index=False)
    print(f"Generated synthetic dataset: {len(df)} rows → data/cashbook_large.csv")
    print(df.head(10).to_string(index=False))
