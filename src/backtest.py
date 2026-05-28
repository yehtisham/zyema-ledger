"""
backtest.py — Walk-forward backtest using live transaction data
Trains on May 2024-Sep 2025, predicts Oct-Dec 2025, compares to actual.
"""
import requests
import pandas as pd
import numpy as np
import json
import sys
sys.path.insert(0, 'src')
from forecasting import _holt_forecast

# Fetch all transactions from live API
print("Fetching live transactions...")
r = requests.get("https://zyema-ledger.onrender.com/transactions?limit=1000")
txns = r.json()["transactions"]
print(f"Fetched {len(txns)} transactions")

# Build monthly aggregates
data = []
for t in txns:
    data.append({
        'date': t['date'],
        'debit': t['debit'] or 0,
        'credit': t['credit'] or 0,
        'category': t['category'] or ''
    })

df = pd.DataFrame(data)
df['date'] = pd.to_datetime(df['date'])
df['month'] = df['date'].dt.to_period('M').dt.to_timestamp()

# Build monthly aggregates - filter by category
df['is_purchase'] = df['category'].str.startswith('Purchases') | (df['category'] == 'Refurbishing Costs')
df['is_payment'] = ~df['category'].isin(['Accounts Payable Payment', 'Goods Return', 'Bank Charges', 'Office Supplies', 'Miscellaneous', 'Rent', 'Utilities', 'Freight & Conveyance', 'Labour & Wages']) & (df['credit'] > 0)

monthly = df.groupby('month').agg(
    purchases=('debit', lambda x: x[df.loc[x.index, 'is_purchase']].sum()),
    payments=('credit', lambda x: x[df.loc[x.index, 'is_payment']].sum())
).reset_index()

print("\nFull monthly data:")
for _, row in monthly.iterrows():
    print(f"  {row['month'].strftime('%Y-%m')}: purchases={row['purchases']:>12,.0f} payments={row['payments']:>12,.0f}")

# Split: train on May 2024 - Sep 2025, test on Oct-Dec 2025
train = monthly[monthly['month'] <= '2025-09-01'].copy()
test  = monthly[(monthly['month'] >= '2025-10-01') & (monthly['month'] <= '2025-12-01')].copy()

print(f"\nTraining months: {len(train)} ({train['month'].min().strftime('%b %Y')} - {train['month'].max().strftime('%b %Y')})")
print(f"Test months: {len(test)}")
print()

# Forecast 3 months
purch_fc = _holt_forecast(train['purchases'], steps=3)
pay_fc   = _holt_forecast(train['payments'],  steps=3)

# Compare
results = []
months = ['Oct 2025', 'Nov 2025', 'Dec 2025']
for i in range(min(3, len(test))):
    actual_p = test.iloc[i]['purchases']
    actual_r = test.iloc[i]['payments']
    pred_p   = purch_fc.iloc[i]['point']
    pred_r   = pay_fc.iloc[i]['point']

    results.append({
        'month': months[i],
        'actual_purchases': round(actual_p),
        'predicted_purchases': round(pred_p),
        'purchases_error': round(abs(actual_p - pred_p)),
        'actual_payments': round(actual_r),
        'predicted_payments': round(pred_r),
        'payments_error': round(abs(actual_r - pred_r)),
    })
    print(f"{months[i]}:")
    print(f"  Purchases — Actual: PKR {actual_p:>12,.0f} | Predicted: PKR {pred_p:>12,.0f} | Error: PKR {abs(actual_p-pred_p):>10,.0f}")
    print(f"  Payments  — Actual: PKR {actual_r:>12,.0f} | Predicted: PKR {pred_r:>12,.0f} | Error: PKR {abs(actual_r-pred_r):>10,.0f}")
    print()

# RMSE
p_errors = [(r['actual_purchases'] - r['predicted_purchases'])**2 for r in results]
r_errors = [(r['actual_payments']  - r['predicted_payments'])**2  for r in results]
rmse_p = round(np.sqrt(np.mean(p_errors)))
rmse_r = round(np.sqrt(np.mean(r_errors)))

print(f"RMSE Purchases: PKR {rmse_p:,.0f}")
print(f"RMSE Payments:  PKR {rmse_r:,.0f}")
print()
print(json.dumps({
    'train_period': f"{train['month'].min().strftime('%b %Y')} - {train['month'].max().strftime('%b %Y')}",
    'results': results,
    'rmse_purchases': rmse_p,
    'rmse_payments': rmse_r
}, indent=2))
