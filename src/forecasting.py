"""
forecasting.py

6-month cash flow and purchase forecast for a drums / IBC-tanks
sole-trader business.

Method: Holt's Linear Trend (double exponential smoothing)
  - Captures both level and trend in the data
  - Works with limited history (12 monthly points)
  - Produces confidence intervals that widen honestly over time

Forecasts produced
------------------
  1. Monthly purchase spend      (COGS outflows)
  2. Monthly supplier payments   (cash paid out)
  3. Net cash flow per month     (payments - purchases)
  4. Projected supplier balance  (running amount owed)

Outputs
-------
  - Console summary table
  - data/forecast.csv  (for the app / API layer later)
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import Holt
import warnings

warnings.filterwarnings("ignore")

SEP  = "=" * 70
THIN = "-" * 70


# ── Data preparation ──────────────────────────────────────────────────

def load_monthly(path: str = "data/cleaned_cashbook.csv") -> pd.DataFrame:
    """
    Aggregate the cleaned cashbook to monthly totals.
    Returns a DataFrame indexed by month-start date.
    """
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()

    monthly = (
        df.groupby("month")
        .agg(purchases=("debit", "sum"), payments=("credit", "sum"))
        .reset_index()
    )
    monthly["net_cash_flow"]  = monthly["payments"] - monthly["purchases"]
    monthly["supplier_balance"] = (
        monthly["purchases"] - monthly["payments"]
    ).cumsum()

    # Fill any missing months with 0 so the series is continuous
    full_range = pd.date_range(
        start=monthly["month"].min(),
        end=monthly["month"].max(),
        freq="MS",
    )
    monthly = (
        monthly.set_index("month")
        .reindex(full_range, fill_value=0)
        .reset_index()
        .rename(columns={"index": "month"})
    )
    # Recompute running balance after fill
    monthly["net_cash_flow"]    = monthly["payments"] - monthly["purchases"]
    monthly["supplier_balance"] = (
        monthly["purchases"] - monthly["payments"]
    ).cumsum()

    return monthly


# ── Forecasting engine ────────────────────────────────────────────────

def _holt_forecast(series: pd.Series, steps: int = 6) -> pd.DataFrame:
    """
    Fit Holt's Linear Trend model and return point forecast +
    80 % / 95 % confidence intervals for `steps` periods ahead.

    Falls back to a flat mean if the series has fewer than 4
    non-zero observations.
    """
    clean = series.replace(0, np.nan).dropna()

    if len(clean) < 4:
        # Not enough data — use mean as flat forecast
        mean = series.mean()
        std  = series.std(ddof=1) if len(series) > 1 else mean * 0.3
        idx  = range(1, steps + 1)
        return pd.DataFrame({
            "point":    [mean] * steps,
            "lower_80": [mean - 1.28 * std] * steps,
            "upper_80": [mean + 1.28 * std] * steps,
            "lower_95": [mean - 1.96 * std] * steps,
            "upper_95": [mean + 1.96 * std] * steps,
            "method":   ["flat mean"] * steps,
        })

    model  = Holt(clean, damped_trend=True, initialization_method="estimated")
    fitted = model.fit(optimized=True)
    pred   = fitted.forecast(steps)
    sim    = fitted.simulate(nsimulations=steps, repetitions=500, error="add")

    point    = pred.values
    lower_95 = np.percentile(sim, 2.5,  axis=1)
    upper_95 = np.percentile(sim, 97.5, axis=1)
    lower_80 = np.percentile(sim, 10,   axis=1)
    upper_80 = np.percentile(sim, 90,   axis=1)

    return pd.DataFrame({
        "point":    point,
        "lower_80": lower_80,
        "upper_80": upper_80,
        "lower_95": lower_95,
        "upper_95": upper_95,
        "method":   ["Holt linear trend"] * steps,
    })


# ── Main forecast function ────────────────────────────────────────────

def generate_forecast(
    data_path: str = "data/cleaned_cashbook.csv",
    horizon:   int = 6,
    output_path: str = "data/forecast.csv",
    opening_supplier_balance: float | None = None,
) -> pd.DataFrame:
    """
    Run the full forecast pipeline.

    Parameters
    ----------
    data_path                : path to cleaned_cashbook.csv
    horizon                  : months to forecast ahead (default 6)
    output_path              : where to save forecast CSV
    opening_supplier_balance : override the starting balance for the
                               projected supplier balance column.
                               If None, uses the last observed balance.

    Returns
    -------
    DataFrame with columns:
        month, purchases_forecast, payments_forecast,
        net_cf_forecast, supplier_balance_forecast,
        [lower/upper 80/95 for each series]
    """
    monthly = load_monthly(data_path)

    last_month   = monthly["month"].max()
    last_balance = (
        opening_supplier_balance
        if opening_supplier_balance is not None
        else monthly["supplier_balance"].iloc[-1]
    )

    # Future month dates
    future_months = pd.date_range(
        start=last_month + pd.offsets.MonthBegin(1),
        periods=horizon,
        freq="MS",
    )

    # Forecast each series
    purch_fc = _holt_forecast(monthly["purchases"],   horizon)
    pay_fc   = _holt_forecast(monthly["payments"],    horizon)

    # Clip negatives — you can't have negative purchases or payments
    for col in ["point", "lower_80", "lower_95"]:
        purch_fc[col] = purch_fc[col].clip(lower=0)
        pay_fc[col]   = pay_fc[col].clip(lower=0)

    # Net cash flow = payments - purchases (point estimate only)
    net_cf_point = pay_fc["point"] - purch_fc["point"]

    # Projected supplier balance = last observed + cumulative (purchases - payments)
    # Floored at 0 — you cannot owe a supplier negative money
    balance_proj = [last_balance]
    for p, py in zip(purch_fc["point"], pay_fc["point"]):
        new_bal = balance_proj[-1] + p - py
        balance_proj.append(max(0.0, new_bal))
    balance_proj = balance_proj[1:]  # drop seed value

    fc = pd.DataFrame({
        "month":                       future_months,

        "purchases_forecast":          purch_fc["point"].round(0),
        "purchases_lower_80":          purch_fc["lower_80"].round(0),
        "purchases_upper_80":          purch_fc["upper_80"].round(0),
        "purchases_lower_95":          purch_fc["lower_95"].round(0),
        "purchases_upper_95":          purch_fc["upper_95"].round(0),

        "payments_forecast":           pay_fc["point"].round(0),
        "payments_lower_80":           pay_fc["lower_80"].round(0),
        "payments_upper_80":           pay_fc["upper_80"].round(0),
        "payments_lower_95":           pay_fc["lower_95"].round(0),
        "payments_upper_95":           pay_fc["upper_95"].round(0),

        "net_cash_flow_forecast":      net_cf_point.round(0),
        "supplier_balance_forecast":   [round(b, 0) for b in balance_proj],
    })

    fc.to_csv(output_path, index=False)
    return fc


# ── Pretty-print summary ──────────────────────────────────────────────

def print_forecast(
    fc: pd.DataFrame,
    monthly: pd.DataFrame | None = None,
):
    W = 14

    print(f"\n{SEP}")
    print(f"  6-MONTH CASH FLOW FORECAST  (PKR)")
    print(f"  Method: Holt's Linear Trend  |  80% & 95% confidence intervals")
    print(SEP)

    # Historical summary
    if monthly is not None:
        print(f"\n  HISTORICAL MONTHLY AVERAGES (last 12 months)")
        print(f"  {'Purchases':>{W}}  {'Payments':>{W}}  {'Net CF':>{W}}")
        print(THIN)
        print(
            f"  {monthly['purchases'].mean():>{W},.0f}  "
            f"{monthly['payments'].mean():>{W},.0f}  "
            f"{monthly['net_cash_flow'].mean():>{W},.0f}"
        )

    print(f"\n  FORECAST")
    header = (
        f"  {'Month':<10}"
        f"{'Purchases':>{W}}"
        f"{'[80% range]':>26}"
        f"{'Payments':>{W}}"
        f"{'Net CF':>{W}}"
        f"{'Supplier Bal':>{W}}"
    )
    print(header)
    print(THIN)

    for _, row in fc.iterrows():
        month_str = row["month"].strftime("%b %Y")
        purch_range = f"[{row['purchases_lower_80']:>10,.0f} – {row['purchases_upper_80']:>10,.0f}]"
        print(
            f"  {month_str:<10}"
            f"{row['purchases_forecast']:>{W},.0f}"
            f"  {purch_range}"
            f"{row['payments_forecast']:>{W},.0f}"
            f"{row['net_cash_flow_forecast']:>{W},.0f}"
            f"{row['supplier_balance_forecast']:>{W},.0f}"
        )

    print(f"\n{SEP}")
    total_purch = fc["purchases_forecast"].sum()
    total_pay   = fc["payments_forecast"].sum()
    net_total   = fc["net_cash_flow_forecast"].sum()
    ending_bal  = fc["supplier_balance_forecast"].iloc[-1]
    print(f"  {'6-month totals':}")
    print(f"  Projected purchases : PKR {total_purch:>12,.0f}")
    print(f"  Projected payments  : PKR {total_pay:>12,.0f}")
    print(f"  Net cash flow       : PKR {net_total:>12,.0f}")
    print(f"  Ending supplier bal : PKR {ending_bal:>12,.0f}")
    print(f"\n  NOTE: Confidence intervals widen over time — treat month 4–6")
    print(f"  as directional only. Rerun as new data arrives.")
    print(f"{SEP}\n")


# ── Entry point ───────────────────────────────────────────────────────

if __name__ == "__main__":
    monthly = load_monthly()
    fc      = generate_forecast()
    print_forecast(fc, monthly)
    print(f"Forecast saved → data/forecast.csv")
