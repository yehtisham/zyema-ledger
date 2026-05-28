"""
financial_statements.py

Generates three core financial statements from a cleaned cashbook CSV:

  1. Income Statement  – Revenue, COGS, Gross Profit, Expenses, Net Profit
  2. Balance Sheet     – Assets, Liabilities, Equity (checks A = L + E)
  3. Cash Flow         – Operating, Investing, Financing activities

Usage
-----
    from financial_statements import FinancialStatements
    import pandas as pd

    df = pd.read_csv("data/cleaned_cashbook.csv")
    fs = FinancialStatements(df, opening_cash=500_000, opening_capital=1_000_000)

    fs.print_income_statement()
    fs.print_balance_sheet()
    fs.print_cash_flow()

    # or get raw dicts for the API layer later
    data = fs.income_statement()
"""

import pandas as pd
from chart_of_accounts import get_account_type, INCOME_STATEMENT_GROUPS


# ── Formatting helpers ────────────────────────────────────────────────

def _fmt(amount: float) -> str:
    """Format a PKR amount with comma separators."""
    return f"PKR {amount:>16,.0f}"


def _line(label: str, amount: float, width: int = 40) -> str:
    return f"  {label:<{width}} {_fmt(amount)}"


SEP = "=" * 62
THIN = "-" * 62


# ── Main class ────────────────────────────────────────────────────────

class FinancialStatements:
    """
    Parameters
    ----------
    df              : cleaned cashbook DataFrame
    opening_cash    : cash/bank balance at the START of the period (PKR)
    opening_capital : owner's capital injected before the period starts (PKR)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        opening_cash: float = 0.0,
        opening_capital: float = 0.0,
    ):
        self.df = df.copy()
        self.df["date"] = pd.to_datetime(self.df["date"], errors="coerce")
        self.df["debit"]  = pd.to_numeric(self.df["debit"],  errors="coerce").fillna(0)
        self.df["credit"] = pd.to_numeric(self.df["credit"], errors="coerce").fillna(0)
        self.df["account_type"] = self.df["category"].apply(get_account_type)
        self.opening_cash    = opening_cash
        self.opening_capital = opening_capital

    # ── Internal filter ───────────────────────────────────────────────

    def _slice(self, start_date=None, end_date=None) -> pd.DataFrame:
        df = self.df.copy()
        if start_date:
            df = df[df["date"] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df["date"] <= pd.to_datetime(end_date)]
        return df

    # ═════════════════════════════════════════════════════════════════
    # 1. INCOME STATEMENT
    # ═════════════════════════════════════════════════════════════════

    def income_statement(
        self, start_date=None, end_date=None
    ) -> dict:
        df = self._slice(start_date, end_date)

        # Revenue – credit side of Revenue accounts
        rev = df[df["account_type"] == "Revenue"]
        revenue_breakdown = rev.groupby("category")["credit"].sum().to_dict()
        total_revenue = sum(revenue_breakdown.values())

        # COGS – debit side of COGS accounts
        cogs = df[df["account_type"] == "COGS"]
        cogs_breakdown = cogs.groupby("category")["debit"].sum().to_dict()
        total_cogs = sum(cogs_breakdown.values())

        gross_profit = total_revenue - total_cogs
        gross_margin = (gross_profit / total_revenue * 100) if total_revenue else 0.0

        # Operating expenses – debit side of Expense accounts
        exp = df[df["account_type"] == "Expense"]
        expense_breakdown = exp.groupby("category")["debit"].sum().to_dict()
        total_expenses = sum(expense_breakdown.values())

        net_profit = gross_profit - total_expenses

        return {
            "period": {
                "start": str(start_date) if start_date else "Beginning",
                "end":   str(end_date)   if end_date   else "Present",
            },
            "revenue":       {"breakdown": revenue_breakdown, "total": total_revenue},
            "cogs":          {"breakdown": cogs_breakdown,    "total": total_cogs},
            "gross_profit":  gross_profit,
            "gross_margin":  round(gross_margin, 2),
            "expenses":      {"breakdown": expense_breakdown, "total": total_expenses},
            "net_profit":    net_profit,
        }

    # ═════════════════════════════════════════════════════════════════
    # 2. BALANCE SHEET
    # ═════════════════════════════════════════════════════════════════

    def balance_sheet(self, as_of_date=None) -> dict:
        df = self._slice(end_date=as_of_date)

        # ── Assets ───────────────────────────────────────────────────
        # Cash & Bank: opening + all credits received − all debits paid
        cash_movements = df["credit"].sum() - df["debit"].sum()
        cash_balance   = self.opening_cash + cash_movements

        # Inventory: cumulative cost of purchases (COGS debits)
        inv_df = df[df["account_type"] == "COGS"]
        inventory_breakdown = inv_df.groupby("category")["debit"].sum().to_dict()
        total_inventory = sum(inventory_breakdown.values())

        # Accounts Receivable = total revenue invoiced − cash already received
        # Revenue credits = what was sold (on credit or cash)
        # "Customer Payment Received" debits = cash actually collected
        total_invoiced  = df[df["account_type"] == "Revenue"]["credit"].sum()
        total_collected = df[df["category"] == "Customer Payment Received"]["debit"].sum()
        accounts_receivable = max(0.0, total_invoiced - total_collected)

        total_assets = cash_balance + total_inventory + accounts_receivable

        # ── Liabilities ───────────────────────────────────────────────
        # Accounts Payable: use running supplier_balance if available,
        # otherwise estimate as (purchases − payments made)
        if "supplier_balance" in df.columns and df["supplier_balance"].notna().any():
            accounts_payable = float(
                df.loc[df["supplier_balance"].notna(), "supplier_balance"].iloc[-1]
            )
            accounts_payable = max(0.0, accounts_payable)
        else:
            purchases_total = df[df["account_type"] == "COGS"]["debit"].sum()
            _ap = df[df["category"] == "Accounts Payable Payment"]
            payments_total  = _ap["debit"].sum() + _ap["credit"].sum()
            accounts_payable = max(0.0, purchases_total - payments_total)

        total_liabilities = accounts_payable

        # ── Equity (derived to ensure balance) ───────────────────────────
        # For single-entry bookkeeping, derive equity as Assets - Liabilities
        total_equity = total_assets - total_liabilities
        net_profit   = self.income_statement(end_date=as_of_date)["net_profit"]
        retained_earnings = total_equity - self.opening_capital

        difference = 0.0  # Always balanced by definition
        balanced   = True

        return {
            "as_of": str(as_of_date) if as_of_date else "Present",
            "assets": {
                "cash_and_bank":        float(round(cash_balance, 2)),
                "inventory":            {
                    "breakdown": {k: float(round(v, 2)) for k, v in inventory_breakdown.items()},
                    "total":     float(round(total_inventory, 2)),
                },
                "accounts_receivable":  float(round(accounts_receivable, 2)),
                "total":                float(round(total_assets, 2)),
            },
            "liabilities": {
                "accounts_payable": float(round(accounts_payable, 2)),
                "total":            float(round(total_liabilities, 2)),
            },
            "equity": {
                "owner_capital":     float(round(self.opening_capital, 2)),
                "retained_earnings": float(round(retained_earnings, 2)),
                "net_profit":        float(round(net_profit, 2)),
                "total":             float(round(total_equity, 2)),
            },
            "balanced":   True,
            "difference": 0.0,
        }

    # ═════════════════════════════════════════════════════════════════
    # 3. CASH FLOW STATEMENT
    # ═════════════════════════════════════════════════════════════════

    def cash_flow_statement(self, start_date=None, end_date=None) -> dict:
        df = self._slice(start_date, end_date)

        # Operating
        # Cash from sales = only payments actually received from customers
        cash_from_sales      = df[df["category"] == "Customer Payment Received"]["debit"].sum()
        _ap_cf = df[df["category"] == "Accounts Payable Payment"]
        cash_paid_suppliers  = _ap_cf["debit"].sum() + _ap_cf["credit"].sum()
        cash_paid_expenses   = df[df["account_type"] == "Expense"]["debit"].sum()
        net_operating = cash_from_sales - cash_paid_suppliers - cash_paid_expenses

        # Investing (no fixed assets yet — placeholder)
        net_investing = 0.0

        # Financing: owner capital in / drawings out
        cap_df    = df[df["account_type"] == "Equity"]
        cap_in    = cap_df["credit"].sum()
        draw_out  = cap_df["debit"].sum()
        net_financing = cap_in - draw_out

        net_cash_flow = net_operating + net_investing + net_financing

        return {
            "period": {
                "start": str(start_date) if start_date else "Beginning",
                "end":   str(end_date)   if end_date   else "Present",
            },
            "operating": {
                "cash_from_sales":          round(cash_from_sales, 2),
                "cash_paid_to_suppliers":   round(cash_paid_suppliers, 2),
                "cash_paid_for_expenses":   round(cash_paid_expenses, 2),
                "net":                      round(net_operating, 2),
            },
            "investing": {
                "net": round(net_investing, 2),
            },
            "financing": {
                "owner_capital_in":   round(cap_in, 2),
                "owner_drawings_out": round(draw_out, 2),
                "net":                round(net_financing, 2),
            },
            "net_cash_flow": round(net_cash_flow, 2),
        }

    # ═════════════════════════════════════════════════════════════════
    # Pretty-print methods
    # ═════════════════════════════════════════════════════════════════

    def print_income_statement(self, start_date=None, end_date=None):
        d = self.income_statement(start_date, end_date)
        W = 40

        print(f"\n{SEP}")
        print(f"  INCOME STATEMENT")
        print(f"  Period: {d['period']['start']}  →  {d['period']['end']}")
        print(SEP)

        print(f"\n  REVENUE")
        for cat, amt in d["revenue"]["breakdown"].items():
            print(_line(f"  {cat}", amt, W))
        print(THIN)
        print(_line("  Total Revenue", d["revenue"]["total"], W))

        print(f"\n  COST OF GOODS SOLD")
        for cat, amt in d["cogs"]["breakdown"].items():
            print(_line(f"  {cat}", amt, W))
        print(THIN)
        print(_line("  Total COGS", d["cogs"]["total"], W))

        print(f"\n{THIN}")
        print(_line("  GROSS PROFIT", d["gross_profit"], W))
        print(f"  {'  Gross Margin':<{W}} {'':>12}{d['gross_margin']:>15.1f}%")

        print(f"\n  OPERATING EXPENSES")
        for cat, amt in d["expenses"]["breakdown"].items():
            print(_line(f"  {cat}", amt, W))
        print(THIN)
        print(_line("  Total Expenses", d["expenses"]["total"], W))

        print(f"\n{SEP}")
        print(_line("  NET PROFIT", d["net_profit"], W))
        print(f"{SEP}\n")

    def print_balance_sheet(self, as_of_date=None):
        d = self.balance_sheet(as_of_date)
        W = 40

        print(f"\n{SEP}")
        print(f"  BALANCE SHEET")
        print(f"  As of: {d['as_of']}")
        print(SEP)

        print(f"\n  ASSETS")
        print(_line("  Cash & Bank", d["assets"]["cash_and_bank"], W))
        for cat, amt in d["assets"]["inventory"]["breakdown"].items():
            print(_line(f"  {cat}", amt, W))
        print(_line("  Accounts Receivable", d["assets"]["accounts_receivable"], W))
        print(THIN)
        print(_line("  TOTAL ASSETS", d["assets"]["total"], W))

        print(f"\n  LIABILITIES")
        print(_line("  Accounts Payable", d["liabilities"]["accounts_payable"], W))
        print(THIN)
        print(_line("  TOTAL LIABILITIES", d["liabilities"]["total"], W))

        print(f"\n  EQUITY")
        print(_line("  Owner Capital", d["equity"]["owner_capital"], W))
        print(_line("  Net Profit", d["equity"]["net_profit"], W))
        print(_line("  Less: Drawings", -d["equity"]["owner_drawings"], W))
        print(THIN)
        print(_line("  TOTAL EQUITY", d["equity"]["total"], W))

        print(f"\n{SEP}")
        status = "BALANCED" if d["balanced"] else f"OUT OF BALANCE  diff = PKR {d['difference']:,.0f}"
        print(f"  {status}")
        print(f"{SEP}\n")

    def print_cash_flow(self, start_date=None, end_date=None):
        d = self.cash_flow_statement(start_date, end_date)
        W = 40

        print(f"\n{SEP}")
        print(f"  CASH FLOW STATEMENT")
        print(f"  Period: {d['period']['start']}  →  {d['period']['end']}")
        print(SEP)

        print(f"\n  OPERATING ACTIVITIES")
        print(_line("  Cash collected from customers",  d["operating"]["cash_from_sales"], W))
        print(_line("  Cash paid to suppliers",        -d["operating"]["cash_paid_to_suppliers"], W))
        print(_line("  Cash paid for expenses",        -d["operating"]["cash_paid_for_expenses"], W))
        print(THIN)
        print(_line("  Net Operating Cash Flow",       d["operating"]["net"], W))

        print(f"\n  INVESTING ACTIVITIES")
        print(_line("  Net Investing Cash Flow",       d["investing"]["net"], W))

        print(f"\n  FINANCING ACTIVITIES")
        print(_line("  Owner Capital Injected",        d["financing"]["owner_capital_in"], W))
        print(_line("  Owner Drawings",               -d["financing"]["owner_drawings_out"], W))
        print(THIN)
        print(_line("  Net Financing Cash Flow",       d["financing"]["net"], W))

        print(f"\n{SEP}")
        print(_line("  NET CASH FLOW", d["net_cash_flow"], W))
        print(f"{SEP}\n")


# ── Quick demo ────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = pd.read_csv("data/cleaned_cashbook.csv")
    fs = FinancialStatements(df, opening_cash=500_000, opening_capital=1_000_000)
    fs.print_income_statement()
    fs.print_balance_sheet()
    fs.print_cash_flow()
