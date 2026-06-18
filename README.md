# LedgerSmart

[![Live Demo](https://img.shields.io/badge/Live%20Demo-ledgersmart--web.netlify.app-teal)](https://ledgersmart-web.netlify.app)
[![Python](https://img.shields.io/badge/Python-3.9+-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.95+-green)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18+-61dafb)](https://react.dev/)

A full-stack financial intelligence system built for small trading and manufacturing businesses. Transforms raw transaction data into actionable business insights through machine learning, forecasting, and analysis.

**[Live demo: ledgersmart-web.netlify.app](https://ledgersmart-web.netlify.app)**

---

## What it does

Small trading businesses typically rely on manual Excel bookkeeping with no forecasting, no anomaly detection, and no financial intelligence. LedgerSmart addresses this by building an end-to-end ML pipeline on top of operational transaction data.

---

## Features

- **Dashboard** — Real-time KPIs: revenue, costs, net profit, gross margin, receivables, payables
- **Analytics** — Monthly revenue vs COGS trends, sales breakdown by product, expense breakdown, gross margin anomaly detection
- **Income Statement** — Auto-generated P&L from raw transactions
- **Balance Sheet** — Assets, liabilities, and derived equity
- **Forecast** — 6-month cash flow projection with 80% confidence intervals and walk-forward backtest
- **Transactions** — Full ledger with search, filter, and invoice creation
- **Receivables & Payables** — Net position, top customers, top suppliers
- **AR Aging** — Bucket analysis (0–30, 31–60, 61–90, 90+ days) with ML risk scoring
- **AI Assistant** — Claude-powered natural language interface with live chart generation

---

## ML Models

### 1. Transaction Classifier
- **Algorithm:** TF-IDF + Logistic Regression
- **Task:** Automatically categorizes transaction descriptions into 17 accounting categories
- **Training:** labeled samples
- **Features:** Text only — description-based classification, amount-independent

### 2. Cash Flow Forecasting
- **Algorithm:** SARIMAX (with your exogenous variables), Prophet, seasonal navi with dummies
- **Task:** 6-month revenue and COGS projection
- **Validation:** 3-fold time series walk-forward cross-validation

### 3. AR Risk Scoring
- **Algorithm:** Weighted scoring model
- **Formula:** score = (days_outstanding / 365 × 0.75) + (amount_percentile × 0.25)
- **Output:** High / Medium / Low risk labels per customer
- **Business use:** Prioritizes collection outreach

### 4. Gross Margin Anomaly Detection
- **Algorithm:** Rolling Z-score (unsupervised)
- **Task:** Flags months where gross margin deviates >1.5σ from 3-month rolling mean
- **Output:** Anomalous months with Z-scores and direction (above/below average)

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **Backend** | Python, FastAPI, SQLAlchemy, SQLite |
| **ML** | scikit-learn, statsmodels, pandas, numpy |
| **Frontend** | React, Vite, Recharts |
| **AI** | Anthropic Claude API |
| **Deployment** | Render (backend), Netlify (frontend) |

---

## Architecture
