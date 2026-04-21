"""
train_category.py

Trains a TF-IDF + Logistic Regression pipeline to suggest ledger
categories from transaction text and amount.

Training strategy
-----------------
1. Real data  (full_cashbook.csv)   — primary source, actual business patterns
2. Synthetic  (cashbook_realistic.csv) — top-up ONLY for categories that have
   fewer than MIN_REAL_SAMPLES real examples, so rare categories aren't missed

This hybrid approach means the model learns your dad's real transaction
language first, and falls back to synthetic patterns only where real data
is too sparse to train on.

As real labeled data grows over time (through app usage), the synthetic
top-up becomes less and less necessary.
"""

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

MIN_REAL_SAMPLES = 10   # categories below this get topped up from synthetic
SYNTHETIC_TOPUP  = 200  # how many synthetic rows to add per sparse category

# ── 1. Load real data ─────────────────────────────────────────────────
real = pd.read_csv("data/full_cashbook.csv")
real["description"]  = real["description"].fillna("")
real["counterparty"] = real["counterparty"].fillna("")
real["text"]         = real["description"] + " " + real["counterparty"]
real["amount"]       = real["credit"] - real["debit"]
real["source"]       = "real"
real = real[real["text"].str.strip() != ""]
real = real[real["category"].notna()]

# ── 2. Top-up sparse categories from synthetic ────────────────────────
synth = pd.read_csv("data/cashbook_realistic.csv")
synth["description"]  = synth["description"].fillna("")
synth["counterparty"] = synth["counterparty"].fillna("")
synth["text"]         = synth["description"] + " " + synth["counterparty"]
synth["amount"]       = synth["credit"] - synth["debit"]
synth["source"]       = "synthetic"
synth = synth[synth["text"].str.strip() != ""]
synth = synth[synth["category"].notna() & (synth["category"] != "Uncategorized")]

real_counts  = real["category"].value_counts()
sparse_cats  = real_counts[real_counts < MIN_REAL_SAMPLES].index.tolist()

# Categories only in synthetic (not yet seen in real data at all)
synth_only   = set(synth["category"].unique()) - set(real["category"].unique())
topup_cats   = list(set(sparse_cats) | synth_only)

topup_rows = []
for cat in topup_cats:
    cat_rows = synth[synth["category"] == cat]
    if len(cat_rows) > 0:
        topup_rows.append(cat_rows.sample(min(SYNTHETIC_TOPUP, len(cat_rows)), random_state=42))

df = pd.concat([real] + topup_rows, ignore_index=True) if topup_rows else real.copy()

print(f"Training data composition:")
print(f"  Real rows      : {(df['source']=='real').sum()}")
print(f"  Synthetic topup: {(df['source']=='synthetic').sum()}")
print(f"  Total          : {len(df)}")
print(f"  Categories     : {df['category'].nunique()}")
print(f"\nReal data categories (with counts):")
print(real["category"].value_counts().to_string())
if topup_cats:
    print(f"\nTopped up from synthetic: {topup_cats}")

# ── 3. Train / test split ─────────────────────────────────────────────
X = df[["text", "amount"]]
y = df["category"]

min_class_count = y.value_counts().min()
stratify = y if min_class_count >= 2 else None

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=stratify
)

# ── 4. Pipeline ───────────────────────────────────────────────────────
pipe = Pipeline([
    ("prep", ColumnTransformer([
        ("text", TfidfVectorizer(max_features=3000, ngram_range=(1, 2)), "text"),
        ("num",  StandardScaler(), ["amount"]),
    ])),
    ("clf", LogisticRegression(max_iter=400, C=1.0, class_weight="balanced")),
])

# ── 5. Train ──────────────────────────────────────────────────────────
pipe.fit(X_train, y_train)

# ── 6. Evaluate ───────────────────────────────────────────────────────
preds = pipe.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, preds))

# ── 7. Save ───────────────────────────────────────────────────────────
joblib.dump(pipe, "models/category_suggester.joblib")
print("Model saved → models/category_suggester.joblib")
