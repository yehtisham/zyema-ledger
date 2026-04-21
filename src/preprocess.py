import pandas as pd

def load_cashbook(path="data/cleaned_cashbook.csv"):
    df = pd.read_csv(path, parse_dates=["date"])
    # compute net amount (credit - debit)
    df["amount"] = df["credit"] - df["debit"]
    # combine text fields for modeling
    df["text"] = df["description"].fillna("") + " " + df["counterparty"].fillna("")
    # remove zero-amount rows
    df = df[df["amount"] != 0]
    return df

# quick test
if __name__ == "__main__":
    df = load_cashbook()
    print(df.head())
