import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------- Load & clean ----------
df = pd.read_csv("wrds_data.csv")
df.columns = df.columns.str.lower()

# date
date_col = next((c for c in ["date","crcdt","rcrddt"] if c in df.columns), None)
if date_col is None:
    raise ValueError("No date-like column found (date/crcdt/rcrddt).")
df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
df = df.dropna(subset=[date_col])

# returns (percent)
if "ret" not in df.columns:
    raise ValueError("No 'ret' column found.")
df["ret"] = pd.to_numeric(df["ret"], errors="coerce") * 100
df = df.dropna(subset=["ret"])

# ticker normalization
if "ticker" not in df.columns:
    raise ValueError("No 'ticker' column found.")
df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
df["ticker_std"] = df["ticker"].str.extract(r'^([A-Z]+)', expand=False)  # e.g., 'AAPL.OQ' -> 'AAPL'

targets = ["AAPL","PFE","JNJ","MRK"]
df_target = df[df["ticker_std"].isin(targets)].copy().sort_values([date_col, "ticker_std"])

print("Counts by normalized ticker:")
print(df_target["ticker_std"].value_counts())

# ---------- Descriptive stats (robust .agg version) ----------
# df_target already has: columns [ ... , date_col, 'ret', 'ticker', 'ticker_std' ]
targets = ["AAPL","PFE","JNJ","MRK"]

stats = (
    df_target.groupby("ticker_std")["ret"]
    .agg(
        N="size",
        Mean="mean",
        Median="median",
        Std=lambda s: s.std(ddof=1),
        Skewness=lambda s: s.skew(),
        **{"Excess Kurtosis": lambda s: s.kurtosis()},  # Fisher definition (normal => 0)
        Min="min",
        Max="max",
    )
    .reindex(targets)   # keep order AAPL, PFE, JNJ, MRK
)

# sanity check types
print(stats.dtypes)

# pretty display + save
display(
    stats.style.format({
        "Mean":"{:.4f}","Median":"{:.4f}","Std":"{:.4f}",
        "Skewness":"{:.4f}","Excess Kurtosis":"{:.4f}",
        "Min":"{:.4f}","Max":"{:.4f}"
    }).set_caption("Descriptive statistics of daily returns (percent)")
)
stats.to_csv("table_q3_descriptives.csv")
print("[saved] table_q3_descriptives.csv")


# ---------- Four-panel plot ----------
fig, axes = plt.subplots(2, 2, figsize=(11, 6), sharex=False, sharey=False)
axes = axes.ravel()
for i, tkr in enumerate(targets):
    sub = df_target[df_target["ticker_std"] == tkr]
    ax = axes[i]
    ax.plot(sub[date_col], sub["ret"])
    ax.set_title(f"{tkr} daily returns (%)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Return (%)")
    ax.grid(True, alpha=0.3)

fig.suptitle("Daily Returns by Stock (percent scale)", y=1.02, fontsize=12)
plt.tight_layout()
fig.savefig("fig_returns_panels.png", dpi=300, bbox_inches="tight")
fig.savefig("fig_returns_panels.pdf", bbox_inches="tight")
print("[saved] fig_returns_panels.(png|pdf)")
